#pragma once

#include "row_projection.hpp"

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace bitfsai {

enum class Mode : int {
  FP4_E2M1_Split = 0,
  FP16_Split     = 1,
  FP64_Exact     = 2,
  INT4_Inclusive = 3,
  INT4_Split     = 4,
};

constexpr int kMaxPatch = 16;

namespace detail {

constexpr int kSetupMissingDiagonal = 1;
constexpr int kSetupNonsymmetricPattern = 2;
constexpr int kSetupInvalidDiagonal = 4;
constexpr int kSetupUnrepresentableFactor = 8;
constexpr int kSetupDiagonalFallback = 16;
constexpr int kSetupWideApply = 32;
constexpr int kSetupInvalidCsr = 64;

inline void check_cuda(cudaError_t err, const char *expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " +
                             cudaGetErrorString(err));
  }
}

__device__ __constant__ float kFp4Lut[16] = {
    0.0f / 6.0f,  0.5f / 6.0f,  1.0f / 6.0f,  1.5f / 6.0f,
    2.0f / 6.0f,  3.0f / 6.0f,  4.0f / 6.0f,  6.0f / 6.0f,
   -0.0f / 6.0f, -0.5f / 6.0f, -1.0f / 6.0f, -1.5f / 6.0f,
   -2.0f / 6.0f, -3.0f / 6.0f, -4.0f / 6.0f, -6.0f / 6.0f,
};

__device__ __forceinline__ uint8_t encode_fp4_e2m1(double r) {
  const uint8_t sgn = (r < 0.0) ? 8u : 0u;
  const double ar = fabs(r) * 6.0;
  uint8_t mag = 0u;
  if      (ar < 0.25) mag = 0u;
  else if (ar < 0.75) mag = 1u;
  else if (ar < 1.25) mag = 2u;
  else if (ar < 1.75) mag = 3u;
  else if (ar < 2.50) mag = 4u;
  else if (ar < 3.50) mag = 5u;
  else if (ar < 5.00) mag = 6u;
  else                mag = 7u;
  return (mag == 0u) ? 0u : static_cast<uint8_t>(sgn | mag);
}

__device__ __forceinline__ uint8_t encode_int4(double r) {
  int q = static_cast<int>(llround(r * 7.0));
  if (q > 7) q = 7;
  if (q < -7) q = -7;
  return static_cast<uint8_t>(q + 8);
}

__device__ __forceinline__ float decode_nibble(uint8_t nib, Mode mode) {
  if (mode == Mode::FP4_E2M1_Split) return kFp4Lut[nib & 0x0Fu];
  return static_cast<float>(static_cast<int>(nib & 0x0Fu) - 8) * (1.0f / 7.0f);
}

template <typename Scalar>
__device__ __forceinline__ double csr_lookup(const int *__restrict__ rp,
                                             const int *__restrict__ ci,
                                             const Scalar *__restrict__ aval,
                                             int row, int col) {
  for (int k = rp[row], re = rp[row + 1]; k < re; ++k) {
    if (ci[k] == col) return static_cast<double>(aval[k]);
  }
  return 0.0;
}

template <typename Scalar>
struct CsrPatchAt {
  const int *__restrict__ rp;
  const int *__restrict__ ci;
  const Scalar *__restrict__ aval;
  const int *__restrict__ rows;

  __device__ __forceinline__ double operator()(int row, int col) const {
    return csr_lookup(rp, ci, aval, rows[row], rows[col]);
  }
};

__device__ __forceinline__ int csr_find_pos(const int *__restrict__ rp,
                                            const int *__restrict__ ci,
                                            int row, int col) {
  for (int k = rp[row], re = rp[row + 1]; k < re; ++k) {
    if (ci[k] == col) return k;
  }
  return -1;
}

__global__ void validate_csr_rows_kernel(int n, int nnz,
                                         const int *__restrict__ rp,
                                         int *__restrict__ err_flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    if (rp[0] != 0 || rp[n] != nnz) atomicOr(err_flags, kSetupInvalidCsr);
  }
  if (i >= n) return;
  const int rs = rp[i], re = rp[i + 1];
  if (rs < 0 || re < rs || re > nnz) atomicOr(err_flags, kSetupInvalidCsr);
}

__global__ void validate_csr_cols_kernel(int n, const int *__restrict__ rp,
                                         const int *__restrict__ ci,
                                         const int *__restrict__ row_flags,
                                         int *__restrict__ col_flags) {
  if (*row_flags & kSetupInvalidCsr) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int rs = rp[i], re = rp[i + 1];
  bool strictly_sorted = true;
  int prev = -1;
  for (int k = rs; k < re; ++k) {
    const int c = ci[k];
    if (c < 0 || c >= n) {
      atomicOr(col_flags, kSetupInvalidCsr);
      return;
    }
    if (k > rs && c <= prev) strictly_sorted = false;
    prev = c;
  }
  if (!strictly_sorted) {
    for (int a = rs; a < re; ++a) {
      const int ca = ci[a];
      for (int b = a + 1; b < re; ++b) {
        if (ci[b] == ca) {
          atomicOr(col_flags, kSetupInvalidCsr);
          return;
        }
      }
    }
  }
}

template <typename Scalar>
__global__ void build_kernel(
    int n, Mode mode, int max_patch, bool reproject,
    const int *__restrict__ rp, const int *__restrict__ ci,
    const Scalar *__restrict__ aval,
    float *__restrict__ g_diag,
    float *__restrict__ g_off_scale,
    double *__restrict__ g_diag_f64,
    uint8_t *__restrict__ q_tmp,
    __half *__restrict__ g_f16,
    double *__restrict__ g_f64,
    int *__restrict__ err_flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const int rs = rp[i], re = rp[i + 1];
  int J[kMaxPatch];
  int Kpos[kMaxPatch];
  double Wabs[kMaxPatch];
  int m_low = 0;
  const int max_low = max_patch - 1;
  double a_ii = 1.0;
  bool found_diag = false;
  bool nonsymmetric = false;

  for (int k = rs; k < re; ++k) {
    const int col = ci[k];
    if (col < 0 || col >= n) {
      nonsymmetric = true;
      continue;
    }
    const double v = static_cast<double>(aval[k]);
    if (col == i) {
      a_ii = v;
      found_diag = true;
    } else {
      if (csr_find_pos(rp, ci, col, i) < 0) {
        nonsymmetric = true;
      }
      if (col < i) {
        const double av = fabs(v);
        if (m_low < max_low) {
          J[m_low] = col;
          Kpos[m_low] = k;
          Wabs[m_low] = av;
          ++m_low;
        } else if (max_low > 0) {
          int min_idx = 0;
          double min_w = Wabs[0];
          for (int t = 1; t < max_low; ++t) {
            if (Wabs[t] < min_w) {
              min_w = Wabs[t];
              min_idx = t;
            }
          }
          if (av > min_w) {
            J[min_idx] = col;
            Kpos[min_idx] = k;
            Wabs[min_idx] = av;
          }
        }
      }
    }
  }
  if (!found_diag) {
    if (err_flags) atomicOr(err_flags, kSetupMissingDiagonal);
    return;
  }
  if (nonsymmetric) {
    if (err_flags) atomicOr(err_flags, kSetupNonsymmetricPattern);
    return;
  }
  if (!(a_ii > 0.0) || !row_projection_finite(a_ii)) {
    if (err_flags) atomicOr(err_flags, kSetupInvalidDiagonal);
    return;
  }

  const int m = m_low + 1;
  J[m - 1] = i;

  double L[kMaxPatch * kMaxPatch];
  bool ok = true;
  for (int a = 0; a < m && ok; ++a) {
    for (int b = 0; b <= a; ++b) {
      double sum = (a == m - 1 && b == m - 1)
                       ? a_ii
                       : csr_lookup(rp, ci, aval, J[a], J[b]);
      for (int k = 0; k < b; ++k) {
        sum -= L[a * kMaxPatch + k] * L[b * kMaxPatch + k];
      }
      if (a == b) {
        if (!(sum > 0.0) || !row_projection_finite(sum)) {
          ok = false;
          break;
        } else {
          L[a * kMaxPatch + a] = sqrt(sum);
        }
      } else {
        const double pivot = L[b * kMaxPatch + b];
        L[a * kMaxPatch + b] = sum / pivot;
        if (!row_projection_finite(L[a * kMaxPatch + b])) {
          ok = false;
          break;
        }
      }
    }
  }

  double g[kMaxPatch] = {};
  if (ok) {
    const double llast = L[(m - 1) * kMaxPatch + (m - 1)];
    g[m - 1] = 1.0 / llast;
    if (!row_projection_finite(g[m - 1])) {
      ok = false;
    } else {
      for (int a = m - 2; a >= 0; --a) {
        double s = 0.0;
        for (int b = a + 1; b < m; ++b) s += L[b * kMaxPatch + a] * g[b];
        g[a] = -s / L[a * kMaxPatch + a];
        if (!row_projection_finite(g[a])) {
          ok = false;
          break;
        }
      }
    }
  }
  double g_d = 0.0;
  double scale = 0.0;
  bool diagonal_only = !ok || m == 1;
  if (!ok) {
    if (err_flags) atomicOr(err_flags, kSetupDiagonalFallback);
    for (int a = 0; a < m - 1; ++a) g[a] = 0.0;
    const RowProjectionStatus fallback = diagonal_row(a_ii, &g_d, &scale);
    if (fallback == RowProjectionStatus::InvalidInput) {
      if (err_flags) atomicOr(err_flags, kSetupInvalidDiagonal);
      return;
    }
    g[m - 1] = g_d;
  } else {
    g_d = g[m - 1];
  }

  if (mode == Mode::FP64_Exact) {
    if (!(g_d > 0.0) || !row_projection_finite(g_d)) {
      if (err_flags) atomicOr(err_flags, kSetupInvalidDiagonal);
      return;
    }
    g_diag_f64[i] = g_d;
    for (int a = 0; a < m - 1; ++a) {
      const int k_low = Kpos[a];
      const int k_up = csr_find_pos(rp, ci, J[a], i);
      if (k_up < 0 && err_flags) atomicOr(err_flags, kSetupNonsymmetricPattern);
      g_f64[k_low] = g[a];
      if (k_up >= 0) g_f64[k_up] = g[a];
    }
    return;
  }

  double omax = 0.0;
  for (int a = 0; a < m - 1; ++a) omax = fmax(omax, fabs(g[a]));
  if (ok) {
    scale = (mode == Mode::INT4_Inclusive)
                ? fmax(fabs(g_d), omax)
                : ((omax > 0.0) ? omax : fabs(g_d));
  }
  if (!(scale != 0.0) || !row_projection_finite(scale)) {
    if (err_flags) atomicOr(err_flags, kSetupUnrepresentableFactor);
    return;
  }

  uint8_t q_code[kMaxPatch];
  __half q_f16[kMaxPatch];
  double q_dec[kMaxPatch];
  const bool is_4bit = (mode == Mode::FP4_E2M1_Split ||
                        mode == Mode::INT4_Inclusive ||
                        mode == Mode::INT4_Split);

  for (int a = 0; a < m - 1; ++a) {
    const double u = g[a] / scale;
    if (is_4bit) {
      const uint8_t code =
          (mode == Mode::FP4_E2M1_Split) ? encode_fp4_e2m1(u) : encode_int4(u);
      q_code[a] = code;
      q_dec[a] = static_cast<double>(decode_nibble(code, mode));
    } else {
      const __half hv = __double2half(u);
      q_f16[a] = hv;
      q_dec[a] = static_cast<double>(__half2float(hv));
    }
  }

  if (reproject && ok && m > 1) {
    const CsrPatchAt<Scalar> patch_at{rp, ci, aval, J};
    const RowProjectionStatus projection = project_row(
        m, q_dec, L, kMaxPatch, patch_at, &g_d, &scale);
    if (projection != RowProjectionStatus::Applied) {
      if (err_flags) atomicOr(err_flags, kSetupDiagonalFallback);
      const RowProjectionStatus fallback = diagonal_row(a_ii, &g_d, &scale);
      if (fallback == RowProjectionStatus::InvalidInput) {
        if (err_flags) atomicOr(err_flags, kSetupInvalidDiagonal);
        return;
      }
      diagonal_only = true;
      for (int a = 0; a < m - 1; ++a) {
        q_code[a] = (mode == Mode::FP4_E2M1_Split) ? 0u : 8u;
        q_f16[a] = __double2half(0.0);
      }
    }
  }

  if (!row_projection_storable_f32(g_d) ||
      !row_projection_storable_f32(scale)) {
    if (err_flags) atomicOr(err_flags, kSetupUnrepresentableFactor);
    return;
  }
  if (diagonal_only) {
    const double stored_diag = static_cast<double>(static_cast<float>(g_d));
    const double stored_energy = stored_diag * stored_diag * a_ii;
    if (!row_projection_finite(stored_energy) ||
        fabs(stored_energy - 1.0) > kRowProjectionEnergyTolerance) {
      if (err_flags) atomicOr(err_flags, kSetupUnrepresentableFactor);
      return;
    }
  }

  // Keep a wide intermediate when scale factoring would compress the exponent range.
  const double ratio = fabs(scale) / g_d;
  if (err_flags && (ratio < 1e-6 || ratio > 1e6 || fabs(scale) < sqrt(double(FLT_MIN)) ||
                    fabs(scale) > sqrt(double(FLT_MAX))))
    atomicOr(err_flags, kSetupWideApply);
  g_diag[i] = static_cast<float>(g_d);
  g_off_scale[i] = static_cast<float>(scale);

  for (int a = 0; a < m - 1; ++a) {
    const int k_low = Kpos[a];
    const int k_up = csr_find_pos(rp, ci, J[a], i);
    if (k_up < 0 && err_flags) atomicOr(err_flags, kSetupNonsymmetricPattern);
    if (is_4bit) {
      q_tmp[k_low] = q_code[a];
      if (k_up >= 0) q_tmp[k_up] = q_code[a];
    } else {
      g_f16[k_low] = q_f16[a];
      if (k_up >= 0) g_f16[k_up] = q_f16[a];
    }
  }
}

__global__ void pack_nibbles_kernel(int nnz, const uint8_t *__restrict__ q_tmp,
                                    uint8_t *__restrict__ q_packed) {
  const uint64_t b = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= (static_cast<uint64_t>(nnz) + 1ULL) / 2ULL) return;
  const int idx = static_cast<int>(b << 1);
  const uint8_t lo = q_tmp[idx] & 0x0Fu;
  const uint8_t hi = (idx + 1 < nnz) ? (q_tmp[idx + 1] & 0x0Fu) : 0u;
  q_packed[b] = static_cast<uint8_t>(lo | (hi << 4));
}

template <Mode M>
__device__ __forceinline__ float off_weight(int k, const uint8_t *packed, const __half *half) {
  if constexpr (M == Mode::FP16_Split) return __half2float(half[k]);
  const uint8_t byte = packed[k >> 1];
  return decode_nibble((k & 1) ? byte >> 4 : byte & 15u, M);
}

template <typename VecT, Mode M, typename Acc>
__global__ void apply_pass1_kernel(
    int n, double pass1_scale, const int *rp, const int *ci, const float *diag, const float *scale,
    const double *diag64, const uint8_t *packed, const __half *half, const double *exact,
    const VecT *r, Acc *tmp, int *oor_flag = nullptr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if constexpr (M == Mode::FP64_Exact) {
    double sum = diag64[i] * double(r[i]);
    for (int k = rp[i]; k < rp[i + 1]; ++k)
      if (ci[k] < i) sum += exact[k] * double(r[ci[k]]);
    tmp[i] = pass1_scale * sum;
  } else if constexpr (std::is_same_v<Acc, double>) {
    // Standard IEEE-754 binary64 normal-range Gram accumulation.
    double sum = double(diag[i]) * double(r[i]);
    for (int k = rp[i]; k < rp[i + 1]; ++k)
      if (ci[k] < i)
        sum = fma(double(scale[i]) * off_weight<M>(k, packed, half), double(r[ci[k]]), sum);
    tmp[i] = pass1_scale * sum;
  } else {
    const float sc = scale[i];
    const float r_i = float(r[i]);
    bool any_nz = (r_i != 0.0f);
    float t_i = diag[i] * r_i;
    for (int k = rp[i]; k < rp[i + 1]; ++k) {
      if (ci[k] < i) {
        const float w = off_weight<M>(k, packed, half);
        const float r_c = float(r[ci[k]]);
        if (w != 0.0f && r_c != 0.0f) any_nz = true;
        t_i = fmaf(sc * w, r_c, t_i);
      }
    }
    const float y_i = sc * t_i;
    tmp[i] = y_i;
    if (oor_flag && (!isfinite(y_i) ||
                     (t_i != 0.0f && (fabsf(t_i) < FLT_MIN || fabsf(y_i) < FLT_MIN)) ||
                     (any_nz && fabsf(t_i) < FLT_MIN))) {
      atomicOr(oor_flag, 1);
    }
  }
}

template <typename VecT, Mode M, typename Acc>
__global__ void apply_pass2_kernel(
    int n, double omega, const int *rp, const int *ci, const float *diag, const float *scale,
    const double *diag64, const uint8_t *packed, const __half *half, const double *exact,
    const Acc *tmp, VecT *x, bool accumulate, const int *oor_flag = nullptr) {
  if constexpr (!std::is_same_v<Acc, double> && M != Mode::FP64_Exact) {
    if (oor_flag && *oor_flag != 0) return;
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  double sum_d;
  if constexpr (M == Mode::FP64_Exact) {
    double sum = diag64[j] * tmp[j];
    for (int k = rp[j]; k < rp[j + 1]; ++k)
      if (ci[k] > j) sum += exact[k] * tmp[ci[k]];
    sum_d = sum;
  } else if constexpr (std::is_same_v<Acc, double>) {
    double sum = double(diag[j]) * tmp[j];
    for (int k = rp[j]; k < rp[j + 1]; ++k)
      if (ci[k] > j)
        sum = fma(double(scale[ci[k]]) * off_weight<M>(k, packed, half), tmp[ci[k]], sum);
    sum_d = sum;
  } else {
    float sum = diag[j] * (tmp[j] / scale[j]);
    for (int k = rp[j]; k < rp[j + 1]; ++k)
      if (ci[k] > j) sum = fmaf(off_weight<M>(k, packed, half), tmp[ci[k]], sum);
    if (!isfinite(sum) || (sum != 0.0f && fabsf(sum) < FLT_MIN)) {
      double dsum = double(diag[j]) * (double(tmp[j]) / double(scale[j]));
      for (int k = rp[j]; k < rp[j + 1]; ++k)
        if (ci[k] > j)
          dsum = fma(double(off_weight<M>(k, packed, half)), double(tmp[ci[k]]), dsum);
      sum_d = dsum;
    } else {
      sum_d = double(sum);
    }
  }
  const double update = omega * sum_d;
  x[j] = static_cast<VecT>(accumulate ? double(x[j]) + update : update);
}

template <typename VecT, Mode M>
__global__ void apply_fallback_pass1_kernel(
    int n, const int *rp, const int *ci, const float *diag, const float *scale,
    const uint8_t *packed, const __half *half, const VecT *r, double *dtmp,
    const int *oor_flag) {
  if (*oor_flag == 0) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double sum = double(diag[i]) * double(r[i]);
  for (int k = rp[i]; k < rp[i + 1]; ++k)
    if (ci[k] < i)
      sum = fma(double(scale[i]) * double(off_weight<M>(k, packed, half)), double(r[ci[k]]), sum);
  dtmp[i] = sum;
}

template <typename VecT, Mode M>
__global__ void apply_fallback_pass2_kernel(
    int n, double omega, const int *rp, const int *ci, const float *diag, const float *scale,
    const uint8_t *packed, const __half *half, const double *dtmp, VecT *x,
    bool accumulate, const int *oor_flag) {
  if (*oor_flag == 0) return;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  double sum = double(diag[j]) * dtmp[j];
  for (int k = rp[j]; k < rp[j + 1]; ++k)
    if (ci[k] > j)
      sum = fma(double(scale[ci[k]]) * double(off_weight<M>(k, packed, half)), dtmp[ci[k]], sum);
  const double update = omega * sum;
  x[j] = static_cast<VecT>(accumulate ? double(x[j]) + update : update);
}

}

struct Smoother {
  int n = 0;
  int nnz = 0;
  Mode mode = Mode::INT4_Split;
  bool reproject = true;
  int max_patch = 12;
  bool wide_apply = false;
  bool used_diagonal_fallback = false;

  float *d_diag = nullptr;
  float *d_off_scale = nullptr;
  uint8_t *d_q_packed = nullptr;
  __half *d_f16 = nullptr;
  double *d_diag_f64 = nullptr;
  double *d_f64 = nullptr;

  Smoother() = default;
  Smoother(const Smoother &) = delete;
  Smoother &operator=(const Smoother &) = delete;
  ~Smoother() { free_buffers(); }

  void free_buffers() {
    auto freep = [](auto *&p) {
      if (p) {
        cudaFree(p);
        p = nullptr;
      }
    };
    freep(d_diag);
    freep(d_off_scale);
    freep(d_q_packed);
    freep(d_f16);
    freep(d_diag_f64);
    freep(d_f64);
    n = nnz = 0;
    wide_apply = false;
    used_diagonal_fallback = false;
  }

  size_t factor_bytes() const {
    if (mode == Mode::FP64_Exact)
      return (static_cast<size_t>(nnz) + static_cast<size_t>(n)) * sizeof(double);
    if (mode == Mode::FP16_Split)
      return static_cast<size_t>(nnz) * sizeof(__half) + static_cast<size_t>(n) * 2 * sizeof(float);
    return (static_cast<size_t>(nnz) + 1) / 2 + static_cast<size_t>(n) * 2 * sizeof(float);
  }

  template <typename Scalar>
  void setup(int rows, int nonzeros,
             const int *d_rp, const int *d_ci, const Scalar *d_val,
             Mode m = Mode::INT4_Split, bool use_reproject = true,
             int patch_cap = 12, cudaStream_t stream = 0) try {
    free_buffers();
    if (m != Mode::FP4_E2M1_Split && m != Mode::FP16_Split &&
        m != Mode::FP64_Exact && m != Mode::INT4_Inclusive &&
        m != Mode::INT4_Split)
      throw std::runtime_error("BitFSAI setup failed: invalid Mode enum value");
    if (rows <= 0)
      throw std::runtime_error("BitFSAI setup failed: row count must be positive");
    if (nonzeros < 0)
      throw std::runtime_error("BitFSAI setup failed: nonzero count must be non-negative");
    if (d_rp == nullptr || (nonzeros > 0 && (d_ci == nullptr || d_val == nullptr)))
      throw std::runtime_error("BitFSAI setup failed: null CSR device pointer");

    constexpr uint64_t kMaxGridX = 2147483647ULL;
    const uint64_t n_u64 = static_cast<uint64_t>(rows);
    const uint64_t nnz_u64 = static_cast<uint64_t>(nonzeros);
    const uint64_t threads_u64 = 256ULL;
    const uint64_t blocks_u64 = (n_u64 + threads_u64 - 1ULL) / threads_u64;
    const uint64_t packed_u64 = (nnz_u64 + 1ULL) / 2ULL;
    const uint64_t pblocks_u64 = (packed_u64 + threads_u64 - 1ULL) / threads_u64;
    if (blocks_u64 > kMaxGridX || pblocks_u64 > kMaxGridX ||
        n_u64 > SIZE_MAX / sizeof(double) || nnz_u64 > SIZE_MAX / sizeof(double)) {
      throw std::runtime_error("BitFSAI setup failed: dimensions exceed allocation or grid limits");
    }

    n = rows;
    nnz = nonzeros;
    mode = m;
    reproject = use_reproject;
    max_patch = (patch_cap < 1) ? 1 : (patch_cap > kMaxPatch ? kMaxPatch : patch_cap);

    const bool is_4bit = (mode == Mode::FP4_E2M1_Split ||
                          mode == Mode::INT4_Inclusive ||
                          mode == Mode::INT4_Split);

    struct SetupScratch {
      uint8_t *codes = nullptr;
      int *flags = nullptr;
      ~SetupScratch() {
        if (codes) cudaFree(codes);
        if (flags) cudaFree(flags);
      }
    } scratch;
    detail::check_cuda(cudaMalloc(&scratch.flags, 2 * sizeof(int)), "cudaMalloc(scratch.flags)");
    detail::check_cuda(cudaMemsetAsync(scratch.flags, 0, 2 * sizeof(int), stream), "cudaMemset(scratch.flags)");

    const int threads = 256;
    const int blocks = static_cast<int>(blocks_u64);
    detail::validate_csr_rows_kernel<<<blocks, threads, 0, stream>>>(n, nnz, d_rp, scratch.flags);
    detail::check_cuda(cudaGetLastError(), "validate_csr_rows_kernel");
    detail::validate_csr_cols_kernel<<<blocks, threads, 0, stream>>>(n, d_rp, d_ci, scratch.flags, scratch.flags + 1);
    detail::check_cuda(cudaGetLastError(), "validate_csr_cols_kernel");
    int h_csr_err[2] = {0, 0};
    detail::check_cuda(cudaMemcpyAsync(h_csr_err, scratch.flags, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream),
                       "copy_csr_err");
    detail::check_cuda(cudaStreamSynchronize(stream), "sync_csr_validate");
    if ((h_csr_err[0] | h_csr_err[1]) & detail::kSetupInvalidCsr)
      throw std::runtime_error(
          "BitFSAI setup failed: malformed CSR row offsets, out-of-range column, or duplicate column");

    if (mode == Mode::FP64_Exact) {
      detail::check_cuda(cudaMalloc(&d_diag_f64, static_cast<size_t>(n_u64) * sizeof(double)), "diag_f64");
      if (nnz_u64 > 0) {
        detail::check_cuda(cudaMalloc(&d_f64, static_cast<size_t>(nnz_u64) * sizeof(double)), "f64");
        detail::check_cuda(cudaMemsetAsync(d_f64, 0, static_cast<size_t>(nnz_u64) * sizeof(double), stream), "zero_f64");
      }
    } else {
      detail::check_cuda(cudaMalloc(&d_diag, static_cast<size_t>(n_u64) * sizeof(float)), "diag");
      detail::check_cuda(cudaMalloc(&d_off_scale, static_cast<size_t>(n_u64) * sizeof(float)), "scale");
      if (is_4bit) {
        const size_t packed_bytes = static_cast<size_t>(packed_u64);
        if (packed_bytes > 0) {
          detail::check_cuda(cudaMalloc(&d_q_packed, packed_bytes), "q_packed");
          detail::check_cuda(cudaMalloc(&scratch.codes, static_cast<size_t>(nnz_u64)), "q_tmp");
          const uint8_t zero_code = (mode == Mode::FP4_E2M1_Split) ? 0u : 8u;
          detail::check_cuda(cudaMemsetAsync(scratch.codes, zero_code, static_cast<size_t>(nnz_u64), stream), "zero_q_tmp");
        }
      } else if (nnz_u64 > 0) {
        detail::check_cuda(cudaMalloc(&d_f16, static_cast<size_t>(nnz_u64) * sizeof(__half)), "f16");
        detail::check_cuda(cudaMemsetAsync(d_f16, 0, static_cast<size_t>(nnz_u64) * sizeof(__half), stream), "zero_f16");
      }
    }

    detail::build_kernel<Scalar><<<blocks, threads, 0, stream>>>(
        n, mode, max_patch, reproject, d_rp, d_ci, d_val,
        d_diag, d_off_scale, d_diag_f64, scratch.codes, d_f16, d_f64, scratch.flags);
    detail::check_cuda(cudaGetLastError(), "build_kernel");

    if (is_4bit && packed_u64 > 0) {
      const int pblocks = static_cast<int>(pblocks_u64);
      detail::pack_nibbles_kernel<<<pblocks, threads, 0, stream>>>(nnz, scratch.codes, d_q_packed);
      detail::check_cuda(cudaGetLastError(), "pack_nibbles_kernel");
    }

    int h_err = 0;
    detail::check_cuda(cudaMemcpyAsync(&h_err, scratch.flags, sizeof(int), cudaMemcpyDeviceToHost, stream), "copy_err");
    detail::check_cuda(cudaStreamSynchronize(stream), "sync_setup");

    used_diagonal_fallback = (h_err & detail::kSetupDiagonalFallback) != 0;
    wide_apply = mode == Mode::FP64_Exact || (h_err & detail::kSetupWideApply) != 0;
    if (h_err & detail::kSetupInvalidCsr)
      throw std::runtime_error(
          "BitFSAI setup failed: malformed CSR row offsets, out-of-range column, or duplicate column");
    if (h_err & detail::kSetupMissingDiagonal)
      throw std::runtime_error("BitFSAI setup failed: CSR row missing explicit diagonal");
    if (h_err & detail::kSetupNonsymmetricPattern)
      throw std::runtime_error("BitFSAI setup failed: CSR sparsity pattern is not symmetric");
    if (h_err & detail::kSetupInvalidDiagonal)
      throw std::runtime_error("BitFSAI setup failed: diagonal must be finite and positive");
    if (h_err & detail::kSetupUnrepresentableFactor)
      throw std::runtime_error("BitFSAI setup failed: factor cannot be stored in FP32");
  } catch (...) {
    free_buffers();
    throw;
  }

  template <typename VecT>
  size_t workspace_bytes() const {
    if (wide_apply || std::is_same_v<VecT, double> || mode == Mode::FP64_Exact)
      return size_t(n) * sizeof(double);
    return (size_t(n) + 1) * sizeof(double);
  }

  // A supplied workspace needs workspace_bytes<VecT>() bytes for each overlapping call.
  template <typename VecT>
  void apply(const int *d_rp, const int *d_ci, const VecT *d_r, VecT *d_x,
             double damp = 1.0, bool accumulate = false, cudaStream_t stream = 0,
             void *workspace = nullptr) const {
    if (n == 0) throw std::logic_error("BitFSAI apply requires successful setup");
    const int threads = 256;
    const int blocks = 1 + (n - 1) / threads;
    struct WorkspaceGuard {
      void *ptr = nullptr;
      bool owned = false;
      bool pooled = false;
      cudaStream_t stream = 0;
      ~WorkspaceGuard() {
        if (owned && ptr) {
#if CUDART_VERSION >= 11020
          if (pooled) {
            (void)cudaFreeAsync(ptr, stream);
            return;
          }
#endif
          (void)cudaFree(ptr);
        }
      }
      void release_checked() {
        if (!owned || !ptr) return;
        void *p = ptr;
        ptr = nullptr;
        owned = false;
#if CUDART_VERSION >= 11020
        if (pooled) {
          detail::check_cuda(cudaFreeAsync(p, stream), "cudaFreeAsync(apply workspace)");
          return;
        }
#endif
        detail::check_cuda(cudaFree(p), "cudaFree(apply workspace)");
      }
    } ws_guard{workspace, workspace == nullptr, false, stream};
    if (ws_guard.owned) {
#if CUDART_VERSION >= 11020
      cudaError_t err = cudaMallocAsync(&ws_guard.ptr, workspace_bytes<VecT>(), stream);
      if (err == cudaSuccess) {
        ws_guard.pooled = true;
      } else if (err == cudaErrorNotSupported || err == cudaErrorInsufficientDriver) {
        (void)cudaGetLastError();
      } else {
        detail::check_cuda(err, "cudaMallocAsync(apply workspace)");
      }
#endif
      if (!ws_guard.pooled)
        detail::check_cuda(cudaMalloc(&ws_guard.ptr, workspace_bytes<VecT>()), "cudaMalloc(apply workspace)");
      workspace = ws_guard.ptr;
    }
    auto launch = [&](auto tag, auto precision) {
      constexpr Mode M = decltype(tag)::value;
      using Acc = decltype(precision);
      if constexpr (std::is_same_v<Acc, float> && M != Mode::FP64_Exact) {
        auto *tmp = static_cast<float *>(workspace);
        auto *dtmp = static_cast<double *>(workspace);
        int *oor_flag = reinterpret_cast<int *>(dtmp + n);
        detail::check_cuda(cudaMemsetAsync(oor_flag, 0, sizeof(int), stream), "zero_oor_flag");
        detail::apply_pass1_kernel<VecT, M, float><<<blocks, threads, 0, stream>>>(
            n, 1.0, d_rp, d_ci, d_diag, d_off_scale, d_diag_f64, d_q_packed, d_f16, d_f64,
            d_r, tmp, oor_flag);
        detail::check_cuda(cudaGetLastError(), "apply_pass1_kernel");
        detail::apply_pass2_kernel<VecT, M, float><<<blocks, threads, 0, stream>>>(
            n, damp, d_rp, d_ci, d_diag, d_off_scale, d_diag_f64, d_q_packed, d_f16, d_f64,
            tmp, d_x, accumulate, oor_flag);
        detail::check_cuda(cudaGetLastError(), "apply_pass2_kernel");
        detail::apply_fallback_pass1_kernel<VecT, M><<<blocks, threads, 0, stream>>>(
            n, d_rp, d_ci, d_diag, d_off_scale, d_q_packed, d_f16, d_r, dtmp, oor_flag);
        detail::check_cuda(cudaGetLastError(), "apply_fallback_pass1_kernel");
        detail::apply_fallback_pass2_kernel<VecT, M><<<blocks, threads, 0, stream>>>(
            n, damp, d_rp, d_ci, d_diag, d_off_scale, d_q_packed, d_f16, dtmp, d_x,
            accumulate, oor_flag);
        detail::check_cuda(cudaGetLastError(), "apply_fallback_pass2_kernel");
      } else {
        auto *tmp = static_cast<Acc *>(workspace);
        const bool damp_pass1 = (damp >= 0.0 && damp <= 1.0);
        const double p1_scale = damp_pass1 ? damp : 1.0;
        const double p2_scale = damp_pass1 ? 1.0 : damp;
        detail::apply_pass1_kernel<VecT, M, Acc><<<blocks, threads, 0, stream>>>(
            n, p1_scale, d_rp, d_ci, d_diag, d_off_scale, d_diag_f64, d_q_packed, d_f16, d_f64, d_r, tmp);
        detail::check_cuda(cudaGetLastError(), "apply_pass1_kernel");
        detail::apply_pass2_kernel<VecT, M, Acc><<<blocks, threads, 0, stream>>>(
            n, p2_scale, d_rp, d_ci, d_diag, d_off_scale, d_diag_f64, d_q_packed, d_f16, d_f64,
            tmp, d_x, accumulate);
        detail::check_cuda(cudaGetLastError(), "apply_pass2_kernel");
      }
    };
    auto dispatch = [&](auto tag) {
      if constexpr (decltype(tag)::value == Mode::FP64_Exact || std::is_same_v<VecT, double>)
        launch(tag, double{});
      else if (wide_apply)
        launch(tag, double{});
      else
        launch(tag, float{});
    };
    switch (mode) {
      case Mode::FP4_E2M1_Split: dispatch(std::integral_constant<Mode, Mode::FP4_E2M1_Split>{}); break;
      case Mode::FP16_Split:     dispatch(std::integral_constant<Mode, Mode::FP16_Split>{}); break;
      case Mode::FP64_Exact:     dispatch(std::integral_constant<Mode, Mode::FP64_Exact>{}); break;
      case Mode::INT4_Inclusive: dispatch(std::integral_constant<Mode, Mode::INT4_Inclusive>{}); break;
      case Mode::INT4_Split:     dispatch(std::integral_constant<Mode, Mode::INT4_Split>{}); break;
      default: throw std::logic_error("BitFSAI apply failed: invalid Mode enum value");
    }
    ws_guard.release_checked();
  }
};

}
