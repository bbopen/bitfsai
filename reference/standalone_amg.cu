#include "workloads.hpp"
#include "row_projection.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>

#define CHECK(expr)                                                            \
  do {                                                                         \
    auto err = (expr);                                                         \
    if (err != cudaSuccess)                                                    \
      throw std::runtime_error(std::string(#expr) + " " +                      \
                               cudaGetErrorString(err));                       \
  } while (0)
#define DEVICE [=] __device__
using std::vector;
cudaStream_t stream = nullptr;

template <class T> struct Buffer {
  T *p = nullptr;
  size_t count = 0;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  ~Buffer() {
    if (p)
      cudaFree(p);
  }
  void free() {
    if (p) {
      cudaFree(p);
      p = nullptr;
      count = 0;
    }
  }
  void alloc(size_t n) {
    free();
    if (n) {
      CHECK(cudaMalloc(&p, n * sizeof(T)));
      count = n;
    }
  }
  void init(const vector<T> &v) {
    alloc(v.size());
    if (p) {
      CHECK(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T),
                            cudaMemcpyHostToDevice, stream));
      CHECK(cudaStreamSynchronize(stream));
    }
  }
  size_t bytes() const { return count * sizeof(T); }
};
template <class F> __global__ void rows(int n, F f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    f(i);
}
template <class F> void launch(int n, F f) {
  rows<<<(n + 255) / 256, 256, 0, stream>>>(n, f);
  CHECK(cudaGetLastError());
}
struct Timer {
  cudaEvent_t a, b;
  Timer() {
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, stream));
  }
  float stop() {
    CHECK(cudaEventRecord(b, stream));
    CHECK(cudaEventSynchronize(b));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
  ~Timer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
};
enum Method { Jacobi, Half, Int4, Double, Int4Raw };
constexpr int MaxLower = 16;
template <class T> struct Work {
  Buffer<T> rhs, x, res, tmp, poly_rhs, poly_acc;
  void alloc(int n) {
    rhs.alloc(n);
    x.alloc(n);
    res.alloc(n);
    tmp.alloc(n);
    poly_rhs.alloc(n);
    poly_acc.alloc(n);
  }
};
struct Level {
  int n, nz, nc = 0;
  Buffer<int> rp, ci, interp, source, members, mstart, rap, starts, anchor;
  Buffer<int2> slots;
  Buffer<double> a, l1, pw, dense, exact;
  Buffer<float> af, inv, diag, scale, ac_diag;
  Buffer<__half> half;
  Buffer<unsigned char> packed;
  Buffer<uint8_t> ac_fp8;
  Buffer<unsigned> failures, invalid, oor;
  size_t bytes[5]{};
  size_t lower_nnz = 0, selected_lower_nnz = 0, truncated_rows = 0;
  int max_lower = 0;
  bool cap_three = false;
  bool use_factored = true;
  bool use_fp8_level = false;
  uint64_t alloc_generation = 0;
  Buffer<int> tentative;
  double omega = 0;
  Work<float> mix;
  Work<double> full;

  void alloc_fp8_buffers() {
    ac_fp8.alloc(nz);
    ac_diag.alloc(n);
    if (nz) CHECK(cudaMemsetAsync(ac_fp8.p, 0, ac_fp8.bytes(), stream));
    if (n) CHECK(cudaMemsetAsync(ac_diag.p, 0, ac_diag.bytes(), stream));
    ++alloc_generation;
  }
  void free_fp8_buffers() {
    if (ac_fp8.p || ac_diag.p) {
      ac_fp8.free();
      ac_diag.free();
      ++alloc_generation;
    }
  }
  void alloc_af_buffer() {
    af.alloc(nz);
    ++alloc_generation;
  }
  void free_af_buffer() {
    if (af.p) {
      af.free();
      ++alloc_generation;
    }
  }

  size_t active_coarse_bytes(bool c_fp8) const {
    if (dense.p)
      return 0;
    if (c_fp8 && use_fp8_level)
      return ac_fp8.bytes() + ac_diag.bytes();
    return af.bytes();
  }
  size_t allocated_smoother_bytes() const {
    if (dense.p)
      return 0;
    return diag.bytes() + scale.bytes() + packed.bytes() + half.bytes() +
           exact.bytes();
  }
  size_t allocated_coarse_bytes() const {
    if (dense.p)
      return 0;
    return af.bytes() + ac_fp8.bytes() + ac_diag.bytes();
  }
};

void smooth_interpolation(Level &l, const HostCSR &h, vector<int> &col,
                          vector<double> &w, int nc) {
  vector<int> agg(h.n);
  for (int i = 0; i < h.n; i++)
    agg[i] = col[2 * i];
  vector<int> anchor(nc, -1);
  for (int i = 0; i < h.n; i++)
    if (anchor[agg[i]] < 0)
      anchor[agg[i]] = i;
  double bound = 1;
  for (int i = 0; i < h.n; i++) {
    double row_norm = 0, diagonal = 0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; k++) {
      row_norm += std::abs(h.val[k]);
      if (h.ci[k] == i)
        diagonal = h.val[k];
    }
    bound = std::max(bound, row_norm / diagonal);
  }
  l.omega = 1.3333333333333333 / bound;
  vector<std::pair<int, double>> contributions;
  for (int i = 0; i < h.n; i++) {
    if (anchor[agg[i]] == i) {
      col[2 * i + 1] = -1;
      w[2 * i] = 1;
      w[2 * i + 1] = 0;
      continue;
    }
    contributions.clear();
    double diagonal = 0, row_sum = 0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; k++) {
      row_sum += h.val[k];
      contributions.emplace_back(agg[h.ci[k]], h.val[k]);
      if (h.ci[k] == i)
        diagonal = h.val[k];
    }
    std::stable_sort(contributions.begin(), contributions.end(),
                     [](auto &a, auto &b) { return a.first < b.first; });
    col[2 * i + 1] = -1;
    w[2 * i] = 1;
    w[2 * i + 1] = 0;
    for (size_t t = 0; t < contributions.size();) {
      int c = contributions[t].first;
      double value = 0;
      do {
        value += contributions[t++].second;
      } while (t < contributions.size() && contributions[t].first == c);
      double weight = -l.omega * value / diagonal;
      if (c == agg[i])
        w[2 * i] += weight;
      else if (std::abs(weight) > std::abs(w[2 * i + 1])) {
        col[2 * i + 1] = c;
        w[2 * i + 1] = weight;
      }
    }
    w[2 * i] = 1 - l.omega * row_sum / diagonal - w[2 * i + 1];
  }
  for (int c = 0; c < nc; ++c)
    if (anchor[c] < 0)
      throw std::runtime_error("smoothed aggregation lost a tentative aggregate");
  l.tentative.init(agg);
  l.anchor.init(anchor);
}

HostCSR symbolic(Level &l, const HostCSR &h, bool direct, bool sa,
                 size_t level_idx = 0, bool coarse_fp8 = false,
                 bool only_int4 = false) {
  int n = l.n = h.n;
  l.nz = h.nnz;
  l.rp.init(h.rp);
  l.ci.init(h.ci);
  l.a.alloc(h.nnz);
  l.mix.alloc(n);
  l.full.alloc(n);
  l.failures.alloc(1);
  l.invalid.alloc(1);
  l.oor.alloc(1);
  CHECK(cudaMemsetAsync(l.failures.p, 0, sizeof(unsigned), stream));
  CHECK(cudaMemsetAsync(l.invalid.p, 0, sizeof(unsigned), stream));
  CHECK(cudaMemsetAsync(l.oor.p, 0, sizeof(unsigned), stream));
  if (n <= 64) {
    l.dense.alloc(n * n);
    if (n > 0) CHECK(cudaMemsetAsync(l.dense.p, 0, l.dense.bytes(), stream));
    return {};
  }
  if (level_idx > 0 && coarse_fp8) {
    l.ac_fp8.alloc(h.nnz);
    l.ac_diag.alloc(n);
    if (h.nnz) CHECK(cudaMemsetAsync(l.ac_fp8.p, 0, l.ac_fp8.bytes(), stream));
    if (n) CHECK(cudaMemsetAsync(l.ac_diag.p, 0, l.ac_diag.bytes(), stream));
    l.use_fp8_level = true;
  } else {
    l.af.alloc(h.nnz);
    l.use_fp8_level = false;
  }
  l.l1.alloc(n);
  l.inv.alloc(n);
  l.diag.alloc(n);
  l.scale.alloc(n);
  size_t packed_size = (size_t(h.nnz) + n + 1) / 2;
  l.packed.alloc(packed_size);
  if (packed_size)
    CHECK(cudaMemsetAsync(l.packed.p, 0, packed_size, stream));
  if (!only_int4) {
    l.half.alloc(h.nnz);
    l.exact.alloc(h.nnz);
    if (h.nnz) {
      CHECK(cudaMemsetAsync(l.half.p, 0, l.half.bytes(), stream));
      CHECK(cudaMemsetAsync(l.exact.p, 0, l.exact.bytes(), stream));
    }
  }
  l.bytes[Half] = size_t(8) * n + size_t(2) * h.nnz;
  l.bytes[Int4] = size_t(8) * n + packed_size;
  l.bytes[Int4Raw] = l.bytes[Int4];
  l.bytes[Double] = size_t(8) * h.nnz;
  vector<int> coarse_id(n, -1), col(2 * n, -1), src(2 * n, -1);
  vector<double> w(2 * n, 0), threshold(n);
  for (int i = 0; i < n; i++) {
    double mx = 0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; k++)
      if (h.ci[k] != i)
        mx = std::max(mx, std::abs(h.val[k]));
    threshold[i] = 0.25 * mx;
    int lower = std::lower_bound(h.ci.begin() + h.rp[i],
                                 h.ci.begin() + h.rp[i + 1], i) -
                (h.ci.begin() + h.rp[i]);
    l.max_lower = std::max(l.max_lower, lower);
    constexpr int kDefaultMaxLower = 16;
    l.lower_nnz += static_cast<size_t>(lower) + 1;
    l.selected_lower_nnz += static_cast<size_t>(std::min(lower, kDefaultMaxLower)) + 1;
    if (lower > kDefaultMaxLower)
      ++l.truncated_rows;
  }
  int nc = 0;

  for (int i = 0; i < n; i++) {
    bool covered = false;
    for (int k = h.rp[i]; k < h.rp[i + 1] && !covered; k++)
      covered = coarse_id[h.ci[k]] >= 0 && std::abs(h.val[k]) >= threshold[i] &&
                h.val[k] != 0;
    if (!covered)
      coarse_id[i] = nc++;
  }
  bool pair_injection = nc > 0.95 * n;
  if (pair_injection)
    nc = (n + 1) / 2;
  vector<int> off(nc + 1);
  for (int i = 0; i < n; i++) {
    if (pair_injection) {
      col[2 * i] = i / 2;
      w[2 * i] = 1;
    } else if (coarse_id[i] >= 0) {
      col[2 * i] = coarse_id[i];
      w[2 * i] = 1;
    } else {
      int best[2] = {-1, -1};
      for (int k = h.rp[i]; k < h.rp[i + 1]; k++) {
        if (coarse_id[h.ci[k]] < 0 || std::abs(h.val[k]) < threshold[i] ||
            h.val[k] == 0)
          continue;
        int candidate = k;
        for (int j = 0; j < (direct ? 2 : 1); j++)
          if (best[j] < 0 ||
              std::abs(h.val[candidate]) > std::abs(h.val[best[j]])) {
            std::swap(candidate, best[j]);
            if (candidate < 0)
              break;
          }
      }
      double sum = 0;
      for (int j = 0; j < 2 && best[j] >= 0; j++) {
        src[2 * i + j] = best[j];
        col[2 * i + j] = coarse_id[h.ci[best[j]]];
        sum += (w[2 * i + j] = std::abs(h.val[best[j]]));
      }
      for (int j = 0; j < 2; j++)
        if (best[j] >= 0)
          w[2 * i + j] = std::abs(h.val[best[j]]) / sum;
    }
  }
  if (sa)
    smooth_interpolation(l, h, col, w, nc);
  for (int c : col)
    if (c >= 0)
      ++off[c + 1];
  std::partial_sum(off.begin(), off.end(), off.begin());
  vector<int> cursor = off, members(off.back());
  for (int t = 0; t < 2 * n; t++)
    if (col[t] >= 0)
      members[cursor[col[t]]++] = t;
  l.nc = nc;
  l.interp.init(col);
  l.source.init(src);
  l.pw.alloc(2 * n);
  l.members.init(members);
  l.mstart.init(off);
  HostCSR coarse;
  coarse.n = nc;
  coarse.rp.push_back(0);
  vector<int> rap, starts{0};
  vector<int2> slots;
  struct Term {
    int col, k, left, right;
  };
  vector<Term> row;
  for (int u = 0; u < nc; u++) {
    row.clear();
    for (int t = off[u]; t < off[u + 1]; t++) {
      int left = members[t], i = left / 2;
      for (int k = h.rp[i]; k < h.rp[i + 1]; k++)
        for (int j = 0; j < 2; j++) {
          int right = 2 * h.ci[k] + j;
          if (col[right] >= 0)
            row.push_back({col[right], k, left, right});
        }
    }
    std::sort(row.begin(), row.end(),
              [](auto &a, auto &b) { return a.col < b.col; });
    for (size_t t = 0; t < row.size();) {
      int c = row[t].col;
      double sum = 0;
      do {
        auto e = row[t++];
        rap.push_back(e.k);
        slots.push_back(make_int2(e.left, e.right));
        sum += w[e.left] * h.val[e.k] * w[e.right];
      } while (t < row.size() && row[t].col == c);
      coarse.ci.push_back(c);
      coarse.val.push_back(sum);
      starts.push_back(rap.size());
    }
    coarse.rp.push_back(coarse.ci.size());
  }
  coarse.nnz = coarse.ci.size();
  l.rap.init(rap);
  l.slots.init(slots);
  l.starts.init(starts);
  return coarse;
}
__device__ double entry(int i, int j, const int *rp, const int *ci,
                        const double *a) {
  int lo = rp[i], hi = rp[i + 1];
  while (lo < hi) {
    int m = (lo + hi) / 2;
    if (ci[m] < j)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < rp[i + 1] && ci[lo] == j ? a[lo] : 0.;
}
template <int K> void numeric(Level &l, Method method) {
  auto rp = l.rp.p, ci = l.ci.p;
  auto exact = l.exact.p;
  auto failures = l.failures.p;
  auto invalid = l.invalid.p;
  auto a = l.a.p, l1 = l.l1.p;
  auto af = l.af.p, inv = l.inv.p, d = l.diag.p, s = l.scale.p;
  auto half = l.half.p;
  auto packed = l.packed.p;
  if ((method == Double && !exact) || (method == Half && !half) ||
      ((method == Int4 || method == Int4Raw) && !packed))
    throw std::logic_error("requested smoother representation is not allocated");
  if (af)
    launch(l.nz, DEVICE(int k) { af[k] = a[k]; });
  launch(
      l.n, DEVICE(int i) {
        int ids[K + 1], m = 0;
        int base = rp[i];
        int pos[K], diagpos = base;
        while (diagpos < rp[i + 1] && ci[diagpos] < i) {
          if (method != Jacobi) {
            if (m < K)
              pos[m++] = diagpos;
            else {
              int weakest = 0;
              for (int j = 1; j < K; j++)
                if (fabs(a[pos[j]]) < fabs(a[pos[weakest]]))
                  weakest = j;
              if (fabs(a[diagpos]) > fabs(a[pos[weakest]]))
                pos[weakest] = diagpos;
            }
          }
          ++diagpos;
        }
        if (diagpos >= rp[i + 1] || ci[diagpos] != i ||
            !(a[diagpos] > 0.0 && isfinite(a[diagpos]))) {
          atomicOr(invalid, 1u);
          return;
        }
        double sum = 0;
        for (int k = rp[i]; k < rp[i + 1]; ++k)
          sum += fabs(a[k]);
        if (!(sum > 0.0 && isfinite(sum))) {
          atomicOr(invalid, 1u);
          return;
        }
        l1[i] = 1 / sum;
        inv[i] = 1 / sum;
        if (method == Jacobi)
          return;
        if (diagpos - base > K) {
          for (int k = base; k < diagpos; k++) {
            if (method == Double)
              exact[k] = 0;
            else if (method == Half)
              half[k] = __float2half(0);
          }
        }
        if (method == Int4 || method == Int4Raw)
          for (int j = 0; j <= diagpos - base; j += 2)
            packed[(base + i) / 2 + j / 2] = 0;
        for (int j = 0; j < m; j++)
          ids[j] = ci[pos[j]];
        ids[m] = i;
        double L[K + 1][K + 1] = {}, g[K + 1] = {};
        for (int r = 0; r <= m; ++r)
          for (int c = 0; c <= r; ++c) {
            double v = entry(ids[r], ids[c], rp, ci, a);
            for (int k = 0; k < c; ++k)
              v -= L[r][k] * L[c][k];
            L[r][c] = r == c ? sqrt(v) : v / L[c][c];
          }

        for (int r = m; r >= 0; --r) {
          double v = (r == m);
          for (int k = r + 1; k <= m; ++k)
            v -= L[k][r] * g[k];
          g[r] = v / L[r][r];
        }
        bool factor_valid = true;
        for (int j = 0; j <= m; j++)
          factor_valid &= isfinite(g[j]);
        factor_valid &= g[m] > 0;
        if (!factor_valid) {
          for (int j = 0; j < m; j++)
            g[j] = 0;
          g[m] = sqrt(l1[i]);
        }
        if (method == Double) {
          if (!factor_valid)
            atomicAdd(failures, 1u);
          exact[diagpos] = g[m];
          for (int j = 0; j < m; j++)
            exact[pos[j]] = g[j];
          return;
        }
        double mx = 0;
        for (int j = 0; j < m; ++j)
          mx = fmax(mx, fabs(g[j]));
        double decoded[K] = {};
        unsigned q = 0;
        for (int j = 0; j < m; ++j) {
          double u = mx ? g[j] / mx : 0;
          if (method == Half) {
            half[pos[j]] = __float2half(u);
            decoded[j] = __half2float(half[pos[j]]);
          } else {
            int t = pos[j] - base;
            q = (int(round(u * 7)) & 15) << (4 * (t & 1));
            packed[(base + i) / 2 + t / 2] |= q;
            int code = int((q >> (4 * (t & 1))) & 15);
            decoded[j] = double((code ^ 8) - 8) / 7;
          }
        }
        double stored_diag = 0, stored_scale = 0;
        bool fallback = false, use_diagonal = m == 0;
        if (method == Int4Raw) {
          stored_diag = g[m];
          stored_scale = mx ? mx / 7 : g[m];
          fallback = !factor_valid;
          use_diagonal = false;
        } else if (m) {
          auto patch_at = DEVICE(int row, int col) {
            return entry(ids[row], ids[col], rp, ci, a);
          };
          auto status = factor_valid ? bitfsai::project_row(
                                            m + 1, decoded, &L[0][0], K + 1,
                                            patch_at, &stored_diag,
                                            &stored_scale)
                                      : bitfsai::RowProjectionStatus::Fallback;
          if (status != bitfsai::RowProjectionStatus::Applied) {
            fallback = true;
            use_diagonal = true;
          }
        }
        if (use_diagonal) {
          auto status = bitfsai::diagonal_row(a[diagpos], &stored_diag,
                                              &stored_scale);
          double persisted_diag = static_cast<float>(stored_diag);
          double energy = persisted_diag * persisted_diag * a[diagpos];
          if (status == bitfsai::RowProjectionStatus::InvalidInput ||
              !bitfsai::row_projection_storable_f32(stored_diag) ||
              !bitfsai::row_projection_storable_f32(stored_scale) ||
              !isfinite(energy) ||
              fabs(energy - 1) > bitfsai::kRowProjectionEnergyTolerance) {
            atomicOr(invalid, 1u);
            stored_diag = 1;
            stored_scale = 1;
          }
        }
        if (fallback)
          atomicAdd(failures, 1u);
        if (!bitfsai::row_projection_storable_f32(stored_diag) ||
            !bitfsai::row_projection_storable_f32(stored_scale)) {
          atomicOr(invalid, 1u);
          stored_diag = 1;
          stored_scale = 1;
          use_diagonal = true;
        }
        if (use_diagonal)
          for (int j = 0; j < m; ++j) {
            if (method == Half) {
              half[pos[j]] = __float2half(0);
            } else {
              int t = pos[j] - base;
              int at = (base + i) / 2 + t / 2;
              packed[at] = (t & 1) ? (packed[at] & 15) : (packed[at] & 240);
            }
          }
        float fd = static_cast<float>(stored_diag);
        float fs = static_cast<float>(stored_scale);
        d[i] = fd;
        s[i] = fs;
        float eff_sc = (method == Int4) ? (fs * (1.0f / 7.0f)) : fs;
        if (!(fabsf(eff_sc) >= 1e-12f && fabsf(eff_sc) <= 1e12f))
          atomicOr(invalid, 2u);
      });
  if (method == Jacobi)
    return;

  launch(
      l.n, DEVICE(int i) {
        int base = rp[i], k = base;
        while (k < rp[i + 1] && ci[k] < i)
          ++k;
        if (k >= rp[i + 1] || ci[k] != i) {
          atomicOr(invalid, 1u);
          return;
        }
        for (++k; k < rp[i + 1]; k++) {
          int c = ci[k], b = rp[c], lo = b, hi = rp[c + 1];
          while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ci[mid] < i)
              lo = mid + 1;
            else
              hi = mid;
          }
          bool found = lo < rp[c + 1] && ci[lo] == i;
          if (method == Double)
            exact[k] = found ? exact[lo] : 0.;
          else if (method == Half)
            half[k] = found ? half[lo] : __float2half(0);
          else {
            int j = lo - b;
            unsigned q =
                found ? ((packed[(b + c) / 2 + j / 2] >> (4 * (j & 1))) & 15)
                      : 0;
            int t = k - base, pos = (base + i) / 2 + t / 2;

            packed[pos] = (t & 1) ? ((packed[pos] & 15) | (q << 4)) : q;
          }
        }
      });
}
using Levels = vector<std::unique_ptr<Level>>;
void reset(Levels &levels, Method method, const double *values,
           bool coarse_fp8 = false) {
  auto &fine = *levels.front();
  CHECK(cudaMemcpyAsync(fine.a.p, values, fine.nz * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream));
  for (size_t j = 0; j < levels.size(); ++j) {
    auto &l = *levels[j];
    CHECK(cudaMemsetAsync(l.invalid.p, 0, sizeof(unsigned), stream));
    if (l.dense.p) {
      auto rp = l.rp.p, ci = l.ci.p;
      auto a = l.a.p, L = l.dense.p;
      auto failures = l.failures.p;
      int n = l.n;
      CHECK(cudaMemsetAsync(failures, 0, sizeof(unsigned), stream));
      launch(
          1, DEVICE(int) {
            for (int i = 0; i < n; i++)
              for (int j = 0; j <= i; j++) {
                double v = entry(i, j, rp, ci, a);
                for (int k = 0; k < j; k++)
                  v -= L[i * n + k] * L[j * n + k];
                if (i == j) {
                  if (!(v > 0 && isfinite(v))) {
                    atomicAdd(failures, 1u);
                    return;
                  }
                  L[i * n + j] = sqrt(v);
                } else {
                  double pivot = L[j * n + j];
                  if (!(pivot > 0 && isfinite(pivot))) {
                    atomicAdd(failures, 1u);
                    return;
                  }
                  double val = v / pivot;
                  if (!isfinite(val)) {
                    atomicAdd(failures, 1u);
                    return;
                  }
                  L[i * n + j] = val;
                }
              }
          });
      unsigned invalid = 0;
      CHECK(cudaMemcpyAsync(&invalid, failures, sizeof(invalid),
                            cudaMemcpyDeviceToHost, stream));
      CHECK(cudaStreamSynchronize(stream));
      if (invalid)
        throw std::runtime_error(
            "dense coarse operator is not SPD (non-positive or non-finite Cholesky pivot)");
      continue;
    }
    CHECK(cudaMemsetAsync(l.failures.p, 0, sizeof(unsigned), stream));
    if (coarse_fp8 && j > 0) {
      if (!l.ac_fp8.p) {
        l.alloc_fp8_buffers();
      }
      auto rp = l.rp.p, ci = l.ci.p;
      auto a = l.a.p;
      auto ac_fp8 = l.ac_fp8.p;
      auto ac_diag = l.ac_diag.p;
      auto invalid = l.invalid.p;
      launch(
          l.n, DEVICE(int i) {
            double diag_val = 0.0;
            bool has_diag = false;
            double row_sum = 0.0, row_comp = 0.0;
            double dec_off_sum = 0.0, dec_comp = 0.0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) {
              int c = ci[k];
              if (c == i) {
                diag_val = a[k];
                has_diag = true;
                ac_fp8[k] = 0u;
                double t1 = row_sum + diag_val;
                row_comp += (fabs(row_sum) >= fabs(diag_val))
                                ? ((row_sum - t1) + diag_val)
                                : ((diag_val - t1) + row_sum);
                row_sum = t1;
              } else {
                int lo = rp[c], hi = rp[c + 1];
                while (lo < hi) {
                  int mid = (lo + hi) / 2;
                  if (ci[mid] < i)
                    lo = mid + 1;
                  else
                    hi = mid;
                }
                if (lo >= rp[c + 1] || ci[lo] != i) {
                  atomicOr(invalid, 4u);
                  ac_fp8[k] = 0u;
                  continue;
                }
                double a_sym = 0.5 * (a[k] + a[lo]);
                if (!isfinite(a_sym) || !isfinite(a[k])) {
                  atomicOr(invalid, 4u);
                  ac_fp8[k] = 0u;
                  continue;
                }
                __nv_fp8_storage_t q =
                    __nv_cvt_double_to_fp8(a_sym, __NV_SATFINITE, __NV_E5M2);
                ac_fp8[k] = static_cast<uint8_t>(q);
                double dec = static_cast<double>(
                    __half2float(__nv_cvt_fp8_to_halfraw(q, __NV_E5M2)));
                const double a_abs = fabs(a_sym);
                if (a_sym > 0.0 || dec > 0.0 || a_abs > 57344.0 ||
                    (a_sym != 0.0 && a_abs < ldexp(1.0, -14))) {
                  atomicOr(invalid, 4u);
                }

                // Neumaier compensated summation over all row entries (including diagonal).
                double x1 = a[k];
                double t1 = row_sum + x1;
                row_comp += (fabs(row_sum) >= fabs(x1)) ? ((row_sum - t1) + x1)
                                                        : ((x1 - t1) + row_sum);
                row_sum = t1;

                double t2 = dec_off_sum + dec;
                dec_comp += (fabs(dec_off_sum) >= fabs(dec))
                                ? ((dec_off_sum - t2) + dec)
                                : ((dec - t2) + dec_off_sum);
                dec_off_sum = t2;
              }
            }
            if (!has_diag || !isfinite(diag_val)) {
              atomicOr(invalid, 4u);
              ac_diag[i] = 1.0f;
              return;
            }
            double exact_row_sum = row_sum + row_comp;
            dec_off_sum += dec_comp;
            double repaired_diag = exact_row_sum - dec_off_sum;
            float rep_d = static_cast<float>(repaired_diag);
            if (exact_row_sum >= 0.0 &&
                static_cast<double>(rep_d) + dec_off_sum < 0.0) {
              rep_d = nextafterf(rep_d, INFINITY);
            }
            if (exact_row_sum > 0.0 &&
                static_cast<double>(rep_d) + dec_off_sum <= 0.0) {
              rep_d = nextafterf(rep_d, INFINITY);
            }
            double dec_row_sum = static_cast<double>(rep_d) + dec_off_sum;
            if (!(exact_row_sum >= 0.0) || !(rep_d > 0.0f && isfinite(rep_d)) ||
                !(dec_row_sum >= 0.0) ||
                (exact_row_sum > 0.0 && !(dec_row_sum > 0.0))) {
              atomicOr(invalid, 4u);
              rep_d = 1.0f;
            } else if (dec_row_sum > 0.0) {
              atomicOr(invalid, 16u);
            }
            ac_diag[i] = rep_d;
          });
    } else if (!l.af.p) {
      l.alloc_af_buffer();
    }
    if (l.cap_three || l.max_lower <= 3)
      numeric<3>(l, method);
    else if (l.max_lower <= 8)
      numeric<8>(l, method);
    else
      numeric<MaxLower>(l, method);
    if (l.nc) {
      auto a = l.a.p, b = levels[j + 1]->a.p;
      auto start = l.starts.p, map = l.rap.p;
      auto src = l.source.p;
      auto pw = l.pw.p;
      auto slots = l.slots.p;
      if (l.tentative.p) {
        auto agg = l.tentative.p, col = l.interp.p, rp = l.rp.p, ci = l.ci.p;
        auto anchor = l.anchor.p;
        double omega = l.omega;
        launch(
            l.n, DEVICE(int i) {
              if (anchor[agg[i]] == i) {
                pw[2 * i] = 1;
                pw[2 * i + 1] = 0;
                return;
              }
              double row_sum = 0, other = 0, diagonal = 0;
              for (int k = rp[i]; k < rp[i + 1]; k++) {
                int c = ci[k];
                if (c == i)
                  diagonal = a[k];
                row_sum += a[k];
                if (agg[c] == col[2 * i + 1])
                  other += a[k];
              }
              pw[2 * i + 1] = -omega * other / diagonal;
              pw[2 * i] = 1 - omega * row_sum / diagonal - pw[2 * i + 1];
            });
      } else
        launch(
            l.n, DEVICE(int i) {
              int k = src[2 * i], j = src[2 * i + 1];
              double x = k < 0 ? 1 : fabs(a[k]), y = j < 0 ? 0 : fabs(a[j]);
              pw[2 * i] = x + y > 0 ? x / (x + y) : 1;
              pw[2 * i + 1] = 1 - pw[2 * i];
            });
      launch(
          levels[j + 1]->nz, DEVICE(int i) {
            double v = 0;
            for (int k = start[i]; k < start[i + 1]; ++k)
              v += pw[slots[k].x] * a[map[k]] * pw[slots[k].y];
            b[i] = v;
          });
    }
  }
  vector<unsigned> invalid(levels.size());
  for (size_t j = 0; j < levels.size(); ++j)
    CHECK(cudaMemcpyAsync(&invalid[j], levels[j]->invalid.p, sizeof(unsigned),
                          cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));
  for (size_t j = 0; j < levels.size(); ++j) {
    if (invalid[j] & 1u)
      throw std::runtime_error(
          "factor setup cannot store a positive FP32 diagonal or unit-energy row");
    levels[j]->use_factored = !(invalid[j] & 2u);
    if (levels[j]->dense.p || j == 0 || !coarse_fp8) {
      levels[j]->use_fp8_level = false;
    } else {
      bool fp8_ok = ((invalid[j] & 4u) == 0u) && ((invalid[j] & 16u) != 0u);
      if (fp8_ok) {
        // Verify every connected component of the decoded FP8 graph contains at
        // least one strictly grounded row (guaranteed SPD by SDDM M-matrix).
        auto &lv = *levels[j];
        vector<int> h_rp(lv.n + 1), h_ci(lv.nz);
        vector<uint8_t> h_fp8(lv.nz);
        vector<float> h_diag(lv.n);
        CHECK(cudaMemcpy(h_rp.data(), lv.rp.p, (lv.n + 1) * sizeof(int),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_ci.data(), lv.ci.p, lv.nz * sizeof(int),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_fp8.data(), lv.ac_fp8.p, lv.nz * sizeof(uint8_t),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_diag.data(), lv.ac_diag.p, lv.n * sizeof(float),
                         cudaMemcpyDeviceToHost));
        vector<uint8_t> grounded(lv.n, 0);
        for (int i = 0; i < lv.n; ++i) {
          double row_s = static_cast<double>(h_diag[i]), row_c = 0.0;
          for (int k = h_rp[i]; k < h_rp[i + 1]; ++k) {
            if (h_ci[k] == i) continue;
            __half_raw hr = __nv_cvt_fp8_to_halfraw(
                static_cast<__nv_fp8_storage_t>(h_fp8[k]), __NV_E5M2);
            double dec = static_cast<double>(__half2float(__half(hr)));
            double t = row_s + dec;
            row_c += (std::fabs(row_s) >= std::fabs(dec)) ? ((row_s - t) + dec)
                                                          : ((dec - t) + row_s);
            row_s = t;
          }
          if (row_s + row_c > 0.0)
            grounded[i] = 1;
        }
        vector<uint8_t> seen(lv.n, 0);
        vector<int> q;
        q.reserve(lv.n);
        for (int s0 = 0; s0 < lv.n && fp8_ok; ++s0) {
          if (seen[s0]) continue;
          q.clear();
          q.push_back(s0);
          seen[s0] = 1;
          bool comp_grounded = (grounded[s0] != 0);
          for (size_t head = 0; head < q.size(); ++head) {
            int u = q[head];
            for (int k = h_rp[u]; k < h_rp[u + 1]; ++k) {
              int c = h_ci[k];
              if (c == u || seen[c]) continue;
              __half_raw hr = __nv_cvt_fp8_to_halfraw(
                  static_cast<__nv_fp8_storage_t>(h_fp8[k]), __NV_E5M2);
              if (__half2float(__half(hr)) != 0.0f) {
                seen[c] = 1;
                comp_grounded |= (grounded[c] != 0);
                q.push_back(c);
              }
            }
          }
          if (!comp_grounded) fp8_ok = false;
        }
      }
      if (!fp8_ok) {
        levels[j]->use_fp8_level = false;
        levels[j]->free_fp8_buffers();
        if (!levels[j]->af.p)
          levels[j]->alloc_af_buffer();
        auto af = levels[j]->af.p;
        auto a = levels[j]->a.p;
        launch(levels[j]->nz,
               DEVICE(int k) { af[k] = static_cast<float>(a[k]); });
        CHECK(cudaStreamSynchronize(stream));
      } else {
        levels[j]->use_fp8_level = true;
        levels[j]->free_af_buffer();
      }
    }
  }
}
template <class T> Work<T> &work(Level &l) {
  if constexpr (sizeof(T) == 4)
    return l.mix;
  else
    return l.full;
}
__device__ double weight(int i, int j, int base, Method m, const __half *half,
                         const unsigned char *packed, const float *scale,
                         const double *exact) {
  if (m == Double)
    return exact[base + j];
  if (m == Half)
    return double(scale[i]) * __half2float(half[base + j]);
  int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
  return m == Int4Raw ? double(scale[i]) * ((q ^ 8) - 8)
                       : double(scale[i]) * (((q ^ 8) - 8) / 7.0);
}
template <class T> void smooth(Level &l, Method m, bool add) {
  auto &w = work<T>(l);
  auto x = w.x.p, r = add ? w.res.p : w.rhs.p, tmp = w.tmp.p;
  auto rp = l.rp.p, ci = l.ci.p;
  auto d = l.diag.p, s = l.scale.p;
  auto half = l.half.p;
  auto packed = l.packed.p;
  auto exact = l.exact.p;
  if (m == Jacobi) {
    const T *inv;
    if constexpr (sizeof(T) == 4)
      inv = l.inv.p;
    else
      inv = l.l1.p;
    launch(l.n, DEVICE(int i) { x[i] = (add ? x[i] : T(0)) + inv[i] * r[i]; });
    return;
  }
  const int k_eff = l.cap_three ? 3 : MaxLower;
  const double damping = l.max_lower <= k_eff ? 0.65 : 0.35;
  if constexpr (sizeof(T) == 4) {
    if (m != Double && l.use_factored) {
      constexpr float kInv7 = 1.0f / 7.0f;
      const bool is_half = (m == Half);
      const bool is_int4 = (m == Int4);
      auto oor = l.oor.p;
      auto dtmp = l.full.tmp.p;
      CHECK(cudaMemsetAsync(oor, 0, sizeof(unsigned), stream));
      launch(
          l.n, DEVICE(int i) {
            int base = rp[i], k = base;
            const float sc = is_int4 ? (s[i] * kInv7) : s[i];
            const float r_i = r[i];
            bool any_nz = (r_i != 0.0f);
            float t_i = d[i] * r_i;
            if (is_half) {
              for (; k < rp[i + 1] && ci[k] < i; ++k) {
                const float w = __half2float(half[k]);
                const float r_c = r[ci[k]];
                if (w != 0.0f && r_c != 0.0f) any_nz = true;
                t_i = fmaf(sc * w, r_c, t_i);
              }
            } else {
              for (; k < rp[i + 1] && ci[k] < i; ++k) {
                int j = k - base;
                int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                const float w = float((q ^ 8) - 8);
                const float r_c = r[ci[k]];
                if (w != 0.0f && r_c != 0.0f) any_nz = true;
                t_i = fmaf(sc * w, r_c, t_i);
              }
            }
            float y_i = sc * t_i;
            tmp[i] = y_i;
            if (!isfinite(y_i) ||
                (t_i != 0.0f && (fabsf(t_i) < FLT_MIN || fabsf(y_i) < FLT_MIN)) ||
                (any_nz && fabsf(t_i) < FLT_MIN))
              atomicOr(oor, 1u);
          });
      launch(
          l.n, DEVICE(int i) {
            if (*oor)
              return;
            int base = rp[i];
            const float sc = is_int4 ? (s[i] * kInv7) : s[i];
            float v = 0.0f;
            if (is_half) {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fmaf(d[i], tmp[i] / sc, v);
                else
                  v = fmaf(__half2float(half[k]), tmp[c], v);
              }
            } else {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fmaf(d[i], tmp[i] / sc, v);
                else {
                  int j = k - base;
                  int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                  v = fmaf(float((q ^ 8) - 8), tmp[c], v);
                }
              }
            }
            double dv = double(v);
            if (!isfinite(v) || (v != 0.0f && fabsf(v) < FLT_MIN)) {
              dv = 0.0;
              if (is_half) {
                for (int k = base; k < rp[i + 1]; ++k) {
                  int c = ci[k];
                  if (c < i) continue;
                  if (c == i)
                    dv = fma(double(d[i]), double(tmp[i]) / double(sc), dv);
                  else
                    dv = fma(double(__half2float(half[k])), double(tmp[c]), dv);
                }
              } else {
                for (int k = base; k < rp[i + 1]; ++k) {
                  int c = ci[k];
                  if (c < i) continue;
                  if (c == i)
                    dv = fma(double(d[i]), double(tmp[i]) / double(sc), dv);
                  else {
                    int j = k - base;
                    int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                    dv = fma(double((q ^ 8) - 8), double(tmp[c]), dv);
                  }
                }
              }
            }
            x[i] = float((add ? double(x[i]) : 0.0) + damping * dv);
          });
      launch(
          l.n, DEVICE(int i) {
            if (!*oor)
              return;
            int base = rp[i], k = base;
            const double sc = is_int4 ? (double(s[i]) / 7.0) : double(s[i]);
            double t_i = double(d[i]) * double(r[i]);
            if (is_half) {
              for (; k < rp[i + 1] && ci[k] < i; ++k)
                t_i = fma(sc * double(__half2float(half[k])), double(r[ci[k]]), t_i);
            } else {
              for (; k < rp[i + 1] && ci[k] < i; ++k) {
                int j = k - base;
                int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                t_i = fma(sc * double((q ^ 8) - 8), double(r[ci[k]]), t_i);
              }
            }
            dtmp[i] = t_i;
          });
      launch(
          l.n, DEVICE(int i) {
            if (!*oor)
              return;
            int base = rp[i];
            double v = 0.0;
            if (is_half) {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fma(double(d[i]), dtmp[i], v);
                else
                  v = fma(double(s[c]) * double(__half2float(half[k])), dtmp[c], v);
              }
            } else {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fma(double(d[i]), dtmp[i], v);
                else {
                  int j = k - base;
                  int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                  const double sc_c = is_int4 ? (double(s[c]) / 7.0) : double(s[c]);
                  v = fma(sc_c * double((q ^ 8) - 8), dtmp[c], v);
                }
              }
            }
            x[i] = float((add ? double(x[i]) : 0.0) + damping * v);
          });
      return;
    }
    if (m != Double) {
      // Unfactored extreme-scale path: promote intermediate and accumulation to double
      // to prevent FP32 overflow/underflow when |s_i|/7 lies outside [1e-12, 1e12].
      const bool is_half = (m == Half);
      const bool is_int4 = (m == Int4);
      auto dtmp = l.full.tmp.p;
      launch(
          l.n, DEVICE(int i) {
            int base = rp[i], k = base;
            const double sc = is_int4 ? (double(s[i]) / 7.0) : double(s[i]);
            double v = double(d[i]) * double(r[i]);
            if (is_half) {
              for (; k < rp[i + 1] && ci[k] < i; ++k)
                v = fma(sc * double(__half2float(half[k])), double(r[ci[k]]), v);
            } else {
              for (; k < rp[i + 1] && ci[k] < i; ++k) {
                int j = k - base;
                int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                v = fma(sc * double((q ^ 8) - 8), double(r[ci[k]]), v);
              }
            }
            dtmp[i] = v;
          });
      launch(
          l.n, DEVICE(int i) {
            int base = rp[i];
            double v = 0.0;
            if (is_half) {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fma(double(d[i]), dtmp[i], v);
                else
                  v = fma(double(s[c]) * double(__half2float(half[k])), dtmp[c], v);
              }
            } else {
              for (int k = base; k < rp[i + 1]; ++k) {
                int c = ci[k];
                if (c < i)
                  continue;
                if (c == i)
                  v = fma(double(d[i]), dtmp[i], v);
                else {
                  int j = k - base;
                  int q = (packed[(base + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
                  const double sc_c = is_int4 ? (double(s[c]) / 7.0) : double(s[c]);
                  v = fma(sc_c * double((q ^ 8) - 8), dtmp[c], v);
                }
              }
            }
            x[i] = float((add ? double(x[i]) : 0.0) + damping * v);
          });
      return;
    }
  }
  launch(
      l.n, DEVICE(int i) {
        int base = rp[i], k = base;
        T v = 0;
        for (; k < rp[i + 1] && ci[k] < i; k++) {
          v += T(weight(i, k - base, base, m, half, packed, s, exact)) *
               r[ci[k]];
        }
        tmp[i] = v + T(m == Double ? exact[k] : d[i]) * r[i];
      });
  launch(
      l.n, DEVICE(int i) {
        T v = 0;
        for (int k = rp[i]; k < rp[i + 1]; k++) {
          int c = ci[k];
          if (c < i)
            continue;
          if (c == i) {
            v += T(m == Double ? exact[k] : d[i]) * tmp[i];
            continue;
          }
          if (m == Double)
            v += T(exact[k]) * tmp[c];
          else if (m == Half)
            v += T(double(s[c]) * __half2float(half[k])) * tmp[c];
          else {
            int j = k - rp[i];
            int q = (packed[(rp[i] + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
            double off = m == Int4Raw ? double(s[c]) * ((q ^ 8) - 8)
                                       : double(s[c]) * (((q ^ 8) - 8) / 7.0);
            v += T(off) * tmp[c];
          }
        }
        x[i] = (add ? x[i] : T(0)) + T(damping) * v;
      });
}
template <class T>
void residual(Level &l, const T *b, const T *x, T *r, bool use_fp8 = false) {
  auto rp = l.rp.p, ci = l.ci.p;
  if constexpr (sizeof(T) == 4) {
    if (use_fp8 && l.use_fp8_level) {
      auto ac_fp8 = l.ac_fp8.p;
      auto ac_diag = l.ac_diag.p;
      launch(
          l.n, DEVICE(int i) {
            float sum = 0.0f;
            for (int k = rp[i]; k < rp[i + 1]; ++k) {
              int c = ci[k];
              float v = (c == i)
                            ? ac_diag[i]
                            : __half2float(__nv_cvt_fp8_to_halfraw(
                                  static_cast<__nv_fp8_storage_t>(ac_fp8[k]),
                                  __NV_E5M2));
              sum += v * x[c];
            }
            r[i] = b ? b[i] - sum : sum;
          });
      return;
    }
  }
  const T *a;
  if constexpr (sizeof(T) == 4)
    a = l.af.p;
  else
    a = l.a.p;
  launch(
      l.n, DEVICE(int i) {
        T sum = 0;
        for (int k = rp[i]; k < rp[i + 1]; ++k)
          sum += a[k] * x[ci[k]];
        r[i] = b ? b[i] - sum : sum;
      });
}
template <class T>
void cycle(Levels &ls, size_t j, Method m, bool coarse_fp8 = false,
           int poly_stride = 0, int poly_offset = 0, float poly_scale = 1.0f) {
  auto &l = *ls[j];
  auto &w = work<T>(l);
  const bool use_fp8 = coarse_fp8 && (j > 0);
  if (l.dense.p) {
    int n = l.n;
    auto L = l.dense.p;
    auto b = w.rhs.p, x = w.x.p;
    launch(
        1, DEVICE(int) {
          double y[64];
          for (int i = 0; i < n; ++i) {
            double v = double(b[i]);
            for (int j = 0; j < i; ++j)
              v -= L[i * n + j] * y[j];
            y[i] = v / L[i * n + i];
          }
          for (int i = n - 1; i >= 0; --i) {
            double v = y[i];
            for (int j = i + 1; j < n; ++j)
              v -= L[j * n + i] * y[j];
            y[i] = v / L[i * n + i];
            x[i] = static_cast<T>(y[i]);
          }
        });
    return;
  }
  smooth<T>(l, m, false);
  residual(l, w.rhs.p, w.x.p, w.res.p, use_fp8);
  auto &coarse = work<T>(*ls[j + 1]);
  auto cb = coarse.rhs.p, r = w.res.p, cx = coarse.x.p, x = w.x.p;
  auto ms = l.mstart.p, mem = l.members.p, col = l.interp.p;
  auto pw = l.pw.p;
  launch(
      l.nc, DEVICE(int i) {
        T sum = 0;
        for (int k = ms[i]; k < ms[i + 1]; ++k)
          sum += T(pw[mem[k]]) * r[mem[k] / 2];
        cb[i] = sum;
      });
  const bool boost_child = (poly_stride > 0) &&
                           (static_cast<int>(j) % poly_stride == poly_offset) &&
                           !ls[j + 1]->dense.p;
  if (boost_child) {
    int cn = ls[j + 1]->n;
    auto prhs = coarse.poly_rhs.p, pacc = coarse.poly_acc.p;
    launch(cn, DEVICE(int i) { prhs[i] = cb[i]; });
    cycle<T>(ls, j + 1, m, coarse_fp8, poly_stride, poly_offset, poly_scale);
    launch(cn, DEVICE(int i) { pacc[i] = T(poly_scale) * cx[i]; });
    residual(*ls[j + 1], prhs, pacc, cb, coarse_fp8);
    cycle<T>(ls, j + 1, m, coarse_fp8, poly_stride, poly_offset, poly_scale);
    launch(cn, DEVICE(int i) { cx[i] = pacc[i] + T(poly_scale) * cx[i]; });
  } else {
    cycle<T>(ls, j + 1, m, coarse_fp8, poly_stride, poly_offset, poly_scale);
  }
  launch(
      l.n, DEVICE(int i) {
        x[i] += T(pw[2 * i]) * cx[col[2 * i]];
        if (col[2 * i + 1] >= 0)
          x[i] += T(pw[2 * i + 1]) * cx[col[2 * i + 1]];
      });
  residual(l, w.rhs.p, x, r, use_fp8);
  smooth<T>(l, m, true);
}
__global__ void reduce(int n, const double *x, const double *y, double *out) {
  __shared__ double sums[256];
  double sum = 0;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
    sum += x[i] * (y ? y[i] : 1.);
  int t = threadIdx.x;
  sums[t] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (t < stride)
      sums[t] += sums[t + stride];
    __syncthreads();
  }
  if (!t)
    out[blockIdx.x] = sums[0];
}
struct Result {
  int iterations = 0;
  std::string trigger = "none";
  float recovery = 0;
  double rel = INFINITY;
};
struct Engine {
  Levels ls;
  Buffer<double> rhs, solution, res, dir, product, z, partial, scalar;
  cudaGraphExec_t graphs[5] = {};
  vector<uint8_t> graph_factored[5];
  vector<uint64_t> graph_gen[5];
  int n;
  double rhs_norm_sq;
  bool inject = false;
  bool graph;
  bool coarse_fp8 = false;
  int poly_stride = 0;
  int poly_offset = 0;
  float poly_scale = 1.0f;
  Engine(const HostCSR &a, const vector<double> &b, bool direct, bool enable,
         bool sa = false, bool c_fp8 = false, int p_stride = 0,
         float p_scale = 1.0f, int p_offset = 0, bool only_int4 = false,
         bool allow_unsafe_poly_scale = false)
      : n(a.n), graph(enable), coarse_fp8(c_fp8), poly_stride(p_stride),
        poly_offset(p_offset), poly_scale(p_scale) {
    if (poly_stride < 0 ||
        (poly_stride > 0 && (poly_offset < 0 || poly_offset >= poly_stride)) ||
        !(std::isfinite(poly_scale) && poly_scale > 0.0f) ||
        (poly_stride > 0 && !allow_unsafe_poly_scale && poly_scale >= 2.0f))
      throw std::runtime_error(
          "invalid polynomial boost configuration (poly_scale must lie in (0, 2) unless unsafe_poly is enabled)");
    HostCSR coarse;
    const HostCSR *cur = &a;
    size_t level_idx = 0;
    do {
      ls.emplace_back(std::make_unique<Level>());
      coarse = symbolic(*ls.back(), *cur, direct, sa, level_idx++, coarse_fp8,
                        only_int4);
      cur = &coarse;
    } while (coarse.n);
    rhs.init(b);
    rhs_norm_sq = std::inner_product(b.begin(), b.end(), b.begin(), 0.);
    solution.alloc(n);
    res.alloc(n);
    dir.alloc(n);
    product.alloc(n);
    z.alloc(n);
    partial.alloc(512);
    scalar.alloc(1);
  }
  ~Engine() {
    for (auto g : graphs)
      if (g)
        cudaGraphExecDestroy(g);
  }
  double dot(const double *x, const double *y) {
    int blocks = std::min(512, (n + 255) / 256);
    reduce<<<blocks, 256, 0, stream>>>(n, x, y, partial.p);
    CHECK(cudaGetLastError());
    reduce<<<1, 256, 0, stream>>>(blocks, partial.p, nullptr, scalar.p);
    CHECK(cudaGetLastError());
    double v;
    CHECK(cudaMemcpyAsync(&v, scalar.p, sizeof(v), cudaMemcpyDeviceToHost,
                          stream));
    CHECK(cudaStreamSynchronize(stream));
    return v;
  }
  template <class T> void apply(Method m) {
    auto &w = work<T>(*ls[0]);
    auto r = res.p, out = z.p;
    auto b = w.rhs.p, x = w.x.p;
    launch(n, DEVICE(int i) { b[i] = r[i]; });
    const bool use_fp8 = (sizeof(T) == 4) && coarse_fp8;
    const int p_stride = (sizeof(T) == 4) ? poly_stride : 0;
    cycle<T>(ls, 0, m, use_fp8, p_stride, poly_offset, poly_scale);
    launch(n, DEVICE(int i) { out[i] = x[i]; });
  }
  vector<uint8_t> factored_state() const {
    vector<uint8_t> state(ls.size(), 0);
    for (size_t j = 0; j < ls.size(); ++j)
      state[j] = uint8_t(ls[j]->use_factored ? 1u : 0u) |
                 uint8_t(ls[j]->use_fp8_level ? 2u : 0u);
    return state;
  }
  vector<uint64_t> alloc_generations() const {
    vector<uint64_t> gens(ls.size(), 0);
    for (size_t j = 0; j < ls.size(); ++j)
      gens[j] = ls[j]->alloc_generation;
    return gens;
  }
  void invalidate_stale_graphs() {
    vector<uint64_t> gens = alloc_generations();
    for (int k = 0; k < 5; ++k) {
      if (graphs[k] && graph_gen[k] != gens) {
        CHECK(cudaGraphExecDestroy(graphs[k]));
        graphs[k] = nullptr;
        graph_factored[k].clear();
        graph_gen[k].clear();
      }
    }
  }
  void reset(Method method, const double *values) {
    ::reset(ls, method, values, coarse_fp8);
    invalidate_stale_graphs();
  }
  void capture(Method m) {
    if (!graph)
      return;
    vector<uint8_t> state = factored_state();
    vector<uint64_t> gens = alloc_generations();
    if (graphs[m] && graph_factored[m] == state && graph_gen[m] == gens)
      return;
    if (graphs[m]) {
      CHECK(cudaGraphExecDestroy(graphs[m]));
      graphs[m] = nullptr;
    }
    cudaGraph_t g;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    apply<float>(m);
    CHECK(cudaStreamEndCapture(stream, &g));
    CHECK(cudaGraphInstantiate(&graphs[m], g, 0));
    CHECK(cudaGraphDestroy(g));
    graph_factored[m] = std::move(state);
    graph_gen[m] = std::move(gens);
  }
  Result solve(Method m) {
    if (graph)
      capture(m);
    Result result;
    auto r = res.p, b = rhs.p, x = solution.p, p = dir.p, q = product.p,
         v = z.p;
    launch(
        n, DEVICE(int i) {
          r[i] = b[i];
          x[i] = 0;
        });
    double b2 = rhs_norm_sq;
    std::unique_ptr<Timer> recovery;
    for (int stage = 0; stage < 2; ++stage) {
      if (stage) {
        recovery = std::make_unique<Timer>();
        residual(*ls[0], b, x, r);
        if (!std::isfinite(dot(r, r))) {
          launch(
              n, DEVICE(int i) {
                x[i] = 0;
                r[i] = b[i];
              });
          result.trigger += "/restart_zero";
        }
      }
      double prev = 0, best = INFINITY;
      int stale = 0;
      std::string reason = "iteration_limit";
      for (int it = 0; it < 2000; ++it) {
        if (stage)
          apply<double>(Jacobi);
        else if (graph)
          CHECK(cudaGraphLaunch(graphs[m], stream));
        else
          apply<float>(m);
        if (inject && !stage && !it)
          launch(n, DEVICE(int i) { v[i] = -v[i]; });
        double rho = dot(r, v);
        if (!(rho > 0 && std::isfinite(rho))) {
          reason = "rz";
          break;
        }
        double beta = it ? rho / prev : 0;
        launch(n, DEVICE(int i) { p[i] = it ? v[i] + beta * p[i] : v[i]; });
        residual(*ls[0], (const double *)nullptr, p, q);
        double den = dot(p, q);
        if (!(den > 0 && std::isfinite(den))) {
          reason = "curvature";
          break;
        }
        double alpha = rho / den;
        launch(
            n, DEVICE(int i) {
              x[i] += alpha * p[i];
              r[i] -= alpha * q[i];
            });
        ++result.iterations;
        double err = sqrt(dot(r, r) / b2);
        prev = rho;
        if (err < 7e-9) {
          residual(*ls[0], b, x, q);
          if (sqrt(dot(q, q) / b2) <= 1e-8) {
            reason = "ok";
            break;
          }
          reason = "residual_drift";
          break;
        }
        if (err < best * .99) {
          best = err;
          stale = 0;
        } else if (++stale >= 100) {
          reason = "stagnation";
          break;
        }
      }
      if (stage) {
        result.recovery = recovery->stop();
        if (reason != "ok")
          result.trigger += "/" + reason;
      }
      if (reason == "ok")
        break;
      if (!stage)
        result.trigger = reason;
    }
    return result;
  }
};

void test_recovery() {
  HostCSR a{2, 4, {0, 2, 4}, {0, 1, 0, 1}, {2, -1, -1, 2}};
  vector<double> rhs{1, 2};
  Engine engine(a, rhs, true, true);
  Buffer<double> values;
  values.init(a.val);
  reset(engine.ls, Jacobi, values.p);
  engine.capture(Jacobi);
  engine.inject = true;
  Result result = engine.solve(Jacobi);
  double x[2];
  CHECK(cudaMemcpyAsync(x, engine.solution.p, sizeof(x), cudaMemcpyDeviceToHost,
                        stream));
  CHECK(cudaStreamSynchronize(stream));
  double rel = std::hypot(1 - 2 * x[0] + x[1], 2 + x[0] - 2 * x[1]) / sqrt(5.);
  if (result.trigger != "rz" || !(rel <= 1e-8) || !(result.recovery > 0))
    throw std::runtime_error("FP64 recovery self-test failed");
  printf("Injected negative rTz: trigger=%s, recovered residual=%.12e, "
         "recovery=%.4f ms\n",
         result.trigger.c_str(), rel, result.recovery);
}
#include "solver_checks.hpp"

int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    test_recovery();
    test_numeric_refresh();
    test_smoothed_aggregation_anchor_refresh();
    test_invalid_coarse_cholesky_rejected();
    test_int4_raw_projection_control();
    test_fp8_coarse_and_poly_boost();
    test_fp8_symmetrization_and_grounding_preservation();
    test_dense_polynomial_boost_oracle();
    test_factored_refresh_and_guards();
    test_fp8_sddm_counterexamples_and_neumaier();
    test_fp8_graph_reallocation_safety();
    test_terminal_tiny_cholesky_and_poly_boundaries();
    CHECK(cudaStreamDestroy(stream));
    return 0;
  }
  if (argc >= 3 && std::string(argv[1]) == "--pattern-stats") {
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    puts("case_label,levels,rows,nnz,full_lower_nnz,selected_lower_nnz,"
         "truncated_rows,int4_weight_bytes,fp64_mirrored_bytes,"
         "fp16_mirrored_bytes");
    for (int argi = 2; argi < argc; ++argi) {
      std::string cname = argv[argi];
      auto load_case = [&](const std::string &nm) {
        if (nm == "grid128" || nm == "grid192" || nm == "grid16")
          return build_3d_structured(std::stoi(nm.substr(4)), 0);
        if (nm == "elastic28")
          return build_3d_elasticity(28, 0);
        if (nm == "tet48")
          return build_p1_3d_tet_fem(48, 0);
        if (std::filesystem::exists(nm))
          return load_mtx_csr(nm);
        const char *dir = std::getenv("MTX_DIR");
        std::string base = dir ? std::string(dir) : ".";
        return load_mtx_csr(base + "/" + nm);
      };
      auto a = load_case(cname);
      vector<double> b(a.n, 1.0);
      Engine engine(a, b, /*direct=*/true, /*graph=*/false, /*sa=*/false,
                    /*coarse_fp8=*/false, 0, 1.0f, 0, /*only_int4=*/false,
                    false);
      size_t levels = 0, rows = 0, nnz = 0, full_lower = 0, sel_lower = 0,
             trunc = 0, wi4 = 0, w64 = 0, w16 = 0;
      for (const auto &lp : engine.ls) {
        const auto &l = *lp;
        if (l.dense.p)
          continue;
        ++levels;
        rows += static_cast<size_t>(l.n);
        nnz += static_cast<size_t>(l.nz);
        full_lower += l.lower_nnz;
        sel_lower += l.selected_lower_nnz;
        trunc += l.truncated_rows;
        wi4 += l.bytes[Int4];
        w64 += l.bytes[Double];
        w16 += l.bytes[Half];
      }
      std::string label = std::filesystem::path(cname).filename().string();
      printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", label.c_str(), levels,
             rows, nnz, full_lower, sel_lower, trunc, wi4, w64, w16);
      fflush(stdout);
    }
    CHECK(cudaStreamDestroy(stream));
    return 0;
  }
  if (argc < 3)
    throw std::runtime_error(
        "solver CASE REPEATS [injection|sa] [eager] [int4|rawint4] [cap3] "
        "[fp8coarse] [polySTRIDE[@OFFSET][:SCALE]] [unsafe_poly]");
  std::string name = argv[1];
  int trials = std::stoi(argv[2]);
  if (trials < 1)
    throw std::runtime_error("repeat count must be positive");
  bool direct = true, graph = true, only = false, raw_int4 = false, sa = false,
       cap_three = false, coarse_fp8 = false, allow_unsafe_poly = false;
  int poly_stride = 0, poly_offset = 0;
  float poly_scale = 1.0f;
  for (int i = 3; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "fp8coarse") {
      coarse_fp8 = true;
      continue;
    }
    if (flag == "unsafe_poly") {
      allow_unsafe_poly = true;
      continue;
    }
    if (flag.rfind("poly", 0) == 0 && flag.size() >= 5) {
      std::string rest = flag.substr(4);
      auto colon = rest.find(':');
      if (colon != std::string::npos) {
        std::string scale_str = rest.substr(colon + 1);
        if (scale_str.empty())
          throw std::runtime_error("invalid poly scale");
        size_t pos = 0;
        poly_scale = std::stof(scale_str, &pos);
        if (pos != scale_str.size() || !std::isfinite(poly_scale) ||
            !(poly_scale > 0.0f))
          throw std::runtime_error("poly scale must be finite and positive");
        rest = rest.substr(0, colon);
      }
      auto at = rest.find('@');
      if (at != std::string::npos) {
        std::string off_str = rest.substr(at + 1);
        if (off_str.empty())
          throw std::runtime_error("invalid poly offset");
        size_t pos = 0;
        poly_offset = std::stoi(off_str, &pos);
        if (pos != off_str.size())
          throw std::runtime_error("invalid poly offset");
        rest = rest.substr(0, at);
      }
      if (rest.empty())
        throw std::runtime_error("invalid poly stride");
      size_t pos = 0;
      poly_stride = std::stoi(rest, &pos);
      if (pos != rest.size() || poly_stride < 1 || poly_offset < 0 ||
          poly_offset >= poly_stride)
        throw std::runtime_error("invalid poly stride/offset");
      continue;
    }
    if (flag != "injection" && flag != "eager" && flag != "int4" &&
        flag != "rawint4" && flag != "sa" && flag != "cap3")
      throw std::runtime_error("unknown option " + flag);
    cap_three |= flag == "cap3";
    sa |= flag == "sa";
    direct &= flag != "injection" && flag != "sa";
    graph &= flag != "eager";
    raw_int4 |= flag == "rawint4";
    only |= flag == "int4" || raw_int4;
  }
  if (raw_int4 && only && argc > 3) {
    for (int i = 3; i < argc; ++i)
      if (std::string(argv[i]) == "int4")
        throw std::runtime_error("choose either int4 or rawint4");
  }
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto matrix = [&](int step) {
    if (name == "grid128" || name == "grid192" || name == "grid16")
      return build_3d_structured(std::stoi(name.substr(4)), step);
    if (name == "elastic28")
      return build_3d_elasticity(28, step);
    if (name == "tet48")
      return build_p1_3d_tet_fem(48, step);
    if (std::filesystem::exists(name))
      return load_mtx_csr(name);
    const char *dir = std::getenv("MTX_DIR");
    std::string base = dir ? std::string(dir) : ".";
    return load_mtx_csr(base + "/" + name);
  };
  auto a = matrix(0), next = matrix(2);
  assert_same_pattern(a, next, name, 2);
  std::mt19937 rng(12345);
  std::normal_distribution<double> nd(0, 1);
  vector<double> b(a.n);
  for (auto &v : b)
    v = nd(rng);
  Buffer<double> initial, changed;
  initial.init(a.val);
  changed.init(next.val);
  auto start = std::chrono::steady_clock::now();
  Engine engine(a, b, direct, graph, sa, coarse_fp8, poly_stride, poly_scale,
                poly_offset, only, allow_unsafe_poly);
  for (auto &l : engine.ls)
    l->cap_three = cap_three;
  CHECK(cudaDeviceSynchronize());
  double symbolic_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  vector<double> x(a.n);
  bool all_verified = true;
  puts("case,trial,phase,method,direct,graph,fp8coarse,poly_stride,poly_offset,"
       "poly_scale,symbolic_ms,reset_ms,solve_ms,iters,true_rel,trigger,"
       "recovery_ms,pass,factored_levels,weight_bytes,coarse_bytes,"
       "resident_bytes,index_bytes,diagonal_fallbacks");
  for (int t = 0; t < trials; ++t)
    for (int j = 0; j < (only ? 1 : 4); ++j) {
      Method m = raw_int4 ? Int4Raw : only ? Int4 : Method((t + j) % 4);
      for (int step : {0, 2}) {
        Timer setup;
        engine.reset(m, step ? changed.p : initial.p);
        engine.capture(m);
        float reset_ms = setup.stop();
        Timer solve;
        Result rr = engine.solve(m);
        float solve_ms = solve.stop();
        CHECK(cudaMemcpy(x.data(), engine.solution.p, a.n * sizeof(double),
                         cudaMemcpyDeviceToHost));
        const auto &h = step ? next : a;
        double r2 = 0, b2 = 0;
        for (int i = 0; i < a.n; ++i) {
          double ax = 0;
          for (int k = h.rp[i]; k < h.rp[i + 1]; ++k)
            ax += h.val[k] * x[h.ci[k]];
          double r = b[i] - ax;
          r2 += r * r;
          b2 += b[i] * b[i];
        }
        rr.rel = sqrt(r2 / b2);
        bool passed = std::isfinite(rr.rel) && rr.rel <= 1e-8;
        all_verified &= passed;
        size_t weight_bytes = 0, coarse_bytes = 0, resident_bytes = 0;
        unsigned fall = 0;
        int factored_cnt = 0, nonterminal_cnt = 0;
        for (size_t lv = 0; lv < engine.ls.size(); ++lv) {
          auto &l = *engine.ls[lv];
          weight_bytes += l.bytes[m];
          resident_bytes += l.allocated_smoother_bytes();
          if (lv > 0) {
            coarse_bytes += l.active_coarse_bytes(coarse_fp8);
            resident_bytes += l.allocated_coarse_bytes();
          }
          if (!l.dense.p) {
            ++nonterminal_cnt;
            if (m != Jacobi && m != Double && l.use_factored)
              ++factored_cnt;
            if (l.failures.p) {
              unsigned v;
              CHECK(cudaMemcpy(&v, l.failures.p, sizeof(v),
                               cudaMemcpyDeviceToHost));
              fall += v;
            }
          }
        }
        printf("%s,%d,%s,%s,%d,%d,%d,%d,%d,%.6g,%.4f,%.5f,%.5f,%d,%.12e,%s,"
               "%.4f,%d,%d/%d,%zu,%zu,%zu,0,%u\n",
               name.c_str(), t, step ? "timed" : "warm",
               m == Int4Raw  ? "rawint4"
               : m == Int4   ? "int4"
               : m == Double ? "double"
               : m == Half   ? "half"
                             : "jacobi",
               direct, graph, coarse_fp8 ? 1 : 0, poly_stride, poly_offset,
               poly_scale, symbolic_ms, reset_ms, solve_ms, rr.iterations,
               rr.rel, rr.trigger.c_str(), rr.recovery, passed, factored_cnt,
               nonterminal_cnt, weight_bytes, coarse_bytes, resident_bytes,
               fall);
        fflush(stdout);
      }
    }
  CHECK(cudaStreamDestroy(stream));
  return all_verified ? 0 : 2;
} catch (const std::exception &e) {
  fprintf(stderr, "%s\n", e.what());
  return 1;
}
