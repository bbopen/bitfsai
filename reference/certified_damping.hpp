#pragma once
// Standalone host reference for Theorem 5 (Certified Damping Guard):
// Computes a conservative IEEE binary64 upper bound U_upper >= || |G| |A| |G|^T 1 ||_inf
// and downward-rounded safe damping omega_lower <= min(18/25, 1 / U_upper) in three
// sparse passes (C++17, IEEE binary64, round-to-nearest, gradual underflow, no -ffast-math).
//
// Input A must be a full symmetric CSR matrix (A == A^T) with explicit finite
// positive diagonals (A_{ii} > 0); G is a split lower-triangular factor with
// positive diagonals and strictly lower INT4 codes in [-7, 7].
//
// Caller SPD obligation / precondition:
//   This O(nnz) routine validates CSR structure, finiteness, exact symmetry
//   (A == A^T), and positive diagonals (A_{ii} > 0), but it does NOT prove that
//   A is symmetric positive definite (e.g., an indefinite symmetric matrix can
//   still have positive diagonals). Positive definiteness (A = A^T > 0) is an
//   explicit precondition on the caller: under A = A^T > 0, the returned
//   omega_lower <= min(18/25, 1 / U_upper) certifies
//   lambda_max(omega_lower * G A G^T) <= 1 < 2, hence the damped Gram smoother
//   M_omega = omega_lower * G^T G satisfies M_omega <= A^{-1} (so A^{-1} - M_omega >= 0
//   and the symmetrized error-propagation / two-grid smoother R <= A^{-1} holds).

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bitfsai {
namespace certificate {

struct Csr {
  std::vector<std::size_t> row;
  std::vector<std::size_t> col;
  std::vector<double> value;
};

struct SplitFactor {
  std::vector<std::size_t> row;
  std::vector<std::size_t> col;
  std::vector<std::int8_t> code;
  std::vector<double> scale;
  std::vector<double> diagonal;
};

struct DampingCertificate {
  double U_upper;
  double omega_lower;
};

inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

inline void check_environment() {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                    std::numeric_limits<double>::digits == 53,
                "IEEE binary64 required");
  require(std::fegetround() == FE_TONEAREST, "round-to-nearest required");
  volatile double d = std::numeric_limits<double>::denorm_min(), one = 1.0;
  require(d * one != 0.0, "gradual underflow required; disable FTZ/DAZ");
}

inline double finite_abs(double x) {
  require(std::isfinite(x), "nonfinite matrix/factor value");
  return std::abs(x);
}

inline double upward(double rounded) {
  require(std::isfinite(rounded) && rounded >= 0.0,
          "invalid or overflowed upper bound");
  double up = std::nextafter(rounded, std::numeric_limits<double>::infinity());
  require(std::isfinite(up), "upper bound overflow");
  return up;
}

inline double add_up(double a, double b) {
  if (a == 0.0) return b;
  if (b == 0.0) return a;
  volatile double x = a + b;
  return upward(x);
}

inline double mul_up(double a, double b) {
  if (a == 0.0 || b == 0.0) return 0.0;
  volatile double x = a * b;
  return upward(x);
}

inline double div_up(double a, double b) {
  require(b > 0.0 && std::isfinite(b), "invalid divisor");
  if (a == 0.0) return 0.0;
  volatile double x = a / b;
  return upward(x);
}

inline void check_csr(const std::vector<std::size_t> &row,
                      const std::vector<std::size_t> &col,
                      std::size_t count, std::size_t n) {
  require(row.size() == n + 1 && row.front() == 0 && row.back() == count &&
              col.size() == count,
          "invalid CSR dimensions");
  for (std::size_t i = 0; i < n; ++i)
    require(row[i] <= row[i + 1], "nonmonotonic CSR");
  for (auto j : col) require(j < n, "CSR column outside matrix");
}

// Adapter from BitFSAI INT4_Split row scales (where decoded off-diagonal weights
// are row_scale[i] * decode_nibble(q) = row_scale[i] * float(q) * (1.0f / 7.0f),
// with q in [-7, 7]) into a conservative SplitFactor whose step scale[i] satisfies
// |q| * scale[i] >= max(|row_scale[i] * (q / 7)|, |row_scale[i] * float(q) * (1.0f / 7.0f)|).
inline SplitFactor make_int4_split_factor(
    std::vector<std::size_t> row,
    std::vector<std::size_t> col,
    std::vector<std::int8_t> code,
    const std::vector<double> &row_scale,
    std::vector<double> diagonal) {
  check_environment();
  const std::size_t n = diagonal.size();
  require(n > 0 && row_scale.size() == n, "invalid INT4_Split factor dimensions");
  check_csr(row, col, code.size(), n);
  // Conservative binary64 upper bound on max(1/7, max_{1<=q<=7} float(q)*(1.0f/7.0f)/q).
  const double inv7_upper = upward(static_cast<double>(std::nextafter(1.0f / 7.0f, 1.0f)));
  std::vector<double> step_scale(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double s = finite_abs(row_scale[i]);
    step_scale[i] = std::max(div_up(s, 7.0), mul_up(s, inv7_upper));
  }
  return SplitFactor{std::move(row), std::move(col), std::move(code),
                     std::move(step_scale), std::move(diagonal)};
}

// Adapter extracting a conservative SplitFactor directly from a downloaded
// bitfsai::Smoother (Mode::INT4_Split) built on a symmetric CSR matrix.
inline SplitFactor make_int4_split_factor_from_packed_csr(
    int n,
    const int *rp,
    const int *ci,
    const float *diag,
    const float *row_scale,
    const std::uint8_t *q_packed) {
  check_environment();
  require(n > 0 && rp != nullptr && ci != nullptr &&
              diag != nullptr && row_scale != nullptr,
          "invalid packed CSR inputs");
  require(rp[0] == 0, "packed CSR row pointers must start at 0");
  std::vector<std::size_t> g_row;
  std::vector<std::size_t> g_col;
  std::vector<std::int8_t> g_code;
  std::vector<double> g_scale(static_cast<std::size_t>(n));
  std::vector<double> g_diag(static_cast<std::size_t>(n));
  g_row.reserve(static_cast<std::size_t>(n) + 1);
  g_row.push_back(0);
  for (int i = 0; i < n; ++i) {
    require(rp[i + 1] >= rp[i], "packed CSR row pointers must be non-decreasing");
    g_diag[static_cast<std::size_t>(i)] = static_cast<double>(diag[i]);
    g_scale[static_cast<std::size_t>(i)] = static_cast<double>(row_scale[i]);
    for (int k = rp[i]; k < rp[i + 1]; ++k) {
      const int c = ci[k];
      require(c >= 0 && c < n, "packed CSR column index out of range");
      if (c < i) {
        require(q_packed != nullptr, "missing packed INT4 nibble array");
        const std::uint8_t byte = q_packed[k >> 1];
        const int nib = (k & 1) ? (byte >> 4) : (byte & 15u);
        const int q = nib - 8;
        require(q >= -7 && q <= 7, "decoded packed INT4 code outside [-7, 7]");
        g_col.push_back(static_cast<std::size_t>(c));
        g_code.push_back(static_cast<std::int8_t>(q));
      }
    }
    g_row.push_back(g_col.size());
  }
  return make_int4_split_factor(std::move(g_row), std::move(g_col),
                                std::move(g_code), g_scale, std::move(g_diag));
}

inline DampingCertificate certify_damping(const Csr &A, const SplitFactor &G) {
  check_environment();
  const auto n = G.diagonal.size();
  require(n > 0 && G.scale.size() == n, "invalid factor dimensions");
  check_csr(A.row, A.col, A.value.size(), n);
  check_csr(G.row, G.col, G.code.size(), n);

  std::vector<double> weights(G.code.size()), z(n, 0.0), y(n, 0.0);

  // Validate G rows have unique strictly-lower columns, finite scales, and positive diagonals.
  // Pass 1: z = |G|^T 1
  for (std::size_t i = 0; i < n; ++i) {
    require(G.diagonal[i] > 0.0 && std::isfinite(G.diagonal[i]),
            "invalid factor diagonal");
    auto scale = finite_abs(G.scale[i]);
    z[i] = add_up(z[i], G.diagonal[i]);
    std::vector<std::size_t> g_cols;
    g_cols.reserve(G.row[i + 1] - G.row[i]);
    for (auto k = G.row[i]; k < G.row[i + 1]; ++k) {
      auto j = G.col[k];
      int q = G.code[k];
      require(j < i && q >= -7 && q <= 7,
              "not a valid strict-lower INT4 entry");
      g_cols.push_back(j);
      weights[k] = mul_up(scale, static_cast<double>(std::abs(q)));
      z[j] = add_up(z[j], weights[k]);
    }
    std::sort(g_cols.begin(), g_cols.end());
    require(std::adjacent_find(g_cols.begin(), g_cols.end()) == g_cols.end(),
            "duplicate column index in factor row");
  }

  // Validate A is finite, duplicate-free, symmetric (A == A^T), and has an explicit
  // positive diagonal on every row (rejecting zero rows and nonsymmetric inputs).
  std::vector<std::vector<std::pair<std::size_t, double>>> a_rows(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto &entries = a_rows[i];
    entries.reserve(A.row[i + 1] - A.row[i]);
    bool has_pos_diag = false;
    for (auto k = A.row[i]; k < A.row[i + 1]; ++k) {
      const auto j = A.col[k];
      const double val = A.value[k];
      require(std::isfinite(val), "nonfinite matrix/factor value");
      if (j == i) {
        require(val > 0.0, "matrix diagonal must be positive");
        has_pos_diag = true;
      }
      entries.emplace_back(j, val);
    }
    require(has_pos_diag, "matrix row missing positive diagonal");
    std::sort(entries.begin(), entries.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (std::size_t t = 1; t < entries.size(); ++t) {
      require(entries[t - 1].first != entries[t].first,
              "duplicate column index in matrix row");
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (const auto &entry : a_rows[i]) {
      const auto j = entry.first;
      if (j == i) continue;
      const auto &mirror = a_rows[j];
      const auto it = std::lower_bound(
          mirror.begin(), mirror.end(), i,
          [](const auto &elem, std::size_t col) { return elem.first < col; });
      require(it != mirror.end() && it->first == i && it->second == entry.second,
              "matrix A must be symmetric");
    }
  }

  // Pass 2: y = |A| z
  for (std::size_t i = 0; i < n; ++i) {
    for (auto k = A.row[i]; k < A.row[i + 1]; ++k) {
      y[i] = add_up(y[i], mul_up(finite_abs(A.value[k]), z[A.col[k]]));
    }
  }

  // Pass 3: U = max_i (|G| y)_i
  double U = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double v = mul_up(G.diagonal[i], y[i]);
    for (auto k = G.row[i]; k < G.row[i + 1]; ++k) {
      v = add_up(v, mul_up(weights[k], y[G.col[k]]));
    }
    U = std::max(U, v);
  }
  require(U > 0.0 && std::isfinite(U), "no finite positive certificate");

  // Round downward so stored omega <= 18/25 and omega <= 1/U.
  volatile double rounded_cap = 18.0 / 25.0;
  double cap = std::nextafter(rounded_cap, 0.0);
  double omega = cap;
  if (U >= 1.0) {
    volatile double reciprocal = 1.0 / U;
    omega = std::min(cap, std::nextafter(reciprocal, 0.0));
  }
  require(omega > 0.0 && std::isfinite(omega),
          "no representable certified damping");
  return {U, omega};
}

// Downward-rounded IEEE binary32 helper for callers that require a float scalar:
// guarantees 0 < static_cast<double>(out) <= omega_f64.
inline float certified_damping_f32(double omega_f64) {
  require(std::isfinite(omega_f64) && omega_f64 > 0.0,
          "invalid certified damping scalar");
  volatile float rounded = static_cast<float>(omega_f64);
  float out = rounded;
  if (static_cast<double>(out) > omega_f64) {
    out = std::nextafter(out, 0.0f);
  }
  require(out > 0.0f && std::isfinite(out) &&
              static_cast<double>(out) <= omega_f64,
          "no representable FP32 certified damping");
  return out;
}

}  // namespace certificate
}  // namespace bitfsai
