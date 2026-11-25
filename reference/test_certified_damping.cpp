#include "certified_damping.hpp"

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace bitfsai::certificate;

static void check(bool cond, const char *msg) {
  if (!cond) throw std::runtime_error(std::string("CHECK FAILED: ") + msg);
}

int main() {
  std::mt19937 rng(7331);
  for (int trial = 0; trial < 300; ++trial) {
    const std::size_t n = 1 + rng() % 22;
    Csr A;
    SplitFactor G;
    A.row = {0};
    G.row = {0};
    std::vector<std::vector<long double>> a(n, std::vector<long double>(n, 0.0L));
    auto g = a;

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (rng() % 3 == 0) {
          a[i][j] = a[j][i] = -static_cast<int>(1 + rng() % 8) / 8.0L;
        }
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      a[i][i] = 1.0L;
      for (std::size_t j = 0; j < n; ++j) {
        if (i != j) a[i][i] += std::abs(a[i][j]);
      }
      for (std::size_t j = 0; j < n; ++j) {
        if (a[i][j] != 0.0L) {
          A.col.push_back(j);
          A.value.push_back(static_cast<double>(a[i][j]));
        }
      }
      A.row.push_back(A.col.size());

      double diag = (1 + rng() % 8) / 8.0;
      double scale = (1 + rng() % 8) / 64.0;
      G.diagonal.push_back(diag);
      G.scale.push_back(scale);
      g[i][i] = diag;
      for (std::size_t j = 0; j < i; ++j) {
        if (rng() % 3 == 0) {
          int q = static_cast<int>(rng() % 15) - 7;
          G.col.push_back(j);
          G.code.push_back(static_cast<std::int8_t>(q));
          g[i][j] = static_cast<long double>(scale) * q;
        }
      }
      G.row.push_back(G.col.size());
    }

    // Dyadic inputs are exact in binary64/long double; compare against exact U.
    long double exact = 0.0L;
    std::vector<long double> z(n, 0.0L), y(n, 0.0L);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) z[j] += std::abs(g[i][j]);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) y[i] += std::abs(a[i][j]) * z[j];
    for (std::size_t i = 0; i < n; ++i) {
      long double w = 0.0L;
      for (std::size_t j = 0; j < n; ++j) w += std::abs(g[i][j]) * y[j];
      exact = std::max(exact, w);
    }

    auto result = certify_damping(A, G);
    check(result.U_upper >= exact, "U_upper must upper-bound exact U");
    check(result.omega_lower > 0.0, "omega_lower must be positive");
    check(static_cast<long double>(result.omega_lower) * result.U_upper <= 1.0L,
          "omega_lower * U_upper must be <= 1");
    check(static_cast<long double>(result.omega_lower) <= 18.0L / 25.0L,
          "omega_lower must be <= 18/25");
  }

  // Valid small and large 1x1 cases; overflow and zero diagonal must fail closed.
  Csr a{{0, 1}, {0}, {1.0}};
  SplitFactor g{{0, 0}, {}, {}, {0.0}, {1.0}};
  auto c = certify_damping(a, g);
  check(c.U_upper >= 1.0 && c.omega_lower <= 0.72, "1x1 identity certificate");

  a.value[0] = std::numeric_limits<double>::denorm_min();
  c = certify_damping(a, g);
  check(c.U_upper >= a.value[0] && c.omega_lower > 0.0, "1x1 subnormal certificate");

  a.value[0] = std::numeric_limits<double>::max();
  bool overflow = false;
  try {
    certify_damping(a, g);
  } catch (const std::runtime_error &) {
    overflow = true;
  }
  check(overflow, "overflowing U must fail closed");

  a.value[0] = 1.0;
  g.diagonal[0] = 0.0;
  bool bad_diag = false;
  try {
    certify_damping(a, g);
  } catch (const std::runtime_error &) {
    bad_diag = true;
  }
  check(bad_diag, "zero factor diagonal must fail closed");

  auto expect_reject = [](const Csr &mat, const SplitFactor &fac, const char *why) {
    bool rejected = false;
    try {
      certify_damping(mat, fac);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    check(rejected, why);
  };

  SplitFactor g2{{0, 0, 0}, {}, {}, {0.0, 0.0}, {1.0, 1.0}};

  // Nonsymmetric inputs: structural asymmetry and value asymmetry must fail closed.
  expect_reject(Csr{{0, 2, 3}, {0, 1, 1}, {2.0, -1.0, 2.0}}, g2, "reject structural asymmetry");
  expect_reject(Csr{{0, 2, 4}, {0, 1, 0, 1}, {2.0, -1.0, -0.5, 2.0}}, g2, "reject value asymmetry");

  // Mixed zero/nonzero-row matrices (empty row and explicit zero diagonal) must fail closed.
  expect_reject(Csr{{0, 1, 1}, {0}, {1.0}}, g2, "reject empty row");
  expect_reject(Csr{{0, 1, 2}, {0, 1}, {1.0, 0.0}}, g2, "reject zero diagonal");

  // NaN and +-infinity in A or G must fail closed.
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  const double inf_v = std::numeric_limits<double>::infinity();
  Csr valid2{{0, 2, 4}, {0, 1, 0, 1}, {2.0, -0.5, -0.5, 2.0}};
  check(certify_damping(valid2, g2).omega_lower > 0.0, "valid 2x2 certificate");

  expect_reject(Csr{{0, 2, 4}, {0, 1, 0, 1}, {nan_v, -0.5, -0.5, 2.0}}, g2, "reject NaN diag in A");
  expect_reject(Csr{{0, 2, 4}, {0, 1, 0, 1}, {2.0, nan_v, nan_v, 2.0}}, g2, "reject NaN off-diag in A");
  expect_reject(Csr{{0, 2, 4}, {0, 1, 0, 1}, {inf_v, -0.5, -0.5, 2.0}}, g2, "reject Inf diag in A");
  expect_reject(Csr{{0, 2, 4}, {0, 1, 0, 1}, {2.0, -inf_v, -inf_v, 2.0}}, g2, "reject -Inf off-diag in A");
  expect_reject(valid2, SplitFactor{{0, 0, 0}, {}, {}, {nan_v, 0.0}, {1.0, 1.0}}, "reject NaN scale");
  expect_reject(valid2, SplitFactor{{0, 0, 0}, {}, {}, {inf_v, 0.0}, {1.0, 1.0}}, "reject Inf scale");
  expect_reject(valid2, SplitFactor{{0, 0, 0}, {}, {}, {-inf_v, 0.0}, {1.0, 1.0}}, "reject -Inf scale");
  expect_reject(valid2, SplitFactor{{0, 0, 0}, {}, {}, {0.0, 0.0}, {nan_v, 1.0}}, "reject NaN factor diag");
  expect_reject(valid2, SplitFactor{{0, 0, 0}, {}, {}, {0.0, 0.0}, {inf_v, 1.0}}, "reject Inf factor diag");

  // Modified IEEE rounding modes (FE_DOWNWARD, FE_UPWARD) must fail closed; restore FE_TONEAREST.
  for (int mode : {FE_DOWNWARD, FE_UPWARD}) {
    const int set_rc = std::fesetround(mode);
    check(set_rc == 0, "fesetround(non-default) failed");
    expect_reject(valid2, g2, "non-FE_TONEAREST rounding mode must fail closed");
    const int restore_rc = std::fesetround(FE_TONEAREST);
    check(restore_rc == 0, "fesetround(FE_TONEAREST) failed");
  }
  check(std::fegetround() == FE_TONEAREST, "FE_TONEAREST must be restored");
  auto restored = certify_damping(valid2, g2);
  check(restored.U_upper >= 2.5 && restored.omega_lower > 0.0, "restored FE_TONEAREST certificate");

  // A = [3], G = [1]: exact 1/U is 1/3; both omega_lower (binary64) and
  // certified_damping_f32 (binary32) must strictly satisfy 3 * omega <= 1.0
  // (unlike naive float(1.0/3.0) which rounds upward above 1/3).
  Csr a3{{0, 1}, {0}, {3.0}};
  SplitFactor g1{{0, 0}, {}, {}, {0.0}, {1.0}};
  auto c3 = certify_damping(a3, g1);
  check(c3.U_upper >= 3.0, "U_upper >= 3 for A=[3]");
  check(3.0L * static_cast<long double>(c3.omega_lower) <= 1.0L, "3 * omega_lower <= 1");
  float omega_f32 = certified_damping_f32(c3.omega_lower);
  check(omega_f32 > 0.0f, "omega_f32 > 0");
  check(static_cast<double>(omega_f32) <= c3.omega_lower, "omega_f32 <= omega_lower");
  check(3.0L * static_cast<long double>(omega_f32) <= 1.0L, "3 * omega_f32 <= 1");
  check(3.0L * static_cast<long double>(static_cast<float>(1.0 / 3.0)) > 1.0L,
        "naive float(1/3) must exceed 1/3 as negative control");

  // INT4_Split adapter: verify step scale upper-bounds both exact q/7 and FP32 decode_nibble
  // across all q in [-7, 7], and certifies a real 2x2 split factor.
  {
    const double row_sc = 3.5;
    auto sf = make_int4_split_factor({0, 0, 1}, {0}, {3}, {0.0, row_sc}, {1.0, 1.0});
    for (int q = 1; q <= 7; ++q) {
      const double exact_mag = row_sc * (static_cast<double>(q) / 7.0);
      const double f32_mag = static_cast<double>(static_cast<float>(row_sc)) *
                             static_cast<double>(static_cast<float>(q) * (1.0f / 7.0f));
      const double bound_mag = sf.scale[1] * static_cast<double>(q);
      check(bound_mag >= exact_mag && bound_mag >= f32_mag,
            "make_int4_split_factor step scale must upper-bound exact and FP32 nibble weights");
    }
    auto cert = certify_damping(valid2, sf);
    check(cert.U_upper > 0.0 && cert.omega_lower > 0.0 &&
              static_cast<long double>(cert.omega_lower) * cert.U_upper <= 1.0L,
          "make_int4_split_factor must produce a valid certificate");

    const int bad_rp[2] = {1, 1};
    const int ci1[1] = {0};
    const float d1[1] = {1.0f}, s1[1] = {1.0f};
    const std::uint8_t q1[1] = {0x88};
    bool bad_rp_rejected = false;
    try {
      (void)make_int4_split_factor_from_packed_csr(1, bad_rp, ci1, d1, s1, q1);
    } catch (const std::runtime_error &) {
      bad_rp_rejected = true;
    }
    check(bad_rp_rejected, "make_int4_split_factor_from_packed_csr must reject rp[0] != 0");

    const int dec_rp[3] = {0, 1, 0};
    const float d2[2] = {1.0f, 1.0f}, s2[2] = {1.0f, 1.0f};
    bool dec_rp_rejected = false;
    try {
      (void)make_int4_split_factor_from_packed_csr(2, dec_rp, ci1, d2, s2, q1);
    } catch (const std::runtime_error &) {
      dec_rp_rejected = true;
    }
    check(dec_rp_rejected, "make_int4_split_factor_from_packed_csr must reject decreasing rp");

    const int ok_rp[2] = {0, 1};
    const int bad_col[1] = {1};
    bool bad_col_rejected = false;
    try {
      (void)make_int4_split_factor_from_packed_csr(1, ok_rp, bad_col, d1, s1, q1);
    } catch (const std::runtime_error &) {
      bad_col_rejected = true;
    }
    check(bad_col_rejected, "make_int4_split_factor_from_packed_csr must reject out-of-range col");
  }

  std::cout << "PASS: 300 dyadic matrix tests + nonsymmetric/zero-row/NaN/Inf/rounding-mode/f32-damping/INT4_Split-adapter guards passed.\n";
  return 0;
}
