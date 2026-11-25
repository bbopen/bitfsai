#include "row_projection.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cstdio>

namespace {

struct Patch2 {
  double value[4];

  double operator()(int row, int col) const { return value[row * 2 + col]; }
};

void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", message);
    std::abort();
  }
}

void test_tiny_positive_diagonal() {
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::diagonal_row(1.0e-30, &diag, &scale);
  const double stored = static_cast<double>(static_cast<float>(diag));
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "positive tiny pivot must use the diagonal row");
  require(std::fabs(stored * stored * 1.0e-30 - 1.0) < 1.0e-5,
          "tiny positive pivot must keep stored FP32 unit energy");
}

void test_subnormal_float_diagonal() {
  const float source = 1.0e-40f;
  const double a_ii = static_cast<double>(source);
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::diagonal_row(a_ii, &diag, &scale);
  const double stored = static_cast<double>(static_cast<float>(diag));
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "positive FP32 subnormal must not take an absolute-pivot failure");
  require(std::fabs(stored * stored * a_ii - 1.0) < 1.0e-5,
          "stored FP32 diagonal must retain subnormal-input accuracy");
}

void test_small_scale_ratio() {
  constexpr double ratio = 0.5e-6;
  const double lower_l[4] = {1.0, 0.0, -ratio, 1.0};
  const double decoded[1] = {1.0};
  const Patch2 patch{{1.0, -ratio, -ratio, 1.0 + ratio * ratio}};
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::project_row(2, decoded, lower_l, 2, patch,
                                            &diag, &scale);
  require(status == bitfsai::RowProjectionStatus::Applied,
          "small valid scale must not take a heuristic fallback");
  require(std::fabs(scale / diag - ratio) < 1.0e-12,
          "projection must retain the small valid scale");
}

void test_unstable_stored_energy_fallback() {
  const double a = 2.0;
  const double c = 3.0;
  const double b = std::sqrt(a * c) * (1.0 - 1.0e-13);
  const double lower_l[4] = {std::sqrt(a), 0.0, b / std::sqrt(a),
                             std::sqrt(c - b * b / a)};
  const double decoded[1] = {-1.0};  // INT4 code -7 decoded as -7 / 7.
  const Patch2 patch{{a, b, b, c}};
  double diag = 0.0, scale = 0.0, energy = 0.0, bound = 0.0;
  const auto status = bitfsai::project_row(2, decoded, lower_l, 2, patch,
                                            &diag, &scale, &energy, &bound);
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "uncertain stored FP32 energy must fall back");
  require(std::fabs(energy - 1.0) + bound >
              bitfsai::kRowProjectionEnergyTolerance,
          "the stored-energy check must explain the fallback");
}

void test_invalid_fp32_store() {
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::diagonal_row(1.0e100, &diag, &scale);
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "large positive diagonal is valid before FP32 storage");
  require(!bitfsai::row_projection_storable_f32(diag),
          "out-of-range FP32 diagonal must be rejected by setup");
}

void test_seventeen_entry_patch() {
  constexpr int m = 17;
  double lower_l[m * m] = {};
  double patch[m * m] = {};
  double decoded[m - 1] = {};
  for (int i = 0; i < m; ++i) lower_l[i * m + i] = 1.0;
  lower_l[(m - 1) * m] = -0.1;
  decoded[0] = 1.0;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < m; ++col) {
      double value = 0.0;
      for (int k = 0; k < m; ++k) value += lower_l[row * m + k] * lower_l[col * m + k];
      patch[row * m + col] = value;
    }
  }
  const auto patch_at = [&patch](int row, int col) { return patch[row * m + col]; };
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::project_row(m, decoded, lower_l, m, patch_at,
                                            &diag, &scale);
  require(status == bitfsai::RowProjectionStatus::Applied,
          "projection must support sixteen lower entries and one diagonal");
  require(std::fabs(diag - 1.0) < 1.0e-12 && std::fabs(scale - 0.1) < 1.0e-12,
          "seventeen-entry projection must retain its row factors");
}

void test_minimum_subnormal_diagonal() {
  const float source = std::numeric_limits<float>::denorm_min();
  const double a_ii = static_cast<double>(source);
  double diag = 0.0, scale = 0.0;
  const auto status = bitfsai::diagonal_row(a_ii, &diag, &scale);
  const double stored = static_cast<double>(static_cast<float>(diag));
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "minimum positive subnormal must keep a diagonal fallback");
  require(bitfsai::row_projection_storable_f32(diag),
          "minimum positive subnormal must produce a nonzero FP32 factor");
  require(std::fabs(stored * stored * a_ii - 1.0) < 1.0e-5,
          "minimum positive subnormal must keep stored diagonal accuracy");
}

void test_energy_tolerance_threshold_discrimination() {
  // Stored FP32 energy error + bound is ~2.1e-3 (strictly between 1e-4 and 1e-2),
  // so relaxing kRowProjectionEnergyTolerance from 1e-4 to 1e-2 would falsely accept it.
  const double a = 2.0;
  const double c = 3.0;
  const double b = std::sqrt(a * c) * (1.0 - std::ldexp(1.0, -39));
  const double lower_l[4] = {std::sqrt(a), 0.0, b / std::sqrt(a),
                             std::sqrt(c - b * b / a)};
  const double decoded[1] = {-1.0};
  const Patch2 patch{{a, b, b, c}};
  double diag = 0.0, scale = 0.0, energy = 0.0, bound = 0.0;
  const auto status = bitfsai::project_row(2, decoded, lower_l, 2, patch,
                                            &diag, &scale, &energy, &bound);
  const double err = std::fabs(energy - 1.0) + bound;
  require(status == bitfsai::RowProjectionStatus::Fallback,
          "stored FP32 energy error above 1e-4 must fall back");
  require(err > bitfsai::kRowProjectionEnergyTolerance && err < 1.0e-2,
          "discrimination fixture must sit strictly between 1e-4 and 1e-2");
}

}  // namespace

int main() {
  test_tiny_positive_diagonal();
  test_subnormal_float_diagonal();
  test_small_scale_ratio();
  test_unstable_stored_energy_fallback();
  test_invalid_fp32_store();
  test_seventeen_entry_patch();
  test_minimum_subnormal_diagonal();
  test_energy_tolerance_threshold_discrimination();
  std::puts("row projection checks passed");
  return 0;
}
