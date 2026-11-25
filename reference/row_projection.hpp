#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define BITFSAI_ROW_HD __host__ __device__
#else
#define BITFSAI_ROW_HD
#endif

namespace bitfsai {

enum class RowProjectionStatus : uint8_t {
  Applied,
  Fallback,
  InvalidInput,
  InvalidStorage,
};

constexpr double kRowProjectionEnergyTolerance = 1.0e-4;

BITFSAI_ROW_HD inline bool row_projection_finite(double value) {
  return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

BITFSAI_ROW_HD inline bool row_projection_storable_f32(double value) {
  const float stored = static_cast<float>(value);
  return row_projection_finite(value) && stored == stored && stored != 0.0f &&
         stored >= -FLT_MAX && stored <= FLT_MAX;
}

BITFSAI_ROW_HD inline RowProjectionStatus diagonal_row(double a_ii,
                                                        double *diag,
                                                        double *scale) {
  if (!(a_ii > 0.0) || !row_projection_finite(a_ii)) {
    return RowProjectionStatus::InvalidInput;
  }
  const double value = 1.0 / sqrt(a_ii);
  if (!(value > 0.0) || !row_projection_finite(value)) {
    return RowProjectionStatus::InvalidInput;
  }
  *diag = value;
  *scale = value;
  return RowProjectionStatus::Fallback;
}

template <typename PatchAt>
BITFSAI_ROW_HD inline RowProjectionStatus project_row(
    int m, const double *decoded, const double *lower_l, int lower_ld,
    PatchAt patch_at, double *diag, double *scale,
    double *stored_energy = nullptr, double *cancellation_bound = nullptr) {
  if (stored_energy) *stored_energy = 0.0;
  if (cancellation_bound) *cancellation_bound = 0.0;
  if (m < 2 || lower_ld < m) {
    return RowProjectionStatus::Fallback;
  }

  const double c22 = patch_at(m - 1, m - 1);
  if (!(c22 > 0.0) || !row_projection_finite(c22)) {
    return RowProjectionStatus::InvalidInput;
  }

  double c11 = 0.0;
  double c12 = 0.0;
  for (int b = 0; b < m - 1; ++b) {
    double ub = 0.0;
    for (int a = b; a < m - 1; ++a) {
      ub += lower_l[a * lower_ld + b] * decoded[a];
    }
    c11 += ub * ub;
    c12 += ub * lower_l[(m - 1) * lower_ld + b];
  }
  if (!(c11 > 0.0) || !row_projection_finite(c11) ||
      !row_projection_finite(c12)) {
    return RowProjectionStatus::Fallback;
  }

  const double projection = c12 / c11;
  if (!row_projection_finite(projection)) {
    return RowProjectionStatus::Fallback;
  }

  // Compute the component of L^T e_m orthogonal to L^T q. This avoids the
  // cancellation in c11 * c22 - c12 * c12.
  double residual_sq = 0.0;
  for (int b = 0; b < m; ++b) {
    double ub = 0.0;
    if (b < m - 1) {
      for (int a = b; a < m - 1; ++a) {
        ub += lower_l[a * lower_ld + b] * decoded[a];
      }
    }
    const double vb = lower_l[(m - 1) * lower_ld + b];
    const double residual = vb - projection * ub;
    residual_sq += residual * residual;
  }
  if (!(residual_sq > 0.0) || !row_projection_finite(residual_sq)) {
    return RowProjectionStatus::Fallback;
  }

  const double projected_diag = 1.0 / sqrt(residual_sq);
  const double projected_scale = -projection * projected_diag;
  if (!(projected_diag > 0.0) || !row_projection_finite(projected_diag) ||
      !row_projection_finite(projected_scale)) {
    return RowProjectionStatus::Fallback;
  }
  const float stored_diag_f32 = static_cast<float>(projected_diag);
  const float stored_scale_f32 = static_cast<float>(projected_scale);
  if (stored_scale_f32 == 0.0f) return RowProjectionStatus::Fallback;
  if (!row_projection_storable_f32(projected_diag) ||
      !row_projection_storable_f32(projected_scale)) {
    return RowProjectionStatus::InvalidStorage;
  }

  const double stored_diag = static_cast<double>(stored_diag_f32);
  const double stored_scale = static_cast<double>(stored_scale_f32);
  double energy = 0.0;
  double compensation = 0.0;
  double sum_abs = 0.0;
  for (int a = 0; a < m; ++a) {
    const double wa = (a == m - 1) ? stored_diag : stored_scale * decoded[a];
    for (int b = 0; b < m; ++b) {
      const double wb = (b == m - 1) ? stored_diag : stored_scale * decoded[b];
      const double term = wa * patch_at(a, b) * wb;
      if (!row_projection_finite(term)) return RowProjectionStatus::InvalidStorage;
      const double adjusted = term - compensation;
      const double next = energy + adjusted;
      compensation = (next - energy) - adjusted;
      energy = next;
      sum_abs += fabs(term);
    }
  }
  if (!row_projection_finite(energy) || !row_projection_finite(sum_abs)) {
    return RowProjectionStatus::InvalidStorage;
  }

  // Bound product and compensated-sum rounding with the absolute term sum.
  const double bound = 8.0 * DBL_EPSILON * sum_abs;
  if (stored_energy) *stored_energy = energy;
  if (cancellation_bound) *cancellation_bound = bound;
  if (!(fabs(energy - 1.0) + bound <= kRowProjectionEnergyTolerance)) {
    return RowProjectionStatus::Fallback;
  }

  *diag = projected_diag;
  *scale = projected_scale;
  return RowProjectionStatus::Applied;
}

}  // namespace bitfsai

#undef BITFSAI_ROW_HD
