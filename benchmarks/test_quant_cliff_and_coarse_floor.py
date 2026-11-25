#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from quant_cliff_and_coarse_floor import (
    _f32_metadata,
    compute_fsai1_lower,
    est_spectral_radius,
    fmt_res,
    quantize_coarse_matrix,
    quantize_fsai_split_reprojected,
    quantize_lower_rowwise,
    quantize_lower_rowwise_inclusive,
    run_pcg,
    verify_sddm_grounding_spd,
)


class QuantizationRegressionTest(unittest.TestCase):
    def test_split_preserves_the_counterexample_off_diagonal(self):
        A = sp.csr_matrix([[1.0, 0.04], [0.04, 0.0017]])
        G = sp.csr_matrix([[1.0, 0.0], [-4.0, 100.0]])

        split = quantize_fsai_split_reprojected(A, G, "INT4").toarray()
        inclusive = quantize_lower_rowwise_inclusive(G, "INT4").toarray()

        np.testing.assert_allclose(split, G.toarray(), rtol=0.0, atol=1e-5)
        self.assertAlmostEqual(float(split[1] @ A.toarray() @ split[1]), 1.0,
                               places=5)
        self.assertEqual(inclusive[1, 0], 0.0)

    def test_recursive_convergence_does_not_pass_the_true_residual_gate(self):
        A = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0 + 2.0 ** -40]])
        b = np.array([1.0, 0.1])

        _, true_rel, status = run_pcg(A, b, lambda r: r)

        self.assertEqual(status, "TRUE_RESIDUAL_FAIL")
        self.assertGreater(true_rel, 1e-8)

    def test_large_coupling_does_not_trigger_an_absolute_cutoff(self):
        A = sp.csr_matrix([[2.0 ** 42, 2.0 ** 21],
                           [2.0 ** 21, 1.0 + 2.0 ** -10]])
        G = quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")
        row = G.toarray()[1]

        self.assertNotEqual(row[0], 0.0)
        self.assertAlmostEqual(float(row @ A.toarray() @ row), 1.0, places=5)

    def test_fsai_pattern_matches_the_cuda_patch_cap(self):
        n = 14
        dense = np.eye(n) * 1e4
        for col in range(n - 1):
            value = float(col + 1)
            dense[n - 1, col] = dense[col, n - 1] = value

        G = compute_fsai1_lower(sp.csr_matrix(dense))
        kept = G.indices[G.indptr[-2]:G.indptr[-1]]

        np.testing.assert_array_equal(kept, np.arange(2, n))

    def test_tiny_valid_diagonal_is_not_clipped(self):
        A = sp.csr_matrix([[2.0 ** -100]])
        G = quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")
        diag = G.toarray()[0, 0]

        self.assertGreater(diag, 1e14)
        self.assertAlmostEqual(float(diag * A.toarray()[0, 0] * diag), 1.0,
                               places=5)

    def test_nearly_singular_projection_uses_the_diagonal_fallback(self):
        A = sp.csr_matrix([[1.0, 1.0 - 2.0 ** -45],
                           [1.0 - 2.0 ** -45, 1.0]])
        G = quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")
        row = G.toarray()[1]

        self.assertEqual(row[0], 0.0)
        self.assertAlmostEqual(float(row @ A.toarray() @ row), 1.0, places=5)

    def test_unstorable_fp32_diagonal_fails(self):
        A = sp.csr_matrix([[2.0 ** -300]])

        with self.assertRaisesRegex(ValueError, "does not fit FP32"):
            quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")

    def test_metadata_accepts_a_value_that_rounds_to_a_fp32_subnormal(self):
        minimum = float(np.nextafter(np.float32(0.0), np.float32(1.0)))

        stored = _f32_metadata(0.75 * minimum, 0, "diagonal")

        self.assertEqual(stored, minimum)

    def test_projected_scale_that_rounds_to_zero_uses_diagonal_fallback(self):
        A = sp.csr_matrix([[1.0, 1e-50], [1e-50, 1.0]])
        G = quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")
        row = G.toarray()[1]

        self.assertEqual(row[0], 0.0)
        self.assertAlmostEqual(float(row @ A.toarray() @ row), 1.0, places=5)

    def test_triangular_fsai_solve_keeps_a_tiny_valid_singleton(self):
        A = sp.csr_matrix([[1e-310]])
        G = compute_fsai1_lower(A).toarray()

        self.assertTrue(np.isfinite(G[0, 0]))
        self.assertAlmostEqual(float(G[0, 0] * 1e-310 * G[0, 0]), 1.0,
                               places=5)

    def test_non_spd_patch_with_valid_diagonal_uses_diagonal_fallback(self):
        A = sp.csr_matrix([[1.0, 2.0], [2.0, 1.0]])
        G = compute_fsai1_lower(A).toarray()

        np.testing.assert_array_equal(G[1], np.array([0.0, 1.0]))

    def test_legacy_inclusive_quantizer_name_remains_available(self):
        G = sp.csr_matrix([[1.0]])

        np.testing.assert_array_equal(quantize_lower_rowwise(G, "INT4").toarray(),
                                      quantize_lower_rowwise_inclusive(G, "INT4").toarray())

    def test_max_iters_formats_as_max_iters_not_breakdown(self):
        A = sp.diags([1.0, 2.0, 3.0, 4.0], format="csr")
        b = np.ones(4, dtype=np.float64)

        it, _, status = run_pcg(A, b, lambda r: r, rtol=1e-12, maxit=2)

        self.assertEqual(it, 2)
        self.assertEqual(status, "MAX_ITERS")
        self.assertEqual(fmt_res(it, status), "MAX_ITERS")
        self.assertEqual(fmt_res(5, "CONVERGED"), "5 it")
        self.assertEqual(fmt_res(3, "TRUE_RESIDUAL_FAIL"), "TRUE-FAIL")
        self.assertEqual(fmt_res(1, "NON-SPD BREAKDOWN"), "BREAKDOWN")

    def test_fp32_persisted_repaired_coarse_diagonal_preserves_grounding(self):
        diag = 1.0 + 2.0 ** -26
        Ac = sp.csr_matrix([[diag, -1.0], [-1.0, diag]], dtype=np.float64)

        q_f64 = quantize_coarse_matrix(
            Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=False
        ).toarray()
        q_f32 = quantize_coarse_matrix(
            Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
        ).toarray()

        self.assertAlmostEqual(q_f64[0, 0] + q_f64[0, 1], 2.0 ** -26, places=15)
        # Naive float32(1 + 2^-26) == 1.0 would make the 2x2 block singular;
        # upward rounding keeps the FP32-persisted repaired matrix strictly SPD.
        self.assertEqual(np.float32(q_f32[0, 0]), q_f32[0, 0])
        self.assertGreater(q_f32[0, 0], 1.0)
        self.assertGreater(np.linalg.eigvalsh(q_f32).min(), 0.0)

    def test_fp8_coarse_rejects_disconnected_ungrounded_component(self):
        # Counterexample A: 3-node path with weak edge 2^-18 and edge 1, unit
        # grounding only at node 0. E5M2 underflows 2^-18 to 0, leaving {1, 2}
        # singular ([[1, 0, 0], [0, 1, -1], [0, -1, 1]]).
        w = 2.0 ** -18
        Ac = sp.csr_matrix(
            [[1.0 + w, -w, 0.0], [-w, 1.0 + w, -1.0], [0.0, -1.0, 1.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "below normal min"):
            quantize_coarse_matrix(
                Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
            )
        # Unchecked FP64 repair indeed produces a singular matrix; helper flags it.
        q_unchecked = quantize_coarse_matrix(
            Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=False, verify_sddm=False
        )
        ok, reason = verify_sddm_grounding_spd(q_unchecked)
        self.assertFalse(ok)
        self.assertIn("ungrounded", reason)
        self.assertAlmostEqual(float(np.linalg.eigvalsh(q_unchecked.toarray()).min()), 0.0, places=12)

    def test_fp8_coarse_rejects_saturation_and_subnormal_normal_range_violations(self):
        # 10^6-scaled SDDM matrix saturates E5M2 (> 57344.0) and must fail closed.
        Ac_sat = sp.csr_matrix([[2.0e6, -1.0e6], [-1.0e6, 2.0e6]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "57344"):
            quantize_coarse_matrix(
                Ac_sat, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
            )
        # Subnormal off-diagonal 2^-15 violates E5M2 normal range (< 2^-14) and must fail closed.
        sub = 2.0 ** -15
        Ac_sub = sp.csr_matrix([[1.0 + sub, -sub], [-sub, 1.0 + sub]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "2\\^-14"):
            quantize_coarse_matrix(
                Ac_sub, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
            )

    def test_fp8_coarse_rejects_positive_off_diagonal_indefinite_repair(self):
        # Counterexample B: symmetric SPD matrix with positive off-diagonals
        # [[1.14, 1.13], [1.13, 1.14]] turns indefinite (lambda_min = -0.23)
        # after E5M2 row-sum repair.
        Ac = sp.csr_matrix([[1.14, 1.13], [1.13, 1.14]], dtype=np.float64)
        self.assertGreater(float(np.linalg.eigvalsh(Ac.toarray()).min()), 0.0)
        with self.assertRaisesRegex(ValueError, "positive off-diagonals"):
            quantize_coarse_matrix(
                Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
            )
        q_unchecked = quantize_coarse_matrix(
            Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=False, verify_sddm=False
        )
        ok, reason = verify_sddm_grounding_spd(q_unchecked)
        self.assertFalse(ok)
        self.assertIn("positive off-diagonals", reason)
        self.assertLess(float(np.linalg.eigvalsh(q_unchecked.toarray()).min()), 0.0)

    def test_power_iteration_estimate_is_heuristic_when_orthogonal_to_dominant_mode(self):
        # For n = 4, the alternating start vector [-1, 1, -1, 1] / 2 = [-0.5, 0.5, -0.5, 0.5]
        # is exact in binary64 and orthogonal to the dominant eigenvector [1, 1, 0, 0]
        # (lambda_max = 100), so power iteration returns lambda = 1.0 instead of 100.0.
        A = sp.csr_matrix(
            [
                [50.5, 49.5, 0.0, 0.0],
                [49.5, 50.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        exact_lmax = float(np.linalg.eigvalsh(A.toarray()).max())
        est = est_spectral_radius(lambda v: v, A, 4, iters=25)
        self.assertAlmostEqual(exact_lmax, 100.0, places=12)
        self.assertAlmostEqual(est, 1.0, places=12)

    def test_verify_sddm_grounding_rejects_singular_cancellation_star(self):
        # Singular 4-node star (center last) with edge weights 2^54, 2, 2:
        # every exact row sum is 0, whereas naive FP64 A @ ones gives [0, 0, 0, 4].
        w0 = float(2 ** 54)
        A = sp.csr_matrix(
            [
                [w0, 0.0, 0.0, -w0],
                [0.0, 2.0, 0.0, -2.0],
                [0.0, 0.0, 2.0, -2.0],
                [-w0, -2.0, -2.0, w0 + 4.0],
            ],
            dtype=np.float64,
        )
        ok, reason = verify_sddm_grounding_spd(A)
        self.assertFalse(ok)
        self.assertIn("ungrounded", reason)

    def test_run_pcg_zero_rhs_converges_immediately(self):
        A = sp.csr_matrix([[1.0]], dtype=np.float64)
        b = np.array([0.0], dtype=np.float64)
        it, rel, status = run_pcg(A, b, lambda r: r)
        self.assertEqual((it, rel, status), (0, 0.0, "CONVERGED"))

    def test_run_pcg_tiny_nonzero_rhs_does_not_false_converge(self):
        A = sp.csr_matrix([[1.0]], dtype=np.float64)
        b = np.array([2.0 ** -600], dtype=np.float64)
        it, rel, status = run_pcg(A, b, lambda r: r)
        self.assertEqual(status, "CONVERGED")
        self.assertEqual(it, 1)
        self.assertLessEqual(rel, 1e-8)

    def test_fp8_coarse_preserves_tiny_ground_via_exact_row_sum(self):
        s = 0.09999999999999996
        Ac = sp.csr_matrix(
            [[s, -s, 0.0], [-s, 1.0, -0.9], [0.0, -0.9, 0.9]], dtype=np.float64
        )
        q = quantize_coarse_matrix(
            Ac, "FP8_E5M2", preserve_rowsum=True, persist_diag_fp32=True
        )
        ok, reason = verify_sddm_grounding_spd(q)
        self.assertTrue(ok, reason)
        self.assertGreater(float(np.linalg.eigvalsh(q.toarray()).min()), 0.0)

    def test_row_energy_tolerance_falls_back_between_1e4_and_1e2(self):
        # Choose b = sqrt(2*3) * (1 - 2^-39) so stored FP32 row-energy error +
        # bound is ~2.1e-3 (strictly between 1e-4 and 1e-2), forcing fallback.
        a_val, c_val = 2.0, 3.0
        b_val = float(np.sqrt(a_val * c_val) * (1.0 - 2.0 ** -39))
        A = sp.csr_matrix([[a_val, b_val], [b_val, c_val]], dtype=np.float64)
        G = quantize_fsai_split_reprojected(A, compute_fsai1_lower(A), "INT4")
        row = G.toarray()[1]
        self.assertEqual(row[0], 0.0)
        self.assertAlmostEqual(float(row @ A.toarray() @ row), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
