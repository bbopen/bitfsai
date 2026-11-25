import Bitfsai.Core
import Bitfsai.Multilevel
import Bitfsai.BitLimits

-- Core algebraic and re-projection theorems (21)
#print axioms BitFSAI.lowerTriangular_posDiag_injective
#print axioms BitFSAI.gram_posDef_of_lowerTriangular_posDiag
#print axioms BitFSAI.diagonal_split_preserves_posDef
#print axioms BitFSAI.lowerTriangular_zero_diag_not_posDef
#print axioms BitFSAI.conditional_spectral_distortion_bound
#print axioms BitFSAI.eta_five_over_28_bound
#print axioms BitFSAI.rayleigh_condition_distortion_bound
#print axioms BitFSAI.A_ex_posDef
#print axioms BitFSAI.G_ex_exact_inverse
#print axioms BitFSAI.inclusive_int4_rounds_offdiag_to_zero
#print axioms BitFSAI.split_int4_recovers_exact_factor
#print axioms BitFSAI.G_inc_preconditioned_matrix
#print axioms BitFSAI.counterexample_2x2_kappa_gt_321
#print axioms BitFSAI.damped_richardson_contraction
#print axioms BitFSAI.gram_spd_not_sufficient_for_fixed_damping
#print axioms BitFSAI.rowSum_repair_preserves_ones
#print axioms BitFSAI.rowSum_repair_diag_dom
#print axioms BitFSAI.scale_factored_gram_identity
#print axioms BitFSAI.optimal_2x2_reprojection_properties
#print axioms BitFSAI.zero_offdiag_reprojection_fallback
#print axioms BitFSAI.renormalized_2x2_counterexample_posDef

-- Multilevel convergence, polynomial-cycle, and certificate theorems (28)
#print axioms BitFSAI.variance_budget_3d
#print axioms BitFSAI.scalar_smoothing_3d
#print axioms BitFSAI.vcycle_induction_scalar
#print axioms BitFSAI.coarseError_nonneg
#print axioms BitFSAI.coarseError_le
#print axioms BitFSAI.fp8_e5m2_wcycle_constants
#print axioms BitFSAI.fp8_e5m2_defect_constants
#print axioms BitFSAI.guarded_smoothing_scalar
#print axioms BitFSAI.defect_cycle_homogeneous
#print axioms BitFSAI.defect_cycle_matrix_reduction
#print axioms BitFSAI.vcycle_upper_bound_of_galerkin_sandwich
#print axioms BitFSAI.two_call_interval
#print axioms BitFSAI.two_call_gain_upper
#print axioms BitFSAI.two_call_concave_endpoint_min
#print axioms BitFSAI.monotone_effectiveness_scalar
#print axioms BitFSAI.monotone_effectiveness_matrix
#print axioms BitFSAI.monotone_effectiveness_operator
#print axioms BitFSAI.work_convergence_frontier
#print axioms BitFSAI.work_convergence_frontier_periodic
#print axioms BitFSAI.work_convergence_frontier_converse
#print axioms BitFSAI.four_phase_sddm_budgets
#print axioms BitFSAI.four_phase_pruned_sddm_retention
#print axioms BitFSAI.four_phase_kappa_17_certificate
#print axioms BitFSAI.four_phase_kappa_9_certificate
#print axioms BitFSAI.four_phase_recurrence_induction
#print axioms BitFSAI.four_phase_arbitrary_depth_induction
#print axioms BitFSAI.singleton_block_truncation_bounds
#print axioms BitFSAI.absolute_guard_overdamping_witness

-- Extreme-low-bit precision limits, sign/ternary separation, and frozen-word drift (16)
#print axioms BitFSAI.projection_line_pythagorean
#print axioms BitFSAI.exact_prefix_kappa_threshold
#print axioms BitFSAI.global_kaporin_product_identity
#print axioms BitFSAI.grounded_star_sign_min
#print axioms BitFSAI.grounded_star_ternary_min
#print axioms BitFSAI.grounded_star_sign_kappa3_iff
#print axioms BitFSAI.grounded_star_ternary_kappa3_iff
#print axioms BitFSAI.grounded_star_separation_one_fifth
#print axioms BitFSAI.largest_voltage_gap_exists
#print axioms BitFSAI.sddm_ternary_cut_bound
#print axioms BitFSAI.unit_path_ternary_lower_bound
#print axioms BitFSAI.frozen_word_loewner_drift
#print axioms BitFSAI.kantorovich_pointwise_majorant
#print axioms BitFSAI.frozen_word_kantorovich_quadratic_drift
#print axioms BitFSAI.projective_packing_separation
#print axioms BitFSAI.harmonic_bin_minimax_relerr
