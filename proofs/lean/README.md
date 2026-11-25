# BitFSAI Lean 4 Proofs

Machine-checked, `sorry`-free, custom-axiom-free proofs of the algebraic, multilevel-convergence, and precision-limit guarantees for BitFSAI, checked against Mathlib (`leanprover/lean4:v4.37.0-rc1`).

## Verified theorems (65 across three modules)

### 1. Smoother algebra and re-projection (`Bitfsai/Core.lean`, 21 theorems)

| Section | Theorems | Dimension | Formal guarantee |
|---|---|---|---|
| 1 | `lowerTriangular_posDiag_injective`, `gram_posDef_of_lowerTriangular_posDiag` | Any $n \in \mathbb{N}$ | Any lower-triangular $G$ with $\forall i,\; G_{ii} > 0$ and arbitrary off-diagonals yields `(Gᵀ * G).PosDef`. |
| 2 | `diagonal_split_preserves_posDef`, `lowerTriangular_zero_diag_not_posDef` | Any $n \in \mathbb{N}$ | Diagonal-split quantization preserves `Matrix.PosDef` unconditionally. Zeroing any diagonal destroys `PosDef` ($\det = 0$). |
| 3 | `conditional_spectral_distortion_bound`, `eta_five_over_28_bound`, `rayleigh_condition_distortion_bound` | Any $n \in \mathbb{N}$ | Relative perturbation $\eta < 1$ bounds generalized Rayleigh-quotient condition distortion `(‖G_q x‖² / xᵀAx) / (‖G_q y‖² / yᵀAy)` by $((1+\eta)/(1-\eta))^2$ (`< 2.06` at $\eta = 5/28$). |
| 4 | `A_ex_posDef`, `G_ex_exact_inverse`, `inclusive_int4_rounds_offdiag_to_zero`, `split_int4_recovers_exact_factor`, `G_inc_preconditioned_matrix`, `counterexample_2x2_kappa_gt_321` | $2 \times 2$ exact | Constructive SPD witness $A = \begin{bmatrix}1 & 1/25 \\ 1/25 & 17/10000\end{bmatrix}$ proving diagonal-inclusive INT4 drives $\kappa > 321$, whereas diagonal-split INT4 recovers $G A G^\top = I$. |
| 5 | `damped_richardson_contraction`, `gram_spd_not_sufficient_for_fixed_damping` | Scalar / $2 \times 2$ | Contraction for $\omega \lambda < 2$, plus explicit witness showing $M = G^\top G \succ 0$ does not certify $2M - \omega M A M \succ 0$ at $\omega = 0.65$. |
| 6 | `rowSum_repair_preserves_ones`, `rowSum_repair_diag_dom` | Any $n \in \mathbb{N}$ | Exact row-sum diagonal repair preserves $\tilde{A}\mathbf{1} = A\mathbf{1}$ and M-matrix diagonal dominance. |
| 7 | `scale_factored_gram_identity` | Any $n \in \mathbb{N}$ | Exact algebraic factorization $G = \operatorname{diag}(s)\tilde{G} \implies (G^\top G)x = \tilde{G}^\top \operatorname{diag}(s^2)\tilde{G}x$. |
| 8 | `optimal_2x2_reprojection_properties`, `zero_offdiag_reprojection_fallback`, `renormalized_2x2_counterexample_posDef` | General / $2 \times 2$ | Closed-form $(s_i^*, g_{ii}^*)$ and zero-direction fallback ($s = 0, g = 1/\sqrt{c_{22}}$ when $c_{11} = c_{12} = 0$) guarantee $g > 0$, exact unit $A$-energy, global Kaporin optimality, and `PosDef` of $2M - 0.72 M A M$. |

### 2. Multilevel convergence and polynomial schedules (`Bitfsai/Multilevel.lean`, 28 theorems)

| Section | Theorems | Dimension | Formal guarantee |
|---|---|---|---|
| 1 | `variance_budget_3d`, `scalar_smoothing_3d`, `vcycle_induction_scalar` | Scalar | 3D INT4 variance budget ($t \le 1/294$), smoothing bound at $\omega = 18/25$, and V-cycle induction step. |
| 2 | `coarseError_nonneg`, `coarseError_le`, `fp8_e5m2_wcycle_constants`, `fp8_e5m2_defect_constants`, `guarded_smoothing_scalar`, `defect_cycle_homogeneous`, `defect_cycle_matrix_reduction`, `vcycle_upper_bound_of_galerkin_sandwich` | Any finite $\iota, \kappa$ | FP8 E5M2 rounded W-cycle fixed-point ratios ($\alpha = 49/81 > 1/2$), defect-cycle matrix reduction, and arbitrary-dimensional symmetric Galerkin sandwich ($B^\top = B$, $0 \le R \le B \le A^{-1}$) from SPD $A$, injective $P$, symmetric $A_c$, $0 \le R \le A^{-1}$, $R^\top = R$, $\gamma > 0$, $\gamma P^\top A P \le A_c$, $B_c^\top = B_c$, and $0 \le B_c \le A_c^{-1}$. |
| 3 | `two_call_interval`, `two_call_gain_upper`, `two_call_concave_endpoint_min`, `monotone_effectiveness_scalar`, `monotone_effectiveness_matrix`, `monotone_effectiveness_operator`, `work_convergence_frontier`, `work_convergence_frontier_periodic`, `work_convergence_frontier_converse` | Any finite $\iota, \kappa$ | Monotone one-level effectiveness lemma $e_{\mathrm{new}} = \min(p\,g(e), d)$, including full A-inner-product operator mechanization (`monotone_effectiveness_operator`) proving the coarse-comparison step from `P`, `T = Pᵀ A P`, `R`, and `B_amp` without assuming `hE`, concave endpoint bound, and sharp necessary and sufficient work-convergence frontier $\beta^2 < p$. |
| 4 | `four_phase_sddm_budgets`, `four_phase_pruned_sddm_retention`, `four_phase_kappa_17_certificate`, `four_phase_kappa_9_certificate`, `four_phase_recurrence_induction`, `four_phase_arbitrary_depth_induction`, `singleton_block_truncation_bounds`, `absolute_guard_overdamping_witness` | Exact / arbitrary depth $D \in \mathbb{N}$ / $2 \times 2$ | Four-phase grounded SDDM budgets (including $511/512$ edge-pruning retention $> 11/15$), exact root certificates $\kappa \le 17$ ($t = 2560/1307$) and $\kappa \le 9$ ($t = 30375/15853$), arbitrary-depth induction (`four_phase_arbitrary_depth_induction`) proving phase targets and root effectiveness $\ge 1/17$ and $\ge 1/9$ for all depths $D \in \mathbb{N}$, 4-bit singleton truncation termination, and the 2x2 absolute-guard overdamping witness. |

### 3. Extreme-low-bit precision limits and frozen-word drift (`Bitfsai/BitLimits.lean`, 16 public theorems + 3 helper lemmas)

| Section | Theorems | Dimension | Formal guarantee |
|---|---|---|---|
| 1 | `projection_line_pythagorean`, `exact_prefix_kappa_threshold`, `global_kaporin_product_identity` | Scalar / Any $n \in \mathbb{N}$ | Exact Schur-pivot line loss $1 + D/\sigma$, exact-prefix threshold $\kappa \le K \iff D/\sigma \le (K-1)^2/(4K)$ under $\rho^2(\sigma+D)=D$, and arbitrary-$n$ determinant-ratio identity $\det(G_* A G_*^\top)/\det(G_q A G_q^\top) = \prod_i (G_{*,ii}/G_{q,ii})^2$. |
| 2 | `grounded_star_sign_min`, `grounded_star_ternary_min`, `grounded_star_sign_kappa3_iff`, `grounded_star_ternary_kappa3_iff`, `grounded_star_separation_one_fifth` | $3 \times 3$ exact | Grounded SDDM star separation between 1-bit sign-only (`t >= 3/5` for $\kappa \le 3$, and $\kappa > 9$ at $t=1/5$) and 1.58-bit ternary (`t <= 1/4 ∨ t >= 3/5`, and $\kappa < 3$ at $t=1/5$) with exact real-valued `(g_ii, s_i)`. |
| 3 | `largest_voltage_gap_exists`, `sddm_ternary_cut_bound`, `unit_path_ternary_lower_bound` | Any $r \in \mathbb{N}$ / $r \ge 1$ | Contrast-independent ternary threshold cut $C_k \le (r+1)\sigma$ under Kirchhoff flux `gap_k * cap_k <= σ`, and matching-order unit-path lower bound $E/\sigma \ge (r+1)/2$ for `σ = 1/(r+1)`. |
| 4 | `frozen_word_loewner_drift`, `kantorovich_pointwise_majorant`, `frozen_word_kantorovich_quadratic_drift` | Scalar / Any finite $\iota$ | Local frozen-word Newton drift bounds under variational and spectral-weight hypotheses: general $h/l$ ratio bound and Kantorovich quadratic bound $1 + (h-l)^2/(4hl)$ over any finite spectrum. |
| 5 | `projective_packing_separation`, `harmonic_bin_minimax_relerr` | General | Scalar $K^2$ interval-incompatibility lemma for projective packing and sharp harmonic-mean minimax coarse weight bin relative error bound $\le (r-1)/(r+1)$ on $[1, r]$ (attained at endpoints $1, r$). |

## Reproduction

```bash
lake exe cache get   # optional: fetch precompiled Mathlib .olean cache
lake build
lake env lean Bitfsai.lean
```

All 65 `#print axioms` checks in `Bitfsai.lean` depend strictly on Lean's three standard axioms:
`[propext, Classical.choice, Quot.sound]` (zero `sorryAx`, zero custom axioms).
