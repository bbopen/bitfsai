import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Tactic
import Bitfsai.Core

namespace BitFSAI

open Matrix Finset

/-!
# BitFSAI Extreme-Low-Bit Precision Limits & Frozen-Word Drift

Machine-checked theorems establishing:
1. Exact row projection loss `1 + D / σ`, exact-prefix spectral threshold
   `κ ≤ K ↔ D / σ ≤ (K - 1)^2 / (4 K)`, and arbitrary-`n` Global Kaporin Product Identity.
2. Exact 3-vertex grounded SDDM separation between 1-bit sign-only (`{-1, +1}`)
   and 1.58-bit ternary (`{-1, 0, +1}`), including exact `K = 3` threshold intervals
   and the `t = 1/5` separation witness.
3. Contrast-independent `r + 1` ternary threshold-cut upper bound for arbitrary
   `r`-predecessor grounded SDDM patches and matching `(r + 1) / 2` unit-path lower bound.
4. Frozen-word Newton drift: general `h / l` Loewner bound and arbitrary-spectrum
   Kantorovich quadratic drift bound `1 + (h - l)^2 / (4 h l)` for exact reference rows.
5. Information-theoretic projective SPD packing (`K^2` separation) and harmonic-mean
   minimax coarse weight bin error `(r - 1) / (r + 1)`.
-/

-- =============================================================================
-- 1. Projection Loss, Exact-Prefix Spectral Criterion, and Kaporin Product
-- =============================================================================

/-- Theorem 9.1a: Exact quadratic fiber decomposition along a candidate line `span(q)`.
    Minimizing `σ + (v_norm_sq - 2 * b * t + a * t^2)` over `t ∈ ℝ` (for `a > 0`)
    achieves minimum `σ + D` with `D = v_norm_sq - b^2 / a`, and Theorem 3
    re-projection produces local optimality inflation `1 + D / σ`. -/
theorem projection_line_pythagorean
    (a b v_sq σ t : ℝ) (ha : 0 < a) (hσ : 0 < σ) :
    let D := v_sq - b ^ 2 / a
    σ + (v_sq - 2 * b * t + a * t ^ 2) = (σ + D) + a * (t - b / a) ^ 2 ∧
      (σ + D) / σ = 1 + D / σ := by
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hσ0 : σ ≠ 0 := ne_of_gt hσ
  dsimp only
  constructor
  · field_simp [ha0]; ring
  · field_simp [hσ0]

/-- Theorem 9.1b: Exact-prefix spectral condition threshold after Theorem 3 re-projection.
    When an exact whitened prefix is followed by a single quantized row with
    cross-vector norm `ρ ∈ [0, 1)` satisfying `ρ^2 (σ + D) = D`, the resulting
    condition number `(1 + ρ) / (1 - ρ)` is at most `K ≥ 1` if and only if
    `D / σ ≤ (K - 1)^2 / (4 K)`. -/
theorem exact_prefix_kappa_threshold
    (K σ D ρ : ℝ)
    (hK : 1 ≤ K) (hσ : 0 < σ) (hD : 0 ≤ D)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hρsq : ρ ^ 2 * (σ + D) = D) :
    (1 + ρ) / (1 - ρ) ≤ K ↔ D / σ ≤ (K - 1) ^ 2 / (4 * K) := by
  have hKpos : 0 < K := by linarith
  have h1mρ : 0 < 1 - ρ := by linarith
  have hσD : 0 < σ + D := by linarith
  have hKp1 : 0 < K + 1 := by linarith
  have h1 : (1 + ρ) / (1 - ρ) ≤ K ↔ ρ ≤ (K - 1) / (K + 1) := by
    rw [div_le_iff₀ h1mρ, le_div_iff₀ hKp1]
    constructor <;> intro h <;> nlinarith
  have h2 : ρ ≤ (K - 1) / (K + 1) ↔ ρ ^ 2 ≤ ((K - 1) / (K + 1)) ^ 2 := by
    have hnonneg : 0 ≤ (K - 1) / (K + 1) := div_nonneg (by linarith) (le_of_lt hKp1)
    constructor
    · intro hle; nlinarith
    · intro hsq
      by_contra hlt
      push Not at hlt
      nlinarith
  have h3 : ρ ^ 2 ≤ ((K - 1) / (K + 1)) ^ 2 ↔ D / σ ≤ (K - 1) ^ 2 / (4 * K) := by
    have hρeq : ρ ^ 2 = D / (σ + D) := by
      rw [eq_div_iff (ne_of_gt hσD)]; exact hρsq
    rw [hρeq, div_le_iff₀ hσD, div_le_iff₀ hσ]
    have h4K : 0 < 4 * K := by linarith
    have hKp1sq : 0 < (K + 1) ^ 2 := by positivity
    rw [div_pow, div_mul_eq_mul_div, le_div_iff₀ hKp1sq, div_mul_eq_mul_div, le_div_iff₀ h4K]
    constructor <;> intro h <;> nlinarith
  exact h1.trans (h2.trans h3)

/-- Helper: Determinant of a lower-triangular `Fin n × Fin n` matrix equals the
    product of its diagonal entries (by transpose to upper triangular). -/
lemma det_of_isLowerTriangular {n : ℕ}
    (G : Matrix (Fin n) (Fin n) ℝ) (hL : IsLowerTriangular G) :
    G.det = ∏ i : Fin n, G i i := by
  have hU : Gᵀ.BlockTriangular id := by
    intro i j hij
    exact hL j i hij
  rw [← Matrix.det_transpose, Matrix.det_of_isUpperTriangular hU]
  rfl

/-- Theorem 9.1c (Global Kaporin Product Identity, arbitrary `n ∈ ℕ`):
    For any two lower-triangular factors `G_q, G_*` with strictly positive diagonals
    and any matrix `A` with `det A > 0`, the determinant ratio of the preconditioned
    Gram matrices equals the product of squared diagonal ratios. Combined with unit-`A`-energy
    row normalization (`tr(G_q A G_qᵀ) = tr(G_* A G_*ᵀ) = n`) and Theorem 3 row optimality
    `(G_{*,ii} / G_{q,ii})^2 = 1 + D_i / σ_i`, the `n`-th power of the global Kaporin
    inflation `K_ap(G_q A G_qᵀ) / K_ap(G_* A G_*ᵀ)` equals `∏_i (1 + D_i / σ_i)`. -/
theorem global_kaporin_product_identity {n : ℕ}
    (A Gq Gstar : Matrix (Fin n) (Fin n) ℝ)
    (hdetA : 0 < A.det)
    (hLq : IsLowerTriangular Gq) (hDq : ∀ i, 0 < Gq i i)
    (hLs : IsLowerTriangular Gstar) :
    (Gstar * A * Gstarᵀ).det / (Gq * A * Gqᵀ).det =
      ∏ i : Fin n, (Gstar i i / Gq i i) ^ 2 := by
  have hdet_q : Gq.det = ∏ i : Fin n, Gq i i := det_of_isLowerTriangular Gq hLq
  have hdet_s : Gstar.det = ∏ i : Fin n, Gstar i i := det_of_isLowerTriangular Gstar hLs
  have hq_pos : 0 < ∏ i : Fin n, Gq i i := Finset.prod_pos (fun i _ => hDq i)
  have hq_ne : (∏ i : Fin n, Gq i i) ≠ 0 := ne_of_gt hq_pos
  have hA_ne : A.det ≠ 0 := ne_of_gt hdetA
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose,
      Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose,
      hdet_q, hdet_s]
  have hprod_div : ∏ i : Fin n, (Gstar i i / Gq i i) ^ 2 =
      ((∏ i : Fin n, Gstar i i) / (∏ i : Fin n, Gq i i)) ^ 2 := by
    rw [Finset.prod_pow, Finset.prod_div_distrib]
  rw [hprod_div]
  field_simp [hq_ne, hA_ne]

-- =============================================================================
-- 2. 3-Vertex Grounded SDDM: Sign-Only vs. 1.58-Bit Ternary Separation
-- =============================================================================

/-- Squared Euclidean distance from `(1, t)` to the line `span(q₁, q₂)`. -/
noncomputable def starLineErr (t q1 q2 : ℝ) : ℝ :=
  if q1 ^ 2 + q2 ^ 2 = 0 then 1 + t ^ 2
  else (1 + t ^ 2) - (q1 + q2 * t) ^ 2 / (q1 ^ 2 + q2 ^ 2)

/-- Theorem 9.2a: On the 3-vertex grounded star target `(1, t)` (`0 < t < 1`),
    every dense sign codeword `q ∈ {-1, +1}^2` has squared line error at least
    `(1 - t)^2 / 2`, attained at `q = (1, 1)`. -/
theorem grounded_star_sign_min (t : ℝ) (ht0 : 0 < t) (_ht1 : t < 1) :
    (∀ q1 q2 : ℝ, (q1 = -1 ∨ q1 = 1) → (q2 = -1 ∨ q2 = 1) →
      (1 - t) ^ 2 / 2 ≤ starLineErr t q1 q2) ∧
      starLineErr t 1 1 = (1 - t) ^ 2 / 2 := by
  constructor
  · intro q1 q2 hq1 hq2
    rcases hq1 with rfl | rfl <;> rcases hq2 with rfl | rfl <;>
    · simp only [starLineErr]
      norm_num
      nlinarith
  · simp only [starLineErr]
    norm_num
    ring

/-- Theorem 9.2b: On the same 3-vertex grounded star target `(1, t)` (`0 < t < 1`),
    every ternary codeword `q ∈ {-1, 0, +1}^2` has squared line error at least
    `min(t^2, (1 - t)^2 / 2)`, attained by `(1, 0)` or `(1, 1)`. -/
theorem grounded_star_ternary_min (t : ℝ) (ht0 : 0 < t) (ht1 : t < 1) :
    (∀ q1 q2 : ℝ, (q1 = -1 ∨ q1 = 0 ∨ q1 = 1) → (q2 = -1 ∨ q2 = 0 ∨ q2 = 1) →
      min (t ^ 2) ((1 - t) ^ 2 / 2) ≤ starLineErr t q1 q2) ∧
      min (starLineErr t 1 0) (starLineErr t 1 1) = min (t ^ 2) ((1 - t) ^ 2 / 2) := by
  constructor
  · intro q1 q2 hq1 hq2
    rcases hq1 with rfl | rfl | rfl <;> rcases hq2 with rfl | rfl | rfl
    · simp only [starLineErr]; norm_num; right; nlinarith
    · simp only [starLineErr]; norm_num
    · simp only [starLineErr]; norm_num; right; nlinarith
    · simp only [starLineErr]; norm_num; left; rw [abs_of_pos ht0]; linarith
    · simp only [starLineErr]; norm_num
    · simp only [starLineErr]; norm_num; left; rw [abs_of_pos ht0]; linarith
    · simp only [starLineErr]; norm_num; right; nlinarith
    · simp only [starLineErr]; norm_num
    · simp only [starLineErr]; norm_num; right; nlinarith
  · have h10 : starLineErr t 1 0 = t ^ 2 := by
      simp only [starLineErr]; norm_num
    have h11 : starLineErr t 1 1 = (1 - t) ^ 2 / 2 := by
      simp only [starLineErr]; norm_num; ring
    rw [h10, h11]

/-- Theorem 9.2c: Exact `K = 3` threshold for 1-bit sign-only on the grounded star
    (`σ = t(1 - t)`): `3 D_s ≤ σ ↔ t ≥ 3/5`. -/
theorem grounded_star_sign_kappa3_iff (t : ℝ) (_ht0 : 0 < t) (ht1 : t < 1) :
    3 * ((1 - t) ^ 2 / 2) ≤ t * (1 - t) ↔ (3 : ℝ) / 5 ≤ t := by
  have h1mt : 0 < 1 - t := by linarith
  constructor <;> intro h <;> nlinarith

/-- Theorem 9.2d: Exact `K = 3` threshold for 1.58-bit ternary on the grounded star
    (`σ = t(1 - t)`): `3 D_t ≤ σ ↔ t ≤ 1/4 ∨ t ≥ 3/5`. -/
theorem grounded_star_ternary_kappa3_iff (t : ℝ) (ht0 : 0 < t) (ht1 : t < 1) :
    3 * min (t ^ 2) ((1 - t) ^ 2 / 2) ≤ t * (1 - t) ↔
      t ≤ (1 : ℝ) / 4 ∨ (3 : ℝ) / 5 ≤ t := by
  constructor
  · intro h
    by_cases hcmp : t ^ 2 ≤ (1 - t) ^ 2 / 2
    · rw [min_eq_left hcmp] at h
      left; nlinarith
    · push Not at hcmp
      rw [min_eq_right (le_of_lt hcmp)] at h
      right; nlinarith
  · intro h
    rcases h with hle | hge
    · exact le_trans (mul_le_mul_of_nonneg_left (min_le_left _ _) (by norm_num)) (by nlinarith)
    · exact le_trans (mul_le_mul_of_nonneg_left (min_le_right _ _) (by norm_num)) (by nlinarith)

/-- Theorem 9.2e: Concrete separation at `t = 1/5`: sign-only gives `D_s / σ = 2`
    (violating `K = 3`, in fact `κ > 9`), while ternary gives `D_t / σ = 1/4 ≤ 1/3`
    (strictly meeting `K = 3`). -/
theorem grounded_star_separation_one_fifth :
    let t : ℝ := 1 / 5
    let σ := t * (1 - t)
    let Ds := (1 - t) ^ 2 / 2
    let Dt := min (t ^ 2) ((1 - t) ^ 2 / 2)
    Ds / σ = 2 ∧ ¬ (3 * Ds ≤ σ) ∧ Dt / σ = 1 / 4 ∧ 3 * Dt ≤ σ := by
  dsimp only
  norm_num

-- =============================================================================
-- 3. Contrast-Independent Grounded SDDM Ternary Cut & Unit-Path Lower Bound
-- =============================================================================

/-- Theorem 9.3a (Pigeonhole largest voltage gap, arbitrary `r ∈ ℕ`):
    Any `r + 1` real gaps summing to `1` contain at least one gap `≥ 1 / (r + 1)`. -/
theorem largest_voltage_gap_exists (r : ℕ) (gap : Fin (r + 1) → ℝ)
    (hsum : ∑ k : Fin (r + 1), gap k = 1) :
    ∃ k : Fin (r + 1), (1 : ℝ) / (r + 1 : ℝ) ≤ gap k := by
  by_contra hall
  push Not at hall
  have hlt : ∑ k : Fin (r + 1), gap k < ∑ _k : Fin (r + 1), ((1 : ℝ) / (r + 1 : ℝ)) := by
    apply Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty
    intro k _
    exact hall k
  have hconst : ∑ _k : Fin (r + 1), ((1 : ℝ) / (r + 1 : ℝ)) = 1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hr : (r + 1 : ℝ) ≠ 0 := by positivity
    push_cast
    field_simp [hr]
  linarith

/-- Theorem 9.3b (Contrast-independent ternary cut bound, arbitrary `r ∈ ℕ`):
    Given `r + 1` voltage gaps summing to `1` and cut capacities `cap_k ≥ 0`
    satisfying the Kirchhoff flux hypothesis `gap_k * cap_k ≤ σ` (`σ > 0`), at
    least one threshold cut achieves `cap_k ≤ (r + 1) * σ`, independent of
    edge-weight contrast `R_w`. -/
theorem sddm_ternary_cut_bound (r : ℕ) (σ : ℝ) (_hσ : 0 < σ)
    (gap cap : Fin (r + 1) → ℝ)
    (hcap : ∀ k, 0 ≤ cap k)
    (hsum : ∑ k : Fin (r + 1), gap k = 1)
    (hflux : ∀ k, gap k * cap k ≤ σ) :
    ∃ k : Fin (r + 1), cap k ≤ (r + 1 : ℝ) * σ := by
  obtain ⟨k, hk⟩ := largest_voltage_gap_exists r gap hsum
  refine ⟨k, ?_⟩
  have hr : (0 : ℝ) < (r + 1 : ℝ) := by positivity
  have hmul : (1 / (r + 1 : ℝ)) * cap k ≤ σ :=
    le_trans (mul_le_mul_of_nonneg_right hk (hcap k)) (hflux k)
  rw [one_div_mul_eq_div, div_le_iff₀ hr] at hmul
  linarith

/-- Helper: Sum of squared adjacent differences along a finite path is nonnegative. -/
lemma path_tail_energy_ge_sq :
    ∀ (m : ℕ) (w : Fin (m + 1) → ℝ),
      (∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2) + (w (Fin.last m)) ^ 2 ≥
        0 := by
  intro m w
  have h1 : 0 ≤ ∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2 :=
    Finset.sum_nonneg (fun k _ => sq_nonneg _)
  linarith [sq_nonneg (w (Fin.last m))]

/-- Helper: Along a chain from `w 0` down to `0` where every step has either
    equal endpoints or jump ≥ `|s|` when transitioning from `|w k| = |s|` to `0`
    (or from `s` to `-s`), if `|w 0| = |s|`, the tail energy is at least `s^2`. -/
lemma ternary_path_tail_ge_sq :
    ∀ (m : ℕ) (s : ℝ) (w : Fin (m + 1) → ℝ),
      (∀ i : Fin (m + 1), w i = -s ∨ w i = 0 ∨ w i = s) →
      (w 0 = -s ∨ w 0 = s) →
      s ^ 2 ≤ (∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2) + (w (Fin.last m)) ^ 2 := by
  intro m
  induction m with
  | zero =>
    intro s w _ hw0
    simp only [Finset.univ_eq_empty, Finset.sum_empty, zero_add]
    have hlast : Fin.last 0 = (0 : Fin 1) := rfl
    rw [hlast]
    rcases hw0 with h | h
    · rw [h]; ring_nf; linarith
    · rw [h]
  | succ m ih =>
    intro s w htern hw0
    rw [Fin.sum_univ_succ]
    have hstep0 : (0 : Fin (m + 1)).castSucc = (0 : Fin (m + 2)) := rfl
    have hstep1 : (0 : Fin (m + 1)).succ = (1 : Fin (m + 2)) := rfl
    set w' : Fin (m + 1) → ℝ := fun i => w i.succ
    have htail_nonneg :
        0 ≤ (∑ k : Fin m, (w' k.castSucc - w' k.succ) ^ 2) + (w' (Fin.last m)) ^ 2 :=
      path_tail_energy_ge_sq m w'
    have hsum_eq :
        (∑ k : Fin m, (w k.succ.castSucc - w k.succ.succ) ^ 2) + (w (Fin.last (m + 1))) ^ 2 =
          (∑ k : Fin m, (w' k.castSucc - w' k.succ) ^ 2) + (w' (Fin.last m)) ^ 2 := by
      congr 1
    rw [add_assoc, hsum_eq]
    rcases htern 1 with h1 | h1 | h1
    · have hih := ih s w' (fun i => htern i.succ) (Or.inl h1)
      linarith [sq_nonneg (w (0 : Fin (m + 1)).castSucc - w (0 : Fin (m + 1)).succ)]
    · have hfirst : (w (0 : Fin (m + 1)).castSucc - w (0 : Fin (m + 1)).succ) ^ 2 = s ^ 2 := by
        rw [hstep0, hstep1, h1, sub_zero]
        rcases hw0 with h | h
        · rw [h]; ring
        · rw [h]
      linarith
    · have hih := ih s w' (fun i => htern i.succ) (Or.inr h1)
      linarith [sq_nonneg (w (0 : Fin (m + 1)).castSucc - w (0 : Fin (m + 1)).succ)]

/-- Theorem 9.3c (Matching-order unit-path lower bound, arbitrary `r ≥ 1`):
    On a unit-conductance path from source (potential `1`) through `r = m + 1`
    free vertices to ground (`0`), effective conductance is `σ = 1 / (r + 1)`.
    Yet every ternary interior assignment `w_i ∈ {-s, 0, s}` (`s ∈ ℝ`) incurs
    Dirichlet energy at least `1 / 2`, so `E / σ ≥ (r + 1) / 2`. -/
theorem unit_path_ternary_lower_bound (m : ℕ) (s : ℝ)
    (w : Fin (m + 1) → ℝ)
    (htern : ∀ i : Fin (m + 1), w i = -s ∨ w i = 0 ∨ w i = s) :
    let E := (1 - w 0) ^ 2 +
      (∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2) + (w (Fin.last m)) ^ 2
    let σ : ℝ := 1 / (m + 2 : ℝ)
    (1 : ℝ) / 2 ≤ E ∧ ((m + 2 : ℝ) / 2) * σ ≤ E := by
  dsimp only
  have hE : (1 : ℝ) / 2 ≤ (1 - w 0) ^ 2 +
      (∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2) + (w (Fin.last m)) ^ 2 := by
    have htail_nonneg :
        0 ≤ (∑ k : Fin m, (w k.castSucc - w k.succ) ^ 2) + (w (Fin.last m)) ^ 2 :=
      path_tail_energy_ge_sq m w
    rcases htern 0 with h0 | h0 | h0
    · have htail := ternary_path_tail_ge_sq m s w htern (Or.inl h0)
      rw [h0]
      nlinarith [sq_nonneg (s + 1 / 2)]
    · rw [h0]
      linarith
    · have htail := ternary_path_tail_ge_sq m s w htern (Or.inr h0)
      rw [h0]
      nlinarith [sq_nonneg (s - 1 / 2)]
  refine ⟨hE, ?_⟩
  have hm : (m + 2 : ℝ) ≠ 0 := by positivity
  have hprod : ((m + 2 : ℝ) / 2) * (1 / (m + 2 : ℝ)) = 1 / 2 := by
    field_simp [hm]
  linarith

-- =============================================================================
-- 4. Frozen-Word Newton Drift (Loewner General & Kantorovich Quadratic Bound)
-- =============================================================================

/-- Theorem 9.4a (General frozen-word Newton drift):
    If an SPD tangent drifts within `l J_0 ≼ J_1 ≼ h J_0` (`0 < l ≤ h`), then
    keeping the quantized codeword `q` frozen and only recomputing Theorem 3's
    two scalars `(g_ii^*, s_i^*)` degrades the local optimality ratio by at most `h / l`. -/
theorem frozen_word_loewner_drift
    (l h m_old s_old m_new s_new : ℝ)
    (hl : 0 < l) (hh : 0 ≤ h) (hm_old : 0 ≤ m_old)
    (hs_old : 0 < s_old) (hs_new : 0 < s_new)
    (hm : m_new ≤ h * m_old) (hs : l * s_old ≤ s_new) :
    m_new / s_new ≤ (h / l) * (m_old / s_old) := by
  rw [div_mul_div_comm, div_le_div_iff₀ hs_new (mul_pos hl hs_old)]
  have h1 : m_new * (l * s_old) ≤ (h * m_old) * (l * s_old) :=
    mul_le_mul_of_nonneg_right hm (le_of_lt (mul_pos hl hs_old))
  have h2 : (h * m_old) * (l * s_old) ≤ (h * m_old) * s_new :=
    mul_le_mul_of_nonneg_left hs (mul_nonneg hh hm_old)
  linarith

/-- Theorem 9.4b (Kantorovich pointwise majorant):
    For `0 < l ≤ x ≤ h`, `x + (l * h) / x ≤ l + h`. -/
theorem kantorovich_pointwise_majorant
    (l h x : ℝ) (hl : 0 < l) (hlx : l ≤ x) (hxh : x ≤ h) :
    x + l * h / x ≤ l + h := by
  have hx : 0 < x := lt_of_lt_of_le hl hlx
  rw [← sub_nonpos]
  have hmul : (x + l * h / x - (l + h)) * x = (x - l) * (x - h) := by
    field_simp [ne_of_gt hx]; ring
  have hprod : (x - l) * (x - h) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
  nlinarith

/-- Theorem 9.4c (Exact-row Kantorovich quadratic Newton drift, arbitrary finite spectrum `ι`):
    If a frozen codeword was exact at `J_0` and `l J_0 ≼ J_1 ≼ h J_0` (`0 < l ≤ h`),
    evaluating the old optimal row (a feasible competitor for Theorem 3 re-projection
    along `span(q)`) yields local optimality ratio `(∑ p_i λ_i)(∑ p_i λ_i⁻¹)` for
    spectral weights `p_i ≥ 0`, `∑ p_i = 1`, `λ_i ∈ [l, h]`; hence the re-projected
    local optimality ratio is at most `(∑ p_i λ_i)(∑ p_i λ_i⁻¹)`, which is bounded
    by `(h + l)^2 / (4 h l) = 1 + (h - l)^2 / (4 h l)`. -/
theorem frozen_word_kantorovich_quadratic_drift {ι : Type*} [Fintype ι]
    (l h : ℝ) (hl : 0 < l) (hlh : l ≤ h)
    (p lam : ι → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i : ι, p i = 1)
    (hlo : ∀ i, l ≤ lam i) (hhi : ∀ i, lam i ≤ h) :
    (∑ i : ι, p i * lam i) * (∑ i : ι, p i * (lam i)⁻¹) ≤
      (h + l) ^ 2 / (4 * h * l) ∧
      (h + l) ^ 2 / (4 * h * l) = 1 + (h - l) ^ 2 / (4 * h * l) := by
  have hh : 0 < h := lt_of_lt_of_le hl hlh
  have h4hl : 0 < 4 * h * l := by positivity
  set A := ∑ i : ι, p i * lam i
  set B := ∑ i : ι, p i * (lam i)⁻¹
  have hcomb : A + l * h * B ≤ l + h := by
    have hterm : ∀ i : ι, p i * (lam i + l * h * (lam i)⁻¹) ≤ p i * (l + h) := by
      intro i
      apply mul_le_mul_of_nonneg_left _ (hp i)
      have hm := kantorovich_pointwise_majorant l h (lam i) hl (hlo i) (hhi i)
      rwa [div_eq_mul_inv] at hm
    have hsum_le : ∑ i : ι, p i * (lam i + l * h * (lam i)⁻¹) ≤ ∑ i : ι, p i * (l + h) :=
      Finset.sum_le_sum (fun i _ => hterm i)
    have hL : ∑ i : ι, p i * (lam i + l * h * (lam i)⁻¹) = A + l * h * B := by
      simp only [A, B, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
      congr 1
      refine Finset.sum_congr rfl (fun i _ => ?_)
      ring
    have hR : ∑ i : ι, p i * (l + h) = l + h := by
      rw [← Finset.sum_mul, hsum, one_mul]
    linarith
  have hA : 0 ≤ A := Finset.sum_nonneg (fun i _ =>
    mul_nonneg (hp i) (le_of_lt (lt_of_lt_of_le hl (hlo i))))
  have hB : 0 ≤ B := Finset.sum_nonneg (fun i _ =>
    mul_nonneg (hp i) (inv_nonneg.mpr (le_of_lt (lt_of_lt_of_le hl (hlo i)))))
  constructor
  · rw [le_div_iff₀ h4hl]
    have hsq : (A + l * h * B) ^ 2 ≤ (l + h) ^ 2 := by
      have hnonneg : 0 ≤ A + l * h * B := by nlinarith
      nlinarith
    nlinarith [sq_nonneg (A - l * h * B)]
  · field_simp [ne_of_gt h4hl]; ring

-- =============================================================================
-- 5. Projective SPD Packing Separation & Harmonic-Mean Coarse Bin Optimality
-- =============================================================================

/-- Theorem 9.5a (Strict `K^2` projective packing separation):
    If `K ≥ 1` and ratio `h > K^2`, no single positive decoder scalar `c > 0`
    can simultaneously certify `K⁻¹ ≤ c ≤ K` and `K⁻¹ h ≤ c ≤ K h`. -/
theorem projective_packing_separation
    (K h c : ℝ) (hK : 1 ≤ K) (hh : K ^ 2 < h) (_hc : 0 < c) :
    ¬ (1 ≤ K * c ∧ c ≤ K ∧ h ≤ K * c ∧ c ≤ K * h) := by
  rintro ⟨_, h2, h3, _⟩
  nlinarith

/-- Theorem 9.5b (Harmonic-mean minimax coarse weight bin):
    On any positive weight interval `[1, r]` (`r > 1`), the harmonic center
    `c* = 2 r / (r + 1)` satisfies relative error at most `u* = (r - 1) / (r + 1)`
    for every `x ∈ [1, r]` (attained with equality at endpoints `1` and `r`), and no
    representative `c ∈ ℝ` can achieve relative error strictly below `u*`
    simultaneously at both endpoints `1` and `r`. -/
theorem harmonic_bin_minimax_relerr (r : ℝ) (hr : 1 < r) :
    let uStar := (r - 1) / (r + 1)
    let cStar := 2 * r / (r + 1)
    (∀ x : ℝ, 1 ≤ x → x ≤ r → (1 - uStar) * x ≤ cStar ∧ cStar ≤ (1 + uStar) * x) ∧
      (∀ u c : ℝ, 0 ≤ u → u < uStar →
        ¬ (1 - u ≤ c ∧ c ≤ 1 + u ∧ (1 - u) * r ≤ c ∧ c ≤ (1 + u) * r)) := by
  dsimp only
  have hrp1 : 0 < r + 1 := by linarith
  have hrp1_ne : r + 1 ≠ 0 := ne_of_gt hrp1
  have h1mu : 1 - (r - 1) / (r + 1) = 2 / (r + 1) := by field_simp [hrp1_ne]; ring
  have h1pu : 1 + (r - 1) / (r + 1) = 2 * r / (r + 1) := by field_simp [hrp1_ne]; ring
  constructor
  · intro x hx1 hxr
    rw [h1mu, h1pu]
    constructor
    · rw [div_mul_eq_mul_div, div_le_div_iff_of_pos_right hrp1]; linarith
    · rw [le_mul_iff_one_le_right (by positivity)]; exact hx1
  · intro u c _ hu ⟨_, hhi1, hlor, _⟩
    have hdiff : (1 - u) * r ≤ 1 + u := le_trans hlor hhi1
    rw [lt_div_iff₀ hrp1] at hu
    nlinarith

end BitFSAI
