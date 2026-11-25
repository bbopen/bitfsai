import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Tactic

namespace BitFSAI

open Matrix Finset
open scoped BigOperators

/-!
# Multilevel Convergence and Polynomial-Cycle Theorems for BitFSAI

Machine-checked scalar and arbitrary-finite-dimensional quadratic-form obligations
supporting:
1. Cartesian 3D quantized smoothing and V-cycle scalar contraction;
2. FP8 E5M2 rounded W-cycle and arbitrary-dimensional defect-cycle reduction;
3. The monotone one-level effectiveness recurrence `e_new = min(p * g(e), d)`,
   degree-two Richardson concavity, and the work-convergence frontier `β² < p`;
4. Four-phase grounded SDDM cycle certificates (`κ ≤ 17` at `t = 2560/1307` and
   `κ ≤ 9` at `t = 30375/15853`), decoded-block singleton truncation bounds, and
   the 2×2 absolute-value damping overdamping witness.
-/

-- ============================================================================
-- Section 1: Cartesian 3D Smoothing and V-Cycle Scalar Bounds
-- ============================================================================

theorem variance_budget_3d (v2 err t : ℝ)
    (hv : v2 ≤ 1 / 4) (he0 : 0 ≤ err) (he : err ≤ 1 / 392)
    (ht : t * (1 - v2) = err) : 0 ≤ t ∧ t ≤ 1 / 294 := by
  have hpos : 0 < 1 - v2 := by linarith
  have ht0 : 0 ≤ t := by
    by_contra! hneg
    have hn := mul_neg_of_neg_of_pos hneg hpos
    linarith
  have hh : 0 ≤ 1 / 4 - v2 := by linarith
  have hp := mul_nonneg ht0 hh
  exact ⟨ht0, by nlinarith⟩

theorem scalar_smoothing_3d (lam : ℝ) (h0 : 0 ≤ lam) (h1 : lam ≤ 11 / 4) :
    (18 / 1250 : ℝ) * lam ≤ 1 - (1 - (18 / 25 : ℝ) * lam) ^ 2 := by
  have hp : 0 ≤ lam * ((11 / 4 : ℝ) - lam) := mul_nonneg h0 (by linarith)
  nlinarith

theorem vcycle_induction_scalar (C c q s t : ℝ)
    (hC : 0 < C) (hc : 0 < c) (hq : q * (C + c) = C)
    (hb : c * t ≤ C * (1 - s)) :
    0 < q ∧ q < 1 ∧ q * s + (1 - q) * t ≤ q := by
  have hsum : 0 < C + c := add_pos hC hc
  have hqeq : q = C / (C + c) := (eq_div_iff (ne_of_gt hsum)).2 hq
  have hq0 : 0 < q := by rw [hqeq]; exact div_pos hC hsum
  have hq1 : q < 1 := by
    rw [hqeq]
    exact (div_lt_iff₀ hsum).2 (by linarith)
  have hrel : (1 - q) * C = c * q := by nlinarith [hq]
  have hmul := mul_le_mul_of_nonneg_left hb (show 0 ≤ 1 - q by linarith)
  have h : c * (q * s + (1 - q) * t) ≤ c * q := calc
    c * (q * s + (1 - q) * t) = c * q * s + (1 - q) * (c * t) := by ring
    _ ≤ c * q * s + (1 - q) * (C * (1 - s)) := add_le_add le_rfl hmul
    _ = c * q * s + ((1 - q) * C) * (1 - s) := by ring
    _ = c * q := by rw [hrel]; ring
  refine ⟨hq0, hq1, ?_⟩
  by_contra! hbad
  have hlt := mul_lt_mul_of_pos_left hbad hc
  linarith

-- ============================================================================
-- Section 2: FP8 E5M2 Rounded W-Cycle and Defect-Cycle Matrix Reduction
-- ============================================================================

def coarseError (alpha q : ℝ) : ℝ := 1 - alpha + alpha * q ^ 2

theorem coarseError_nonneg (alpha q : ℝ) (ha : 0 ≤ alpha) (ha1 : alpha ≤ 1) :
    0 ≤ coarseError alpha q := by
  unfold coarseError
  nlinarith [mul_nonneg ha (sq_nonneg q)]

theorem coarseError_le (alpha q : ℝ) (hq : q ≤ 1) (hfix : 1 ≤ alpha * (1 + q)) :
    coarseError alpha q ≤ q := by
  unfold coarseError
  have hp := mul_nonneg (sub_nonneg.mpr hq) (sub_nonneg.mpr hfix)
  nlinarith [hp]

theorem fp8_e5m2_wcycle_constants :
    ((1 - (1 / 8 : ℝ)) / (1 + 1 / 8)) ^ 2 = (49 / 81 : ℝ) ∧
    (1 / 2 : ℝ) < 49 / 81 ∧
    ((1 - (49 / 81 : ℝ)) / (49 / 81)) = 32 / 49 := by
  norm_num

theorem fp8_e5m2_defect_constants :
    2 * (1 / 8 : ℝ) ^ 2 * (9 / 7) = 9 / 224 ∧
    (49 / 81 : ℝ) * (1 - 9 / 224) = 10535 / 18144 ∧
    (1 / 2 : ℝ) < 10535 / 18144 ∧
    (1 - (10535 / 18144 : ℝ)) / (10535 / 18144) = 7609 / 10535 := by
  norm_num

theorem guarded_smoothing_scalar (h : ℝ) (h0 : 0 ≤ h) (h1 : h ≤ 1) :
    0 ≤ 1 - h ∧ h ≤ 1 - (1 - h) ^ 2 := by
  exact ⟨by linarith, by nlinarith [mul_nonneg h0 (sub_nonneg.mpr h1)]⟩

theorem defect_cycle_homogeneous (alpha q eta K N s t : ℝ)
    (ha0 : 0 ≤ alpha) (_ha1 : alpha ≤ 1)
    (_hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (_he0 : 0 ≤ eta) (_he1 : eta ≤ 1)
    (hK : 0 ≤ K) (hs0 : 0 ≤ s) (hs1 : s ≤ N)
    (ht : t ≤ eta * s + K * (N - s))
    (hfix : 1 ≤ alpha * (1 - eta) * (1 + q))
    (hbudget : 2 * alpha * K * (1 - q) ≤ q) :
    coarseError alpha q * s + (1 - coarseError alpha q) * t ≤ q * N := by
  have hq2 : 0 ≤ 1 - q ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hq1) (show 0 ≤ 1 + q by linarith)]
  have hrem : 0 ≤ 1 - coarseError alpha q := by
    unfold coarseError
    nlinarith [mul_nonneg ha0 hq2]
  have hfirst : coarseError alpha q + (1 - coarseError alpha q) * eta ≤ q := by
    have hp := mul_nonneg (sub_nonneg.mpr hq1) (sub_nonneg.mpr hfix)
    unfold coarseError
    nlinarith [hp]
  have hsecond : (1 - coarseError alpha q) * K ≤ q := by
    have haux : 0 ≤ alpha * K := mul_nonneg ha0 hK
    have hp := mul_nonneg haux (sq_nonneg (1 - q))
    unfold coarseError
    nlinarith [hp, hbudget]
  calc
    coarseError alpha q * s + (1 - coarseError alpha q) * t
      ≤ coarseError alpha q * s + (1 - coarseError alpha q) * (eta * s + K * (N - s)) :=
        add_le_add le_rfl (mul_le_mul_of_nonneg_left ht hrem)
    _ = (coarseError alpha q + (1 - coarseError alpha q) * eta) * s +
        ((1 - coarseError alpha q) * K) * (N - s) := by ring
    _ ≤ q * s + q * (N - s) := add_le_add
        (mul_le_mul_of_nonneg_right hfirst hs0)
        (mul_le_mul_of_nonneg_right hsecond (sub_nonneg.mpr hs1))
    _ = q * N := by ring

def quadForm {ι : Type*} [Fintype ι] (A : Matrix ι ι ℝ) (x : ι → ℝ) : ℝ :=
  ∑ i, ∑ j, x i * A i j * x j

def vecNormSq {ι : Type*} [Fintype ι] (x : ι → ℝ) : ℝ :=
  ∑ i, (x i) ^ 2

theorem defect_cycle_matrix_reduction {ι : Type*} [Fintype ι]
    (E S2 T : Matrix ι ι ℝ) (alpha q eta K : ℝ)
    (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (hK : 0 ≤ K)
    (hfix : 1 ≤ alpha * (1 - eta) * (1 + q))
    (hbudget : 2 * alpha * K * (1 - q) ≤ q)
    (hs : ∀ x, 0 ≤ quadForm S2 x ∧ quadForm S2 x ≤ vecNormSq x)
    (ht : ∀ x, quadForm T x ≤ eta * quadForm S2 x + K * (vecNormSq x - quadForm S2 x))
    (hE : ∀ x, quadForm E x ≤ coarseError alpha q * quadForm S2 x +
      (1 - coarseError alpha q) * quadForm T x) :
    ∀ x, quadForm E x ≤ q * vecNormSq x := by
  intro x
  exact le_trans (hE x) (defect_cycle_homogeneous alpha q eta K
    (vecNormSq x) (quadForm S2 x) (quadForm T x) ha0 ha1 hq0 hq1 he0 he1 hK
    (hs x).1 (hs x).2 (ht x) hfix hbudget)

private lemma quadForm_eq_dotProduct {ι : Type*} [Fintype ι]
    (M : Matrix ι ι ℝ) (x : ι → ℝ) :
    quadForm M x = x ⬝ᵥ M.mulVec x := by
  simp only [quadForm, dotProduct, mulVec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

private lemma quadForm_add {ι : Type*} [Fintype ι]
    (M N : Matrix ι ι ℝ) (x : ι → ℝ) :
    quadForm (M + N) x = quadForm M x + quadForm N x := by
  simp only [quadForm_eq_dotProduct, Matrix.add_mulVec, dotProduct_add]

private lemma quadForm_sub {ι : Type*} [Fintype ι]
    (M N : Matrix ι ι ℝ) (x : ι → ℝ) :
    quadForm (M - N) x = quadForm M x - quadForm N x := by
  simp only [quadForm_eq_dotProduct, Matrix.sub_mulVec, dotProduct_sub]

private lemma quadForm_smul {ι : Type*} [Fintype ι]
    (c : ℝ) (M : Matrix ι ι ℝ) (x : ι → ℝ) :
    quadForm (c • M) x = c * quadForm M x := by
  simp only [quadForm_eq_dotProduct, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

private lemma quadForm_conj_transpose {ι κ : Type*} [Fintype ι] [Fintype κ]
    (P : Matrix ι κ ℝ) (M : Matrix κ κ ℝ) (x : ι → ℝ) :
    quadForm (P * M * Pᵀ) x = quadForm M (Pᵀ.mulVec x) := by
  rw [quadForm_eq_dotProduct, quadForm_eq_dotProduct, Matrix.mul_assoc,
    ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec]
  have hvt : x ᵥ* P = Pᵀ.mulVec x := by
    have := Matrix.vecMul_transpose Pᵀ x
    simpa [Matrix.transpose_transpose] using this
  rw [hvt]

private lemma symm_inv_of_mul_eq_one {κ : Type*} [Fintype κ] [DecidableEq κ]
    (M Minv : Matrix κ κ ℝ) (hMs : Mᵀ = M)
    (_hR : M * Minv = 1) (hL : Minv * M = 1) :
    Minvᵀ = Minv := by
  have h1 : M * Minvᵀ = 1 := by
    have ht := congr_arg Matrix.transpose hL
    simpa [Matrix.transpose_mul, hMs] using ht
  calc
    Minvᵀ = (Minv * M) * Minvᵀ := by rw [hL, Matrix.one_mul]
    _ = Minv * (M * Minvᵀ) := by rw [Matrix.mul_assoc]
    _ = Minv := by rw [h1, Matrix.mul_one]

private lemma loewner_inv_antitone_explicit {κ : Type*} [Fintype κ] [DecidableEq κ]
    (M Minv N Ninv : Matrix κ κ ℝ)
    (hMs : Mᵀ = M) (hNs : Nᵀ = N)
    (hMR : M * Minv = 1) (hML : Minv * M = 1)
    (hNR : N * Ninv = 1) (hNL : Ninv * N = 1)
    (hM0 : ∀ y, 0 ≤ quadForm M y)
    (hMN : ∀ y, quadForm M y ≤ quadForm N y) :
    ∀ y, quadForm Ninv y ≤ quadForm Minv y := by
  have hMinv_s : Minvᵀ = Minv := symm_inv_of_mul_eq_one M Minv hMs hMR hML
  have hNinv_s : Ninvᵀ = Ninv := symm_inv_of_mul_eq_one N Ninv hNs hNR hNL
  have hMinv0 : ∀ w, 0 ≤ quadForm Minv w := by
    intro w
    have hid : Minv = Minv * M * Minvᵀ := by
      rw [hML, Matrix.one_mul, hMinv_s]
    rw [hid, quadForm_conj_transpose]
    exact hM0 (Minvᵀ.mulVec w)
  intro y
  let K := (1 : Matrix κ κ ℝ) - Ninv * M
  have hKT : Kᵀ = (1 : Matrix κ κ ℝ) - M * Ninv := by
    simp only [K, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_mul, hMs, hNinv_s]
  have hident : Minv - Ninv = Ninv * (N - M) * Ninvᵀ + K * Minv * Kᵀ := by
    rw [hNinv_s, hKT]
    have h1 : Ninv * (N - M) * Ninv = Ninv - Ninv * M * Ninv := by
      rw [Matrix.mul_sub, Matrix.sub_mul, hNL, Matrix.one_mul]
    have h2 : K * Minv * ((1 : Matrix κ κ ℝ) - M * Ninv) =
        Minv - Ninv - Ninv + Ninv * M * Ninv := by
      simp only [K]
      rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc Ninv M Minv, hMR, Matrix.mul_one,
        Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul,
        ← Matrix.mul_assoc Minv M Ninv, hML, Matrix.one_mul, ← Matrix.mul_assoc Ninv M Ninv]
      abel
    rw [h1, h2]
    abel
  have hdiff_nn : 0 ≤ quadForm (Minv - Ninv) y := by
    rw [hident, quadForm_add, quadForm_conj_transpose, quadForm_conj_transpose]
    have ht1 : 0 ≤ quadForm (N - M) (Ninvᵀ.mulVec y) := by
      rw [quadForm_sub]
      linarith [hMN (Ninvᵀ.mulVec y)]
    have ht2 : 0 ≤ quadForm Minv (Kᵀ.mulVec y) := hMinv0 (Kᵀ.mulVec y)
    linarith
  rw [quadForm_sub] at hdiff_nn
  linarith

private lemma posDef_quadForm_nonneg {ι : Type*} [Fintype ι]
    (M : Matrix ι ι ℝ) (hM : M.PosDef) (x : ι → ℝ) :
    0 ≤ quadForm M x := by
  rw [quadForm_eq_dotProduct]
  by_cases hx : x = 0
  · subst hx; simp
  · exact le_of_lt (by simpa using (Matrix.posDef_iff_dotProduct_mulVec.mp hM).2 hx)

/-- Arbitrary finite-dimensional Galerkin V-cycle Loewner sandwich (Theorem 5):
    if `A` is SPD, `P` has full column rank, `R` is symmetric with `0 ≤ R ≤ A⁻¹`,
    `A_c` is symmetric with `γ * (Pᵀ A P) ≤ A_c` (`γ > 0`), and `B_c` is symmetric
    with `0 ≤ B_c ≤ A_c⁻¹`, then `T = Pᵀ A P` and `A_c` are SPD, `Bᵀ = B`,
    `γ * A_c⁻¹ ≤ (Pᵀ A P)⁻¹`, and the symmetric two-level V-cycle preconditioner
    `B = 2 R - R A R + γ (I - R A) P B_c Pᵀ (I - A R)` satisfies `0 ≤ R ≤ B ≤ A⁻¹`. -/
theorem vcycle_upper_bound_of_galerkin_sandwich
    {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]
    (A R : Matrix ι ι ℝ) (P : Matrix ι κ ℝ) (Ac Bc : Matrix κ κ ℝ) (γ : ℝ)
    (hA : A.PosDef)
    (hR_symm : Rᵀ = R)
    (hR0 : ∀ x, 0 ≤ quadForm R x)
    (hR_ub : ∀ x, quadForm R x ≤ quadForm A⁻¹ x)
    (hP : Function.Injective P.mulVec)
    (hAc_symm : Acᵀ = Ac)
    (hBc_symm : Bcᵀ = Bc)
    (hγ : 0 < γ)
    (hAc_lb : ∀ y, γ * quadForm (Pᵀ * A * P) y ≤ quadForm Ac y)
    (hBc0 : ∀ y, 0 ≤ quadForm Bc y)
    (hBc_ub : ∀ y, quadForm Bc y ≤ quadForm Ac⁻¹ y) :
    let T := Pᵀ * A * P
    let S := (1 : Matrix ι ι ℝ) - R * A
    let B := (2 : ℝ) • R - R * A * R + γ • (S * P * Bc * Pᵀ * Sᵀ)
    T.PosDef ∧
    Ac.PosDef ∧
    Bᵀ = B ∧
    (∀ y, γ * quadForm Ac⁻¹ y ≤ quadForm T⁻¹ y) ∧
    (∀ x, quadForm R x ≤ quadForm B x ∧ quadForm B x ≤ quadForm A⁻¹ x) := by
  intro T S B
  have hAs : Aᵀ = A := by
    simpa [Matrix.IsHermitian] using (Matrix.posDef_iff_dotProduct_mulVec.mp hA).1
  have hAdet : IsUnit A.det := isUnit_iff_ne_zero.mpr (ne_of_gt hA.det_pos)
  have hAR : A * A⁻¹ = 1 := Matrix.mul_nonsing_inv A hAdet
  have hAL : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul A hAdet
  have hAinvs : (A⁻¹)ᵀ = A⁻¹ := symm_inv_of_mul_eq_one A A⁻¹ hAs hAR hAL
  have hA0 : ∀ x, 0 ≤ quadForm A x := posDef_quadForm_nonneg A hA
  have hAinv0 : ∀ x, 0 ≤ quadForm A⁻¹ x := by
    intro x
    have hid : A⁻¹ = A⁻¹ * A * (A⁻¹)ᵀ := by rw [hAL, Matrix.one_mul, hAinvs]
    rw [hid, quadForm_conj_transpose]
    exact hA0 _
  have hTs : Tᵀ = T := by
    simp only [T, Matrix.transpose_mul, Matrix.transpose_transpose, hAs, Matrix.mul_assoc]
  have hT_pd : T.PosDef := by
    rw [Matrix.posDef_iff_dotProduct_mulVec]
    refine ⟨by simpa [Matrix.IsHermitian] using hTs, ?_⟩
    intro y hy
    have hPy : P.mulVec y ≠ 0 := fun h0 => hy (hP (by simpa using h0))
    have hquad : star y ⬝ᵥ T.mulVec y = star (P.mulVec y) ⬝ᵥ A.mulVec (P.mulVec y) := by
      simp only [T, star_trivial, ← quadForm_eq_dotProduct]
      exact quadForm_conj_transpose Pᵀ A y
    rw [hquad]
    exact (Matrix.posDef_iff_dotProduct_mulVec.mp hA).2 hPy
  have hAc_pd : Ac.PosDef := by
    rw [Matrix.posDef_iff_dotProduct_mulVec]
    refine ⟨by simpa [Matrix.IsHermitian] using hAc_symm, ?_⟩
    intro y hy
    have hTpos : 0 < quadForm T y := by
      rw [quadForm_eq_dotProduct]
      simpa using (Matrix.posDef_iff_dotProduct_mulVec.mp hT_pd).2 hy
    have hAcpos : 0 < quadForm Ac y := lt_of_lt_of_le (mul_pos hγ hTpos) (hAc_lb y)
    rw [quadForm_eq_dotProduct] at hAcpos
    simpa using hAcpos
  have hTdet : IsUnit T.det := isUnit_iff_ne_zero.mpr (ne_of_gt hT_pd.det_pos)
  have hTR : T * T⁻¹ = 1 := Matrix.mul_nonsing_inv T hTdet
  have hTL : T⁻¹ * T = 1 := Matrix.nonsing_inv_mul T hTdet
  have hTinvs : (T⁻¹)ᵀ = T⁻¹ := symm_inv_of_mul_eq_one T T⁻¹ hTs hTR hTL
  have hAcdet : IsUnit Ac.det := isUnit_iff_ne_zero.mpr (ne_of_gt hAc_pd.det_pos)
  have hAcR : Ac * Ac⁻¹ = 1 := Matrix.mul_nonsing_inv Ac hAcdet
  have hAcL : Ac⁻¹ * Ac = 1 := Matrix.nonsing_inv_mul Ac hAcdet
  have hcoarse_inv : ∀ y, γ * quadForm Ac⁻¹ y ≤ quadForm T⁻¹ y := by
    have hγ_ne : γ ≠ 0 := ne_of_gt hγ
    have h1 : (γ • T) * (γ⁻¹ • T⁻¹) = 1 := by
      rw [smul_mul_smul_comm, mul_inv_cancel₀ hγ_ne, hTR, one_smul]
    have h2 : (γ⁻¹ • T⁻¹) * (γ • T) = 1 := by
      rw [smul_mul_smul_comm, inv_mul_cancel₀ hγ_ne, hTL, one_smul]
    have hanti := loewner_inv_antitone_explicit (γ • T) (γ⁻¹ • T⁻¹) Ac Ac⁻¹
      (by simp [Matrix.transpose_smul, hTs]) hAc_symm h1 h2 hAcR hAcL
      (fun y => by rw [quadForm_smul]; exact mul_nonneg hγ.le (posDef_quadForm_nonneg T hT_pd y))
      (fun y => by rw [quadForm_smul]; exact hAc_lb y)
    intro y
    have hy := hanti y
    rw [quadForm_smul] at hy
    have hmul := mul_le_mul_of_nonneg_left hy hγ.le
    rwa [← mul_assoc, mul_inv_cancel₀ hγ_ne, one_mul] at hmul
  have hgalerkin_proj : ∀ z, quadForm (P * T⁻¹ * Pᵀ) z ≤ quadForm A⁻¹ z := by
    intro z
    let P_proj := (1 : Matrix ι ι ℝ) - P * T⁻¹ * Pᵀ * A
    have hP_projT : P_projᵀ = (1 : Matrix ι ι ℝ) - A * P * T⁻¹ * Pᵀ := by
      simp only [P_proj, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_mul,
        Matrix.transpose_transpose, hAs, hTinvs, Matrix.mul_assoc]
    have hproj_id : A⁻¹ - P * T⁻¹ * Pᵀ = P_proj * A⁻¹ * P_projᵀ := by
      rw [hP_projT]
      simp only [P_proj]
      have hmid : (P * T⁻¹ * Pᵀ * A * A⁻¹) = P * T⁻¹ * Pᵀ := by
        rw [Matrix.mul_assoc (P * T⁻¹ * Pᵀ) A A⁻¹, hAR, Matrix.mul_one]
      have hPAP : (P * T⁻¹ * Pᵀ) * (A * P * T⁻¹ * Pᵀ) = P * T⁻¹ * Pᵀ := by
        calc
          (P * T⁻¹ * Pᵀ) * (A * P * T⁻¹ * Pᵀ)
            = P * T⁻¹ * (Pᵀ * A * P) * T⁻¹ * Pᵀ := by simp only [Matrix.mul_assoc]
          _ = P * (T⁻¹ * T) * T⁻¹ * Pᵀ := by simp only [T, Matrix.mul_assoc]
          _ = P * T⁻¹ * Pᵀ := by rw [hTL, Matrix.mul_one]
      have hA1 : A⁻¹ * (A * P * T⁻¹ * Pᵀ) = P * T⁻¹ * Pᵀ := by
        simp only [← Matrix.mul_assoc, hAL, Matrix.one_mul]
      rw [Matrix.sub_mul, Matrix.one_mul, hmid, Matrix.mul_sub, Matrix.mul_one,
        Matrix.sub_mul, hA1, hPAP]
      abel
    have hnn : 0 ≤ quadForm (A⁻¹ - P * T⁻¹ * Pᵀ) z := by
      rw [hproj_id, quadForm_conj_transpose]
      exact hAinv0 (P_projᵀ.mulVec z)
    rw [quadForm_sub] at hnn
    linarith
  have hRAR_le : ∀ x, quadForm (R * A * R) x ≤ quadForm R x := by
    intro x
    suffices h_eps : ∀ ε : ℝ, 0 < ε →
        quadForm (R * A * R) x - quadForm R x ≤ ε * (quadForm A⁻¹ x - quadForm R x) by
      by_contra! hbad
      let δ := quadForm (R * A * R) x - quadForm R x
      let C := quadForm A⁻¹ x - quadForm R x
      have hδ : 0 < δ := sub_pos.mpr hbad
      have hC : 0 ≤ C := sub_nonneg.mpr (hR_ub x)
      by_cases hC0 : C = 0
      · have h1 := h_eps 1 one_pos
        change δ ≤ 1 * C at h1
        rw [hC0] at h1
        linarith
      · have hCpos : 0 < C := lt_of_le_of_ne hC (Ne.symm hC0)
        have h1 := h_eps (δ / (2 * C)) (div_pos hδ (by positivity))
        change δ ≤ (δ / (2 * C)) * C at h1
        have h2 : (δ / (2 * C)) * C = δ / 2 := by field_simp [ne_of_gt hCpos]
        linarith
    intro ε hε
    let Mε := R + ε • A⁻¹
    have hMε_s : Mεᵀ = Mε := by
      simp only [Mε, Matrix.transpose_add, Matrix.transpose_smul, hR_symm, hAinvs]
    have hMε_pd : Mε.PosDef := by
      rw [Matrix.posDef_iff_dotProduct_mulVec]
      refine ⟨by simpa [Matrix.IsHermitian] using hMε_s, ?_⟩
      intro v hv
      have hAv_ne : A⁻¹.mulVec v ≠ 0 := by
        intro h0
        have hv0 : v = (A * A⁻¹).mulVec v := by rw [hAR, Matrix.one_mulVec]
        rw [← Matrix.mulVec_mulVec, h0, Matrix.mulVec_zero] at hv0
        exact hv hv0
      have hAinv_pos : 0 < quadForm A⁻¹ v := by
        have hid : A⁻¹ = A⁻¹ * A * (A⁻¹)ᵀ := by rw [hAL, Matrix.one_mul, hAinvs]
        rw [hid, quadForm_conj_transpose, quadForm_eq_dotProduct, hAinvs]
        simpa using (Matrix.posDef_iff_dotProduct_mulVec.mp hA).2 hAv_ne
      have hsum : quadForm Mε v = quadForm R v + ε * quadForm A⁻¹ v := by
        simp only [Mε, quadForm_add, quadForm_smul]
      rw [star_trivial, ← quadForm_eq_dotProduct]
      simpa [hsum] using add_pos_of_nonneg_of_pos (hR0 v) (mul_pos hε hAinv_pos)
    have hMε_det : IsUnit Mε.det := isUnit_iff_ne_zero.mpr (ne_of_gt hMε_pd.det_pos)
    have hMεR : Mε * Mε⁻¹ = 1 := Matrix.mul_nonsing_inv Mε hMε_det
    have hMεL : Mε⁻¹ * Mε = 1 := Matrix.nonsing_inv_mul Mε hMε_det
    have h1ε_pos : 0 < 1 + ε := by linarith
    have h1ε_ne : 1 + ε ≠ 0 := ne_of_gt h1ε_pos
    have hN1 : ((1 + ε) • A⁻¹) * ((1 + ε)⁻¹ • A) = 1 := by
      rw [smul_mul_smul_comm, mul_inv_cancel₀ h1ε_ne, hAL, one_smul]
    have hN2 : ((1 + ε)⁻¹ • A) * ((1 + ε) • A⁻¹) = 1 := by
      rw [smul_mul_smul_comm, inv_mul_cancel₀ h1ε_ne, hAR, one_smul]
    have hanti := loewner_inv_antitone_explicit Mε Mε⁻¹ ((1 + ε) • A⁻¹) ((1 + ε)⁻¹ • A)
      hMε_s (by simp [Matrix.transpose_smul, hAinvs]) hMεR hMεL hN1 hN2
      (posDef_quadForm_nonneg Mε hMε_pd)
      (fun v => by
        simp only [Mε, quadForm_add, quadForm_smul]
        linarith [hR_ub v])
    have hx_step := hanti (Mε.mulVec x)
    have hlhs : quadForm ((1 + ε)⁻¹ • A) (Mε.mulVec x) =
        (1 + ε)⁻¹ * quadForm (Mε * A * Mε) x := by
      rw [quadForm_smul]
      have hconj := quadForm_conj_transpose Mεᵀ A x
      simpa [hMε_s] using congr_arg ((1 + ε)⁻¹ * ·) hconj.symm
    have hrhs : quadForm Mε⁻¹ (Mε.mulVec x) = quadForm Mε x := by
      have hconj := quadForm_conj_transpose Mεᵀ Mε⁻¹ x
      rw [Matrix.transpose_transpose, hMε_s, hMεR, Matrix.one_mul] at hconj
      exact hconj.symm
    rw [hlhs, hrhs] at hx_step
    have hmul := mul_le_mul_of_nonneg_left hx_step h1ε_pos.le
    rw [← mul_assoc, mul_inv_cancel₀ h1ε_ne, one_mul] at hmul
    have hexp_lhs : Mε * A * Mε = R * A * R + (2 * ε) • R + (ε ^ 2) • A⁻¹ := by
      have hRA : R * A * A⁻¹ = R := by rw [Matrix.mul_assoc, hAR, Matrix.mul_one]
      simp only [Mε, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
        smul_smul, hAL, Matrix.one_mul, hRA]
      ext i j
      simp [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      ring
    rw [hexp_lhs, quadForm_add, quadForm_add, quadForm_smul, quadForm_smul] at hmul
    simp only [Mε, quadForm_add, quadForm_smul] at hmul
    nlinarith
  have hBs : Bᵀ = B := by
    simp only [B, Matrix.transpose_add, Matrix.transpose_sub, Matrix.transpose_smul,
      Matrix.transpose_mul, Matrix.transpose_transpose, hR_symm, hAs, hBc_symm, Matrix.mul_assoc]
  refine ⟨hT_pd, hAc_pd, hBs, hcoarse_inv, ?_⟩
  intro x
  constructor
  · have hcoarse_nn : 0 ≤ quadForm (γ • (S * P * Bc * Pᵀ * Sᵀ)) x := by
      rw [quadForm_smul]
      have hassoc : S * P * Bc * Pᵀ * Sᵀ = (S * P) * Bc * (S * P)ᵀ := by
        simp only [Matrix.transpose_mul, Matrix.mul_assoc]
      rw [hassoc, quadForm_conj_transpose]
      exact mul_nonneg hγ.le (hBc0 _)
    have hsm : quadForm ((2 : ℝ) • R - R * A * R) x = 2 * quadForm R x - quadForm (R * A * R) x := by
      rw [quadForm_sub, quadForm_smul]
    simp only [B, quadForm_add, hsm]
    linarith [hRAR_le x]
  · have hST : Sᵀ = (1 : Matrix ι ι ℝ) - A * R := by
      simp only [S, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_mul, hAs, hR_symm]
    have herr_id : A⁻¹ - B = S * (A⁻¹ - γ • (P * Bc * Pᵀ)) * Sᵀ := by
      have hsm_err : A⁻¹ - ((2 : ℝ) • R - R * A * R) = S * A⁻¹ * Sᵀ := by
        rw [hST]
        simp only [S]
        rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc R A A⁻¹, hAR, Matrix.mul_one,
          Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul, ← Matrix.mul_assoc A⁻¹ A R, hAL,
          Matrix.one_mul, ← Matrix.mul_assoc R A R]
        ext i j
        simp [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
        ring
      have hcoarse_eq : γ • (S * P * Bc * Pᵀ * Sᵀ) = S * (γ • (P * Bc * Pᵀ)) * Sᵀ := by
        simp only [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]
      simp only [B]
      rw [sub_add_eq_sub_sub, hsm_err, hcoarse_eq, ← Matrix.sub_mul, ← Matrix.mul_sub]
    have herr_nn : 0 ≤ quadForm (A⁻¹ - B) x := by
      rw [herr_id, quadForm_conj_transpose, quadForm_sub, quadForm_smul, quadForm_conj_transpose]
      let z := Sᵀ.mulVec x
      have h1 : γ * quadForm Bc (Pᵀ.mulVec z) ≤ γ * quadForm Ac⁻¹ (Pᵀ.mulVec z) :=
        mul_le_mul_of_nonneg_left (hBc_ub (Pᵀ.mulVec z)) hγ.le
      have h2 : γ * quadForm Ac⁻¹ (Pᵀ.mulVec z) ≤ quadForm T⁻¹ (Pᵀ.mulVec z) :=
        hcoarse_inv (Pᵀ.mulVec z)
      have h3 : quadForm T⁻¹ (Pᵀ.mulVec z) ≤ quadForm A⁻¹ z := by
        rw [← quadForm_conj_transpose P T⁻¹ z]
        exact hgalerkin_proj z
      linarith
    rw [quadForm_sub] at herr_nn
    linarith

-- ============================================================================
-- Section 3: Monotone Effectiveness Recurrence and Polynomial Frontier
-- ============================================================================

theorem two_call_interval (e x : ℝ) (hx0 : e ≤ x) (hx1 : x ≤ 1) :
    (1 + e - 2 * x) ^ 2 ≤ (1 - e) ^ 2 := by
  have h := mul_nonneg (sub_nonneg.mpr hx0) (sub_nonneg.mpr hx1)
  nlinarith

theorem two_call_gain_upper (e f : ℝ) (hf : 0 ≤ f) (h : f * (1 + e) ^ 2 = 4 * e) :
    f ≤ 4 * e := by
  have hh : 0 ≤ (1 + e) ^ 2 - 1 := by
    have hpos : 0 ≤ e := by
      by_contra! hneg
      have : f * (1 + e) ^ 2 ≥ 0 := mul_nonneg hf (sq_nonneg _)
      linarith
    nlinarith [sq_nonneg e]
  nlinarith [mul_nonneg hf hh]

/-- Concavity of the two-call Richardson gain `1 - (1 - t*x)^2` on `[u, 1]`:
    its minimum on `[u, 1]` is attained at one of the two endpoints. -/
theorem two_call_concave_endpoint_min (t u x : ℝ)
    (hu : u ≤ x) (hx : x ≤ 1) :
    min (1 - (1 - t * u) ^ 2) (1 - (1 - t) ^ 2) ≤ 1 - (1 - t * x) ^ 2 := by
  have hprod : 0 ≤ (x - u) * (1 - x) :=
    mul_nonneg (sub_nonneg.mpr hu) (sub_nonneg.mpr hx)
  have hlen : 0 ≤ 1 - u := by linarith
  by_cases h0 : 1 - u = 0
  · have hxu : x = u := by linarith
    subst hxu
    exact min_le_left _ _
  · have hpos : 0 < 1 - u := lt_of_le_of_ne hlen (Ne.symm h0)
    have hcv : (1 - x) * (1 - (1 - t * u) ^ 2) + (x - u) * (1 - (1 - t) ^ 2) ≤
        (1 - u) * (1 - (1 - t * x) ^ 2) := by
      have hid : (1 - u) * (1 - (1 - t * x) ^ 2) -
          ((1 - x) * (1 - (1 - t * u) ^ 2) + (x - u) * (1 - (1 - t) ^ 2)) =
          (1 - u) * t ^ 2 * ((x - u) * (1 - x)) := by ring
      have hnn : 0 ≤ (1 - u) * t ^ 2 * ((x - u) * (1 - x)) :=
        mul_nonneg (mul_nonneg hlen (sq_nonneg t)) hprod
      linarith
    have hm1 : min (1 - (1 - t * u) ^ 2) (1 - (1 - t) ^ 2) ≤ 1 - (1 - t * u) ^ 2 :=
      min_le_left _ _
    have hm2 : min (1 - (1 - t * u) ^ 2) (1 - (1 - t) ^ 2) ≤ 1 - (1 - t) ^ 2 :=
      min_le_right _ _
    have hcomb : (1 - u) * min (1 - (1 - t * u) ^ 2) (1 - (1 - t) ^ 2) ≤
        (1 - x) * (1 - (1 - t * u) ^ 2) + (x - u) * (1 - (1 - t) ^ 2) := by
      have h1 := mul_le_mul_of_nonneg_left hm1 (sub_nonneg.mpr hx)
      have h2 := mul_le_mul_of_nonneg_left hm2 (sub_nonneg.mpr hu)
      linarith
    nlinarith

/-- One-level monotone effectiveness recurrence in homogeneous scalar form:
    taking `c = min(alpha * gain, 1 / (K + 1 - zeta))` yields effectiveness
    `e_new = min(alpha * (1 - zeta) * gain, (1 - zeta) / (K + 1 - zeta))`. -/
theorem monotone_effectiveness_scalar (alpha gain zeta K N s t : ℝ)
    (ha : 0 ≤ alpha) (hg : 0 ≤ gain)
    (_hz0 : 0 ≤ zeta) (hz1 : zeta < 1) (_hK : 0 ≤ K)
    (_hs0 : 0 ≤ s) (hs1 : s ≤ N)
    (ht : t ≤ zeta * s + K * (N - s)) :
    let c := min (alpha * gain) (1 / (K + 1 - zeta))
    let e_new := min (alpha * (1 - zeta) * gain) ((1 - zeta) / (K + 1 - zeta))
    (1 - c) * s + c * t ≤ (1 - e_new) * N := by
  intro c e_new
  have hden : 0 < K + 1 - zeta := by linarith
  have h1z : 0 < 1 - zeta := by linarith
  have hc0 : 0 ≤ c := le_min (mul_nonneg ha hg) (div_nonneg zero_le_one hden.le)
  have he_eq : e_new = (1 - zeta) * c := by
    dsimp only [e_new, c]
    rw [mul_min_of_nonneg _ _ h1z.le]
    congr 1 <;> ring
  have hc_le : c ≤ 1 / (K + 1 - zeta) := min_le_right _ _
  have hK_le : K * c ≤ 1 - e_new := by
    rw [he_eq]
    have hmul : c * (K + 1 - zeta) ≤ 1 := by
      have := (le_div_iff₀ hden).mp hc_le
      linarith
    linarith
  have hfirst : 1 - c + c * zeta = 1 - e_new := by
    rw [he_eq]; ring
  calc
    (1 - c) * s + c * t
      ≤ (1 - c) * s + c * (zeta * s + K * (N - s)) :=
        add_le_add le_rfl (mul_le_mul_of_nonneg_left ht hc0)
    _ = (1 - c + c * zeta) * s + (K * c) * (N - s) := by ring
    _ ≤ (1 - e_new) * s + (1 - e_new) * (N - s) := by
        rw [hfirst]
        exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hK_le (sub_nonneg.mpr hs1))
    _ = (1 - e_new) * N := by ring

/-- Arbitrary finite-dimensional quadratic-form version of the monotone
    one-level effectiveness lemma. -/
theorem monotone_effectiveness_matrix {ι : Type*} [Fintype ι]
    (E S2 T : Matrix ι ι ℝ) (alpha gain zeta K : ℝ)
    (ha : 0 ≤ alpha) (hg : 0 ≤ gain)
    (hz0 : 0 ≤ zeta) (hz1 : zeta < 1) (hK : 0 ≤ K)
    (hs : ∀ x, 0 ≤ quadForm S2 x ∧ quadForm S2 x ≤ vecNormSq x)
    (ht : ∀ x, quadForm T x ≤ zeta * quadForm S2 x + K * (vecNormSq x - quadForm S2 x))
    (hE : ∀ x,
      let c := min (alpha * gain) (1 / (K + 1 - zeta))
      quadForm E x ≤ (1 - c) * quadForm S2 x + c * quadForm T x) :
    let e_new := min (alpha * (1 - zeta) * gain) ((1 - zeta) / (K + 1 - zeta))
    ∀ x, quadForm E x ≤ (1 - e_new) * vecNormSq x := by
  intro e_new x
  exact le_trans (hE x) (monotone_effectiveness_scalar alpha gain zeta K
    (vecNormSq x) (quadForm S2 x) (quadForm T x)
    ha hg hz0 hz1 hK (hs x).1 (hs x).2 (ht x))

/-- Arbitrary finite-dimensional operator mechanization of Theorem 6 (monotone
    one-level effectiveness in A-inner-product dual form, without matrix square roots
    and without assuming `hE`): given SPD `A`, injective prolongation `P` with
    Galerkin operator `T = Pᵀ A P`, symmetric smoother `0 ≤ R ≤ A⁻¹`, and symmetric
    coarse operator `B_amp` whose damped copy `γ B_amp` lies in `[α g T⁻¹, T⁻¹]`,
    the two-level error `A⁻¹ - B_level = S (A⁻¹ - γ P B_amp Pᵀ) Sᵀ` satisfies the
    exact coarse comparison inequality `A⁻¹ - B_level ≤ (1 - c) S A⁻¹ Sᵀ + c S (A⁻¹ - P T⁻¹ Pᵀ) Sᵀ`
    for `c = min(α g, 1 / (K + 1 - ζ))`. Combining this with the weak approximation
    property yields the full two-sided Loewner effectiveness sandwich
    `e_new A⁻¹ ≤ B_level ≤ A⁻¹` with `e_new = min(α (1 - ζ) g, (1 - ζ) / (K + 1 - ζ))`. -/
theorem monotone_effectiveness_operator
    {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]
    (A R : Matrix ι ι ℝ) (P : Matrix ι κ ℝ) (B_amp : Matrix κ κ ℝ)
    (γ alpha gain zeta K : ℝ)
    (hA : A.PosDef)
    (hR_symm : Rᵀ = R)
    (hR0 : ∀ x, 0 ≤ quadForm R x)
    (hR_ub : ∀ x, quadForm R x ≤ quadForm A⁻¹ x)
    (hP : Function.Injective P.mulVec)
    (hBamp_symm : B_ampᵀ = B_amp)
    (hγ : 0 < γ) (ha : 0 ≤ alpha) (hg : 0 ≤ gain)
    (hz0 : 0 ≤ zeta) (hz1 : zeta < 1) (hK : 0 ≤ K)
    (hBamp_lb : ∀ y, (alpha * gain) * quadForm (Pᵀ * A * P)⁻¹ y ≤ γ * quadForm B_amp y)
    (hBamp_ub : ∀ y, γ * quadForm B_amp y ≤ quadForm (Pᵀ * A * P)⁻¹ y)
    (happrox : ∀ x,
      let S := (1 : Matrix ι ι ℝ) - R * A
      let T := Pᵀ * A * P
      quadForm (S * (A⁻¹ - P * T⁻¹ * Pᵀ) * Sᵀ) x ≤
        zeta * quadForm (S * A⁻¹ * Sᵀ) x +
        K * (quadForm A⁻¹ x - quadForm (S * A⁻¹ * Sᵀ) x)) :
    let T := Pᵀ * A * P
    let S := (1 : Matrix ι ι ℝ) - R * A
    let B_level := (2 : ℝ) • R - R * A * R + γ • (S * P * B_amp * Pᵀ * Sᵀ)
    let c := min (alpha * gain) (1 / (K + 1 - zeta))
    let e_new := min (alpha * (1 - zeta) * gain) ((1 - zeta) / (K + 1 - zeta))
    T.PosDef ∧
    B_levelᵀ = B_level ∧
    (∀ x, quadForm (A⁻¹ - B_level) x ≤
      (1 - c) * quadForm (S * A⁻¹ * Sᵀ) x +
      c * quadForm (S * (A⁻¹ - P * T⁻¹ * Pᵀ) * Sᵀ) x) ∧
    (∀ x, e_new * quadForm A⁻¹ x ≤ quadForm B_level x ∧ quadForm B_level x ≤ quadForm A⁻¹ x) := by
  intro T S B_level c e_new
  have hAs : Aᵀ = A := by
    simpa [Matrix.IsHermitian] using (Matrix.posDef_iff_dotProduct_mulVec.mp hA).1
  have hAdet : IsUnit A.det := isUnit_iff_ne_zero.mpr (ne_of_gt hA.det_pos)
  have hAR : A * A⁻¹ = 1 := Matrix.mul_nonsing_inv A hAdet
  have hAL : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul A hAdet
  have hAinvs : (A⁻¹)ᵀ = A⁻¹ := symm_inv_of_mul_eq_one A A⁻¹ hAs hAR hAL
  have hA0 : ∀ x, 0 ≤ quadForm A x := posDef_quadForm_nonneg A hA
  have hAinv0 : ∀ x, 0 ≤ quadForm A⁻¹ x := by
    intro x
    have hid : A⁻¹ = A⁻¹ * A * (A⁻¹)ᵀ := by rw [hAL, Matrix.one_mul, hAinvs]
    rw [hid, quadForm_conj_transpose]
    exact hA0 _
  have hTs : Tᵀ = T := by
    simp only [T, Matrix.transpose_mul, Matrix.transpose_transpose, hAs, Matrix.mul_assoc]
  have hT_pd : T.PosDef := by
    rw [Matrix.posDef_iff_dotProduct_mulVec]
    refine ⟨by simpa [Matrix.IsHermitian] using hTs, ?_⟩
    intro y hy
    have hPy : P.mulVec y ≠ 0 := fun h0 => hy (hP (by simpa using h0))
    have hquad : star y ⬝ᵥ T.mulVec y = star (P.mulVec y) ⬝ᵥ A.mulVec (P.mulVec y) := by
      simp only [T, star_trivial, ← quadForm_eq_dotProduct]
      exact quadForm_conj_transpose Pᵀ A y
    rw [hquad]
    exact (Matrix.posDef_iff_dotProduct_mulVec.mp hA).2 hPy
  have hTdet : IsUnit T.det := isUnit_iff_ne_zero.mpr (ne_of_gt hT_pd.det_pos)
  have hTR : T * T⁻¹ = 1 := Matrix.mul_nonsing_inv T hTdet
  have hTL : T⁻¹ * T = 1 := Matrix.nonsing_inv_mul T hTdet
  have hTinvs : (T⁻¹)ᵀ = T⁻¹ := symm_inv_of_mul_eq_one T T⁻¹ hTs hTR hTL
  have hTinv0 : ∀ y, 0 ≤ quadForm T⁻¹ y := by
    intro y
    have hid : T⁻¹ = T⁻¹ * T * (T⁻¹)ᵀ := by rw [hTL, Matrix.one_mul, hTinvs]
    rw [hid, quadForm_conj_transpose]
    exact posDef_quadForm_nonneg T hT_pd _
  have hgalerkin_proj : ∀ z, quadForm (P * T⁻¹ * Pᵀ) z ≤ quadForm A⁻¹ z := by
    intro z
    let P_proj := (1 : Matrix ι ι ℝ) - P * T⁻¹ * Pᵀ * A
    have hP_projT : P_projᵀ = (1 : Matrix ι ι ℝ) - A * P * T⁻¹ * Pᵀ := by
      simp only [P_proj, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_mul,
        Matrix.transpose_transpose, hAs, hTinvs, Matrix.mul_assoc]
    have hproj_id : A⁻¹ - P * T⁻¹ * Pᵀ = P_proj * A⁻¹ * P_projᵀ := by
      rw [hP_projT]
      simp only [P_proj]
      have hmid : (P * T⁻¹ * Pᵀ * A * A⁻¹) = P * T⁻¹ * Pᵀ := by
        rw [Matrix.mul_assoc (P * T⁻¹ * Pᵀ) A A⁻¹, hAR, Matrix.mul_one]
      have hPAP : (P * T⁻¹ * Pᵀ) * (A * P * T⁻¹ * Pᵀ) = P * T⁻¹ * Pᵀ := by
        calc
          (P * T⁻¹ * Pᵀ) * (A * P * T⁻¹ * Pᵀ)
            = P * T⁻¹ * (Pᵀ * A * P) * T⁻¹ * Pᵀ := by simp only [Matrix.mul_assoc]
          _ = P * (T⁻¹ * T) * T⁻¹ * Pᵀ := by simp only [T, Matrix.mul_assoc]
          _ = P * T⁻¹ * Pᵀ := by rw [hTL, Matrix.mul_one]
      have hA1 : A⁻¹ * (A * P * T⁻¹ * Pᵀ) = P * T⁻¹ * Pᵀ := by
        simp only [← Matrix.mul_assoc, hAL, Matrix.one_mul]
      rw [Matrix.sub_mul, Matrix.one_mul, hmid, Matrix.mul_sub, Matrix.mul_one,
        Matrix.sub_mul, hA1, hPAP]
      abel
    have hnn : 0 ≤ quadForm (A⁻¹ - P * T⁻¹ * Pᵀ) z := by
      rw [hproj_id, quadForm_conj_transpose]
      exact hAinv0 (P_projᵀ.mulVec z)
    rw [quadForm_sub] at hnn
    linarith
  have hBamp0 : ∀ y, 0 ≤ quadForm B_amp y := by
    intro y
    have h1 : 0 ≤ γ * quadForm B_amp y :=
      le_trans (mul_nonneg (mul_nonneg ha hg) (hTinv0 y)) (hBamp_lb y)
    nlinarith [hγ]
  have hvc := vcycle_upper_bound_of_galerkin_sandwich A R P T (γ • B_amp) 1
    hA hR_symm hR0 hR_ub hP hTs (by simp [Matrix.transpose_smul, hBamp_symm]) one_pos
    (fun y => by simp [T])
    (fun y => by rw [quadForm_smul]; exact mul_nonneg hγ.le (hBamp0 y))
    (fun y => by rw [quadForm_smul]; exact hBamp_ub y)
  have hBeq : (2 : ℝ) • R - R * A * R + (1 : ℝ) • (S * P * (γ • B_amp) * Pᵀ * Sᵀ) = B_level := by
    simp only [B_level, one_smul, Matrix.mul_smul, Matrix.smul_mul]
  have hB_symm : B_levelᵀ = B_level := by
    have h := hvc.2.2.1
    rw [hBeq] at h
    exact h
  have hB_ub : ∀ x, quadForm B_level x ≤ quadForm A⁻¹ x := by
    intro x
    have h := (hvc.2.2.2.2 x).2
    rw [hBeq] at h
    exact h
  have hST : Sᵀ = (1 : Matrix ι ι ℝ) - A * R := by
    simp only [S, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_mul, hAs, hR_symm]
  have hsm_err : A⁻¹ - ((2 : ℝ) • R - R * A * R) = S * A⁻¹ * Sᵀ := by
    rw [hST]
    simp only [S]
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc R A A⁻¹, hAR, Matrix.mul_one,
      Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul, ← Matrix.mul_assoc A⁻¹ A R, hAL,
      Matrix.one_mul, ← Matrix.mul_assoc R A R]
    ext i j
    simp [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have herr_id : A⁻¹ - B_level = S * (A⁻¹ - γ • (P * B_amp * Pᵀ)) * Sᵀ := by
    have hcoarse_eq : γ • (S * P * B_amp * Pᵀ * Sᵀ) = S * (γ • (P * B_amp * Pᵀ)) * Sᵀ := by
      simp only [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]
    simp only [B_level]
    rw [sub_add_eq_sub_sub, hsm_err, hcoarse_eq, ← Matrix.sub_mul, ← Matrix.mul_sub]
  have hs_bounds : ∀ x, 0 ≤ quadForm (S * A⁻¹ * Sᵀ) x ∧ quadForm (S * A⁻¹ * Sᵀ) x ≤ quadForm A⁻¹ x := by
    intro x
    constructor
    · rw [quadForm_conj_transpose]
      exact hAinv0 _
    · have hvc0 := vcycle_upper_bound_of_galerkin_sandwich A R P T (0 : Matrix κ κ ℝ) 1
        hA hR_symm hR0 hR_ub hP hTs (by simp) one_pos (fun y => by simp [T])
        (fun y => by simp [quadForm]) (fun y => by simp [quadForm]; exact hTinv0 y)
      have hsm_nn : quadForm R x ≤ quadForm ((2 : ℝ) • R - R * A * R) x := by
        have h := (hvc0.2.2.2.2 x).1
        simpa [Matrix.mul_zero, Matrix.zero_mul] using h
      have hsub := congr_arg (fun M => quadForm M x) hsm_err
      rw [quadForm_sub] at hsub
      linarith [hR0 x]
  have hE_step : ∀ x, quadForm (A⁻¹ - B_level) x ≤
      (1 - c) * quadForm (S * A⁻¹ * Sᵀ) x +
      c * quadForm (S * (A⁻¹ - P * T⁻¹ * Pᵀ) * Sᵀ) x := by
    intro x
    let z := Sᵀ.mulVec x
    have hc_le : c ≤ alpha * gain := min_le_left _ _
    have h1 : quadForm (A⁻¹ - B_level) x = quadForm A⁻¹ z - γ * quadForm B_amp (Pᵀ.mulVec z) := by
      rw [herr_id, quadForm_conj_transpose, quadForm_sub, quadForm_smul, quadForm_conj_transpose]
    have h2 : c * quadForm T⁻¹ (Pᵀ.mulVec z) ≤ γ * quadForm B_amp (Pᵀ.mulVec z) :=
      le_trans (mul_le_mul_of_nonneg_right hc_le (hTinv0 _)) (hBamp_lb _)
    have h3 : quadForm (S * A⁻¹ * Sᵀ) x = quadForm A⁻¹ z := quadForm_conj_transpose S A⁻¹ x
    have h4 : quadForm (S * (A⁻¹ - P * T⁻¹ * Pᵀ) * Sᵀ) x =
        quadForm A⁻¹ z - quadForm T⁻¹ (Pᵀ.mulVec z) := by
      rw [quadForm_conj_transpose, quadForm_sub, quadForm_conj_transpose]
    rw [h1, h3, h4]
    linarith
  refine ⟨hT_pd, hB_symm, hE_step, ?_⟩
  intro x
  have hsc := monotone_effectiveness_scalar alpha gain zeta K
    (quadForm A⁻¹ x) (quadForm (S * A⁻¹ * Sᵀ) x)
    (quadForm (S * (A⁻¹ - P * T⁻¹ * Pᵀ) * Sᵀ) x)
    ha hg hz0 hz1 hK (hs_bounds x).1 (hs_bounds x).2 (happrox x)
  have herr_le : quadForm (A⁻¹ - B_level) x ≤ (1 - e_new) * quadForm A⁻¹ x :=
    le_trans (hE_step x) hsc
  rw [quadForm_sub] at herr_le
  exact ⟨by linarith, hB_ub x⟩

/-- Work-convergence frontier: if `p ≤ β²` and `β * m < 1` (linear work), then
    the degree-`m` gain multiplier satisfies `p * m² < 1`. -/
theorem work_convergence_frontier (beta p m : ℝ)
    (hb : 0 < beta) (hm : 0 < m)
    (hp : p ≤ beta ^ 2) (hw : beta * m < 1) :
    p * m ^ 2 < 1 := by
  have h₁ : p * m ^ 2 ≤ beta ^ 2 * m ^ 2 := mul_le_mul_of_nonneg_right hp (sq_nonneg m)
  have h₂ : 0 ≤ beta * m := mul_nonneg hb.le hm.le
  have h₃ : (beta * m) ^ 2 < 1 := by
    nlinarith [mul_pos (sub_pos.mpr hw) (show 0 < 1 + beta * m by linarith)]
  nlinarith only [h₁, h₃]

/-- Periodic-schedule impossibility frontier: if `p ≤ β²`, no positive periodic
    degree schedule `m : Fin k → ℝ` (`k ≥ 1`) can simultaneously satisfy linear
    work (`∏ i, (beta * m i) < 1`) and uniform gain (`1 ≤ ∏ i, (p * (m i) ^ 2)`). -/
theorem work_convergence_frontier_periodic {k : ℕ} (_hk : 1 ≤ k)
    (beta p : ℝ) (m : Fin k → ℝ)
    (hb : 0 < beta) (hp0 : 0 ≤ p) (hp : p ≤ beta ^ 2)
    (hm : ∀ i : Fin k, 0 < m i)
    (hw : (∏ i : Fin k, (beta * m i)) < 1) :
    (∏ i : Fin k, (p * (m i) ^ 2)) < 1 := by
  have hw' : beta ^ k * (∏ i : Fin k, m i) < 1 := by
    simpa [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin] using hw
  have hpk : p ^ k ≤ (beta ^ k) ^ 2 := by
    calc
      p ^ k ≤ (beta ^ 2) ^ k := pow_le_pow_left₀ hp0 hp k
      _ = (beta ^ k) ^ 2 := by rw [← pow_mul, mul_comm, pow_mul]
  have hres := work_convergence_frontier (beta ^ k) (p ^ k) (∏ i : Fin k, m i)
    (pow_pos hb k) (Finset.prod_pos fun i _ => hm i) hpk hw'
  simpa [Finset.prod_mul_distrib, Finset.prod_const, Finset.prod_pow,
    Finset.card_univ, Fintype.card_fin] using hres

/-- Converse of the work-convergence frontier: whenever `0 < β < 1`, `0 < p < 1`,
    and `β² < p`, there exist a positive integer period `k ≥ 1` and an integer boost
    degree `m ≥ 2` (implementing the periodic integer schedule `(1, …, 1, m)` of
    period `k`) that simultaneously achieve strictly linear work (`β^k * m < 1`)
    and strictly super-unit multi-level gain (`1 < p^k * m²`). -/
theorem work_convergence_frontier_converse (beta p : ℝ)
    (hb0 : 0 < beta) (_hb1 : beta < 1)
    (hp0 : 0 < p) (hp1 : p < 1)
    (hbp : beta ^ 2 < p) :
    ∃ k m : ℕ, 1 ≤ k ∧ 2 ≤ m ∧ beta ^ k * (m : ℝ) < 1 ∧ 1 < p ^ k * (m : ℝ) ^ 2 := by
  classical
  let q := Real.sqrt p
  have hq0 : 0 < q := Real.sqrt_pos.mpr hp0
  have hq_sq : q ^ 2 = p := Real.sq_sqrt hp0.le
  have hq1 : q < 1 := by nlinarith [hq_sq, hp1, hq0]
  have hbq : beta < q := by nlinarith [hq_sq, hbp, hb0, hq0]
  let δ := q / beta - 1
  have hδ : 0 < δ := by
    have : 1 < q / beta := (one_lt_div hb0).mpr hbq
    linarith
  have hbern : ∀ n : ℕ, 1 + (n : ℝ) * δ ≤ (q / beta) ^ n := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have hstep : (q / beta) ^ (n + 1) = (q / beta) ^ n * (1 + δ) := by
        simp only [pow_succ, δ]; ring
      rw [hstep]
      have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
      have h1δ : 0 ≤ 1 + δ := by linarith
      have hmul := mul_le_mul_of_nonneg_right ih h1δ
      push_cast
      nlinarith [mul_nonneg hn0 (sq_nonneg δ)]
  obtain ⟨k0, hk0⟩ := exists_nat_gt (1 / δ)
  let k := k0 + 1
  have hk1 : 1 ≤ k := Nat.succ_le_succ (Nat.zero_le k0)
  have hk_gt : 1 < (k : ℝ) * δ := by
    have hk_cast : 1 / δ < (k : ℝ) := by
      have : (k0 : ℝ) < (k : ℝ) := by simp [k]
      linarith
    have := (div_lt_iff₀ hδ).mp hk_cast
    linarith
  have hratio : 2 < (q / beta) ^ k := by
    have hb := hbern k
    linarith
  have hqk_pos : 0 < q ^ k := pow_pos hq0 k
  have hqk_lt1 : q ^ k < 1 := pow_lt_one₀ hq0.le hq1 (by omega)
  have hbk_pos : 0 < beta ^ k := pow_pos hb0 k
  have hgap : 1 / q ^ k + 1 < 1 / beta ^ k := by
    have hpow_div : (q / beta) ^ k = q ^ k / beta ^ k := div_pow q beta k
    have hq_gt_2b : 2 * beta ^ k < q ^ k := by
      have := (lt_div_iff₀ hbk_pos).mp (hratio.trans_eq hpow_div)
      linarith
    have hdiff : 1 / beta ^ k - 1 / q ^ k = (q ^ k - beta ^ k) / (beta ^ k * q ^ k) := by
      field_simp [ne_of_gt hbk_pos, ne_of_gt hqk_pos]
    have hden_pos : 0 < beta ^ k * q ^ k := mul_pos hbk_pos hqk_pos
    have hprod_lt : beta ^ k * q ^ k < beta ^ k := mul_lt_of_lt_one_right hbk_pos hqk_lt1
    have hgt1 : 1 < (q ^ k - beta ^ k) / (beta ^ k * q ^ k) := by
      rw [one_lt_div hden_pos]
      linarith
    linarith
  have hex : ∃ n : ℕ, 1 / q ^ k < (n : ℝ) := exists_nat_gt (1 / q ^ k)
  let m := Nat.find hex
  have hm_gt : 1 / q ^ k < (m : ℝ) := Nat.find_spec hex
  have hm_ne : m ≠ 0 := by
    intro hm0
    have hneg : 1 / q ^ k < (0 : ℝ) := by simpa [hm0] using hm_gt
    have hpos : 0 < 1 / q ^ k := div_pos one_pos hqk_pos
    linarith
  obtain ⟨m', hm_eq⟩ := Nat.exists_eq_succ_of_ne_zero hm_ne
  have hm'_lt : m' < m := by omega
  have hm'_le : (m' : ℝ) ≤ 1 / q ^ k :=
    le_of_not_gt (Nat.find_min hex hm'_lt)
  have hm_lt : (m : ℝ) < 1 / beta ^ k := by
    have hcast : (m : ℝ) = (m' : ℝ) + 1 := by simp [hm_eq]
    linarith
  have hwork : beta ^ k * (m : ℝ) < 1 := by
    have := (lt_div_iff₀ hbk_pos).mp hm_lt
    linarith
  have hqm_gt1 : 1 < q ^ k * (m : ℝ) := by
    have := (div_lt_iff₀ hqk_pos).mp hm_gt
    linarith
  have hconv : 1 < p ^ k * (m : ℝ) ^ 2 := by
    have hp_eq : p ^ k * (m : ℝ) ^ 2 = (q ^ k * (m : ℝ)) ^ 2 := by
      rw [← hq_sq, mul_pow, ← pow_mul, mul_comm 2 k, pow_mul]
    rw [hp_eq]
    nlinarith
  have hm2 : 2 ≤ m := by
    by_contra! hm_lt2
    have hm_le1 : (m : ℝ) ≤ 1 := by
      have : m ≤ 1 := by omega
      exact_mod_cast this
    have hpk_lt1 : p ^ k < 1 := pow_lt_one₀ hp0.le hp1 (by omega)
    have hm_nn : (0 : ℝ) ≤ m := Nat.cast_nonneg m
    have hm_sq_le1 : (m : ℝ) ^ 2 ≤ 1 := by nlinarith
    have : p ^ k * (m : ℝ) ^ 2 < 1 := by nlinarith [pow_pos hp0 k]
    linarith
  exact ⟨k, m, hk1, hm2, hwork, hconv⟩

-- ============================================================================
-- Section 4: Four-Phase Grounded SDDM Certificates (κ ≤ 17 and κ ≤ 9)
-- ============================================================================

theorem four_phase_sddm_budgets :
    let eps : ℝ := 1 / 1024
    let tau : ℝ := 1 / 8
    let eta : ℝ := 1 / 14
    let zeta : ℝ := (1 + 1 / tau) * eps
    let K : ℝ := (1 + tau) / (2 * eta)
    let cap : ℝ := (1 - zeta) / (K + 1 - zeta)
    let alpha : ℝ := (7 / 9) * (127 / 129) * (63 / 64) / (1 + eps)
    (7 : ℝ) * (3 / 4) ^ 32 < eps ∧
    zeta = 9 / 1024 ∧
    K = 63 / 8 ∧
    cap = 145 / 1297 ∧
    (1 / 9 : ℝ) < cap ∧
    (11 / 15 : ℝ) < alpha * (1 - zeta) ∧
    eta ≤ (3 / 40 : ℝ) ∧
    (2 : ℝ) * (13 / 16) ^ 4 < 1 := by
  norm_num

/-- Edge-pruning retention certificate (Theorem 8): deleting weak sparsifier edges
    below `a_i / (512 * (n_i + 1))` tightens the lower coarse sandwich by `511 / 512`
    while keeping `α_pruned * (1 - ζ) > 11 / 15`, so both `κ ≤ 17` and `κ ≤ 9`
    four-phase certificates remain valid with `γ = l_pruned`. -/
theorem four_phase_pruned_sddm_retention :
    let eps : ℝ := 1 / 1024
    let zeta : ℝ := 9 / 1024
    let alpha_pruned : ℝ := (511 / 512) * (7 / 9) * (127 / 129) * (63 / 64) / (1 + eps)
    (11 / 15 : ℝ) < alpha_pruned * (1 - zeta) := by
  norm_num

/-- Exact root condition-number certificate `κ(B₀ A₀) ≤ 17` for the
    four-phase grounded SDDM schedule at Richardson scale `t = 2560/1307`,
    plus the negative control showing that `t = 2560/1307` cannot certify `κ ≤ 9`. -/
theorem four_phase_kappa_17_certificate :
    let p : ℝ := 11 / 15
    let cap : ℝ := 145 / 1297
    let x : ℝ := 1 / 17
    let u : ℝ := 27 / 1280
    let t : ℝ := 2560 / 1307
    t = 2 / (1 + u) ∧
    x ≤ cap ∧
    u < p ^ 3 * x ∧
    x ≤ p * (1 - (1 - t * (p ^ 3 * x)) ^ 2) ∧
    x ≤ p * (1 - (1 - t) ^ 2) ∧
    p * (1 - (1 - t) ^ 2) < (1 / 9 : ℝ) := by
  norm_num

/-- Exact root condition-number certificate `κ(B₀ A₀) ≤ 9` for the same
    four-phase grounded SDDM schedule at retuned scale `t = 30375/15853`. -/
theorem four_phase_kappa_9_certificate :
    let p : ℝ := 11 / 15
    let cap : ℝ := 145 / 1297
    let x : ℝ := 1 / 9
    let u : ℝ := p ^ 3 * x
    let t : ℝ := 30375 / 15853
    u = 1331 / 30375 ∧
    t = 2 / (1 + u) ∧
    x ≤ cap ∧
    x ≤ p * (1 - (1 - t * u) ^ 2) ∧
    x ≤ p * (1 - (1 - t) ^ 2) := by
  norm_num

/-- Four-phase effectiveness target at hierarchy level `lvl` (`lvl % 4 = 0` is boosted). -/
def fourPhaseTarget (p x : ℝ) (lvl : ℕ) : ℝ :=
  match lvl % 4 with
  | 0 => x
  | 1 => p ^ 3 * x
  | 2 => p ^ 2 * x
  | _ => p * x

/-- One-level monotone four-phase recurrence step from child effectiveness `e_child`
    at level `lvl + 1` to parent effectiveness at level `lvl`. -/
def fourPhaseStep (p cap t : ℝ) (lvl : ℕ) (e_child : ℝ) : ℝ :=
  if lvl % 4 = 0 then
    min (p * min (1 - (1 - t * e_child) ^ 2) (1 - (1 - t) ^ 2)) cap
  else
    min (p * e_child) cap

/-- Bottom-up effectiveness produced by `D` levels of the four-phase monotone
    recurrence starting from exact terminal solve effectiveness `1` at level `lvl + D`. -/
def fourPhaseEff (p cap t : ℝ) : ℕ → ℕ → ℝ
  | _, 0 => 1
  | lvl, h + 1 => fourPhaseStep p cap t lvl (fourPhaseEff p cap t (lvl + 1) h)

/-- General arbitrary-depth induction for the four-phase monotone effectiveness
    recurrence: from exact terminal effectiveness `1`, every level `lvl` at every
    remaining height `h : ℕ` satisfies `fourPhaseTarget p x lvl ≤ fourPhaseEff p cap t lvl h ≤ 1`. -/
theorem four_phase_recurrence_induction (p cap t x : ℝ)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hx0 : 0 ≤ x)
    (hx_cap : x ≤ cap) (hcap1 : cap ≤ 1)
    (hboost_lo : x ≤ p * (1 - (1 - t * (p ^ 3 * x)) ^ 2))
    (hboost_hi : x ≤ p * (1 - (1 - t) ^ 2)) :
    ∀ h lvl : ℕ,
      fourPhaseTarget p x lvl ≤ fourPhaseEff p cap t lvl h ∧
      fourPhaseEff p cap t lvl h ≤ 1 := by
  have hpx_le_x : p * x ≤ x := by nlinarith
  have hp2_le_1 : p ^ 2 ≤ 1 := by nlinarith
  have hp3_le_1 : p ^ 3 ≤ 1 := by nlinarith [pow_nonneg hp0 2]
  have hp2x_le_x : p ^ 2 * x ≤ x := by nlinarith
  have hp3x_le_x : p ^ 3 * x ≤ x := by nlinarith
  have ht_le_x : ∀ lvl, fourPhaseTarget p x lvl ≤ x := by
    intro lvl
    unfold fourPhaseTarget
    have hmod : lvl % 4 = 0 ∨ lvl % 4 = 1 ∨ lvl % 4 = 2 ∨ lvl % 4 = 3 := by omega
    rcases hmod with h0 | h1 | h2 | h3
    · simp [h0]
    · simp [h1]; linarith
    · simp [h2]; linarith
    · simp [h3]; linarith
  intro h
  induction h with
  | zero =>
    intro lvl
    simp only [fourPhaseEff]
    exact ⟨le_trans (ht_le_x lvl) (le_trans hx_cap hcap1), le_rfl⟩
  | succ h ih =>
    intro lvl
    obtain ⟨hchild_lo, hchild_hi⟩ := ih (lvl + 1)
    simp only [fourPhaseEff, fourPhaseStep]
    have hmod : lvl % 4 = 0 ∨ lvl % 4 = 1 ∨ lvl % 4 = 2 ∨ lvl % 4 = 3 := by omega
    rcases hmod with h0 | h1 | h2 | h3
    · have hnext : (lvl + 1) % 4 = 1 := by omega
      have hch : p ^ 3 * x ≤ fourPhaseEff p cap t (lvl + 1) h := by
        simpa [fourPhaseTarget, hnext] using hchild_lo
      have hconc := two_call_concave_endpoint_min t (p ^ 3 * x) (fourPhaseEff p cap t (lvl + 1) h)
        hch hchild_hi
      have hmin_ge : min (1 - (1 - t * (p ^ 3 * x)) ^ 2) (1 - (1 - t) ^ 2) ≤
          min (1 - (1 - t * fourPhaseEff p cap t (lvl + 1) h) ^ 2) (1 - (1 - t) ^ 2) :=
        le_min hconc (min_le_right _ _)
      have hmul : p * min (1 - (1 - t * (p ^ 3 * x)) ^ 2) (1 - (1 - t) ^ 2) ≤
          p * min (1 - (1 - t * fourPhaseEff p cap t (lvl + 1) h) ^ 2) (1 - (1 - t) ^ 2) :=
        mul_le_mul_of_nonneg_left hmin_ge hp0
      have hx_le_pm : x ≤ p * min (1 - (1 - t * (p ^ 3 * x)) ^ 2) (1 - (1 - t) ^ 2) := by
        rw [mul_min_of_nonneg _ _ hp0]
        exact le_min hboost_lo hboost_hi
      simp only [h0, ↓reduceIte, fourPhaseTarget]
      exact ⟨le_min (le_trans hx_le_pm hmul) hx_cap, le_trans (min_le_right _ _) hcap1⟩
    · have hnext : (lvl + 1) % 4 = 2 := by omega
      have hch : p ^ 2 * x ≤ fourPhaseEff p cap t (lvl + 1) h := by
        simpa [fourPhaseTarget, hnext] using hchild_lo
      have hmul : p ^ 3 * x ≤ p * fourPhaseEff p cap t (lvl + 1) h := by
        have := mul_le_mul_of_nonneg_left hch hp0
        linarith [show p * (p ^ 2 * x) = p ^ 3 * x by ring]
      simp only [h1, Nat.one_ne_zero, ↓reduceIte, fourPhaseTarget]
      exact ⟨le_min hmul (le_trans hp3x_le_x hx_cap), le_trans (min_le_right _ _) hcap1⟩
    · have hnext : (lvl + 1) % 4 = 3 := by omega
      have hch : p * x ≤ fourPhaseEff p cap t (lvl + 1) h := by
        simpa [fourPhaseTarget, hnext] using hchild_lo
      have hmul : p ^ 2 * x ≤ p * fourPhaseEff p cap t (lvl + 1) h := by
        have := mul_le_mul_of_nonneg_left hch hp0
        linarith [show p * (p * x) = p ^ 2 * x by ring]
      simp only [h2, OfNat.ofNat_ne_zero, ↓reduceIte, fourPhaseTarget]
      exact ⟨le_min hmul (le_trans hp2x_le_x hx_cap), le_trans (min_le_right _ _) hcap1⟩
    · have hnext : (lvl + 1) % 4 = 0 := by omega
      have hch : x ≤ fourPhaseEff p cap t (lvl + 1) h := by
        simpa [fourPhaseTarget, hnext] using hchild_lo
      have hmul : p * x ≤ p * fourPhaseEff p cap t (lvl + 1) h :=
        mul_le_mul_of_nonneg_left hch hp0
      simp only [h3, OfNat.ofNat_ne_zero, ↓reduceIte, fourPhaseTarget]
      exact ⟨le_min hmul (le_trans hpx_le_x hx_cap), le_trans (min_le_right _ _) hcap1⟩

/-- Arbitrary-depth four-phase induction for BOTH grounded SDDM schedules:
    for every hierarchy depth `D : ℕ` (starting from exact terminal effectiveness `1`),
    every level `lvl` maintains its phase target and the root (`lvl = 0`) achieves
    effectiveness `≥ 1/17` (`κ ≤ 17` at `t = 2560/1307`) and `≥ 1/9` (`κ ≤ 9` at
    `t = 30375/15853`), bounded above by `1`. -/
theorem four_phase_arbitrary_depth_induction (D : ℕ) :
    let p : ℝ := 11 / 15
    let cap : ℝ := 145 / 1297
    let t17 : ℝ := 2560 / 1307
    let t9 : ℝ := 30375 / 15853
    (∀ lvl, fourPhaseTarget p (1 / 17) lvl ≤ fourPhaseEff p cap t17 lvl D ∧
            fourPhaseEff p cap t17 lvl D ≤ 1) ∧
    ((1 / 17 : ℝ) ≤ fourPhaseEff p cap t17 0 D ∧ fourPhaseEff p cap t17 0 D ≤ 1) ∧
    (∀ lvl, fourPhaseTarget p (1 / 9) lvl ≤ fourPhaseEff p cap t9 lvl D ∧
            fourPhaseEff p cap t9 lvl D ≤ 1) ∧
    ((1 / 9 : ℝ) ≤ fourPhaseEff p cap t9 0 D ∧ fourPhaseEff p cap t9 0 D ≤ 1) := by
  intro p cap t17 t9
  have h17_cert := four_phase_kappa_17_certificate
  have h9_cert := four_phase_kappa_9_certificate
  have h17 : ∀ lvl, fourPhaseTarget p (1 / 17) lvl ≤ fourPhaseEff p cap t17 lvl D ∧
      fourPhaseEff p cap t17 lvl D ≤ 1 := fun lvl =>
    four_phase_recurrence_induction p cap t17 (1 / 17)
      (by norm_num) (by norm_num) (by norm_num)
      h17_cert.2.1 (by norm_num)
      h17_cert.2.2.2.1 h17_cert.2.2.2.2.1 D lvl
  have h9 : ∀ lvl, fourPhaseTarget p (1 / 9) lvl ≤ fourPhaseEff p cap t9 lvl D ∧
      fourPhaseEff p cap t9 lvl D ≤ 1 := fun lvl =>
    four_phase_recurrence_induction p cap t9 (1 / 9)
      (by norm_num) (by norm_num) (by norm_num)
      h9_cert.2.2.1 (by norm_num)
      h9_cert.2.2.2.1 h9_cert.2.2.2.2 D lvl
  refine ⟨h17, ?_, h9, ?_⟩
  · simpa [fourPhaseTarget] using h17 0
  · simpa [fourPhaseTarget] using h9 0

/-- Singleton fallback termination for decoded-block certification: rounding
    `1 / √d` downward to 4 significand bits guarantees `d * s² ∈ [49/64, 1] ⊂ [3/4, 5/4]`. -/
theorem singleton_block_truncation_bounds (d s : ℝ) (hd : 0 < d)
    (hlo : (7 / 8 : ℝ) / Real.sqrt d ≤ s) (hhi : s ≤ 1 / Real.sqrt d) :
    (3 / 4 : ℝ) < d * s ^ 2 ∧ (49 / 64 : ℝ) ≤ d * s ^ 2 ∧ d * s ^ 2 ≤ 1 := by
  have hsq : Real.sqrt d ^ 2 = d := Real.sq_sqrt hd.le
  have hrt : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd
  have hs0 : 0 ≤ s := le_trans (div_nonneg (by norm_num) hrt.le) hlo
  have hlo' : (7 / 8 : ℝ) ≤ Real.sqrt d * s := by
    have := (div_le_iff₀ hrt).mp hlo
    linarith
  have hhi' : Real.sqrt d * s ≤ 1 := by
    have := (le_div_iff₀ hrt).mp hhi
    linarith
  have hds : d * s ^ 2 = (Real.sqrt d * s) ^ 2 := by
    rw [mul_pow, hsq]
  rw [hds]
  have h1 : (49 / 64 : ℝ) ≤ (Real.sqrt d * s) ^ 2 := by nlinarith
  have h2 : (Real.sqrt d * s) ^ 2 ≤ 1 := by nlinarith [mul_nonneg hrt.le hs0]
  exact ⟨by linarith, h1, h2⟩

/-- Two-vertex grounded counterexample showing that an exact inverse factor
    (`G * A * Gᵀ = I`) can still be arbitrarily overdamped by the entrywise
    absolute-value bound `|G| |A| |G|ᵀ` as grounding weight `s → 0⁺`. -/
theorem absolute_guard_overdamping_witness (s : ℝ) (hs : 0 < s) :
    let A : Matrix (Fin 2) (Fin 2) ℝ := !![(1 : ℝ), -1; -1, 1 + s]
    let G : Matrix (Fin 2) (Fin 2) ℝ := !![(1 : ℝ), 0; 1 / Real.sqrt s, 1 / Real.sqrt s]
    let absA : Matrix (Fin 2) (Fin 2) ℝ := !![(1 : ℝ), 1; 1, 1 + s]
    let absG : Matrix (Fin 2) (Fin 2) ℝ := G
    G * A * Gᵀ = (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧
    (absG * absA * absGᵀ) 1 1 = (4 + s) / s := by
  intro A G absA absG
  have hsq : Real.sqrt s ^ 2 = s := Real.sq_sqrt hs.le
  have hrt : Real.sqrt s ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hs)
  have hsne : s ≠ 0 := ne_of_gt hs
  constructor
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [G, A, Matrix.mul_apply, Fin.sum_univ_two]
    · field_simp [hrt]
      nlinarith [hsq]
  · simp [absG, absA, G, Matrix.mul_apply, Fin.sum_univ_two]
    field_simp [hrt, hsne]
    nlinarith [hsq]

end BitFSAI
