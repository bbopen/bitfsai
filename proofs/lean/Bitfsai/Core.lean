import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

namespace BitFSAI

open Matrix Finset

def IsLowerTriangular {n : ℕ} (G : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j : Fin n, i < j → G i j = 0

theorem lowerTriangular_posDiag_injective {n : ℕ}
    (G : Matrix (Fin n) (Fin n) ℝ)
    (hL : IsLowerTriangular G)
    (hD : ∀ i : Fin n, 0 < G i i)
    (x : Fin n → ℝ) (hx : G.mulVec x = 0) : x = 0 := by
  have hmain : ∀ k : ℕ, ∀ i : Fin n, i.val = k → x i = 0 := by
    intro k
    induction k using Nat.strong_induction_on with
    | ind k ih =>
      intro i hik
      have hrow : (G.mulVec x) i = 0 := by
        rw [hx]; rfl
      simp only [mulVec, dotProduct] at hrow
      have hsplit : ∑ j : Fin n, G i j * x j = G i i * x i := by
        rw [← Finset.add_sum_erase (s := Finset.univ) (f := fun j => G i j * x j)
          (Finset.mem_univ i)]
        have hrest : ∑ j ∈ Finset.univ.erase i, G i j * x j = 0 := by
          apply Finset.sum_eq_zero
          intro j hj
          have hne : j ≠ i := (Finset.mem_erase.mp hj).1
          rcases lt_or_gt_of_ne hne with hlt | hgt
          · have hltv : j.val < k := hik ▸ hlt
            rw [ih j.val hltv j rfl, mul_zero]
          · rw [hL i j hgt, zero_mul]
        linarith
      rw [hsplit] at hrow
      exact (mul_eq_zero.mp hrow).resolve_left (ne_of_gt (hD i))
  funext i
  exact hmain i.val i rfl

theorem gram_posDef_of_lowerTriangular_posDiag {n : ℕ}
    (G : Matrix (Fin n) (Fin n) ℝ)
    (hL : IsLowerTriangular G)
    (hD : ∀ i : Fin n, 0 < G i i) :
    (Gᵀ * G).PosDef := by
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_, ?_⟩
  · simp [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial]
  · intro x hx
    have hGx_ne : G.mulVec x ≠ 0 := fun h0 =>
      hx (lowerTriangular_posDiag_injective G hL hD x h0)
    have hquad : star x ⬝ᵥ (Gᵀ * G).mulVec x = ∑ i : Fin n, (G.mulVec x i) ^ 2 := by
      rw [star_trivial, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec,
        Matrix.vecMul_transpose]
      simp only [dotProduct, sq]
    rw [hquad]
    obtain ⟨i, hi⟩ : ∃ i : Fin n, G.mulVec x i ≠ 0 := by
      by_contra h
      push Not at h
      exact hGx_ne (funext h)
    exact Finset.sum_pos' (fun j _ => sq_nonneg (G.mulVec x j))
      ⟨i, Finset.mem_univ i, sq_pos_of_ne_zero hi⟩

def diagSplitQuant {n : ℕ} (G : Matrix (Fin n) (Fin n) ℝ)
    (q : Fin n → Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i = j then G i i else if j < i then q i j else 0

theorem diagonal_split_preserves_posDef {n : ℕ}
    (G : Matrix (Fin n) (Fin n) ℝ)
    (hD : ∀ i : Fin n, 0 < G i i)
    (q : Fin n → Fin n → ℝ) :
    ((diagSplitQuant G q)ᵀ * diagSplitQuant G q).PosDef := by
  apply gram_posDef_of_lowerTriangular_posDiag
  · intro i j hij
    simp only [diagSplitQuant, ne_of_lt hij, ↓reduceIte, not_lt.mpr (le_of_lt hij)]
  · intro i
    simp only [diagSplitQuant, ↓reduceIte]
    exact hD i

theorem lowerTriangular_zero_diag_not_posDef {n : ℕ}
    (G : Matrix (Fin n) (Fin n) ℝ)
    (hL : IsLowerTriangular G)
    (k : Fin n) (hk : G k k = 0) :
    ¬ (Gᵀ * G).PosDef := by
  intro hpd
  have hdetG : G.det = 0 := by
    rw [Matrix.det_of_isLowerTriangular G (by
      intro i j hij
      exact hL i j hij)]
    exact Finset.prod_eq_zero (Finset.mem_univ k) hk
  have hdetM : (Gᵀ * G).det = 0 := by
    rw [Matrix.det_mul, Matrix.det_transpose, hdetG, mul_zero]
  exact ne_of_gt hpd.det_pos hdetM

theorem conditional_spectral_distortion_bound {n : ℕ}
    (u e : EuclideanSpace ℝ (Fin n)) (η : ℝ)
    (hη0 : 0 ≤ η) (hη1 : η < 1)
    (he : ‖e‖ ≤ η * ‖u‖) :
    (1 - η) ^ 2 * ‖u‖ ^ 2 ≤ ‖u + e‖ ^ 2 ∧ ‖u + e‖ ^ 2 ≤ (1 + η) ^ 2 * ‖u‖ ^ 2 := by
  have hu : 0 ≤ ‖u‖ := norm_nonneg u
  have h1 : 0 ≤ 1 - η := sub_nonneg.mpr (le_of_lt hη1)
  have h2 : 0 ≤ 1 + η := by linarith
  constructor
  · have hlow : (1 - η) * ‖u‖ ≤ ‖u + e‖ := by
      have htri : ‖u‖ - ‖e‖ ≤ ‖u + e‖ := by
        have := norm_sub_norm_le u (-e)
        simpa [sub_neg_eq_add, norm_neg] using this
      nlinarith
    have hsq := mul_self_le_mul_self (mul_nonneg h1 hu) hlow
    nlinarith
  · have hhi : ‖u + e‖ ≤ (1 + η) * ‖u‖ := by
      have htri : ‖u + e‖ ≤ ‖u‖ + ‖e‖ := norm_add_le u e
      nlinarith
    have hsq := mul_self_le_mul_self (norm_nonneg (u + e)) hhi
    nlinarith

theorem eta_five_over_28_bound :
    ((1 + (5 : ℝ) / 28) / (1 - 5 / 28)) ^ 2 < (206 : ℝ) / 100 := by
  norm_num

noncomputable def l2Norm {n : ℕ} (v : Fin n → ℝ) : ℝ :=
  Real.sqrt (∑ i : Fin n, (v i) ^ 2)

noncomputable def rayleighQuotient {n : ℕ}
    (G A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  l2Norm (G.mulVec x) ^ 2 / (x ⬝ᵥ A.mulVec x)

private lemma l2Norm_sq {n : ℕ} (v : Fin n → ℝ) :
    l2Norm v ^ 2 = ∑ i : Fin n, (v i) ^ 2 :=
  Real.sq_sqrt (Finset.sum_nonneg fun i _ => sq_nonneg (v i))

private lemma l2Norm_add_sq_bounds {n : ℕ}
    (u e : Fin n → ℝ) (η : ℝ)
    (hη0 : 0 ≤ η) (hη1 : η < 1)
    (he : l2Norm e ≤ η * l2Norm u) :
    (1 - η) ^ 2 * l2Norm u ^ 2 ≤ l2Norm (u + e) ^ 2 ∧
    l2Norm (u + e) ^ 2 ≤ (1 + η) ^ 2 * l2Norm u ^ 2 := by
  let U := ∑ i : Fin n, (u i) ^ 2
  let E2 := ∑ i : Fin n, (e i) ^ 2
  let IP := ∑ i : Fin n, u i * e i
  have hU_eq : l2Norm u ^ 2 = U := l2Norm_sq u
  have hE_eq : l2Norm e ^ 2 = E2 := l2Norm_sq e
  have hsum_eq : l2Norm (u + e) ^ 2 = U + 2 * IP + E2 := by
    rw [l2Norm_sq]
    simp only [Pi.add_apply, U, E2, IP]
    have hexp : ∀ i : Fin n, (u i + e i) ^ 2 = (u i) ^ 2 + 2 * (u i * e i) + (e i) ^ 2 := by
      intro i; ring
    simp_rw [hexp, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hU_eq, hsum_eq]
  have hU_nn : 0 ≤ U := Finset.sum_nonneg fun i _ => sq_nonneg (u i)
  have h1 : 0 ≤ 1 - η := sub_nonneg.mpr hη1.le
  by_cases hlu0 : l2Norm u = 0
  · have hU0 : U = 0 := by rw [← hU_eq, hlu0, sq]; ring
    have hle0 : l2Norm e = 0 :=
      le_antisymm (by simpa [hlu0] using he) (Real.sqrt_nonneg _)
    have hE0 : E2 = 0 := by rw [← hE_eq, hle0, sq]; ring
    have he_i : ∀ i : Fin n, e i = 0 := by
      intro i
      have hsq0 : (e i) ^ 2 ≤ 0 := by
        rw [← hE0]
        exact Finset.single_le_sum (fun j _ => sq_nonneg (e j)) (Finset.mem_univ i)
      exact sq_eq_zero_iff.mp (le_antisymm hsq0 (sq_nonneg (e i)))
    have hIP0 : IP = 0 := Finset.sum_eq_zero fun i _ => by simp [he_i i]
    simp [hU0, hE0, hIP0]
  · have hlu_pos : 0 < l2Norm u :=
      lt_of_le_of_ne (Real.sqrt_nonneg _) (Ne.symm hlu0)
    let r := l2Norm e / l2Norm u
    have hr0 : 0 ≤ r := div_nonneg (Real.sqrt_nonneg _) hlu_pos.le
    have hrη : r ≤ η := (div_le_iff₀ hlu_pos).mpr (by linarith)
    have hE_rU : E2 = r ^ 2 * U := by
      rw [← hE_eq, ← hU_eq]
      simp only [r, div_pow]
      exact (div_mul_cancel₀ _ (ne_of_gt (sq_pos_of_pos hlu_pos))).symm
    by_cases hr_zero : r = 0
    · have hE0 : E2 = 0 := by rw [hE_rU, hr_zero]; ring
      have he_i : ∀ i : Fin n, e i = 0 := by
        intro i
        have hsq0 : (e i) ^ 2 ≤ 0 := by
          rw [← hE0]
          exact Finset.single_le_sum (fun j _ => sq_nonneg (e j)) (Finset.mem_univ i)
        exact sq_eq_zero_iff.mp (le_antisymm hsq0 (sq_nonneg (e i)))
      have hIP0 : IP = 0 := Finset.sum_eq_zero fun i _ => by simp [he_i i]
      rw [hE0, hIP0]
      constructor <;> nlinarith [mul_nonneg hU_nn (sq_nonneg η), mul_nonneg hU_nn hη0]
    · have hr_pos : 0 < r := lt_of_le_of_ne hr0 (Ne.symm hr_zero)
      have hcs_lo : 0 ≤ ∑ i : Fin n, (r * u i + e i) ^ 2 :=
        Finset.sum_nonneg fun i _ => sq_nonneg _
      have hcs_hi : 0 ≤ ∑ i : Fin n, (r * u i - e i) ^ 2 :=
        Finset.sum_nonneg fun i _ => sq_nonneg _
      have hlo_expand : ∑ i : Fin n, (r * u i + e i) ^ 2 = r ^ 2 * U + 2 * r * IP + E2 := by
        have hexp : ∀ i : Fin n, (r * u i + e i) ^ 2 =
            r ^ 2 * (u i) ^ 2 + 2 * r * (u i * e i) + (e i) ^ 2 := fun i => by ring
        simp_rw [hexp, Finset.sum_add_distrib, ← Finset.mul_sum]
        rfl
      have hhi_expand : ∑ i : Fin n, (r * u i - e i) ^ 2 = r ^ 2 * U - 2 * r * IP + E2 := by
        have hexp : ∀ i : Fin n, (r * u i - e i) ^ 2 =
            r ^ 2 * (u i) ^ 2 - 2 * r * (u i * e i) + (e i) ^ 2 := fun i => by ring
        simp_rw [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
        rfl
      rw [hlo_expand, hE_rU] at hcs_lo
      rw [hhi_expand, hE_rU] at hcs_hi
      have hIP_lo : -r * U ≤ IP := by nlinarith
      have hIP_hi : IP ≤ r * U := by nlinarith
      have h1r : 1 - η ≤ 1 - r := by linarith
      have hsq_lo : (1 - η) ^ 2 ≤ (1 - r) ^ 2 := by nlinarith
      have hsq_hi : (1 + r) ^ 2 ≤ (1 + η) ^ 2 := by nlinarith
      rw [hE_rU]
      constructor
      · have : (1 - r) ^ 2 * U ≤ U + 2 * IP + r ^ 2 * U := by nlinarith
        nlinarith [mul_le_mul_of_nonneg_right hsq_lo hU_nn]
      · have : U + 2 * IP + r ^ 2 * U ≤ (1 + r) ^ 2 * U := by nlinarith
        nlinarith [mul_le_mul_of_nonneg_right hsq_hi hU_nn]

/-- Arbitrary-`n` Rayleigh-quotient condition-distortion corollary of Theorem 2:
    for any SPD matrix `A`, invertible factor `G`, perturbation `E` with
    `‖E u‖₂ ≤ η ‖u‖₂` (`0 ≤ η < 1`), and `G_q = (I + E) G`, both Rayleigh
    denominators at `y ≠ 0` are strictly positive and the generalized Rayleigh-quotient
    ratio `(‖G_q x‖₂² / xᵀ A x) / (‖G_q y‖₂² / yᵀ A y)` is bounded by
    `((1 + η) / (1 - η))²` times `(‖G x‖₂² / xᵀ A x) / (‖G y‖₂² / yᵀ A y)`. -/
theorem rayleigh_condition_distortion_bound {n : ℕ}
    (A G E : Matrix (Fin n) (Fin n) ℝ) (η : ℝ)
    (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hA : A.PosDef) (hG : IsUnit G)
    (he : ∀ u : Fin n → ℝ, l2Norm (E.mulVec u) ≤ η * l2Norm u)
    (x y : Fin n → ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
    let G_q := (1 + E) * G
    0 < rayleighQuotient G A y ∧
    0 < rayleighQuotient G_q A y ∧
    rayleighQuotient G_q A x / rayleighQuotient G_q A y ≤
      ((1 + η) / (1 - η)) ^ 2 * (rayleighQuotient G A x / rayleighQuotient G A y) := by
  intro G_q
  have hA_quad := (Matrix.posDef_iff_dotProduct_mulVec.mp hA).2
  have hxA : 0 < x ⬝ᵥ A.mulVec x := by simpa using hA_quad hx
  have hyA : 0 < y ⬝ᵥ A.mulVec y := by simpa using hA_quad hy
  have hGy_ne : G.mulVec y ≠ 0 := by
    intro h0
    rcases hG with ⟨u, rfl⟩
    let G_inv : Matrix (Fin n) (Fin n) ℝ := ↑u⁻¹
    let G_val : Matrix (Fin n) (Fin n) ℝ := ↑u
    have hinv : G_inv * G_val = 1 := Units.inv_mul u
    have hy0 : y = G_inv.mulVec (G_val.mulVec y) := by
      rw [Matrix.mulVec_mulVec, hinv, Matrix.one_mulVec]
    rw [show G_val.mulVec y = 0 from h0, Matrix.mulVec_zero] at hy0
    exact hy hy0
  have hGy_sq_pos : 0 < l2Norm (G.mulVec y) ^ 2 := by
    rw [l2Norm_sq]
    obtain ⟨i, hi⟩ : ∃ i : Fin n, G.mulVec y i ≠ 0 := by
      by_contra h
      push Not at h
      exact hGy_ne (funext h)
    exact Finset.sum_pos' (fun j _ => sq_nonneg (G.mulVec y j))
      ⟨i, Finset.mem_univ i, sq_pos_of_ne_zero hi⟩
  have hmul_eq : ∀ v : Fin n → ℝ, G_q.mulVec v = G.mulVec v + E.mulVec (G.mulVec v) := by
    intro v
    simp only [G_q, ← Matrix.mulVec_mulVec, Matrix.add_mulVec, Matrix.one_mulVec]
  have hbx :=
    (l2Norm_add_sq_bounds (G.mulVec x) (E.mulVec (G.mulVec x)) η hη0 hη1 (he (G.mulVec x))).2
  have hby :=
    (l2Norm_add_sq_bounds (G.mulVec y) (E.mulVec (G.mulVec y)) η hη0 hη1 (he (G.mulVec y))).1
  rw [← hmul_eq x] at hbx
  rw [← hmul_eq y] at hby
  have h1_pos : 0 < 1 - η := sub_pos.mpr hη1
  have hry_pos : 0 < rayleighQuotient G A y := div_pos hGy_sq_pos hyA
  have hry_lb : (1 - η) ^ 2 * rayleighQuotient G A y ≤ rayleighQuotient G_q A y := by
    simp only [rayleighQuotient]
    rw [← mul_div_assoc]
    exact div_le_div_of_nonneg_right hby hyA.le
  have hry_q_pos : 0 < rayleighQuotient G_q A y :=
    lt_of_lt_of_le (mul_pos (sq_pos_of_pos h1_pos) hry_pos) hry_lb
  have hrx_nn : 0 ≤ rayleighQuotient G A x :=
    div_nonneg (sq_nonneg _) hxA.le
  have hrx_q_nn : 0 ≤ rayleighQuotient G_q A x :=
    div_nonneg (sq_nonneg _) hxA.le
  have hrx_ub : rayleighQuotient G_q A x ≤ (1 + η) ^ 2 * rayleighQuotient G A x := by
    simp only [rayleighQuotient]
    rw [← mul_div_assoc]
    exact div_le_div_of_nonneg_right hbx hxA.le
  refine ⟨hry_pos, hry_q_pos, ?_⟩
  have hden_pos : 0 < (1 - η) ^ 2 * rayleighQuotient G A y :=
    mul_pos (sq_pos_of_pos h1_pos) hry_pos
  have hrhs_eq : ((1 + η) / (1 - η)) ^ 2 * (rayleighQuotient G A x / rayleighQuotient G A y) =
      ((1 + η) ^ 2 * rayleighQuotient G A x) / ((1 - η) ^ 2 * rayleighQuotient G A y) := by
    rw [div_pow, mul_div_mul_comm]
  rw [hrhs_eq, div_le_div_iff₀ hry_q_pos hden_pos]
  exact mul_le_mul hrx_ub hry_lb hden_pos.le (mul_nonneg (sq_nonneg _) hrx_nn)

noncomputable def A_ex : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1 : ℝ), 1 / 25; 1 / 25, 17 / 10000]

def G_ex : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1 : ℝ), 0; -4, 100]

def G_inc_ex : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1 : ℝ), 0; 0, 100]

theorem A_ex_posDef : A_ex.PosDef := by
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [A_ex]
  · intro x hx
    have hne : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra h
      push Not at h
      apply hx
      funext i
      fin_cases i
      · exact h.1
      · exact h.2
    have hval : star x ⬝ᵥ A_ex.mulVec x =
        (x 0 + (1 / 25 : ℝ) * x 1) ^ 2 + (1 / 10000 : ℝ) * (x 1) ^ 2 := by
      simp [A_ex, dotProduct, mulVec, Fin.sum_univ_two]
      ring
    rw [hval]
    rcases hne with h0 | h1
    · by_cases hx1 : x 1 = 0
      · rw [hx1]
        have hpos : 0 < (x 0) ^ 2 := sq_pos_of_ne_zero h0
        linarith
      · have hpos : 0 < (1 / 10000 : ℝ) * (x 1) ^ 2 := by positivity
        linarith [sq_nonneg (x 0 + (1 / 25 : ℝ) * x 1)]
    · have hpos : 0 < (1 / 10000 : ℝ) * (x 1) ^ 2 := by positivity
      linarith [sq_nonneg (x 0 + (1 / 25 : ℝ) * x 1)]

theorem G_ex_exact_inverse :
    G_ex * A_ex * G_exᵀ = (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [G_ex, A_ex, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

theorem inclusive_int4_rounds_offdiag_to_zero :
    |((-4 : ℝ) / (100 / 7))| < (1 : ℝ) / 2 := by
  norm_num

theorem split_int4_recovers_exact_factor :
    diagSplitQuant G_ex
      (fun i j => if i = 1 ∧ j = 0 then (-7 : ℝ) * (4 / 7) else 0) = G_ex := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [diagSplitQuant, G_ex] <;> norm_num

theorem G_inc_preconditioned_matrix :
    G_inc_ex * A_ex * G_inc_exᵀ = !![(1 : ℝ), 4; 4, 17] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [G_inc_ex, A_ex, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

theorem counterexample_2x2_kappa_gt_321 :
    let P := G_inc_ex * A_ex * G_inc_exᵀ
    Matrix.trace P = 18 ∧
    Matrix.det P = 1 ∧
    ∀ lmax lmin : ℝ,
      lmax + lmin = Matrix.trace P →
      lmax * lmin = Matrix.det P →
      0 < lmin → lmin ≤ lmax →
      (321 : ℝ) < lmax / lmin := by
  have hP := G_inc_preconditioned_matrix
  have htr : Matrix.trace (G_inc_ex * A_ex * G_inc_exᵀ) = 18 := by
    rw [hP]; simp [Matrix.trace, Fin.sum_univ_two]; norm_num
  have hdet : Matrix.det (G_inc_ex * A_ex * G_inc_exᵀ) = 1 := by
    rw [hP, Matrix.det_fin_two]; norm_num
  refine ⟨htr, hdet, ?_⟩
  intro lmax lmin hsum hprod hpos hle
  rw [htr] at hsum
  rw [hdet] at hprod
  rw [lt_div_iff₀ hpos]
  have hsmall : lmin < 9 / 161 := by
    by_contra hge
    push Not at hge
    have hmax_le : lmax ≤ 18 - 9 / 161 := by linarith
    have hmid : lmin ≤ 9 := by linarith
    have hprod_gt : 1 < lmax * lmin := by nlinarith
    linarith
  linarith

theorem damped_richardson_contraction (ω lam : ℝ)
    (hω : 0 < ω) (hlam : 0 < lam) (hbound : ω * lam < 2) :
    |1 - ω * lam| < 1 := by
  rw [abs_lt]
  constructor <;> nlinarith

theorem gram_spd_not_sufficient_for_fixed_damping :
    ∃ (A G : Matrix (Fin 2) (Fin 2) ℝ) (x : Fin 2 → ℝ),
      A.PosDef ∧
      IsLowerTriangular G ∧
      (∀ i, 0 < G i i) ∧
      (Gᵀ * G).PosDef ∧
      let M := Gᵀ * G
      let ω : ℝ := 65 / 100
      x ⬝ᵥ (2 • M - ω • (M * A * M)).mulVec x < 0 := by
  refine ⟨A_ex, G_inc_ex, ![0, 1], A_ex_posDef, ?_, ?_, ?_, ?_⟩
  · intro i j hij
    fin_cases i <;> fin_cases j <;> simp [G_inc_ex] at hij ⊢
  · intro i
    fin_cases i <;> simp [G_inc_ex]
  · apply gram_posDef_of_lowerTriangular_posDiag
    · intro i j hij
      fin_cases i <;> fin_cases j <;> simp [G_inc_ex] at hij ⊢
    · intro i
      fin_cases i <;> simp [G_inc_ex]
  · simp only [A_ex, G_inc_ex, dotProduct, mulVec, Matrix.sub_apply, Matrix.smul_apply,
      Matrix.mul_apply, Matrix.transpose_apply, smul_eq_mul, Fin.sum_univ_two,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    norm_num

def rowSumRepair {n : ℕ} (A Q : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i = j then A i i + ∑ k ∈ Finset.univ.erase i, (A i k - Q i k) else Q i j

theorem rowSum_repair_preserves_ones {n : ℕ} (A Q : Matrix (Fin n) (Fin n) ℝ) :
    (rowSumRepair A Q).mulVec (fun _ => 1) = A.mulVec (fun _ => 1) := by
  funext i
  simp only [mulVec, dotProduct, mul_one]
  rw [← Finset.add_sum_erase (s := Finset.univ)
    (f := fun j => rowSumRepair A Q i j) (Finset.mem_univ i)]
  rw [← Finset.add_sum_erase (s := Finset.univ)
    (f := fun j => A i j) (Finset.mem_univ i)]
  have hdiag : rowSumRepair A Q i i =
      A i i + ∑ k ∈ Finset.univ.erase i, (A i k - Q i k) := by
    simp [rowSumRepair]
  have hoff : ∑ j ∈ Finset.univ.erase i, rowSumRepair A Q i j =
      ∑ j ∈ Finset.univ.erase i, Q i j := by
    apply Finset.sum_congr rfl
    intro j hj
    have hne : i ≠ j := Ne.symm (Finset.mem_erase.mp hj).1
    simp [rowSumRepair, hne]
  rw [hdiag, hoff, Finset.sum_sub_distrib]
  ring

theorem rowSum_repair_diag_dom {n : ℕ} (A Q : Matrix (Fin n) (Fin n) ℝ)
    (hrow : ∀ i : Fin n, 0 ≤ ∑ j : Fin n, A i j)
    (hQ : ∀ i j : Fin n, i ≠ j → Q i j ≤ 0) :
    ∀ i : Fin n,
      ∑ j ∈ Finset.univ.erase i, |rowSumRepair A Q i j| ≤ rowSumRepair A Q i i := by
  intro i
  have hones := congr_fun (rowSum_repair_preserves_ones A Q) i
  simp only [mulVec, dotProduct, mul_one] at hones
  have hsum_nonneg : 0 ≤ ∑ j : Fin n, rowSumRepair A Q i j := by
    rw [hones]; exact hrow i
  rw [← Finset.add_sum_erase (s := Finset.univ)
    (f := fun j => rowSumRepair A Q i j) (Finset.mem_univ i)] at hsum_nonneg
  have habs : ∑ j ∈ Finset.univ.erase i, |rowSumRepair A Q i j| =
      - ∑ j ∈ Finset.univ.erase i, rowSumRepair A Q i j := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j hj
    have hne : i ≠ j := Ne.symm (Finset.mem_erase.mp hj).1
    simp only [rowSumRepair, hne, ↓reduceIte]
    exact abs_of_nonpos (hQ i j hne)
  linarith

theorem scale_factored_gram_identity {n : ℕ}
    (s : Fin n → ℝ) (Gtilde : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    let G := Matrix.diagonal s * Gtilde
    (Gᵀ * G).mulVec x =
      Gtildeᵀ.mulVec ((Matrix.diagonal (fun i => s i * s i)).mulVec (Gtilde.mulVec x)) := by
  intro G
  have hdiag : (Matrix.diagonal s)ᵀ * Matrix.diagonal s =
      Matrix.diagonal (fun i => s i * s i) := by
    rw [Matrix.diagonal_transpose, Matrix.diagonal_mul_diagonal]
  have hmat : Gᵀ * G = Gtildeᵀ * (Matrix.diagonal (fun i => s i * s i) * Gtilde) := by
    rw [show G = Matrix.diagonal s * Gtilde from rfl, Matrix.transpose_mul, Matrix.mul_assoc,
      ← Matrix.mul_assoc (Matrix.diagonal s)ᵀ, hdiag]
  rw [hmat, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]

theorem optimal_2x2_reprojection_properties (c11 c12 c22 : ℝ)
    (hc11 : 0 < c11) (hdelta : 0 < c11 * c22 - c12 ^ 2) :
    let Δ := c11 * c22 - c12 ^ 2
    let g_opt := Real.sqrt (c11 / Δ)
    let s_opt := -c12 / Real.sqrt (c11 * Δ)
    0 < g_opt ∧
    s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22 = 1 ∧
    ∀ s g : ℝ, 0 < g →
      (s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22) / g_opt ^ 2 ≤
        (s ^ 2 * c11 + 2 * s * g * c12 + g ^ 2 * c22) / g ^ 2 := by
  intro Δ g_opt s_opt
  have hΔ : 0 < Δ := hdelta
  have hprod : 0 < c11 * Δ := mul_pos hc11 hΔ
  have hg_pos : 0 < g_opt := Real.sqrt_pos.mpr (div_pos hc11 hΔ)
  have hg_sq : g_opt ^ 2 = c11 / Δ := Real.sq_sqrt (le_of_lt (div_pos hc11 hΔ))
  have hs_sq : s_opt ^ 2 = c12 ^ 2 / (c11 * Δ) := by
    simp only [s_opt, div_pow, neg_sq, Real.sq_sqrt (le_of_lt hprod)]
  have hsg : s_opt * g_opt = -c12 / Δ := by
    have hmul : Real.sqrt (c11 * Δ) * Real.sqrt (c11 / Δ) = c11 := by
      rw [← Real.sqrt_mul (le_of_lt hprod)]
      have heq : c11 * Δ * (c11 / Δ) = c11 ^ 2 := by field_simp [ne_of_gt hΔ]
      rw [heq, Real.sqrt_sq (le_of_lt hc11)]
    have hden : Real.sqrt (c11 * Δ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hprod)
    have h1 : s_opt * g_opt * Real.sqrt (c11 * Δ) = -c12 * Real.sqrt (c11 / Δ) := by
      simp only [s_opt, g_opt]
      field_simp [hden]
    have h2 : (-c12 / Δ) * Real.sqrt (c11 * Δ) =
        (-c12 / Δ) * (Real.sqrt (c11 * Δ)) := rfl
    have hprod_eq : Real.sqrt (c11 / Δ) / Real.sqrt (c11 * Δ) = 1 / Δ := by
      rw [div_eq_div_iff hden (ne_of_gt hΔ)]
      have hsq1 : (Real.sqrt (c11 / Δ) * Δ) ^ 2 = c11 * Δ := by
        rw [mul_pow, Real.sq_sqrt (le_of_lt (div_pos hc11 hΔ))]
        field_simp [ne_of_gt hΔ]
      have hsq2 : (1 * Real.sqrt (c11 * Δ)) ^ 2 = c11 * Δ := by
        rw [one_mul, Real.sq_sqrt (le_of_lt hprod)]
      have hn1 : 0 ≤ Real.sqrt (c11 / Δ) * Δ := by positivity
      have hn2 : 0 ≤ 1 * Real.sqrt (c11 * Δ) := by positivity
      nlinarith
    calc
      s_opt * g_opt = -c12 * (Real.sqrt (c11 / Δ) / Real.sqrt (c11 * Δ)) := by
        simp only [s_opt, g_opt]; ring
      _ = -c12 * (1 / Δ) := by rw [hprod_eq]
      _ = -c12 / Δ := by ring
  have hunit : s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22 = 1 := by
    have hassoc : 2 * s_opt * g_opt * c12 = 2 * (s_opt * g_opt) * c12 := by ring
    rw [hassoc, hs_sq, hsg, hg_sq]
    have hc11_ne : c11 ≠ 0 := ne_of_gt hc11
    have hΔ_ne : Δ ≠ 0 := ne_of_gt hΔ
    field_simp [hc11_ne, hΔ_ne]
    simp only [Δ]
    ring
  refine ⟨hg_pos, hunit, ?_⟩
  intro s g hg
  rw [hunit, hg_sq]
  have h1 : 1 / (c11 / Δ) = Δ / c11 := by field_simp [ne_of_gt hc11, ne_of_gt hΔ]
  rw [h1]
  have hdiff :
      (s ^ 2 * c11 + 2 * s * g * c12 + g ^ 2 * c22) / g ^ 2 - Δ / c11 =
        c11 * (s / g + c12 / c11) ^ 2 := by
    have hg_ne : g ≠ 0 := ne_of_gt hg
    have hc11_ne : c11 ≠ 0 := ne_of_gt hc11
    field_simp [hg_ne, hc11_ne]
    simp only [Δ]
    ring
  have hnonneg : 0 ≤ c11 * (s / g + c12 / c11) ^ 2 := by positivity
  linarith

/-- Theorem 3 zero-direction fallback branch: when the frozen off-diagonal vector
    vanishes (`c11 = 0`, `c12 = 0`, `c22 > 0`), setting `s_opt = 0` and
    `g_opt = 1 / √c22` preserves `g_opt > 0`, enforces exact unit `A`-energy
    `g_opt² * c22 = 1`, and globally minimizes Kaporin's local ratio over all
    `(s, g) ∈ ℝ × ℝ_{>0}`, directly grounding the `c11 = 0` diagonal renormalization
    used in `renormalized_2x2_counterexample_posDef`. -/
theorem zero_offdiag_reprojection_fallback (c11 c12 c22 : ℝ)
    (hc11 : c11 = 0) (hc12 : c12 = 0) (hc22 : 0 < c22) :
    let g_opt := 1 / Real.sqrt c22
    let s_opt : ℝ := 0
    0 < g_opt ∧
    g_opt ^ 2 * c22 = 1 ∧
    s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22 = 1 ∧
    (s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22) / g_opt ^ 2 = c22 ∧
    ∀ s g : ℝ, 0 < g →
      (s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22) / g_opt ^ 2 ≤
        (s ^ 2 * c11 + 2 * s * g * c12 + g ^ 2 * c22) / g ^ 2 := by
  intro g_opt s_opt
  have hrt : 0 < Real.sqrt c22 := Real.sqrt_pos.mpr hc22
  have hsq : Real.sqrt c22 ^ 2 = c22 := Real.sq_sqrt hc22.le
  have hg_pos : 0 < g_opt := div_pos one_pos hrt
  have hg_unit : g_opt ^ 2 * c22 = 1 := by
    simp only [g_opt, div_pow, one_pow, hsq]
    exact one_div_mul_cancel (ne_of_gt hc22)
  have hfull_unit : s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22 = 1 := by
    simp only [s_opt, hc11, hc12, mul_zero, zero_add]
    exact hg_unit
  have hrat : (s_opt ^ 2 * c11 + 2 * s_opt * g_opt * c12 + g_opt ^ 2 * c22) / g_opt ^ 2 = c22 := by
    simp only [s_opt, hc11, hc12, mul_zero, zero_add]
    exact mul_div_cancel_left₀ c22 (ne_of_gt (sq_pos_of_pos hg_pos))
  refine ⟨hg_pos, hg_unit, hfull_unit, hrat, ?_⟩
  intro s g hg
  rw [hrat]
  simp only [hc11, hc12, mul_zero, zero_add]
  rw [mul_div_cancel_left₀ c22 (ne_of_gt (sq_pos_of_pos hg))]

theorem renormalized_2x2_counterexample_posDef :
    let G_ren : Matrix (Fin 2) (Fin 2) ℝ := !![(1 : ℝ), 0; 0, 100 / Real.sqrt 17]
    let M := G_renᵀ * G_ren
    let ω : ℝ := 72 / 100
    (2 • M - ω • (M * A_ex * M)).PosDef := by
  intro G_ren M ω
  have h17 : (0 : ℝ) < 17 := by norm_num
  have hsq17 : Real.sqrt 17 ^ 2 = 17 := Real.sq_sqrt (le_of_lt h17)
  have hM : M = !![(1 : ℝ), 0; 0, 10000 / 17] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [M, G_ren, Matrix.mul_apply, Fin.sum_univ_two]
    · rw [← sq, div_pow, hsq17]; norm_num
  have hOp : 2 • M - ω • (M * A_ex * M) =
      !![(32 / 25 : ℝ), -(288 / 17); -(288 / 17), 12800 / 17] := by
    rw [hM]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [A_ex, ω] <;> norm_num
  rw [hOp, Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp
  · intro x hx
    have hne : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
      by_contra h
      push Not at h
      apply hx
      funext i
      fin_cases i
      · exact h.1
      · exact h.2
    have hquad : star x ⬝ᵥ (!![(32 / 25 : ℝ), -(288 / 17); -(288 / 17), 12800 / 17]).mulVec x =
        (32 / 25 : ℝ) * (x 0 - (225 / 17 : ℝ) * x 1) ^ 2 +
        (152800 / 289 : ℝ) * (x 1) ^ 2 := by
      simp [dotProduct, mulVec, Fin.sum_univ_two]
      ring
    rw [hquad]
    rcases hne with h0 | h1
    · by_cases hx1 : x 1 = 0
      · rw [hx1]
        have hpos : 0 < (x 0) ^ 2 := sq_pos_of_ne_zero h0
        linarith
      · have hpos : 0 < (152800 / 289 : ℝ) * (x 1) ^ 2 := by positivity
        have hnn : 0 ≤ (32 / 25 : ℝ) * (x 0 - (225 / 17 : ℝ) * x 1) ^ 2 := by positivity
        linarith
    · have hpos : 0 < (152800 / 289 : ℝ) * (x 1) ^ 2 := by positivity
      have hnn : 0 ≤ (32 / 25 : ℝ) * (x 0 - (225 / 17 : ℝ) * x 1) ^ 2 := by positivity
      linarith

end BitFSAI
