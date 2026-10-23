/- Retained exact two-subspace bounds and arbitrary-energy sharpness.
Historical scope, information, anisotropic and projection developments are omitted. -/
import PaperReliability.ModalModel

open scoped BigOperators
open Classical
namespace PaperReliability
variable {I : Type*} [Fintype I]

noncomputable def sloppySet (lam : I → ℝ) (ε : ℝ) : Finset I :=
  Finset.univ.filter (fun i => lam i ≤ ε)

noncomputable def stiffSet (lam : I → ℝ) (ε : ℝ) : Finset I :=
  Finset.univ.filter (fun i => ε < lam i)

noncomputable def normSqSloppy
  (c : I → ℝ) (lam : I → ℝ) (ε : ℝ) : ℝ :=
  Finset.sum (sloppySet lam ε) (fun i => (c i)^2)

noncomputable def normSqStiff
  (c : I → ℝ) (lam : I → ℝ) (ε : ℝ) : ℝ :=
  Finset.sum (stiffSet lam ε) (fun i => (c i)^2)

noncomputable def sigmaSq
  (c : I → ℝ) (lam : I → ℝ) (α : ℝ) : ℝ :=
  Finset.sum Finset.univ (fun i => ((c i)^2 / (α + lam i)))

private lemma predictive_variance_upper_bound
  (c : I → ℝ)
  (lam : I → ℝ)
  (α ε : ℝ)
  (hα : 0 < α)
  (hε : 0 < ε)
  (hlam : ∀ i, 0 ≤ lam i) :
  sigmaSq c lam α ≤
    normSqSloppy c lam ε / α +
    normSqStiff c lam ε / (α + ε) := by
  unfold sigmaSq normSqSloppy normSqStiff sloppySet stiffSet

  have h_filter :
    (Finset.univ.filter (fun x => ¬ lam x ≤ ε))
    =
    (Finset.univ.filter (fun x => ε < lam x)) := by
    apply Finset.ext
    intro x
    simp

  have h_split :
    Finset.sum Finset.univ (fun i => (c i)^2 / (α + lam i))
    =
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + lam i))
    +
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + lam i)) := by
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => lam i ≤ ε)]
    rw [h_filter]

  have h_sloppy :
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + lam i))
    ≤
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / α) := by
    apply Finset.sum_le_sum
    intro i hi
    have hsquare : 0 ≤ (c i)^2 := by
      exact sq_nonneg (c i)
    have hden : α ≤ α + lam i := by
      linarith [hlam i]
    gcongr

  have h_stiff :
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + lam i))
    ≤
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + ε)) := by
    apply Finset.sum_le_sum
    intro i hi
    have hgt : ε < lam i := by
      exact (Finset.mem_filter.mp hi).2
    have hsquare : 0 ≤ (c i)^2 := by
      exact sq_nonneg (c i)
    gcongr

  have h_sloppy_rewrite :
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / α)
    =
    (Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2)) / α := by
    rw [← Finset.sum_div]

  have h_stiff_rewrite :
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + ε))
    =
    (Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2)) / (α + ε) := by
    rw [← Finset.sum_div]

  rw [h_split]

  have h_sum := add_le_add h_sloppy h_stiff
  rw [h_sloppy_rewrite, h_stiff_rewrite] at h_sum
  exact h_sum

private lemma predictive_variance_lower_bound_two_term
  (c : I → ℝ)
  (lam : I → ℝ)
  (α ε Λ : ℝ)
  (hα : 0 < α)
  (hlam : ∀ i, 0 ≤ lam i)
  (hlam_ub : ∀ i, lam i ≤ Λ) :
  normSqSloppy c lam ε / (α + ε) + normSqStiff c lam ε / (α + Λ)
    ≤ sigmaSq c lam α := by
  unfold sigmaSq normSqSloppy normSqStiff sloppySet stiffSet

  have h_filter :
    (Finset.univ.filter (fun x => ¬ lam x ≤ ε))
    =
    (Finset.univ.filter (fun x => ε < lam x)) := by
    apply Finset.ext
    intro x
    simp

  have h_split :
    Finset.sum Finset.univ (fun i => (c i)^2 / (α + lam i))
    =
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + lam i))
    +
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + lam i)) := by
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => lam i ≤ ε)]
    rw [h_filter]

  have h_sloppy :
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + ε))
    ≤
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + lam i)) := by
    apply Finset.sum_le_sum
    intro i hi
    have hle : lam i ≤ ε := by
      exact (Finset.mem_filter.mp hi).2
    have hpos : 0 < α + lam i := by
      linarith [hlam i]
    have hsquare : 0 ≤ (c i)^2 := by
      exact sq_nonneg (c i)
    gcongr

  have h_stiff :
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + Λ))
    ≤
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + lam i)) := by
    apply Finset.sum_le_sum
    intro i hi
    have hub : lam i ≤ Λ := hlam_ub i
    have hpos : 0 < α + lam i := by
      linarith [hlam i]
    have hsquare : 0 ≤ (c i)^2 := by
      exact sq_nonneg (c i)
    gcongr

  have h_sloppy_rewrite :
    (Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2)) / (α + ε)
    =
    Finset.sum (Finset.univ.filter (fun i => lam i ≤ ε))
      (fun i => (c i)^2 / (α + ε)) := by
    rw [Finset.sum_div]

  have h_stiff_rewrite :
    (Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2)) / (α + Λ)
    =
    Finset.sum (Finset.univ.filter (fun i => ε < lam i))
      (fun i => (c i)^2 / (α + Λ)) := by
    rw [Finset.sum_div]

  rw [h_split, h_sloppy_rewrite, h_stiff_rewrite]
  exact add_le_add h_sloppy h_stiff

private theorem geometric_ceiling_upper_bound
  (c_s c_b : I → ℝ)
  (lam : I → ℝ)
  (α ε Λ : ℝ)
  (hα : 0 < α)
  (hε : 0 < ε)
  (hlam : ∀ i, 0 ≤ lam i)
  (hlam_ub : ∀ i, lam i ≤ Λ)
  (h_bulk : 0 < normSqSloppy c_b lam ε / (α + ε) +
              normSqStiff c_b lam ε / (α + Λ)) :
  sigmaSq c_s lam α / sigmaSq c_b lam α ≤
    (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)) /
    (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)) := by

  have h_num :=
    predictive_variance_upper_bound c_s lam α ε hα hε hlam

  have h_den :=
    predictive_variance_lower_bound_two_term c_b lam α ε Λ hα hlam hlam_ub

  have hαε : 0 < α + ε := by
    linarith

  have h_bulk_sigma_pos : 0 < sigmaSq c_b lam α :=
    lt_of_lt_of_le h_bulk h_den

  have h_num_nonneg :
    0 ≤ normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε) := by
    apply add_nonneg
    · apply div_nonneg
      · unfold normSqSloppy
        apply Finset.sum_nonneg
        intro i _
        exact sq_nonneg (c_s i)
      · exact le_of_lt hα
    · apply div_nonneg
      · unfold normSqStiff
        apply Finset.sum_nonneg
        intro i _
        exact sq_nonneg (c_s i)
      · exact le_of_lt hαε

  let A := sigmaSq c_s lam α
  let E := sigmaSq c_b lam α
  let N := normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)
  let D := normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)

  have hAD_le_ND : A * D ≤ N * D := by
    apply mul_le_mul_of_nonneg_right
    · simpa [A, N] using h_num
    · simpa [D] using le_of_lt h_bulk

  have hND_le_NE : N * D ≤ N * E := by
    apply mul_le_mul_of_nonneg_left
    · simpa [D, E] using h_den
    · simpa [N] using h_num_nonneg

  have h_cross : A * D ≤ N * E := by
    exact le_trans hAD_le_ND hND_le_NE

  have hD_pos : 0 < D := by
    simpa [D] using h_bulk

  have hE_pos : 0 < E := by
    simpa [E] using h_bulk_sigma_pos

  have h_div : A / E ≤ N / D := by
    have hD_ne : D ≠ 0 := ne_of_gt hD_pos
    have hE_ne : E ≠ 0 := ne_of_gt hE_pos
    rw [div_le_div_iff₀ hE_pos hD_pos]
    simpa [mul_comm, mul_left_comm, mul_assoc] using h_cross

  simpa [A, E, N, D] using h_div

private lemma sigmaSq_two_level
  (c : I → ℝ) (lam : I → ℝ) (α ε μ ν : ℝ)
  (hsl : ∀ i ∈ sloppySet lam ε, c i ≠ 0 → lam i = μ)
  (hst : ∀ i ∈ stiffSet lam ε, c i ≠ 0 → lam i = ν) :
  sigmaSq c lam α
    = normSqSloppy c lam ε / (α + μ) + normSqStiff c lam ε / (α + ν) := by

  have h_filter :
    (Finset.univ.filter (fun x => ¬ lam x ≤ ε))
    = (Finset.univ.filter (fun x => ε < lam x)) := by
    apply Finset.ext
    intro x
    simp

  have h_split :
    sigmaSq c lam α
      = Finset.sum (sloppySet lam ε) (fun i => (c i)^2 / (α + lam i))
      + Finset.sum (stiffSet lam ε) (fun i => (c i)^2 / (α + lam i)) := by
    unfold sigmaSq sloppySet stiffSet
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => lam i ≤ ε)]
    rw [h_filter]

  have h_sl :
    Finset.sum (sloppySet lam ε) (fun i => (c i)^2 / (α + lam i))
      = normSqSloppy c lam ε / (α + μ) := by
    unfold normSqSloppy
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i hi
    by_cases h0 : c i = 0
    · simp [h0]
    · rw [hsl i hi h0]

  have h_st :
    Finset.sum (stiffSet lam ε) (fun i => (c i)^2 / (α + lam i))
      = normSqStiff c lam ε / (α + ν) := by
    unfold normSqStiff
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i hi
    by_cases h0 : c i = 0
    · simp [h0]
    · rw [hst i hi h0]

  rw [h_split, h_sl, h_st]

private theorem upper_envelope_witness_value
  (c_s c_b : I → ℝ) (lam : I → ℝ) (α ε ε' Λ : ℝ)
  (hs_sl : ∀ i ∈ sloppySet lam ε, c_s i ≠ 0 → lam i = 0)
  (hs_st : ∀ i ∈ stiffSet lam ε, c_s i ≠ 0 → lam i = ε')
  (hb_sl : ∀ i ∈ sloppySet lam ε, c_b i ≠ 0 → lam i = ε)
  (hb_st : ∀ i ∈ stiffSet lam ε, c_b i ≠ 0 → lam i = Λ) :
  sigmaSq c_s lam α / sigmaSq c_b lam α
    = (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε')) /
      (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)) := by
  rw [sigmaSq_two_level c_s lam α ε 0 ε' hs_sl hs_st,
      sigmaSq_two_level c_b lam α ε ε Λ hb_sl hb_st]
  norm_num

private theorem upper_envelope_sharp
  (c_s c_b : I → ℝ) (lam : I → ℝ) (α ε ε' Λ : ℝ)
  (hα : 0 < α) (hε : 0 < ε) (hεε' : ε ≤ ε')
  (hD : 0 < normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ))
  (hs_sl : ∀ i ∈ sloppySet lam ε, c_s i ≠ 0 → lam i = 0)
  (hs_st : ∀ i ∈ stiffSet lam ε, c_s i ≠ 0 → lam i = ε')
  (hb_sl : ∀ i ∈ sloppySet lam ε, c_b i ≠ 0 → lam i = ε)
  (hb_st : ∀ i ∈ stiffSet lam ε, c_b i ≠ 0 → lam i = Λ) :
  (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)) /
      (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ))
    ≤ ((α + ε') / (α + ε)) * (sigmaSq c_s lam α / sigmaSq c_b lam α) := by

  have hαε : 0 < α + ε := by linarith
  have hαε' : 0 < α + ε' := by linarith
  have hαε_ne : (α + ε) ≠ 0 := ne_of_gt hαε
  have hαε'_ne : (α + ε') ≠ 0 := ne_of_gt hαε'

  have h_sl_nonneg : 0 ≤ normSqSloppy c_s lam ε := by
    unfold normSqSloppy
    exact Finset.sum_nonneg (fun i _ => sq_nonneg (c_s i))

  have hk : 1 ≤ (α + ε') / (α + ε) := by
    have h1 : (α + ε) / (α + ε) ≤ (α + ε') / (α + ε) := by
      rw [div_le_div_iff₀ hαε hαε]
      nlinarith
    rwa [div_self hαε_ne] at h1

  have h_num :
    normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)
      ≤ ((α + ε') / (α + ε)) *
        (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε')) := by
    have h_stiff_eq :
      ((α + ε') / (α + ε)) * (normSqStiff c_s lam ε / (α + ε'))
        = normSqStiff c_s lam ε / (α + ε) := by
      field_simp
    have h_sloppy_le :
      normSqSloppy c_s lam ε / α
        ≤ ((α + ε') / (α + ε)) * (normSqSloppy c_s lam ε / α) := by
      nlinarith [div_nonneg h_sl_nonneg (le_of_lt hα), hk]
    calc normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)
        ≤ ((α + ε') / (α + ε)) * (normSqSloppy c_s lam ε / α)
            + ((α + ε') / (α + ε)) * (normSqStiff c_s lam ε / (α + ε')) := by
          rw [h_stiff_eq]; linarith
      _ = ((α + ε') / (α + ε)) *
            (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε')) := by
          ring

  rw [upper_envelope_witness_value c_s c_b lam α ε ε' Λ hs_sl hs_st hb_sl hb_st]

  have h_pull :
    ((α + ε') / (α + ε)) *
      ((normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε')) /
       (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)))
    = (((α + ε') / (α + ε)) *
        (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε'))) /
      (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)) := by
    ring
  rw [h_pull, div_le_div_iff₀ hD hD]
  exact mul_le_mul_of_nonneg_right h_num (le_of_lt hD)

noncomputable def regAvg {X : Type*} (S : Finset X) (c : X → I → ℝ) (i : I) : ℝ :=
  Real.sqrt ((S.card : ℝ)⁻¹ * Finset.sum S (fun x => (c x i) ^ 2))

private theorem mean_sigmaSq_eq_sigmaSq_regAvg
    {X : Type*} (S : Finset X) (c : X → I → ℝ) (lam : I → ℝ) (α : ℝ) :
    (S.card : ℝ)⁻¹ * Finset.sum S (fun x => sigmaSq (c x) lam α)
      = sigmaSq (regAvg S c) lam α := by
  classical
  unfold sigmaSq regAvg

  have hnn : ∀ i : I, 0 ≤ (S.card : ℝ)⁻¹ * Finset.sum S (fun x => (c x i) ^ 2) := fun i =>
    mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg _)) (Finset.sum_nonneg (fun x _ => sq_nonneg _))
  simp_rw [Real.sq_sqrt (hnn _)]

  rw [Finset.sum_comm, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [← Finset.sum_div, mul_div_assoc]

/-- The upper two-subspace envelope is an arbitrarily close supremum for prescribed nonnegative regional loading energies. -/
theorem upper_envelope_arbitrary_energy_sharpness
    (α ε ε' Λ qS_sl qS_st qB_sl qB_st : ℝ)
    (hα : 0 < α) (hε : 0 < ε) (hεε' : ε < ε') (hε'Λ : ε' ≤ Λ)
    (hqS_sl : 0 ≤ qS_sl) (hqS_st : 0 ≤ qS_st)
    (hqB_sl : 0 ≤ qB_sl) (hqB_st : 0 ≤ qB_st)
    (hqB : 0 < qB_sl + qB_st) :
    ∃ (c_s c_b lam : Fin 4 → ℝ),
      (∀ i, 0 ≤ lam i) ∧
      (∀ i, lam i ≤ Λ) ∧
      normSqSloppy c_s lam ε = qS_sl ∧
      normSqStiff c_s lam ε = qS_st ∧
      normSqSloppy c_b lam ε = qB_sl ∧
      normSqStiff c_b lam ε = qB_st ∧
      sigmaSq c_s lam α / sigmaSq c_b lam α ≤
        (qS_sl / α + qS_st / (α + ε)) /
        (qB_sl / (α + ε) + qB_st / (α + Λ)) ∧
      (qS_sl / α + qS_st / (α + ε)) /
          (qB_sl / (α + ε) + qB_st / (α + Λ)) ≤
        ((α + ε') / (α + ε)) *
          (sigmaSq c_s lam α / sigmaSq c_b lam α) := by
  classical
  let c_s : Fin 4 → ℝ := ![Real.sqrt qS_sl, 0, Real.sqrt qS_st, 0]
  let c_b : Fin 4 → ℝ := ![0, Real.sqrt qB_sl, 0, Real.sqrt qB_st]
  let lam : Fin 4 → ℝ := ![0, ε, ε', Λ]

  have hsl : sloppySet lam ε = {0, 1} := by
    ext i
    fin_cases i <;> simp [sloppySet, lam, Finset.mem_filter] <;> linarith
  have hst : stiffSet lam ε = {2, 3} := by
    ext i
    fin_cases i <;> simp [stiffSet, lam, Finset.mem_filter] <;> linarith

  have hcS_sl : normSqSloppy c_s lam ε = qS_sl := by
    simp [normSqSloppy, hsl, c_s, Real.sq_sqrt hqS_sl]
  have hcS_st : normSqStiff c_s lam ε = qS_st := by
    simp [normSqStiff, hst, c_s, Real.sq_sqrt hqS_st]
  have hcB_sl : normSqSloppy c_b lam ε = qB_sl := by
    simp [normSqSloppy, hsl, c_b, Real.sq_sqrt hqB_sl]
  have hcB_st : normSqStiff c_b lam ε = qB_st := by
    simp [normSqStiff, hst, c_b, Real.sq_sqrt hqB_st]

  have hlam : ∀ i, 0 ≤ lam i := by
    intro i
    fin_cases i <;> simp [lam] <;> linarith
  have hlam_ub : ∀ i, lam i ≤ Λ := by
    intro i
    fin_cases i <;> simp [lam] <;> linarith

  have hs_sl : ∀ i ∈ sloppySet lam ε, c_s i ≠ 0 → lam i = 0 := by
    intro i hi hci
    fin_cases i <;> simp_all [c_s, lam]
  have hs_st : ∀ i ∈ stiffSet lam ε, c_s i ≠ 0 → lam i = ε' := by
    intro i hi hci
    fin_cases i <;> simp_all [c_s, lam]
  have hb_sl : ∀ i ∈ sloppySet lam ε, c_b i ≠ 0 → lam i = ε := by
    intro i hi hci
    fin_cases i <;> simp_all [c_b, lam]
  have hb_st : ∀ i ∈ stiffSet lam ε, c_b i ≠ 0 → lam i = Λ := by
    intro i hi hci
    fin_cases i <;> simp_all [c_b, lam]

  have hαε : 0 < α + ε := by linarith
  have hαΛ : 0 < α + Λ := by linarith
  have hDq : 0 < qB_sl / (α + ε) + qB_st / (α + Λ) := by
    by_cases hz : qB_sl = 0
    · have hqB_st_pos : 0 < qB_st := by linarith
      rw [hz]
      norm_num
      exact div_pos hqB_st_pos hαΛ
    · have hqB_sl_pos : 0 < qB_sl := lt_of_le_of_ne hqB_sl (Ne.symm hz)
      exact add_pos_of_pos_of_nonneg (div_pos hqB_sl_pos hαε)
        (div_nonneg hqB_st (le_of_lt hαΛ))
  have hD : 0 < normSqSloppy c_b lam ε / (α + ε) +
      normSqStiff c_b lam ε / (α + Λ) := by
    simpa only [hcB_sl, hcB_st] using hDq

  have hupper :=
    geometric_ceiling_upper_bound c_s c_b lam α ε Λ hα hε hlam hlam_ub hD
  have hsharp :=
    upper_envelope_sharp c_s c_b lam α ε ε' Λ hα hε (le_of_lt hεε') hD
      hs_sl hs_st hb_sl hb_st

  refine ⟨c_s, c_b, lam, hlam, hlam_ub, hcS_sl, hcS_st, hcB_sl, hcB_st, ?_, ?_⟩
  · simpa only [hcS_sl, hcS_st, hcB_sl, hcB_st] using hupper
  · simpa only [hcS_sl, hcS_st, hcB_sl, hcB_st] using hsharp

private lemma ratio_enclosure_of_enclosures
  {a b La Ua Lb Ub : ℝ}
  (hLa : 0 ≤ La) (ha_lo : La ≤ a) (ha_hi : a ≤ Ua)
  (hLb : 0 < Lb) (hb_lo : Lb ≤ b) (hb_hi : b ≤ Ub) :
  La / Ub ≤ a / b ∧ a / b ≤ Ua / Lb := by
  have hb : 0 < b := lt_of_lt_of_le hLb hb_lo
  have hUb : 0 < Ub := lt_of_lt_of_le hb hb_hi
  have ha : 0 ≤ a := le_trans hLa ha_lo
  constructor
  · rw [div_le_div_iff₀ hUb hb]
    nlinarith
  · rw [div_le_div_iff₀ hb hLb]
    nlinarith

/-- Two spectral groups enclose the variance ratio, equivalently the squared RMS uncertainty contrast. -/
theorem two_subspace_contrast_enclosure
    (c_s c_b lam : I → ℝ) (α ε Λ : ℝ)
    (hα : 0 < α) (hε : 0 < ε) (hΛ : 0 ≤ Λ)
    (hlam : ∀ i, 0 ≤ lam i) (hlam_ub : ∀ i, lam i ≤ Λ)
    (h_bulk : 0 < normSqSloppy c_b lam ε / (α + ε) +
      normSqStiff c_b lam ε / (α + Λ)) :
    (normSqSloppy c_s lam ε / (α + ε) + normSqStiff c_s lam ε / (α + Λ)) /
        (normSqSloppy c_b lam ε / α + normSqStiff c_b lam ε / (α + ε))
      ≤ sigmaSq c_s lam α / sigmaSq c_b lam α ∧
    sigmaSq c_s lam α / sigmaSq c_b lam α ≤
      (normSqSloppy c_s lam ε / α + normSqStiff c_s lam ε / (α + ε)) /
        (normSqSloppy c_b lam ε / (α + ε) + normSqStiff c_b lam ε / (α + Λ)) := by
  have hsl : 0 ≤ normSqSloppy c_s lam ε :=
    Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  have hst : 0 ≤ normSqStiff c_s lam ε :=
    Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  have hm : 0 ≤ normSqSloppy c_s lam ε / (α + ε) +
      normSqStiff c_s lam ε / (α + Λ) :=
    add_nonneg (div_nonneg hsl (by linarith)) (div_nonneg hst (by linarith))
  exact ratio_enclosure_of_enclosures hm
    (predictive_variance_lower_bound_two_term c_s lam α ε Λ hα hlam hlam_ub)
    (predictive_variance_upper_bound c_s lam α ε hα hε hlam) h_bulk
    (predictive_variance_lower_bound_two_term c_b lam α ε Λ hα hlam hlam_ub)
    (predictive_variance_upper_bound c_b lam α ε hα hε hlam)

/-- Averaging squared tangent coefficients yields the same two-subspace enclosure for regional mean variances. -/
theorem regional_two_subspace_enclosure
    {X : Type*} (Sh Bk : Finset X)
    (c : X → I → ℝ) (lam : I → ℝ) (α ε Λ : ℝ)
    (hα : 0 < α) (hε : 0 < ε) (hΛ : 0 ≤ Λ)
    (hlam : ∀ i, 0 ≤ lam i) (hlam_ub : ∀ i, lam i ≤ Λ)
    (h_bulk : 0 < normSqSloppy (regAvg Bk c) lam ε / (α + ε) +
      normSqStiff (regAvg Bk c) lam ε / (α + Λ)) :
    (normSqSloppy (regAvg Sh c) lam ε / (α + ε) +
        normSqStiff (regAvg Sh c) lam ε / (α + Λ)) /
      (normSqSloppy (regAvg Bk c) lam ε / α +
        normSqStiff (regAvg Bk c) lam ε / (α + ε))
      ≤ ((Sh.card : ℝ)⁻¹ * Finset.sum Sh (fun x => sigmaSq (c x) lam α)) /
        ((Bk.card : ℝ)⁻¹ * Finset.sum Bk (fun x => sigmaSq (c x) lam α)) ∧
    ((Sh.card : ℝ)⁻¹ * Finset.sum Sh (fun x => sigmaSq (c x) lam α)) /
        ((Bk.card : ℝ)⁻¹ * Finset.sum Bk (fun x => sigmaSq (c x) lam α)) ≤
      (normSqSloppy (regAvg Sh c) lam ε / α +
        normSqStiff (regAvg Sh c) lam ε / (α + ε)) /
      (normSqSloppy (regAvg Bk c) lam ε / (α + ε) +
        normSqStiff (regAvg Bk c) lam ε / (α + Λ)) := by
  rw [mean_sigmaSq_eq_sigmaSq_regAvg, mean_sigmaSq_eq_sigmaSq_regAvg]
  exact two_subspace_contrast_enclosure (regAvg Sh c) (regAvg Bk c) lam
    α ε Λ hα hε hΛ hlam hlam_ub h_bulk

/-- The lower two-subspace envelope is an arbitrarily close infimum for prescribed nonnegative regional loading energies. -/
theorem lower_envelope_arbitrary_energy_sharpness
    (α ε ε' Λ qS_sl qS_st qB_sl qB_st : ℝ)
    (hα : 0 < α) (hε : 0 < ε) (hεε' : ε < ε') (hε'Λ : ε' ≤ Λ)
    (hqS_sl : 0 ≤ qS_sl) (hqS_st : 0 ≤ qS_st)
    (hqB_sl : 0 ≤ qB_sl) (hqB_st : 0 ≤ qB_st)
    (hqB : 0 < qB_sl + qB_st) :
    ∃ (c_s c_b lam : Fin 4 → ℝ),
      (∀ i, 0 ≤ lam i) ∧ (∀ i, lam i ≤ Λ) ∧
      normSqSloppy c_s lam ε = qS_sl ∧
      normSqStiff c_s lam ε = qS_st ∧
      normSqSloppy c_b lam ε = qB_sl ∧
      normSqStiff c_b lam ε = qB_st ∧
      (qS_sl / (α + ε) + qS_st / (α + Λ)) /
          (qB_sl / α + qB_st / (α + ε)) ≤
        sigmaSq c_s lam α / sigmaSq c_b lam α ∧
      sigmaSq c_s lam α / sigmaSq c_b lam α ≤
        ((α + ε') / (α + ε)) *
          ((qS_sl / (α + ε) + qS_st / (α + Λ)) /
            (qB_sl / α + qB_st / (α + ε))) := by
  classical
  let c_s : Fin 4 → ℝ := ![Real.sqrt qS_sl, Real.sqrt qS_st, 0, 0]
  let c_b : Fin 4 → ℝ := ![0, 0, Real.sqrt qB_sl, Real.sqrt qB_st]
  let lam : Fin 4 → ℝ := ![ε, Λ, 0, ε']
  have hsl : sloppySet lam ε = {0, 2} := by
    ext i
    fin_cases i <;> simp [sloppySet, lam, Finset.mem_filter] <;> linarith
  have hst : stiffSet lam ε = {1, 3} := by
    ext i
    fin_cases i <;> simp [stiffSet, lam, Finset.mem_filter] <;> linarith
  have hcS_sl : normSqSloppy c_s lam ε = qS_sl := by
    simp [normSqSloppy, hsl, c_s, Real.sq_sqrt hqS_sl]
  have hcS_st : normSqStiff c_s lam ε = qS_st := by
    simp [normSqStiff, hst, c_s, Real.sq_sqrt hqS_st]
  have hcB_sl : normSqSloppy c_b lam ε = qB_sl := by
    simp [normSqSloppy, hsl, c_b, Real.sq_sqrt hqB_sl]
  have hcB_st : normSqStiff c_b lam ε = qB_st := by
    simp [normSqStiff, hst, c_b, Real.sq_sqrt hqB_st]
  have hlam : ∀ i, 0 ≤ lam i := by
    intro i
    fin_cases i <;> simp [lam] <;> linarith
  have hlam_ub : ∀ i, lam i ≤ Λ := by
    intro i
    fin_cases i <;> simp [lam] <;> linarith
  have hsigS : sigmaSq c_s lam α = qS_sl / (α + ε) + qS_st / (α + Λ) := by
    simp [sigmaSq, Fin.sum_univ_succ, c_s, lam, Real.sq_sqrt hqS_sl,
      Real.sq_sqrt hqS_st]
  have hsigB : sigmaSq c_b lam α = qB_sl / α + qB_st / (α + ε') := by
    simp [sigmaSq, Fin.sum_univ_succ, c_b, lam, Real.sq_sqrt hqB_sl,
      Real.sq_sqrt hqB_st]
  have hαε : 0 < α + ε := by linarith
  have hαε' : 0 < α + ε' := by linarith
  have hαΛ : 0 < α + Λ := by linarith
  let N := qS_sl / (α + ε) + qS_st / (α + Λ)
  let D := qB_sl / α + qB_st / (α + ε)
  let D' := qB_sl / α + qB_st / (α + ε')
  let k := (α + ε') / (α + ε)
  have hN : 0 ≤ N :=
    add_nonneg (div_nonneg hqS_sl hαε.le) (div_nonneg hqS_st hαΛ.le)
  have hD' : 0 < D' := by
    dsimp [D']
    by_cases hz : qB_sl = 0
    · have hqB_st_pos : 0 < qB_st := by linarith
      simpa [hz] using div_pos hqB_st_pos hαε'
    · have hqB_sl_pos : 0 < qB_sl := lt_of_le_of_ne hqB_sl (Ne.symm hz)
      exact add_pos_of_pos_of_nonneg (div_pos hqB_sl_pos hα)
        (div_nonneg hqB_st hαε'.le)
  have hD_le : D' ≤ D := by
    dsimp [D', D]
    apply add_le_add_right
    rw [div_le_div_iff₀ hαε' hαε]
    nlinarith
  have hD : 0 < D := lt_of_lt_of_le hD' hD_le
  have hk : 1 ≤ k := by
    dsimp [k]
    rw [le_div_iff₀ hαε]
    linarith
  have hscaled : D ≤ k * D' := by
    have hstiff : k * (qB_st / (α + ε')) = qB_st / (α + ε) := by
      dsimp [k]
      field_simp
    have hsloppy : qB_sl / α ≤ k * (qB_sl / α) := by
      nlinarith [div_nonneg hqB_sl hα.le]
    dsimp [D, D']
    rw [mul_add, hstiff]
    exact add_le_add_left hsloppy _
  refine ⟨c_s, c_b, lam, hlam, hlam_ub, hcS_sl, hcS_st, hcB_sl, hcB_st, ?_, ?_⟩
  · rw [hsigS, hsigB]
    change N / D ≤ N / D'
    rw [div_le_div_iff₀ hD hD']
    exact mul_le_mul_of_nonneg_left hD_le hN
  · rw [hsigS, hsigB]
    change N / D' ≤ k * (N / D)
    have hpull : k * (N / D) = (k * N) / D := by ring
    rw [hpull, div_le_div_iff₀ hD' hD]
    calc N * D ≤ N * (k * D') := mul_le_mul_of_nonneg_left hscaled hN
      _ = (k * N) * D' := by ring

end PaperReliability
