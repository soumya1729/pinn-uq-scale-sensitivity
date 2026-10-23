import PaperReliability.ModalModel

open scoped BigOperators Topology
open Filter

namespace PaperReliability

variable {ι : Type*} [Fintype ι]

/-- The whole-kernel regional energy, including every zero eigenvalue. -/
noncomputable def kernelEnergy (spectrum energy : ι → ℝ) : ℝ :=
  ∑ i, if spectrum i = 0 then energy i else 0

private lemma scaled_modalVariance (spectrum energy : ι → ℝ) (γ : ℝ) :
    γ * modalVariance spectrum energy γ =
      ∑ i, energy i * (γ / (spectrum i + γ)) := by
  unfold modalVariance
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

private lemma strong_weight_limit (lam : ℝ) :
    Tendsto (fun γ : ℝ => γ / (lam + γ)) atTop (𝓝 1) := by
  have hdiv : Tendsto (fun γ : ℝ => lam / γ) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_id
  have hsum : Tendsto (fun γ : ℝ => 1 + lam / γ) atTop (𝓝 1) := by
    simpa using (tendsto_const_nhds (x := (1 : ℝ))).add hdiv
  have hinv : Tendsto (fun γ : ℝ => (1 + lam / γ)⁻¹) atTop (𝓝 1) := by
    simpa using hsum.inv₀ one_ne_zero
  apply hinv.congr'
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with γ hγ
  have heq : 1 + lam / γ = (lam + γ) / γ := by
    field_simp [ne_of_gt hγ]
    ring
  rw [heq, inv_div]

private lemma weak_weight_limit (lam : ℝ) :
    Tendsto (fun γ : ℝ => γ / (lam + γ)) (𝓝[Set.Ioi 0] 0)
      (𝓝 (if lam = 0 then 1 else 0)) := by
  by_cases hlam : lam = 0
  · simp only [hlam, if_true]
    apply tendsto_const_nhds.congr'
    filter_upwards [self_mem_nhdsWithin] with γ hγ
    simp only [Set.mem_Ioi] at hγ
    simp [ne_of_gt hγ]
  · simp only [hlam, if_false]
    have hid : Tendsto (fun γ : ℝ => γ) (𝓝[Set.Ioi 0] 0) (𝓝 0) :=
      tendsto_id.mono_left nhdsWithin_le_nhds
    have hden : Tendsto (fun γ : ℝ => lam + γ) (𝓝[Set.Ioi 0] 0) (𝓝 lam) := by
      simpa using (tendsto_const_nhds (x := lam)).add hid
    simpa using hid.div hden hlam

private lemma strong_scaled_variance_limit (spectrum energy : ι → ℝ) :
    Tendsto (fun γ : ℝ => γ * modalVariance spectrum energy γ)
      atTop (𝓝 (∑ i, energy i)) := by
  simp_rw [scaled_modalVariance]
  simpa using tendsto_finsetSum Finset.univ (fun i _ =>
    tendsto_const_nhds.mul (strong_weight_limit (spectrum i)))

private lemma weak_scaled_variance_limit (spectrum energy : ι → ℝ) :
    Tendsto (fun γ : ℝ => γ * modalVariance spectrum energy γ)
      (𝓝[Set.Ioi 0] 0) (𝓝 (kernelEnergy spectrum energy)) := by
  simp_rw [scaled_modalVariance]
  unfold kernelEnergy
  have h := tendsto_finsetSum Finset.univ (fun i _ =>
    (tendsto_const_nhds (x := energy i)).mul (weak_weight_limit (spectrum i)))
  simpa only [mul_ite, mul_one, mul_zero] using h

/-- Strong precision removes spectral weighting: the RMS contrast tends to the
ratio of the total regional tangent energies. -/
theorem regional_contrast_strong_limit (spectrum sharp bulk : ι → ℝ)
    (hbulk : 0 < ∑ i, bulk i) :
    Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
      atTop (𝓝 (Real.sqrt ((∑ i, sharp i) / (∑ i, bulk i)))) := by
  have h := (strong_scaled_variance_limit spectrum sharp).div
    (strong_scaled_variance_limit spectrum bulk) (ne_of_gt hbulk)
  have hratio : Tendsto (contrastSq spectrum sharp bulk)
      atTop (𝓝 ((∑ i, sharp i) / (∑ i, bulk i))) := by
    apply h.congr'
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with γ hγ
    simp [contrastSq, mul_div_mul_left, ne_of_gt hγ]
  exact Real.continuous_sqrt.continuousAt.tendsto.comp hratio

/-- At weak positive precision, positive bulk kernel energy makes the whole-kernel
loading ratio determine the RMS contrast. A QR complement alone need not be the kernel. -/
theorem regional_contrast_weak_limit (spectrum sharp bulk : ι → ℝ)
    (hbulk : 0 < kernelEnergy spectrum bulk) :
    Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
      (𝓝[Set.Ioi 0] 0)
      (𝓝 (Real.sqrt (kernelEnergy spectrum sharp / kernelEnergy spectrum bulk))) := by
  have h := (weak_scaled_variance_limit spectrum sharp).div
    (weak_scaled_variance_limit spectrum bulk) (ne_of_gt hbulk)
  have hratio : Tendsto (contrastSq spectrum sharp bulk)
      (𝓝[Set.Ioi 0] 0)
      (𝓝 (kernelEnergy spectrum sharp / kernelEnergy spectrum bulk)) := by
    apply h.congr'
    filter_upwards [self_mem_nhdsWithin] with γ hγ
    simp only [Set.mem_Ioi] at hγ
    simp [contrastSq, mul_div_mul_left, ne_of_gt hγ]
  exact Real.continuous_sqrt.continuousAt.tendsto.comp hratio

private lemma peak_with_equal_endpoint_limits_is_nonmonotone (f : ℝ → ℝ)
    (hstrong : Tendsto f atTop (𝓝 1))
    (hweak : Tendsto f (𝓝[Set.Ioi 0] 0) (𝓝 1)) (hpeak : 1 < f 1) :
    ¬ MonotoneOn f (Set.Ioi 0) ∧ ¬ AntitoneOn f (Set.Ioi 0) := by
  constructor
  · intro hmono
    have hle : f 1 ≤ 1 := ge_of_tendsto hstrong (by
      filter_upwards [eventually_ge_atTop (1 : ℝ)] with γ hγ
      exact hmono (by norm_num) (by change 0 < γ; linarith) hγ)
    linarith
  · intro hanti
    have hlt : ∀ᶠ γ : ℝ in 𝓝[Set.Ioi 0] 0, γ < 1 :=
      (eventually_lt_nhds (by norm_num : (0 : ℝ) < 1)).filter_mono
        nhdsWithin_le_nhds
    have hle : f 1 ≤ 1 := ge_of_tendsto hweak (by
      filter_upwards [self_mem_nhdsWithin, hlt] with γ hγ hγ1
      exact hanti hγ (by norm_num) hγ1.le)
    linarith

/-- Physically admissible finite modal data can have both endpoint RMS limits equal
to one while the intermediate squared contrast exceeds twenty. Thus the endpoints
do not bound the curve, and it is neither increasing nor decreasing. -/
theorem endpoint_limits_do_not_bound_contrast :
    ∃ (spectrum sharp bulk : Fin 3 → ℝ),
      (∀ i, 0 ≤ spectrum i) ∧ (∀ i, 0 ≤ sharp i) ∧ (∀ i, 0 < bulk i) ∧
      Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
        atTop (𝓝 1) ∧
      Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
        (𝓝[Set.Ioi 0] 0) (𝓝 1) ∧
      contrastSq spectrum sharp bulk 1 = 10304 / 503 ∧
      20 < contrastSq spectrum sharp bulk 1 ∧
      1 < Real.sqrt (contrastSq spectrum sharp bulk 1) ∧
      ¬ MonotoneOn (fun γ => Real.sqrt (contrastSq spectrum sharp bulk γ))
        (Set.Ioi 0) ∧
      ¬ AntitoneOn (fun γ => Real.sqrt (contrastSq spectrum sharp bulk γ))
        (Set.Ioi 0) := by
  let spectrum : Fin 3 → ℝ := ![0, 1, 100]
  let sharp : Fin 3 → ℝ := ![1, 100, 1]
  let bulk : Fin 3 → ℝ := ![1, 1, 100]
  have hspectrum : ∀ i, 0 ≤ spectrum i := by
    intro i
    fin_cases i <;> norm_num [spectrum]
  have hsharp : ∀ i, 0 ≤ sharp i := by
    intro i
    fin_cases i <;> norm_num [sharp]
  have hbulk : ∀ i, 0 < bulk i := by
    intro i
    fin_cases i <;> norm_num [bulk]
  have hsumS : ∑ i, sharp i = 102 := by norm_num [sharp, Fin.sum_univ_succ]
  have hsumB : ∑ i, bulk i = 102 := by norm_num [bulk, Fin.sum_univ_succ]
  have hkernelS : kernelEnergy spectrum sharp = 1 := by
    norm_num [kernelEnergy, spectrum, sharp, Fin.sum_univ_succ]
  have hkernelB : kernelEnergy spectrum bulk = 1 := by
    norm_num [kernelEnergy, spectrum, bulk, Fin.sum_univ_succ]
  have hstrong := regional_contrast_strong_limit spectrum sharp bulk
    (by rw [hsumB]; norm_num)
  have hweak := regional_contrast_weak_limit spectrum sharp bulk
    (by rw [hkernelB]; norm_num)
  have hvalue : contrastSq spectrum sharp bulk 1 = 10304 / 503 := by
    norm_num [contrastSq, modalVariance, spectrum, sharp, bulk, Fin.sum_univ_succ]
  have hlarge : 20 < contrastSq spectrum sharp bulk 1 := by rw [hvalue]; norm_num
  have hpeak : 1 < Real.sqrt (contrastSq spectrum sharp bulk 1) := by
    have hnonneg : 0 ≤ contrastSq spectrum sharp bulk 1 := by linarith
    have hsqrt := Real.sqrt_nonneg (contrastSq spectrum sharp bulk 1)
    have hsquare := Real.sq_sqrt hnonneg
    nlinarith
  have hstrong_one : Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
      atTop (𝓝 1) := by simpa [hsumS, hsumB] using hstrong
  have hweak_one : Tendsto (fun γ : ℝ => Real.sqrt (contrastSq spectrum sharp bulk γ))
      (𝓝[Set.Ioi 0] 0) (𝓝 1) := by simpa [hkernelS, hkernelB] using hweak
  exact ⟨spectrum, sharp, bulk, hspectrum, hsharp, hbulk, hstrong_one,
    hweak_one, hvalue, hlarge, hpeak,
    peak_with_equal_endpoint_limits_is_nonmonotone _ hstrong_one hweak_one hpeak⟩

end PaperReliability
