import PaperReliability.VarianceEnvelopes

open scoped BigOperators

namespace PaperReliability

private noncomputable def subsetMaximum {β : Type*} (s : Finset β)
    (f : β → ℝ) : ℝ :=
  if hs : s.Nonempty then s.sup' hs f else 0

/-- The zero-based `rank`th observation, equivalently the minimum of the
maxima of all subsets of size `rank+1`. This retains multiplicities through
the observation indices and represents the empirical higher quantile. -/
noncomputable def higherQuantile {β : Type*} [DecidableEq β]
    (sample : Finset β) (f : β → ℝ) (rank : ℕ) (hrank : rank < sample.card) : ℝ :=
  (sample.powersetCard (rank + 1)).inf'
    (Finset.powersetCard_nonempty.mpr (by omega)) (fun s => subsetMaximum s f)

/-- Higher empirical quantiles preserve pointwise order, including tied data. -/
private theorem higherQuantile_mono {β : Type*} [DecidableEq β]
    (sample : Finset β) {f g : β → ℝ} (rank : ℕ) (hrank : rank < sample.card)
    (hfg : ∀ x ∈ sample, f x ≤ g x) :
    higherQuantile sample f rank hrank ≤ higherQuantile sample g rank hrank := by
  unfold higherQuantile
  apply Finset.inf'_mono_fun
  intro s hs
  have hsubset := (Finset.mem_powersetCard.mp hs).1
  unfold subsetMaximum
  split_ifs with hn
  · exact Finset.sup'_mono_fun fun x hx => hfg x (hsubset hx)
  · exact le_rfl

/-- The empirical order statistic is below a threshold exactly when at least
`rank+1` sample observations are at or below that threshold. -/
private theorem higherQuantile_le_iff_count {β : Type*} [DecidableEq β]
    (sample : Finset β) (f : β → ℝ) (rank : ℕ) (hrank : rank < sample.card)
    (c : ℝ) :
    higherQuantile sample f rank hrank ≤ c ↔
      rank + 1 ≤ (sample.filter fun x => f x ≤ c).card := by
  classical
  unfold higherQuantile
  rw [Finset.inf'_le_iff]
  constructor
  · rintro ⟨s, hs, hmax⟩
    obtain ⟨hsubset, hcard⟩ := Finset.mem_powersetCard.mp hs
    have hn : s.Nonempty := Finset.card_pos.mp (by omega)
    have hpoint : ∀ x ∈ s, f x ≤ c := by
      simpa [subsetMaximum, hn] using hmax
    have hfilter : s ⊆ sample.filter (fun x => f x ≤ c) := by
      intro x hx
      exact Finset.mem_filter.mpr ⟨hsubset hx, hpoint x hx⟩
    simpa [hcard] using Finset.card_le_card hfilter
  · intro hcount
    obtain ⟨s, hs, hcard⟩ := Finset.exists_subset_card_eq hcount
    have hn : s.Nonempty := Finset.card_pos.mp (by omega)
    refine ⟨s, Finset.mem_powersetCard.mpr ⟨?_, hcard⟩, ?_⟩
    · exact hs.trans (Finset.filter_subset _ _)
    · unfold subsetMaximum
      rw [dif_pos hn]
      apply Finset.sup'_le
      intro x hx
      exact (Finset.mem_filter.mp (hs hx)).2

private theorem quantile_nonneg {β : Type*} [DecidableEq β]
    (sample : Finset β) (f : β → ℝ) (rank : ℕ) (hrank : rank < sample.card)
    (hf : ∀ x ∈ sample, 0 ≤ f x) :
    0 ≤ higherQuantile sample f rank hrank := by
  classical
  have hcount := (higherQuantile_le_iff_count sample f rank hrank
    (higherQuantile sample f rank hrank)).mp le_rfl
  have hn : (sample.filter fun x => f x ≤ higherQuantile sample f rank hrank).Nonempty :=
    Finset.card_pos.mp (by omega)
  obtain ⟨x, hx⟩ := hn
  obtain ⟨hxs, hxq⟩ := Finset.mem_filter.mp hx
  exact (hf x hxs).trans hxq

/-- Empirical coverage on a finite region at a nonnegative interval multiplier. -/
noncomputable def empiricalCoverage {β : Type*} (sample : Finset β)
    (error width : β → ℝ) (multiplier : ℝ) : ℝ := by
  classical
  exact ((sample.filter fun x => |error x| ≤ multiplier * width x).card : ℝ) /
    sample.card

private theorem coverage_mono_width {β : Type*} (sample : Finset β)
    (error lower upper : β → ℝ)
    (hlower : ∀ x ∈ sample, lower x ≤ upper x) :
    empiricalCoverage sample error lower 1 ≤ empiricalCoverage sample error upper 1 := by
  classical
  unfold empiricalCoverage
  apply div_le_div_of_nonneg_right
  · exact_mod_cast Finset.card_le_card (show
      sample.filter (fun x => |error x| ≤ 1 * lower x) ⊆
        sample.filter (fun x => |error x| ≤ 1 * upper x) from by
      intro x hx
      obtain ⟨hxs, hxe⟩ := Finset.mem_filter.mp hx
      exact Finset.mem_filter.mpr ⟨hxs, by simpa using hxe.trans (by simpa using hlower x hxs)⟩)
  · positivity

private theorem calibrated_coverage_from_enclosures {β : Type*} [DecidableEq β]
    (bulk sharp : Finset β) (error low q upper : β → ℝ)
    (rank : ℕ) (hrank : rank < bulk.card) {z : ℝ} (hz : 0 ≤ z)
    (hbounds : ∀ x ∈ bulk ∪ sharp, 0 < low x ∧ low x ≤ q x ∧ q x ≤ upper x) :
    empiricalCoverage sharp error (fun x => Real.sqrt (low x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (upper x)) rank hrank) ≤
      empiricalCoverage sharp error (fun x => Real.sqrt (q x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (q x)) rank hrank) ∧
      empiricalCoverage sharp error (fun x => Real.sqrt (q x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (q x)) rank hrank) ≤
      empiricalCoverage sharp error (fun x => Real.sqrt (upper x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (low x)) rank hrank) := by
  let kmin := higherQuantile bulk (fun x => |error x| / Real.sqrt (upper x)) rank hrank
  let kq := higherQuantile bulk (fun x => |error x| / Real.sqrt (q x)) rank hrank
  let kmax := higherQuantile bulk (fun x => |error x| / Real.sqrt (low x)) rank hrank
  have hkmin : kmin ≤ kq := by
    apply higherQuantile_mono
    intro x hx
    obtain ⟨hlo, hloq, hqu⟩ := hbounds x (Finset.mem_union_left sharp hx)
    exact div_le_div_of_nonneg_left (abs_nonneg _) (Real.sqrt_pos.mpr (lt_of_lt_of_le hlo hloq))
      (Real.sqrt_le_sqrt hqu)
  have hkmax : kq ≤ kmax := by
    apply higherQuantile_mono
    intro x hx
    obtain ⟨hlo, hloq, _⟩ := hbounds x (Finset.mem_union_left sharp hx)
    exact div_le_div_of_nonneg_left (abs_nonneg _) (Real.sqrt_pos.mpr hlo)
      (Real.sqrt_le_sqrt hloq)
  have hkq0 : 0 ≤ kq := by
    apply quantile_nonneg
    intro x hx
    exact div_nonneg (abs_nonneg _) (Real.sqrt_nonneg _)
  have hkmax0 : 0 ≤ kmax := hkq0.trans hkmax
  constructor
  · have hwidth : ∀ x ∈ sharp, (z * kmin) * Real.sqrt (low x) ≤
        (z * kq) * Real.sqrt (q x) := by
      intro x hx
      obtain ⟨_, hloq, _⟩ := hbounds x (Finset.mem_union_right bulk hx)
      calc
        _ ≤ (z * kq) * Real.sqrt (low x) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hkmin hz)
            (Real.sqrt_nonneg _)
        _ ≤ _ := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hloq) (mul_nonneg hz hkq0)
    simpa only [empiricalCoverage, one_mul] using
      coverage_mono_width sharp error (fun x => (z * kmin) * Real.sqrt (low x))
        (fun x => (z * kq) * Real.sqrt (q x)) hwidth
  · have hwidth : ∀ x ∈ sharp, (z * kq) * Real.sqrt (q x) ≤
        (z * kmax) * Real.sqrt (upper x) := by
      intro x hx
      obtain ⟨_, _, hqu⟩ := hbounds x (Finset.mem_union_right bulk hx)
      calc
        _ ≤ (z * kmax) * Real.sqrt (q x) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hkmax hz)
            (Real.sqrt_nonneg _)
        _ ≤ _ := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hqu) (mul_nonneg hz hkmax0)
    simpa only [empiricalCoverage, one_mul] using
      coverage_mono_width sharp error (fun x => (z * kq) * Real.sqrt (q x))
        (fun x => (z * kmax) * Real.sqrt (upper x)) hwidth

/-- The positive modal endpoint envelopes propagate through the empirical bulk
higher quantile to enclose sharp-region coverage at every retained precision.
Errors are fixed across precision, as required for the trained-centre oracle. -/
theorem calibrated_modal_coverage_enclosure {ι β : Type*}
    [Fintype ι] [DecidableEq β]
    (bulk sharp : Finset β) (lambda : ι → ℝ) (energy : β → ι → ℝ)
    (error : β → ℝ) (hlambda : ∀ i, 0 ≤ lambda i)
    (henergy : ∀ x ∈ bulk ∪ sharp, ∀ i, 0 ≤ energy x i)
    (henergy_pos : ∀ x ∈ bulk ∪ sharp, ∃ i, 0 < energy x i)
    (rank : ℕ) (hrank : rank < bulk.card)
    {l h a b gamma z : ℝ} (hl : 0 < l) (hlh : l < h)
    (hla : l ≤ a) (hag : a ≤ gamma) (hgb : gamma ≤ b) (hbh : b ≤ h)
    (hz : 0 ≤ z) :
    let ql := fun x => modalVariance lambda (energy x) l
    let qh := fun x => modalVariance lambda (energy x) h
    let low := fun x => lowerVarianceEnvelope l h b (ql x) (qh x)
    let upper := fun x => upperVarianceEnvelope l h a (ql x) (qh x)
    let q := fun x => modalVariance lambda (energy x) gamma
    empiricalCoverage sharp error (fun x => Real.sqrt (low x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (upper x)) rank hrank) ≤
      empiricalCoverage sharp error (fun x => Real.sqrt (q x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (q x)) rank hrank) ∧
      empiricalCoverage sharp error (fun x => Real.sqrt (q x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (q x)) rank hrank) ≤
      empiricalCoverage sharp error (fun x => Real.sqrt (upper x))
        (z * higherQuantile bulk (fun x => |error x| / Real.sqrt (low x)) rank hrank) := by
  dsimp only
  apply calibrated_coverage_from_enclosures bulk sharp error _ _ _ rank hrank hz
  intro x hx
  exact variance_enclosure_on_precision_bracket lambda (energy x) hlambda
    (henergy x hx) (henergy_pos x hx) hl hlh hla hag hgb hbh

end PaperReliability
