import PaperReliability.ModalModel

open scoped BigOperators

namespace PaperReliability

/-- Lower endpoint envelope for a positive modal variance. -/
noncomputable def lowerVarianceEnvelope (l h gamma ql qh : ℝ) : ℝ :=
  let t := (gamma - l) / (h - l)
  ((1 - t) * l * ql + t * h * qh) / gamma

/-- Harmonic upper endpoint envelope for a positive modal variance. -/
noncomputable def upperVarianceEnvelope (l h gamma ql qh : ℝ) : ℝ :=
  let t := (gamma - l) / (h - l)
  ((1 - t) / ql + t / qh)⁻¹

private theorem modal_pos {ι : Type*} [Fintype ι]
    (lambda a : ι → ℝ) (hlambda : ∀ i, 0 ≤ lambda i)
    (ha : ∀ i, 0 ≤ a i) (ha_pos : ∃ i, 0 < a i)
    {gamma : ℝ} (hgamma : 0 < gamma) :
    0 < modalVariance lambda a gamma := by
  unfold modalVariance
  apply Finset.sum_pos'
  · intro i _
    exact div_nonneg (ha i) (by linarith [hlambda i])
  · obtain ⟨i, hi⟩ := ha_pos
    exact ⟨i, Finset.mem_univ i, div_pos hi (by linarith [hlambda i])⟩

private theorem mode_lower {lambda l h gamma t : ℝ}
    (hlambda : 0 ≤ lambda) (hl : 0 < l) (hh : 0 < h)
    (hgamma : 0 < gamma) (ht : 0 ≤ t) (ht' : t ≤ 1)
    (heq : gamma = (1 - t) * l + t * h) :
    (1 - t) * l / (lambda + l) + t * h / (lambda + h) ≤
      gamma / (lambda + gamma) := by
  have hdl : 0 < lambda + l := by linarith
  have hdh : 0 < lambda + h := by linarith
  have hdg : 0 < lambda + gamma := by linarith
  have identity :
      gamma / (lambda + gamma) -
        ((1 - t) * l / (lambda + l) + t * h / (lambda + h)) =
      lambda * t * (1 - t) * (h - l) ^ 2 /
        ((lambda + gamma) * (lambda + l) * (lambda + h)) := by
    field_simp [hdl.ne', hdh.ne', hdg.ne']
    rw [heq]
    ring
  have hdefect : 0 ≤ lambda * t * (1 - t) * (h - l) ^ 2 /
      ((lambda + gamma) * (lambda + l) * (lambda + h)) := by positivity
  linarith

private theorem weight_properties {l h gamma : ℝ} (hlh : l < h)
    (hlg : l ≤ gamma) (hgh : gamma ≤ h) :
    0 ≤ (gamma - l) / (h - l) ∧ (gamma - l) / (h - l) ≤ 1 ∧
      gamma = (1 - (gamma - l) / (h - l)) * l +
        ((gamma - l) / (h - l)) * h := by
  have hd : 0 < h - l := sub_pos.mpr hlh
  refine ⟨div_nonneg (sub_nonneg.mpr hlg) hd.le, ?_, ?_⟩
  · apply (div_le_one hd).2
    linarith
  · field_simp [hd.ne']
    ring

/-- Positive finite modal variances lie between a weighted lower chord and a
harmonic upper chord. The hypotheses allow zero eigenvalues and zero weights. -/
theorem positive_modal_variance_enclosure {ι : Type*} [Fintype ι]
    (lambda a : ι → ℝ) (hlambda : ∀ i, 0 ≤ lambda i)
    (ha : ∀ i, 0 ≤ a i) (ha_pos : ∃ i, 0 < a i)
    {l h gamma : ℝ} (hl : 0 < l) (hlh : l < h)
    (hlg : l ≤ gamma) (hgh : gamma ≤ h) :
    lowerVarianceEnvelope l h gamma (modalVariance lambda a l)
        (modalVariance lambda a h) ≤ modalVariance lambda a gamma ∧
      modalVariance lambda a gamma ≤
        upperVarianceEnvelope l h gamma (modalVariance lambda a l)
          (modalVariance lambda a h) := by
  let t := (gamma - l) / (h - l)
  obtain ⟨ht, ht', heq⟩ := weight_properties hlh hlg hgh
  change 0 ≤ t at ht
  change t ≤ 1 at ht'
  change gamma = (1 - t) * l + t * h at heq
  have hh : 0 < h := lt_trans hl hlh
  have hg : 0 < gamma := lt_of_lt_of_le hl hlg
  have hql : 0 < modalVariance lambda a l := modal_pos lambda a hlambda ha ha_pos hl
  have hqh : 0 < modalVariance lambda a h := modal_pos lambda a hlambda ha ha_pos hh
  have hqg : 0 < modalVariance lambda a gamma := modal_pos lambda a hlambda ha ha_pos hg
  constructor
  · change ((1 - t) * l * modalVariance lambda a l +
      t * h * modalVariance lambda a h) / gamma ≤ modalVariance lambda a gamma
    apply (div_le_iff₀ hg).2
    have H := Finset.sum_le_sum (s := Finset.univ) fun i _ =>
      mul_le_mul_of_nonneg_left
        (mode_lower (hlambda i) hl hh hg ht ht' heq) (ha i)
    have hleft : (1 - t) * l * modalVariance lambda a l +
        t * h * modalVariance lambda a h =
        ∑ i, a i * ((1 - t) * l / (lambda i + l) + t * h / (lambda i + h)) := by
      unfold modalVariance
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      ring
    have hright : modalVariance lambda a gamma * gamma =
        ∑ i, a i * (gamma / (lambda i + gamma)) := by
      unfold modalVariance
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i _
      ring
    rw [hleft, hright]
    exact H
  · let el := ∑ i, a i * (lambda i + l) / (lambda i + gamma) ^ 2
    let eh := ∑ i, a i * (lambda i + h) / (lambda i + gamma) ^ 2
    have csl : modalVariance lambda a gamma ^ 2 ≤ modalVariance lambda a l * el := by
      apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul
        (s := Finset.univ) (r := fun i => a i / (lambda i + gamma))
        (f := fun i => a i / (lambda i + l))
        (g := fun i => a i * (lambda i + l) / (lambda i + gamma) ^ 2)
      · intro i _; exact div_nonneg (ha i) (by linarith [hlambda i])
      · intro i _
        exact div_nonneg (mul_nonneg (ha i) (by linarith [hlambda i])) (sq_nonneg _)
      · intro i _
        have hdl : lambda i + l ≠ 0 := ne_of_gt (by linarith [hlambda i])
        have hdg : lambda i + gamma ≠ 0 := ne_of_gt (by linarith [hlambda i])
        apply le_of_eq
        field_simp [hdl, hdg]
    have csh : modalVariance lambda a gamma ^ 2 ≤ modalVariance lambda a h * eh := by
      apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul
        (s := Finset.univ) (r := fun i => a i / (lambda i + gamma))
        (f := fun i => a i / (lambda i + h))
        (g := fun i => a i * (lambda i + h) / (lambda i + gamma) ^ 2)
      · intro i _; exact div_nonneg (ha i) (by linarith [hlambda i])
      · intro i _
        exact div_nonneg (mul_nonneg (ha i) (by linarith [hlambda i])) (sq_nonneg _)
      · intro i _
        have hdh : lambda i + h ≠ 0 := ne_of_gt (by linarith [hlambda i])
        have hdg : lambda i + gamma ≠ 0 := ne_of_gt (by linarith [hlambda i])
        apply le_of_eq
        field_simp [hdh, hdg]
    have hlenergy : modalVariance lambda a gamma ^ 2 / modalVariance lambda a l ≤ el :=
      (div_le_iff₀ hql).2 (by simpa [mul_comm] using csl)
    have hhenergy : modalVariance lambda a gamma ^ 2 / modalVariance lambda a h ≤ eh :=
      (div_le_iff₀ hqh).2 (by simpa [mul_comm] using csh)
    have energy_identity : (1 - t) * el + t * eh = modalVariance lambda a gamma := by
      dsimp [el, eh, modalVariance]
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      have hdg : lambda i + gamma ≠ 0 := ne_of_gt (by linarith [hlambda i])
      field_simp [hdg]
      rw [heq]
      ring
    have hmain : ((1 - t) / modalVariance lambda a l + t / modalVariance lambda a h) *
        modalVariance lambda a gamma ^ 2 ≤ modalVariance lambda a gamma := by
      calc
        _ = (1 - t) * (modalVariance lambda a gamma ^ 2 / modalVariance lambda a l) +
            t * (modalVariance lambda a gamma ^ 2 / modalVariance lambda a h) := by ring
        _ ≤ (1 - t) * el + t * eh :=
          add_le_add (mul_le_mul_of_nonneg_left hlenergy (sub_nonneg.mpr ht'))
            (mul_le_mul_of_nonneg_left hhenergy ht)
        _ = _ := energy_identity
    have hchord : 0 < (1 - t) / modalVariance lambda a l + t / modalVariance lambda a h := by
      have hsplit : 0 < 1 - t ∨ 0 < t := by
        by_cases ht0 : 0 < t
        · exact Or.inr ht0
        · left; linarith
      rcases hsplit with hw | htpos <;> positivity
    change modalVariance lambda a gamma ≤
      ((1 - t) / modalVariance lambda a l + t / modalVariance lambda a h)⁻¹
    rw [inv_eq_one_div]
    apply (le_div_iff₀ hchord).2
    apply (mul_le_mul_iff_left₀ hqg).mp
    simpa [pow_two, mul_comm, mul_left_comm, mul_assoc] using hmain

private theorem lower_expanded {l h gamma ql qh : ℝ}
    (hlh : l < h) (hgamma : 0 < gamma) :
    lowerVarianceEnvelope l h gamma ql qh =
      (h * qh - l * ql) / (h - l) +
        (l * h * (ql - qh) / (h - l)) / gamma := by
  unfold lowerVarianceEnvelope
  field_simp [ne_of_gt (sub_pos.mpr hlh), hgamma.ne']
  ring

private theorem lower_antitone {l h u v ql qh : ℝ}
    (hl : 0 < l) (hlh : l < h) (hu : 0 < u) (huv : u ≤ v)
    (hqlqh : qh ≤ ql) :
    lowerVarianceEnvelope l h v ql qh ≤ lowerVarianceEnvelope l h u ql qh := by
  have hv : 0 < v := lt_of_lt_of_le hu huv
  rw [lower_expanded hlh hu, lower_expanded hlh hv]
  have hc : 0 ≤ l * h * (ql - qh) / (h - l) := by
    have hh : 0 < h := lt_trans hl hlh
    positivity
  exact add_le_add le_rfl (div_le_div_of_nonneg_left hc hu huv)

private theorem upper_antitone {l h u v ql qh : ℝ}
    (hlh : l < h) (hlu : l ≤ u) (huv : u ≤ v) (hvh : v ≤ h)
    (hql : 0 < ql) (hqh : 0 < qh) (hqlqh : qh ≤ ql) :
    upperVarianceEnvelope l h v ql qh ≤ upperVarianceEnvelope l h u ql qh := by
  let tu := (u - l) / (h - l)
  let tv := (v - l) / (h - l)
  obtain ⟨htu, htu', _⟩ := weight_properties hlh hlu (huv.trans hvh)
  obtain ⟨htv, htv', _⟩ := weight_properties hlh (hlu.trans huv) hvh
  change 0 ≤ tu at htu
  change tu ≤ 1 at htu'
  change 0 ≤ tv at htv
  change tv ≤ 1 at htv'
  have htt : tu ≤ tv := by
    apply div_le_div_of_nonneg_right (by linarith) (sub_nonneg.mpr hlh.le)
  have hinv : 1 / ql ≤ 1 / qh := one_div_le_one_div_of_le hqh hqlqh
  have hden : (1 - tu) / ql + tu / qh ≤ (1 - tv) / ql + tv / qh := by
    have hp := mul_nonneg (sub_nonneg.mpr htt) (sub_nonneg.mpr hinv)
    calc
      _ ≤ (1 - tu) / ql + tu / qh + (tv - tu) * (1 / qh - 1 / ql) :=
        le_add_of_nonneg_right hp
      _ = _ := by ring
  have hpos : 0 < (1 - tu) / ql + tu / qh := by
    have hsplit : 0 < 1 - tu ∨ 0 < tu := by
      by_cases hp : 0 < tu
      · exact Or.inr hp
      · left; linarith
    rcases hsplit with hw | ht <;> positivity
  change ((1 - tv) / ql + tv / qh)⁻¹ ≤ ((1 - tu) / ql + tu / qh)⁻¹
  simpa [one_div] using one_div_le_one_div_of_le hpos hden

private theorem lower_pos {l h gamma ql qh : ℝ}
    (hl : 0 < l) (hlh : l < h) (hlg : l ≤ gamma) (hgh : gamma ≤ h)
    (hql : 0 < ql) (hqh : 0 < qh) :
    0 < lowerVarianceEnvelope l h gamma ql qh := by
  let t := (gamma - l) / (h - l)
  obtain ⟨ht, ht', _⟩ := weight_properties hlh hlg hgh
  change 0 ≤ t at ht
  change t ≤ 1 at ht'
  have hh : 0 < h := lt_trans hl hlh
  have hg : 0 < gamma := lt_of_lt_of_le hl hlg
  have hsplit : 0 < 1 - t ∨ 0 < t := by
    by_cases hp : 0 < t
    · exact Or.inr hp
    · left; linarith
  change 0 < ((1 - t) * l * ql + t * h * qh) / gamma
  rcases hsplit with hw | htpos <;> positivity

/-- Saved endpoint variances enclose every pointwise modal variance throughout
a retained precision bracket. The bracket may contain a contrast-matching root;
no uniqueness or numerical root certificate is assumed by this analytic result. -/
theorem variance_enclosure_on_precision_bracket {ι : Type*} [Fintype ι]
    (lambda energy : ι → ℝ) (hlambda : ∀ i, 0 ≤ lambda i)
    (henergy : ∀ i, 0 ≤ energy i) (henergy_pos : ∃ i, 0 < energy i)
    {l h a b gamma : ℝ} (hl : 0 < l) (hlh : l < h)
    (hla : l ≤ a) (hag : a ≤ gamma) (hgb : gamma ≤ b) (hbh : b ≤ h) :
    0 < lowerVarianceEnvelope l h b (modalVariance lambda energy l)
        (modalVariance lambda energy h) ∧
      lowerVarianceEnvelope l h b (modalVariance lambda energy l)
        (modalVariance lambda energy h) ≤ modalVariance lambda energy gamma ∧
      modalVariance lambda energy gamma ≤
        upperVarianceEnvelope l h a (modalVariance lambda energy l)
          (modalVariance lambda energy h) := by
  have hh : 0 < h := lt_trans hl hlh
  have hg : 0 < gamma := lt_of_lt_of_le hl (hla.trans hag)
  have hql := modal_pos lambda energy hlambda henergy henergy_pos hl
  have hqh := modal_pos lambda energy hlambda henergy henergy_pos hh
  have hordered : modalVariance lambda energy h ≤ modalVariance lambda energy l := by
    unfold modalVariance
    apply Finset.sum_le_sum
    intro i _
    exact div_le_div_of_nonneg_left (henergy i) (by linarith [hlambda i]) (by linarith)
  obtain ⟨hlo, hhi⟩ := positive_modal_variance_enclosure lambda energy hlambda henergy
    henergy_pos hl hlh (hla.trans hag) (hgb.trans hbh)
  refine ⟨lower_pos hl hlh (hla.trans (hag.trans hgb)) hbh hql hqh, ?_, ?_⟩
  · exact (lower_antitone hl hlh hg hgb hordered).trans hlo
  · exact hhi.trans (upper_antitone hlh hla hag (hgb.trans hbh) hql hqh hordered)

end PaperReliability
