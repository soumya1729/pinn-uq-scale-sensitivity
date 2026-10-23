import PaperReliability.ModalModel

open scoped BigOperators
open Classical

namespace PaperReliability

/-- Likelihood contribution to the affine parameter shift. Exact zero singular modes
have zero likelihood shift; prior-only shrinkage is not included. -/
noncomputable def likelihoodShift {ι : Type*} (singular residual : ι → ℝ)
    (precision : ℝ) (i : ι) : ℝ :=
  -(singular i * residual i) / (singular i ^ 2 + precision)

/-- Prior-shrinkage contribution for a zero-mean prior. It is absent when the
prior is centred at the trained parameters. -/
noncomputable def priorShrinkage {ι : Type*} (trained singular : ι → ℝ)
    (precision : ℝ) (i : ι) : ℝ :=
  -precision * trained i / (singular i ^ 2 + precision)

/-- Modal affine shift for the zero-mean prior, split into likelihood and prior
contributions. -/
noncomputable def zeroMeanShift {ι : Type*} (trained singular residual : ι → ℝ)
    (precision : ℝ) (i : ι) : ℝ :=
  likelihoodShift singular residual precision i + priorShrinkage trained singular precision i

/-- Projected residual energy in the precision metric. -/
noncomputable def likelihoodEnergy {ι : Type*} [Fintype ι]
    (singular residual : ι → ℝ) (precision : ℝ) : ℝ :=
  ∑ i, singular i ^ 2 * residual i ^ 2 / (singular i ^ 2 + precision)

/-- Output shift of the theta-centred affine Gaussian. -/
noncomputable def predictiveShift {ι : Type*} [Fintype ι]
    (tangent singular residual : ι → ℝ) (precision : ℝ) : ℝ :=
  ∑ i, tangent i * likelihoodShift singular residual precision i

private theorem denominator_pos {ι : Type*} (singular : ι → ℝ)
    {precision : ℝ} (hp : 0 < precision) (i : ι) :
    0 < singular i ^ 2 + precision := by
  nlinarith [sq_nonneg (singular i)]

private theorem variance_nonneg {ι : Type*} [Fintype ι]
    (tangent singular : ι → ℝ) {precision : ℝ} (hp : 0 < precision) :
    0 ≤ modalVariance (fun i => singular i ^ 2) (fun i => tangent i ^ 2) precision := by
  exact Finset.sum_nonneg fun i _ => div_nonneg (sq_nonneg _) (denominator_pos singular hp i).le

private theorem energy_nonneg {ι : Type*} [Fintype ι]
    (singular residual : ι → ℝ) {precision : ℝ} (hp : 0 < precision) :
    0 ≤ likelihoodEnergy singular residual precision := by
  exact Finset.sum_nonneg fun i _ => div_nonneg (mul_nonneg (sq_nonneg _) (sq_nonneg _))
    (denominator_pos singular hp i).le

/-- Covariance-metric Cauchy--Schwarz for the actual modal likelihood shift.
It remains valid for exact kernel directions, including a QR complement encoded
by zero singular values. This theorem certifies the analytic inequality, not
floating-point values or a numerical run count. -/
private theorem likelihood_predictive_shift_sq_bound {ι : Type*} [Fintype ι]
    (tangent singular residual : ι → ℝ) {precision : ℝ} (hp : 0 < precision) :
    predictiveShift tangent singular residual precision ^ 2 ≤
      modalVariance (fun i => singular i ^ 2) (fun i => tangent i ^ 2) precision *
        likelihoodEnergy singular residual precision := by
  have hf : ∀ i ∈ (Finset.univ : Finset ι),
      0 ≤ tangent i ^ 2 / (singular i ^ 2 + precision) :=
    fun i _ => div_nonneg (sq_nonneg _) (denominator_pos singular hp i).le
  have hg : ∀ i ∈ (Finset.univ : Finset ι),
      0 ≤ singular i ^ 2 * residual i ^ 2 / (singular i ^ 2 + precision) :=
    fun i _ => div_nonneg (mul_nonneg (sq_nonneg _) (sq_nonneg _))
      (denominator_pos singular hp i).le
  have hi : ∀ i ∈ (Finset.univ : Finset ι),
      (tangent i * likelihoodShift singular residual precision i) ^ 2 ≤
        (tangent i ^ 2 / (singular i ^ 2 + precision)) *
          (singular i ^ 2 * residual i ^ 2 / (singular i ^ 2 + precision)) := by
    intro i _
    apply le_of_eq
    simp only [likelihoodShift]
    ring
  exact Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul Finset.univ hf hg hi

/-- In every exact kernel mode the likelihood contributes zero; the zero-mean
prior removes the trained kernel prediction at every positive precision. This
statement concerns the whole exact kernel, not a numerical rank truncation. -/
theorem kernel_prediction_shift_attribution {ι : Type*} [Fintype ι]
    (trained tangent singular residual : ι → ℝ) {precision : ℝ} (hp : 0 < precision) :
    (∑ i ∈ Finset.univ.filter (fun i => singular i = 0),
      tangent i * likelihoodShift singular residual precision i) = 0 ∧
    (∑ i ∈ Finset.univ.filter (fun i => singular i = 0),
      tangent i * zeroMeanShift trained singular residual precision i) =
      -(∑ i ∈ Finset.univ.filter (fun i => singular i = 0), tangent i * trained i) := by
  classical
  constructor
  · apply Finset.sum_eq_zero
    intro i hi
    have hzero := (Finset.mem_filter.mp hi).2
    simp [likelihoodShift, hzero]
  · rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    have hzero := (Finset.mem_filter.mp hi).2
    simp [zeroMeanShift, likelihoodShift, priorShrinkage, hzero]
    field_simp [hp.ne']

/-- Theta-centred output displacement is at most sigma times sqrt(K / v).
The prior and noise enter covariance through gamma = alpha v; the separate
noise multiplier is retained explicitly. -/
theorem likelihood_predictive_shift_bound {ι : Type*} [Fintype ι]
    (tangent singular residual : ι → ℝ) {precision noise : ℝ}
    (hp : 0 < precision) (hv : 0 < noise) :
    |predictiveShift tangent singular residual precision| ≤
      Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
        (fun i => tangent i ^ 2) precision) *
      Real.sqrt (likelihoodEnergy singular residual precision / noise) := by
  have hq := variance_nonneg tangent singular hp
  have hk := energy_nonneg singular residual hp
  have hcs := likelihood_predictive_shift_sq_bound tangent singular residual hp
  have h1 := Real.sq_sqrt (mul_nonneg hv.le hq)
  have h2 := Real.sq_sqrt (div_nonneg hk hv.le)
  have halg : (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent i ^ 2) precision) *
      (likelihoodEnergy singular residual precision / noise) =
      modalVariance (fun i => singular i ^ 2) (fun i => tangent i ^ 2) precision *
      likelihoodEnergy singular residual precision := by field_simp
  have hn : 0 ≤ Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent i ^ 2) precision) *
      Real.sqrt (likelihoodEnergy singular residual precision / noise) := by positivity
  have hs : (Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent i ^ 2) precision) *
      Real.sqrt (likelihoodEnergy singular residual precision / noise)) ^ 2 =
      modalVariance (fun i => singular i ^ 2) (fun i => tangent i ^ 2) precision *
      likelihoodEnergy singular residual precision := by
    rw [mul_pow, h1, h2]
    exact halg
  calc
    |predictiveShift tangent singular residual precision| =
      Real.sqrt (predictiveShift tangent singular residual precision ^ 2) :=
        (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (modalVariance (fun i => singular i ^ 2)
        (fun i => tangent i ^ 2) precision * likelihoodEnergy singular residual precision) :=
      Real.sqrt_le_sqrt hcs
    _ = Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
        (fun i => tangent i ^ 2) precision) *
        Real.sqrt (likelihoodEnergy singular residual precision / noise) := by
      rw [← hs, Real.sqrt_sq hn]

/-- RMS on a finite region. The region is the finite index type itself. -/
noncomputable def regionalRMS {X : Type*} [Fintype X] (field : X → ℝ) : ℝ :=
  Real.sqrt (∑ x, field x ^ 2) / Real.sqrt (Fintype.card X : ℝ)

private theorem rms_nonneg {X : Type*} [Fintype X] (f : X → ℝ) :
    0 ≤ regionalRMS f := by unfold regionalRMS; positivity

private theorem rms_norm {X : Type*} [Fintype X] (f : X → ℝ) :
    regionalRMS f = ‖(WithLp.toLp 2 f : EuclideanSpace ℝ X)‖ /
      Real.sqrt (Fintype.card X : ℝ) := by
  rw [EuclideanSpace.norm_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  rfl

private theorem rms_triangle {X : Type*} [Fintype X] (f g : X → ℝ) :
    regionalRMS (fun x => f x + g x) ≤ regionalRMS f + regionalRMS g := by
  simp only [rms_norm]
  have heq : (WithLp.toLp 2 (fun x => f x + g x) : EuclideanSpace ℝ X) =
      WithLp.toLp 2 f + WithLp.toLp 2 g := rfl
  rw [heq, ← add_div]
  exact div_le_div_of_nonneg_right (norm_add_le _ _) (Real.sqrt_nonneg _)

private theorem rms_neg {X : Type*} [Fintype X] (f : X → ℝ) :
    regionalRMS (fun x => -f x) = regionalRMS f := by
  simp [regionalRMS]

private theorem rms_pointwise {X : Type*} [Fintype X]
    (field width : X → ℝ) {c : ℝ} (hc : 0 ≤ c) (hw : ∀ x, 0 ≤ width x)
    (hb : ∀ x, |field x| ≤ c * width x) :
    regionalRMS field ≤ c * regionalRMS width := by
  have hs : (∑ x, field x ^ 2) ≤ c ^ 2 * (∑ x, width x ^ 2) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro x _
    have ht := hb x
    have hn := mul_nonneg hc (hw x)
    simpa only [sq_abs, mul_pow] using (sq_le_sq₀ (abs_nonneg (field x)) hn).2 ht
  have hroot : Real.sqrt (∑ x, field x ^ 2) ≤ c * Real.sqrt (∑ x, width x ^ 2) := by
    have h := Real.sqrt_le_sqrt hs
    rwa [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc] at h

  unfold regionalRMS
  calc
    Real.sqrt (∑ x, field x ^ 2) / Real.sqrt (Fintype.card X : ℝ) ≤
      (c * Real.sqrt (∑ x, width x ^ 2)) / Real.sqrt (Fintype.card X : ℝ) :=
        div_le_div_of_nonneg_right hroot (Real.sqrt_nonneg _)
    _ = c * (Real.sqrt (∑ x, width x ^ 2) / Real.sqrt (Fintype.card X : ℝ)) := by ring

private theorem rms_modal_minkowski {X ι : Type*} [Fintype X] [Fintype ι]
    (columns : X → ι → ℝ) (coefficient : ι → ℝ) :
    regionalRMS (fun x => ∑ i, coefficient i * columns x i) ≤
      ∑ i, |coefficient i| * regionalRMS (fun x => columns x i) := by
  simp only [rms_norm]
  have heq : (WithLp.toLp 2 (fun x => ∑ i, coefficient i * columns x i) :
      EuclideanSpace ℝ X) =
      ∑ i, coefficient i • (WithLp.toLp 2 (fun x => columns x i) : EuclideanSpace ℝ X) := by
    ext x
    simp
  rw [heq]
  have hn := norm_sum_le Finset.univ (fun i => coefficient i •
    (WithLp.toLp 2 (fun x => columns x i) : EuclideanSpace ℝ X))
  simp only [norm_smul, Real.norm_eq_abs] at hn
  calc
    ‖∑ i, coefficient i • (WithLp.toLp 2 (fun x => columns x i) : EuclideanSpace ℝ X)‖ /
        Real.sqrt (Fintype.card X : ℝ) ≤
      (∑ i, |coefficient i| * ‖(WithLp.toLp 2 (fun x => columns x i) : EuclideanSpace ℝ X)‖) /
        Real.sqrt (Fintype.card X : ℝ) :=
          div_le_div_of_nonneg_right hn (Real.sqrt_nonneg _)
    _ = ∑ i, |coefficient i| *
        (‖(WithLp.toLp 2 (fun x => columns x i) : EuclideanSpace ℝ X)‖ /
          Real.sqrt (Fintype.card X : ℝ)) := by rw [Finset.sum_div]; apply Finset.sum_congr rfl; intro i _; ring

/-- The theta-centred affine error is bracketed by the smaller of the pointwise
covariance-metric bound and the modal-column Minkowski bound. Both bounds are
derived from the actual likelihood shift and finite regional data. The signed
error is prediction minus reference, so recentering adds the predictive shift. -/
theorem likelihood_regional_error_bounds {X ι : Type*} [Fintype X] [Nonempty X] [Fintype ι]
    (error : X → ℝ) (tangent : X → ι → ℝ) (singular residual : ι → ℝ)
    {precision noise : ℝ} (hp : 0 < precision) (hv : 0 < noise) :
    let shift := fun x => predictiveShift (tangent x) singular residual precision
    let width := fun x => Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent x i ^ 2) precision)
    let c := Real.sqrt (likelihoodEnergy singular residual precision / noise)
    let B := min (c * regionalRMS width)
      (∑ i, |likelihoodShift singular residual precision i| *
        regionalRMS (fun x => tangent x i))
    max 0 (regionalRMS error - B) ≤ regionalRMS (fun x => error x + shift x) ∧
      regionalRMS (fun x => error x + shift x) ≤ regionalRMS error + B := by
  dsimp only
  let shift := fun x => predictiveShift (tangent x) singular residual precision
  let width := fun x => Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent x i ^ 2) precision)
  let c := Real.sqrt (likelihoodEnergy singular residual precision / noise)
  let B := min (c * regionalRMS width)
    (∑ i, |likelihoodShift singular residual precision i| * regionalRMS (fun x => tangent x i))
  have hs1 : regionalRMS shift ≤ c * regionalRMS width :=
    rms_pointwise shift width (Real.sqrt_nonneg _) (fun x => Real.sqrt_nonneg _)
      (fun x => by simpa [width, c, mul_comm] using
        likelihood_predictive_shift_bound (tangent x) singular residual hp hv)
  have hs2 : regionalRMS shift ≤ ∑ i, |likelihoodShift singular residual precision i| *
      regionalRMS (fun x => tangent x i) := by
    simpa [shift, predictiveShift, mul_comm] using
      rms_modal_minkowski tangent (likelihoodShift singular residual precision)
  have hs : regionalRMS shift ≤ B := le_min hs1 hs2
  have hu : regionalRMS (fun x => error x + shift x) ≤ regionalRMS error + regionalRMS shift :=
    rms_triangle error shift
  have hl : regionalRMS error ≤ regionalRMS (fun x => error x + shift x) + regionalRMS shift := by
    have h := rms_triangle (fun x => error x + shift x) (fun x => -shift x)
    simpa only [add_assoc, add_neg_cancel, add_zero, rms_neg] using h

  constructor
  · exact max_le (rms_nonneg _) (by linarith)
  · linarith

/-- Empirical coverage on a finite region, with the paper's closed inequality. -/
noncomputable def centreCoverage {X : Type*} [Fintype X]
    (error width : X → ℝ) (threshold : ℝ) : ℝ := by
  classical
  exact ((Finset.univ.filter fun x => |error x| ≤ threshold * width x).card : ℝ) /
    (Fintype.card X : ℝ)

/-- Finite-grid coverage brackets from a pointwise centre displacement.
The proof is by event inclusion and cardinality, without distributional
assumptions or independence assumptions. -/
private theorem centre_coverage_sandwich {X : Type*} [Fintype X] [Nonempty X]
    (error shift width : X → ℝ) {c threshold : ℝ}
    (_hc : 0 ≤ c) (_ht : c ≤ threshold) (_hw : ∀ x, 0 ≤ width x)
    (hs : ∀ x, |shift x| ≤ c * width x) :
    centreCoverage error width (threshold - c) ≤
      centreCoverage (fun x => error x + shift x) width threshold ∧
    centreCoverage (fun x => error x + shift x) width threshold ≤
      centreCoverage error width (threshold + c) := by
  classical
  have hlo : (Finset.univ.filter fun x => |error x| ≤ (threshold - c) * width x) ⊆
      (Finset.univ.filter fun x => |error x + shift x| ≤ threshold * width x) := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    calc |error x + shift x| ≤ |error x| + |shift x| := abs_add_le _ _
      _ ≤ (threshold - c) * width x + c * width x := add_le_add hx (hs x)
      _ = threshold * width x := by ring
  have hhi : (Finset.univ.filter fun x => |error x + shift x| ≤ threshold * width x) ⊆
      (Finset.univ.filter fun x => |error x| ≤ (threshold + c) * width x) := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    calc |error x| = |(error x + shift x) - shift x| := by rw [add_sub_cancel_right]
      _ ≤ |error x + shift x| + |shift x| := abs_sub _ _
      _ ≤ threshold * width x + c * width x := add_le_add hx (hs x)
      _ = (threshold + c) * width x := by ring
  constructor
  · exact div_le_div_of_nonneg_right (by exact_mod_cast Finset.card_le_card hlo) (by positivity)
  · exact div_le_div_of_nonneg_right (by exact_mod_cast Finset.card_le_card hhi) (by positivity)

/-- The covariance-metric likelihood bound implies empirical theta-centred
coverage brackets directly. No arbitrary centre-displacement assumption is
needed: the displacement is the modal likelihood update. Signed errors use
prediction minus reference, so the recentered error is error plus shift. -/
theorem likelihood_centre_coverage {X ι : Type*} [Fintype X] [Nonempty X] [Fintype ι]
    (error : X → ℝ) (tangent : X → ι → ℝ) (singular residual : ι → ℝ)
    {precision noise threshold : ℝ} (hp : 0 < precision) (hv : 0 < noise)
    (ht : Real.sqrt (likelihoodEnergy singular residual precision / noise) ≤ threshold) :
    let shift := fun x => predictiveShift (tangent x) singular residual precision
    let width := fun x => Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent x i ^ 2) precision)
    let c := Real.sqrt (likelihoodEnergy singular residual precision / noise)
    centreCoverage error width (threshold - c) ≤
      centreCoverage (fun x => error x + shift x) width threshold ∧
    centreCoverage (fun x => error x + shift x) width threshold ≤
      centreCoverage error width (threshold + c) := by
  dsimp only
  apply centre_coverage_sandwich error
    (fun x => predictiveShift (tangent x) singular residual precision)
    (fun x => Real.sqrt (noise * modalVariance (fun i => singular i ^ 2)
      (fun i => tangent x i ^ 2) precision)) (Real.sqrt_nonneg _) ht
    (fun x => Real.sqrt_nonneg _)
  intro x
  simpa only [mul_comm] using likelihood_predictive_shift_bound (tangent x) singular residual hp hv

end PaperReliability
