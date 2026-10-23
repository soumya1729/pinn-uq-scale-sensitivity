import Mathlib

open scoped BigOperators

namespace PaperReliability

/-- Variance in a finite modal model. Zero eigenvalues represent kernel directions. -/
noncomputable def modalVariance {ι : Type*} [Fintype ι]
    (spectrum energy : ι → ℝ) (precision : ℝ) : ℝ :=
  ∑ i, energy i / (spectrum i + precision)

/-- Regional RMS contrast squared, before applying the square root. -/
noncomputable def contrastSq {ι : Type*} [Fintype ι]
    (spectrum sharp bulk : ι → ℝ) (precision : ℝ) : ℝ :=
  modalVariance spectrum sharp precision / modalVariance spectrum bulk precision

end PaperReliability
