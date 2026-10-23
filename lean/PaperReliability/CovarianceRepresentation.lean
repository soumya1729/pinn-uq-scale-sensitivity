/- Spectral covariance representation, isolated from historical modules. -/
import PaperReliability.ModalModel

open Matrix
open scoped BigOperators

namespace PaperReliability

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The explicit candidate posterior covariance in eigencoordinates. -/
noncomputable def covarianceOperator (Q : Matrix n n ℝ) (lam : n → ℝ) (α : ℝ) : Matrix n n ℝ :=
  Qᵀ * Matrix.diagonal (fun i => (α + lam i)⁻¹) * Q

/-- The precision αI + H, written through the eigendecomposition. -/
noncomputable def precisionOperator (Q : Matrix n n ℝ) (lam : n → ℝ) (α : ℝ) : Matrix n n ℝ :=
  Qᵀ * Matrix.diagonal (fun i => α + lam i) * Q

/-- The explicit covariance is a right inverse of the precision. -/
private lemma precisionOperator_mul_covarianceOperator
  (Q : Matrix n n ℝ) (lam : n → ℝ) (α : ℝ)
  (hQ : Qᵀ * Q = 1) (hQ' : Q * Qᵀ = 1)
  (hpos : ∀ i, 0 < α + lam i) :
  precisionOperator Q lam α * covarianceOperator Q lam α = 1 := by
  unfold precisionOperator covarianceOperator
  have hDD :
    Matrix.diagonal (fun i => α + lam i)
      * Matrix.diagonal (fun i => (α + lam i)⁻¹) = 1 := by
    rw [Matrix.diagonal_mul_diagonal]
    have : (fun i => (α + lam i) * (α + lam i)⁻¹) = fun _ => (1:ℝ) := by
      funext i
      exact mul_inv_cancel₀ (ne_of_gt (hpos i))
    rw [this, Matrix.diagonal_one]
  calc Qᵀ * Matrix.diagonal (fun i => α + lam i) * Q
        * (Qᵀ * Matrix.diagonal (fun i => (α + lam i)⁻¹) * Q)
      = Qᵀ * Matrix.diagonal (fun i => α + lam i)
        * (Q * Qᵀ) * Matrix.diagonal (fun i => (α + lam i)⁻¹) * Q := by
        simp only [Matrix.mul_assoc]
    _ = Qᵀ * (Matrix.diagonal (fun i => α + lam i)
        * Matrix.diagonal (fun i => (α + lam i)⁻¹)) * Q := by
        rw [hQ']
        simp only [Matrix.mul_one, Matrix.mul_assoc]
    _ = Qᵀ * Q := by
        rw [hDD]
        simp only [Matrix.mul_one]
    _ = 1 := hQ

/-- The explicit covariance is a left inverse of the precision. -/
private lemma covarianceOperator_mul_precisionOperator
  (Q : Matrix n n ℝ) (lam : n → ℝ) (α : ℝ)
  (hQ : Qᵀ * Q = 1) (hQ' : Q * Qᵀ = 1)
  (hpos : ∀ i, 0 < α + lam i) :
  covarianceOperator Q lam α * precisionOperator Q lam α = 1 := by
  unfold precisionOperator covarianceOperator
  have hDD :
    Matrix.diagonal (fun i => (α + lam i)⁻¹)
      * Matrix.diagonal (fun i => α + lam i) = 1 := by
    rw [Matrix.diagonal_mul_diagonal]
    have : (fun i => (α + lam i)⁻¹ * (α + lam i)) = fun _ => (1:ℝ) := by
      funext i
      exact inv_mul_cancel₀ (ne_of_gt (hpos i))
    rw [this, Matrix.diagonal_one]
  calc Qᵀ * Matrix.diagonal (fun i => (α + lam i)⁻¹) * Q
        * (Qᵀ * Matrix.diagonal (fun i => α + lam i) * Q)
      = Qᵀ * Matrix.diagonal (fun i => (α + lam i)⁻¹)
        * (Q * Qᵀ) * Matrix.diagonal (fun i => α + lam i) * Q := by
        simp only [Matrix.mul_assoc]
    _ = Qᵀ * (Matrix.diagonal (fun i => (α + lam i)⁻¹)
        * Matrix.diagonal (fun i => α + lam i)) * Q := by
        rw [hQ']
        simp only [Matrix.mul_one, Matrix.mul_assoc]
    _ = Qᵀ * Q := by
        rw [hDD]
        simp only [Matrix.mul_one]
    _ = 1 := hQ

/-- A diagonal matrix acts by pointwise multiplication. -/
private lemma diag_mulVec (d w : n → ℝ) (i : n) :
    ((Matrix.diagonal d).mulVec w) i = d i * w i := by
  simp [Matrix.mulVec, dotProduct, Matrix.diagonal_apply, ite_mul, zero_mul,
        Finset.sum_ite_eq]

omit [DecidableEq n] in
/-- Transposes move across the dot product: g ⬝ (Qᵀ *ᵥ u) = (Q *ᵥ g) ⬝ u.
    Proved by exchanging the two finite sums. -/
private lemma dot_transpose_mulVec (Q : Matrix n n ℝ) (g u : n → ℝ) :
    g ⬝ᵥ (Qᵀ.mulVec u) = (Q.mulVec g) ⬝ᵥ u := by
  simp only [Matrix.mulVec, dotProduct, Matrix.transpose_apply,
             Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The covariance quadratic form is the finite positive-modal variance. -/
private theorem covarianceOperator_quadForm
  (Q : Matrix n n ℝ) (lam : n → ℝ) (α : ℝ)
  (g : n → ℝ) :
  g ⬝ᵥ (covarianceOperator Q lam α).mulVec g
    = Finset.sum Finset.univ (fun i => ((Q.mulVec g) i)^2 / (α + lam i)) := by
  unfold covarianceOperator
  have h1 :
    (Qᵀ * Matrix.diagonal (fun i => (α + lam i)⁻¹) * Q).mulVec g
      = Qᵀ.mulVec ((Matrix.diagonal (fun i => (α + lam i)⁻¹)).mulVec (Q.mulVec g)) := by
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, Matrix.mul_assoc]
  rw [h1, dot_transpose_mulVec]
  unfold dotProduct
  apply Finset.sum_congr rfl
  intro i _
  rw [diag_mulVec]
  rw [div_eq_mul_inv]
  ring


/-- An orthogonal spectral representation supplies both the inverse precision
and its output quadratic form. No numerical eigendecomposition is certified. -/
theorem spectral_covariance_representation
    (Q : Matrix n n ℝ) (spectrum : n → ℝ) (precision : ℝ)
    (hQ : Qᵀ * Q = 1) (hQ' : Q * Qᵀ = 1)
    (hpos : ∀ i, 0 < precision + spectrum i) :
    precisionOperator Q spectrum precision * covarianceOperator Q spectrum precision = 1 ∧
    covarianceOperator Q spectrum precision * precisionOperator Q spectrum precision = 1 ∧
    ∀ g : n → ℝ,
      g ⬝ᵥ (covarianceOperator Q spectrum precision).mulVec g =
        modalVariance spectrum (fun i => ((Q.mulVec g) i)^2) precision := by
  refine ⟨precisionOperator_mul_covarianceOperator Q spectrum precision hQ hQ' hpos,
    covarianceOperator_mul_precisionOperator Q spectrum precision hQ hQ' hpos, ?_⟩
  intro g
  simpa [modalVariance, add_comm] using covarianceOperator_quadForm Q spectrum precision g

end PaperReliability
