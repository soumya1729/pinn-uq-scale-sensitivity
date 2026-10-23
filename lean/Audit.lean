import PaperReliability

/- Standard logical axioms are expected; sorryAx and custom axioms are not. -/
#print axioms PaperReliability.spectral_covariance_representation
#print axioms PaperReliability.two_subspace_contrast_enclosure
#print axioms PaperReliability.regional_two_subspace_enclosure
#print axioms PaperReliability.upper_envelope_arbitrary_energy_sharpness
#print axioms PaperReliability.lower_envelope_arbitrary_energy_sharpness
#print axioms PaperReliability.positive_modal_variance_enclosure
#print axioms PaperReliability.variance_enclosure_on_precision_bracket
#print axioms PaperReliability.calibrated_modal_coverage_enclosure
#print axioms PaperReliability.kernel_prediction_shift_attribution
#print axioms PaperReliability.likelihood_predictive_shift_bound
#print axioms PaperReliability.likelihood_regional_error_bounds
#print axioms PaperReliability.likelihood_centre_coverage
#print axioms PaperReliability.regional_contrast_strong_limit
#print axioms PaperReliability.regional_contrast_weak_limit
#print axioms PaperReliability.endpoint_limits_do_not_bound_contrast

/- Confirm that imported centre theorems use the paper's error + shift convention. -/
#check @PaperReliability.likelihood_regional_error_bounds
#check @PaperReliability.likelihood_centre_coverage
