"""Direct-J spectral algebra for the shared, frozen-checkpoint round-two pass.

No Gram matrix, rank truncation or added numerical noise is used. QR's full
orthogonal matrix is applied through Householder reflectors, never stored.
"""
import numpy as np
from scipy.linalg import qr, svd
from scipy.linalg.lapack import dormqr


def factor_jacobian(jacobian):
    """J.T = Q R; R = U diag(s) V.T, with J shaped N by P (P >= N)."""
    (packed, tau), triangular = qr(jacobian.T, mode="raw", overwrite_a=True)
    rotation, singular, observation_rotation = svd(
        triangular, full_matrices=False, overwrite_a=True)
    return packed, tau, rotation, singular, observation_rotation


def transform(packed, tau, values, transpose=True):
    """Apply Q.T (default) or Q to P-by-K values, including QR complement."""
    vector = np.ndim(values) == 1
    work_array = np.array(np.asarray(values).reshape(len(packed), -1),
                          dtype=np.float64, order="F", copy=True)
    trans = b"T" if transpose else b"N"
    _, work, info = dormqr(b"L", trans, packed, tau, work_array, -1)
    if info:
        raise RuntimeError(f"Householder workspace query returned {info}")
    result, _, info = dormqr(b"L", trans, packed, tau, work_array,
                            int(work[0]), overwrite_c=1)
    if info:
        raise RuntimeError(f"Householder transform returned {info}")
    return result[:, 0] if vector else result


def posterior_modes(theta, residual, packed, tau, rotation, singular,
                    observation_rotation, gammas):
    """Affine mean shift; every mode is retained and only gamma = alpha*v enters."""
    theta_q = transform(packed, tau, theta)
    theta_modes = rotation.T @ theta_q[:len(singular)]
    residual_modes = observation_rotation @ residual
    gammas = np.atleast_1d(gammas)
    delta_modes = -(singular[None, :] * residual_modes[None, :]
                    + gammas[:, None] * theta_modes[None, :]) / (
                        singular[None, :]**2 + gammas[:, None])
    return dict(theta_q=theta_q, theta_modes=theta_modes,
                residual_modes=residual_modes, delta_modes=delta_modes,
                complement_coefficients=-theta_q[len(singular):])


def reconstruct_shift(packed, tau, rotation, delta_modes, complement_coefficients):
    """Return one parameter shift without forming the orthogonal basis."""
    coefficients = np.concatenate((rotation @ delta_modes, complement_coefficients))
    return transform(packed, tau, coefficients, transpose=False)


def evidence_grid(singular, projected_data, alpha_grid, noise_grid):
    """Full affine log evidence including its normalising constant."""
    singular, projected_data = np.asarray(singular), np.asarray(projected_data)
    alpha_grid, noise_grid = np.asarray(alpha_grid), np.asarray(noise_grid)
    surface = np.empty((len(alpha_grid), len(noise_grid)))
    for i, alpha in enumerate(alpha_grid):
        diagonal = noise_grid[:, None] + singular[None, :]**2 / alpha
        surface[i] = -0.5 * (len(singular) * np.log(2 * np.pi)
            + np.sum(np.log(diagonal) + projected_data[None, :]**2 / diagonal, axis=1))
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    near = np.argwhere(surface >= surface[i, j] - 1)
    best = dict(alpha=float(alpha_grid[i]), noise_variance=float(noise_grid[j]),
        effective_precision=float(alpha_grid[i] * noise_grid[j]),
        log_evidence=float(surface[i, j]), alpha_index=int(i), noise_index=int(j),
        alpha_boundary=bool(i in (0, len(alpha_grid)-1)),
        noise_boundary=bool(j in (0, len(noise_grid)-1)),
        within_one_log_unit_count=int(len(near)),
        within_one_log_unit_alpha_range=[float(alpha_grid[near[:, 0]].min()),
                                       float(alpha_grid[near[:, 0]].max())],
        within_one_log_unit_noise_range=[float(noise_grid[near[:, 1]].min()),
                                       float(noise_grid[near[:, 1]].max())])
    return best, surface


def evidence_slope(singular, projected_data, alpha, noise_variance):
    """Derivative with respect to log(v), holding alpha and observation fixed.

    Near-null modes are grouped by lambda/gamma: far below .01, .01 to 100,
    and far above 100. Group energy is meaningful without identifying a
    particular singular-vector basis within a nearly degenerate cluster.
    """
    eigenvalues = np.asarray(singular)**2
    z = np.asarray(projected_data)
    diagonal = noise_variance + eigenvalues / alpha
    ratio = eigenvalues / (alpha * noise_variance)
    terms = -0.5 * (noise_variance / diagonal) * (1-z**2 / diagonal)
    groups = []
    for label, mask in (("lambda_below_gamma/100", ratio < .01),
                        ("lambda_near_gamma", (ratio >= .01) & (ratio <= 100)),
                        ("lambda_above_100gamma", ratio > 100)):
        groups.append(dict(group=label, count=int(mask.sum()),
                           projected_data_energy=float(np.sum(z[mask]**2)),
                           slope=float(np.sum(terms[mask]))))
    return dict(slope=float(terms.sum()), groups=groups,
                mode_slope=terms, lambda_over_gamma=ratio,
                data_variance=eigenvalues/alpha, projected_data_energy=z**2)


def evidence_stability(singular, projected_data, theta_modes, residual_modes,
                       alpha_grid, noise_grid, matrix_shape, selected_settings):
    """Finite coherent perturbations, not certified backward-error bounds.

    The envelope is eps*max(N,P)*s_max and ten times that value. Singular
    vectors stay fixed: a perturbed J changes the affine observation through
    z' = z + (s'-s)*theta_modes. The original projected observation remains
    the baseline; its factorisation consistency is reported separately.
    Both signs and a deterministic
    alternating pattern are tested. Squaring s' retains the 2*s*ds + ds^2 term.
    For an unresolved cluster, only grouped energy is interpreted; no random
    per-mode projected observations are introduced.
    """
    s, z = np.asarray(singular), np.asarray(projected_data)
    if isinstance(selected_settings, dict):
        selected_settings = [dict(setting=name, **values)
                             for name, values in selected_settings.items()]
    base_best, _ = evidence_grid(s, z, alpha_grid, noise_grid)
    scale = float(np.finfo(float).eps * max(matrix_shape) * s.max())
    signs = np.where(np.arange(len(s)) % 2 == 0, 1.0, -1.0)
    records = []
    for multiple in (1.0, 10.0):
        for label, pattern in (("increase", np.ones_like(s)),
                               ("decrease", -np.ones_like(s)),
                               ("alternating", signs)):
            perturbed_s = np.maximum(s + multiple*scale*pattern, 0)
            perturbed_z = z+(perturbed_s-s)*np.asarray(theta_modes)
            best, surface = evidence_grid(perturbed_s, perturbed_z, alpha_grid, noise_grid)
            row = dict(case=f"{label}_{multiple:g}x", singular_value_envelope=multiple*scale,
                max_singular_change=float(np.max(np.abs(perturbed_s-s))),
                max_eigenvalue_change=float(np.max(np.abs(perturbed_s**2-s**2))),
                best=best, selected=[])
            for setting in selected_settings:
                a, v = setting["alpha"], setting["noise_variance"]
                original = evidence_grid(s, z, [a], [v])[1][0, 0]
                perturbed = evidence_grid(perturbed_s, perturbed_z, [a], [v])[1][0, 0]
                row["selected"].append(dict(setting=setting["setting"],
                    log_evidence_change=float(perturbed-original),
                    original_slope=evidence_slope(s, z, a, v)["slope"],
                    perturbed_slope=evidence_slope(perturbed_s, perturbed_z, a, v)["slope"]))
            records.append(row)
    # Equal singular values have isotropic covariance: rotations of z inside
    # such a cluster preserve its energy and evidence. This records that
    # identity directly; it does not assert that unequal modes are identical.
    repeated_groups = []
    for value in np.unique(s):
        idx = np.flatnonzero(s == value)
        if len(idx) > 1:
            rotated = z[idx].copy()
            rotated[:2] = np.array([[.6, -.8], [.8, .6]]) @ rotated[:2]
            repeated_groups.append(dict(singular_value=float(value), count=len(idx),
                energy_difference=float(rotated@rotated-z[idx]@z[idx])))
    return dict(envelope_definition="eps*max(N,P)*s_max; sensitivity, not certification",
                matrix_shape=list(matrix_shape), base_best=base_best,
                affine_projection_consistency_norm=float(np.linalg.norm(
                    z-(s*np.asarray(theta_modes)-np.asarray(residual_modes)))),
                perturbations=records, equal_singular_cluster_rotations=repeated_groups)
