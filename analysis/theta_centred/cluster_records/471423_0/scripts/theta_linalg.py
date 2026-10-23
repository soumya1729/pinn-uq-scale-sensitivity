"""Direct-J algebra for a prior centred on the frozen checkpoint.

J.T = Q R and R = U diag(s) V.T. Q is applied using its Householder
reflectors. No Gram matrix, rank cutoff, or added variance floor is used.
"""
import numpy as np
from scipy.linalg import qr, svd
from scipy.linalg.lapack import dormqr


def factor_jacobian(jacobian):
    """Factor an N-by-P Jacobian, P >= N; the input may be overwritten."""
    if jacobian.shape[0] > jacobian.shape[1]:
        raise ValueError("This full-network protocol assumes P >= N.")
    (packed, tau), triangular = qr(jacobian.T, mode="raw", overwrite_a=True)
    rotation, singular, residual_rotation = svd(
        triangular, full_matrices=False, overwrite_a=True)
    return packed, tau, rotation, singular, residual_rotation


def transform(packed, tau, values, transpose=True):
    """Apply Q.T or Q to one vector or a matrix whose columns are vectors."""
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


def posterior_modes(theta, residual, factors, gammas):
    """Likelihood, zero-prior and structural-complement shift coordinates.

The theta-centred shift is delta_L. The zero-mean-prior shift adds delta_P,
whose QR-complement coordinates are -Q_c.T theta at every gamma.
"""
    packed, tau, rotation, singular, residual_rotation = factors
    theta_q = transform(packed, tau, theta)
    theta_modes = rotation.T @ theta_q[:len(singular)]
    residual_modes = residual_rotation @ residual
    gammas = np.asarray(gammas)
    denominator = singular[None, :]**2 + gammas[:, None]
    likelihood = -singular[None, :]*residual_modes[None, :]/denominator
    prior = -gammas[:, None]*theta_modes[None, :]/denominator
    return dict(theta_modes=theta_modes, residual_modes=residual_modes,
                likelihood_modes=likelihood, prior_modes=prior,
                zero_modes=likelihood+prior,
                complement_coefficients=-theta_q[len(singular):])


def reconstruct_shifts(factors, mode_values, complement=None):
    """Reconstruct K-by-P shifts, including a shared complement if supplied."""
    packed, tau, rotation, singular, _ = factors
    mode_values = np.atleast_2d(mode_values)
    coefficients = np.zeros((len(packed), len(mode_values)))
    coefficients[:len(singular)] = rotation @ mode_values.T
    if complement is not None:
        coefficients[len(singular):] = np.asarray(complement)[:, None]
    return transform(packed, tau, coefficients, transpose=False).T


def gradient_components(factors, gradient):
    """Return signed modal tangents, complement tangents and squared norm."""
    packed, tau, rotation, singular, _ = factors
    qt = transform(packed, tau, np.asarray(gradient).T)
    modes = rotation.T @ qt[:len(singular)]
    return modes, qt[len(singular):], np.sum(np.asarray(gradient)**2, axis=1)


def unit_variance(singular, gammas, mode_energy, complement_energy):
    """g.T (J.T J + gamma I)^-1 g as a sum of nonnegative terms."""
    gammas = np.asarray(gammas)
    return ((1/(singular[None, :]**2+gammas[:, None])) @ mode_energy
            + complement_energy[None, :]/gammas[:, None])


def theta_log_evidence(singular, residual_modes, alpha_grid, noise_grid):
    """Affine evidence under N(theta, I/alpha), including its constant."""
    surface = np.empty((len(alpha_grid), len(noise_grid)))
    for k, alpha in enumerate(alpha_grid):
        variance = noise_grid[:, None]+singular[None, :]**2/alpha
        surface[k] = -.5*(len(singular)*np.log(2*np.pi)
            + np.sum(np.log(variance)+residual_modes[None, :]**2/variance, axis=1))
    return surface
