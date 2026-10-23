"""Small dense and nonlinear checks; no production checkpoint is differentiated."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from theta_linalg import (factor_jacobian, posterior_modes, reconstruct_shifts,
                          gradient_components, unit_variance, theta_log_evidence)
from theta_pass import independent_normal_checks, evaluate_grid, region_masks, scientific_rows

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def matrix_checks():
    rng = np.random.default_rng(71)
    maximum = {name: 0. for name in ("theta_shift", "zero_shift", "prior_shift",
               "complement_shift", "variance", "evidence", "gradient_norm")}
    for singular_values in ([3., 1., .4], [3., 1., 0.], [3., 1e-6, 1e-10]):
        left, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        right, _ = np.linalg.qr(rng.normal(size=(8, 3)))
        jacobian = left @ np.diag(singular_values) @ right.T
        theta, residual = rng.normal(size=8), rng.normal(size=3)
        gradients = rng.normal(size=(7, 8))
        factors = factor_jacobian(jacobian.copy())
        gammas = np.array([.03, 1., 90.])
        modes = posterior_modes(theta, residual, factors, gammas)
        likelihood = reconstruct_shifts(factors, modes['likelihood_modes'])
        zero = reconstruct_shifts(factors, modes['zero_modes'], modes['complement_coefficients'])
        prior = zero-likelihood
        complement = reconstruct_shifts(factors, np.zeros((1, 3)),
                                        modes['complement_coefficients'])[0]
        # The QR complement is structural (P-N); a rank-deficient J may have
        # additional kernel directions within its first N QR coordinates.
        full_q, _ = np.linalg.qr(jacobian.T, mode="complete")
        expected_complement = -full_q[:, 3:] @ (full_q[:, 3:].T @ theta)
        np.testing.assert_allclose(complement, expected_complement, rtol=1e-10, atol=1e-11)
        maximum['complement_shift'] = max(maximum['complement_shift'],
                                         float(np.max(np.abs(complement-expected_complement))))
        c, cc, norm = gradient_components(factors, gradients)
        variance = unit_variance(factors[3], gammas, c**2, np.sum(cc**2, axis=0))
        np.testing.assert_allclose(norm, np.sum(c**2, axis=0)+np.sum(cc**2, axis=0), rtol=1e-12)
        maximum['gradient_norm'] = max(maximum['gradient_norm'], float(np.max(
            np.abs(norm-np.sum(c**2, axis=0)-np.sum(cc**2, axis=0)))))
        fisher = jacobian.T @ jacobian
        for k, gamma in enumerate(gammas):
            precision = fisher+gamma*np.eye(8)
            references = dict(theta_shift=np.linalg.solve(precision, -jacobian.T @ residual),
                              zero_shift=np.linalg.solve(precision, -jacobian.T @ residual-gamma*theta),
                              prior_shift=np.linalg.solve(precision, -gamma*theta))
            for label, actual in (("theta_shift", likelihood[k]), ("zero_shift", zero[k]),
                                  ("prior_shift", prior[k])):
                np.testing.assert_allclose(actual, references[label], rtol=1e-10, atol=1e-11)
                maximum[label] = max(maximum[label], float(np.max(np.abs(actual-references[label]))))
            expected_variance = np.einsum('ij,ji->i', gradients,
                                          np.linalg.solve(precision, gradients.T))
            np.testing.assert_allclose(variance[k], expected_variance, rtol=1e-11, atol=1e-11)
            maximum['variance'] = max(maximum['variance'],
                                      float(np.max(np.abs(variance[k]-expected_variance))))
        alphas, noises = np.array([.1, 2., 40.]), np.array([.05, .9])
        surface = theta_log_evidence(factors[3], modes['residual_modes'], alphas, noises)
        for i, alpha in enumerate(alphas):
            for j, noise in enumerate(noises):
                covariance = noise*np.eye(3)+(jacobian @ jacobian.T)/alpha
                _, logdet = np.linalg.slogdet(covariance)
                expected = -.5*(3*np.log(2*np.pi)+logdet+residual @ np.linalg.solve(covariance, residual))
                np.testing.assert_allclose(surface[i, j], expected, rtol=1e-11, atol=1e-11)
                maximum['evidence'] = max(maximum['evidence'], abs(float(surface[i, j]-expected)))
    return maximum


def nonlinear_checks():
    theta = jnp.array([.7, -.4, .3, .2, -.1, .5])

    def u(a, x, t):
        return a[0]*jnp.tanh(a[1]*x+a[2]*t+a[3])+a[4]*x+a[5]

    coordinates = np.array([-.8, -.2, .6])

    def residual(a, x):
        return u(a, x, .2)-jnp.sin(x)

    blocks = [(residual, (coordinates,))]
    r = np.asarray(jax.vmap(residual, in_axes=(None, 0))(theta, coordinates))
    jacobian = np.asarray(jax.vmap(jax.grad(residual), in_axes=(None, 0))(theta, coordinates))
    factors = factor_jacobian(jacobian.copy())
    gammas = np.array([.05, 2.])
    modes = posterior_modes(np.asarray(theta), r, factors, gammas)
    deltas = reconstruct_shifts(factors, modes['likelihood_modes'])
    xs, ts = np.linspace(-1, 1, 8), np.linspace(0, 1, 5)
    xx, tt = np.meshgrid(xs, ts)
    predictions = np.asarray(jax.vmap(u, in_axes=(None, 0, 0))(theta, xx.ravel(), tt.ravel()))
    data = dict(xs=xs, ts=ts, u_pred=predictions,
                u_exact=np.sin(xx.ravel())+.2*tt.ravel(), grad_abs=np.abs(xx.ravel())+tt.ravel())
    fields = evaluate_grid(theta, u, data, factors, modes, gammas, deltas, 11)
    gradient = np.asarray(jax.vmap(jax.grad(u), in_axes=(None, 0, 0))(theta, xx.ravel(), tt.ravel()))
    np.testing.assert_allclose(fields['theta_shift'], deltas @ gradient.T, rtol=1e-10, atol=1e-11)
    for k, delta in enumerate(deltas):
        expected = np.asarray(jax.vmap(u, in_axes=(None, 0, 0))(
            theta+jnp.asarray(delta), xx.ravel(), tt.ravel()))
        np.testing.assert_allclose(fields['theta_nonlinear'][k], expected, rtol=1e-12, atol=1e-12)
    normal = independent_normal_checks(theta, blocks, r, gammas, deltas, 2, factors[3][0]**2)
    assert max(row['normwise_backward_error'] for row in normal) < 1e-12
    masks, _ = region_masks(data['grad_abs'], tt.ravel(), 'burgers')
    choice = dict(setting="toy_root", selection="reference_informed_trained_root",
                  gamma=float(gammas[0]), gamma_index=0, alpha=float(gammas[0]), noise_variance=1.)
    rows = scientific_rows('toy', data, [choice], fields, masks)
    assert set(row['centre'] for row in rows) == {'trained', 'theta_affine', 'theta_nonlinear', 'zero_affine'}
    for row in rows:
        if row['calibration'] == 'bulk_68_oracle' and row['region'] == 'bulk':
            assert row['coverage68'] >= .68
    return dict(grid_checks=fields['diagnostics'], normal_checks=normal,
                rows=len(rows), nonlinear_remainder_is_nonzero=bool(np.any(
                    np.abs(fields['theta_nonlinear']-data['u_pred'][None, :]-fields['theta_shift']) > 1e-8)))


def main():
    result = dict(status="PASS", dense_matrix_checks=matrix_checks(),
                  nonlinear_checks=nonlinear_checks(), versions=dict(jax=jax.__version__, numpy=np.__version__))
    path = Path(__file__).resolve().parents[1]/"verification/theta_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
