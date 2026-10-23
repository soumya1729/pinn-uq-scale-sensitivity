"""Small-matrix checks of the production numerical core; no PINN inputs."""
import json
from pathlib import Path
import numpy as np
from round2_linalg import (factor_jacobian, transform, posterior_modes,
                          reconstruct_shift, evidence_grid, evidence_slope,
                          evidence_stability)


def main():
    rng = np.random.default_rng(7072026)
    n, p = 7, 11
    j = rng.normal(size=(n, p))
    theta, residual = rng.normal(size=p), rng.normal(size=n)
    y = j@theta-residual
    packed, tau, u, s, vt = factor_jacobian(j.copy())
    gammas = np.array([.1, 1., 100.])
    modes = posterior_modes(theta, residual, packed, tau, u, s, vt, gammas)
    q = transform(packed, tau, np.eye(p), transpose=False)
    reconstruction_error = np.linalg.norm(j-(vt.T*s)@u.T@q[:, :n].T)/np.linalg.norm(j)
    np.testing.assert_allclose(q.T@q, np.eye(p), atol=1e-14)
    result = dict(reconstruction_relative_error=float(reconstruction_error), settings=[])
    for k, gamma in enumerate(gammas):
        alpha, noise = 2., gamma/2
        delta = reconstruct_shift(packed, tau, u, modes["delta_modes"][k],
                                  modes["complement_coefficients"])
        dense_delta = -np.linalg.solve(j.T@j+gamma*np.eye(p), j.T@residual+gamma*theta)
        np.testing.assert_allclose(delta, dense_delta, rtol=1e-12, atol=1e-12)
        c = noise*np.eye(n)+j@j.T/alpha
        sign, logdet = np.linalg.slogdet(c)
        expected = -.5*(n*np.log(2*np.pi)+logdet+y@np.linalg.solve(c, y))
        actual = evidence_grid(s, vt@y, [alpha], [noise])[1][0, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        h = 1e-5
        finite = evidence_grid(s, vt@y, [alpha], noise*np.exp([-h,h]))[1][0]
        derivative = (finite[1]-finite[0])/(2*h)
        slope = evidence_slope(s, vt@y, alpha, noise)
        np.testing.assert_allclose(slope["slope"], derivative, atol=1e-7)
        np.testing.assert_allclose(sum(r["slope"] for r in slope["groups"]), slope["slope"])
        result["settings"].append(dict(gamma=float(gamma), mean_shift_max_error=float(np.max(np.abs(delta-dense_delta))),
            log_evidence_error=float(abs(actual-expected)), slope_error=float(abs(slope["slope"]-derivative))))
    diagnostic = evidence_stability(s, vt@y, modes["theta_modes"], modes["residual_modes"],
        np.logspace(-2,2,9), np.logspace(-3,1,9), j.shape,
        [dict(setting="reference", alpha=2., noise_variance=.5)])
    assert len(diagnostic["perturbations"]) == 6
    result["stability_envelopes"] = diagnostic
    # Exact repeated singular values: likelihood depends on cluster energy.
    same_s = np.array([4., 1., 1., 0., 0.])
    z = rng.normal(size=len(same_s))
    rotated = z.copy()
    rotation = np.array([[.6,-.8],[.8,.6]])
    rotated[1:3], rotated[3:] = rotation@z[1:3], rotation@z[3:]
    np.testing.assert_allclose(evidence_grid(same_s,z,[2.],[.1])[1],
                               evidence_grid(same_s,rotated,[2.],[.1])[1], atol=1e-13)
    result["status"] = "PASS"
    out = Path(__file__).resolve().parent/"outputs"/"preparation"
    out.mkdir(parents=True,exist_ok=True)
    (out/"linalg_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Production core small-matrix checks: PASS")


if __name__ == "__main__":
    main()
