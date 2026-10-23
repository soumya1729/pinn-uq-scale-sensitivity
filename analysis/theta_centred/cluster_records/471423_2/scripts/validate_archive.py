"""Small archive round trips and independent dense comparisons, no PINN pass."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from theta_archive import NativeArchive, verify_archive_manifest
from theta_linalg import factor_jacobian, posterior_modes, gradient_components, unit_variance


def round_trip(path, rank_deficient=False):
    rng = np.random.default_rng(91+int(rank_deficient))
    n, p, count, gamma = 7, 23, 29, .4
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(p, n)))
    singular = np.geomspace(3., .01, n)
    if rank_deficient:
        singular[-2:] = 0.
    jacobian = left @ np.diag(singular) @ right.T
    theta, residual = rng.normal(size=p), rng.normal(size=n)
    gradients = rng.normal(size=(count, p))
    factors = factor_jacobian(jacobian.copy())
    modes = posterior_modes(theta, residual, factors, [gamma])
    c, complement, _ = gradient_components(factors, gradients)
    energy = np.sum(complement**2, axis=0)
    complement_shift = modes["complement_coefficients"] @ complement
    variance = unit_variance(factors[3], np.array([gamma]), c*c, energy)[0]
    shift = modes["zero_modes"][0] @ c+complement_shift
    precision = jacobian.T @ jacobian+gamma*np.eye(p)
    expected_variance = np.einsum("ij,ji->i", gradients,
        np.linalg.solve(precision, gradients.T))
    expected_shift = gradients @ np.linalg.solve(precision, -jacobian.T @ residual-gamma*theta)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shift, expected_shift, rtol=1e-12, atol=1e-12)

    tag = "rank_deficient" if rank_deficient else "full_row_rank"
    archive = NativeArchive(path, tag, factors, theta, residual, modes, count,
        metadata=dict(grid_order="test columns", fixed_gamma=gamma))
    for start in range(0, count, 8):
        end = min(start+8, count)
        samples = dict(gradient_samples=gradients[:3], sample_indices=np.arange(3)) if start == 0 else {}
        archive.write_grid_chunk(start, c[:, start:end], energy[start:end],
            complement_shift[start:end], **samples)
    result = archive.finalize(gamma, variance, shift)
    manifest = verify_archive_manifest(result["manifest_path"], checksums=True)
    assert manifest["arrays"]["grid_modal_tangents.npy"]["order"] == "F"
    assert manifest["arrays"]["packed_qr.npy"]["order"] == "F"
    assert result["checks_passed"]
    report = dict(rank_deficient=rank_deficient,
        dense_variance_relative_defect=float(np.max(np.abs(variance/expected_variance-1))),
        dense_zero_shift_absolute_defect=float(np.max(np.abs(shift-expected_shift))),
        reload_checks=result["reload_checks"])

    # A same-shaped, valid .npy can still have changed data after transfer.
    target = Path(result["manifest_path"]).parent/"residual.npy"
    changed = np.load(target, mmap_mode="r+")
    changed[0] += .25
    changed.flush()
    try:
        verify_archive_manifest(result["manifest_path"], checksums=True)
    except ValueError as error:
        assert "checksum changed" in str(error)
        report["post_write_corruption_detected"] = True
    else:
        raise AssertionError("A changed stored residual was accepted by the checksum audit")
    return report


def layout_corruption(path, nonfinite=False):
    rng = np.random.default_rng(94)
    jacobian = rng.normal(size=(5, 17))
    theta, residual = rng.normal(size=17), rng.normal(size=5)
    gradients = rng.normal(size=(9, 17))
    factors = factor_jacobian(jacobian.copy())
    modes = posterior_modes(theta, residual, factors, [.7])
    c, cc, _ = gradient_components(factors, gradients)
    energy = np.sum(cc*cc, axis=0)
    shift_c = modes["complement_coefficients"] @ cc
    variance = unit_variance(factors[3], np.array([.7]), c*c, energy)[0]
    shift = modes["zero_modes"][0] @ c+shift_c
    archive = NativeArchive(path, "corrupt_grid" if nonfinite else "corrupt_layout",
                            factors, theta, residual, modes, 9)
    archive.write_grid_chunk(0, c, energy, shift_c,
        gradient_samples=gradients[:3], sample_indices=np.arange(3))
    if nonfinite:
        changed = np.load(archive.path/"grid_modal_tangents.npy", mmap_mode="r+")
        changed[0, -1] = np.nan
    else:
        changed = np.load(archive.path/"packed_qr.npy", mmap_mode="r+")
        changed[-1, 0] += .1
    changed.flush()
    try:
        archive.finalize(.7, variance, shift)
    except ValueError as error:
        assert ("nonfinite/nonpositive" if nonfinite else "reload check failed") in str(error)
        assert not (archive.path/"manifest.json").exists()
        return dict(nonfinite_grid_detected=nonfinite,
            corrupt_householder_layout_detected=not nonfinite,
            interpretation="Reload checks inspect native factors, not only saved grid fields.")
    raise AssertionError("A corrupted native archive was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
        default=Path(__file__).resolve().parents[1]/"outputs/local/archive_validation.json")
    args = parser.parse_args()
    temporary_root = Path(__file__).resolve().parents[1]/"verification"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="theta_archive_validation_", dir=temporary_root) as folder:
        assert Path(folder).resolve().is_relative_to(temporary_root.resolve())
        report = dict(numerical_status="PASS", cases=[round_trip(folder, False),
            round_trip(folder, True)], layout_check=layout_corruption(folder),
            nonfinite_check=layout_corruption(folder, nonfinite=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: native archive round trips, dense references and corruption checks. Saved {args.out}")


if __name__ == "__main__":
    main()
