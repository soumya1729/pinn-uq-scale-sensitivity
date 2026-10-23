"""Reusable float64 QR/SVD and grid archive for the isolated theta study.

For J.T = Q R and R = U diag(s) V.T, Q is retained as LAPACK's
packed Householder reflectors, not as a dense orthogonal matrix. Grid columns
are c(x) = U.T Q_1.T g(x); the structural QR complement is kept separately.
These are computed-grid data, not an off-grid or weak-mode accuracy certificate.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from theta_linalg import transform


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} in the archive manifest")


def _blas_metadata(module):
    dependencies = getattr(module.__config__, "CONFIG", {}).get("Build Dependencies", {})
    return {name: {key: value for key, value in dependencies.get(name, {}).items()
        if key in ("name", "version", "openblas configuration", "has ilp64")}
        for name in ("blas", "lapack")}


def verify_archive_manifest(path, checksums=False):
    """Check the completed manifest and arrays; optionally hash every large file.

    The fast resume check verifies file sizes, shapes and dtypes. It does not
    replace a checksum audit after transfer or long-term storage.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["numerical_status"] != "PASS":
        raise ValueError(f"Not a completed native archive: {path}")
    for name, record in manifest["arrays"].items():
        target = path.parent/name
        if target.stat().st_size != record["file_bytes"]:
            raise ValueError(f"Archive file size changed: {target}")
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Archive array layout changed: {target}")
        if record["order"] == "F" and not array.flags.f_contiguous:
            raise ValueError(f"Archive array is not Fortran contiguous: {target}")
        if checksums and file_sha256(target) != record["sha256"]:
            raise ValueError(f"Archive checksum changed: {target}")
    return manifest


class NativeArchive:
    """Write one independent checkpoint archive; finish with verified metadata.

    `root/tag` must be unused. The driver supplies its scientific input/source
    signature and coordinate layout in metadata. All large arrays use .npy and
    memory mapping; there is no compression or second in-memory factor copy.
    """

    def __init__(self, root, tag, factors, theta, residual, modes, point_count,
                 metadata=None):
        self.path = Path(root).resolve()/tag
        self.path.mkdir(parents=True, exist_ok=False)
        self.tag, self.point_count = tag, int(point_count)
        self.metadata = metadata or {}
        packed, tau, rotation, singular, residual_rotation = factors
        self.parameter_count, self.mode_count = packed.shape
        self.names = []
        for name, value in (
            ("packed_qr", packed), ("tau", tau), ("svd_U", rotation),
            ("singular", singular), ("svd_Vt", residual_rotation),
            ("theta", theta), ("residual", residual),
            ("theta_modes", modes["theta_modes"]),
            ("residual_modes", modes["residual_modes"]),
            ("complement_coefficients", modes["complement_coefficients"]),
        ):
            self._save(name, np.asarray(value, dtype=np.float64))
        self.grid_modes = self._memmap("grid_modal_tangents",
            (self.mode_count, self.point_count), order="F")
        self.complement_energy = self._memmap("grid_complement_energy",
            (self.point_count,))
        self.complement_shift = self._memmap("grid_complement_shift",
            (self.point_count,))
        self.next_point = 0
        self.has_gradient_samples = False

    def _save(self, name, value):
        target = self.path/f"{name}.npy"
        np.save(target, value, allow_pickle=False)
        self.names.append(target.name)

    def _memmap(self, name, shape, order="C"):
        target = self.path/f"{name}.npy"
        value = np.lib.format.open_memmap(target, mode="w+", dtype=np.float64,
            shape=shape, fortran_order=(order == "F"))
        self.names.append(target.name)
        return value

    def write_grid_chunk(self, start, modal_tangents, complement_energy,
                         complement_shift, gradient_samples=None,
                         sample_indices=None):
        count = modal_tangents.shape[1]
        if start != self.next_point or modal_tangents.shape[0] != self.mode_count:
            raise ValueError("Native grid archive requires consecutive N-by-chunk columns")
        end = start+count
        self.grid_modes[:, start:end] = modal_tangents
        self.complement_energy[start:end] = complement_energy
        self.complement_shift[start:end] = complement_shift
        self.next_point = end
        if gradient_samples is not None:
            if self.has_gradient_samples:
                raise ValueError("Independent archive gradient samples are supplied once")
            self._save("gradient_samples", np.asarray(gradient_samples, dtype=np.float64))
            self._save("gradient_sample_indices", np.asarray(sample_indices, dtype=np.int64))
            self.has_gradient_samples = True

    def _reload_checks(self, gamma, reference_variance, reference_shift, chunk=200):
        load = lambda name: np.load(self.path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
        # LAPACK's reflector application temporarily writes/restores reflector
        # diagonals. A read-only map can fault; copy-on-write permits this without
        # changing the archive or copying the entire P-by-N factor in advance.
        packed = np.load(self.path/"packed_qr.npy", mmap_mode="c", allow_pickle=False)
        tau, rotation = load("tau"), load("svd_U")
        singular, vt = load("singular"), load("svd_Vt")
        theta, residual = load("theta"), load("residual")
        theta_modes, residual_modes = load("theta_modes"), load("residual_modes")
        complement_coefficients = load("complement_coefficients")
        theta_q = transform(packed, tau, theta)
        norm = np.linalg.norm(theta)
        theta_modal_defect = float(np.linalg.norm(rotation.T @ theta_q[:len(singular)]-theta_modes)/norm)
        theta_complement_defect = float(np.linalg.norm(theta_q[len(singular):]+complement_coefficients)/norm)
        residual_modal_defect = float(np.linalg.norm(vt @ residual-residual_modes)
            /max(np.linalg.norm(residual), np.finfo(float).tiny))

        grid = load("grid_modal_tangents")
        energy, comp_shift = load("grid_complement_energy"), load("grid_complement_shift")
        gradients, indices = load("gradient_samples"), load("gradient_sample_indices")
        gradient_q = transform(packed, tau, gradients.T)
        expected_grid_modes = rotation.T @ gradient_q[:len(singular)]
        norm_g = np.linalg.norm(gradients, axis=1)
        gradient_modal_defect = float(np.max(np.linalg.norm(
            expected_grid_modes-grid[:, indices], axis=0)/norm_g))
        complement = gradient_q[len(singular):]
        gradient_energy_defect = float(np.max(np.abs(np.sum(complement**2, axis=0)
            -energy[indices])/(norm_g**2)))
        gradient_shift_defect = float(np.max(np.abs(complement_coefficients @ complement
            -comp_shift[indices])/(norm_g*norm)))

        inverse = 1/(singular**2+gamma)
        zero_modes = -(singular*residual_modes+gamma*theta_modes)*inverse
        variance_defect, shift_defect = 0., 0.
        for start in range(0, self.point_count, chunk):
            end = min(start+chunk, self.point_count)
            c = grid[:, start:end]
            variance = inverse @ (c*c)+energy[start:end]/gamma
            shift = zero_modes @ c+comp_shift[start:end]
            if not np.isfinite(variance).all() or np.any(variance <= 0) or not np.isfinite(shift).all():
                raise ValueError("Native archive reload produced nonfinite/nonpositive fields")
            reference = reference_variance[start:end]
            variance_defect = max(variance_defect, float(np.max(np.abs(variance/reference-1))))
            scale = np.maximum(np.sqrt(reference), np.abs(reference_shift[start:end]))
            shift_defect = max(shift_defect, float(np.max(np.abs(
                shift-reference_shift[start:end])/scale)))
        checks = dict(theta_modal_relative_defect=theta_modal_defect,
            theta_complement_relative_defect=theta_complement_defect,
            residual_modal_relative_defect=residual_modal_defect,
            sample_gradient_modal_scaled_defect=gradient_modal_defect,
            sample_gradient_complement_energy_scaled_defect=gradient_energy_defect,
            sample_gradient_complement_shift_scaled_defect=gradient_shift_defect,
            full_variance_relative_defect=variance_defect,
            full_zero_shift_scaled_defect=shift_defect,
            gradient_sample_indices=indices.tolist(), gamma=float(gamma), tolerance=1e-10,
            interpretation="Reload/layout consistency; not a weak-mode forward-error certificate.")
        values = [value for name, value in checks.items() if name.endswith("defect")]
        if not all(np.isfinite(value) and value <= checks["tolerance"] for value in values):
            raise ValueError(f"Native archive reload check failed: {checks}")
        return checks

    def finalize(self, baseline_gamma, baseline_unit_variance, baseline_zero_shift):
        if self.next_point != self.point_count or not self.has_gradient_samples:
            raise ValueError("Native archive grid or independent gradient samples are incomplete")
        for array in (self.grid_modes, self.complement_energy, self.complement_shift):
            array.flush()
        del array
        del self.grid_modes, self.complement_energy, self.complement_shift
        checks = self._reload_checks(baseline_gamma, baseline_unit_variance, baseline_zero_shift)
        records = {}
        for name in self.names:
            path = self.path/name
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            records[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                order="F" if array.ndim > 1 and array.flags.f_contiguous else "C",
                file_bytes=path.stat().st_size, sha256=file_sha256(path))
            print(f"Retained archive checksum: {self.tag}/{name}", flush=True)
        source_paths = [Path(__file__), Path(__file__).with_name("theta_linalg.py")]
        manifest = dict(schema_version=1, tag=self.tag, numerical_status="PASS",
            parameter_count=self.parameter_count, residual_count=self.mode_count,
            grid_point_count=self.point_count, metadata=self.metadata,
            libraries=dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, qr="scipy.linalg.qr(raw); LAPACK dgeqrf",
                svd="scipy.linalg.svd; default LAPACK gesdd", Q_application="LAPACK dormqr",
                numpy_blas_lapack=_blas_metadata(np), scipy_blas_lapack=_blas_metadata(scipy)),
            conventions=dict(jacobian="J has shape (N,P); J.T = Q R",
                packed_qr="P-by-N, Fortran contiguous; R above/on diagonal, Householder vectors below; tau stores reflector coefficients; use mmap_mode='c' when applying dormqr (temporary LAPACK writes)",
                svd="R = svd_U @ diag(singular) @ svd_Vt; U and Vt have shape (N,N)",
                singular_order="descending, untruncated; QR complement is structural, not the entire kernel when J is rank deficient",
                theta_modes="U.T @ (Q.T theta)[:N]", residual_modes="Vt @ residual",
                complement_coefficients="-(Q.T theta)[N:]",
                grid_modal_tangents="N-by-M, Fortran contiguous; column j = U.T @ (Q.T g_j)[:N]",
                grid_complement_energy="sum((Q.T g_j)[N:]**2), computed directly",
                grid_complement_shift="complement_coefficients.T @ (Q.T g_j)[N:]",
                gradient_samples="K-by-P gradients, linked to grid columns by integer gradient_sample_indices",
                grid_scope="Only the supplied existing evaluation grid; new points require new gradients"),
            sources={p.name: file_sha256(p) for p in source_paths}, arrays=records,
            reload_checks=checks)
        path = self.path/"manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=_json_value)+"\n", encoding="utf-8")
        return dict(numerical_status="PASS", checks_passed=True,
            manifest_path=str(path), manifest_sha256=file_sha256(path),
            retained_bytes=sum(record["file_bytes"] for record in records.values()),
            reload_checks=checks)
