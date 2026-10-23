# Environments

Use `envs/reviewer_requirements.txt` for the public saved-data route. The other
environment records describe historical training or specialised production stages.

| Stage | Recorded dependencies |
| --- | --- |
| Original training | Python 3.12.5; JAX/jaxlib 0.10.1; NumPy 2.4.6; SciPy 1.17.1; Optax 0.2.8. Full saved package freeze: `envs/training_recorded_freeze.txt`. |
| N1 and all-17 production | JAX 0.10.1, NumPy 2.4.6, SciPy 1.17.1; GPU devices recorded in the cluster logs. CUDA packages and environment are preserved in the executed records. |
| N5 | NumPy; no JAX or GPU computation. |
| Saved-array reviews/plots | NumPy, SciPy for numerical-core utilities, Matplotlib; `pypdf` for the combined Burgers figure audit. |
| Third-round saved consumer | Recorded NumPy 2.5.3. Its SHA-256 matches the consumer fingerprint in the saved summary. |
| Formal development | Lean 4.30.0 and pinned mathlib v4.30.0 manifest under `lean/`. |

`envs/production_requirements.txt` pins the measured core production versions. For the recorded GPU route, install the CUDA-enabled JAX wheel corresponding to the preserved environment; the historical `slurm/env.sh` documents that setup but must be adapted to another cluster. For CPU-only saved-array replay, JAX is unnecessary.

`envs/reviewer_requirements.txt` pins the current reviewer dependencies: Python
3.12, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, pypdf 6.10.0 and openpyxl
3.1.5. This route needs no JAX or CUDA. Numerical comparisons use the reported
tolerances; identical figure/PDF bytes across environments are not promised.

`envs/saved_analysis_requirements.txt` preserves an earlier consumer environment,
including its recorded NumPy 2.5.3. It is not the installation command for the
current reviewer route. A complete historical figure-build freeze was not recovered.

`envs/training_recorded_freeze.txt` includes packages used by the wider original project. It is evidence of the historical environment, not a list of packages all required by this paper. HMC, dropout and other unused methods are not part of the replication target.

The original N1 results reside in a folder named `outputs_cpu`, but the successful production logs show CUDA devices and job 469488. The failed CPU-partition attempt is not the numerical provenance. The all-17 production records are the successful pilot 469955 and main array 470013; duplicate cancelled array 470288 is only visible in accounting history.

## Theta-centred execution - 16 October 2026

The successful pilot `471403_0` and array `471423_0..2` recorded JAX 0.10.1,
NumPy 2.4.6 and SciPy 1.17.1 on CUDA. The four executed source snapshots and
accounting are in `analysis/theta_centred/cluster_records`. The private local CPU
environment is excluded. The pinned core is in
`analysis/theta_centred/requirements-cpu.txt`; use a CUDA-enabled JAX installation
matched to the target site for deliberate production recomputation.

Saved-field summary/table replay requires NumPy; metadata review uses the standard
library. The small algebra/archive validators need SciPy, and the nonlinear AD
validator needs JAX. Retained manifests record QR/SVD library versions and BLAS/LAPACK
backend metadata. Applying the reloaded packed Householder array uses a copy-on-write
mapping (`mmap_mode='c'`), because LAPACK temporarily writes/restores reflector
diagonals; the archive source documents that detail.
