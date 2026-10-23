#!/bin/bash
#SBATCH --job-name=acref
#SBATCH --partition=gpu
#SBATCH --qos=gpu_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# AC reference-resolution validity check: re-evaluate the AC full-Laplace ceiling metric against a
# 2048-mode ETDRK4 reference with spectral (trig) x-interpolation, reusing the existing MAPs. If the
# ceiling sigma-ratio, shock meanZ, and bulk rel-L2 move <few%, the paper's 512-mode + bilinear
# reference was adequate. Outputs to results/ac_ref2048_*/ so the canonical results are untouched.
# Requires the updated common.py (AC_REF_MODES / AC_REF_SPECTRAL knobs; defaults unchanged).
ENVSH="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"; source "$ENVSH"
export RAW_METRIC=1 REUSE_MAP=1 AC_REF_MODES=2048 AC_REF_SPECTRAL=1

PDE=allen_cahn ARCH=mlp     SEED=0 FOURIER_SCALE=1 REL_L2_GATE=0.03 \
  MAP_PKL=results/allen_cahn_mlp_seed0/allen_cahn_mlp_seed0_params.pkl \
  OUTDIR=results/ac_ref2048_mlp_seed0 python m4_full_laplace.py
PDE=allen_cahn ARCH=fourier SEED=0 FOURIER_SCALE=1 REL_L2_GATE=0.03 \
  MAP_PKL=results/allen_cahn_fourier_seed0/allen_cahn_fourier_seed0_params.pkl \
  OUTDIR=results/ac_ref2048_fourier_seed0 python m4_full_laplace.py
echo ACREF DONE
