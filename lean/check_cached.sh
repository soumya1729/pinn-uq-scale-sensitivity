#!/usr/bin/env bash
# Source recompilation using an existing, matching Mathlib package cache.
set -eu
project=$(cd "$(dirname "$0")" && pwd)
cache=${LEAN_CACHE_PROJECT:?Set LEAN_CACHE_PROJECT to an existing Lean 4.30.0 Mathlib project}
compiler=${LEAN_COMPILER:-lean}
mkdir -p "$project/build/PaperReliability"
export LEAN_PATH="$project/build"
for package in "$cache"/.lake/packages/*/.lake/build/lib/lean; do
  LEAN_PATH="$LEAN_PATH:$package"
done
cd "$project"
printf 'Command: %s --version\n' "$compiler"
"$compiler" --version
printf 'Compiling all project modules from source; warnings are errors.\n'
for module in ModalModel CovarianceRepresentation CoarseSpectralBounds VarianceEnvelopes CoverageEnvelopes PredictiveCentres RegionalLimits; do
  printf 'Command: %s -j4 -DwarningAsError=true -o build/PaperReliability/%s.olean PaperReliability/%s.lean\n' "$compiler" "$module" "$module"
  "$compiler" -j4 -DwarningAsError=true -o "build/PaperReliability/$module.olean" "PaperReliability/$module.lean"
  printf 'DIRECT_MODULE_PASS: %s\n' "$module"
done
printf 'Command: %s -j4 -DwarningAsError=true -o build/PaperReliability.olean PaperReliability.lean\n' "$compiler"
"$compiler" -j4 -DwarningAsError=true -o build/PaperReliability.olean PaperReliability.lean
printf 'Command: %s -j4 -DwarningAsError=true Audit.lean\n' "$compiler"
"$compiler" -j4 -DwarningAsError=true Audit.lean
printf 'DIRECT_BUILD_AND_AXIOM_AUDIT: PASS\n'
