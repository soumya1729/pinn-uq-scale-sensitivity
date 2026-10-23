"""Copy frozen inputs into this independent study; never modify the source project."""
import ast
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\dev\myshortened\pinn_uq")


def main():
    cohort = json.loads((SOURCE / "analysis/round2/cohort.json").read_text())
    files, runs = [], []

    def copy(source, destination, expected=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ValueError(f"Input must be a real file: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        wanted = expected or hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != wanted:
            raise ValueError(f"Copied input does not match its source: {destination}")
        files.append(dict(path=destination.relative_to(ROOT).as_posix(),
                          bytes=destination.stat().st_size, sha256=digest,
                          copied_from=str(source)))

    for entry in cohort["cohort"]:
        tag = entry["tag"]
        summary_path = SOURCE / f"analysis/round2/outputs/production/{tag}_round2.json"
        summary = json.loads(summary_path.read_text())
        group = next(i for i, tags in enumerate(cohort["groups"]) if tag in tags)
        runs.append(dict(tag=tag, group=group, config=summary["config"]))
        for record in entry["files"]:
            source = SOURCE / record["path"]
            subdir = "baseline" if source.parent.name == "baseline" else f"checkpoints/{tag}"
            copy(source, ROOT / "inputs" / subdir / source.name, record["sha256"])
        for suffix in (".json", ".csv", "_fields.npz", "_spectral.npz"):
            source = SOURCE / f"analysis/round2/outputs/production/{tag}_round2{suffix}"
            copy(source, ROOT / "inputs/round2" / source.name)

    roots_source = SOURCE / "analysis/round3_local/outputs/refined_roots.csv"
    copy(roots_source, ROOT / "inputs/refined_roots.csv")
    roots = {}
    with roots_source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            roots[row["tag"]] = [dict(setting="trained_centre_refined_root",
                name="trained_centre_refined_root", gamma=float(row["root_gamma"]),
                lower=float(row["root_low"]), upper=float(row["root_high"]),
                saved_bracket_low=float(row["saved_bracket_low"]),
                saved_bracket_high=float(row["saved_bracket_high"]))]
    local = ROOT / "outputs/local"
    local.mkdir(parents=True, exist_ok=True)
    (local / "root_settings.json").write_text(json.dumps(roots, indent=2) + "\n")

    # The three model/residual functions are copied verbatim, not imported from
    # another project. Their original training-weight and PDE conventions remain.
    source = SOURCE / "analysis/n1/n1_sensitivity.py"
    original = source.read_text(encoding="utf-8")
    wanted = {"make_model", "residual_blocks", "build_jacobian"}
    bodies = [ast.get_source_segment(original, node) for node in ast.parse(original).body
              if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(bodies) == 3
    helper = ('"""Frozen checkpoint model and raw residual definitions, copied verbatim.\n'
              'Source: original n1_sensitivity.py; no external project import.\n"""\n'
              'import jax\nimport jax.numpy as jnp\nimport numpy as np\n'
              'from jax.flatten_util import ravel_pytree\n\n'
              'jax.config.update("jax_enable_x64", True)\n'
              'jax.config.update("jax_default_matmul_precision", "highest")\n\n'
              + '\n\n'.join(bodies) + '\n')
    (ROOT / "scripts/model_residual.py").write_text(helper, encoding="utf-8")

    assert len(runs) == len({r["tag"] for r in runs}) == 17
    manifest = dict(study="pinn_theta_followup", created="2026-10-14",
                    runs=runs, groups=cohort["groups"], files=files,
                    conventions="Frozen checkpoints and original raw residual weights; copied inputs only.",
                    source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for directory in ("outputs/evidence", "outputs/predictions", "logs", "cluster_records", "verification"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    print(f"Copied and hash-checked {len(files)} real input files for {len(runs)} checkpoints.")
    print(f"Independent study root: {ROOT}")
    print(f"Input size: {sum(f['bytes'] for f in files)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
