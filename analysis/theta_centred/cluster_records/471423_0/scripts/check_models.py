"""Check one saved model per family without constructing a parameter Jacobian."""
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from model_residual import make_model, residual_blocks
from theta_pass import early_checks

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("burgers_mlp_seed2", "burgers_fourier_seed2",
        "allen_cahn_mlp_seed1", "allen_cahn_fourier_seed1")


def main():
    results = {}
    for tag in TAGS:
        run = ROOT / "inputs/checkpoints" / tag
        metadata = json.loads((run / f"{tag}_map.json").read_text(encoding="utf-8-sig"))
        config = metadata["provenance"]["config"]
        with (run / f"{tag}_params.pkl").open("rb") as stream:
            params = pickle.load(stream)
        with np.load(run / f"{tag}_fields.npz") as archive:
            data = {key: archive[key] for key in
                    ("xs", "ts", "x_u", "t_b", "x_f", "t_f", "u_pred", "u_exact", "grad_abs")}
        for key in ("u_pred", "u_exact", "grad_abs"):
            data[key] = data[key].ravel()
        fourier = None
        if config["arch"] == "fourier":
            with np.load(run / f"{tag}_fourier_B.npz") as archive:
                fourier = {key: jnp.asarray(archive[key]) for key in ("B_X", "B_T")}
        theta, u = make_model(params, config, fourier)
        blocks = residual_blocks(u, config, data)
        print(tag, flush=True)
        results[tag] = early_checks(theta, u, data, blocks,
                                   metadata["final_loss"], 100, 200)
    output = ROOT / "verification/checkpoint_models.json"
    output.write_text(json.dumps(dict(status="PASS", checks=results), indent=2) + "\n")
    print("Four-family prediction and residual-loss checks passed; no Jacobian constructed.")


if __name__ == "__main__":
    main()
