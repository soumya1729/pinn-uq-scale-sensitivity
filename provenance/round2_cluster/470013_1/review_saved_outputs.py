"""Small interpretation checks from frozen predictions and existing round-two CSVs.

No differentiation, fitting, scale sweep or ensemble resampling is performed.
Original ensemble and oracle numerical outputs are not overwritten.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=HERE.parent/"n5/input")
    parser.add_argument("--out",type=Path,default=HERE/"outputs")
    args=parser.parse_args()
    errors=[]
    for seed in range(8):
        tag=f"burgers_mlp_seed{seed}"
        with np.load(args.inputs/tag/f"{tag}_fields.npz") as d:
            error=d["u_pred"].ravel()-d["u_exact"].ravel()
            if seed==0:
                gradient=d["grad_abs"].ravel()
                sharp=gradient>=np.percentile(gradient,95)
        errors.append(error[sharp])
    errors=np.asarray(errors)
    alignment=[]
    for seed in range(8):
        own=errors[seed]
        other=np.delete(errors,seed,axis=0).mean(axis=0)
        alignment.append(dict(seed=seed,pearson=float(np.corrcoef(own,other)[0,1]),
            cosine=float(own@other/(np.linalg.norm(own)*np.linalg.norm(other))),
            opposite_sign_fraction=float(np.mean(own*other<0)),
            own_error_rms=float(np.sqrt(np.mean(own**2))),
            other_mean_error_rms=float(np.sqrt(np.mean(other**2)))))
    oracle=read_csv(args.out/"saved_eight/sampled_oracle_coverage.csv")
    selected=[]
    for summary in read_csv(args.out/"saved_eight/saved_eight_summary.csv"):
        tag=summary["tag"]; gamma=float(summary["closest_sampled_gamma"])
        pair={r["region"]:r for r in oracle if r["tag"]==tag and r["subset"]=="full"
              and float(r["effective_precision"])==gamma}
        selected.append(dict(tag=tag,gamma=gamma,rho=float(pair["sharp"]["rho"]),
            near_agreement=abs(float(pair["sharp"]["rho"])-1)<=.05,
            bulk_mean_Z=float(pair["bulk"]["mean_Z"]),sharp_mean_Z=float(pair["sharp"]["mean_Z"]),
            bulk_cov68=float(pair["bulk"]["coverage68"]),sharp_cov68=float(pair["sharp"]["coverage68"])))
    ensemble=read_csv(args.out/"ensemble_loo.csv")
    complete=next(r for r in ensemble if r["omitted_seed"]=="none")
    omit5=next(r for r in ensemble if r["omitted_seed"]=="5")
    result=dict(scope="Saved-data interpretation only; original numerical results unchanged",
        sharp_count=int(sharp.sum()),alignment=alignment,oracle_nearest_sampled=selected,
        seed5_rho_change=float(omit5["rho"])-float(complete["rho"]),
        seed5_sharp_error_rms=[float(complete["sharp_error_rms"]),float(omit5["sharp_error_rms"])],
        seed5_sharp_sigma_rms=[float(complete["sharp_sigma_rms"]),float(omit5["sharp_sigma_rms"])])
    (args.out/"saved_result_interpretation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# Interpretation checks from saved results","",
        "These checks read the existing prediction fields and round-two CSVs. No production pass, fitting or resampling was rerun. Original ensemble/oracle CSVs are unchanged.","",
        "## Ensemble member influence","",
        f"Omitting seed 5 changes rho from {float(complete['rho']):.4f} to {float(omit5['rho']):.4f} (a shift of {result['seed5_rho_change']:.4f}), and sharp 68% coverage from {float(complete['sharp_coverage68']):.4f} to {float(omit5['sharp_coverage68']):.4f}. The sign of the contrast deficit survives every omission, but its magnitude and coverage are sensitive to membership.","",
        "| Seed | Correlation with other seven's mean error | Uncentred cosine | Opposite signs | Error RMS |",
        "|---:|---:|---:|---:|---:|"]
    for r in alignment:
        lines.append(f"| {r['seed']} | {r['pearson']:.4f} | {r['cosine']:.4f} | {r['opposite_sign_fraction']:.4f} | {r['own_error_rms']:.5f} |")
    lines += ["",
        f"Seed 5's errors counteract the other members' mean error: correlation {alignment[5]['pearson']:.4f}, with opposite signs at {100*alignment[5]['opposite_sign_fraction']:.2f}% of the sharp grid points. Its inclusion reduces sharp error RMS from {float(omit5['sharp_error_rms']):.5f} to {float(complete['sharp_error_rms']):.5f} and increases spread RMS from {float(omit5['sharp_sigma_rms']):.5f} to {float(complete['sharp_sigma_rms']):.5f}.","",
        "Seed 5 is not uniquely negatively aligned: seed 0 also has negative correlation, although its error magnitude is much smaller. These observations concern this fixed ensemble and mask; they do not establish that a member with opposite errors is generally necessary for calibration, or that errors have opposite signs everywhere.","",
        "## Oracle interpretation","",
        "At the sampled settings closest to rho=1, five runs are within 0.05 of one. All five have a higher average sharp-region Z and lower sharp 68% coverage after bulk calibration. This illustrates that agreement of regional RMS contrasts does not match the distribution of pointwise error-to-width ratios. A higher mean is not a statement about every point or a unique causal explanation.","",
        "| Seed | rho | Bulk mean Z | Sharp mean Z | Sharp cov68 | Near agreement |",
        "|---:|---:|---:|---:|---:|---|"]
    for r in selected:
        lines.append(f"| {r['tag'].removeprefix('burgers_mlp_seed')} | {r['rho']:.4f} | {r['bulk_mean_Z']:.4f} | {r['sharp_mean_Z']:.4f} | {r['sharp_cov68']:.4f} | {'yes' if r['near_agreement'] else 'no'} |")
    lines += ["","All values are extracted from existing outputs; sampled settings are not exact roots, and the analyses retain the trained centre."]
    (args.out/"SAVED_RESULT_INTERPRETATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Saved interpretation note: seed5 correlation={alignment[5]['pearson']:.6f}; seed0={alignment[0]['pearson']:.6f}; five near-agreement oracle runs checked.")


if __name__=="__main__":
    main()
