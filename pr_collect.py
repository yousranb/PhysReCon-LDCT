"""Collect few-shot results: metric vs number of support units k, per adaptation mode, mean +- std over the
random support draws (and over folds when several result folders are given).

Inputs: evaluate.py result folders (summary.csv; categories A-C) and/or pr_eval.py outputs (*_summary.csv
per-unit means; categories D-F). Method names follow pr_adapt.py: physrecon_<mode>_k<k>[_d<draw>].

    python pr_collect.py --evaluate results_fs/100_IR_to_10_IR_f0 results_fs/100_IR_to_25_IR_f0 \
        --pr_eval results_fs_pr --out fewshot_norsvin
"""
import argparse
import glob
import os
import re

import pandas as pd

PAT = re.compile(r"^physrecon_(zero|z|zref|zfilm|full|scratch)_k(\d+)(?:_d(\d+))?$")
DEFAULT_METRICS = ["nps_mae_abs", "nps_shape_err_abs", "f_mean_err_abs", "noise_std_err_abs", "noise_map_corr",
                   "lf_psnr", "ssim_01", "dists_01", "w1", "swd", "mmd2", "hu_response_err", "edge_width_err_lo",
                   "edge_width_err_hi", "lf_psnr_vs_ndct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", nargs="*", default=[], help="evaluate.py output folders (with summary.csv)")
    ap.add_argument("--pr_eval", nargs="*", default=[], help="pr_eval.py output folders")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--out", default="fewshot")
    a = ap.parse_args()
    frames = []
    for d in a.evaluate:
        f = os.path.join(d, "summary.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = df.rename(columns={df.columns[0]: "method"})
            frames.append(df)
    for d in a.pr_eval:
        for f in glob.glob(os.path.join(d, "*_summary.csv")):
            df = pd.read_csv(f)
            df = df.rename(columns={df.columns[0]: "method"})
            m = re.match(r"(.+)_to_(.+)_f(\d+)_summary.csv", os.path.basename(f))
            df["src"], df["tgt"], df["fold"] = m.group(1), m.group(2), int(m.group(3))
            df = df.rename(columns={c: c[:-5] for c in df.columns if c.endswith("_mean")})
            frames.append(df)
    if not frames:
        raise SystemExit("no input found")
    df = pd.concat(frames, ignore_index=True)
    df = df.groupby(["method", "tgt", "fold"], as_index=False).first()
    rows = []
    for _, r in df.iterrows():
        m = PAT.match(str(r["method"]))
        if m:
            rows.append({**r.to_dict(), "mode": m.group(1), "k": int(m.group(2)), "draw": int(m.group(3) or 0)})
        else:
            rows.append({**r.to_dict(), "mode": "reference", "k": -1, "draw": 0})
    df = pd.DataFrame(rows)
    mets = [m for m in a.metrics if m in df.columns]
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    fs = df[df["mode"] != "reference"]
    tab = fs.groupby(["tgt", "mode", "k"])[mets].agg(["mean", "std", "count"])
    tab.columns = [f"{c}_{s}" for c, s in tab.columns]
    tab.to_csv(os.path.join(a.out, "fewshot_table.csv"))
    ref = df[df["mode"] == "reference"].groupby(["tgt", "method"])[mets].mean()
    ref.to_csv(os.path.join(a.out, "reference_methods.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(fs.groupby(["tgt", "mode", "k"])[mets].mean().round(4).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [m for m in ["nps_mae_abs", "noise_std_err_abs", "swd", "edge_width_err_lo"] if m in mets] or mets[:4]
    for tgt, g in fs.groupby("tgt"):
        fig, ax = plt.subplots(1, len(show), figsize=(4 * len(show), 3.2), squeeze=False)
        for j, met in enumerate(show):
            for mode, gm in g.groupby("mode"):
                s = gm.groupby("k")[met].agg(["mean", "std"]).reset_index()
                ax[0, j].errorbar(s["k"], s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3, label=mode)
            for meth, v in ref.loc[tgt][met].items() if tgt in ref.index.get_level_values(0) else []:
                ax[0, j].axhline(v, ls=":", lw=1, color="gray")
                ax[0, j].text(ax[0, j].get_xlim()[1], v, f" {meth}", fontsize=6, va="center")
            ax[0, j].set_xlabel("support units k")
            ax[0, j].set_title(met, fontsize=9)
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"few-shot adaptation, target {tgt}")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"fewshot_{tgt}.png"), dpi=130)
    print(f"[collect] saved {a.out}/fewshot_table.csv, reference_methods.csv, fewshot_*.png")


if __name__ == "__main__":
    main()
