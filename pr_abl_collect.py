"""Aggregate the RSW / conditioning ablation: mean +- s.d. over seeds, paired differences to the full model,
CSV + LaTeX checkbox table.   python pr_abl_collect.py --results results_abl_norsvin"""
import argparse, glob, os, re
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_abl_norsvin"); a = ap.parse_args()
M = [("nps_mae_rel_abs", "NPS$_\\mathrm e$"), ("nps_shape_err_abs", "NPS$_\\mathrm s$"), ("lpips", "LPIPS"),
     ("dists", "DISTS"), ("noise_std_err_abs", "$\\sigma_\\mathrm{err}$")]
CFG = [  # name, RSW, anatomy, signal transfer S_eta, z (domain embedding), FiLM adapted, label
    ("zeroshot", 0, 1, 1, "mean", 0, "zero-shot (no target data)"),
    ("noCond", 1, 1, 1, 0, 0, "no domain conditioning"),
    ("noZ", 1, 1, 1, 0, 1, "$-$ embedding $\\bm z$"),
    ("noFiLM", 1, 1, 1, 1, 0, "$-$ FiLM adaptation"),
    ("noRSW", 0, 1, 1, 1, 1, "$-$ ray-wise weighting"),
    ("noAnat", 1, 0, 1, 1, 1, "$-$ anatomy conditioning"),
    ("noST", 1, 1, 0, 1, 1, "$-$ signal transfer"),
    ("full", 1, 1, 1, 1, 1, "\\textbf{full}")]
rows = []
for f in sorted(glob.glob(os.path.join(a.results, "*", "summary.csv"))):
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        m = re.match(r"abl_(\w+?)_s(\d+)$", str(r["method"]))
        if m:
            rows.append(dict(cfg=m.group(1), seed=int(m.group(2)), tgt=r["tgt"], **{k: r[k] for k, _ in M}))
D = pd.DataFrame(rows)
if D.empty:
    raise SystemExit("no abl_* runs found")
D.to_csv(os.path.join(a.results, "ablation_per_seed.csv"), index=False)
out, tex = [], []
Y, X = "\\cmark", "\\xmark"
for name, rsw, anat, st, z, film, label in CFG:
    line = [label, Y if rsw else X, Y if anat else X, Y if st else X, {1: Y, 0: X, "mean": "mean"}[z], Y if film else X]
    for t in ["10_IR", "25_IR"]:
        for k, _ in M[:3]:
            v = D[(D.cfg == name) & (D.tgt == t)][k]
            full = D[(D.cfg == "full") & (D.tgt == t)].set_index("seed")[k]
            mine = D[(D.cfg == name) & (D.tgt == t)].set_index("seed")[k]
            if v.empty:
                line.append("--"); continue
            s = f"{v.mean():.3f}" + (f"$\\pm${v.std(ddof=1):.3f}" if len(v) > 1 else "")
            line.append(s)
            common = full.index.intersection(mine.index)
            out.append(dict(cfg=name, tgt=t, metric=k, mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                            n=len(v), paired_delta_vs_full=(mine[common] - full[common]).mean() if name != "full" and len(common) else np.nan,
                            full_better_in=int((mine[common] > full[common]).sum()) if name != "full" else np.nan,
                            n_paired=len(common)))
    tex.append(" & ".join(line) + " \\\\")
pd.DataFrame(out).to_csv(os.path.join(a.results, "ablation_summary.csv"), index=False)
hdr = " & ".join(h for _ in range(2) for _, h in M[:3])
T = ("\\begin{tabular}{l|ccccc|ccc|ccc}\n\\toprule\n & RSW & Anat. & $S_\\eta$ & $\\bm z$ & FiLM & \\multicolumn{3}{c|}{Norsvin 10\\,\\%} & "
     "\\multicolumn{3}{c}{Norsvin 25\\,\\%} \\\\\n & & & & & & " + hdr + " \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}")
open(os.path.join(a.results, "tab_ablation_rsw_film.tex"), "w").write(T)
print(pd.DataFrame(out).round(4).to_string())
print(f"[collect] {a.results}/ablation_summary.csv, ablation_per_seed.csv, tab_ablation_rsw_film.tex")
