"""Evaluation categories D (distributional), F (signal transfer) and E (anatomical identity) for prediction
runs of one (src, tgt, fold) - complements evaluate.py (categories A, B, C).

Statistics are computed in matched tissue contexts (flat regions of each class, defined on the NDCT), never
on whole images, and aggregated per statistical unit: per PIG on multi-subject data (Norsvin), per slab
group of 10 slices on the single piglet (descriptive only). The real LDCT is the reference; 'identity'
(the NDCT itself) is reported as the no-degradation reference.

D  W1, KS, |d skewness|, |d excess kurtosis| of the undecimated Haar detail coefficients per (scale,
   orientation, class) [mean over groups]; SWD on 81-D multiscale neighbourhoods (3 scales x 3
   orientations x 3x3), standardised per class with the real statistics; MMD^2 (unbiased, Gaussian
   kernel, median heuristic) on per-ROI log radial NPS shapes.
F  HU response error per class; edge-spread width error (low/high contrast boundaries); boundary contrast
   ratio error  |C(gen)/C(ndct) - C(real)/C(ndct)|.  (On registered data, residual misalignment of the
   real LDCT broadens its profiles slightly - stated as a limitation.)
E  LF-PSNR of the output against its own NDCT input (G4, body) - anatomical identity, alignment-free.

    python -u pr_eval.py --runs runs_norsvin/physrecon_z_k3_100_IR_to_10_IR_f0_s0 runs_norsvin/cddpm_ex_cal16_100_IR_to_10_IR_f0_s0 \
        --out results_norsvin_pr
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats as sst

import datasets
from physrecon import data as prdata
from physrecon import stats, wavelet


def mmd2(X, Y, gamma=None):
    X, Y = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(Y, dtype=torch.float64)
    if len(X) < 4 or len(Y) < 4:
        return float("nan")
    Z = torch.cat([X, Y])
    D = torch.cdist(Z, Z) ** 2
    if gamma is None:
        gamma = 1.0 / max(float(D[D > 0].median()), 1e-12)
    K = torch.exp(-gamma * D)
    n, m = len(X), len(Y)
    kxx = (K[:n, :n].sum() - K[:n, :n].diag().sum()) / (n * (n - 1))
    kyy = (K[n:, n:].sum() - K[n:, n:].diag().sum()) / (m * (m - 1))
    kxy = K[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def neighbourhoods(det, pos):
    """det: list of J tensors (1,1,3,H,W); pos (K,2) -> (K, J*3*9) 3x3 neighbourhoods of every band."""
    feats = []
    for d in det:
        b = d[0, 0]                                      # (3,H,W)
        for dy in (-1, 0, 1):
            for dx_ in (-1, 0, 1):
                feats.append(b[:, pos[:, 0] + dy, pos[:, 1] + dx_].T)     # (K,3)
    return torch.cat(feats, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results_pr")
    ap.add_argument("--stride", type=int, default=2, help="use every n-th test slice")
    ap.add_argument("--coef_per_slice", type=int, default=1500)
    ap.add_argument("--nb_per_slice", type=int, default=300)
    ap.add_argument("--n_proj", type=int, default=128)
    ap.add_argument("--no_identity", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [json.load(open(os.path.join(r, "config.json"))) for r in a.runs]
    c0 = cfgs[0]
    for c in cfgs[1:]:
        assert (c["src"], c["tgt"], c["fold"]) == (c0["src"], c0["tgt"], c0["fold"]), "runs must share src/tgt/fold"
    ds = prdata.Dataset(c0["data"])
    zl = ds.pair_z(c0["src"], c0["tgt"])
    snd, sld = ds.series(c0["src"]), ds.series(c0["tgt"])
    dx = float(snd.spacing[0])
    stats.set_roi_size(snd.get(zl[0]).shape[0])
    test = set(c0["test_idx"])
    for c in cfgs[1:]:
        test &= set(c["test_idx"])
    test = sorted(test)
    test = [i for i in test if (snd.get(zl[i])[::8, ::8] > -500).mean() >= 0.05][:: a.stride]
    preds = {}
    for r, c in zip(a.runs, cfgs):
        P = np.load(os.path.join(r, "pred.npy"), mmap_mode="r")
        posm = {int(k): j for j, k in enumerate(c["test_idx"])}
        name = c["model"]
        preds[name] = (P, posm, tuple(c.get("hu_range", (-1024, 2048))))
    methods = ["real"] + ([] if a.no_identity else ["identity"]) + list(preds)
    lo = max(rng[0] for _, _, rng in preds.values()) if preds else -1024
    hi = min(rng[1] for _, _, rng in preds.values()) if preds else 2048
    multi = zl[0] >= datasets.SUBJECT_Z
    names = snd.subjects if multi else None
    print(f"[pr_eval] {c0['src']} -> {c0['tgt']} fold {c0['fold']}: {len(test)} test slices, methods {methods}, "
          f"common HU range [{lo:g}, {hi:g}], unit = {'pig' if multi else 'slab group'}")

    def group_of(i):
        z = zl[i]
        if multi:
            s = int(z // datasets.SUBJECT_Z)
            return names[s - 1] if names else str(s)
        return f"slab{test.index(i) // 10 if i in test else 0}"

    acc = {}       # group -> method -> lists
    rng = np.random.default_rng(0)
    for t, i in enumerate(test):
        z = zl[i]
        x = torch.tensor(snd.get(z), device=dev)[None, None]
        imgs = {"real": torch.tensor(sld.get(z), device=dev)[None, None]}
        if not a.no_identity:
            imgs["identity"] = x.clone()
        for nme, (P, posm, _) in preds.items():
            imgs[nme] = torch.tensor(np.asarray(P[posm[i]], np.float32), device=dev)[None, None]
        for k in imgs:
            imgs[k] = imgs[k].clamp(lo, hi)
        xc = x.clamp(lo, hi)
        masks = stats.class_masks(x)
        flat = masks[:, :4].any(1, keepdim=True)
        g = group_of(i)
        A = acc.setdefault(g, {m: dict(coef={}, nb=[], roi=[], cls=[], prof={"lo": [], "hi": []}, lf=[])
                               for m in methods})
        # shared sampling locations for all methods
        cpos = {}
        for c in range(4):
            yx = torch.nonzero(masks[0, c])
            if len(yx) > 50:
                cpos[c] = yx[torch.as_tensor(rng.choice(len(yx), min(a.coef_per_slice, len(yx)), replace=False),
                                             device=yx.device)]
        fyx = torch.nonzero(stats.erode(flat, 1)[0, 0])
        fyx = fyx[(fyx[:, 0] > 2) & (fyx[:, 1] > 2) & (fyx[:, 0] < x.shape[-2] - 3) & (fyx[:, 1] < x.shape[-1] - 3)]
        npos = fyx[torch.as_tensor(rng.choice(len(fyx), min(a.nb_per_slice, len(fyx)), replace=False),
                                   device=fyx.device)] if len(fyx) > 20 else None
        rpos = stats.roi_positions(flat, generator=torch.Generator().manual_seed(t))
        pts = {k: stats.edge_points(x, k, K=512, generator=torch.Generator().manual_seed(t)) for k in ("lo", "hi")}
        body = stats.body_mask(x)
        gx = stats.gauss(xc, 4.0)
        for m, img in imgs.items():
            S = A[m]
            det, _ = wavelet.forward(img)
            for c, p in cpos.items():
                for j, dj in enumerate(det):
                    for o in range(3):
                        S["coef"].setdefault((j, o, c), []).append(dj[0, 0, o][p[:, 0], p[:, 1]].cpu().numpy())
            if npos is not None:
                S["nb"].append(neighbourhoods(det, npos).cpu().numpy())
            if len(rpos):
                r = stats.detrend(stats.extract(img, rpos))
                Pn = torch.fft.fft2(r).abs() ** 2
                rad = torch.stack([stats.radial_nps(q, 16) for q in Pn])
                rad = torch.log(rad / rad.sum(1, keepdim=True) + 1e-12)
                S["roi"].append(rad.cpu().numpy())
            S["cls"].append([float(img[stats.erode(masks[:, c:c + 1], 2)].mean()) if stats.erode(masks[:, c:c + 1], 2).sum() > 200
                             else np.nan for c in range(len(stats.CLASSES))])
            for k in ("lo", "hi"):
                prof = stats.mean_profile(img, pts[k])
                if prof is not None:
                    S["prof"][k].append(prof.cpu().numpy())
            mse = ((stats.gauss(img, 4.0) - gx)[body] ** 2).mean().item()
            S["lf"].append(min(100.0, 10 * np.log10((hi - lo) ** 2 / max(mse, 1e-8))))   # capped (identity = inf)
        if (t + 1) % 25 == 0:
            print(f"[pr_eval] {t + 1}/{len(test)} slices", flush=True)

    # ------------------------------------------------------------------ metrics per group
    # NDCT profiles (for contrast ratios) = 'identity' if present, else recomputed lazily
    rows = []
    for g, A in acc.items():
        R = A["real"]
        Rnb = np.concatenate(R["nb"]) if R["nb"] else None
        mu, sd = (Rnb.mean(0), Rnb.std(0) + 1e-6) if Rnb is not None else (None, None)
        Rroi = np.concatenate(R["roi"]) if R["roi"] else None
        nd_prof = {k: np.mean(A["identity"]["prof"][k], 0) if "identity" in A and A["identity"]["prof"][k] else None
                   for k in ("lo", "hi")}
        for m in methods:
            if m == "real":
                continue
            M = A[m]
            row = dict(group=g, method=m, n_slices=len(M["lf"]))
            w1, ks, dsk, dku = [], [], [], []
            for key, rv in R["coef"].items():
                if key not in M["coef"]:
                    continue
                a_, b_ = np.concatenate(M["coef"][key]), np.concatenate(rv)
                if len(a_) < 100:
                    continue
                w1.append(sst.wasserstein_distance(a_, b_))
                ks.append(sst.ks_2samp(a_, b_).statistic)
                dsk.append(abs(sst.skew(a_) - sst.skew(b_)))
                dku.append(abs(sst.kurtosis(a_) - sst.kurtosis(b_)))
            row.update(w1=np.mean(w1) if w1 else np.nan, ks=np.mean(ks) if ks else np.nan,
                       d_skew=np.mean(dsk) if dsk else np.nan, d_kurt=np.mean(dku) if dku else np.nan)
            if Rnb is not None and M["nb"]:
                Mnb = np.concatenate(M["nb"])
                row["swd"] = float(stats.swd(torch.tensor((Mnb - mu) / sd), torch.tensor((Rnb - mu) / sd),
                                             a.n_proj, generator=torch.Generator().manual_seed(0)))
            if Rroi is not None and M["roi"]:
                Mroi = np.concatenate(M["roi"])
                n = min(len(Mroi), len(Rroi), 2000)
                row["mmd2"] = mmd2(Mroi[rng.choice(len(Mroi), n, replace=False)], Rroi[rng.choice(len(Rroi), n, replace=False)])
            cm, cr = np.nanmean(np.array(M["cls"], float), 0), np.nanmean(np.array(R["cls"], float), 0)
            row["hu_response_err"] = float(np.nanmean(np.abs(cm - cr)))
            for k in ("lo", "hi"):
                if M["prof"][k] and R["prof"][k]:
                    pm = torch.tensor(np.mean(M["prof"][k], 0))
                    pr = torch.tensor(np.mean(R["prof"][k], 0))
                    wm, cmk = stats.esf_stats(pm)
                    wr, crk = stats.esf_stats(pr)
                    row[f"edge_width_err_{k}"] = abs(float(wm) - float(wr))
                    if nd_prof[k] is not None:
                        _, cn = stats.esf_stats(torch.tensor(nd_prof[k]))
                        row[f"contrast_ratio_err_{k}"] = abs(float(cmk) / float(cn) - float(crk) / float(cn))
            row["lf_psnr_vs_ndct"] = float(np.mean(M["lf"]))
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    tag = f"{c0['src']}_to_{c0['tgt']}_f{c0['fold']}"
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"{tag}_per_unit.csv"), index=False)
    num = [c for c in df.columns if c not in ("group", "method")]
    summ = df.groupby("method")[num].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.to_csv(os.path.join(a.out, f"{tag}_summary.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(df.groupby("method")[num].mean().round(4).to_string())
    print(f"[pr_eval] saved {a.out}/{tag}_per_unit.csv and _summary.csv (mean/std over {df['group'].nunique()} units)")


if __name__ == "__main__":
    main()
