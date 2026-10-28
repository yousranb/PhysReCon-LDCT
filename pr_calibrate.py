"""Stage 0 / E0 - calibrate and validate the shared acquisition physics on piglet FBP.

Fit (TRAINING slabs only), one parameter set shared by all doses:
    I0     solved so that the INCREMENTAL variance matches:  var(n_phys) = var(real LDCT) - var(NDCT) in flat ROIs
           (fixed-point on 1/I0 scaling, geometric mean over doses)
    kappa  Hann cutoff of the reference FBP (noise texture)      grid
    se2    electronic noise variance                             grid
    objective: mean |log radial NPS(NDCT + n_phys) - log radial NPS(real LDCT)| over doses (same ROIs)
Validate (VALIDATION slabs): for each dose, simulated LDCT (NDCT + n_phys) vs real FBP LDCT, both measured as
IMAGES with the same estimator (never the difference image, which would add the full-dose noise twice):
    noise std, radial NPS (log L1), NPS anisotropy, noise std per tissue class, noise std vs radial position.
Acceptance = practical equivalence: error <= max(Monte-Carlo tolerance, pre-declared margin, see MARGINS).
Monte-Carlo test (reported as p-values; H0: "the real LDCT is one more realisation of the simulator"): with R independent
physics realisations on the same validation slices, error = median_i |stat(real) - stat(sim_i)| must not exceed
the 95th percentile of |stat(sim_i) - stat(sim_j)| (i<j). Both quantities compare two independent noise
realisations on identical anatomy, so the test does not depend on how many slabs are available.

    python -u pr_calibrate.py --data data/piglet --recon FBP --doses 5 10 25 --out pr_calib
"""
import argparse
import json
import os

import numpy as np
import torch

from physrecon import data as prdata
from physrecon import stats
from physrecon.physics import PhysicsSim

NR = 32
# Pre-declared practical-equivalence margins (decided BEFORE looking at real data). A strict Monte-Carlo test
# rejects even a perfect simulator once the validation set is large, because I0 is itself estimated from
# finite calibration data; acceptance therefore = within the Monte-Carlo tolerance OR within the margin.
MARGINS = dict(std=0.05,      # relative noise-std error (5 %)
               nps=0.15,      # mean |log radial NPS| difference (~15 % per frequency)
               aniso=0.05,    # absolute NPS anisotropy difference
               cls=0.05,      # relative noise std per tissue class
               radial=0.05)   # relative noise std per radial position (centre / middle / periphery)


def slice_stats(img, nd, dx, masks):
    """Per-slice sufficient statistics of an IMAGE (1,1,H,W): ROI NPS sum + count, per-class texture
    variance, texture variance in 3 radial position bins."""
    flat = masks[:, :4].any(1, keepdim=True)
    pos = stats.roi_positions(flat, generator=torch.Generator().manual_seed(0))
    out = {}
    if len(pos):
        r = stats.detrend(stats.extract(img, pos))
        P = (torch.fft.fft2(r).abs() ** 2).sum(0) * dx * dx / r.shape[-1] ** 2
        out["nps_sum"], out["n"] = P, len(pos)
    t = stats.texture(img, 2.0)
    out["cls"] = torch.stack([t[masks[:, c:c + 1]].var() if masks[:, c:c + 1].sum() > 200
                              else torch.tensor(float("nan"), device=img.device) for c in range(len(stats.CLASSES))])
    body = stats.body_mask(nd)
    yy, xx = torch.nonzero(body[0, 0], as_tuple=True)
    if len(yy):
        cy, cx = yy.float().mean(), xx.float().mean()
        H, W = img.shape[-2:]
        gy, gx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
        rr = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
        R = rr[body[0, 0]].quantile(0.95)
        rad = []
        for lo, hi in ((0, 0.33), (0.33, 0.66), (0.66, 1.0)):
            m = flat[0, 0] & (rr >= lo * R) & (rr < hi * R)
            rad.append(t[0, 0][m].var() if m.sum() > 200 else torch.tensor(float("nan"), device=img.device))
        out["rad"] = torch.stack(rad)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--recon", default="FBP")
    ap.add_argument("--doses", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--n_fit", type=int, default=24, help="training slices used for the fit")
    ap.add_argument("--n_val", type=int, default=40, help="max validation slices")
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--se2s", type=float, nargs="+", default=[0.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--I0", type=float, default=1e5, help="initial guess")
    ap.add_argument("--n_views", type=int, default=720)
    ap.add_argument("--n_real_val", type=int, default=20, help="simulator realisations for the Monte-Carlo test")
    ap.add_argument("--out", default="pr_calib")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = prdata.Dataset(a.data)
    ld_keys = {d: f"{d}_{a.recon}" for d in a.doses}
    nd_key = ds.nd_key(a.recon, ld_keys[a.doses[0]])
    split = prdata.benchmark_split(ds, nd_key, ld_keys[a.doses[0]], a.n_folds, a.fold, a.gap)
    snd = ds.series(nd_key)
    dx = float(snd.spacing[0])
    common = set(snd.z)
    for k in ld_keys.values():
        common &= set(ds.series(k).z)
    tr = prdata.body_z(snd, sorted(split["train"] & common))
    va = prdata.body_z(snd, sorted(split["val"] & common))
    fit_z = [tr[i] for i in np.linspace(0, len(tr) - 1, min(a.n_fit, len(tr))).astype(int)]
    val_z = [va[i] for i in np.linspace(0, len(va) - 1, min(a.n_val, len(va))).astype(int)] if va else []
    print(f"[E0] {nd_key} -> {list(ld_keys.values())}; fit on {len(fit_z)} training slices, "
          f"validate on {len(val_z)} validation slices; pixel {dx:.3f} mm")
    H = snd.get(fit_z[0]).shape[0]
    stats.set_roi_size(H)
    sim = PhysicsSim(H, a.I0, 10.0, 1.0, a.n_views, dev)

    def load(z):
        x = torch.tensor(snd.get(z), device=dev)[None, None]
        return x, stats.class_masks(x)

    # ---------------------------------------------------------------- real statistics (fit slices)
    real_fit = {d: [] for d in a.doses}
    nd_var = {}
    cache = {}
    for z in fit_z:
        x, m = load(z)
        cache[z] = (x, m)
        sx = slice_stats(x, x, dx, m)
        nd_var.setdefault("v", []).append(sx)
        for d in a.doses:
            y = torch.tensor(ds.series(ld_keys[d]).get(z), device=dev)[None, None]
            real_fit[d].append(slice_stats(y, x, dx, m))

    def nps_mean(sl):
        n = sum(s["n"] for s in sl if "nps_sum" in s)
        if n == 0:
            raise SystemExit("[E0] no flat ROI found in the slices (check HU scale / masks)")
        return sum(s["nps_sum"] for s in sl if "nps_sum" in s) / n

    var_nd = nps_mean(nd_var["v"]).sum().item()
    var_real = {d: nps_mean(real_fit[d]).sum().item() for d in a.doses}
    rad_real = {d: stats.radial_nps(nps_mean(real_fit[d]), NR).cpu().numpy() for d in a.doses}
    print("[E0] ROI noise power (sum of NPS): NDCT %.4g | " % var_nd +
          " | ".join(f"{d}%: {var_real[d]:.4g}" for d in a.doses))
    for d in a.doses:
        if var_real[d] <= var_nd:
            print(f"[E0][warn] {d}%: real LDCT noise not above NDCT noise - incremental variance undefined")

    def simulate(zs, d, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        out = []
        for z in zs:
            x, m = cache[z] if z in cache else load(z)
            p = sim.sinogram(x, dx, keys=[("nd", z)])
            n = sim.noise(p, torch.tensor([d / 100.0], device=dev), dx, generator=g)
            out.append((x + n, x, m, n))
        return out

    # ---------------------------------------------------------------- fit
    best = None
    for kappa in a.kappas:
        for se2 in a.se2s:
            sim.kappa, sim.se2 = kappa, se2
            for it in range(3):
                ratios = []
                for d in a.doses:
                    inc_real = var_real[d] - var_nd
                    sl = [slice_stats(n, x, dx, m) for _, x, m, n in simulate(fit_z, d)]
                    inc_sim = nps_mean(sl).sum().item()
                    if inc_real > 0:
                        ratios.append(inc_sim / inc_real)
                sim.I0 *= float(np.exp(np.mean(np.log(ratios))))       # var ~ 1/I0 (quantum-dominated)
            err, per = 0.0, {}
            for d in a.doses:
                sl = [slice_stats(y, x, dx, m) for y, x, m, _ in simulate(fit_z, d)]
                rs = stats.radial_nps(nps_mean(sl), NR).cpu().numpy()
                e = float(np.mean(np.abs(np.log(rs[1:] + 1e-12) - np.log(rad_real[d][1:] + 1e-12))))
                per[d] = e
                err += e / len(a.doses)
            print(f"[fit] kappa={kappa:.2f} se2={se2:g} I0={sim.I0:.4g}  log-NPS L1 per dose "
                  + " ".join(f"{d}%:{per[d]:.3f}" for d in a.doses) + f"  mean {err:.3f}")
            if best is None or err < best[0]:
                best = (err, kappa, se2, sim.I0, per)
    err, kappa, se2, I0, per = best
    sim.kappa, sim.se2, sim.I0 = kappa, se2, I0
    params = dict(I0=I0, se2=se2, kappa=kappa, n_views=a.n_views, recon=a.recon, data=a.data,
                  fit_error=per, fit_slices=len(fit_z))
    print(f"[E0] best: kappa={kappa} se2={se2} I0={I0:.4g}  (mean log-NPS L1 {err:.3f})")

    # ---------------------------------------------------------------- validation + acceptance
    report = {}
    if val_z:
        def agg(sl, idx):
            sub = [sl[i] for i in idx]
            P = nps_mean(sub)
            rad = stats.radial_nps(P, NR).cpu().numpy()
            a4 = stats.polar_nps(P, 8, 4).view(8, 4).sum(0).cpu().numpy()
            ang = np.exp(2j * (np.arange(4) + 0.5) * np.pi / 4)
            cls = np.sqrt(np.nanmean(np.stack([s["cls"].cpu().numpy() for s in sub]), 0))
            radial = np.sqrt(np.nanmean(np.stack([s["rad"].cpu().numpy() for s in sub if "rad" in s]), 0))
            return dict(std=float(np.sqrt(P.sum().item())) / (dx * P.shape[-1]), rad=rad,
                        aniso=float(np.abs((a4 * ang).sum()) / max(a4.sum(), 1e-12)), cls=cls, radial=radial)

        def dist(s1, s2):
            return dict(std=abs(s1["std"] - s2["std"]) / max(s2["std"], 1e-9),
                        nps=float(np.mean(np.abs(np.log(s1["rad"][1:] + 1e-12) - np.log(s2["rad"][1:] + 1e-12)))),
                        aniso=abs(s1["aniso"] - s2["aniso"]),
                        cls=float(np.nanmean(np.abs(s1["cls"] - s2["cls"]) / s2["cls"])),
                        radial=float(np.nanmean(np.abs(s1["radial"] - s2["radial"]) / s2["radial"])))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(a.doses), figsize=(4.2 * len(a.doses), 3.4), squeeze=False)
        for j, d in enumerate(a.doses):
            real_sl, sim_sl = [], []
            for z in val_z:
                x, m = load(z)
                y = torch.tensor(ds.series(ld_keys[d]).get(z), device=dev)[None, None]
                real_sl.append(slice_stats(y, x, dx, m))
            allidx = list(range(len(val_z)))
            R = agg(real_sl, allidx)
            sims = []
            for r in range(a.n_real_val):
                sl = [slice_stats(y, x, dx, m) for y, x, m, _ in simulate(val_z, d, seed=1000 + r)]
                sims.append(agg(sl, allidx))
            S = sims[0]
            # statistic: median over simulator realisations of |real - sim_i|; its null distribution is
            # obtained leave-one-out: treat each sim_k as the "real" data and recompute the same median
            errs = {k: float(np.median([dist(Si, R)[k] for Si in sims])) for k in dist(S, R)}
            null = {k: [] for k in errs}
            for kk in range(len(sims)):
                others = [dist(sims[ii], sims[kk]) for ii in range(len(sims)) if ii != kk]
                for k in errs:
                    null[k].append(float(np.median([o[k] for o in others])))
            rep = {}
            for k in errs:
                tol = float(np.nanmax(null[k]))     # = 1 - 1/R quantile of the leave-one-out null
                pval = (1 + sum(v >= errs[k] for v in null[k])) / (len(null[k]) + 1)
                marg = MARGINS[k]
                rep[k] = dict(error=float(errs[k]), mc_tolerance=tol, p_value=float(pval), margin=marg,
                              indistinguishable=bool(errs[k] <= tol),
                              passed=bool(errs[k] <= max(tol, marg)))
            rep["std_sim"], rep["std_real"] = S["std"], R["std"]
            rep["radial_sim"], rep["radial_real"] = S["radial"].tolist(), R["radial"].tolist()
            report[d] = rep
            print(f"[E0][val {d}%] noise std sim {S['std']:.2f} vs real {R['std']:.2f} HU | " +
                  " | ".join(f"{k}: {rep[k]['error']:.3f} (MC tol {rep[k]['mc_tolerance']:.3f}, p={rep[k]['p_value']:.2f}, "
                             f"margin {rep[k]['margin']}) {'PASS' if rep[k]['passed'] else 'FAIL'}" for k in errs))
            f = np.arange(NR) / NR * 0.5 / dx
            ax[0, j].semilogy(f, R["rad"], label="real FBP LDCT")
            ax[0, j].semilogy(f, S["rad"], "--", label="NDCT + n_phys")
            ax[0, j].set_title(f"{d}% dose (validation slabs)")
            ax[0, j].set_xlabel("spatial frequency (1/mm)")
        ax[0, 0].set_ylabel("radial NPS (HU$^2$mm$^2$)")
        ax[0, 0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "e0_radial_nps.png"), dpi=130)
    else:
        print("[E0] no validation slices (multi-subject data?) - acceptance test skipped")
    params["validation"] = report
    params["accepted"] = {str(d): all(v["passed"] for k, v in r.items() if isinstance(v, dict) and "passed" in v)
                          for d, r in report.items()}
    json.dump(params, open(os.path.join(a.out, "physics.json"), "w"), indent=1, default=float)
    print(f"[E0] saved {a.out}/physics.json  accepted per dose: {params['accepted']}")


if __name__ == "__main__":
    main()
