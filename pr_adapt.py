"""Stage 2 - few-shot adaptation to a (new) domain + synthesis of the test set for the benchmark.

Modes (F_theta trunk frozen except 'full'/'scratch'):
  zero     z = mean of the training-domain embeddings (no target data)                       k = 0
  z        z = embedding of k support units (no gradient)                     PRIMARY few-shot mode
  zref     z refined by gradient on the support set
  zfilm    z + last FiLM layer + signal-transfer head refined on the support set
  full     all parameters fine-tuned on the support set (baseline)
  scratch  all parameters re-initialised and trained on the support set only (baseline)
Support units: Norsvin = pigs (all slices of a pig, capped), piglet = blocks of 10 consecutive slices,
always taken from the TRAINING part of the benchmark split. Refinement uses the stage-1 loss against the
support set's real LDCT only, with a fixed number of steps (no selection on test data).

Output: prediction runs (pred.npy + config.json) readable by evaluate.py / run_benchmark.py / pr_eval.py,
scored on exactly the same test slices as the previously benchmarked models.

    # Norsvin, 3 pigs, embedding only
    python -u pr_adapt.py --ckpt pr_runs/physrecon_main/final.pt --data data/norsvin --recon IR --doses 25 10 \
        --k 3 --mode z --runs runs_norsvin --name physrecon_z_k3
    # piglet FBP (training domain): embedding from all training slabs
    python -u pr_adapt.py --ckpt pr_runs/physrecon_main/final.pt --data data/piglet --recon FBP --doses 5 10 25 \
        --k 999 --mode z --runs runs_v3 --name physrecon
"""
import argparse
import json
import math
import os

import numpy as np
import torch

import datasets
from diffusion_common import write_pred_run
from physrecon import data as prdata
from physrecon import stats
from physrecon.losses import PRLoss
from physrecon.model import PhysReCon
from physrecon.physics import PhysicsSim
from pr_train import make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--doses", type=int, nargs="+", required=True, help="test doses to synthesise")
    ap.add_argument("--support_doses", type=int, nargs="*", default=None, help="default: = --doses")
    ap.add_argument("--k", type=int, default=3, help="support units (pigs / slabs); 0 = zero-shot")
    ap.add_argument("--draw", type=int, default=0, help="random draw of the support units")
    ap.add_argument("--mode", default="z", choices=["zero", "z", "zref", "zfilm", "heads", "full", "scratch"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--unit_len", type=int, default=10)
    ap.add_argument("--max_per_unit", type=int, default=30)
    ap.add_argument("--exclude_subjects", nargs="*", default=[], help="never used as support (e.g. badly registered)")
    ap.add_argument("--nd_recon", default=None, help="force the NDCT reconstruction (default: automatic, see data.nd_key)")
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_stride", type=int, default=1,
                    help="synthesise every n-th test slice (few-shot sweeps; all compared runs must use the same value)")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--ng_J", type=int, default=5, help="wavelet scales of the noise-transfer gains (1..32 px for 5)")
    ap.add_argument("--no_proj_resp", action="store_true", help="ablation: no projection-domain statistical weighting")
    ap.add_argument("--no_blur", action="store_true", help="ablation: no linear resolution-loss branch")
    ap.add_argument("--z_init", default="support", choices=["support", "zero", "mean"],
                    help="initial domain embedding: from the support set, zero vector (no domain information) or "
                         "mean of the training-domain embeddings (ablation of the domain conditioning)")
    ap.add_argument("--freeze_z", action="store_true", help="do not refine z during adaptation (ablation)")
    ap.add_argument("--val_every", type=int, default=50, help="early stopping on a hold-out part of the support set")
    ap.add_argument("--val_slices", type=int, default=16, help="hold-out support slices per dose")
    a = ap.parse_args()
    if a.k <= 0:
        a.mode = "zero"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    targs = ck["args"]
    cfg = dict(ck["cfg"])
    cfg.update(noise_gain=True, ng_J=a.ng_J, proj_resp=not a.no_proj_resp, sig_blur=not a.no_blur)
    model = PhysReCon(**cfg).to(dev)
    model.load_compatible(ck["model"])
    if a.mode == "scratch":
        fresh = PhysReCon(**cfg).to(dev)
        for buf in ("cnn.s_n", "cnn.s_na", "cnn.s_x", "emb.mu", "emb.sd"):   # keep normalisation constants
            fresh.get_buffer(buf).copy_(model.get_buffer(buf))
        model = fresh
    phys = ck["physics"]
    ds = prdata.Dataset(a.data)
    ld_keys = {d: f"{d}_{a.recon}" for d in a.doses}
    nd_key = ds.nd_key(a.recon, ld_keys[a.doses[0]], force=a.nd_recon)
    sup_doses = a.support_doses or a.doses
    split = prdata.benchmark_split(ds, nd_key, ld_keys[a.doses[0]], a.n_folds, a.fold, a.gap, val_frac=0.0)
    snd = ds.series(nd_key)
    dx = float(snd.spacing[0])
    H = snd.get(snd.z[0]).shape[0]
    stats.set_roi_size(H)
    sim = PhysicsSim(H, phys["I0"], phys["se2"], phys["kappa"], phys.get("n_views", 720), dev)
    name = a.name or f"physrecon_{a.mode}_k{a.k}" + (f"_d{a.draw}" if a.draw else "")
    print(f"[adapt] {name}: domain {a.recon} of {a.data}, mode {a.mode}, k={a.k} (draw {a.draw}); "
          f"physics I0={sim.I0:.4g} se2={sim.se2:g} kappa={sim.kappa}; pixel {dx:.3f} mm")

    # ------------------------------------------------------------------ support set (training part only)
    train_body = prdata.body_z(snd, sorted(split["train"] & set(snd.z)))
    if a.exclude_subjects and train_body and train_body[0] >= datasets.SUBJECT_Z:
        names = snd.subjects
        train_body = [z for z in train_body if names[int(z // datasets.SUBJECT_Z) - 1] not in a.exclude_subjects]
    sup_z, labels = ([], [])
    if a.mode != "zero":
        sup_z, labels = prdata.support_units(ds, nd_key, train_body, a.k, a.unit_len, a.draw, a.max_per_unit)
        print(f"[adapt] support units: {labels} ({len(sup_z)} slices per dose)")
    pairs = []
    for d in sup_doses:
        k = f"{d}_{a.recon}"
        zz = sorted(set(sup_z) & set(ds.pair_z(nd_key, k)))
        if zz:
            pairs.append(dict(r=a.recon, d=d, nd=nd_key, ld=k, train=zz, val=[]))

    class Doms:            # minimal interface used by make_batch
        pass
    doms = Doms()
    doms.ds = ds
    doms.slices = lambda p, z: (ds.series(p["nd"]).get(z), ds.series(p["ld"]).get(z))

    # ------------------------------------------------------------------ embedding
    g = torch.Generator(device=dev).manual_seed(a.seed)
    if a.mode == "zero":
        zD = ck["z0"].to(dev)
    else:
        phis, lds = [], []
        with torch.no_grad():
            for p in pairs:
                for z in p["train"][:: max(1, len(p["train"]) // 40)]:
                    x, y = doms.slices(p, z)
                    x = torch.tensor(x, device=dev)[None, None]
                    y = torch.tensor(y, device=dev)[None, None]
                    m = stats.class_masks(x)
                    n = sim.noise(sim.sinogram(x, dx), torch.tensor([p["d"] / 100], device=dev), dx, generator=g)
                    phis.append(stats.descriptor(y, x, dx, m) - stats.descriptor(x + n, x, dx, m))
                    lds.append(math.log(p["d"] / 100))
            zD = model.emb(torch.stack(phis), torch.tensor(lds, device=dev))
        print(f"[adapt] embedding from {len(phis)} support descriptors")
        if "z_domains" in ck:
            dist = {r: float((zD.detach().cpu() - v.detach().cpu()).norm()) for r, v in ck["z_domains"].items()}
            print("[adapt] distance to training-domain embeddings: " + "  ".join(f"{r}:{v:.2f}" for r, v in dist.items()))

    # ------------------------------------------------------------------ refinement (optional)
    if a.z_init == "zero":
        zD = torch.zeros_like(zD)
        print("[adapt] z_init=zero: domain embedding set to 0 (no domain information)")
    elif a.z_init == "mean":
        zD = ck["z0"].to(dev).clone()
        print("[adapt] z_init=mean: mean of the training-domain embeddings")
    if a.mode in ("zref", "zfilm", "heads", "full", "scratch"):
        zpar = torch.nn.Parameter(zD.detach().clone())
        for prm in model.parameters():
            prm.requires_grad_(False)
        params = ([] if a.freeze_z else [zpar]) + model.adapted_parameters(a.mode)
        for prm in params:
            prm.requires_grad_(True)
        zpar.requires_grad_(not a.freeze_z)
        n_adapt = sum(p.numel() for p in params)
        print(f"[adapt] refining {n_adapt} parameters for {a.steps} steps")
        opt = torch.optim.Adam(params, lr=a.lr if a.mode in ("zref", "zfilm", "heads") else min(a.lr, 2e-4))
        lossf = PRLoss(dx, targs["lam_cov"], targs["lam_dist"], targs["lam_st"], targs["lam_mean"],
                       generator=torch.Generator().manual_seed(a.seed))
        rng = np.random.default_rng(a.seed)
        from pr_train import forward_pair

        class A:
            no_anat = targs.get("no_anat", False)
            no_st = targs.get("no_st", False)
        # ---- hold-out part of the SUPPORT set for early stopping (never the test set)
        def subj(z):
            return int(z // datasets.SUBJECT_Z) if z >= datasets.SUBJECT_Z else None
        fit_pairs, val_pairs = [], []
        for pp in pairs:
            subs = sorted({subj(z) for z in pp["train"]} - {None})
            if len(subs) >= 3:                       # Norsvin: hold out one support pig
                hold = {subs[a.seed % len(subs)]}
                tr = [z for z in pp["train"] if subj(z) not in hold]
                va = [z for z in pp["train"] if subj(z) in hold]
            else:                                    # piglet / few units: every 5th support slice
                tr = [z for i, z in enumerate(pp["train"]) if i % 5]
                va = [z for i, z in enumerate(pp["train"]) if i % 5 == 0]
            if not tr or not va:
                tr, va = pp["train"], pp["train"]
            fit_pairs.append(dict(pp, train=tr))
            val_pairs.append(dict(pp, train=va[:: max(1, len(va) // a.val_slices)][: a.val_slices]))

        def val_loss():
            tot, n = 0.0, 0
            with torch.no_grad():
                for vp in val_pairs:
                    vg = torch.Generator(device=dev).manual_seed(12345)
                    vr = np.random.default_rng(12345)
                    for i in range(0, len(vp["train"]), a.batch):
                        b = make_batch(doms, vp, vp["train"][i:i + a.batch], sim, dx, dev, a.crop, vr, vg,
                                       varmap=model.needs_varmap, basis=model.needs_basis)
                        yh1, yh2, nh1, nh2 = forward_pair(model, b, zpar, A)
                        l, _ = lossf(yh1, yh2, nh1, nh2, b["y"], b["x"])
                        tot, n = tot + float(l), n + 1
            return tot / max(n, 1)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps, 1), eta_min=opt.param_groups[0]["lr"] * 0.05)
        best = (val_loss(), 0, [q.detach().clone() for q in params])
        print(f"[refine 0/{a.steps}] val={best[0]:.4f} (support hold-out)", flush=True)
        for it in range(a.steps):
            p = fit_pairs[rng.integers(len(fit_pairs))]
            zs = list(rng.choice(p["train"], min(a.batch, len(p["train"])), replace=False))
            b = make_batch(doms, p, zs, sim, dx, dev, a.crop, rng, g, varmap=model.needs_varmap, basis=model.needs_basis)
            yh1, yh2, nh1, nh2 = forward_pair(model, b, zpar, A)
            loss, terms = lossf(yh1, yh2, nh1, nh2, b["y"], b["x"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if (it + 1) % a.val_every == 0 or it + 1 == a.steps:
                v = val_loss()
                if v < best[0]:
                    best = (v, it + 1, [q.detach().clone() for q in params])
                msg = " ".join(f"{k}={vv:.4f}" for k, vv in terms.items())
                msg += f"  val={v:.4f} (best {best[0]:.4f} @ {best[1]})  "
                msg += model.describe(zpar[None], torch.tensor([math.log(p["d"] / 100)], device=dev))
                print(f"[refine {it + 1}/{a.steps}] {msg}", flush=True)
        with torch.no_grad():
            for q, bq in zip(params, best[2]):
                q.copy_(bq)
        print(f"[adapt] kept the parameters of step {best[1]} (support hold-out loss {best[0]:.4f})")
        print("[adapt] final response: " + "  ".join(
            f"{d}%: " + model.describe(zpar[None], torch.tensor([math.log(d / 100)], device=dev)) for d in sup_doses))
        zD = zpar.detach()
        model.eval()
    else:
        n_adapt = 0 if a.mode == "zero" else zD.numel()

    # ------------------------------------------------------------------ synthesis of the test set
    model.eval()
    os.makedirs(a.runs, exist_ok=True)
    for d in a.doses:
        lk = ld_keys[d]
        zl = ds.pair_z(nd_key, lk)
        pos = {z: i for i, z in enumerate(zl)}
        test = prdata.body_z(snd, sorted(split["test"] & set(zl)))[:: a.test_stride]
        train_idx = [pos[z] for z in sorted(split["train"] & set(zl))]
        test_idx = [pos[z] for z in test]
        gs = torch.Generator(device=dev).manual_seed(a.seed * 1000 + d)
        preds = []
        with torch.no_grad():
            for i, z in enumerate(test):
                x = torch.tensor(snd.get(z), device=dev)[None, None]
                P = sim.sinogram(x, dx)
                dd = torch.tensor([d / 100], device=dev)
                if model.needs_basis:
                    n, nb = sim.noise_basis(P, dd, dx, generator=gs)
                else:
                    n, nb = sim.noise(P, dd, dx, generator=gs), None
                v = sim.variance_map(P, dd, dx) if model.needs_varmap else None
                yh, nh, s = model(x, n, torch.log(dd), zD[None], v,
                                  x_cnn=torch.full_like(x, 40.0) if targs.get("no_anat") else None, n_basis=nb)
                if targs.get("no_st"):
                    yh = x + nh
                preds.append(np.clip(yh[0, 0].float().cpu().numpy(), -1024, 3071).astype(np.float16))
                if (i + 1) % 100 == 0:
                    print(f"[sample {d}%] {i + 1}/{len(test)}", flush=True)
        write_pred_run(a.runs, name, nd_key, lk, a.fold, a.n_folds, 0, a.data, train_idx, test_idx,
                       np.stack(preds), (-1024.0, 2048.0),
                       extra=dict(method="PhysReCon-LDCT", mode=a.mode, k=a.k, draw=a.draw, support=labels, test_stride=a.test_stride,
                                  adapted_parameters=int(n_adapt), response=model.describe(zD[None], torch.log(torch.tensor([d / 100], device=dev))), ckpt=a.ckpt, physics=phys,
                                  z=[float(v) for v in zD.flatten().tolist()]))
    if a.save_ckpt:
        torch.save(dict(model=model.state_dict(), z=zD.cpu(), args=vars(a)), os.path.join(a.runs, f"{name}_adapted.pt"))


if __name__ == "__main__":
    main()
