"""Stage 1 - shared pretraining of PhysReCon-LDCT on piglet reconstruction domains.

Default: training domains FBP, ASIR20, ASIR60 (ASIR40 and MBIR held out for few-shot adaptation),
doses 5/10/25 %, benchmark slab split (test slab never used), validation block carved from training slabs.

Per step: one (domain, dose); a support set of 2-8 cached descriptors of that domain (any dose) gives z_D;
a batch of training slices -> physics noise (2 independent realisations, full slices, cached sinograms)
-> random body crops -> model -> loss against the REAL LDCT of that domain and dose (statistics only,
plus a paired term in strategies 'real'/'joint').

    python -u pr_train.py --data data/piglet --physics pr_calib/physics.json --recons FBP ASIR20 ASIR60 \
        --doses 5 10 25 --iters 6000 --out pr_runs --tag main
"""
import argparse
import json
import math
import os
import time

import numpy as np
import datasets
import torch

from physrecon import data as prdata
from physrecon import stats
from physrecon.losses import PRLoss
from physrecon.model import PhysReCon
from physrecon.physics import PhysicsSim


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--physics", required=True, help="physics.json from pr_calibrate.py")
    ap.add_argument("--recons", nargs="+", default=["FBP", "ASIR20", "ASIR60"])
    ap.add_argument("--doses", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--ref_recon", default="FBP", help="reference pair of the benchmark split")
    ap.add_argument("--nd_recon", default=None,
                    help="force the NDCT reconstruction for every domain (e.g. FBP); default: 100_<recon> when it is on "
                         "the same pixel grid as the LDCT, else 100_FBP")
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--exclude_subjects", nargs="*", default=[], help="multi-subject data: subjects never used")
    ap.add_argument("--strategy", default="sim", choices=["sim", "real", "joint"],
                    help="sim: simulated physics noise + statistical losses (primary); real: real FBP residual "
                         "input + paired target (diagnostic, needs pr_dicom_check); joint: alternate")
    ap.add_argument("--use_varmap", action="store_true", help="projection-feature ablation P1")
    ap.add_argument("--proj_resp", action="store_true", help="projection-domain statistical-weighting response (pw)")
    ap.add_argument("--sig_blur", action="store_true", help="linear resolution-loss branch after the signal transfer")
    ap.add_argument("--ng_J", type=int, default=5, help="wavelet scales of the noise-transfer gains")
    ap.add_argument("--st_varmap", action="store_true",
                    help="noise-adaptive signal transfer: the low-contrast detail gain depends on the local noise "
                         "variance map from the noisy sinogram (IR smooths more where projection data are noisier)")
    ap.add_argument("--no_st", action="store_true", help="ablation: identity signal transfer (A4)")
    ap.add_argument("--no_anat", action="store_true", help="ablation: no NDCT subbands in F (A3)")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam_cov", type=float, default=1.0)
    ap.add_argument("--lam_dist", type=float, default=1.0)
    ap.add_argument("--lam_st", type=float, default=0.5)
    ap.add_argument("--lam_mean", type=float, default=1.0)
    ap.add_argument("--lam_pair", type=float, default=1.0)
    ap.add_argument("--support_min", type=int, default=2)
    ap.add_argument("--support_max", type=int, default=8)
    ap.add_argument("--desc_stride", type=int, default=3)
    ap.add_argument("--desc_max", type=int, default=40)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_batches", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pr_runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


class Domains:
    """All (domain, dose) pairs of a dataset with their train/val z lists."""

    def __init__(self, ds, recons, doses, split, nd_recon=None):
        self.ds = ds
        self.nd = {r: ds.nd_key(r, f"{doses[0]}_{r}", force=nd_recon) for r in recons}
        self.pairs = []
        for r in recons:
            for d in doses:
                k = f"{d}_{r}"
                if k not in ds.index:
                    print(f"[data] {k} missing - skipped")
                    continue
                zs = set(ds.pair_z(self.nd[r], k))
                tr = prdata.body_z(ds.series(self.nd[r]), sorted(split["train"] & zs))
                va = prdata.body_z(ds.series(self.nd[r]), sorted(split["val"] & zs))
                self.pairs.append(dict(r=r, d=d, nd=self.nd[r], ld=k, train=tr, val=va))
                print(f"[data] {self.nd[r]} -> {k}: train {len(tr)}, val {len(va)} body slices")

    def slices(self, pair, z):
        return self.ds.series(pair["nd"]).get(z), self.ds.series(pair["ld"]).get(z)


def to_dev(a, dev):
    return torch.tensor(np.asarray(a, np.float32), device=dev)[None, None]


@torch.no_grad()
def descriptor_cache(doms, sim, dx, stride, max_n, dev, seed=0):
    """dPhi = Phi(real LDCT) - Phi(NDCT + n_phys) for training slices of every (domain, dose)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    cache = {}
    for p in doms.pairs:
        zs = p["train"][::stride][:max_n]
        out = []
        for z in zs:
            x, y = doms.slices(p, z)
            x, y = to_dev(x, dev), to_dev(y, dev)
            m = stats.class_masks(x)
            n = sim.noise(sim.sinogram(x, dx, keys=[(p["nd"], z)]), torch.tensor([p["d"] / 100], device=dev), dx,
                          generator=g)
            out.append(stats.descriptor(y, x, dx, m) - stats.descriptor(x + n, x, dx, m))
        cache.setdefault(p["r"], []).append((torch.stack(out), torch.full((len(out),), math.log(p["d"] / 100),
                                                                          device=dev)))
        print(f"[desc] {p['ld']}: {len(out)} descriptors")
    return {r: (torch.cat([a for a, _ in v]), torch.cat([b for _, b in v])) for r, v in cache.items()}


@torch.no_grad()
def set_normalisation(model, doms, sim, dx, dev, n=12):
    """Fixed per-scale scales of the physics-noise subbands (at 25 % dose) and NDCT subbands (robust MAD),
    from training slices of the first domain. Stored as buffers in the checkpoint."""
    from physrecon import wavelet
    p = doms.pairs[0]
    zs = p["train"][:: max(1, len(p["train"]) // n)][:n]
    sn, sx, sa = [], [], []
    g = torch.Generator(device=dev).manual_seed(1)
    for z in zs:
        x, _ = doms.slices(p, z)
        x = to_dev(x, dev)
        nz = sim.noise(sim.sinogram(x, dx, keys=[(p["nd"], z)]), torch.tensor([0.25], device=dev), dx, generator=g)
        body = stats.body_mask(x)[0, 0]
        dn, an = wavelet.forward(nz)
        dxw, _ = wavelet.forward(x)
        sn.append(torch.stack([d[0, 0][:, body].std() for d in dn]))
        sx.append(torch.stack([1.4826 * d[0, 0][:, body].abs().median() for d in dxw]))
        sa.append(an[0, 0][body].std())
    model.cnn.s_n.copy_(torch.stack(sn).median(0).values.clamp(min=1e-3))
    model.cnn.s_x.copy_(torch.stack(sx).median(0).values.clamp(min=1e-3))
    model.cnn.s_na.copy_(torch.stack(sa).median().clamp(min=1e-3))
    print(f"[norm] s_n {[round(v, 2) for v in model.cnn.s_n.tolist()]}  s_na {model.cnn.s_na.item():.2f}  "
          f"s_x {[round(v, 2) for v in model.cnn.s_x.tolist()]}")


def make_batch(doms, pair, zs, sim, dx, dev, crop, rng, g, n_real=2, real_noise=False, varmap=False, basis=False):
    """Full-slice physics on the GPU, then identical random body crops of x, y, noise (and variance map)."""
    xs, ys, keys = [], [], []
    for z in zs:
        x, y = doms.slices(pair, z)
        xs.append(x)
        ys.append(y)
        keys.append((pair["nd"], z))
    X = torch.tensor(np.stack(xs)[:, None], device=dev)
    Y = torch.tensor(np.stack(ys)[:, None], device=dev)
    d = torch.full((len(zs),), pair["d"] / 100, device=dev)
    P = sim.sinogram(X, dx, keys=keys)
    if real_noise:
        fbp_nd = doms.ds.nd_key("FBP")
        fbp_ld = f"{pair['d']}_FBP"
        Nr = torch.tensor(np.stack([doms.ds.series(fbp_ld).get(z) - doms.ds.series(fbp_nd).get(z) for z in zs])[:, None],
                          device=dev)
        N = torch.cat([Nr, sim.noise(P, d, dx, generator=g)])
    elif basis:
        N, NB = sim.noise_basis(P, d, dx, generator=g, n_real=n_real)
    else:
        N = sim.noise(P, d, dx, generator=g, n_real=n_real)
    basis = basis and not real_noise
    V = sim.variance_map(P, d, dx).repeat(2, 1, 1, 1) if varmap else None
    out = dict(x=[], y=[], n1=[], n2=[], v=[], nb1=[], nb2=[])
    B = len(zs)
    for b in range(B):
        y0, x0 = prdata.body_crop(xs[b], crop, rng)
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        out["x"].append(X[b, :, sl[0], sl[1]])
        out["y"].append(Y[b, :, sl[0], sl[1]])
        out["n1"].append(N[b, :, sl[0], sl[1]])
        out["n2"].append(N[B + b, :, sl[0], sl[1]])
        if varmap:
            out["v"].append(V[b, :, sl[0], sl[1]])
        if basis:
            out["nb1"].append(NB[b, :, sl[0], sl[1]])
            out["nb2"].append(NB[B + b, :, sl[0], sl[1]])
    res = {k: torch.stack(v) for k, v in out.items() if v}
    res["logd"] = torch.log(d)
    return res


def forward_pair(model, b, z, args):
    """Run the model on both noise realisations with the same x, d, z."""
    x2 = torch.cat([b["x"], b["x"]])
    n2 = torch.cat([b["n1"], b["n2"]])
    ld2 = torch.cat([b["logd"], b["logd"]])
    z2 = z[None].expand(len(x2), -1)
    v2 = torch.cat([b["v"], b["v"]]) if "v" in b else None
    nb = torch.cat([b["nb1"], b["nb2"]]) if "nb1" in b else None
    yh, nh, s = model(x2, n2, ld2, z2, v2, x_cnn=torch.full_like(x2, 40.0) if args.no_anat else None, n_basis=nb)
    if args.no_st:
        yh = x2 + nh
    else:
        yh = s + nh
    B = len(b["x"])
    return yh[:B], yh[B:], nh[:B], nh[B:]


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phys = json.load(open(a.physics))
    ds = prdata.Dataset(a.data)
    ref_nd = ds.nd_key(a.ref_recon, f"{a.doses[0]}_{a.ref_recon}")
    split = prdata.benchmark_split(ds, ref_nd, f"{a.doses[0]}_{a.ref_recon}", a.n_folds, a.fold, a.gap, a.val_frac)
    doms = Domains(ds, a.recons, a.doses, split, a.nd_recon)
    if a.exclude_subjects:
        for p in doms.pairs:
            names = ds.series(p["nd"]).subjects
            keep = lambda zz: [z for z in zz if z < datasets.SUBJECT_Z or names[int(z // datasets.SUBJECT_Z) - 1] not in a.exclude_subjects]
            n0 = len(p["train"])
            p["train"], p["val"] = keep(p["train"]), keep(p["val"])
            print(f"[data] {p['ld']}: excluded {a.exclude_subjects}: train {n0} -> {len(p['train'])}")
    if not doms.pairs:
        raise SystemExit("no (domain, dose) pair available")
    dx = float(ds.series(doms.pairs[0]["nd"]).spacing[0])
    H = ds.series(doms.pairs[0]["nd"]).get(doms.pairs[0]["train"][0]).shape[0]
    stats.set_roi_size(H)
    sim = PhysicsSim(H, phys["I0"], phys["se2"], phys["kappa"], phys.get("n_views", 720), dev)
    print(f"[physics] I0={sim.I0:.4g} se2={sim.se2:g} kappa={sim.kappa}  pixel {dx:.3f} mm")

    name = "physrecon" + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)

    model = PhysReCon(a.channels, a.units, use_varmap=a.use_varmap, st_varmap=a.st_varmap, ng_J=a.ng_J,
                      proj_resp=a.proj_resp, sig_blur=a.sig_blur).to(dev)
    set_normalisation(model, doms, sim, dx, dev)
    cache = descriptor_cache(doms, sim, dx, a.desc_stride, a.desc_max, dev)
    allphi = torch.cat([v[0] for v in cache.values()])
    model.emb.mu.copy_(allphi.mean(0))
    model.emb.sd.copy_(allphi.std(0).clamp(min=1e-3))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / max(a.iters, 1))))
    lossf = PRLoss(dx, a.lam_cov, a.lam_dist, a.lam_st, a.lam_mean, a.lam_pair, generator=torch.Generator().manual_seed(a.seed))
    start, best = 0, float("inf")
    if a.resume and os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location=dev)
        model.load_compatible(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["it"], ck["best"]
        print(f"[resume] from iteration {start}")

    def support_z(r, k, gen):
        phi, ld = cache[r]
        idx = torch.randperm(len(phi), generator=gen)[:k]
        return model.emb(phi[idx], ld[idx])

    # fixed validation batches
    vpairs = [p for p in doms.pairs if p["val"]]
    vrng = np.random.default_rng(1234)
    vbatches = []
    for i in range(a.val_batches if vpairs else 0):
        p = vpairs[i % len(vpairs)]
        vbatches.append((p, list(vrng.choice(p["val"], min(a.batch, len(p["val"])), replace=False)), int(vrng.integers(1 << 30))))

    def validate():
        model.eval()
        tot = []
        with torch.no_grad():
            for p, zs, sd in vbatches:
                g = torch.Generator(device=dev).manual_seed(sd)
                b = make_batch(doms, p, zs, sim, dx, dev, a.crop, np.random.default_rng(sd), g, varmap=model.needs_varmap, basis=model.needs_basis)
                z = support_z(p["r"], a.support_max, torch.Generator().manual_seed(sd))
                yh1, yh2, nh1, nh2 = forward_pair(model, b, z, a)
                l, _ = lossf(yh1, yh2, nh1, nh2, b["y"], b["x"])
                tot.append(float(l))
        model.train()
        return float(np.mean(tot)) if tot else float("nan")

    gpu_g = torch.Generator(device=dev).manual_seed(a.seed)
    cpu_g = torch.Generator().manual_seed(a.seed)
    t0, log = time.time(), {}
    for it in range(start, a.iters):
        p = doms.pairs[rng.integers(len(doms.pairs))]
        zs = list(rng.choice(p["train"], min(a.batch, len(p["train"])), replace=False))
        real = a.strategy == "real" or (a.strategy == "joint" and it % 2 == 1)
        b = make_batch(doms, p, zs, sim, dx, dev, a.crop, rng, gpu_g, real_noise=real, varmap=model.needs_varmap, basis=model.needs_basis)
        k = int(rng.integers(a.support_min, a.support_max + 1))
        z = support_z(p["r"], k, cpu_g)
        yh1, yh2, nh1, nh2 = forward_pair(model, b, z, a)
        loss, terms = lossf(yh1, yh2, nh1, nh2, b["y"], b["x"], paired=real)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for kk, v in terms.items():
            log[kk] = log.get(kk, 0) + v
        log["n"] = log.get("n", 0) + 1
        if (it + 1) % a.log_every == 0:
            n = log.pop("n")
            el = time.time() - t0
            eta = el / (it + 1 - start) * (a.iters - it - 1) / 3600
            print(f"[{it + 1}/{a.iters}] " + " ".join(f"{kk}={v / n:.4f}" for kk, v in log.items()) +
                  f"  {el / (it + 1 - start):.2f}s/it  eta {eta:.1f} h", flush=True)
            log = {}
        if (it + 1) % a.val_every == 0 or it + 1 == a.iters:
            v = validate()
            if v < best:
                best = v
                torch.save(dict(model=model.state_dict()), os.path.join(out, "best_state.pt"))
            print(f"[val] it {it + 1}: {v:.4f} (best {best:.4f})", flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it + 1,
                            best=best), os.path.join(out, "last.pt"))

    # final checkpoint: best weights (or last if no validation), domain embeddings, zero-shot embedding
    if os.path.exists(os.path.join(out, "best_state.pt")):
        model.load_state_dict(torch.load(os.path.join(out, "best_state.pt"), map_location=dev)["model"])
    with torch.no_grad():
        zdom = {r: model.emb(*cache[r]).cpu() for r in cache}
    z0 = torch.stack(list(zdom.values())).mean(0)
    torch.save(dict(model=model.state_dict(), cfg=model.cfg, physics=phys, args=vars(a), z_domains=zdom, z0=z0,
                    split={k: sorted(float(v) for v in s) for k, s in split.items()}, dx=dx),
               os.path.join(out, "final.pt"))
    print(f"[done] {out}/final.pt  domain embeddings: {sorted(zdom)}")


if __name__ == "__main__":
    main()
