"""Cross-domain generalisation without adaptation: apply a GAN/VAE trained on the piglet source data to the
unseen Norsvin scanner, on the same test slices as the joint evaluation (every --test_stride-th test slice).

    python cross_apply.py --run runs_old_piglet/cut_100_FBP_to_10_FBP_f0_s0 --data data/norsvin --recon IR \
        --dose 10 --test_stride 8 --runs runs_cross_norsvin --name cut_piglet
"""
import argparse, json, os
import numpy as np, torch
import datasets
from diffusion_common import load_multi_dose, write_pred_run
from evaluate import load_run

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="piglet-trained run folder containing model.pt")
ap.add_argument("--data", default="data/norsvin")
ap.add_argument("--recon", default="IR")
ap.add_argument("--dose", type=int, required=True)
ap.add_argument("--src_dose", type=int, default=100)
ap.add_argument("--n_folds", type=int, default=3)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--gap", type=int, default=5)
ap.add_argument("--test_stride", type=int, default=8)
ap.add_argument("--runs", default="runs_cross_norsvin")
ap.add_argument("--name", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(os.path.join(a.run, "config.json")))
lo, hi = cfg.get("hu_range", [-1024.0, 2048.0])
datasets.set_hu_range(lo, hi)
m = load_run(a.run, dev)
if isinstance(m, np.ndarray):
    raise SystemExit(f"{a.run} has predictions only (no model.pt)")
D = load_multi_dose(a.data, f"{a.src_dose}_{a.recon}", a.recon, [a.dose], a.n_folds, a.fold, a.gap)[a.dose]
te = D["test"][:: a.test_stride]
preds = []
with torch.no_grad():
    for i in te:
        x = torch.from_numpy(datasets.hu_to_net(D["src"][i].astype(np.float32)).astype(np.float32))[None, None].to(dev)
        y = datasets.net_to_hu(m.translate(x)[0, 0].float().cpu().numpy())
        preds.append(np.clip(y, lo, hi))
print(f"[cross] {a.run} -> {a.data} {a.dose}% : {len(te)} slices (no adaptation)")
write_pred_run(a.runs, a.name, f"{a.src_dose}_{a.recon}", D["tgt"], a.fold, a.n_folds, 0, a.data, D["train"], te,
               np.stack(preds), (lo, hi), extra=dict(trained_on=cfg.get("data"), source_run=a.run, adaptation="none"))
