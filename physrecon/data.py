"""Domains, splits and support sets.

A domain D is a reconstruction configuration of a dataset (piglet FBP / ASIR20 / ... ; Norsvin IR).
For each domain: NDCT key  100_<recon>  (falls back to 100_FBP with a warning) and LDCT keys <d>_<recon>.
Slices are addressed by their z value (index.json), so all domains of the piglet share one split.

Split = exactly the benchmark split (datasets.slab_split on the reference pair), so PhysReCon is scored on
the same test slices as the previously benchmarked models. A validation block is carved out of the
training slices for the single-subject piglet (with a gap on both sides).
"""
import json
import os

import numpy as np

import datasets


class Series:
    """Memory-mapped int16 volume addressed by z."""

    def __init__(self, data, key, index):
        self.key = key
        self.vol = np.load(os.path.join(data, key + ".npy"), mmap_mode="r")
        self.z = [float(v) for v in index[key]["z"]]
        self.pos = {v: i for i, v in enumerate(self.z)}
        self.spacing = index[key]["spacing"]
        self.subjects = index[key].get("subjects")

    def get(self, z):
        return np.asarray(self.vol[self.pos[float(z)]], np.float32)


class Dataset:
    def __init__(self, data):
        self.data = data
        self.index = json.load(open(os.path.join(data, "index.json")))
        self._series = {}

    def series(self, key):
        if key not in self._series:
            if key not in self.index:
                raise KeyError(f"{key} not in {self.data}/index.json; available: {sorted(self.index)}")
            self._series[key] = Series(self.data, key, self.index)
        return self._series[key]

    def nd_key(self, recon, ld_key=None, src_dose=100, force=None):
        """NDCT series of a domain: 100_<recon> if it exists AND lies on the same pixel grid as the domain's
        LDCT (ld_key); otherwise the full-dose FBP (e.g. piglet: the full-dose ASIR/MBIR series were
        reconstructed with a 320 mm field of view, all low-dose series and full-dose FBP with 212 mm)."""
        if force:
            return f"{src_dose}_{force}"
        k = f"{src_dose}_{recon}"
        fb = f"{src_dose}_FBP"
        if k not in self.index:
            print(f"[data] {k} not available: using {fb} as the NDCT of domain {recon}")
            return fb
        if ld_key and ld_key in self.index:
            sp_nd = [round(float(v), 4) for v in self.index[k]["spacing"]]
            sp_ld = [round(float(v), 4) for v in self.index[ld_key]["spacing"]]
            if sp_nd != sp_ld:
                print(f"[data] {k} pixel spacing {sp_nd} != {ld_key} {sp_ld} (different field of view): "
                      f"using {fb} as the NDCT of domain {recon}")
                return fb
        return k

    def pair_z(self, nd_key, ld_key):
        return sorted(set(self.series(nd_key).z) & set(self.series(ld_key).z))

    def available(self):
        return sorted(self.index)


def benchmark_split(ds, ref_nd, ref_ld, n_folds=3, fold=0, gap=5, val_frac=0.1):
    """train / val / test z-sets identical to datasets.slab_split on the reference pair (the benchmark)."""
    z = np.array(ds.pair_z(ref_nd, ref_ld))
    tr, te = datasets.slab_split(len(z), n_folds, fold, gap, z=z)
    tr_z, te_z = list(z[tr]), list(z[te])
    val_z = []
    single = not len(z) or np.min(z) < datasets.SUBJECT_Z
    if single and val_frac > 0 and len(tr):
        tr_sorted = np.sort(tr)
        runs = np.split(tr_sorted, np.where(np.diff(tr_sorted) > 1)[0] + 1)
        run = max(runs, key=len)
        nv = max(4, int(round(val_frac * len(z))))
        mid = len(run) // 2
        v = run[max(0, mid - nv // 2): mid - nv // 2 + nv]
        val_z = list(z[v])
        lo, hi = v.min() - gap, v.max() + gap
        tr_z = [zz for i, zz in zip(tr, z[tr]) if not (lo <= i <= hi)]
    return dict(train=set(tr_z), val=set(val_z), test=set(te_z))


def body_z(series, zs, min_frac=0.05):
    out = []
    for z in zs:
        if (series.get(z)[::8, ::8] > -500).mean() >= min_frac:
            out.append(z)
    return out


def support_units(ds, nd_key, zs, k, unit_len=10, draw=0, max_per_unit=30):
    """Split training z values into units and pick k of them.
    Multi-subject data (Norsvin): a unit is a subject (pig). Single subject (piglet): a unit is a block of
    `unit_len` consecutive slices. Returns (chosen z list, unit names)."""
    zs = sorted(zs)
    if not zs or k <= 0:
        return [], []
    rng = np.random.default_rng(draw)
    if zs[0] >= datasets.SUBJECT_Z:
        subj = datasets.subject_of(zs)
        names = ds.series(nd_key).subjects
        units = {}
        for z, s in zip(zs, subj):
            units.setdefault(int(s), []).append(z)
        keys = sorted(units)
        pick = rng.choice(len(keys), size=min(k, len(keys)), replace=False)
        chosen, labels = [], []
        for p in sorted(pick):
            u = units[keys[p]]
            step = max(1, len(u) // max_per_unit)
            chosen += u[::step][:max_per_unit]
            labels.append(names[keys[p] - 1] if names else str(keys[p]))
        return chosen, labels
    blocks = [zs[i:i + unit_len] for i in range(0, len(zs), unit_len) if len(zs[i:i + unit_len]) == unit_len]
    pick = rng.choice(len(blocks), size=min(k, len(blocks)), replace=False)
    chosen = [z for p in sorted(pick) for z in blocks[p]]
    return chosen, [f"slab@{blocks[p][0]:g}" for p in sorted(pick)]


def body_crop(nd, size, rng):
    """Random size x size crop location (y0, x0) centred on the body of a full NDCT slice (numpy)."""
    H, W = nd.shape
    if size >= H:
        return 0, 0
    ys, xs = np.nonzero(nd[::4, ::4] > -500)
    if len(ys) == 0:
        return (H - size) // 2, (W - size) // 2
    i = rng.integers(len(ys))
    cy, cx = ys[i] * 4, xs[i] * 4
    y0 = int(np.clip(cy - size // 2, 0, H - size))
    x0 = int(np.clip(cx - size // 2, 0, W - size))
    return y0, x0
