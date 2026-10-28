"""Statistics shared by the losses, the degradation descriptor and the evaluation.

All functions work on torch tensors (B,1,H,W) in HU and are differentiable w.r.t. the image whenever the
image is the synthetic output; masks, ROI positions and edge points are always derived from the NDCT
(no gradient), so real and synthetic images are measured at the SAME anatomical locations.
"""
import math

import torch
import torch.nn.functional as F

from . import wavelet

# tissue classes on the smoothed NDCT (HU ranges); order is fixed (descriptor layout)
CLASSES = [("fat", -150, -50), ("fluid", -50, 20), ("soft", 20, 80), ("dense", 80, 200),
           ("bone", 300, 4000), ("lung", -900, -500)]
SOFT = 2
ROI = 64          # ROI size (px); set_roi_size() adapts it for small images (tests)


def set_roi_size(H):
    global ROI
    ROI = int(min(64, max(16, H // 4)))
    return ROI


# ----------------------------------------------------------------------------- filters
def gauss(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)


def texture(x, sigma=2.0):
    return x - gauss(x, sigma)


def erode(mask, r):
    if r <= 0:
        return mask
    return -F.max_pool2d(-mask.float(), 2 * r + 1, stride=1, padding=r) > 0.5


# ----------------------------------------------------------------------------- masks
@torch.no_grad()
def class_masks(nd, flat_thr=20.0, erode_px=3):
    """nd (B,1,H,W) HU -> (B, n_classes, H, W) bool masks of FLAT regions of each tissue class.
    Flat = local std (9x9) of the slightly smoothed NDCT below flat_thr HU (no strong edges)."""
    g = gauss(nd, 1.5)
    m = box(g, 4)
    sd = torch.sqrt(torch.clamp(box(g * g, 4) - m * m, min=0))
    flat = sd < flat_thr
    s2 = gauss(nd, 2.0)
    out = []
    for _, lo, hi in CLASSES:
        c = (s2 >= lo) & (s2 < hi)
        if lo >= 300:                      # bone: no flatness requirement (cortical bone is never flat)
            out.append(erode(c, 1))
        else:
            out.append(erode(c & flat, erode_px))
    return torch.cat(out, 1)


@torch.no_grad()
def body_mask(nd):
    return gauss(nd, 2.0) > -500


@torch.no_grad()
def roi_positions(mask, size=None, stride=None, frac=0.9, max_per_image=64, generator=None):
    """mask (B,1,H,W) bool -> LongTensor (K,3) of (b, y0, x0) ROI corners fully (>= frac) inside mask."""
    size = size or ROI
    stride = stride or size // 2
    f = F.avg_pool2d(mask.float(), size, stride=stride)
    b, yy, xx = torch.nonzero(f[:, 0] >= frac, as_tuple=True)
    pos = torch.stack([b, yy * stride, xx * stride], 1)
    if len(pos) == 0:
        return pos
    keep = []
    for bi in pos[:, 0].unique():
        sel = torch.nonzero(pos[:, 0] == bi).flatten()
        if len(sel) > max_per_image:
            perm = torch.randperm(len(sel), generator=generator, device="cpu")[:max_per_image].to(sel.device)
            sel = sel[perm]
        keep.append(sel)
    return pos[torch.cat(keep)]


def extract(img, pos, size=None):
    """img (B,1,H,W), pos (K,3) -> (K,size,size); differentiable w.r.t. img."""
    size = size or ROI
    if len(pos) == 0:
        return img.new_zeros(0, size, size)
    ar = torch.arange(size, device=img.device)
    ys = (pos[:, 1:2] + ar[None]).long()                         # (K,size)
    xs = (pos[:, 2:3] + ar[None]).long()
    return img[pos[:, 0, None, None].long(), 0, ys[:, :, None], xs[:, None, :]]


_PLANE = {}


def detrend(rois):
    """Subtract a least-squares plane from every ROI (K,N,N)."""
    K, N, _ = rois.shape
    key = (N, rois.device, rois.dtype)
    if key not in _PLANE:
        a = torch.arange(N, device=rois.device, dtype=rois.dtype) - (N - 1) / 2
        yy, xx = torch.meshgrid(a, a, indexing="ij")
        A = torch.stack([torch.ones_like(xx), xx, yy], -1).view(-1, 3)
        _PLANE[key] = (A, torch.linalg.pinv(A))
    A, Ap = _PLANE[key]
    v = rois.reshape(K, -1)
    return (v - (v @ Ap.T) @ A.T).view(K, N, N)


# ----------------------------------------------------------------------------- NPS
def nps2d(rois, dx=1.0):
    """ICRU-87 style NPS of detrended ROIs (no window): mean over ROIs of dx^2/N^2 |DFT|^2  (N,N), HU^2 mm^2.
    Zero frequency at index [0,0] (not shifted)."""
    if len(rois) == 0:
        return None
    N = rois.shape[-1]
    r = detrend(rois)
    return (torch.fft.fft2(r).abs() ** 2).mean(0) * dx * dx / (N * N)


_POLAR = {}


def polar_index(N, nr=16, na=4, device="cpu"):
    key = (N, nr, na, str(device))
    if key not in _POLAR:
        f = torch.fft.fftfreq(N, device=device)
        fy, fx = torch.meshgrid(f, f, indexing="ij")
        r = torch.sqrt(fx ** 2 + fy ** 2) / 0.5                  # 1 = Nyquist
        ang = torch.remainder(torch.atan2(fy, fx), math.pi)       # NPS is symmetric -> [0, pi)
        rb = torch.clamp((r * nr).long(), max=nr)                 # nr = beyond Nyquist (corners) -> dropped
        ab = torch.clamp((ang / math.pi * na).long(), max=na - 1)
        idx = rb * na + ab
        idx[(r == 0) | (rb >= nr)] = nr * na                      # DC and corners -> trash bin
        _POLAR[key] = idx.flatten()
    return _POLAR[key]


def polar_nps(nps, nr=16, na=4):
    """2-D NPS (N,N) -> (nr*na,) mean power per (radial, angular) bin."""
    N = nps.shape[-1]
    idx = polar_index(N, nr, na, nps.device)
    s = torch.zeros(nr * na + 1, device=nps.device, dtype=nps.dtype).index_add(0, idx, nps.flatten())
    c = torch.zeros(nr * na + 1, device=nps.device, dtype=nps.dtype).index_add(0, idx, torch.ones_like(nps.flatten()))
    return (s / c.clamp(min=1))[:-1]


def radial_nps(nps, nr=32):
    return polar_nps(nps, nr, 1)


# ----------------------------------------------------------------------------- wavelet statistics
def subband_cov(img, mask, J=3):
    """Orientation covariance (J,3,3) of the detail coefficients inside mask (B,1,H,W)."""
    det, _ = wavelet.forward(img, J)
    m = mask.float()
    n = m.sum().clamp(min=1)
    covs = []
    for d in det:                                   # (B,1,3,H,W)
        v = d[:, 0] * m                             # (B,3,H,W)
        covs.append(torch.einsum("bihw,bjhw->ij", v, d[:, 0]) / n)
    return torch.stack(covs)


def subband_energy(img, mask, J=3):
    det, a = wavelet.forward(texture(img, 4.0), J)
    m = mask.float()
    n = m.sum().clamp(min=1)
    e = [((d[:, 0] ** 2) * m).sum((0, 2, 3)) / n for d in det]          # each (3,)
    e = torch.cat(e + [((a ** 2) * m).sum().view(1) / n])
    return e / e.sum().clamp(min=1e-8)


def logm_spd(C, eps=1e-6):
    w, V = torch.linalg.eigh(C + eps * torch.eye(C.shape[-1], device=C.device, dtype=C.dtype))
    return V @ torch.diag_embed(torch.log(w.clamp(min=eps))) @ V.transpose(-1, -2)


# ----------------------------------------------------------------------------- sliced Wasserstein
def sample_patches(t, mask, n, size=8, generator=None):
    """Random size x size patches of t (B,1,H,W) centred in mask; mean removed. -> (n, size*size)."""
    B, _, H, W = t.shape
    valid = mask.clone()
    h = size // 2
    valid[..., :h, :] = False
    valid[..., H - size + h:, :] = False
    valid[..., :, :h] = False
    valid[..., :, W - size + h:] = False
    idx = torch.nonzero(valid[:, 0])
    if len(idx) == 0:
        return t.new_zeros(0, size * size)
    sel = idx[torch.randint(len(idx), (n,), generator=generator, device="cpu").to(idx.device)]
    pos = torch.stack([sel[:, 0], sel[:, 1] - h, sel[:, 2] - h], 1)
    p = extract(t, pos, size).reshape(n, -1)
    return p - p.mean(1, keepdim=True)


def swd(a, b, n_proj=128, generator=None):
    """Sliced 2-Wasserstein between two equal-size point clouds (n,D)."""
    if len(a) == 0 or len(b) == 0:
        return a.new_zeros(())
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    P = torch.randn(a.shape[1], n_proj, generator=generator, device="cpu").to(a.device, a.dtype)
    P = P / P.norm(dim=0, keepdim=True)
    pa, _ = torch.sort(a @ P, 0)
    pb, _ = torch.sort(b @ P, 0)
    return ((pa - pb) ** 2).mean().sqrt()


# ----------------------------------------------------------------------------- edge profiles (signal transfer)
@torch.no_grad()
def edge_points(nd, kind, K=256, generator=None, margin=10):
    """Boundary points of the smoothed NDCT and their unit normals (pointing to higher HU).
    kind 'hi': strong edges (|grad| > 60 HU/px, bone/air); 'lo': soft-tissue edges (8..25 HU/px).
    -> list (per image) of (pts (k,2) [y,x], nrm (k,2))."""
    g = gauss(nd, 2.0)
    gy = (F.pad(g, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(g, (0, 0, 1, 1), mode="replicate")[..., :-2, :]) / 2
    gx = (F.pad(g, (1, 1, 0, 0), mode="replicate")[..., :, 2:] - F.pad(g, (1, 1, 0, 0), mode="replicate")[..., :, :-2]) / 2
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    sel = (mag > 60) if kind == "hi" else ((mag > 8) & (mag < 25))
    sel[..., :margin, :] = False
    sel[..., -margin:, :] = False
    sel[..., :, :margin] = False
    sel[..., :, -margin:] = False
    out = []
    for b in range(nd.shape[0]):
        yx = torch.nonzero(sel[b, 0])
        if len(yx) > K:
            yx = yx[torch.randperm(len(yx), generator=generator, device="cpu")[:K].to(yx.device)]
        m = mag[b, 0, yx[:, 0], yx[:, 1]].clamp(min=1e-6)
        nrm = torch.stack([gy[b, 0, yx[:, 0], yx[:, 1]] / m, gx[b, 0, yx[:, 0], yx[:, 1]] / m], 1)
        out.append((yx.float(), nrm))
    return out


def mean_profile(img, pts_list, half=6.0, step=0.5):
    """Average edge profile (S,) of img (B,1,H,W) across all boundary points (bilinear sampling along
    the normal). Averaging over many points removes the zero-mean noise and keeps the signal transfer."""
    s = torch.arange(-half, half + 1e-6, step, device=img.device, dtype=img.dtype)
    B, _, H, W = img.shape
    acc, n = 0, 0
    for b, (p, nrm) in enumerate(pts_list):
        if len(p) == 0:
            continue
        yy = p[:, 0:1] + s[None] * nrm[:, 0:1]
        xx = p[:, 1:2] + s[None] * nrm[:, 1:2]
        grid = torch.stack([(xx + 0.5) / W * 2 - 1, (yy + 0.5) / H * 2 - 1], -1)[None]   # (1,k,S,2)
        v = F.grid_sample(img[b:b + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)
        acc = acc + v[0, 0].sum(0)
        n += len(p)
    return acc / max(n, 1) if n else None


def esf_stats(prof, step=0.5):
    """Edge width (10-90 % for a Gaussian ESF = 2.563 sigma, in px) and contrast (HU) of a mean profile."""
    d = torch.relu(prof[1:] - prof[:-1])
    w = d / d.sum().clamp(min=1e-6)
    s = torch.arange(len(d), device=prof.device, dtype=prof.dtype) * step
    mu = (w * s).sum()
    sig = torch.sqrt((w * (s - mu) ** 2).sum().clamp(min=1e-8))
    k = max(2, len(prof) // 6)
    return 2.563 * sig, prof[-k:].mean() - prof[:k].mean()


# ----------------------------------------------------------------------------- degradation descriptor
DESC_DIM = 16 * 4 + 10 + 18 + 6


@torch.no_grad()
def descriptor(img, nd, dx=1.0, masks=None, J=3):
    """Physics-spectral degradation descriptor Phi of ONE image (1,1,H,W), measured in flat regions
    defined on its NDCT nd. Layout: log polar NPS normalised by variance (64) | subband energy
    fractions (10) | per-scale log-Euclidean orientation covariance, trace-normalised (18) |
    log noise std per class relative to soft tissue (6)."""
    masks = class_masks(nd) if masks is None else masks
    flat = masks[:, :4].any(1, keepdim=True)              # fat, fluid, soft, dense
    pos = roi_positions(flat)
    feats = []
    nps = nps2d(extract(img, pos), dx) if len(pos) else None
    if nps is None:
        feats.append(torch.zeros(64, device=img.device))
    else:
        pol = polar_nps(nps)
        feats.append(torch.log(pol + 1e-8) - torch.log(pol.mean() + 1e-8))
    feats.append(subband_energy(img, flat, J))
    C = subband_cov(texture(img, 4.0), flat, J)
    L = []
    for j in range(J):
        Cj = C[j] / torch.trace(C[j]).clamp(min=1e-8)
        Lj = logm_spd(Cj)
        iu = torch.triu_indices(3, 3)
        L.append(Lj[iu[0], iu[1]])
    feats.append(torch.cat(L))
    t = texture(img, 2.0)
    sd = []
    for c in range(len(CLASSES)):
        m = masks[:, c:c + 1]
        sd.append(t[m].std() if m.sum() > 200 else torch.tensor(float("nan"), device=img.device))
    sd = torch.stack(sd)
    ref = sd[SOFT] if torch.isfinite(sd[SOFT]) else torch.nanmean(sd)
    rel = torch.log(sd / ref)
    feats.append(torch.nan_to_num(rel, nan=0.0))
    return torch.cat([f.float().flatten() for f in feats])
