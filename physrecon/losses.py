"""Training objectives. Every target is a REAL LDCT of the target domain and dose; the NDCT is only used
to define WHERE statistics are measured (masks, ROIs, boundary points). No term uses the NDCT as a target,
and no term can be lowered by adding less degradation:

  L_NPS   log 2-D NPS per tissue class (ROIs at matched locations; statistics, not pixel pairs)
  L_cov   log-Euclidean distance of per-scale orientation covariances + log noise std per class
  L_dist  sliced Wasserstein between 8x8 texture-patch distributions (pixel space, single scale)
  L_ST    signal transfer: class mean HU, edge-spread width and boundary contrast (low/high contrast)
  L_mean  zero-mean degradation constraint, UNBIASED: E[<G n1, G n2>] = |E[G n]|^2 for two independent
          physics realisations n1, n2 (penalises a systematic low-frequency bias of the added degradation,
          not its variance - IR noise may legitimately have low-frequency power)
  L_pair  (strategies 'real'/'joint' only) subband L1 to a real LDCT reconstructed from the SAME raw data
          as the real FBP noise input, outside edges.
"""
import torch
import torch.nn.functional as F

from . import stats, wavelet


def _smooth_freq(P):
    """3x3 circular box smoothing of a 2-D NPS (reduces estimator variance)."""
    Q = P[None, None]
    Q = F.pad(Q, (1, 1, 1, 1), mode="circular")
    return F.avg_pool2d(Q, 3, stride=1)[0, 0]


class PRLoss:
    def __init__(self, dx, lam_cov=1.0, lam_dist=1.0, lam_st=0.5, lam_mean=1.0, lam_pair=1.0,
                 n_patches=2048, generator=None):
        self.dx = dx
        self.lam = dict(cov=lam_cov, dist=lam_dist, st=lam_st, mean=lam_mean, pair=lam_pair)
        self.n_patches = n_patches
        self.gen = generator

    def statistics(self, yh, y, x, masks):
        """Statistical / distributional / signal-transfer terms for one synthetic batch yh vs real y."""
        out = {}
        # ---- local 2-D NPS per class (fat, fluid, soft, dense)
        # loss ROIs are half the evaluation ROI size (32 px on 512 images) so that thin tissue classes and
        # crops still provide enough ROIs; classes with < 4 ROIs are replaced by the pooled flat-tissue NPS
        rs = max(16, stats.ROI // 2)
        flat = masks[:, :4].any(1, keepdim=True)

        def nps_term(mask, min_rois):
            pos = stats.roi_positions(mask, size=rs, generator=self.gen)
            if len(pos) < min_rois:
                return None
            Ph = _smooth_freq(stats.nps2d(stats.extract(yh, pos, rs), self.dx))
            Pr = _smooth_freq(stats.nps2d(stats.extract(y, pos, rs), self.dx))
            eps = 1e-3 * Pr.mean()
            return (torch.log(Ph + eps) - torch.log(Pr + eps)).abs().mean()

        parts = [t for t in (nps_term(masks[:, c:c + 1], 4) for c in range(4)) if t is not None]
        if not parts:
            t = nps_term(flat, 2)
            parts = [t] if t is not None else []
        out["nps"] = sum(parts) / len(parts) if parts else yh.new_zeros(())
        # ---- multiscale orientation covariance + noise std per class
        Ch = stats.subband_cov(stats.texture(yh, 4.0), flat)
        Cr = stats.subband_cov(stats.texture(y, 4.0), flat)
        l_cov = sum(torch.linalg.matrix_norm(stats.logm_spd(Ch[j]) - stats.logm_spd(Cr[j])) for j in range(len(Ch)))
        th, tr = stats.texture(yh, 2.0), stats.texture(y, 2.0)
        for c in range(len(stats.CLASSES)):
            m = masks[:, c:c + 1]
            if m.sum() > 200:
                l_cov = l_cov + (torch.log(th[m].std() + 1e-3) - torch.log(tr[m].std() + 1e-3)).abs()
        out["cov"] = l_cov
        # ---- distributional: SWD of texture patches (normalised by the real texture std)
        if flat.sum() > 1000:
            g = self.gen
            sref = tr[flat].std().detach().clamp(min=1.0)
            seed = int(torch.randint(1 << 30, (1,), generator=g).item()) if g is not None else 0
            g1 = torch.Generator().manual_seed(seed)
            ph = stats.sample_patches(th / sref, flat, self.n_patches, generator=g1)
            g1 = torch.Generator().manual_seed(seed)                     # same locations for real patches
            pr = stats.sample_patches(tr / sref, flat, self.n_patches, generator=g1)
            out["dist"] = stats.swd(ph, pr, generator=torch.Generator().manual_seed(seed + 1))
        else:
            out["dist"] = yh.new_zeros(())
        # ---- signal transfer vs real LDCT
        l_st = 0.0
        for c in range(len(stats.CLASSES)):
            m = stats.erode(masks[:, c:c + 1], 2)
            if m.sum() > 200:
                l_st = l_st + (yh[m].mean() - y[m].mean()).abs() / 10.0
        for kind in ("lo", "hi"):
            pts = stats.edge_points(x, kind, generator=torch.Generator().manual_seed(7))
            if sum(len(p) for p, _ in pts) < 64:
                continue
            ph, pr = stats.mean_profile(yh, pts), stats.mean_profile(y, pts)
            wh, ch = stats.esf_stats(ph)
            wr, cr = stats.esf_stats(pr)
            l_st = l_st + (wh - wr).abs() + (ch - cr).abs() / cr.abs().clamp(min=10.0)
        out["st"] = l_st if torch.is_tensor(l_st) else yh.new_zeros(())
        return out

    def __call__(self, yh1, yh2, nh1, nh2, y, x, paired=False):
        masks = stats.class_masks(x)
        s1 = self.statistics(yh1, y, x, masks)
        s2 = self.statistics(yh2, y, x, masks)
        terms = {k: 0.5 * (s1[k] + s2[k]) for k in s1}
        flat = masks[:, :4].any(1, keepdim=True)
        ref = stats.texture(y, 2.0)[flat].var().detach().clamp(min=1.0) if flat.any() else y.new_tensor(100.0)
        # zero-mean constraint needs two realisations of the SAME simulator (not valid for a real-noise input)
        terms["mean"] = (stats.gauss(nh1, 4.0) * stats.gauss(nh2, 4.0)).mean() / ref if not paired else ref * 0
        if paired:
            body = stats.body_mask(x)
            g = stats.gauss(x, 2.0)
            gy = g[..., 2:, 1:-1] - g[..., :-2, 1:-1]
            gx = g[..., 1:-1, 2:] - g[..., 1:-1, :-2]
            edge = F.pad((torch.sqrt(gx ** 2 + gy ** 2) / 2 > 25).float(), (1, 1, 1, 1))
            M = (body.float() * (1 - edge))[:, :, None]
            dh, _ = wavelet.forward(yh1)
            dr, _ = wavelet.forward(y)
            terms["pair"] = sum((M * (a - b)).abs().mean() for a, b in zip(dh, dr)) / ref.sqrt()
        total = terms["nps"] + sum(self.lam[k] * terms[k] for k in ("cov", "dist", "st", "mean") if k in terms)
        if paired:
            total = total + self.lam["pair"] * terms["pair"]
        return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()}
