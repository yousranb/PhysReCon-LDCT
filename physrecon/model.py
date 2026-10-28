"""PhysReCon-LDCT networks.

    y_hat = S_eta(x_ND; d, z_D) + W^{-1}[ W(n_phys) + F_theta(W(n_phys), W(x_ND), d, z_D) ]

Module C  ResponseCNN  F_theta: residual blocks with weights SHARED across the J wavelet scales (dilation
          2^j, matching the undecimated transform), processed coarse-to-fine, FiLM-modulated by (z, log d, j).
          The output conv is zero-initialised: training starts from the physics (FBP) noise.
Module B  Embedder: z_D = rho( mean_i g_psi([dPhi_i, log d_i]) ), permutation-invariant over the support set.
Module D  SignalTransfer S_eta: reconstruction-dependent signal transfer (smooth OR sharpen):
          S(x) = b( g + beta(x) (x - g) ),  g = mixture of self-guided filters,
          beta = (1-w) beta_lo + w beta_hi,  w = |grad g| / (|grad g| + tau)  (contrast-dependent transfer),
          b = monotone piecewise-linear HU curve, |b(x) - x| <= 15 HU. Identity at initialisation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import stats, wavelet


class SharedConv(nn.Module):
    """3x3 conv whose dilation is chosen at call time (same weights for every scale)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def forward(self, x, dil):
        return F.conv2d(F.pad(x, (dil, dil, dil, dil), mode="replicate"), self.weight, self.bias, dilation=dil)


class ResUnit(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.c1, self.c2 = SharedConv(C, C), SharedConv(C, C)
        self.n1 = nn.GroupNorm(8, C)

    def forward(self, h, dil):
        return h + self.c2(F.silu(self.n1(self.c1(h, dil))), dil)


class ResponseCNN(nn.Module):
    def __init__(self, C=48, n_units=4, J=3, zdim=16, extra_in=0):
        super().__init__()
        self.C, self.n_units, self.J, self.extra_in = C, n_units, J, extra_in
        self.inp = SharedConv(3 + 3 + 1 + 1 + C + extra_in, C)
        self.units = nn.ModuleList([ResUnit(C) for _ in range(n_units)])
        self.out = SharedConv(C, 4)
        nn.init.zeros_(self.out.weight)
        self.film = nn.Sequential(nn.Linear(zdim + 1 + J, 64), nn.SiLU(), nn.Linear(64, n_units * 2 * C))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        # fixed normalisation constants (set from piglet training slabs before training)
        self.register_buffer("s_n", torch.ones(J))
        self.register_buffer("s_na", torch.ones(()))
        self.register_buffer("s_x", torch.ones(J))

    def forward(self, n_phys, x, z, logd, extra=None):
        B, _, H, W = x.shape
        wn, an = wavelet.forward(n_phys, self.J)
        wx, _, ax = wavelet.forward(x, self.J, return_approx=True)
        h = x.new_zeros(B, self.C, H, W)
        out_det = [None] * self.J
        a_hat = an
        for j in reversed(range(self.J)):
            dil = 2 ** j
            parts = [wn[j][:, 0] / self.s_n[j], wx[j][:, 0] / self.s_x[j], (ax[j] - 40.0) / 400.0,
                     an / self.s_na if j == self.J - 1 else torch.zeros_like(an), h]
            if self.extra_in:
                parts.append(extra)
            h = F.silu(self.inp(torch.cat(parts, 1), dil))
            onehot = F.one_hot(torch.full((B,), j, device=x.device, dtype=torch.long), self.J).float()
            gb = self.film(torch.cat([z, logd[:, None], onehot], 1)).view(B, self.n_units, 2, self.C)
            for u, unit in enumerate(self.units):
                h = unit(h, dil)
                h = h * (1 + gb[:, u, 0, :, None, None]) + gb[:, u, 1, :, None, None]
            o = self.out(h, dil)
            out_det[j] = wn[j] + o[:, None, :3] * self.s_n[j]
            if j == self.J - 1:
                a_hat = an + o[:, 3:4] * self.s_na
        return wavelet.inverse(out_det, a_hat)


class Embedder(nn.Module):
    def __init__(self, din=stats.DESC_DIM, zdim=16, hidden=64):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(din + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.rho = nn.Linear(hidden, zdim)
        self.register_buffer("mu", torch.zeros(din))
        self.register_buffer("sd", torch.ones(din))

    def forward(self, dphi, logd):
        """dphi (n, din) raw descriptor differences, logd (n,) -> z (zdim,)."""
        u = (dphi - self.mu) / self.sd
        return self.rho(self.g(torch.cat([u, logd[:, None]], 1)).mean(0))


class SignalTransfer(nn.Module):
    RADII = (1, 2, 4)
    KNOTS = torch.linspace(-1000.0, 1500.0, 8)

    def __init__(self, zdim=16, hidden=32, max_offset=15.0, beta_max=1.5, noise_adaptive=False):
        super().__init__()
        self.max_offset, self.beta_max, self.noise_adaptive = max_offset, beta_max, noise_adaptive
        n_out = 2 + len(self.RADII) + 2 + len(self.KNOTS) + (1 if noise_adaptive else 0)
        self.h = nn.Sequential(nn.Linear(zdim + 1, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.h[-1].weight)
        b0 = torch.zeros(n_out)
        b0[0:2] = math.log(2.0)          # beta = 1.5 * sigmoid(log 2) = 1  (identity transfer)
        self.h[-1].bias.data.copy_(b0)

    def params(self, z, logd):
        o = self.h(torch.cat([z, logd[:, None]], 1))
        k = len(self.RADII)
        beta = self.beta_max * torch.sigmoid(o[:, 0:2])
        mix = torch.softmax(o[:, 2:2 + k], 1)
        eps = 400.0 * torch.exp(o[:, 2 + k])               # HU^2, 400 at init (20 HU)
        tau = 20.0 * torch.exp(o[:, 3 + k])                # HU/px
        off = self.max_offset * torch.tanh(o[:, 4 + k:4 + k + len(self.KNOTS)])
        out = dict(beta_lo=beta[:, 0], beta_hi=beta[:, 1], mix=mix, eps=eps, tau=tau, offsets=off)
        if self.noise_adaptive:
            out["gamma"] = 0.5 * torch.tanh(o[:, -1])          # 0 at initialisation
        return out

    @staticmethod
    def guided(x, r, eps):
        m = stats.box(x, r)
        v = torch.clamp(stats.box(x * x, r) - m * m, min=0)
        a = v / (v + eps)
        b = m - a * m
        return stats.box(a, r) * x + stats.box(b, r)

    def forward(self, x, z, logd, noise_map=None):
        """noise_map (optional, noise-adaptive transfer): standardised log variance map of the physical noise,
        computed from the noisy sinogram. IR regularises more where the projection data are noisier, so the
        low-contrast detail gain becomes beta_lo + gamma * noise_map (gamma learned, 0 at initialisation)."""
        p = self.params(z, logd)
        B = x.shape[0]
        eps = p["eps"].view(B, 1, 1, 1)
        g = sum(p["mix"][:, i].view(B, 1, 1, 1) * self.guided(x, r, eps) for i, r in enumerate(self.RADII))
        gy = F.pad(g, (0, 0, 1, 1), mode="replicate")
        gx = F.pad(g, (1, 1, 0, 0), mode="replicate")
        grad = torch.sqrt(((gy[..., 2:, :] - gy[..., :-2, :]) / 2) ** 2 + ((gx[..., :, 2:] - gx[..., :, :-2]) / 2) ** 2 + 1e-6)
        w = grad / (grad + p["tau"].view(B, 1, 1, 1))
        b_lo = p["beta_lo"].view(B, 1, 1, 1)
        if self.noise_adaptive and noise_map is not None:
            b_lo = torch.clamp(b_lo + p["gamma"].view(B, 1, 1, 1) * noise_map, 0.0, self.beta_max)
        beta = (1 - w) * b_lo + w * p["beta_hi"].view(B, 1, 1, 1)
        pre = g + beta * (x - g)
        knots = self.KNOTS.to(x.device)
        step = knots[1] - knots[0]
        t = torch.clamp((pre - knots[0]) / step, 0, len(knots) - 1 - 1e-4)
        i = t.floor().long()
        f = t - i
        off = p["offsets"]                                  # (B, K)
        o0 = torch.gather(off, 1, i.view(B, -1)).view_as(pre)
        o1 = torch.gather(off, 1, (i + 1).clamp(max=len(knots) - 1).view(B, -1)).view_as(pre)
        return pre + (1 - f) * o0 + f * o1


class NoiseGain(nn.Module):
    """Reconstruction noise-transfer gains: one positive gain per wavelet scale (J detail scales + approximation),
    a function of (z, log d). Models how a (commercial) iterative reconstruction attenuates the noise band by band
    (strong high-frequency suppression -> small fine-scale gains), and absorbs a global noise-level mismatch
    when the physics (I0) is transferred from another scanner. Identity (all gains 1) at initialisation, so a
    checkpoint trained without it behaves exactly as before."""

    def __init__(self, zdim=16, J=3, hidden=32, log_max=2.5):
        super().__init__()
        self.J, self.log_max = J, log_max
        self.h = nn.Sequential(nn.Linear(zdim + 1, hidden), nn.SiLU(), nn.Linear(hidden, J + 1))
        nn.init.zeros_(self.h[-1].weight)
        nn.init.zeros_(self.h[-1].bias)

    def gains(self, z, logd):
        return torch.exp(self.log_max * torch.tanh(self.h(torch.cat([z, logd[:, None]], 1))))   # (B, J+1)

    def forward(self, n, z, logd):
        g = self.gains(z, logd)
        det, a = wavelet.forward(n, self.J)
        B = n.shape[0]
        det = [d * g[:, j].view(B, 1, 1, 1, 1) for j, d in enumerate(det)]
        return wavelet.inverse(det, a * g[:, self.J].view(B, 1, 1, 1))


class ProjectionResponse(nn.Module):
    """Coefficients (a_w, b_w) of the projection-domain statistical-weighting response (see
    PhysicsSim.noise_basis): how strongly the reconstruction smooths the sinogram noise, uniformly (a_w) and
    in proportion to the ray variance (b_w). 0 at initialisation = plain FBP of the physical noise."""

    def __init__(self, zdim=16, n_widths=2, hidden=32):
        super().__init__()
        self.W = n_widths
        self.h = nn.Sequential(nn.Linear(zdim + 1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * n_widths))
        nn.init.zeros_(self.h[-1].weight)
        nn.init.zeros_(self.h[-1].bias)

    def coeffs(self, z, logd):
        o = self.h(torch.cat([z, logd[:, None]], 1)).view(-1, self.W, 2)
        return torch.stack([torch.tanh(o[..., 0]), 0.5 * torch.tanh(o[..., 1])], -1).view(-1, 2 * self.W)

    def forward(self, n_phys, basis, z, logd):
        c = self.coeffs(z, logd)
        return n_phys - (c[:, :, None, None] * basis).sum(1, keepdim=True)


class SignalBlur(nn.Module):
    """Linear (not edge-preserving) resolution loss of the reconstruction, after the edge-preserving signal
    transfer: s' = s + sum_k a_k (G_k * s - s), sigma_k in {1, 2, 4} px, a_k in [0, 1] (sum <= 1).
    Strong commercial IR at very low dose also blurs high-contrast edges, which a guided filter keeps.
    Identity at initialisation."""
    SIGMAS = (1.0, 2.0, 4.0)

    def __init__(self, zdim=16, hidden=32):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(zdim + 1, hidden), nn.SiLU(), nn.Linear(hidden, len(self.SIGMAS)))
        nn.init.zeros_(self.h[-1].weight)
        nn.init.zeros_(self.h[-1].bias)

    def weights(self, z, logd):
        a = torch.clamp(self.h(torch.cat([z, logd[:, None]], 1)), 0.0, 1.0)
        return a / torch.clamp(a.sum(1, keepdim=True), min=1.0)

    @staticmethod
    def gauss(x, sig):
        r = int(math.ceil(3 * sig))
        t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-0.5 * (t / sig) ** 2)
        k = k / k.sum()
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))

    def forward(self, s, z, logd):
        a = self.weights(z, logd)
        B = s.shape[0]
        return s + sum(a[:, k].view(B, 1, 1, 1) * (self.gauss(s, sg) - s) for k, sg in enumerate(self.SIGMAS))


class PhysReCon(nn.Module):
    def __init__(self, C=48, n_units=4, J=3, zdim=16, use_varmap=False, st_varmap=False, noise_gain=True,
                 ng_J=3, proj_resp=False, sig_blur=False):
        super().__init__()
        self.cfg = dict(C=C, n_units=n_units, J=J, zdim=zdim, use_varmap=use_varmap, st_varmap=st_varmap,
                        noise_gain=noise_gain, ng_J=ng_J, proj_resp=proj_resp, sig_blur=sig_blur)
        self.cnn = ResponseCNN(C, n_units, J, zdim, extra_in=1 if use_varmap else 0)
        self.emb = Embedder(zdim=zdim)
        self.st = SignalTransfer(zdim, noise_adaptive=st_varmap)
        self.ng = NoiseGain(zdim, ng_J) if noise_gain else None
        self.pw = ProjectionResponse(zdim) if proj_resp else None
        self.blur = SignalBlur(zdim) if sig_blur else None

    @property
    def needs_varmap(self):
        return self.cfg["use_varmap"] or self.cfg.get("st_varmap", False)

    @property
    def needs_basis(self):
        return self.pw is not None

    def forward(self, x, n_phys, logd, z, varmap=None, x_cnn=None, n_basis=None):
        """x, n_phys (B,1,H,W) HU; logd (B,); z (B,zdim). Returns (y_hat, n_hat, s).
        x_cnn (ablation A3): anatomy given to the response CNN instead of x; the signal transfer always uses x."""
        nmap = None
        if varmap is not None:
            lv = torch.log(varmap.clamp(min=1e-3))
            nmap = (lv - lv.mean((2, 3), keepdim=True)) / (lv.std((2, 3), keepdim=True) + 1e-6)
        if self.pw is not None and n_basis is not None:
            n_phys = self.pw(n_phys, n_basis, z, logd)
        n_hat = self.cnn(n_phys, x if x_cnn is None else x_cnn, z, logd, nmap if self.cfg["use_varmap"] else None)
        if self.ng is not None:
            n_hat = self.ng(n_hat, z, logd)
        s = self.st(x, z, logd, nmap if self.cfg.get("st_varmap") else None)
        if self.blur is not None:
            s = self.blur(s, z, logd)
        return s + n_hat, n_hat, s

    def load_compatible(self, state):
        """Load a checkpoint saved before NoiseGain existed: the missing gains stay at identity."""
        own = self.state_dict()
        state = dict(state)
        for k in list(state):
            if k in own and own[k].shape != state[k].shape and k.startswith("ng."):
                print(f"[model] {k}: shape {tuple(state[k].shape)} -> {tuple(own[k].shape)}, re-initialised (identity)")
                del state[k]
        missing, unexpected = self.load_state_dict(state, strict=False)
        new = ("ng.", "pw.", "blur.")
        bad = [k for k in missing if not k.startswith(new)] + list(unexpected)
        if bad:
            raise RuntimeError(f"checkpoint incompatible: {bad[:5]}")
        if missing:
            print(f"[model] {len(missing)} new-module tensors not in checkpoint -> identity initialisation "
                  f"({sorted(set(k.split('.')[0] for k in missing))})")

    def describe(self, z, logd):
        """Interpretable response parameters for logging (one sample)."""
        out = {}
        with torch.no_grad():
            if self.ng is not None:
                out["noise_gains"] = self.ng.gains(z, logd)[0].tolist()
            if self.pw is not None:
                out["proj_coeffs(a1,b1,a3,b3)"] = self.pw.coeffs(z, logd)[0].tolist()
            if self.blur is not None:
                out["blur(s1,s2,s4)"] = self.blur.weights(z, logd)[0].tolist()
        return "  ".join(f"{k}=" + ",".join(f"{v:.2f}" for v in vs) for k, vs in out.items())

    def adapted_parameters(self, mode):
        """Parameters updated in stage 2 for each adaptation mode (F_theta trunk always frozen except 'full')."""
        heads = list(self.st.h.parameters()) + [p for m in (self.ng, self.pw, self.blur) if m is not None
                                                 for p in m.h.parameters()]
        if mode == "zfilm":                      # domain conditioning of F_theta (last FiLM layer) + response heads
            return list(self.cnn.film[-1].parameters()) + heads
        if mode == "heads":                      # response heads only; FiLM of F_theta frozen (ablation)
            return heads
        if mode in ("full", "scratch"):
            return list(self.parameters())
        return []
