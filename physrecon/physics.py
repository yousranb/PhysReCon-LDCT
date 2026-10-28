"""Module A - shared acquisition physics.

    mu  = mu_w * max(0, 1 + HU/1000)                       attenuation (monochromatic approximation)
    p   = A mu * dx                                        parallel-beam line integrals (full image, padded)
    eps ~ N(0, var_d(p)),  var_d = (1-d)/d * e^p/I0 * (1 + (1+d)/d * se2 * e^p/I0)     (Zeng et al. 2015)
    n_phys = FBP_kappa(eps) / (mu_w dx) * 1000             HU; Hann-apodized ramp with cutoff kappa

The variance is INCREMENTAL (what is missing between full dose and dose d), so the noise already present
in the NDCT is not counted twice. FBP is linear: FBP(p + eps) - FBP(p) = FBP(eps).
Calibrated parameters (piglet FBP training data only): I0, se2 (electronic noise variance), kappa.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

MU_WATER = 0.0192   # 1/mm, effective energy of a ~120 kVp beam


class ParallelBeam:
    """Batched parallel-beam projector / FBP on the GPU.
    The n x n image is zero-padded to m = ceil(sqrt(2) n) (even) so the whole square image is projected
    (no inscribed-circle clipping). Projection: rotate + sum along rows (bilinear). Backprojection: the
    filtered sinogram is sampled at t = x cos(th) + y sin(th) with grid_sample."""

    def __init__(self, n, n_views=720, device=None, chunk=48):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n, self.A, self.chunk = n, n_views, chunk
        m = int(math.ceil(math.sqrt(2) * n))
        self.m = m + (m % 2)
        th = torch.arange(n_views, dtype=torch.float32, device=self.dev) * math.pi / n_views
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        self.P = max(64, int(2 ** math.ceil(math.log2(2 * self.m))))
        self._filters = {}
        c = (self.m - 1) / 2
        ax = torch.arange(self.m, dtype=torch.float32, device=self.dev) - c
        self.yy, self.xx = torch.meshgrid(ax, ax, indexing="ij")   # pixel centres, padded grid
        a0 = torch.arange(n, dtype=torch.float32, device=self.dev) - (n - 1) / 2
        self.yn, self.xn = torch.meshgrid(a0, a0, indexing="ij")   # pixel centres of the n x n output

    # ----------------------------------------------------------------- helpers
    def pad(self, img):
        """(B,1,n,n) -> (B,1,m,m) centred zero padding."""
        p = (self.m - self.n) // 2
        return F.pad(img, (p, self.m - self.n - p, p, self.m - self.n - p))

    def crop(self, img):
        p = (self.m - self.n) // 2
        return img[..., p:p + self.n, p:p + self.n]

    def ramp(self, kappa=1.0):
        """Ramp filter * Hann window with cutoff kappa (fraction of Nyquist), cached by kappa."""
        key = round(float(kappa), 4)
        if key not in self._filters:
            f = np.fft.fftfreq(self.P)                    # cycles/sample, Nyquist = 0.5
            r = 2 * np.abs(f)
            fc = 0.5 * max(1e-3, float(kappa))
            w = np.where(np.abs(f) < fc, 0.5 * (1 + np.cos(np.pi * f / fc)), 0.0)
            self._filters[key] = torch.tensor(r * w, dtype=torch.float32, device=self.dev)
        return self._filters[key]

    # ----------------------------------------------------------------- operators
    def radon(self, img):
        """img (B,1,n,n) -> sinogram (B, A, m)  [sum over pixels, unit pixel size]."""
        x = self.pad(img.to(self.dev, torch.float32))
        B = x.shape[0]
        out = []
        for i in range(0, self.A, self.chunk):
            c, s = self.cos[i:i + self.chunk], self.sin[i:i + self.chunk]
            k = len(c)
            # rotate every image by every angle of the chunk: (B*k, 1, m, m)
            Arot = torch.zeros(B * k, 2, 3, device=self.dev)
            cc, ss = c.repeat(B), s.repeat(B)
            Arot[:, 0, 0], Arot[:, 0, 1], Arot[:, 1, 0], Arot[:, 1, 1] = cc, -ss, ss, cc
            grid = F.affine_grid(Arot, (B * k, 1, self.m, self.m), align_corners=False)
            xr = F.grid_sample(x.repeat_interleave(k, 0), grid, mode="bilinear", padding_mode="zeros",
                               align_corners=False)
            out.append(xr.sum(2)[:, 0].view(B, k, self.m))           # sum along y -> detector t
        return torch.cat(out, 1)

    def fbp(self, sino, kappa=1.0):
        """sino (B, A, m) -> image (B,1,n,n). Normalised so that fbp(radon(f)) ~ f."""
        B = sino.shape[0]
        q = torch.fft.ifft(torch.fft.fft(sino, n=self.P, dim=2) * self.ramp(kappa), dim=2).real[..., :self.m]
        img = torch.zeros(B, self.n * self.n, device=self.dev)
        X, Y = self.xn.reshape(-1), self.yn.reshape(-1)          # only the n x n output pixels
        c0 = (self.m - 1) / 2
        for i in range(0, self.A, self.chunk):
            c, s = self.cos[i:i + self.chunk], self.sin[i:i + self.chunk]
            k = len(c)
            # detector coordinate of every pixel for every angle of the chunk (the forward rotation maps
            # output pixel (x,y) to source (c x - s y, s x + c y); summing over y gives t = x for the
            # rotated image, i.e. t = c X + s Y in the original frame)
            t = c[:, None] * X[None] + s[:, None] * Y[None]                    # (k, m*m)
            gx = (t + c0 + 0.5) / self.m * 2 - 1                                # normalised detector coord
            gy = ((torch.arange(k, device=self.dev, dtype=torch.float32) + 0.5) / k * 2 - 1)[:, None].expand_as(gx)
            grid = torch.stack([gx, gy], -1)[None].expand(B, k, self.n * self.n, 2)
            v = F.grid_sample(q[:, None, i:i + k], grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)                             # (B,1,k,n*n)
            img += v[:, 0].sum(1)
        return img.view(B, 1, self.n, self.n) * math.pi / (2 * self.A)


class PhysicsSim:
    """Incremental dose-reduction noise in the sinogram domain + reference FBP."""

    def __init__(self, n, I0=1e5, se2=10.0, kappa=1.0, n_views=720, device=None):
        self.proj = ParallelBeam(n, n_views, device)
        self.dev = self.proj.dev
        self.I0, self.se2, self.kappa = float(I0), float(se2), float(kappa)
        self._cache = {}

    def params(self):
        return dict(I0=self.I0, se2=self.se2, kappa=self.kappa, n_views=self.proj.A)

    @staticmethod
    def mu(hu):
        return MU_WATER * torch.clamp(1 + torch.clamp(hu, min=-1024) / 1000.0, min=0)

    def sinogram(self, ndct_hu, dx, keys=None):
        """(B,1,n,n) HU -> line integrals (B,A,m). Optional cache keyed by `keys` (one key per image)."""
        if keys is not None and all(k in self._cache for k in keys):
            return torch.stack([self._cache[k] for k in keys]).float()
        p = self.proj.radon(self.mu(ndct_hu.to(self.dev))) * dx
        if keys is not None:
            for k, pi in zip(keys, p):
                self._cache[k] = pi.half()
        return p

    def variance(self, p, d):
        """d: (B,) dose fractions -> var (B,A,m)."""
        d = torch.as_tensor(d, dtype=torch.float32, device=self.dev).view(-1, 1, 1)
        e = torch.exp(torch.clamp(p, max=20.0)) / self.I0
        return (1 - d) / d * e * (1 + (1 + d) / d * self.se2 * e)

    def noise(self, p, d, dx, generator=None, n_real=1):
        """physical noise realisations in HU: (n_real*B, 1, n, n), realisations stacked along the batch."""
        var = self.variance(p, d)
        if n_real > 1:
            var = var.repeat(n_real, 1, 1)
        eps = torch.sqrt(var) * torch.randn(var.shape, device=self.dev, generator=generator)
        return self.proj.fbp(eps, self.kappa) / (MU_WATER * dx) * 1000.0

    @staticmethod
    def _gauss_sino(e, w):
        """2-D Gaussian smoothing of sinograms (B, A, m) along views and detector bins, sigma = w bins."""
        r = int(math.ceil(3 * w))
        t = torch.arange(-r, r + 1, dtype=torch.float32, device=e.device)
        k = torch.exp(-0.5 * (t / w) ** 2)
        k = k / k.sum()
        x = e[:, None]
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="circular"), k.view(1, 1, -1, 1))
        return x[:, 0]

    def noise_basis(self, p, d, dx, generator=None, n_real=1, widths=(1.0, 3.0)):
        """Physical noise n_phys plus the basis of a projection-domain statistical-weighting response.

        Iterative reconstruction weights every ray by its inverse variance and regularises: noisy rays are
        smoothed more (PWLS sinogram smoothing is its classical first-order model). With
            D_w = eps - G_w * eps                    (high-pass part of the sinogram noise at width w)
            s   = standardised log variance of each ray
        a ray-dependent smoothing  eps' = eps - sum_w (a_w + b_w s) D_w  is LINEAR in (a_w, b_w), so
            FBP(eps') = n_phys - sum_w [a_w FBP(D_w) + b_w FBP(s D_w)].
        Returns n_phys (nB,1,n,n) and the basis images (nB, 2W, n, n) in HU; the coefficients (a_w, b_w)
        are learned (model.pw), 0 = pure FBP physics."""
        var = self.variance(p, d)
        if n_real > 1:
            var = var.repeat(n_real, 1, 1)
        eps = torch.sqrt(var) * torch.randn(var.shape, device=self.dev, generator=generator)
        lv = torch.log(var.clamp(min=1e-12))
        w8 = (var > 1e-9).float()
        mu = (lv * w8).sum((1, 2), keepdim=True) / w8.sum((1, 2), keepdim=True).clamp(min=1)
        sd = torch.sqrt((((lv - mu) * w8) ** 2).sum((1, 2), keepdim=True) / w8.sum((1, 2), keepdim=True).clamp(min=1))
        sv = ((lv - mu) / (sd + 1e-6)).clamp(-3, 3) * w8
        parts = [eps]
        for w in widths:
            D = eps - self._gauss_sino(eps, w)
            parts += [D, sv * D]
        B = eps.shape[0]
        img = self.proj.fbp(torch.cat(parts), self.kappa) / (MU_WATER * dx) * 1000.0
        img = img.view(len(parts), B, 1, *img.shape[-2:])
        return img[0], img[1:, :, 0].permute(1, 0, 2, 3).contiguous()

    def variance_map(self, p, d, dx):
        """Analytic per-pixel variance of n_phys (HU^2): backprojection of var filtered by the squared
        kernel (projection-feature ablation P1). Uses the discrete identity Var[sum_k h_k e_k] = sum h_k^2 var_k.
        Verified against 1000 realisations: spatial correlation 0.99; the absolute level is ~0.82x the empirical
        variance because bilinear interpolation in the backprojection averages neighbouring bins. Use it as a
        normalised spatial map (log, standardised), not as an absolute variance."""
        var = self.variance(p, d)
        h = torch.fft.ifft(self.proj.ramp(self.kappa)).real               # spatial kernel (P,)
        H2 = torch.fft.fft(h ** 2)                                         # filter with squared kernel
        q = torch.fft.ifft(torch.fft.fft(var, n=self.proj.P, dim=2) * H2, dim=2).real[..., :self.proj.m]
        # backprojection of independent per-view contributions: variance adds with weight (pi/2A)^2
        B = var.shape[0]
        img = torch.zeros(B, self.proj.n * self.proj.n, device=self.dev)
        X, Y = self.proj.xn.reshape(-1), self.proj.yn.reshape(-1)
        c0 = (self.proj.m - 1) / 2
        for i in range(0, self.proj.A, self.proj.chunk):
            c, s = self.proj.cos[i:i + self.proj.chunk], self.proj.sin[i:i + self.proj.chunk]
            k = len(c)
            t = c[:, None] * X[None] + s[:, None] * Y[None]
            gx = (t + c0 + 0.5) / self.proj.m * 2 - 1
            gy = ((torch.arange(k, device=self.dev, dtype=torch.float32) + 0.5) / k * 2 - 1)[:, None].expand_as(gx)
            grid = torch.stack([gx, gy], -1)[None].expand(B, k, -1, 2)
            img += F.grid_sample(q[:, None, i:i + k], grid, align_corners=False)[:, 0].sum(1)
        img = img.view(B, 1, self.proj.n, self.proj.n) * (math.pi / (2 * self.proj.A)) ** 2
        return img * (1000.0 / (MU_WATER * dx)) ** 2
