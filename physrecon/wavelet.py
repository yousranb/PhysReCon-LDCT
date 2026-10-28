"""Undecimated (a trous) Haar decomposition with an exact inverse by summation.

1-D:  L_s x(n) = (x(n) + x(n+s)) / 2,   H_s x(n) = (x(n) - x(n+s)) / 2,   L_s + H_s = identity.
2-D separable at level j (s = 2^j):  a_j = LL a_j + LH a_j + HL a_j + HH a_j  exactly, so
    x = a_J + sum_{j<J} (LH_j + HL_j + HH_j)        (W^{-1} = sum of all subbands).
Every subband keeps the full H x W size (shift-invariant). W^{-1} is a plain sum, so ANY modified set
of coefficients maps to a valid image; statistics are therefore always computed on the output image.
Boundary handling: replicate padding.
"""
import torch
import torch.nn.functional as F


def _shift(x, s, dim):
    """x(n+s) along dim (2 = rows, 3 = cols), replicate padding at the end."""
    if dim == 3:
        return F.pad(x, (0, s, 0, 0), mode="replicate")[..., :, s:]
    return F.pad(x, (0, 0, 0, s), mode="replicate")[..., s:, :]


def forward(x, J=3, return_approx=False):
    """x (B,C,H,W) -> list of J detail tensors (B,C,3,H,W) [LH, HL, HH] (fine to coarse) and a_J (B,C,H,W).
    With return_approx, also the list of approximations [a_1, ..., a_J]."""
    a, details, approx = x, [], []
    for j in range(J):
        s = 2 ** j
        lr = (a + _shift(a, s, 3)) / 2          # low along columns (horizontal)
        hr = (a - _shift(a, s, 3)) / 2
        ll = (lr + _shift(lr, s, 2)) / 2
        lh = (lr - _shift(lr, s, 2)) / 2        # low horizontally, high vertically
        hl = (hr + _shift(hr, s, 2)) / 2
        hh = (hr - _shift(hr, s, 2)) / 2
        details.append(torch.stack([lh, hl, hh], 2))
        a = ll
        approx.append(a)
    return (details, a, approx) if return_approx else (details, a)


def inverse(details, approx):
    out = approx
    for d in details:
        out = out + d.sum(2)
    return out
