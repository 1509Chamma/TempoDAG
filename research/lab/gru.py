"""Torch GRU cell and its parallel DEER solvers (full + diagonal quasi-DEER).

Validated in NB01 section 5. `deer_gru` runs Newton over the whole sequence; each
iteration solves a (matrix or diagonal) linear recurrence. `make_gru` normalises
the recurrent weights to a fixed spectral norm so stability does not depend on H.
"""

from __future__ import annotations

import torch

DT = torch.float64


def make_gru(H: int, D: int, w_scale: float = 0.4, u_specnorm: float = 0.9, seed: int = 0) -> dict:
    tg = torch.Generator().manual_seed(seed)
    g = {k: torch.randn(*shp, generator=tg, dtype=DT) * w_scale
         for k, shp in {"Wr": (H, D), "Wz": (H, D), "Wn": (H, D)}.items()}
    for k in ("Ur", "Uz", "Un"):                       # fixed spectral norm vs H
        M = torch.randn(H, H, generator=tg, dtype=DT)
        g[k] = M / torch.linalg.matrix_norm(M, ord=2) * u_specnorm
    g |= {b: torch.zeros(H, dtype=DT) for b in ("br", "bz", "bn")}
    return g


def gru_step(g: dict, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r = torch.sigmoid(g["Wr"] @ x + g["Ur"] @ h + g["br"])
    z = torch.sigmoid(g["Wz"] @ x + g["Uz"] @ h + g["bz"])
    n = torch.tanh(g["Wn"] @ x + g["Un"] @ (r * h) + g["bn"])
    return (1 - z) * n + z * h


def gru_sequential(g: dict, xs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    h, out = h0.clone(), []
    for t in range(len(xs)):
        h = gru_step(g, h, xs[t]); out.append(h.clone())
    return torch.stack(out)


def deer_gru(g, xs, h0, diagonal: bool, max_iters: int = 25, tol: float = 1e-8) -> tuple[int, float]:
    """Parallel Newton on a GRU. diagonal=True keeps only diag(J) (O(H) scan).

    Returns (iterations_used, final_max_error_vs_sequential).
    """
    T, H = len(xs), h0.shape[0]
    ref = gru_sequential(g, xs, h0)
    hstar, hist = torch.zeros(T, H, dtype=DT), []
    for _ in range(max_iters):
        A, bvec = [], []
        for t in range(T):
            hp = (h0 if t == 0 else hstar[t - 1]).detach()
            J = torch.autograd.functional.jacobian(lambda hh: gru_step(g, hh, xs[t]), hp)
            f = gru_step(g, hp, xs[t])
            if diagonal:
                Jd = torch.diagonal(J); A.append(Jd); bvec.append(f - Jd * hp)
            else:
                A.append(J); bvec.append(f - J @ hp)
        h, newh = h0.clone(), []
        for t in range(T):
            h = (A[t] * h if diagonal else A[t] @ h) + bvec[t]; newh.append(h.clone())
        hstar = torch.stack(newh)
        hist.append((hstar - ref).abs().max().item())
        if hist[-1] < tol:
            break
    return len(hist), hist[-1]


__all__ = ["make_gru", "gru_step", "gru_sequential", "deer_gru"]
