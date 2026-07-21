"""Validated recurrence primitives shared across notebooks.

These are the exact methods derived and stress-tested in NB01
(01_breaking_sequential_dependencies): the linear parallel scan, and the two
iterative parallel solvers for nonlinear recurrences (Picard, DEER). Keeping one
copy means every notebook and every optimizer experiment scores against the same
implementations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def affine_combine(acc: tuple, step: tuple) -> tuple:
    """Compose affine maps: (A,B) then (a,b) => (a*A, a*B + b). Associative."""
    A, B = acc
    a, b = step
    return (a * A, a * B + b)


def sequential_linear(a: np.ndarray, b: np.ndarray, h0: float = 0.0) -> np.ndarray:
    """Baseline O(T)-depth sequential evaluation of h_t = a_t h_{t-1} + b_t."""
    h, out = h0, np.empty_like(b, dtype=np.float64)
    for t in range(len(b)):
        h = a[t] * h + b[t]
        out[t] = h
    return out


def scan_hillis_steele(a: np.ndarray, b: np.ndarray, h0: float = 0.0) -> tuple[np.ndarray, int]:
    """Log-depth parallel prefix scan of affine maps. Returns (h, num_rounds)."""
    A = np.asarray(a, dtype=np.float64).copy()
    B = np.asarray(b, dtype=np.float64).copy()
    n, d, rounds = len(A), 1, 0
    while d < n:
        A_p, B_p, i = A.copy(), B.copy(), np.arange(d, n)
        A[i] = A_p[i] * A_p[i - d]
        B[i] = A_p[i] * B_p[i - d] + B_p[i]
        d *= 2
        rounds += 1
    return A * h0 + B, rounds


def sequential_nonlinear(
    w: float, c: np.ndarray, h0: float = 0.0, act: Callable = np.tanh
) -> np.ndarray:
    """Ground-truth sequential nonlinear recurrence h_t = act(w h_{t-1} + c_t)."""
    h, out = h0, np.empty_like(c, dtype=np.float64)
    for t in range(len(c)):
        h = float(act(w * h + c[t]))
        out[t] = h
    return out


def picard(
    w: float, c: np.ndarray, h0: float = 0.0, max_iters: int = 500,
    tol: float = 1e-12, act: Callable = np.tanh,
) -> tuple[np.ndarray, list[float]]:
    """Parallel fixed-point iteration. Depth O(1)/iter; converges if contractive."""
    ref = sequential_nonlinear(w, c, h0, act)
    h, hist = np.zeros_like(c), []
    for _ in range(max_iters):
        h_prev = np.concatenate(([h0], h[:-1]))
        h_new = act(w * h_prev + c)
        hist.append(float(np.max(np.abs(h_new - ref))))
        step = float(np.max(np.abs(h_new - h)))
        h = h_new
        if step < tol:
            break
    return h, hist


def deer(
    w: float, c: np.ndarray, h0: float = 0.0, max_iters: int = 60,
    tol: float = 1e-12, act: Callable = np.tanh, dact: Callable | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Parallel Newton: each iter solves a linear recurrence via the scan."""
    if dact is None:
        def dact(pre):
            return 1.0 - np.tanh(pre) ** 2
    ref = sequential_nonlinear(w, c, h0, act)
    h, hist = np.zeros_like(c), []
    for _ in range(max_iters):
        h_prev = np.concatenate(([h0], h[:-1]))
        pre = w * h_prev + c
        J = w * dact(pre)
        beta = act(pre) - J * h_prev
        h = scan_hillis_steele(J, beta, h0=h0)[0]
        hist.append(float(np.max(np.abs(h - ref))))
        if hist[-1] < tol:
            break
    return h, hist


def quantize(x: np.ndarray, frac_bits: int = 8, int_bits: int = 9) -> np.ndarray:
    """Round-to-nearest, saturating fixed-point (format Q<int_bits>.<frac_bits>)."""
    s = 2.0**frac_bits
    lo, hi = -(2.0**int_bits), 2.0**int_bits - 2.0**-frac_bits
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * s) / s, lo, hi)


def _tree_fixed(a, b, h0, q):
    A, B, n, d = q(a).copy(), q(b).copy(), len(a), 1
    while d < n:
        A_p, B_p, i = A.copy(), B.copy(), np.arange(d, n)
        A[i] = q(A_p[i] * A_p[i - d])
        B[i] = q(A_p[i] * B_p[i - d] + B_p[i])
        d *= 2
    return q(A * h0 + B)


def picard_fixed(w, c, h0=0.0, frac_bits=8, int_bits=9, max_iters=300):
    """Fixed-point Picard. Returns (h, error-vs-float-ground-truth by iteration)."""
    def q(x):
        return quantize(x, frac_bits, int_bits)
    ref = sequential_nonlinear(w, c, h0)
    h, hist = np.zeros_like(c), []
    for _ in range(max_iters):
        h_prev = np.concatenate(([h0], h[:-1]))
        h_new = q(np.tanh(q(w * h_prev + c)))
        hist.append(float(np.max(np.abs(h_new - ref))))
        if np.array_equal(h_new, h):      # exact quantized fixed point
            break
        h = h_new
    return h, hist


def deer_fixed(w, c, h0=0.0, frac_bits=8, int_bits=9, max_iters=40):
    """Fixed-point DEER (quantised ops + rounded tree scan). Returns (h, hist)."""
    def q(x):
        return quantize(x, frac_bits, int_bits)
    ref = sequential_nonlinear(w, c, h0)
    h, hist = np.zeros_like(c), []
    for _ in range(max_iters):
        h_prev = np.concatenate(([h0], h[:-1]))
        pre = q(w * h_prev + c)
        th = q(np.tanh(pre))
        J = q(w * (1.0 - q(th * th)))
        beta = q(th - q(J * h_prev))
        h_new = _tree_fixed(J, beta, h0, q)
        hist.append(float(np.max(np.abs(h_new - ref))))
        if np.array_equal(h_new, h):
            break
        h = h_new
    return h, hist


__all__ = [
    "affine_combine", "sequential_linear", "scan_hillis_steele",
    "sequential_nonlinear", "picard", "deer",
    "quantize", "picard_fixed", "deer_fixed",
]
