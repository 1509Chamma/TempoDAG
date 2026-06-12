from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    histogram: np.ndarray
    bin_edges: np.ndarray
    n_samples: int = 0
    n_nan: int = 0
    p01: float = 0.0
    p99: float = 0.0


def compute_stats(
    samples: Iterable[np.ndarray],
    n_bins: int = 50,
) -> DatasetStats:
    """Compute distributional statistics via a single streaming pass."""
    n = 0
    n_nan = 0
    m1 = 0.0
    m2 = 0.0
    global_min = np.inf
    global_max = -np.inf
    all_values: list[np.ndarray] = []

    for arr in samples:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        nan_mask = np.isnan(flat)
        n_nan += int(nan_mask.sum())
        flat = flat[~nan_mask]
        if flat.size == 0:
            continue

        all_values.append(flat)
        count = flat.size
        batch_mean = float(flat.mean())
        batch_var = float(flat.var())

        delta = batch_mean - m1
        new_n = n + count
        m1 = m1 + delta * count / new_n
        m2 = m2 + batch_var * count + delta**2 * n * count / new_n
        n = new_n

        local_min = float(flat.min())
        local_max = float(flat.max())
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max

    if n == 0:
        return DatasetStats(
            mean=0.0,
            std=0.0,
            minimum=0.0,
            maximum=0.0,
            histogram=np.zeros(n_bins, dtype=np.float64),
            bin_edges=np.linspace(0, 1, n_bins + 1),
            n_samples=0,
            n_nan=n_nan,
        )

    all_flat = np.concatenate(all_values)
    hist, bin_edges = np.histogram(
        all_flat,
        bins=n_bins,
        range=(global_min, global_max),
    )
    hist = hist / hist.sum() if hist.sum() > 0 else hist.astype(np.float64)

    return DatasetStats(
        mean=m1,
        std=float(np.sqrt(m2 / n)),
        minimum=global_min,
        maximum=global_max,
        histogram=hist.astype(np.float64),
        bin_edges=bin_edges,
        n_samples=n,
        n_nan=n_nan,
        p01=float(np.percentile(all_flat, 1)),
        p99=float(np.percentile(all_flat, 99)),
    )


def compare_stats(full: DatasetStats, representative: DatasetStats) -> dict[str, float]:
    """Return similarity metrics between two DatasetStats instances."""
    overlap_lo = max(full.minimum, representative.minimum)
    overlap_hi = min(full.maximum, representative.maximum)
    full_range = max(full.maximum - full.minimum, 1e-12)
    range_overlap = max(0.0, (overlap_hi - overlap_lo) / full_range)

    rep_hist = _rebin(representative, full.bin_edges)
    hist_intersect = float(np.minimum(full.histogram, rep_hist).sum())

    return {
        "mean_diff": abs(full.mean - representative.mean),
        "std_diff": abs(full.std - representative.std),
        "range_overlap": range_overlap,
        "histogram_intersect": hist_intersect,
        "kl_divergence": kl_divergence(full, representative),
        "p01_diff": abs(full.p01 - representative.p01),
        "p99_diff": abs(full.p99 - representative.p99),
    }


def kl_divergence(p: DatasetStats, q: DatasetStats) -> float:
    """KL divergence KL(P || Q) between two histogram distributions."""
    p_hist = p.histogram.copy() + 1e-10
    q_hist = _rebin(q, p.bin_edges) + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def _rebin(stats: DatasetStats, target_bin_edges: np.ndarray) -> np.ndarray:
    n_target = len(target_bin_edges) - 1
    midpoints = 0.5 * (stats.bin_edges[:-1] + stats.bin_edges[1:])
    weights = stats.histogram
    mask = weights > 0
    if not mask.any():
        return np.zeros(n_target, dtype=np.float64)

    rebinned, _ = np.histogram(
        midpoints[mask],
        bins=target_bin_edges,
        weights=weights[mask],
    )
    total = rebinned.sum()
    if total > 0:
        return (rebinned / total).astype(np.float64)
    return rebinned.astype(np.float64)
