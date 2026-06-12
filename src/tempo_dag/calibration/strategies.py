from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np


class SamplingStrategy(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def sample(
        self,
        stream: Iterable[np.ndarray],
        max_samples: int,
    ) -> list[np.ndarray]: ...


class UniformSampler(SamplingStrategy):
    """Reservoir sampling across the full dataset (Algorithm R)."""

    def sample(
        self,
        stream: Iterable[np.ndarray],
        max_samples: int,
    ) -> list[np.ndarray]:
        seed = self.config.get("seed", 42)
        return _reservoir_sample(stream, max_samples, seed=seed)


class StratifiedTemporalSampler(SamplingStrategy):
    """
    Split a dataset into time segments and sample proportionally from each one.

    Config keys:
        num_segments: Number of temporal segments to cycle through.
        samples_per_segment: Optional fixed quota per segment.
        seed: Random seed for segment sampling.
    """

    def sample(
        self,
        stream: Iterable[np.ndarray],
        max_samples: int,
    ) -> list[np.ndarray]:
        num_segments = int(self.config.get("num_segments", 10))
        samples_per_segment: int | None = self.config.get("samples_per_segment")
        rng = random.Random(int(self.config.get("seed", 42)))

        segments: list[list[np.ndarray]] = [[] for _ in range(num_segments)]
        segment_counts = [0] * num_segments
        global_idx = 0

        for arr in stream:
            seg_idx = global_idx % num_segments
            segment_counts[seg_idx] += 1
            count = segment_counts[seg_idx]
            if count <= max_samples:
                segments[seg_idx].append(arr)
            else:
                j = rng.randint(0, count - 1)
                if j < max_samples:
                    segments[seg_idx][j] = arr
            global_idx += 1

        if global_idx == 0:
            return []

        result: list[np.ndarray] = []
        for seg_idx, segment in enumerate(segments):
            if not segment:
                continue

            if samples_per_segment is not None:
                quota = samples_per_segment
            else:
                proportion = segment_counts[seg_idx] / global_idx
                quota = max(1, round(proportion * max_samples))

            result.extend(rng.sample(segment, min(quota, len(segment))))

        if len(result) > max_samples:
            rng.shuffle(result)
            result = result[:max_samples]
        return result


class RegimeAwareSampler(SamplingStrategy):
    """
    Cluster samples by simple distributional features, then sample per cluster.

    Config keys:
        n_clusters: Number of K-means clusters to fit.
        kmeans_iters: Number of K-means refinement iterations.
        seed: Random seed for clustering and sampling.
    """

    def sample(
        self,
        stream: Iterable[np.ndarray],
        max_samples: int,
    ) -> list[np.ndarray]:
        n_clusters = int(self.config.get("n_clusters", 8))
        kmeans_iters = int(self.config.get("kmeans_iters", 5))
        rng = np.random.default_rng(int(self.config.get("seed", 42)))

        items = list(stream)
        if not items:
            return []
        if len(items) <= n_clusters:
            return items[:max_samples]

        features = _standardize(np.stack([_extract_features(item) for item in items]))
        labels = _kmeans(
            features,
            n_clusters=n_clusters,
            n_iters=kmeans_iters,
            rng=rng,
        )

        samples_per_cluster = max(1, max_samples // n_clusters)
        result: list[np.ndarray] = []
        for cluster_idx in range(n_clusters):
            indices = np.where(labels == cluster_idx)[0]
            if len(indices) == 0:
                continue

            chosen = rng.choice(
                indices,
                size=min(samples_per_cluster, len(indices)),
                replace=False,
            )
            result.extend(items[idx] for idx in chosen)

        if len(result) < max_samples:
            selected_ids = {id(item) for item in result}
            extras = [item for item in items if id(item) not in selected_ids]
            rng.shuffle(extras)
            result.extend(extras[: max_samples - len(result)])

        return result[:max_samples]


class TailAwareSampler(SamplingStrategy):
    """
    Force-include top and bottom percentile samples, then fill uniformly.

    Config keys:
        tail_percentile: Percentile used to define the tail band.
        seed: Random seed for fill ordering.
    """

    def sample(
        self,
        stream: Iterable[np.ndarray],
        max_samples: int,
    ) -> list[np.ndarray]:
        tail_pct = float(self.config.get("tail_percentile", 0.99))
        rng = random.Random(int(self.config.get("seed", 42)))
        return _apply_tail_pass(list(stream), max_samples, tail_pct=tail_pct, rng=rng)


def apply_tail_pass(
    items: list[np.ndarray],
    max_samples: int,
    tail_percentile: float = 0.99,
    seed: int = 42,
) -> list[np.ndarray]:
    """Force-include tail samples, then fill uniformly."""
    return _apply_tail_pass(
        items,
        max_samples,
        tail_pct=tail_percentile,
        rng=random.Random(seed),
    )


STRATEGY_REGISTRY: dict[str, type[SamplingStrategy]] = {
    "uniform": UniformSampler,
    "stratified_temporal": StratifiedTemporalSampler,
    "regime_aware": RegimeAwareSampler,
    "tail_aware": TailAwareSampler,
}


def get_strategy(config: dict[str, Any]) -> SamplingStrategy:
    method = config.get("method", "uniform")
    cls = STRATEGY_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown sampling method '{method}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return cls(config)


def _reservoir_sample(
    stream: Iterable[np.ndarray],
    k: int,
    seed: int = 42,
) -> list[np.ndarray]:
    rng = random.Random(seed)
    reservoir: list[np.ndarray] = []
    for idx, item in enumerate(stream):
        if idx < k:
            reservoir.append(item)
            continue

        j = rng.randint(0, idx)
        if j < k:
            reservoir[j] = item
    return reservoir


def _extract_features(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    flat = np.where(np.isnan(flat), 0.0, flat)
    if flat.size == 0:
        return np.zeros(3)
    return np.array([flat.mean(), flat.std(), float(flat.max() - flat.min())])


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = np.where(features.std(axis=0) < 1e-10, 1.0, features.std(axis=0))
    return (features - mean) / std


def _kmeans(
    features: np.ndarray,
    n_clusters: int,
    n_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = features.shape[0]
    k = min(n_clusters, n)

    centroid_indices = [int(rng.integers(0, n))]
    for _ in range(k - 1):
        distances = np.array(
            [
                min(
                    np.linalg.norm(features[row_idx] - features[centroid_idx]) ** 2
                    for centroid_idx in centroid_indices
                )
                for row_idx in range(n)
            ],
            dtype=np.float64,
        )

        distance_sum = float(distances.sum())
        if not np.isfinite(distance_sum) or distance_sum <= 0.0:
            remaining = [
                row_idx for row_idx in range(n) if row_idx not in centroid_indices
            ]
            if not remaining:
                break
            centroid_indices.append(int(rng.choice(remaining)))
            continue

        cumulative = np.cumsum(distances / distance_sum)
        centroid_indices.append(int(np.searchsorted(cumulative, float(rng.random()))))

    centroids = features[centroid_indices]
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(n_iters):
        distances = np.linalg.norm(
            features[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        labels = np.argmin(distances, axis=1)
        centroids = np.array(
            [
                features[labels == cluster_idx].mean(axis=0)
                if (labels == cluster_idx).any()
                else centroids[cluster_idx]
                for cluster_idx in range(k)
            ]
        )

    return labels


def _scalar_repr(arr: np.ndarray) -> float:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    flat = flat[~np.isnan(flat)]
    return float(np.abs(flat).max()) if flat.size > 0 else 0.0


def _apply_tail_pass(
    items: list[np.ndarray],
    max_samples: int,
    tail_pct: float,
    rng: random.Random,
) -> list[np.ndarray]:
    if not items:
        return []

    n = len(items)
    n_tail = max(1, int(math.ceil(n * (1.0 - tail_pct))))
    ranked = sorted(range(n), key=lambda idx: _scalar_repr(items[idx]))
    tail_idx = sorted(set(ranked[:n_tail]) | set(ranked[-n_tail:]))

    tail_samples = [items[idx] for idx in tail_idx]
    non_tail = [items[idx] for idx in range(n) if idx not in tail_idx]
    rng.shuffle(non_tail)

    result = tail_samples + non_tail[: max(0, max_samples - len(tail_samples))]
    rng.shuffle(result)
    return result[:max_samples]
