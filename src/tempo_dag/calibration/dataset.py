from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .strategies import get_strategy


def create_representative_dataset(
    dataset: Iterable[np.ndarray],
    max_samples: int,
    strategy_config: dict[str, Any] | None = None,
) -> Iterator[np.ndarray]:
    """
    Build a statistically representative subset of *dataset* for calibration.

    Yields one sample at a time - compatible with calibration runners that
    consume an iterable of model inputs.

    Args:
        dataset: Iterable of numpy arrays (any shape per element).
        max_samples: Hard upper bound on returned samples.
        strategy_config: Sampling behaviour. Key ``"method"`` selects the
            strategy (``"uniform"``, ``"stratified_temporal"``,
            ``"regime_aware"``, ``"tail_aware"``). Shared optional keys:
            ``include_tails`` (bool), ``tail_percentile`` (float, default 0.99),
            ``seed`` (int, default 42).

    Yields:
        numpy arrays (same shape as input elements).

    Raises:
        ValueError: Unknown method or max_samples < 1.
    """
    if strategy_config is None:
        strategy_config = {"method": "uniform"}

    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    include_tails = bool(strategy_config.get("include_tails", False))
    tail_pct = float(strategy_config.get("tail_percentile", 0.99))

    # Materialise and strip all-NaN entries up front.
    # Required anyway when include_tails=True (we need two passes over the data).
    items = [arr for arr in dataset if _valid_values(arr).size > 0]

    if not items:
        warnings.warn(
            "Representative dataset is empty - input dataset may be empty.",
            stacklevel=2,
        )
        return

    if include_tails and strategy_config.get("method") != "tail_aware":
        # Separate tail items before the strategy sees the data so the extremes
        # are guaranteed in the output regardless of the fill strategy.
        tail_items, non_tail_items = _split_tails(items, tail_pct)
        forced_tails = tail_items[: min(len(tail_items), max_samples)]
        remaining_budget = max(0, max_samples - len(forced_tails))

        strategy = get_strategy(strategy_config)
        fill = strategy.sample(iter(non_tail_items), remaining_budget)
        selected = forced_tails + fill
    else:
        strategy = get_strategy(strategy_config)
        selected = strategy.sample(iter(items), max_samples)

    if len(selected) < max_samples:
        warnings.warn(
            f"Dataset has only {len(selected)} valid samples "
            f"(requested {max_samples}). Returning all available.",
            stacklevel=2,
        )

    yield from selected


def _valid_values(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return flat[~np.isnan(flat)]


def _tail_score(arr: np.ndarray) -> float:
    valid = _valid_values(arr)
    if valid.size == 0:
        return 0.0
    return float(np.abs(valid).max())


def _split_tails(
    items: list[np.ndarray],
    tail_pct: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(tail_items, non_tail_items)`` by absolute-max scalar rank."""
    import math

    n = len(items)
    n_tail = max(1, int(math.ceil(n * (1.0 - tail_pct))))
    ranked = sorted(range(n), key=lambda idx: _tail_score(items[idx]))

    tail_idx = sorted(set(ranked[:n_tail]) | set(ranked[-n_tail:]))
    tail_items = [items[idx] for idx in tail_idx]
    non_tail_items = [items[idx] for idx in range(n) if idx not in tail_idx]
    return tail_items, non_tail_items
