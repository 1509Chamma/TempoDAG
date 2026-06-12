"""
edge_lstm.calibration
====================

Representative dataset generation utilities for quantization calibration.

Public API
----------
create_representative_dataset - main entry point
DatasetStats - distributional snapshot
compute_stats - compute stats from a sample stream
compare_stats - diff two DatasetStats instances
kl_divergence - KL divergence between two distributions
"""

from .dataset import create_representative_dataset
from .stats import DatasetStats, compare_stats, compute_stats, kl_divergence
from .strategies import (
    RegimeAwareSampler,
    SamplingStrategy,
    StratifiedTemporalSampler,
    TailAwareSampler,
    UniformSampler,
    apply_tail_pass,
    get_strategy,
)

__all__ = [
    "create_representative_dataset",
    "DatasetStats",
    "compute_stats",
    "compare_stats",
    "kl_divergence",
    "SamplingStrategy",
    "UniformSampler",
    "StratifiedTemporalSampler",
    "RegimeAwareSampler",
    "TailAwareSampler",
    "apply_tail_pass",
    "get_strategy",
]
