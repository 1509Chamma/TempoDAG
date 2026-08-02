"""Degenerate-input branches in the calibration package.

Targets the last uncovered lines in calibration/dataset.py, stats.py, and
strategies.py: all-NaN inputs, empty streams, and histograms with no overlap.
"""

from __future__ import annotations

import numpy as np

from tempo_dag.calibration.dataset import _tail_score
from tempo_dag.calibration.stats import (
    DatasetStats,
    _rebin,
    compute_stats,
    kl_divergence,
)
from tempo_dag.calibration.strategies import (
    RegimeAwareSampler,
    StratifiedTemporalSampler,
    _extract_features,
    apply_tail_pass,
)


def _stats(histogram, bin_edges) -> DatasetStats:
    return DatasetStats(
        mean=0.0,
        std=1.0,
        minimum=float(bin_edges[0]),
        maximum=float(bin_edges[-1]),
        histogram=np.asarray(histogram, dtype=np.float64),
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        n_samples=1,
    )


class TestStats:
    def test_compute_stats_skips_all_nan_batch(self):
        stats = compute_stats([np.array([np.nan, np.nan]), np.array([1.0, 3.0])])
        assert stats.n_samples == 2
        assert stats.n_nan == 2
        assert stats.minimum == 1.0
        assert stats.maximum == 3.0

    def test_rebin_all_zero_histogram_returns_zeros(self):
        empty = _stats([0.0, 0.0], [0.0, 0.5, 1.0])
        target = np.linspace(0.0, 1.0, 4)
        assert _rebin(empty, target).tolist() == [0.0, 0.0, 0.0]

    def test_rebin_disjoint_ranges_returns_unnormalized_zeros(self):
        # The only weighted midpoint (0.5) falls outside the target edges,
        # so the rebinned mass is zero and normalization is skipped.
        source = _stats([1.0, 0.0], [0.0, 1.0, 2.0])
        rebinned = _rebin(source, np.array([10.0, 11.0]))
        assert rebinned.tolist() == [0.0]

    def test_kl_divergence_against_empty_histogram_is_finite(self):
        p = _stats([0.5, 0.5], [0.0, 0.5, 1.0])
        q = _stats([0.0, 0.0], [0.0, 0.5, 1.0])
        assert np.isfinite(kl_divergence(p, q))


class TestDataset:
    def test_tail_score_of_all_nan_array_is_zero(self):
        assert _tail_score(np.array([np.nan, np.nan])) == 0.0


class TestStrategies:
    def test_stratified_empty_stream_returns_empty(self):
        sampler = StratifiedTemporalSampler({"num_segments": 4})
        assert sampler.sample([], max_samples=8) == []

    def test_stratified_skips_empty_segments(self):
        # 3 items round-robin into 5 segments leaves segments 3 and 4 empty.
        items = [np.full((2,), float(i)) for i in range(3)]
        sampler = StratifiedTemporalSampler({"num_segments": 5, "seed": 0})
        result = sampler.sample(items, max_samples=8)
        assert len(result) == 3

    def test_stratified_fixed_quota_per_segment(self):
        items = [np.full((2,), float(i)) for i in range(6)]
        sampler = StratifiedTemporalSampler(
            {"num_segments": 2, "samples_per_segment": 1, "seed": 0}
        )
        result = sampler.sample(items, max_samples=8)
        assert len(result) == 2

    def test_regime_empty_stream_returns_empty(self):
        assert RegimeAwareSampler({}).sample([], max_samples=4) == []

    def test_regime_fewer_items_than_clusters_short_circuits(self):
        items = [np.full((2,), float(i)) for i in range(3)]
        result = RegimeAwareSampler({"n_clusters": 8}).sample(items, max_samples=2)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], items[0])

    def test_apply_tail_pass_public_wrapper(self):
        items = [np.full((2,), float(i)) for i in range(10)]
        result = apply_tail_pass(items, max_samples=4, tail_percentile=0.9, seed=1)
        assert len(result) == 4
        # The extreme sample must survive the tail pass.
        assert any(float(np.abs(arr).max()) == 9.0 for arr in result)

    def test_apply_tail_pass_empty_items(self):
        assert apply_tail_pass([], max_samples=4) == []

    def test_extract_features_of_empty_array_is_zero_vector(self):
        np.testing.assert_array_equal(_extract_features(np.array([])), np.zeros(3))
