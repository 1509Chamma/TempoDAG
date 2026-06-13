from __future__ import annotations

import warnings
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from tempo_dag.calibration import (
    compare_stats,
    compute_stats,
    create_representative_dataset,
    get_strategy,
    kl_divergence,
)
from tempo_dag.quantization_config import (
    QuantizationScheme,
    QuantizationSpec,
    compute_quant_params,
)


def make_normal_dataset(n: int = 2000, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(16).astype(np.float32) for _ in range(n)]


def make_skewed_dataset(n: int = 2000, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.exponential(scale=2.0, size=16).astype(np.float32) for _ in range(n)]


def make_regime_dataset(n: int = 2000, seed: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    low = [rng.normal(0, 0.5, 16).astype(np.float32) for _ in range(n // 3)]
    high = [rng.normal(0, 3.0, 16).astype(np.float32) for _ in range(n // 3)]
    tail = [
        rng.uniform(-20, 20, 16).astype(np.float32) for _ in range(n - 2 * (n // 3))
    ]
    combined = low + high + tail
    rng.shuffle(combined)
    return combined


def chunked(
    items: list[np.ndarray],
    chunk_size: int,
) -> Generator[np.ndarray, None, None]:
    for start in range(0, len(items), chunk_size):
        yield from items[start : start + chunk_size]


def hist_intersection(a: Any, b: Any) -> float:
    return compare_stats(a, b)["histogram_intersect"]


STRATEGY_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("uniform", {"method": "uniform", "seed": 0}),
    (
        "stratified_temporal",
        {"method": "stratified_temporal", "num_segments": 10, "seed": 0},
    ),
    ("regime_aware", {"method": "regime_aware", "n_clusters": 5, "seed": 0}),
    ("tail_aware", {"method": "tail_aware", "tail_percentile": 0.95, "seed": 0}),
]
STRATEGY_IDS = [name for name, _ in STRATEGY_CONFIGS]


@pytest.mark.parametrize(
    ("name", "config"),
    STRATEGY_CONFIGS,
    ids=STRATEGY_IDS,
)
def test_strategy_returns_correct_count(name: str, config: dict[str, Any]) -> None:
    dataset = make_normal_dataset(500)
    max_samples = 100
    result = list(create_representative_dataset(dataset, max_samples, config))
    assert len(result) <= max_samples
    assert len(result) > 0


@pytest.mark.parametrize(
    ("name", "config"),
    STRATEGY_CONFIGS,
    ids=STRATEGY_IDS,
)
def test_strategy_output_elements_are_numpy_arrays(
    name: str,
    config: dict[str, Any],
) -> None:
    dataset = make_normal_dataset(200)
    result = list(create_representative_dataset(dataset, 50, config))
    for item in result:
        assert isinstance(item, np.ndarray)


@pytest.mark.parametrize(
    ("name", "config"),
    STRATEGY_CONFIGS,
    ids=STRATEGY_IDS,
)
def test_strategy_is_reproducible(name: str, config: dict[str, Any]) -> None:
    dataset = make_normal_dataset(300, seed=99)
    result_1 = [a.tobytes() for a in create_representative_dataset(dataset, 60, config)]
    result_2 = [a.tobytes() for a in create_representative_dataset(dataset, 60, config)]
    assert result_1 == result_2


@pytest.mark.parametrize(
    ("name", "config"),
    STRATEGY_CONFIGS,
    ids=STRATEGY_IDS,
)
def test_distribution_preservation_normal(
    name: str,
    config: dict[str, Any],
) -> None:
    dataset = make_normal_dataset(2000)
    max_samples = 200
    representative = list(create_representative_dataset(dataset, max_samples, config))

    full_stats = compute_stats(iter(dataset))
    rep_stats = compute_stats(iter(representative))

    intersection = hist_intersection(full_stats, rep_stats)
    assert intersection >= 0.75, (
        f"[{name}] histogram intersection {intersection:.3f} < 0.75 - "
        "representative dataset does not preserve distribution"
    )


@pytest.mark.parametrize(
    ("name", "config"),
    STRATEGY_CONFIGS,
    ids=STRATEGY_IDS,
)
def test_mean_preservation(name: str, config: dict[str, Any]) -> None:
    dataset = make_normal_dataset(2000)
    representative = list(create_representative_dataset(dataset, 200, config))

    full_stats = compute_stats(iter(dataset))
    rep_stats = compute_stats(iter(representative))

    tolerance = 0.20 * full_stats.std
    assert abs(full_stats.mean - rep_stats.mean) <= tolerance, (
        f"[{name}] mean drift {abs(full_stats.mean - rep_stats.mean):.4f} "
        f"> tolerance {tolerance:.4f}"
    )


def test_std_preservation_stratified() -> None:
    dataset = make_normal_dataset(2000)
    config = {"method": "stratified_temporal", "num_segments": 20, "seed": 7}
    representative = list(create_representative_dataset(dataset, 300, config))
    full_stats = compute_stats(iter(dataset))
    rep_stats = compute_stats(iter(representative))
    assert abs(full_stats.std - rep_stats.std) / full_stats.std < 0.30


def test_tail_aware_includes_extremes() -> None:
    rng = np.random.default_rng(42)
    main = [rng.standard_normal(16).astype(np.float32) for _ in range(480)]
    tail_lo = [np.full(16, -19.0, dtype=np.float32)]
    tail_hi = [np.full(16, +19.0, dtype=np.float32)]
    dataset = main + tail_lo + tail_hi

    config = {"method": "tail_aware", "tail_percentile": 0.98, "seed": 0}
    representative = list(create_representative_dataset(dataset, 100, config))

    flat_rep = np.concatenate([a.ravel() for a in representative])
    assert flat_rep.min() <= -15.0, "Tail-aware sampler missed lower extreme"
    assert flat_rep.max() >= 15.0, "Tail-aware sampler missed upper extreme"


def test_include_tails_flag_composable_with_uniform() -> None:
    rng = np.random.default_rng(7)
    main = [rng.standard_normal(16).astype(np.float32) for _ in range(480)]
    tail_lo = [np.full(16, -50.0, dtype=np.float32)]
    tail_hi = [np.full(16, +50.0, dtype=np.float32)]
    dataset = main + tail_lo + tail_hi

    config = {
        "method": "uniform",
        "include_tails": True,
        "tail_percentile": 0.99,
        "seed": 0,
    }
    representative = list(create_representative_dataset(dataset, 100, config))

    flat_rep = np.concatenate([a.ravel() for a in representative])
    assert flat_rep.min() <= -30.0, "include_tails did not pull in lower extreme"
    assert flat_rep.max() >= 30.0, "include_tails did not pull in upper extreme"


def test_regime_aware_covers_multiple_clusters() -> None:
    dataset = make_regime_dataset(1500)
    config = {"method": "regime_aware", "n_clusters": 6, "seed": 0}
    representative = list(create_representative_dataset(dataset, 150, config))
    full_stats = compute_stats(iter(dataset))
    rep_stats = compute_stats(iter(representative))

    metrics = compare_stats(full_stats, rep_stats)
    assert metrics["range_overlap"] >= 0.80, (
        f"Range overlap {metrics['range_overlap']:.3f} too low - regime-aware "
        "sampler is not covering the full dynamic range"
    )


def test_calibration_scale_stability() -> None:
    dataset = make_normal_dataset(3000)
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.SYMMETRIC)

    full_data = np.concatenate([a.ravel() for a in dataset]).tolist()
    full_scale, _ = compute_quant_params(full_data, spec)

    # include_tails ensures extremes are captured; that matters for scale stability
    config = {
        "method": "stratified_temporal",
        "num_segments": 15,
        "include_tails": True,
        "tail_percentile": 0.99,
        "seed": 0,
    }
    representative = list(create_representative_dataset(dataset, 300, config))
    rep_data = np.concatenate([a.ravel() for a in representative]).tolist()
    rep_scale, _ = compute_quant_params(rep_data, spec)

    relative_error = abs(full_scale - rep_scale) / max(abs(full_scale), 1e-12)
    assert relative_error < 0.10, (
        f"Calibration scale drift {relative_error:.2%} > 10% - "
        "representative dataset does not give stable quantization parameters"
    )


def test_streaming_input_uniform() -> None:
    dataset = make_normal_dataset(500)
    config = {"method": "uniform", "seed": 42}
    from_list = list(create_representative_dataset(dataset, 100, config))
    from_gen = list(create_representative_dataset(chunked(dataset, 32), 100, config))
    assert len(from_list) == len(from_gen)


def test_streaming_input_stratified() -> None:
    dataset = make_normal_dataset(600)
    config = {"method": "stratified_temporal", "num_segments": 6, "seed": 5}
    result = list(create_representative_dataset(chunked(dataset, 50), 100, config))
    assert 0 < len(result) <= 100


def test_tiny_dataset_smaller_than_max_samples() -> None:
    dataset = [np.array([float(i)]) for i in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = list(create_representative_dataset(dataset, 100))
    assert len(result) == 5


def test_single_element_dataset() -> None:
    dataset = [np.array([3.14])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = list(create_representative_dataset(dataset, 50))
    assert len(result) == 1
    assert float(result[0][0]) == pytest.approx(3.14)


def test_all_nan_dataset_yields_empty() -> None:
    dataset = [np.array([np.nan, np.nan]) for _ in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = list(create_representative_dataset(dataset, 10))
    assert result == []


def test_partial_nan_values_pass_through() -> None:
    rng = np.random.default_rng(0)
    dataset: list[np.ndarray] = []
    for _ in range(50):
        arr = rng.standard_normal(8).astype(np.float32)
        arr[0] = np.nan
        dataset.append(arr)

    result = list(create_representative_dataset(dataset, 20))
    assert len(result) > 0
    for arr in result:
        assert not np.all(np.isnan(arr))


def test_constant_dataset() -> None:
    dataset = [np.full(8, 7.0, dtype=np.float32) for _ in range(100)]
    for name, config in STRATEGY_CONFIGS:
        result = list(create_representative_dataset(dataset, 20, config))
        assert len(result) > 0, f"[{name}] returned empty on constant dataset"


def test_highly_skewed_distribution() -> None:
    dataset = make_skewed_dataset(1000)
    config = {"method": "tail_aware", "tail_percentile": 0.99, "seed": 3}
    result = list(create_representative_dataset(dataset, 100, config))
    assert len(result) > 0


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown sampling method"):
        get_strategy({"method": "nonexistent_method"})


def test_max_samples_less_than_one_raises() -> None:
    dataset = make_normal_dataset(100)
    with pytest.raises(ValueError, match="max_samples must be >= 1"):
        list(create_representative_dataset(dataset, 0))


def test_compute_stats_basic() -> None:
    data = [np.array([0.0, 1.0, 2.0, 3.0, 4.0])]
    stats = compute_stats(iter(data))
    assert stats.mean == pytest.approx(2.0)
    assert stats.minimum == pytest.approx(0.0)
    assert stats.maximum == pytest.approx(4.0)
    assert stats.n_samples == 5


def test_compute_stats_nan_count() -> None:
    data = [np.array([1.0, np.nan, 3.0, np.nan])]
    stats = compute_stats(iter(data))
    assert stats.n_nan == 2
    assert stats.n_samples == 2


def test_compare_stats_self() -> None:
    dataset = make_normal_dataset(500)
    stats = compute_stats(iter(dataset))
    metrics = compare_stats(stats, stats)
    assert metrics["mean_diff"] == pytest.approx(0.0)
    assert metrics["std_diff"] == pytest.approx(0.0)
    assert metrics["histogram_intersect"] == pytest.approx(1.0, abs=1e-6)


def test_kl_divergence_self_is_zero() -> None:
    dataset = make_normal_dataset(500)
    stats = compute_stats(iter(dataset))
    assert kl_divergence(stats, stats) == pytest.approx(0.0, abs=1e-4)


def test_kl_divergence_different_distributions() -> None:
    normal_stats = compute_stats(iter(make_normal_dataset(1000)))
    skewed_stats = compute_stats(iter(make_skewed_dataset(1000)))
    assert kl_divergence(normal_stats, skewed_stats) > 0.01


def test_empty_dataset_stats() -> None:
    stats = compute_stats(iter([]))
    assert stats.n_samples == 0
    assert stats.mean == 0.0


def test_stratified_temporal_segments_covered() -> None:
    segments = 5
    n_per_seg = 200
    dataset: list[np.ndarray] = []
    for seg_idx in range(segments):
        mean_val = float(seg_idx * 10)
        for _ in range(n_per_seg):
            dataset.append(np.full(4, mean_val, dtype=np.float32))

    config = {"method": "stratified_temporal", "num_segments": segments, "seed": 0}
    representative = list(create_representative_dataset(dataset, 50, config))

    means_found = {round(float(arr.mean())) for arr in representative}
    expected_means = {seg_idx * 10 for seg_idx in range(segments)}
    covered = len(means_found & expected_means)
    assert covered >= 4, (
        "Only "
        f"{covered}/5 temporal segments represented. Found: {sorted(means_found)}"
    )


def test_output_is_iterator() -> None:
    result = create_representative_dataset(make_normal_dataset(200), 50)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_output_preserves_dtype() -> None:
    dataset = [np.zeros(8, dtype=np.float32) for _ in range(100)]
    for arr in create_representative_dataset(dataset, 20):
        assert arr.dtype == np.float32
