# Calibration Guide

## Purpose

The calibration package exists to answer a practical deployment question:

How do we choose a compact subset of data that still captures the behaviour of
the full dataset closely enough for quantization and downstream hardware
experiments?

The current implementation focuses on representative dataset selection and
distributional diagnostics. It does not yet implement a full observer or
deployment calibration pipeline, but it provides the core building blocks.

## Public API

The main entry points live under `tempo_dag.calibration`:

- `create_representative_dataset(...)`
- `DatasetStats`
- `compute_stats(...)`
- `compare_stats(...)`
- `kl_divergence(...)`
- `get_strategy(...)`

## Representative Dataset Selection

`create_representative_dataset(...)` accepts:

- An iterable of `numpy.ndarray` samples
- A `max_samples` limit
- An optional `strategy_config` dictionary

It returns an iterator of samples rather than a single stacked tensor, which
makes it easy to hand the result to calibration or evaluation loops that already
consume iterables.

### Current Behaviour

- Samples that are entirely `NaN` are dropped up front.
- Partially `NaN` samples are kept.
- If the resulting valid dataset is smaller than `max_samples`, all valid
  samples are returned and a warning is emitted.
- `include_tails=True` can force extreme samples into the output even when the
  fill strategy is something else, such as uniform or stratified temporal
  sampling.

## Sampling Strategies

### Uniform

Reservoir sampling across the full stream. This is the simplest baseline and is
the default when no strategy is supplied.

Good fit:

- Large datasets with no strong temporal or regime structure
- Baseline comparisons

### Stratified Temporal

Splits the stream into temporal buckets and samples proportionally from each
segment.

Good fit:

- Ordered time-series data
- Datasets where early, middle, and late periods all matter

### Regime Aware

Clusters samples using lightweight features derived from each array:

- Mean
- Standard deviation
- Value range

The sampler then draws from each cluster. This is a simple way to keep multiple
distributional regimes in view without assuming explicit labels.

Good fit:

- Mixture-like datasets
- Regime shifts and volatility changes

### Tail Aware

Always includes samples from the lower and upper extremes of the scalarized
value distribution, then fills the remaining budget uniformly.

Good fit:

- Quantization where scale sensitivity depends on rare extremes
- Datasets with meaningful outliers

## Tail Inclusion

The repo currently supports two related patterns:

- `{"method": "tail_aware", ...}` to use a dedicated tail-first strategy
- `{"method": "...", "include_tails": True, ...}` to compose tail inclusion
  with another fill strategy

This second path is especially useful when you want broad temporal or regime
coverage but do not want to lose rare extremes that influence quantization
scale.

## Statistics and Drift Checks

`compute_stats(...)` flattens the sample stream, ignores `NaN` values, and
tracks:

- Mean
- Standard deviation
- Minimum and maximum
- Histogram and bin edges
- Number of valid samples
- Number of `NaN` values
- 1st and 99th percentiles

`compare_stats(...)` then reports:

- Mean drift
- Standard deviation drift
- Range overlap
- Histogram intersection
- KL divergence
- Percentile drift at p01 and p99

These metrics are helpful when deciding whether a sampled calibration subset is
"good enough" relative to the full dataset.

## Example

```python
import numpy as np

from tempo_dag.calibration import (
    compare_stats,
    compute_stats,
    create_representative_dataset,
)

dataset = [np.random.randn(16).astype(np.float32) for _ in range(2000)]

config = {
    "method": "stratified_temporal",
    "num_segments": 10,
    "include_tails": True,
    "tail_percentile": 0.99,
    "seed": 0,
}

representative = list(create_representative_dataset(dataset, 256, config))

full_stats = compute_stats(dataset)
rep_stats = compute_stats(representative)
metrics = compare_stats(full_stats, rep_stats)
```

## Current Caveats

- The implementation materializes valid samples in memory before sampling. That
  is acceptable for the current repo state, but it is not the final answer for
  very large datasets.
- Regime-aware sampling uses simple hand-built features rather than model- or
  tensor-aware observers.
- There is not yet a persisted calibration report format.
- There is not yet a direct bridge from representative-dataset evaluation into
  graph-level quantization passes.

## Good Next Steps

The most natural calibration follow-ons are:

1. Persist calibration metrics and subset metadata for reproducibility.
2. Add tensor-wise or layer-wise observer flows on top of the current data
   selection utilities.
3. Connect sampled calibration outputs directly into quantization attachment for
   graph values.

