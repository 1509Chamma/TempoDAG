# Roadmap

## Summary

TempoDAG is moving from a useful same-timestep compiler core toward an explicit
temporal compiler for streaming time-series workloads. The near-term roadmap is
organized around model stages rather than a single LSTM-focused path.

The current codebase already has the kernel-layer foundation: IR graph
construction, operator registration, parser frontends, quantization helpers,
representative-dataset calibration, device metadata, and operator-level HLS
template rendering. The next work layers temporal state, delayed dependencies,
verification, and streaming hardware generation on top of that core.

## Current Baseline

Today the repo supports:

- Same-timestep IR graph construction and operator registration.
- Primitive operator validation and coarse FPGA cost heuristics.
- ONNX ingestion with PyTorch and TensorFlow wrappers.
- Quantization spec attachment.
- Representative-dataset calibration utilities.
- Operator-level HLS template rendering.
- Device presets and validation.
- Temporal process scaffolding with explicit clocks, kernels, state, buffers,
  same-timestep edges, delayed edges, and validation that same-timestep edges
  form a DAG.

## Stage 1: Streaming Stateful Core

This is the current focus. The goal is to prove that TempoDAG can model and
verify simple streaming workloads with first-class state.

Models unlocked:

- FIR and IIR filters.
- Rolling statistics.
- AR/MA/ARIMA-style inference.
- Exponential smoothing.
- Hybrid feature pipelines with small linear heads.

Compiler capabilities:

- Temporal IR process, kernel, state, buffer, `Edge0`, and `EdgeDelta`
  structures.
- Delay lines and ring buffers.
- Rolling windows and running accumulators.
- Fixed-point range metadata for temporal values and state.
- Golden traces for per-timestep verification.

## Stage 2: Generic Recurrence And TCN

Once the streaming core is stable, the next target is recurrent and causal
neural sequence models.

Models unlocked:

- GRU.
- LSTM.
- Vanilla RNN.
- Kalman filters.
- VAR.
- Causal dilated TCN.

Compiler capabilities:

- Generic scan/state-threading representation.
- Pattern recovery for recurrence and causal convolution.
- Stateful quantization profiles.
- Per-timestep parity against framework references and fixed-point oracles.

## Stage 3: Spectral And Structured State-Space

This stage adds longer-horizon sequence capability while keeping hardware
structure visible.

Models unlocked:

- FFT and wavelet frontends.
- DSS and early S4-style kernels.
- Hybrid DSP plus neural heads.

Compiler capabilities:

- Complex arithmetic.
- Butterfly-style dataflow.
- Diagonal state-space updates.
- Structured recurrent kernels.

## Stage 4: Global Context And Selective SSMs

Global-context models come after the compiler has a strong story for temporal
memory, precision, and verification.

Models unlocked:

- Small quantized transformers.
- Informer/Autoformer-like models.
- Mamba-like selective state-space models.

Compiler capabilities:

- Attention and KV-cache state.
- Selective scan.
- Memory-aware buffer placement.
- Long-horizon drift and saturation diagnostics.

## Cross-Cutting Priorities

- Keep `tempo_dag.ir` as the same-timestep kernel layer.
- Use `tempo_dag.ir_temporal` for streaming processes and delayed
  dependencies.
- Strengthen verification around fixed-point oracles and golden traces.
- Expand HLS from isolated operator templates toward graph-level temporal
  generation.
- Keep documentation, examples, and tests aligned with each implemented stage.

For the current implementation checklist, see
[30-Day Roadmap](roadmap-30day.md).
