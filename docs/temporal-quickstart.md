# Temporal Quickstart

This guide walks through the Week 4 MVP path in TempoDAG:

1. Start from a supported temporal ONNX model.
2. Lower it into `tempo_dag.ir_temporal.Process`.
3. Generate a golden trace from a timestep-by-timestep PyTorch reference.
4. Emit temporal HLS C++ plus a simple testbench.

## Supported MVP Path

The current Week 4 implementation is intentionally narrow:

- `TemporalONNXParser` supports:
  - custom temporal operators such as `RollingMean`
  - recurrent ONNX structures detected through `Scan`, `Loop`, `LSTM`, `GRU`, and `RNN`
- temporal HLS generation currently targets a single-kernel `Process`
- testbench generation is driven from a Week 3 `GoldenTrace`

This is enough for a credible end-to-end flow while the broader temporal lowering
and scheduling story continues to grow.

## Run The Demo

From the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe examples\temporal_demo.py
```

The demo writes artifacts to `examples/generated/`:

- `temporal_demo.onnx`
- `temporal_demo_process.json`
- `temporal_demo_trace.json`
- `temporal_demo.cpp`
- `temporal_demo_tb.cpp`
- `temporal_demo_report.json`

## API Reference

Key Week 4 entry points:

- `tempo_dag.parsers.temporal_onnx.TemporalONNXParser`
- `tempo_dag.parsers.temporal_onnx.build_demo_temporal_onnx_model`
- `tempo_dag.codegen.hls.temporal_generator.render_temporal_process_hls`
- `tempo_dag.codegen.hls.temporal_generator.render_temporal_testbench`
- `tempo_dag.verification.temporal_parity.StreamingPyTorchAdapter`
- `tempo_dag.verification.golden_trace.GoldenTraceRecorder`
- `tempo_dag.verification.golden_trace.GoldenTraceValidator`

## Verification Flow

The verification ladder for the MVP is:

1. Run a PyTorch reference one timestep at a time with `StreamingPyTorchAdapter`.
2. Quantize outputs and temporal state with `FixedPointOracle`.
3. Record a reproducible JSON artifact with `GoldenTraceRecorder`.
4. Validate generated or replayed traces with `GoldenTraceValidator`.

## HLS Walkthrough

The temporal HLS generator produces two artifacts:

- a top-level process wrapper with buffer declarations and rendered operator
  kernels
- a testbench that replays each timestep from a golden trace

The generated C++ is intentionally transparent and inspection-friendly. It is
meant to prove the process/codegen interface and testbench wiring before the
project grows a richer temporal scheduler.

## Sequence Overview

```text
ONNX model
  -> TemporalONNXParser
  -> Process
  -> StreamingPyTorchAdapter
  -> FixedPointOracle
  -> GoldenTraceRecorder
  -> render_temporal_process_hls + render_temporal_testbench
```
