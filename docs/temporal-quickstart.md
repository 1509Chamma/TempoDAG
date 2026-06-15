# Temporal Quickstart

This guide walks through the Week 4 MVP path in TempoDAG:

1. Start from a supported temporal ONNX model.
2. Lower it into `tempo_dag.ir_temporal.Process`.
3. Generate a golden trace from a timestep-by-timestep PyTorch reference.
4. Emit temporal HLS C++ plus a simple testbench.
5. Write a graph-level artifact bundle with a manifest tying the process,
   golden trace, generated HLS, and testbench together.

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
- `temporal_demo_schedule.json`
- `temporal_demo_baseline_report.json`
- `temporal_demo.cpp`
- `temporal_demo_tb.cpp`
- `temporal_demo_manifest.json`
- `temporal_demo_report.json`

The manifest is the stable entry point for generated artifacts. It lists the
temporal process JSON, golden trace, schedule ABI report, baseline cost report,
generated process HLS, and generated testbench for the same pipeline so
downstream notebooks, reports, and future HLS scripts do not need to rediscover
file names.

## API Reference

Key Week 4 entry points:

- `tempo_dag.parsers.temporal_onnx.TemporalONNXParser`
- `tempo_dag.parsers.temporal_onnx.build_demo_temporal_onnx_model`
- `tempo_dag.codegen.hls.temporal_generator.write_temporal_hls_artifact_bundle`
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

The temporal HLS bundle writer produces a graph-level artifact package:

- a temporal process JSON file
- a golden trace JSON file
- a baseline schedule JSON file with node phases, edge ABI roles, storage
  choices, and conservative latency/II estimates
- a baseline report JSON file with node/resource tables, edge traffic
  estimates, storage tables, safe directive defaults, and optional software
  baseline comparison metadata
- a top-level process wrapper with buffer declarations, contract comments, and
  rendered operator kernels
- a testbench that replays each timestep from the golden trace
- a manifest that records the artifact paths and process identifier

The generated C++ is intentionally transparent and inspection-friendly. It is
meant to prove the process/codegen interface and testbench wiring before the
project grows a richer temporal scheduler.

## Schedule And Baseline Reports

The schedule JSON is the first M3 ABI artifact. It classifies the temporal
process into a stable contract that later HLS wiring can consume:

- `stream` edges for same-timestep operator-to-operator values
- `graph_input` and `graph_output` ports for runtime values
- `parameter_block` inputs for immutable weights or constants
- `state_read`, `buffer_read`, and temporal-delay edges for persistent state and
  bounded history
- per-node phase, estimated latency, and initiation interval metadata

The baseline report JSON is the first M3 cost/reporting artifact. It turns the
schedule into inspectable tables:

- resource totals from coarse operator cost records
- per-edge traffic estimates in value elements per timestep
- buffer and temporal-storage rows
- safe default directives such as process-level `DATAFLOW`, operator-level
  `PIPELINE`, channel `STREAM`, and temporal storage binding placeholders
- optional Python/software baseline comparison when latency metadata is present
  in the golden trace

Both reports are deliberately conservative. They do not yet perform directive
search or parse Vitis reports, but they give later optimization passes a
machine-readable target and a stable judge-facing summary.

## HLS Milestone 2 Compile Contract

The current HLS layer is no longer only a text renderer. Unit tests now render
representative operator templates into real C++ translation units and compile
them with a standard C++17 compiler using `-fsyntax-only`. This gives the
project a fast CI check that catches broken template syntax without requiring a
Vitis or Vivado installation.

The compile-smoke contract currently covers:

- scalar/tensor elementwise kernels: `Add`, `Sub`, `Mul`, `Div`, `Sigmoid`,
  `Tanh`, `ReLU`, and `GELU`
- reductions: `Sum`, `Mean`, and `Max`
- structural kernels: `MatMul`, `Transpose`, `Reshape`, `Concat`, `Slice`,
  `Pad`, and `Shift`
- temporal/model kernels: `Softmax`, `LayerNorm`, `Conv1D`, `LSTM`, and the
  generated temporal process/testbench pair

This is still a baseline HLS ABI, not the final optimized accelerator ABI. The
templates are written to be readable, pragma-ready, and compiler-checkable so
the next scheduler layer has a stable target.

## Supported HLS Subset

The Python IR validates a broader graph vocabulary than the first HLS template
set can lower. The current compiler-checked HLS subset intentionally supports:

- `float32` operator examples in the compile-smoke suite
- rank-2 matrix `MatMul`
- rank-2 `Transpose` with `perm=[1, 0]`
- rank-1 `Slice`, `Pad`, and `Shift`
- flattened reductions where the reduced dimension is contiguous in the
  generated layout
- `Conv1D` in `[batch, channel, time]` layout with static shapes
- `LSTM` with `X`, `W`, `R`, optional `B`, and the primary `Y` output

Unsupported HLS forms raise explicit render-time errors rather than emitting
misleading C++. The next graph scheduler should replace those narrow cases with
shape-aware address generation and a consistent operator invocation ABI.

## Sequence Overview

```text
ONNX model
  -> TemporalONNXParser
  -> Process
  -> StreamingPyTorchAdapter
  -> FixedPointOracle
  -> GoldenTraceRecorder
  -> write_temporal_hls_artifact_bundle
  -> process JSON + golden trace + schedule + report + HLS + testbench + manifest
```
