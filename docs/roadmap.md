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

## Dream Platform Milestones

The long-term target is a platform that turns a temporal model graph into
HLS-backed hardware artifacts, validates both correctness and inference
performance, and uses scheduling optimization as the core differentiator.

## Execution Milestone Plan

This is the practical path from the current temporal MVP to the competition
submission. Each milestone should leave behind a runnable artifact, not only a
design note.

### M1: Temporal Semantics And Hardware Contract

Primary output:

- A precise compiler contract for temporal execution and generated HLS blocks.

Required work:

- Define reset, warm-up, flush, state read/write, buffer update, and stream
  termination semantics.
- Define the standard generated HLS block interface for streams, state,
  buffers, and optional memory access.
- Specify how `Edge0` and `EdgeDelta` lower to wires, registers, FIFOs, ring
  buffers, or RAM.
- Add small reference graphs that exercise delay, recurrence, windows, and
  state reset.
- Document the legality checks that every optimizer and backend pass must
  preserve.

Exit gate:

- A developer can implement a graph rewrite or HLS renderer against one written
  contract without guessing temporal or hardware semantics.

### M2: Graph-Level HLS Artifact Path

Primary output:

- Generated HLS and testbench artifacts for one representative temporal
  pipeline.

Required work:

- Lower one representative streaming pipeline into temporal IR.
- Preserve model weights as named constants, state, or immutable parameter
  blocks.
- Generate graph-level HLS wrappers, not only isolated operator snippets.
- Emit golden traces and expected decision outputs.
- Keep the artifact path notebook-friendly.

Exit gate:

- One command or notebook cell produces the temporal process JSON, generated
  HLS, testbench, golden trace, and report for the same pipeline.

### M3: Baseline Scheduler And Cost Model

Primary output:

- A conservative schedule and resource estimate for the generated HLS pipeline.

Required work:

- Define schedule phases, estimated latency, estimated initiation interval, and
  buffer-depth accounting.
- Attach per-node cost records for latency, resources, and memory traffic.
- Generate a baseline directive plan with safe defaults.
- Produce report tables for estimated II, latency, buffers, and resource use.

Exit gate:

- The report can compare Python/software baseline latency against generated HLS
  schedule estimates for at least one model.

### M4: Temporal Graph Optimizer

Primary output:

- An optimized temporal graph that improves the schedule before HLS directive
  tuning.

Required work:

- Implement parameter-preserving fusion for simple stateless chains.
- Share compatible rolling windows and delay buffers.
- Specialize fixed-point metadata for stream and state values.
- Add legality checks for all graph rewrites.
- Report graph-only gains against the default temporal graph.

Exit gate:

- The optimized graph preserves fixed-point decisions and shows a measurable
  schedule, buffer, or traffic improvement with default directives.

### M5: HLS Directive Optimizer

Primary output:

- An optimized Vitis HLS directive plan for the optimized graph.

Required work:

- Model directive choices for `PIPELINE`, `DATAFLOW`, `UNROLL`,
  `ARRAY_PARTITION`, storage binding, stream depth, and inlining.
- Generate inline pragmas or Tcl directives from the directive plan.
- Search or sweep a bounded directive space.
- Parse or ingest HLS reports where available.
- Report directive-only gains and combined graph-plus-directive gains.

Exit gate:

- The same optimized graph can be rendered with default directives and
  optimized directives, with separate metrics for each.

### M6: Judge-Fast Submission Artifact

Primary output:

- A reproducible AMD-focused demo package.

Required work:

- Add a judge-fast notebook with sample trace, model weights, generated HLS,
  golden trace, expected output, and report tables.
- Add a short video script centered on the market stream, generated artifacts,
  before/after metrics, and parity result.
- Prepare README entry points for a fresh evaluator.
- Keep the project AMD-focused until submission.

Exit gate:

- A fresh reader can run or inspect the key result in minutes and understand the
  core claim without reading the whole codebase.

Note:

- Quant-finance streaming inference is a strong candidate showcase workload,
  but it should not be treated as the platform's main purpose. The core platform
  remains temporal graph optimization plus HLS directive optimization for
  stateful streaming inference.

### M7: Board Validation

Primary output:

- KV260 or KR260 validation once software artifacts are stable.

Required work:

- Buy the board only after M2 and M3 are stable.
- Add board setup notes and device-specific constraints.
- Run HLS C simulation first, then RTL co-simulation or board flow where
  feasible.
- Capture logs, resource reports, and final parity/performance evidence.

Exit gate:

- The board or hardware-emulation evidence supports the same report generated
  by the software-only flow.

### Milestone 1: Temporal Semantics Contract

Goal:

- Make the temporal graph semantics precise enough that every later optimizer
  and backend pass has one source of truth.

Needed capabilities:

- Canonical definitions for same-timestep edges, delayed edges, state reads,
  state writes, buffer updates, reset, warm-up, flush, and stream termination.
- A legality checker that proves same-timestep regions are acyclic and all
  cycles include positive temporal lag.
- A small set of reference temporal graphs that exercise delay, recurrence,
  rolling windows, and state reset.

Competition value:

- Shows the project has a real compiler foundation rather than only a
  collection of HLS templates.

### Milestone 2: Hardware Contract For HLS Templates

Goal:

- Define exactly how temporal graph nodes map to synthesizable C++ HLS blocks.

Needed capabilities:

- Standard block interfaces for streaming inputs/outputs, reset, valid/ready or
  FIFO semantics, state initialization, and optional memory access.
- A mapping from temporal edges to registers, FIFOs, ring buffers, or RAM.
- Generated C++ testbenches that consume the same golden traces used by the
  fixed-point oracle.

Competition value:

- Makes the demo inspectable: judges can see graph nodes become HLS blocks with
  a consistent interface.

### Milestone 3: Cost Model And Baseline Scheduler

Goal:

- Estimate inference throughput and resource pressure before expensive
  synthesis runs.

Needed capabilities:

- Per-node cost records for latency, initiation interval, DSP/LUT/FF/BRAM/URAM,
  traffic, and fixed-point configuration.
- Baseline temporal scheduler that computes phases, buffer depths, and a
  conservative initiation interval.
- Report output that compares estimated latency/throughput against HLS C
  simulation results.

Competition value:

- Gives the project a measurable optimization loop, not just code generation.

### Milestone 4: Two-Level Optimization For Inference

Goal:

- Use temporal graph structure and Vitis HLS directive choices together to
  improve inference throughput and hardware efficiency.

Needed capabilities:

- A temporal graph optimizer for fusion, recurrence boundaries, buffer sharing,
  state placement, fixed-point specialization, and legal retiming.
- An HLS directive optimizer for `PIPELINE`, `DATAFLOW`, `UNROLL`,
  `ARRAY_PARTITION`, storage binding, stream depth, and inlining choices.
- Local exact optimization for recurrence-critical subgraphs or small directive
  search spaces.
- Scalable whole-graph heuristic for larger graph and directive plans.
- Objective hierarchy centered on initiation interval first, then latency,
  traffic, and resource balance.
- Before/after reports that separate graph optimization gains, directive
  optimization gains, combined speedup, resource changes, and parity status.
- Parameter-preserving graph rewrites: fused nodes must keep learned weights as
  explicit named constants, state, or immutable parameter blocks.

Competition value:

- This is the strongest novelty hook: temporal graph optimization plus HLS
  directive optimization improves inference schedules while preserving
  streaming semantics and learned parameters.

### Milestone 5: Bit-Exact HLS And RTL Confidence

Goal:

- Prove generated hardware behavior against compiler-owned fixed-point traces.

Needed capabilities:

- Fixed-point oracle as the primary reference.
- HLS C simulation parity against golden traces.
- RTL or hardware-emulation trace comparison for selected designs.
- Diagnostics for per-timestep error, state divergence, clipping, overflow, and
  long-horizon drift.

Competition value:

- Turns the platform from "it generates HLS" into "it generates evidence."

### Milestone 6: Board-Aware Demo And First Showcase Workload

Goal:

- Produce a polished end-to-end demo that is credible for the AMD Open Hardware
  competition.

Needed capabilities:

- One named board target with device constraints and reproducible scripts.
- A flagship streaming workload, such as rolling statistics plus TCN/GRU, a
  DSP-plus-neural pipeline, or a quant-finance order-decision pipeline.
- Generated HLS, simulation traces, performance report, and resource report.
- A clear comparison against Python inference, existing software-library
  inference, hls4ml where applicable, an unoptimized graph with default
  directives, an optimized graph with default directives, and an optimized
  graph with optimized directives.
- A judge-fast notebook with sample traces, generated artifacts, expected
  output, and report tables.

Competition value:

- Keeps the project aligned with what wins: a concrete working system, a clear
  hardware story, strong measurements, and a memorable technical contribution.

Buy the target board only after the software demo can generate stable HLS and a
parity report for one representative streaming pipeline. That keeps early work
fast and turns hardware bring-up into validation rather than discovery.

### Milestone 7: Broader Model Coverage

Goal:

- Expand only after the compiler contract, optimizer, and verification ladder
  are strong.

Needed capabilities:

- GRU and TCN as first neural flagships.
- Kalman/state-space and spectral workloads next.
- Attention/KV cache and selective SSMs later, after memory planning and
  long-horizon verification are mature.

Competition value:

- Prevents the project from chasing fashionable models before the core platform
  can demonstrate a clean win.

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
- Keep the public repo AMD-competition focused until submission, then revisit
  broader naming and packaging after the contest.

For the current implementation checklist, see
[30-Day Roadmap](roadmap-30day.md).
