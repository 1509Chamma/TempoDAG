# Platform Vision

## Summary

The research points to a stronger direction than the current EdgeLSTM name
suggests. The future platform should not be an LSTM-specific compiler. It
should become a general compiler for streaming time-series workloads on edge
and adaptive hardware.

Working thesis:

> Build an open compiler for stateful temporal dataflow: ingest PyTorch,
> TensorFlow, or ONNX time-series models; recover explicit temporal structure;
> optimize state, memory, schedule, precision, and verification; then generate
> reproducible FPGA-oriented hardware artifacts.

This keeps the current repo's useful foundation while expanding the project
from "LSTM acceleration" to a broader and more defensible platform.

## Key Research Synthesis

The uploaded research converges on five conclusions.

1. Ordinary DAG compilation is the wrong abstraction. The central object should
   be a compact temporal graph with same-timestep edges and delayed temporal
   edges.
2. Time-series models need first-class state objects: hidden states, rolling
   buffers, sequence caches, windows, delay lines, Kalman/state-space state,
   and online statistics.
3. The optimizer should target steady-state streaming behavior, especially
   initiation interval, recurrence bounds, buffer depth, state placement, and
   off-chip traffic.
4. Fixed-point correctness is a compiler feature, not a post-hoc notebook
   check. The platform needs a fixed-point oracle, bit-exact HLS/RTL parity,
   long-horizon drift diagnostics, and range/stability analysis.
5. The first credible implementation path should prioritize streaming
   statistical/DSP primitives, TCNs, and GRUs before dense attention or
   Mamba-like selective SSMs.

## Identity

The name `EdgeLSTM` is now too narrow for the research direction. It is still
fine as the current repository name while the compiler core is being built, but
the long-term platform should be branded around temporal graphs rather than one
model family.

Candidate names:

- `TempoDAG`
- `ChronoGraph`
- `TemporalEdge`
- `StreamForge`

Recommended working name: `TempoDAG`.

One-sentence positioning:

> TempoDAG is a compiler and verification platform for streaming time-series
> models represented as stateful temporal dataflow graphs.

## Target Abstraction

The platform should introduce a stateful temporal IR above the current operator
IR.

Core concepts:

- `Process`: a streaming component with one or more logical clocks.
- `Kernel`: an acyclic same-timestep tensor/dataflow region.
- `State`: persistent structured values such as hidden state, Kalman state, or
  running statistics.
- `Buffer`: bounded history such as delay lines, ring buffers, and rolling
  windows.
- `Window`: a causal temporal view over a buffer.
- `Cache`: append or sliding history, especially for attention key/value state.
- `Edge0`: a same-timestep dependency.
- `EdgeDelta`: a positive-lag temporal dependency.

The key invariant:

> Same-timestep edges must form a DAG. Any cycle in the full graph must contain
> at least one positive-lag temporal edge.

That makes recurrence analyzable without pretending every model is feed-forward
or forcing the compiler to fully unroll time.

## Compiler Stack

The future stack should have five layers.

1. Frontends

   PyTorch, TensorFlow, and ONNX should be imported into a canonical graph with
   shapes, dtypes, source metadata, and temporal candidates preserved. ONNX
   remains the most practical interchange path early on, but PyTorch FX/export
   should eventually be used where it preserves source structure better.

2. Temporal IR

   The compiler should recover sequence axes, state-threading, scan/loop
   structure, rolling windows, recurrent cells, causal convolutions, and cache
   behavior. Structured temporal ops should be preserved until after temporal
   analysis.

3. Optimization

   The optimizer should be split into two linked optimization problems.

   First, the temporal graph optimizer rewrites the model-level structure:
   fusion, recurrence boundaries, buffer sharing, state placement, fixed-point
   specialization, and legal retiming across delayed edges.

   Second, the HLS directive optimizer chooses Vitis implementation controls:
   pipeline targets, dataflow regions, unroll factors, array partitioning,
   storage binding, stream depths, and inlining choices.

   The novel platform claim is the combination: optimize the temporal graph
   first, then optimize the HLS directives that implement that graph. This lets
   TempoDAG separate graph-structure gains from directive-tuning gains while
   still reporting the combined inference speedup.

   The research suggests complementary solvers:

   - An exact CP-SAT/MILP-style oracle for small cells, recurrence-critical
     regions, and directive choices with a small search space.
   - A scalable retiming-aware large-neighborhood search for full temporal
     graphs and larger directive plans.

   Any graph rewrite that fuses nodes must preserve model parameters as
   explicit named constants, state, or immutable parameter blocks. Fusion may
   change where weights are stored or how they are indexed, but it must not
   silently change learned values, quantization metadata, or parameter identity.

4. Verification

   The compiler should own a fixed-point software oracle and trace format.
   Hardware correctness should be measured against the fixed-point oracle, not
   directly against framework floating point.

5. Code Generation

   The first backend should target Vitis HLS C++ with generated testbenches,
   scripts, fixed-point specs, and reports. Later backends can target richer
   MLIR/CIRCT, AI Engine, or mixed PL/AIE flows.

## Model Roadmap

The research recommends resisting the temptation to start with the most
fashionable models. The platform should grow from hardware-natural temporal
patterns toward harder global-context models.

### Stage 1: Streaming Stateful Core

Models unlocked:

- FIR and IIR filters
- Rolling statistics
- AR/MA/ARIMA inference
- Exponential smoothing
- Simple hybrid feature pipelines

Required concepts:

- Delay lines
- Ring buffers
- MAC/FMA
- Running accumulators
- Fixed-point range checks

Why first:

This stage proves the temporal IR and stateful streaming runtime with a small
operator surface and strong hardware fit.

### Stage 2: Generic Recurrence And TCN

Models unlocked:

- GRU
- LSTM
- Vanilla RNN
- Kalman filters
- VAR
- Causal dilated TCN

Required concepts:

- Generic `Scan`
- State buffers
- MatVec and small MatMul
- Sigmoid and tanh
- Causal Conv1D with dilation
- Residual add and pointwise ops

Why second:

GRU and TCN are the best balance of credibility, feasibility, and FPGA fit.
TCN should be the first strong neural flagship. GRU should be the first gated
recurrent flagship.

### Stage 3: Spectral And Structured State-Space

Models unlocked:

- FFT and wavelet frontends
- DSS
- early S4-style kernels
- hybrid DSP plus neural heads

Required concepts:

- Complex arithmetic
- FFT/butterfly patterns
- diagonal state-space updates
- structured recurrent kernels

Why third:

This adds modern long-range sequence capability without jumping immediately to
the most difficult selective-scan systems.

### Stage 4: Global Context And Selective SSMs

Models unlocked:

- Small quantized transformers
- Informer/Autoformer-like models
- Mamba-like selective state-space models

Required concepts:

- Attention
- KV caches
- Softmax and normalization
- layout transforms
- selective scan
- larger precision and memory planning

Why later:

These models are valuable, but they stress memory, layout, cache policy,
precision, and global context. They should come after the compiler can already
handle temporal memory and verification reliably.

## Verification Vision

The current numerical parity work is a major asset. It should become the seed
of the platform's verification story.

The future verification ladder should be:

1. Source floating-point reference: PyTorch, TensorFlow, or ONNX Runtime.
2. Canonical IR floating-point reference.
3. Compiler-owned fixed-point Python oracle.
4. HLS C simulation with matching fixed-point semantics.
5. RTL or hardware-emulation trace parity.
6. Board-in-the-loop parity on frozen golden traces.

Primary hardware gate:

> Fixed-point oracle and generated hardware must match bit-exactly on supported
> golden traces.

Secondary temporal diagnostics:

- Per-timestep max error
- Long-horizon drift
- Saturation and clipping events
- State divergence
- Stability margins
- Static range and overflow analysis

This is one of the clearest places where the platform can be more rigorous
than many existing ML-to-hardware flows.

## Novelty Claim

The strongest novelty claim is not "another FPGA ML compiler." Existing tools
already cover graph optimization, HLS generation, quantized neural inference,
and FPGA dataflow in different ways.

The defensible gap is:

> A compiler for explicitly temporal, continuously streaming, stateful models
> with first-class delayed dependencies, managed persistent state, FPGA-aware
> temporal memory planning, mixed-precision temporal correctness, and
> reproducible hardware verification.

This claim is specific enough to guide implementation and broad enough to cover
statistical, DSP, recurrent, convolutional, attention, and state-space models.

## MVP Recommendation

The first platform milestone should be:

> Compile a hybrid streaming time-series pipeline with rolling statistics or
> causal Conv1D/TCN plus a small GRU or linear head into a temporal IR, optimize
> fixed-point formats and state buffers, generate Vitis HLS, and prove parity
> against golden traces.

This MVP would demonstrate the core platform ideas without requiring a full
transformer or Mamba implementation.

Minimum deliverables:

- Temporal IR data structures.
- Import path from ONNX or PyTorch-exported ONNX.
- Pattern recovery for causal Conv1D, state buffers, windows, and simple scan.
- Fixed-point specs attached to temporal values and state.
- Golden trace format.
- HLS generation for one streaming graph.
- End-to-end parity report.

## Competition Focus

For the AMD Open Hardware competition, the platform should stay centered on a
working hardware story:

> Temporal graph in, optimized graph plus optimized HLS directives out, with
> fixed-point parity and inference-performance evidence.

That means the optimization research should serve the demo rather than replace
it. The strongest route is to show that temporal-aware scheduling reduces
initiation interval, buffer pressure, or memory traffic on a real streaming
workload, then verify the generated HLS/RTL behavior against golden traces.

The first-place-oriented path is therefore:

1. Make the temporal semantics and HLS block contract clear.
2. Generate graph-level HLS for a focused workload.
3. Add a baseline scheduler and report estimated throughput/resources.
4. Add temporal graph optimization and show graph-only before/after gains.
5. Add HLS directive optimization and show directive-only plus combined gains.
6. Validate correctness with fixed-point, HLS simulation, and eventually RTL or
   hardware-emulation traces.

Research into exact optimization, complexity, and proofs is useful because it
can sharpen the scheduler and novelty claim. It should not delay the first
working board-aware demo.

The competition application should be a focused stateful streaming workload
rather than a generic model zoo. Quant-finance streaming inference is a strong
candidate because recorded or synthetic market-data streams naturally exercise
rolling features, temporal state, model inference, and decision logic. The
baselines should progress from Python inference, to existing software-library
inference, to hls4ml where the model shape is supported, to naive generated
HLS, to TempoDAG's optimized HLS schedule.

The hls4ml comparison should be framed carefully. hls4ml is a strong model-level
FPGA inference baseline. TempoDAG's differentiator should be whole-pipeline
temporal optimization: rolling features, state buffers, model inference,
decision logic, graph rewrites, directive tuning, fixed-point parity, and HLS
scheduling evidence in one reproducible flow.

For cost and reproducibility, the first board target should be the cheapest
credible AMD platform available to the project. The default recommendation is
Kria KV260 unless the demo needs KR260-specific I/O or networking. The repo can
stay AMD-competition focused through submission and rebrand or broaden after
the contest. Buy the board after the software demo can replay a market trace,
generate stable HLS for one pipeline, and produce a parity report; before that,
simulation and artifact generation are enough to move quickly.

## Immediate Repository Implications

The current repo should evolve in this order:

1. Keep the existing IR and operator registry as the same-timestep kernel layer.
2. Add a new temporal layer rather than overloading the current graph model.
3. Keep documentation and the public story aligned around TempoDAG and temporal
   dataflow.
4. Promote `numerical_parity.py` into a compiler verification subsystem.
5. Add model examples that are not LSTM-specific.
6. Create a benchmark/report schema before adding many more operators.
7. Add HLS generation at graph level, not only operator-template level.

## Open Questions

- Should the long-term implementation become an MLIR dialect, or should the
  Python-native IR mature first?
- How much ONNX `Scan` support is needed before PyTorch FX/export paths become
  worth prioritizing?
- What is the first board target for credible hardware proof?
- Should the first flagship demo be TCN-based, GRU-based, or hybrid DSP plus
  neural head?
- Which fixed-point rounding and overflow policies should be the platform
  defaults?
- How much exact optimization is needed for MVP versus heuristic scheduling?

## Research Inputs

This vision synthesizes the uploaded reports in `research/`:

- `temporal-dag-mapping-and-scheduling.md`: temporal DAG mapping and
  scheduling.
- `amd-open-hardware-winners-2023-2025.md`: AMD Open Hardware winner patterns.
- `temporal-stateful-ml-compilation-landscape.md`: prior-art gap for
  temporal/stateful ML compilation.
- `fpga-time-series-model-roadmap.md`: time-series model roadmap.
- `fixed-point-verification-framework.md`: fixed-point verification framework.
- `temporal-ir-design.md`: temporal IR design.
