# Competition Submission Strategy

This document translates the platform vision into a competition-facing plan.
The goal is to keep the project focused on a winning AMD Open Hardware
submission while leaving room to rebrand and broaden after the contest.

## Submission Thesis

TempoDAG should be presented as an AMD-focused temporal inference compiler:

> Compile stateful streaming inference pipelines into optimized temporal graphs
> and Vitis HLS directive plans, preserve fixed-point correctness, and show
> faster inference on AMD hardware.

The competition story is not "a general compiler for everything." The
competition story is:

> A reproducible AMD hardware reference design for low-latency stateful
> streaming inference, demonstrated on a judge-fast showcase workload.

After the competition, the project can rebrand or broaden back toward the full
platform vision.

## Board Target

Use the cheapest credible AMD board first.

Recommended primary target:

- AMD Kria KV260 Vision AI Starter Kit.

Why:

- It is a low-cost AMD Kria starter kit.
- It is accessible enough for a judge or student to understand.
- It supports an embedded edge-inference story.
- It keeps the project away from expensive Alveo dependency early on.

Fallback target:

- AMD Kria KR260 Robotics Starter Kit.

Use KR260 only if the demo needs its extra networking or I/O story. Otherwise,
the extra cost does not help the core compiler claim.

Avoid as the first target:

- Alveo U280 or similar data-center accelerator cards.

They are powerful and finance-relevant, but the cost and setup burden make them
poor first targets unless the project already has reliable access to one.

Suggested purchase timing:

- Do not buy the board before the software-only demo can generate stable HLS
  for at least one representative streaming pipeline.
- Buy the board once the notebook can replay a market trace, run Python and
  software-library baselines, generate HLS artifacts, and produce a parity
  report.
- The board then becomes a validation and packaging target, not a blocker for
  early compiler work.

## Showcase Application

Quant-finance streaming inference is a strong candidate showcase, but it is not
the platform's main purpose.

Candidate demo:

- A market-data stream enters the model.
- The model predicts a short-horizon signal or risk score.
- A simple order-decision policy emits submit, hold, cancel, or reduce.
- TempoDAG compiles the temporal model to HLS.
- The optimized schedule is compared with a baseline schedule.
- The fixed-point oracle and generated HLS produce matching decisions.

The demo should avoid implying production trading readiness. The framing should
be research-grade and reproducible:

> low-latency order-decision inference on recorded or synthetic market streams.

## Baseline Ladder

The competition result should show a progression of baselines:

1. Python reference inference.
2. Existing software libraries where practical, such as NumPy, PyTorch, ONNX
   Runtime, or a vectorized CPU implementation.
3. hls4ml where the model shape is supported, especially for neural submodels.
4. Naive generated HLS schedule.
5. TempoDAG optimized graph with default HLS directives.
6. TempoDAG optimized graph with optimized HLS directives.
7. Optional board or hardware-emulation result.

The strongest claim is not only that hardware is faster than Python. The
stronger claim is:

> The temporal-aware scheduler improves the generated HLS implementation while
> preserving the exact order decisions under fixed-point semantics.

hls4ml should be treated as an important baseline, not as the enemy. hls4ml is
strong for model-level neural inference. TempoDAG's claim should be that it
optimizes the whole streaming temporal pipeline around the model: rolling
features, state buffers, model inference, decision logic, and schedule
reporting.

## Two-Level Optimization Evidence

The submission should report two separate optimization effects.

Graph optimization:

- fuses compatible stateless chains;
- shares compatible rolling windows and delay buffers;
- moves temporal feature computation closer to consumers;
- specializes fixed-point formats for stream and state values;
- preserves learned weights as named constants, state, or immutable parameter
  blocks when nodes are fused.

HLS directive optimization:

- chooses `PIPELINE`, `DATAFLOW`, `UNROLL`, and `ARRAY_PARTITION` directives;
- selects stream depths and storage bindings;
- controls function inlining and loop structure;
- tunes the generated HLS implementation for initiation interval, latency, and
  resources.

The ideal report table should isolate:

1. default graph plus default directives;
2. optimized graph plus default directives;
3. optimized graph plus optimized directives.

This makes the novelty easier to defend because the graph optimizer and HLS
directive optimizer each have visible evidence.

## Model Suite

Use a small suite rather than a single fragile model.

Initial models:

- Rolling statistics plus linear score.
- Exponential moving average crossover.
- Small TCN classifier.
- Small GRU signal model.
- Optional Kalman-style state estimator.

Each model should share the same demo structure:

- input stream;
- Python reference;
- fixed-point oracle;
- generated HLS;
- optimized schedule report;
- decision/parity report.

## Judge-Fast Demo

Ship a notebook as the executive demo.

Notebook goals:

- Load a small included market trace.
- Run Python inference.
- Run fixed-point oracle inference.
- Render or load generated HLS artifacts.
- Show baseline-vs-optimized schedule metrics.
- Show that the final order decisions match.
- Produce one compact report table and one plot.

Required artifacts:

- Small sample trace.
- Frozen model weights.
- Generated HLS output.
- Golden trace.
- Expected report.
- Optional prebuilt simulation output.

The notebook should be paired with a short video that shows:

- the input stream;
- the model decision output;
- the compiler producing HLS artifacts;
- the before/after performance table;
- parity passing.

## Repository Packaging

During the competition, the repo should be organized around the AMD submission.

Recommended top-level submission folders:

- `benchmarks/`: showcase workloads and baseline runners.
- `demo/`: notebook and judge-fast scripts.
- `artifacts/`: generated reports, traces, and selected generated HLS outputs.
- `docs/`: architecture, roadmap, hardware contract, and submission guide.
- `examples/`: smaller developer examples.
- `hls/`: reusable HLS templates.
- `src/`: compiler package.
- `tests/`: unit and integration tests.

The repo can still keep the broader compiler architecture, but the README should
lead with the competition path until submission.

## Success Metrics

Minimum metrics:

- Python latency per timestep or per sequence.
- Software-library baseline latency.
- hls4ml latency/resource numbers for supported model components.
- Naive HLS estimated or simulated initiation interval.
- Graph-optimized HLS estimated or simulated initiation interval.
- Graph-plus-directive-optimized HLS estimated or simulated initiation interval.
- End-to-end per-event decision latency.
- Resource estimates or synthesis report summary.
- Fixed-point parity pass/fail.
- Order-decision agreement rate.

Best-case metrics:

- HLS C simulation cycle count.
- RTL co-simulation or hardware-emulation trace parity.
- Board runtime for the judge-fast trace.
- Energy or throughput-per-watt estimate if reliable.

## Submission Positioning

Use this structure in the README, report, and video:

1. Problem: low-latency streaming inference for order decisions.
2. Baseline: Python and existing software inference are easy but not hardware
   optimized.
3. Contribution: TempoDAG extracts temporal structure and optimizes the HLS
   schedule.
4. Proof: fixed-point parity plus before/after schedule and inference metrics.
5. Reproducibility: notebook, sample trace, generated HLS, and expected report.

## What Not To Do Before Submission

- Do not lead with broad rebranding.
- Do not make attention, transformers, or Mamba the first showcase.
- Do not let mathematical optimization research delay the demo.
- Do not require expensive data-center FPGA hardware for the first judged path.
- Do not frame the output as trading advice or production finance tooling.

The competition target is a trustworthy compiler and hardware demonstration,
not a live trading product.
