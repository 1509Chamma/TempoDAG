# TempoDAG

[![CI](https://github.com/1509Chamma/TempoDAG/actions/workflows/ci.yml/badge.svg)](https://github.com/1509Chamma/TempoDAG/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/1509Chamma/TempoDAG/branch/main/graph/badge.svg)](https://codecov.io/gh/1509Chamma/TempoDAG)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Verified: RTL co-simulation](https://img.shields.io/badge/verified-RTL%20co--simulation-2ea44f.svg)](docs/benchmarks.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A compiler that turns streaming time-series models into RTL-verified FPGA accelerators.**
On the AMD Kria KV260, TempoDAG-compiled recurrent models run at **60 ns per
sample** — **~35× faster than hls4ml** (the leading open FPGA-ML toolchain) and
**~1300× faster than PyTorch-CPU** — and, unlike hls4ml, the per-sample cost is
**independent of context-window length**. Every hardware number below is
**C/RTL co-simulation verified**, not estimated.

> **Status in one line.** The compiler and its verification are done and proven
> in Vitis simulation (C-synthesis + C/RTL co-sim). The one missing step is a
> physical board to turn simulation into on-silicon measurements — real latency,
> real power, and real accuracy on trained models. See
> [What a test board unlocks](#what-a-test-board-unlocks).

---

## The result

Per-sample streaming latency (batch = 1, one timestep). CPU rows are NumPy /
PyTorch on this workstation; FPGA rows are compiled to the KV260 and
co-simulation-verified.

| model | NumPy (CPU) | PyTorch (CPU) | hls4ml (FPGA) | **TempoDAG (FPGA)** | vs PyTorch | vs hls4ml |
|---|---:|---:|---:|---:|---:|---:|
| RNN  |  9.5 µs | 36.1 µs | *broken¹* | **0.060 µs** | **600×** | — |
| GRU  | 24.2 µs | 76.8 µs | 2.12 µs | **0.060 µs** | **1280×** | **35×** |
| LSTM | 29.0 µs | 90.6 µs | 1.96 µs | **0.060 µs** | **1510×** | **33×** |
| diagonal-linear (SSM) | — | — | — | **0.020 µs** | — | — |
| transformer block (PatchTST) | 118 µs | 267 µs | — | **1.12 µs/token** | **238×** | — |

<sup>¹ hls4ml 1.3.0's SimpleRNN converter silently drops the recurrent layer —
its synthesis report contains only the dense head. GRU and LSTM are
unaffected.</sup>

**Two structural edges the table alone hides:**

- **Window-independence.** hls4ml re-reads the whole context window, so its
  per-sample cost grows with sequence length (GRU: 555 ns @ T=8 → 8355 ns @
  T=128). TempoDAG carries state and stays **flat at 60 ns for any T** — so the
  35× lead becomes ~140× at long context. This is architectural, not a constant.
- **Aggregate throughput.** The datapath is shared, so interleaving independent
  streams (C-slow) reaches aggregate II = 1: **~200M samples/s on one KV260**
  across a 12-instrument portfolio.

A one-page visual summary of the benchmarks is in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Where the speed comes from (scored against no optimization)

Every TempoDAG pass is measured against a correct-but-literal baseline emit:

| stage | RNN | GRU | LSTM |
|---|---:|---:|---:|
| Naive float per-step | 3.04 µs | 6.64 µs | 8.47 µs |
| + tree-matmul & elementwise fusion | 1.35 µs | 1.72 µs | 1.92 µs |
| + **II-bound streaming + fixed-point** | **0.060 µs** | **0.060 µs** | **0.060 µs** |
| **speedup from our compiler** | **51×** | **111×** | **141×** |

The key idea (the "II-bound" reframe): a streaming recurrent model's per-sample
cost is not the full step latency but the **initiation interval** — the depth of
the loop-carried recurrence — because everything off the feedback path overlaps
across samples. A fixed-point datapath with lookup-table activations makes that
interval physically realizable. The
[walkthroughs](research/walkthrough/) explain the idea and reproduce it.

## Verification: every result is certified

TempoDAG does not report estimates it cannot back up. Each compiled design is
checked against a fixed-point oracle through a verification ladder:

1. NumPy reference → **golden trace**
2. **C-simulation** (asserting testbench, `errors=0`)
3. **C-synthesis** (initiation interval, latency, resources)
4. **C/RTL co-simulation** (`*** co-simulation finished: PASS ***`)

All five architectures above pass step 4. The fixed-point arithmetic is an
*oracle-relative certificate*: the golden is generated in the emitter's exact
`ap_fixed` semantics, so the gate is a few LSB, not a loose tolerance.

## What a test board unlocks

Everything above is **Vitis simulation** — correct and verified, but not yet
silicon. A single AMD Kria **KV260** development board (the exact part these
designs target, `xck26-sfvc784-2LV-c`) converts this research into a deployable,
publishable, competition-ready system by enabling the four things simulation
cannot give:

1. **Place-and-route timing closure** — confirm the 3.4–3.6 ns clock holds on
   real fabric (co-sim verifies function; only P&R verifies board timing).
2. **On-silicon latency and power** — measured microseconds and watts, and the
   **performance-per-watt vs a GPU** comparison that edge-inference judges and
   reviewers weight most.
3. **Accuracy on trained models, on silicon** — a GRU trained on the
   Mackey-Glass chaotic benchmark already keeps **99% of its accuracy** through
   the fixed-point deploy *in simulation*
   ([walkthrough 1](research/walkthrough/1_does_the_hardware_stay_accurate.py));
   the board confirms that retention on real hardware, end to end.
4. **A live demo** — a model running on the board, streaming, at 60 ns/sample:
   the "watch it happen" moment, and the basis for a hardware-competition
   submission.

In short: the hard, novel compiler work and its verification are done. A board
is the difference between *"proven in simulation"* and *"running on hardware."*

## Reproduce the results

Python 3.12 is the development target.

```bash
python3 -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the compiler test + lint baseline (no FPGA toolchain needed):

```bash
python -m pytest -q            # 296 tests
python -m ruff check src tests
```

Reproduce the core findings (pure NumPy, no Vitis needed):

```bash
python -m pip install -r research/requirements-research.txt
python research/walkthrough/2_why_it_is_fast.py            # the II-bound idea
python research/walkthrough/3_attention_the_fpga_way.py    # linear attention
```

Regenerate the HLS and benchmark numbers (requires **Vitis HLS 2026.1** and a
licence; targets the KV260 part above):

```bash
python experiments/benchmark/tempodag_backend.py gru --fixedpoint   # emit → csim → synth → cosim
```

## How it works

TempoDAG models a streaming model as a **temporal dataflow graph**: a DAG whose
only cycles pass through explicitly-typed, positive-lag delay edges (state). The
"DAG" name is the legality contract — *everything is acyclic within a timestep;
recurrence exists only through time* — which is exactly how hardware feedback
works (combinational logic is a DAG; loops close through registers).

```
PyTorch / TF / ONNX  →  Temporal IR  →  optimization passes  →  fixed-point HLS  →  Vitis  →  KV260
                        (state, delay      (fusion, II-bound      (ap_fixed +        (csim,
                         edges, buffers)    streaming, C-slow)     LUT activations)   synth, cosim)
```

## Repository layout

```text
src/tempo_dag/          the compiler
  ir/, ir_temporal/     typed IR + temporal layer (state, delay edges, buffers)
  ops/                  built-in operators with validation + cost models
  codegen/hls/          fixed-point burst-loop + transformer emitters, oracle certificate
  parsers/              ONNX + PyTorch/TensorFlow front ends
experiments/
  benchmark/            the full comparison harness (TempoDAG, hls4ml, torch, numpy)
  hls_proof/            one-command Vitis verification ladder
research/
  walkthrough/          START HERE: 3 plain-language notebooks (accuracy, speed, attention)
  lab/                  reproducibility helpers (seeds, provenance) for the benchmarks
docs/                   architecture, benchmarks, explainer, roadmap
tests/                  unit + integration + verification tests
configs/devices/        FPGA board presets (KV260 and others, JSON)
hls/operators/          operator-level HLS templates
```

## Documentation

- **[Research walkthroughs](research/walkthrough/)** — start here: accuracy, speed, attention, in plain language
- [Benchmarks & comparison](docs/benchmarks.md)
- [Architecture](docs/architecture.md) · [Temporal IR guide](docs/temporal-ir-guide.md)
- [Roadmap](docs/roadmap.md) · [Explainer (no hardware background needed)](docs/explainer.md)
- [Environment setup](docs/environment-setup.md) · [Contributing](CONTRIBUTING.md)

## New to FPGAs or time-series models?

There's a plain-language explainer that assumes **no hardware background at
all** — what an FPGA is, why streaming models are hard to accelerate, and how
this project works, with diagrams: **[docs/explainer.md](docs/explainer.md)**.

## How this project was built

TempoDAG is a solo research project by Abdul-Rahman Chamma. I built it with
heavy use of AI assistance — Anthropic's Claude, through the Claude Code
CLI — as a pair-programming and research partner: exploring the optimization
space, writing and debugging the HLS code generator, and drafting
documentation. I'm acknowledging that openly because it's how the work actually
happened.

What the AI did *not* do is make the results true. Every performance number in
this repository comes from an actual Vitis HLS run (C-synthesis and C/RTL
co-simulation), and every research claim is backed by a script you can run
yourself. The verification ladder exists precisely so that nothing has to be
taken on trust — mine or a model's. If a number here is wrong, the test that
would catch it is in the repo.

## License

MIT — see [LICENSE](LICENSE).
