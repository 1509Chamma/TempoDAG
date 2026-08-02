# TempoDAG

[![CI](https://github.com/1509Chamma/TempoDAG/actions/workflows/ci.yml/badge.svg)](https://github.com/1509Chamma/TempoDAG/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/1509Chamma/TempoDAG/branch/main/graph/badge.svg)](https://codecov.io/gh/1509Chamma/TempoDAG)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Verified: RTL co-simulation](https://img.shields.io/badge/verified-RTL%20co--simulation-2ea44f.svg)](docs/benchmarks.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A compiler that turns streaming time-series models into RTL-verified FPGA
accelerators — with the hardware's key properties predicted before synthesis
runs.** The compiler's temporal IR is rich enough that initiation interval
and DSP usage are computed statically, stated in advance (pre-registered,
never tuned after the fact), certified against a machine-checked scheduling
theorem, and confirmed by C/RTL co-simulation. On the AMD Kria KV260 the
result is a **steady-state 60 ns per incoming sample** for compiled
recurrent models — **independent of context-window length** — which works
out to ~35× the leading open FPGA-ML toolchain's window-based approach and
~1300× PyTorch-CPU at the same task.

> **Status in one line.** The compiler and its verification ladder are complete
> and proven in Vitis simulation (C-synthesis + C/RTL co-simulation). The next
> phase of the research is hardware validation on a KV260 board — on-silicon
> latency, power, and trained-model accuracy. See
> [Next step: hardware validation](#next-step-hardware-validation).

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
| **speedup from the compiler** | **51×** | **111×** | **141×** |

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

## Next step: hardware validation

Everything above is **Vitis simulation** — the industry-standard proof of
correctness before committing a design to silicon. The next phase of the
research is to validate these results on a physical AMD Kria **KV260** board
(the exact part these designs target, `xck26-sfvc784-2LV-c`). Place-and-route
timing closure is already done without a board — the flagship designs meet
the 5 ns budget after full Vivado implementation (see
[benchmarks](docs/benchmarks.md)) — which leaves the measurements only
silicon can provide:

1. **On-silicon latency and power** — measured microseconds and watts, and a
   performance-per-watt comparison against CPU/GPU baselines.
2. **Accuracy of trained models on silicon** — deployment is already measured
   as accuracy-free *in simulation* across three tasks
   ([accuracy retention](docs/accuracy-retention.md)); board runs confirm
   that retention end to end.
3. **Live streaming inference** — a trained model running on the board on real
   streaming data at 60 ns/sample.

The cost model also makes testable predictions (DSP-per-gate scaling ~H²,
C-slow aggregate throughput); hardware validation is designed to check them.

## Reproduce the results

The fastest machine-agnostic route is Docker — one build, one run, every
Python-side result (tests, walkthroughs, research scripts, the tutorial, and
HLS emission) reproduced with pinned seeds and frozen dependencies:

```bash
docker build -t tempodag .
docker run --rm tempodag              # everything Python-side
docker run --rm tempodag tests       # or a single stage: tests | walkthroughs |
                                     # research | tutorial | emit
```

The same stages run natively via `scripts/reproduce.sh`. The one deliberately
non-portable stage is the Vitis hardware ladder (`scripts/reproduce.sh hls`) —
AMD's toolchain is proprietary and ~100 GB, so it runs on a host with Vitis
installed rather than inside the image; the reference results it produces are
in [docs/benchmarks.md](docs/benchmarks.md).

For a native setup, Python 3.12 is the development target.

```bash
python3 -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the compiler test + lint baseline (no FPGA toolchain needed):

```bash
python -m pytest -q            # 296 tests
python -m ruff check src tests
```

Re-run the demo notebooks (pure NumPy/PyTorch, no Vitis needed; they ship
pre-executed, so this refreshes the visible outputs in place):

```bash
python -m pip install -r research/requirements-research.txt
python -m nbconvert --to notebook --execute --inplace research/walkthrough/2_why_it_is_fast.ipynb
python -m nbconvert --to notebook --execute --inplace research/walkthrough/3_attention_the_fpga_way.ipynb
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
tutorials/              deploy YOUR model: trained PyTorch/ONNX/Keras -> RTL, step by step
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
- **[Tutorials](tutorials/)** — deploy your own trained model (PyTorch/ONNX/Keras → RTL)
- [Benchmarks & comparison](docs/benchmarks.md) · [Cost-model validation](docs/cost-model-validation.md) · [Accuracy retention](docs/accuracy-retention.md)
- [Scheduling guarantees](docs/scheduling-guarantees.md) · [Streaming-latency classes](docs/streaming-latency-classes.md) — the theory the cost model rests on, [machine-checked in Lean 4](proofs/)
- [Architecture](docs/architecture.md) · [Temporal IR guide](docs/temporal-ir-guide.md)
- [Roadmap](docs/roadmap.md) · [Explainer (no hardware background needed)](docs/explainer.md)
- [Environment setup](docs/environment-setup.md) · [Contributing](CONTRIBUTING.md)

## New to FPGAs or time-series models?

There's a plain-language explainer that assumes **no hardware background at
all** — what an FPGA is, why streaming models are hard to accelerate, and how
this project works, with diagrams: **[docs/explainer.md](docs/explainer.md)**.

## Citing

If this work is useful in your research, please cite it —
[CITATION.cff](CITATION.cff) carries the metadata (GitHub's "Cite this
repository" button uses it).

## Academic integrity

AI assistance (Anthropic's Claude) was used throughout this project's
research and development — exploring the design space, writing code, and
drafting documentation — under human direction and review. The project is
built so that no result rests on trust in either the human or the AI:
every hardware figure comes from a Vitis synthesis or C/RTL co-simulation
run, every research claim has a seeded, reproducible script, the central
scheduling theorem is machine-checked in Lean 4 ([proofs/](proofs/)), and
the verification ladder exists precisely so that a wrong number would be
caught by a test in this repository.

## License

MIT — see [LICENSE](LICENSE).
