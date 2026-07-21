# Benchmarks

All FPGA figures are compiled to the AMD Kria **KV260**
(`xck26-sfvc784-2LV-c`), 5 ns target clock, and **C/RTL co-simulation
verified** against a fixed-point oracle. CPU figures are NumPy 2.x and
PyTorch-CPU on the development workstation, batch = 1 streaming, median of
2000 timed iterations. A visual one-page summary of this data is published as
an interactive artifact (see [Visual summary](#visual-summary)).

> **Honest status:** these are C-synthesis + co-simulation results, not
> place-and-route or on-board measurements. Timing closure on real fabric,
> on-silicon latency/power, and accuracy on trained models require a physical
> board — see the README's "What a test board unlocks."

## Per-sample streaming latency

One timestep of streaming inference. Lower is better.

| model | NumPy (CPU) | PyTorch (CPU) | hls4ml (FPGA) | TempoDAG (FPGA) | vs PyTorch | vs hls4ml |
|---|---:|---:|---:|---:|---:|---:|
| RNN  |  9.5 µs | 36.1 µs | broken¹ | **0.060 µs** | 600× | — |
| GRU  | 24.2 µs | 76.8 µs | 2.12 µs | **0.060 µs** | 1280× | 35× |
| LSTM | 29.0 µs | 90.6 µs | 1.96 µs | **0.060 µs** | 1510× | 33× |
| diagonal-linear (SSM) | — | — | — | **0.020 µs** | — | — |
| transformer block (PatchTST) | 118.4 µs | 267.0 µs | — | **1.12 µs/token**² | 238× | — |

¹ hls4ml 1.3.0's Vitis-backend SimpleRNN converter silently emits only the
dense head, dropping the recurrent layer. Verified from the synthesis report
and logged with evidence; the GRU/LSTM rows are unaffected (their recurrent
instances are present).
² The transformer is one self-attention + FFN encoder block over an 8-token
window: 8.97 µs/block.

## TempoDAG hardware detail

KV260 budget: 1248 DSP, ~117K LUT, 144 BRAM. Every row co-simulation PASS.

| arch | II | per sample | est. clock | DSP | LUT | BRAM | Q-format |
|---|---:|---:|---:|---:|---:|---:|---|
| RNN | 12 | 60 ns | 3.63 ns | 289 (23%) | 14.1K (12%) | 1 | Q6.12 |
| GRU | 12 | 60 ns | 3.61 ns | 871 (70%) | 42.2K (36%) | 4 | Q6.12 |
| LSTM | 12 | 60 ns | 3.62 ns | 1153 (92%) | 56.2K (48%) | 8 | Q6.12 |
| diagonal-linear | 4 | 20 ns | 3.37 ns | 87 (7%) | 3.3K (3%) | 0 | Q6.12 |
| transformer block | — | 8.97 µs / 8 tok | ~3.4 ns | 250 (20%) | 29.9K (26%) | 13 | Q8.16 |

Notes: LSTM at 92% DSP is the single-model fit ceiling at H=16; DSP double-pump
or a narrower Q relieves it. The transformer needs wider integer headroom
(Q8.16) than the recurrent archs (Q6.12) because attention scores and FFN
pre-activations exceed Q6.12's range.

## Where the speed comes from (optimization ladder)

Each stage measured against a correct-but-literal baseline emit.

| stage | RNN | GRU | LSTM |
|---|---:|---:|---:|
| Naive float per-step | 3.04 µs | 6.64 µs | 8.47 µs |
| + tree-matmul + core binding + fusion | 1.35 µs | 1.72 µs | 1.92 µs |
| + II-bound streaming + fixed-point | 0.060 µs | 0.060 µs | 0.060 µs |
| **total speedup** | **51×** | **111×** | **141×** |

Transformer block, separately: serial matmul 95.9 µs → reciprocal-LUT softmax
95.8 µs (area win, −13K LUT) → pipelined tree matmul **8.97 µs** (**10.7×**).

## Window-independence

hls4ml's per-effective-sample cost grows linearly with the context window;
TempoDAG carries state and is flat.

| context window T | hls4ml GRU | TempoDAG GRU |
|---:|---:|---:|
| 8   | 555 ns | 60 ns |
| 32  | 2115 ns | 60 ns |
| 128 | 8355 ns | 60 ns |

Crossover is at T≈33 today; at long context (e.g. PatchTST's 336) the gap is
an order of magnitude and growing. This is a structural property of streaming
state, not a tuning constant.

## Aggregate throughput (C-slow)

The recurrent engines are latency-bound per stream, but the datapath is shared.
Interleaving N ≥ II independent streams (round-robin) is bit-exact to running
them separately and reaches aggregate II = 1: **~200M samples/s on one KV260**.
Streams-to-saturate = II, so the 87-DSP diagonal engine needs only 4 streams
(≈14 such engines fit one board).

## Method and reproducibility

- CPU: `experiments/benchmark/bench.py` (NumpyModel + torch, batch=1, warmup
  200 / timed 2000, median).
- FPGA: `experiments/benchmark/tempodag_backend.py <arch> --fixedpoint` emits
  the design, then runs Vitis HLS 2026.1 `csim → synth → cosim`. hls4ml runs in
  an isolated venv and is synthesized through the same v++ flow.
- Every result is provenance-stamped with the git SHA, seed, and tool
  versions used to produce it.

## Visual summary

An interactive one-page benchmark readout (log-scale latency chart,
optimization ladder, hardware table, window-independence) is published as a
Claude artifact. Regenerate or view it from the source at
`experiments/benchmark/` results, or open the shared artifact link kept with
the project.
