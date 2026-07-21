# Cross-Framework Streaming Inference Benchmark

Repeatable harness comparing **per-sample streaming inference** (batch=1,
state carried across samples — the deployment-realistic mode) across
architectures and execution backends.

## Design

|  | axis | values |
|---|---|---|
| **Architectures** | temporal model families | statistical (EWMA+z-score), RNN, LSTM, GRU, tiny causal Transformer |
| **Backends** | execution stacks | numpy (CPU), torch eager (CPU), torch.compile (CPU), torch (CUDA if present), hls4ml (pending), **TempoDAG->Vitis (pending, fills in as the wiring registry grows)** |
| **Datasets** | domains where inference speed matters | quant-finance stream (synthetic seeded random-walk now; public OHLCV next), network intrusion (NSL-KDD, planned), ECG anomaly (MIT-BIH, planned), turbofan RUL (C-MAPSS, planned) |

Speed is the claim; accuracy is the *invariant*: every backend must produce
outputs matching the reference implementation (parity column), and hardware
rows must match bit-exactly per the TempoDAG legality contracts.

## Fairness rules

1. **Streaming protocol**: one sample at a time, state carried; no batching.
   (This is the honest deployment mode — and where GPUs pay kernel-launch
   overhead; that is a finding, not a bias.)
2. Same trained weights across every backend of a row; seeded everything.
3. Warmup 200 samples, then timed 2000; report median / p95 / p99 ns per
   sample and samples/sec. Wall-clock via perf_counter_ns.
4. Backends that cannot express a workload record `unsupported` with the
   reason (e.g. hls4ml has no stateful streaming cell; TempoDAG transformer
   pending) — visible in the table, never silently dropped.
5. Every result row carries provenance (git sha, versions, host, config).

## Domain latency bars (what "fast enough" means)

| domain | bar | source of the bar |
|---|---|---|
| network IDS | ~1-10 us/flow-update (line rate) | 10G small-packet budget |
| quant finance | ~1-50 us tick-to-signal | exchange colocation practice |
| ECG / wearable | ~1 ms/beat | real-time monitoring |
| turbofan RUL | ~10 ms/cycle | sensor update rate |

(TempoDAG measured so far: demo pipeline II=1 @ 200 MHz = 5 ns/sample
sustained, 0.285 us latency — from the Vitis proof ladder.)

## Run

```
.venv\Scripts\python experiments\benchmark\bench.py --arch gru --dataset synth_finance
.venv\Scripts\python experiments\benchmark\bench.py --all
```

Results append to `results/results.jsonl` (one provenance-stamped row per
arch x backend) and print a table.
