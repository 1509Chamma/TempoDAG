# Accuracy retention across tasks

A model that runs 1000× faster is worthless if quantizing it to the chip's
arithmetic ruins its predictions. This page answers that concern with
measurements: small streaming models are trained on three tasks, deployed
through the exact compiler path (trained weights → temporal process →
numerical parity proof → fixed-point evaluation), and scored before and
after. Q6.12 is the deployed number format — 18-bit fixed-point with 12
fractional bits, the same datapath behind every hardware figure in
[the benchmarks](benchmarks.md). The models deliberately span the
compiler's latency classes, so the table doubles as an accuracy-vs-latency
trade study.

**How deployed accuracy is measured without a board:** the fixed-point
oracle implements the emitter's exact `ap_fixed` semantics, and the
[cost-model validation campaign](cost-model-validation.md) verified
oracle-vs-RTL agreement by C/RTL co-simulation across the design suite.
Every run first asserts that the compiler's graph reproduces the trained
PyTorch model to float round-off (max observed 2×10⁻⁶) — the check that
catches weight-layout mistakes — then evaluates the quantized model through
the oracle. Reproduce with `python experiments/accuracy/run_accuracy.py`;
the raw log is `experiments/accuracy/results/accuracy.jsonl`.

## Tasks

- **Mackey-Glass** (chaotic forecasting, 12 steps ahead — the standard
  horizon where copy-forward baselines collapse).
- **NARMA-10** (nonlinear system identification, the reservoir-computing
  benchmark).
- **ECG5000** (real heartbeats from the UCR archive; binary
  normal-vs-abnormal, shuffled test split).

Forecasting tasks are scored by **NRMSE** — root-mean-squared error divided
by the signal's own spread, so the number is comparable across tasks and
self-interpreting: 0 is perfect, and 1.0 means no better than always
predicting the average. Classification is scored by plain accuracy.

Baselines are the best *legitimate* naive predictor: majority class for
classification; the better of training-mean and last-observed-input for
forecasting (persistence on the target is excluded at horizon h — the
previous target lies h−1 steps in the future).

## Results (H=16, Q6.12)

| task | model | latency class | float | deployed | Δ | naive baseline |
|---|---|---|---:|---:|---:|---:|
| ECG5000 | GRU | 60 ns/sample | **0.942** | **0.942** | 0.000 | 0.573 |
| ECG5000 | diag-SSM | 20 ns/sample | 0.899 | 0.899 | 0.000 | 0.573 |
| ECG5000 | minGRU | 20 ns/sample | 0.842 | 0.842 | 0.000 | 0.573 |
| Mackey-12 | GRU | 60 ns/sample | **0.355** | **0.356** | +0.001 | 1.000 |
| Mackey-12 | diag-SSM | 20 ns/sample | 0.899 | 0.900 | +0.001 | 1.000 |
| Mackey-12 | minGRU | 20 ns/sample | 1.017 | 1.017 | −0.000 | 1.000 |
| NARMA-10 | GRU | 60 ns/sample | **0.669** | **0.668** | −0.001 | 1.002 |
| NARMA-10 | diag-SSM | 20 ns/sample | 0.717 | 0.718 | 0.000 | 1.002 |
| NARMA-10 | minGRU | 20 ns/sample | 0.821 | 0.821 | 0.000 | 1.002 |

## What the table says

1. **Deployment is free, everywhere measured.** Across nine task×model
   cells the largest float-to-deployed change is 0.0012 NRMSE / 0.1%
   accuracy. The Q6.12 datapath with LUT activations does not measurably
   cost task quality on these workloads.
2. **The latency classes have a capability price, and it is task-dependent.**
   On ECG classification the 3× faster cells give up 4–10 accuracy points
   against the GRU. On chaotic forecasting the gap is structural: the GRU
   beats every naive by 65%, the diagonal SSM only slightly, and minGRU —
   whose gates cannot see the state — fails to beat the mean. Speed-class
   cells are real options, not free upgrades; the compiler makes the
   latency side of that trade precise (see the
   [validation campaign](cost-model-validation.md)), and this table
   supplies the accuracy side.
3. **Caveats.** Small models (H=16), three tasks, one quantization format,
   modest tuning (400 epochs, one seed) — this is a deployment-cost study,
   not a model-quality leaderboard. Two earlier defects were caught by the
   protocol itself and corrected: a class-sorted test file (exposed by the
   baseline column) and an information-leaking baseline candidate (recorded
   in the results log).
