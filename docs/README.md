# Documentation

Documentation for TempoDAG, a compiler for streaming time-series models with
stateful temporal dataflow.

## Start here

- **[Explainer](explainer.md)** — what TempoDAG is and how it works, assuming
  no hardware background, with diagrams.
- **[Benchmarks](benchmarks.md)** — measured results, comparisons, and
  post-route timing.
- **[Cost-model validation](cost-model-validation.md)** — the evidence that
  the compiler predicts hardware before synthesis: 26 designs, 7 published
  cells, pre-registered predictions.
- **[Accuracy retention](accuracy-retention.md)** — what deployment costs in
  task accuracy (measured: nothing), across three tasks and three model
  classes.
- **[Research walkthroughs](../research/walkthrough/)** — three short,
  plain-language notebooks (accuracy, speed, attention).

## Architecture & concepts

- [Architecture](architecture.md) — compiler layers and IR design.
- [Temporal IR guide](temporal-ir-guide.md) — stateful dataflow concepts.
- [Temporal execution contract](temporal-execution-contract.md) — execution and
  HLS lowering semantics.
- [Scheduling guarantees](scheduling-guarantees.md) — the iteration-bound
  theorem, its invariance and feed-forward corollaries, and the exactness
  properties, with proofs and an honest proved-vs-measured boundary.
- [Streaming-latency classes](streaming-latency-classes.md) — the
  architecture-level dichotomy: which temporal networks have a fundamental
  streaming-latency floor and which provably do not, decided mechanically
  by the compiler.
- [Roadmap](roadmap.md) — milestone status and what remains.

## Getting set up & contributing

- [Environment setup](environment-setup.md) — local environment and hooks.
- [Temporal quickstart](temporal-quickstart.md) — a first temporal model.
- [Development guide](development.md) — day-to-day workflow and extension.
- [Calibration guide](calibration.md) — quantization and dataset sampling.
- [Contributing guide](../CONTRIBUTING.md) — pull-request expectations.
- [Security policy](../SECURITY.md) — reporting process.
