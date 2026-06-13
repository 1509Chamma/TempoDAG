# TempoDAG

TempoDAG is a compiler and verification platform for streaming time-series models represented as stateful temporal dataflow graphs. It transforms PyTorch, TensorFlow, or ONNX models into FPGA-friendly intermediate representations with first-class state management, temporal memory planning, and fixed-point correctness verification.

Rather than treating time-series workloads as ordinary DAGs, TempoDAG explicitly models delayed temporal dependencies, rolling buffers, stateful processing, and streaming optimization targets like initiation interval and steady-state behavior.

It is not yet a full "train model, emit bitstream, deploy to board" toolchain.
The current focus is building a reliable temporal compiler foundation with
first-class state and delayed-edge semantics before scaling to complex models.

The repo uses a `src/` layout on disk, but `src` is not part of the public
import path. For IR-facing code, prefer package imports such as
`tempo_dag.ir` rather than anything under `src/...`.

## Current Repo State

What exists today:

- A typed IR centered on `Value`, `Operator`, `Graph`, and `OperatorRegistry`
- Built-in primitive operators with validation, coarse FPGA cost estimates, and
  HLS templates under `hls/operators/`
- ONNX parsing plus PyTorch and TensorFlow wrappers that export through ONNX
- Quantization config utilities and representative-dataset calibration helpers
- FPGA device presets and override handling under `configs/devices/`
- A test and lint baseline driven by `pytest` and `ruff`

What is still future work:

- Lowering high-level sequence models into primitive IR subgraphs
- Scheduling, fusion, memory planning, and hardware-oriented graph transforms
- Richer code generation beyond per-operator HLS template rendering
- Packaging, CLI workflows, and end-to-end deployment automation
- Benchmarking and validation on real FPGA targets

## Repository Layout

```text
.
|-- src/tempo_dag/
|   |-- calibration/     Representative dataset sampling and statistics
|   |-- codegen/hls/     Template resolution and HLS rendering
|   |-- device/          FPGA device schemas and preset registry
|   |-- ir/              Graph, values, operators, validation, registry
|   |-- ops/             Built-in primitive operators
|   `-- parsers/         ONNX parser plus PyTorch/TensorFlow wrappers
|-- tests/               Unit and integration coverage
|-- configs/devices/     Example FPGA board definitions
|-- hls/operators/       Operator-level HLS templates
|-- docs/                Architecture, calibration, development, roadmap
`-- scripts/             Git-hook setup helpers
```

## Quick Start

This repo currently uses `requirements.txt` rather than an installable package
workflow. Python 3.12 is the intended development target.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Enable the project Git hooks if you want the repository hook path configured
locally:

Windows:

```powershell
scripts\setup-hooks.ps1
```

Linux/macOS:

```bash
./scripts/setup-hooks.sh
```

## Verification

Run the same core checks used during development:

```powershell
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m pytest -q
```

## Documentation

- [Documentation Index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Calibration Guide](docs/calibration.md)
- [Development Guide](docs/development.md)
- [Environment Setup](docs/environment-setup.md)
- [Roadmap](docs/roadmap.md)

## Future Efforts

The next meaningful steps for TempoDAG are about turning the current compiler
foundation into a more complete hardware flow:

1. Lower LSTM, GRU, and related sequence layers into primitive operator graphs.
2. Add graph transforms for fusion, scheduling, and memory reuse.
3. Tighten quantization and calibration around deployment-oriented metrics.
4. Expand code generation from isolated operator templates to graph-level
   hardware emission and toolchain integration.
5. Package the workflow with clearer CLI, benchmarking, and board-validation
   stories.

That means the repo is already useful for experimenting with IR, parsing,
operator coverage, and calibration, while still being honest about the larger
compiler and deployment work that remains ahead.
