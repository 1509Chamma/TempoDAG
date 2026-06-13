# Development Guide

## Overview

TempoDAG is currently easiest to work on as a Python-first compiler project with
lightweight C++ scaffolding around it. Most day-to-day work happens in
`src/tempo_dag/`, `tests/`, `hls/operators/`, and `configs/devices/`.

Use [Environment Setup](environment-setup.md) first if you have not created a
local environment yet.

## Core Verification Commands

Run these before considering a change complete:

```powershell
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m pytest -q
```

The repo currently relies on:

- `ruff` for linting
- `pytest` for unit and integration coverage

## Git Hooks

The repository ships helper scripts that point Git at `.githooks/`:

Windows:

```powershell
scripts\setup-hooks.ps1
```

Linux/macOS:

```bash
./scripts/setup-hooks.sh
```

The current pre-commit behaviour is intentionally simple:

- It expects an active virtual environment
- It keeps `requirements.txt` aligned with the environment on commit

## Where To Make Changes

### IR and Operator Model

Use `src/tempo_dag/ir/` when working on:

- Graph structure
- Value metadata
- Operator contracts
- Registry behaviour
- Validation helpers

Use `src/tempo_dag/ir_temporal/` when working on:

- Streaming process structure
- Persistent state and bounded history buffers
- Same-timestep versus positive-lag temporal dependencies
- Validation that feedback cycles cross timestep boundaries

Use `src/tempo_dag/ops/` when changing or adding built-in primitive operators.

### Temporal Operator Patterns

Temporal operators should make state movement explicit instead of hiding it in
ordinary tensor edges.

When adding temporal operators:

- Keep pure same-timestep math inside `tempo_dag.ir.Graph` kernels.
- Represent feedback or recurrence with `EdgeDelta`, not an `Edge0` cycle.
- Model bounded history with `BufferSpec` before lowering it to implementation
  details such as shift registers or ring buffers.
- Model persistent values with `StateSpec` and choose the closest `StateKind`
  (`hidden_state`, `rolling_buffer`, or `running_stat`).
- Validate process structure before attaching scheduling, quantization, or HLS
  metadata.

This keeps the temporal layer analyzable while still reusing the existing
operator registry and same-timestep graph validation.

### Parsers

Use `src/tempo_dag/parsers/` when changing model ingestion. The repo currently
prefers a narrow parser stack:

- Direct ONNX support first
- PyTorch and TensorFlow wrappers second

If you add a new framework path, keeping ONNX as the normalized interchange
layer will reduce duplicated lowering logic.

### Calibration and Quantization

Use:

- `src/tempo_dag/calibration/` for representative dataset logic and statistics
- `src/tempo_dag/quantization_config.py` for quantization specs and attachment

Calibration code should stay honest about what is data-selection logic versus
what is full deployment calibration.

### HLS Templates

Use:

- `src/tempo_dag/codegen/hls/` for template resolution and rendering
- `hls/operators/` for the operator templates themselves

The current backend is template-driven and operator-scoped. If you introduce a
new operator, add both Python-side operator support and a matching HLS template.

### Device Presets

Use `src/tempo_dag/device/` and `configs/devices/` for hardware metadata. This
is the right place to extend board capabilities, memory information, or default
policy settings.

## Adding A New Operator

The current operator workflow is:

1. Implement or extend the operator class.
2. Register it with an `OperatorRegistry`.
3. Add validation rules and a coarse FPGA cost estimate.
4. Add or update the matching HLS template under `hls/operators/` or a
   module-local template path.
5. Add tests that cover validation, registry behaviour, and rendering.

This keeps the operator definition coherent across the IR and codegen layers.

## Adding Tests

The existing test suite mixes unit and integration coverage:

- `tests/unit/` for isolated behaviour
- `tests/integration/` for cross-module flows

Good changes usually add tests near the layer they touch most directly.

Examples:

- Parser mapping changes should add parser tests
- Operator validation changes should add operator tests
- Calibration heuristics should add representative-dataset tests

## Docker And Native Scaffolding

The repo includes:

- `Dockerfile` for a Python 3.12-based development image with common native
  tooling installed
- `CMakeLists.txt` and `include/` as minimal native scaffolding

These are helpful for future backend work, but the current center of gravity is
still the Python compiler code.

## Current Development Priorities

If you are deciding where to contribute next, the highest-leverage areas are:

1. Recurrent-model lowering into primitive operators
2. Graph transforms and hardware-aware optimization passes
3. Quantization and calibration integration
4. Broader codegen and validation coverage
5. Documentation and packaging polish

