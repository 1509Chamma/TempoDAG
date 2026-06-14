# Contributing To TempoDAG

TempoDAG is early-stage compiler infrastructure for stateful temporal dataflow
and FPGA-oriented HLS generation. Contributions are welcome, especially when
they keep the project testable, documented, and aligned with the roadmap.

## Project Priorities

Current work is guided by the milestone plan in [docs/roadmap.md](docs/roadmap.md):

- Precise temporal execution semantics and HLS hardware contracts.
- Graph-level HLS artifact generation for representative streaming pipelines.
- Scheduling, cost modeling, temporal graph optimization, and HLS directive
  optimization.
- Fixed-point, HLS C simulation, and eventually RTL or board-level parity.

Please avoid broad model-zoo expansion unless it supports one of those
milestones.

## Development Setup

TempoDAG currently targets Python 3.12.

```powershell
py -3.12 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
scripts\setup-hooks.ps1
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./scripts/setup-hooks.sh
```

## Local Checks

Before opening a pull request, run:

```bash
python -m ruff check src tests
python -m black --check .
python -m pytest -q
```

If a change touches generated HLS, temporal IR, or verification behavior, add
or update a golden-trace or parity test.

## Pull Request Expectations

- Keep PRs focused around one feature, fix, or documentation update.
- Include tests for behavior changes.
- Update docs when public APIs, roadmap claims, examples, or workflows change.
- Preserve fixed-point and temporal semantics unless the PR explicitly changes
  the contract and updates the relevant tests.
- Do not commit private research notes, local artifacts, generated build
  products, or board-tool output.

## Code Style

- Use typed dataclasses and explicit validation for IR objects.
- Keep graph rewrites parameter-preserving: learned weights must remain named
  constants, state, or immutable parameter blocks.
- Prefer small, inspectable compiler passes over clever monoliths.
- Keep HLS generation deterministic so generated artifacts are diff-friendly.

## Reporting Issues

Use GitHub issues for bugs, roadmap tasks, and design proposals. When reporting
a bug, include the command, expected behavior, actual behavior, Python version,
and any relevant trace or generated artifact.
