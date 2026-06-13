# Environment Setup

## Python Version

Python 3.12 is the intended development target for this repository. That matches
the configured lint target and the provided Docker image.

## Local Virtual Environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Baseline Verification

After installing dependencies, verify the environment with:

```powershell
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m pytest -q
```

## Git Hooks

The helper scripts below tell Git to use the repository-managed hooks under
`.githooks/`.

Windows:

```powershell
scripts\setup-hooks.ps1
```

Linux/macOS:

```bash
./scripts/setup-hooks.sh
```

Once enabled, the current pre-commit flow:

- Expects an active virtual environment
- Keeps `requirements.txt` synchronized with the active environment

## Docker Option

If you want a reproducible containerized environment, the repository also ships
a `Dockerfile` based on `python:3.12-slim` with common build and analysis tools
installed.

Build:

```bash
docker build -t tempo_dag .
```

Run:

```bash
docker run --rm -it -v "$PWD:/repo" tempo_dag bash
```

## Notes

- The repo is currently driven by `requirements.txt`, not a packaged
  `pip install -e .` workflow.
- The Python code is the primary development surface today; the native
  scaffolding is present but still minimal.

