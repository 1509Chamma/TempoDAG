"""Reproducibility helpers shared by every research notebook.

Each experiment stamps a provenance block (git SHA, tool versions, seed,
config hash, timestamp) into its result artifact so every number we report is
traceable to the exact commit and configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - provenance must never crash a run
        return "unknown"


def git_sha() -> str:
    return _git("rev-parse", "HEAD")


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def git_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status) and status != "unknown"


def set_seed(seed: int = 0) -> None:
    """Pin every RNG we might touch so runs are byte-reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for pure-numpy experiments
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def provenance(seed: int | None = None, config: dict[str, Any] | None = None) -> dict:
    config = config or {}
    return {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": seed,
        "config": config,
        "config_hash": _config_hash(config),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_result(
    name: str,
    data: dict[str, Any],
    results_dir: Path | str,
    *,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write ``results/<name>.json`` with a provenance block. Returns the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "provenance": provenance(seed=seed, config=config),
        "result": data,
    }
    path = results_dir / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path
