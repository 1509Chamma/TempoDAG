"""One-command Vitis proof harness for TempoDAG (unified 2024.2+/2026.x flow).

Stages (safe to re-run):

    python experiments/hls_proof/run_experiment.py --stage emit
    python experiments/hls_proof/run_experiment.py --stage csim
    python experiments/hls_proof/run_experiment.py --stage synth
    python experiments/hls_proof/run_experiment.py --stage cosim
    python experiments/hls_proof/run_experiment.py --stage all

`emit` needs no Vitis. The Vitis stages use the UNIFIED flow (`v++ --mode
hls` for synthesis, `vitis-run --mode hls --csim/--cosim` for simulation) —
the classic `vitis_hls` binary was removed in 2025.x/2026.x. Tool discovery:
PATH first, then TEMPO_VITIS_BIN env var, then the default 2026.1 install
location. Outputs land in results/ (gitignored); summary.json is the record.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(REPO / "research")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = HERE / "results"
DEMO = RESULTS / "demo"
TOP_FUNCTION = "temporal_demo_step"      # from the generated temporal_demo.cpp
PART = "xck26-sfvc784-2LV-c"             # Kria K26 (KV260)
CLOCK = "5ns"                            # 200 MHz
DEFAULT_VITIS_BIN = Path("C:/AMDDesignTools/2026.1/Vitis/bin")
# Vitis rejects paths containing spaces (HLS 200-2015); this repo lives under
# "Personal Projects", so all Vitis work happens in a space-free workspace.
WORKSPACE = Path(os.environ.get("TEMPO_HLS_WORKSPACE", "C:/tmp/tempodag_hls"))


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    for base in (os.environ.get("TEMPO_VITIS_BIN"), DEFAULT_VITIS_BIN):
        if not base:
            continue
        cand = Path(base) / f"{name}.bat"
        if cand.exists():
            return str(cand)
    print(f"{name} not found (PATH, TEMPO_VITIS_BIN, {DEFAULT_VITIS_BIN}).")
    print("Install AMD Vitis (see ENVIRONMENT.md), then re-run.")
    sys.exit(2)


def stage_emit() -> None:
    from lab.provenance import provenance
    from tempo_dag.examples.temporal_demo import run_demo

    DEMO.mkdir(parents=True, exist_ok=True)
    report = run_demo(output_dir=DEMO)
    manifest = json.loads((DEMO / "temporal_demo_manifest.json").read_text())
    summary = {
        "stage": "emit",
        "validation_passed": report.validation_passed,
        "max_output_abs_error": report.max_output_abs_error,
        "files": [f["path"] for f in manifest.get("files", [])],
        "provenance": provenance(config={"pipeline": "temporal_demo"}),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"emit OK: {len(summary['files'])} artifacts in {DEMO}")
    print(f"golden-trace validation passed: {report.validation_passed}")


def _artifact(kind: str) -> Path:
    manifest = json.loads((DEMO / "temporal_demo_manifest.json").read_text())
    for f in manifest.get("files", []):
        if f.get("kind") == kind:
            path = DEMO / Path(f["path"]).name
            if path.exists():
                return path
    print(f"artifact kind '{kind}' not found - run --stage emit first")
    sys.exit(1)


def _prepare_workspace() -> Path:
    """Copy DUT/TB into the space-free workspace and write hls.cfg there."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    dut = WORKSPACE / _artifact("process_hls").name
    tb = WORKSPACE / _artifact("testbench_hls").name
    shutil.copy2(_artifact("process_hls"), dut)
    shutil.copy2(_artifact("testbench_hls"), tb)
    cfg = WORKSPACE / "hls.cfg"
    cfg.write_text(
        f"part={PART}\n\n"
        f"[hls]\n"
        f"flow_target=vivado\n"
        f"syn.file={dut.as_posix()}\n"
        f"syn.top={TOP_FUNCTION}\n"
        f"tb.file={tb.as_posix()}\n"
        f"clock={CLOCK}\n",
        encoding="utf-8",
    )
    return cfg


def _run(tag: str, cmd: list[str]) -> None:
    print(f"[{tag}] {' '.join(Path(c).name if os.sep in c else c for c in cmd)}")
    env = dict(os.environ)
    lic = Path.home() / ".Xilinx" / "Xilinx.lic"
    if lic.exists():  # be explicit; default search proved unreliable on Windows
        env["XILINXD_LICENSE_FILE"] = str(lic)
        env["LM_LICENSE_FILE"] = str(lic)
    proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True,
                          timeout=3600, env=env)
    (RESULTS / f"{tag}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    tail = "\n".join((proc.stdout or "").splitlines()[-12:])
    print(tail)
    print(f"[{tag}] exit={proc.returncode}; full log -> results/{tag}.log")
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def stage_csim() -> None:
    cfg = _prepare_workspace()
    _run("csim", [_tool("vitis-run"), "--mode", "hls", "--csim",
                  "--config", str(cfg), "--work_dir",
                  str(WORKSPACE / "work")])


def stage_synth() -> None:
    cfg = _prepare_workspace()
    _run("synth", [_tool("v++"), "-c", "--mode", "hls",
                   "--config", str(cfg), "--work_dir",
                   str(WORKSPACE / "work")])


def stage_cosim() -> None:
    cfg = _prepare_workspace()
    _run("cosim", [_tool("vitis-run"), "--mode", "hls", "--cosim",
                   "--config", str(cfg), "--work_dir",
                   str(WORKSPACE / "work")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["emit", "csim", "synth", "cosim", "all"])
    stage = ap.parse_args().stage
    RESULTS.mkdir(parents=True, exist_ok=True)
    if stage in ("emit", "all"):
        stage_emit()
    if stage in ("csim", "all"):
        stage_csim()
    if stage in ("synth", "all"):
        stage_synth()
    if stage in ("cosim", "all"):
        stage_cosim()


if __name__ == "__main__":
    main()
