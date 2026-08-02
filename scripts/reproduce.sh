#!/usr/bin/env bash
# Reproduce the TempoDAG results on any machine.
#
#   scripts/reproduce.sh              # everything that needs only Python
#   scripts/reproduce.sh tests        # test suite + lint
#   scripts/reproduce.sh walkthroughs # the three research walkthroughs
#   scripts/reproduce.sh research     # cost-model predictions from the IR
#   scripts/reproduce.sh tutorial     # train a GRU -> parity proof -> emitted HLS
#   scripts/reproduce.sh emit         # emit the HLS demo bundle (no Vitis needed)
#   scripts/reproduce.sh hls          # the Vitis ladder (needs AMD Vitis on the host)
#
# Every stage is deterministic: seeds are pinned in the scripts themselves.
# The one thing this script cannot make machine-agnostic is AMD Vitis - it
# is proprietary and ~100 GB, so the `hls` stage runs only where Vitis is
# installed and says so plainly where it is not. Everything else runs
# anywhere, including inside the provided Docker image:
#
#   docker build -t tempodag .
#   docker run --rm tempodag                 # = scripts/reproduce.sh
#   docker run --rm tempodag tests           # a single stage

set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
PY="${PYTHON:-python}"

banner() { echo; echo "=== $1 ==================================================="; }

stage_tests() {
  banner "tests: pytest + ruff"
  "$PY" -m pytest -q
  "$PY" -m ruff check src tests
}

stage_walkthroughs() {
  banner "walkthroughs: accuracy / speed / attention (re-executing notebooks)"
  for nb in research/walkthrough/*.ipynb; do
    "$PY" -m nbconvert --to notebook --execute --inplace "$nb"
  done
}

stage_research() {
  banner "research: cost-model predictions (static, no FPGA tools)"
  "$PY" research/cost_model_validation.py --predict-only
}

stage_tutorial() {
  banner "tutorial: trained PyTorch GRU -> parity proof -> emitted HLS"
  "$PY" tutorials/01_pytorch_gru_to_rtl.py
}

stage_emit() {
  banner "emit: HLS demo bundle (csim-ready, Vitis not required to emit)"
  "$PY" experiments/hls_proof/run_experiment.py --stage emit
}

stage_hls() {
  banner "hls: Vitis verification ladder"
  if command -v v++ >/dev/null 2>&1 || [ -n "${VITIS_BIN:-}" ]; then
    "$PY" experiments/hls_proof/run_experiment.py --stage all
  else
    echo "AMD Vitis was not found on this machine, so the hardware rungs"
    echo "(csim / synthesis / RTL co-simulation) cannot run here. This is the"
    echo "one deliberately non-portable step: Vitis is proprietary and large."
    echo "To run it: install Vitis (see experiments/hls_proof/ENVIRONMENT.md),"
    echo "set VITIS_BIN, and re-run: scripts/reproduce.sh hls"
    echo "Reference results produced by that ladder are in docs/benchmarks.md."
  fi
}

case "${1:-all}" in
  tests)        stage_tests ;;
  walkthroughs) stage_walkthroughs ;;
  research)     stage_research ;;
  tutorial)     stage_tutorial ;;
  emit)         stage_emit ;;
  hls)          stage_hls ;;
  all)
    stage_tests
    stage_walkthroughs
    stage_research
    stage_tutorial
    stage_emit
    echo
    echo "All Python-side results reproduced. For the hardware rungs run:"
    echo "  scripts/reproduce.sh hls   (needs AMD Vitis)"
    ;;
  *)
    echo "unknown stage: $1 (expected tests|walkthroughs|research|tutorial|emit|hls|all)"
    exit 2 ;;
esac
