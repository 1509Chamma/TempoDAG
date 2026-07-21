"""Board-constrained analysis for the streaming benchmark.

Two tools, both driven by the repo's device presets (configs/devices/*.json —
add a JSON to support a new board; KV260 is the competition target):

1. `check_fit(resources, device)` — utilization of a synthesized design
   against a board's budget (annotates measured rows).
2. `project_model(...)` — analytic feasibility of LARGE architectures
   (PatchTST / Crossformer / big transformers) under a board's constraints:
   on-chip weight residency, external-bandwidth demand at a target rate, and
   the DSP-bound throughput ceiling. This answers "millions of parameters"
   HONESTLY: not synthesized, but bounded by physics the board JSON declares.

    .venv\\Scripts\\python experiments\\benchmark\\board_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from tempo_dag.device.registry import DeviceRegistry  # noqa: E402

REGISTRY = DeviceRegistry(str(REPO / "configs" / "devices"))


def load_board(name: str) -> dict:
    return REGISTRY.get_preset(name)


def check_fit(used: dict, board_name: str) -> dict:
    """used: {'dsp': int, 'bram': int(18k blocks), 'lut': int, 'ff': int}."""
    board = load_board(board_name)
    res = board["resources"]
    budget = {"dsp": res.get("dsps", 0), "lut": res.get("luts", 0),
              "ff": res.get("ffs", 0),
              "bram": res.get("bram_18k", 0) or 2 * res.get("bram_36k", 0)}
    util = {}
    fits = True
    for key, avail in budget.items():
        u = int(used.get(key, 0) or 0)
        pct = (100.0 * u / avail) if avail else 0.0
        util[key] = {"used": u, "available": avail, "pct": round(pct, 1)}
        if avail and u > avail:
            fits = False
    return {"board": board_name, "fits": fits, "utilization": util}


def project_model(name: str, params_m: float, macs_per_sample_m: float,
                  board_name: str, precision_bits: int = 16) -> dict:
    """Analytic bound for a large model on a board (NOT a synthesis result).

    Assumes weights stream from external memory once per sample when they
    exceed on-chip capacity (the streaming-inference worst case; weight reuse
    across batch does not exist at batch=1).
    """
    board = load_board(board_name)
    res, mem = board["resources"], board["memory"]
    clock_hz = board["policies"]["target_clock_mhz"] * 1e6

    weight_mb = params_m * 1e6 * (precision_bits / 8) / 1e6
    onchip_mb = mem["on_chip_kb"] / 1024.0
    resident = weight_mb <= 0.8 * onchip_mb  # leave 20% for activations/state

    dsps = res.get("dsps", 1)
    macs_per_dsp = 2 if precision_bits <= 16 else 1
    compute_sps = (dsps * macs_per_dsp * clock_hz) / (macs_per_sample_m * 1e6)

    if resident:
        mem_sps = float("inf")
    else:
        bytes_per_sample = params_m * 1e6 * (precision_bits / 8)
        bw_avail = mem["external_bandwidth_gbps"] / 8 * 1e9  # bits->bytes/s
        mem_sps = bw_avail / bytes_per_sample

    achievable = min(compute_sps, mem_sps)
    binding = ("on-chip resident; DSP-bound" if resident
               else ("memory-bandwidth-bound" if mem_sps < compute_sps
                     else "DSP-bound (weights streamed)"))
    return {
        "model": name, "board": board_name, "params_M": params_m,
        "precision_bits": precision_bits,
        "weights_MB": round(weight_mb, 1), "onchip_MB": round(onchip_mb, 1),
        "weights_resident": resident,
        "compute_bound_sps": f"{compute_sps:,.0f}",
        "memory_bound_sps": ("inf" if mem_sps == float("inf")
                             else f"{mem_sps:,.0f}"),
        "achievable_sps": f"{achievable:,.0f}",
        "binding_constraint": binding,
        "meets_1M_sps": achievable >= 1e6,
        "note": "analytic projection from board JSON; not synthesized",
    }


LARGE_MODELS = [
    # (name, params_M, MACs/sample_M) - representative published configs;
    # MACs/sample for streaming single-step inference with cached context.
    ("transformer_tiny (ours H=16)", 0.002, 0.004),
    ("PatchTST-base (L=336)", 1.2, 2.4),
    ("Crossformer-base", 3.5, 7.0),
    ("PatchTST-large", 6.9, 13.8),
    ("small LLM-style (0.1B)", 100.0, 200.0),
]


def main() -> None:
    boards = ["xilinx_kv260", "xilinx_u250", "intel_s10mx"]
    rows = []
    print(f"{'model':>28} {'board':>13} {'wMB':>7} {'res?':>5} "
          f"{'achievable sps':>15}  binding")
    for name, params_m, macs_m in LARGE_MODELS:
        for board in boards:
            r = project_model(name, params_m, macs_m, board)
            rows.append(r)
            print(f"{name:>28} {board:>13} {r['weights_MB']:>7} "
                  f"{str(r['weights_resident']):>5} {r['achievable_sps']:>15}"
                  f"  {r['binding_constraint']}")
    out = HERE / "results" / "board_projections.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
