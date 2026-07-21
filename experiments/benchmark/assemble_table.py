"""Assemble results/results.jsonl into THE benchmark table (markdown + print).

Adds board-fit annotation (board_model.check_fit) to measured hardware rows.

    .venv\\Scripts\\python experiments\\benchmark\\assemble_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

from board_model import check_fit

HERE = Path(__file__).resolve().parent
ARCHS = ["statistical", "rnn", "gru", "lstm", "transformer"]
BACKENDS = ["numpy", "torch_cpu", "torch_compile_cpu", "torch_cuda",
            "hls4ml", "tempodag", "tempodag_opt"]


def fmt_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    if row["status"] == "ok":
        us = row["median_ns"] / 1000.0
        cell = f"{us:.3f} µs" if us < 1 else f"{us:.2f} µs"
        if row.get("ii"):
            cell += f" (II={row['ii']})"
        if row.get("resources"):
            fit = check_fit(
                {k: int(v) for k, v in row["resources"].items()
                 if str(v).isdigit()}, "xilinx_kv260")
            worst = max(v["pct"] for v in fit["utilization"].values())
            cell += f" [{'fits' if fit['fits'] else 'OVER'} {worst:.0f}%]"
        return cell
    reason = row.get("reason", "unsupported")
    short = {"no CUDA device": "no GPU",
             }.get(reason, None)
    if short:
        return short
    if "MSVC" in reason or "Inductor" in reason:
        return "no MSVC"
    if "pending" in reason:
        return "pending"
    if "expressible" in reason or "streaming" in reason:
        return "not expressible*"
    return "unsupported*"


def main() -> None:
    src = HERE / "results" / "results.jsonl"
    rows = [json.loads(line) for line in src.open(encoding="utf-8")]
    latest: dict = {}
    for row in rows:
        latest[(row["arch"], row["backend"])] = row  # last write wins

    header = ["arch"] + BACKENDS
    lines_md = ["| " + " | ".join(header) + " |",
                "|" + "---|" * len(header)]
    print(f"{'arch':>12} | " + " | ".join(f"{b:>16}" for b in BACKENDS))
    for arch in ARCHS:
        cells = [fmt_cell(latest.get((arch, b))) for b in BACKENDS]
        print(f"{arch:>12} | " + " | ".join(f"{c:>16}" for c in cells))
        lines_md.append("| " + " | ".join([arch, *cells]) + " |")
    lines_md.append("")
    lines_md.append("*full reasons recorded per-row in results.jsonl; "
                    "[fits N%] = worst-resource utilization vs KV260 budget")
    (HERE / "results" / "TABLE.md").write_text("\n".join(lines_md),
                                               encoding="utf-8")
    print(f"\nmarkdown -> {HERE / 'results' / 'TABLE.md'}")


if __name__ == "__main__":
    main()
