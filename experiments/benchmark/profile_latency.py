"""Latency-attribution profiler: where is the latency cap coming from?

Parses a Vitis csynth report and breaks the design's latency down by
instance and loop, ranking the contributors. This is the hardware analogue
of a flame graph: it names the op that caps the pipeline.

    .venv\\Scripts\\python experiments\\benchmark\\profile_latency.py <csynth.rpt> [...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

KNOWN_REPORTS = [
    ("demo (rolling_mean+conv+add)",
     Path("C:/tmp/tempodag_hls/work/hls/syn/report/temporal_demo_step_csynth.rpt")),
    ("bench statistical",
     Path("C:/tmp/tempodag_bench/statistical/work/hls/syn/report/"
          "bench_statistical_step_csynth.rpt"),
     ),
    ("bench statistical (flat ws)",
     Path("C:/tmp/tempodag_bench/work/hls/syn/report/"
          "bench_statistical_step_csynth.rpt")),
    ("hls4ml GRU (window)",
     Path("C:/tmp/hls4ml_bench/work/hls/syn/report/myproject_csynth.rpt")),
    ("bench rnn",
     Path("C:/tmp/tempodag_bench/rnn/work/hls/syn/report/"
          "bench_rnn_step_csynth.rpt")),
]


def parse_sections(text: str) -> dict:
    """Extract top latency/II plus per-instance and per-loop tables."""
    result: dict = {"top": {}, "instances": [], "loops": []}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and cells[1].isdigit() and not result["top"]:
            result["top"] = {"latency_cycles": int(cells[1]),
                             "ii": int(cells[5])}
    inst_block = re.search(
        r"\* Instance: \s*\n(.*?)(?:\n\s*\n|\* Loop)", text, re.S)
    if inst_block:
        for line in inst_block.group(1).splitlines():
            cells = [c.strip() for c in line.split("|")]
            # | Instance | Module | lat min | lat max | abs.. | II min | ...
            if len(cells) >= 8 and cells[1] and not cells[1].startswith("-") \
                    and cells[3].isdigit() and "Instance" not in cells[1]:
                result["instances"].append(
                    {"module": cells[1][:60], "latency": int(cells[4]),
                     "ii": int(cells[7]) if cells[7].isdigit() else None})
    loop_block = re.search(r"\* Loop: \s*\n(.*?)(?:\n\s*\n|\Z)", text, re.S)
    if loop_block:
        for line in loop_block.group(1).splitlines():
            cells = [c.strip() for c in line.split("|")]
            # | - Loop Name | lat min | lat max | Iter lat | achieved II | ...
            if len(cells) >= 7 and cells[1].startswith("- ") \
                    and cells[2].isdigit():
                result["loops"].append(
                    {"loop": cells[1][2:][:60], "latency": int(cells[3]),
                     "ii": cells[5]})
    return result


def profile(label: str, path: Path) -> dict | None:
    if not path.exists():
        return None
    parsed = parse_sections(path.read_text(encoding="utf-8", errors="replace"))
    top = parsed["top"]
    print(f"\n=== {label} ===")
    print(f"total latency {top.get('latency_cycles')} cycles, "
          f"II {top.get('ii')}")
    contributors = sorted(
        [{"kind": "instance", "name": i["module"], "latency": i["latency"]}
         for i in parsed["instances"]]
        + [{"kind": "loop", "name": ln["loop"], "latency": ln["latency"]}
           for ln in parsed["loops"]],
        key=lambda r: -r["latency"])[:8]
    total = max(1, top.get("latency_cycles", 1))
    for c in contributors:
        share = 100.0 * c["latency"] / total
        print(f"  {c['latency']:>6} cyc ({share:5.1f}%)  "
              f"[{c['kind']}] {c['name']}")
    if not contributors:
        print("  (no instance/loop breakdown in report - fully inlined; "
              "latency dominated by the flat datapath: float add ~4-7 cyc, "
              "mul ~3-4, div ~12-15, sqrt ~12-16 chained by dependency)")
    return {"label": label, "top": top, "contributors": contributors}


def main() -> None:
    targets = ([(Path(a).stem, Path(a)) for a in sys.argv[1:]]
               if len(sys.argv) > 1 else KNOWN_REPORTS)
    out = []
    for label, path in targets:
        row = profile(label, path)
        if row:
            out.append(row)
    dest = HERE / "results" / "latency_profile.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
