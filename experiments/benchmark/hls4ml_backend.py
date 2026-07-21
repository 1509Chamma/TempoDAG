"""hls4ml column for the streaming benchmark (runs in .venv-hls4ml).

Honest comparison design (library-positioning.md claim C1):
- hls4ml's mature flow compiles fixed-length SEQUENCE models (window in,
  result out). It has no stateful per-sample streaming cell, so for a
  streaming workload every new sample requires re-processing the window.
- We therefore convert a Keras GRU over a T=32 window (same H=16 as the
  benchmark), synthesize it, and report BOTH per-window numbers and the
  effective per-sample rate under streaming (one window per new sample).
- The `statistical` architecture (EWMA recurrences) is not a neural network
  and cannot be expressed in hls4ml at all -> documented unsupported row.
- hls4ml 2026.x note: it drives the removed classic `vitis_hls` binary; we
  therefore take its GENERATED HLS and synthesize through the same unified
  v++ flow used for the TempoDAG column (identical part, clock, toolchain).

    .venv-hls4ml\\Scripts\\python experiments\\benchmark\\hls4ml_backend.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCH = (sys.argv[1] if len(sys.argv) > 1 else "gru").lower()
T_WIN_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 32
WORKSPACE = (Path(os.environ.get("HLS4ML_WORKSPACE", "C:/tmp/hls4ml_bench"))
             / f"{ARCH}_w{T_WIN_ARG}")
VITIS_BIN = Path(os.environ.get("VITIS_BIN", "C:/AMDDesignTools/2026.1/Vitis/bin"))
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 5.0
T_WIN, H, FEATURES = T_WIN_ARG, 16, 4
SEED = 0


def build_and_convert() -> Path:
    import hls4ml
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(T_WIN, FEATURES)),
        ({"lstm": tf.keras.layers.LSTM, "rnn": tf.keras.layers.SimpleRNN}
         .get(ARCH, tf.keras.layers.GRU))(H, name=ARCH),
        tf.keras.layers.Dense(1, name="head"),
    ])
    model.build((None, T_WIN, FEATURES))

    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vitis",
        default_precision="ap_fixed<16,6>",
    )
    out_dir = WORKSPACE / "prj"
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend="Vitis",
        output_dir=str(out_dir), part=PART,
        clock_period=CLOCK_NS, io_type="io_stream",
    )
    hls_model.write()
    print(f"hls4ml conversion OK -> {out_dir}")
    return out_dir


def synthesize(prj: Path) -> dict:
    """Synthesize hls4ml's generated firmware through the unified v++ flow."""
    fw = prj / "firmware"
    srcs = sorted(str(p.as_posix()) for p in fw.glob("*.cpp"))
    cfg = WORKSPACE / "hls.cfg"
    syn_files = "\n".join(f"syn.file={s}" for s in srcs)
    cfg.write_text(
        f"part={PART}\n\n[hls]\nflow_target=vivado\n{syn_files}\n"
        f"syn.top=myproject\n"
        # NOTE: do NOT add -I firmware/ap_types — hls4ml bundles open-source
        # AP headers that refuse synthesis; Vitis supplies the real ones.
        f"syn.cflags=-I{fw.as_posix()} -I{(fw / 'nnet_utils').as_posix()}\n"
        f"clock={CLOCK_NS}ns\n", encoding="utf-8")
    cmd = [str(VITIS_BIN / "v++.bat"), "-c", "--mode", "hls",
           "--config", str(cfg), "--work_dir", str(WORKSPACE / "work")]
    print("[synth]", " ".join(Path(c).name if "/" in c or chr(92) in c else c
                              for c in cmd))
    proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True,
                          timeout=3600)
    (WORKSPACE / "synth.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    print("\n".join((proc.stdout or "").splitlines()[-6:]))
    if proc.returncode != 0:
        return {"status": "synth_failed", "log": str(WORKSPACE / "synth.log")}

    rpt_dir = WORKSPACE / "work" / "hls" / "syn" / "report"
    rpt = (rpt_dir / "myproject_csynth.rpt").read_text(encoding="utf-8")
    ii = latency = None
    for line in rpt.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and cells[1].isdigit():
            latency, ii = int(cells[1]), int(cells[5])
            break
    return {"status": "ok", "latency_cycles": latency, "ii_window": ii}


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        prj = build_and_convert()
        res = synthesize(prj)
    except Exception as exc:
        res = {"status": "convert_failed",
               "reason": f"{type(exc).__name__}: {exc}"}

    if res.get("status") == "ok":
        win_ns = res["latency_cycles"] * CLOCK_NS
        # streaming: each new sample needs one full window pass (no state)
        per_sample_ns = max(res["ii_window"], 1) * CLOCK_NS
        rows.append({
            "arch": ARCH,
            "backend": "hls4ml" if T_WIN == 32 else f"hls4ml_w{T_WIN}",
            "dataset": "synth_finance", "status": "ok",
            "median_ns": per_sample_ns, "p99_ns": per_sample_ns,
            "samples_per_sec": 1e9 / per_sample_ns,
            "latency_cycles": res["latency_cycles"],
            "ii_window": res["ii_window"], "clock_ns": CLOCK_NS,
            "window": T_WIN, "precision": "ap_fixed<16,6>",
            "note": (
                f"sequence-mode {ARCH.upper()} (no stateful streaming cell in hls4ml): "
                f"one T={T_WIN} window per new sample; per-window latency "
                f"{win_ns:.0f} ns; parity vs float reference not asserted "
                "(fixed-point model, different numerics - documented)"),
        })
    else:
        rows.append({"arch": ARCH, "backend": "hls4ml",
                     "dataset": "synth_finance", "status": "unsupported",
                     "reason": json.dumps(res)})
    rows.append({
        "arch": "statistical", "backend": "hls4ml",
        "dataset": "synth_finance", "status": "unsupported",
        "reason": "partially expressible only: EWMA/EW-var could be faked as "
                  "linear SimpleRNNs (sequence mode), but the z-score needs "
                  "sqrt+divide (no supported layer/merge) and state does not "
                  "persist across invocations (no streaming cell). Full "
                  "pipeline not compilable as a streaming process; the "
                  "extension API would mean hand-writing the HLS ourselves. "
                  "(attempted 2026-07; library-positioning.md C1)",
    })
    out = HERE / "results" / "results.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            row["timestamp"] = time.time()
            fh.write(json.dumps(row, default=str) + "\n")
    print("\nrows appended:")
    for row in rows:
        print(" ", row["arch"], row["backend"], row["status"],
              row.get("median_ns", row.get("reason", "")))


if __name__ == "__main__":
    sys.exit(main())
