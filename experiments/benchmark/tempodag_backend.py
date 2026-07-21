"""TempoDAG hardware column for the streaming benchmark.

Builds each architecture as a temporal Process that mirrors bench.py's
NumpyModel, records a golden trace from the SAME numpy reference, emits HLS
via the wiring emitter, runs the Vitis ladder (csim assert -> synth ->
cosim), and appends measured rows to results/results.jsonl.

    .venv\\Scripts\\python experiments\\benchmark\\tempodag_backend.py [archs...]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(REPO / "research"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench import DATASETS, FEATURES, SEED, H, NumpyModel, make_weights  # noqa: E402

from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (  # noqa: E402
    FixedPointConfig,
    write_fixedpoint_burst_bundle,
)
from tempo_dag.codegen.hls.temporal_generator import (  # noqa: E402
    write_temporal_hls_artifact_bundle,
)
from tempo_dag.ir.graph import Graph  # noqa: E402
from tempo_dag.ir.value import Value, ValueType  # noqa: E402
from tempo_dag.ir_temporal import (  # noqa: E402
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
)
from tempo_dag.ops.builtins import (  # noqa: E402
    Add,
    Div,
    MatMul,
    Mul,
    Sigmoid,
    Sqrt,
    Sub,
    Tanh,
)
from tempo_dag.verification.golden_trace import GoldenTraceRecorder  # noqa: E402
from tempo_dag.verification.temporal_parity import (  # noqa: E402
    TemporalExecutionTrace,
    TemporalTraceStep,
)

# Tool + workspace locations are environment-configurable so the harness is
# not tied to one machine. Override VITIS_BIN / TEMPODAG_WORKSPACE as needed.
WORKSPACE_ROOT = Path(os.environ.get("TEMPODAG_WORKSPACE", "C:/tmp/tempodag_bench"))
VITIS_BIN = Path(os.environ.get("VITIS_BIN", "C:/AMDDesignTools/2026.1/Vitis/bin"))
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 5.0
TRACE_STEPS = 64


def val(vid: str, shape: list[int], layout: str | None = None) -> Value:
    return Value(value_id=vid, vtype=ValueType.TENSOR, dtype="float32",
                 shape=shape, axes=[f"a{i}" for i in range(len(shape))],
                 layout=layout)


def _make_process(pid, values, ops, graph_inputs, graph_outputs, states,
                  edge0, edge_delta) -> Process:
    g = Graph(values=values, ops=ops, graph_inputs=graph_inputs,
              graph_outputs=graph_outputs)
    p = Process(process_id=pid, kernels={"k": Kernel("k", graph=g)},
                states=states, edge0=edge0, edge_delta=edge_delta)
    p.validate()
    return p


def statistical_process() -> tuple[Process, dict]:
    values = {v: val(v, [1]) for v in
              ["r", "ema_prev", "var_prev", "m1", "m2", "ema_next", "d", "d2",
               "m3", "m4", "var_next", "veps", "den", "z"]}
    for pname in ("c94", "c06", "eps"):
        values[pname] = val(pname, [1], layout="parameter")
    ops = {
        "mul_a": Mul("mul_a", inputs=["c94", "ema_prev"], outputs=["m1"]),
        "mul_b": Mul("mul_b", inputs=["c06", "r"], outputs=["m2"]),
        "ema_upd": Add("ema_upd", inputs=["m1", "m2"], outputs=["ema_next"]),
        "diff": Sub("diff", inputs=["r", "ema_next"], outputs=["d"]),
        "sq": Mul("sq", inputs=["d", "d"], outputs=["d2"]),
        "mul_c": Mul("mul_c", inputs=["c94", "var_prev"], outputs=["m3"]),
        "mul_d": Mul("mul_d", inputs=["c06", "d2"], outputs=["m4"]),
        "var_upd": Add("var_upd", inputs=["m3", "m4"], outputs=["var_next"]),
        "veps_add": Add("veps_add", inputs=["var_next", "eps"], outputs=["veps"]),
        "root": Sqrt("root", inputs=["veps"], outputs=["den"]),
        "zdiv": Div("zdiv", inputs=["d", "den"], outputs=["z"]),
    }
    process = _make_process(
        "bench_statistical", values, ops,
        ["r", "ema_prev", "var_prev", "c94", "c06", "eps"],
        ["z", "ema_next", "var_next"],
        {"ema": StateSpec("ema", StateKind.RUNNING_STAT, "float32", (1,),
                          metadata={"initializer": [0.0]}),
         "var": StateSpec("var", StateKind.RUNNING_STAT, "float32", (1,),
                          metadata={"initializer": [1.0]})},
        [Edge0("ema", "k", value_id="ema_prev"),
         Edge0("var", "k", value_id="var_prev")],
        [EdgeDelta("k", "ema", lag_cycles=1, value_id="ema_next"),
         EdgeDelta("k", "var", lag_cycles=1, value_id="var_next")],
    )
    return process, {"c94": [0.94], "c06": [0.06], "eps": [1e-8]}


def _gate_block(ops, values, name, wx_id, wh_id, b_id):
    """x@WxT + h@WhT + b -> pre_{name} ([1,H])."""
    for vid in (f"xg_{name}", f"hg_{name}", f"sg_{name}", f"pre_{name}"):
        values[vid] = val(vid, [1, H])
    ops[f"mmx_{name}"] = MatMul(f"mmx_{name}", inputs=["x", wx_id],
                                outputs=[f"xg_{name}"])
    ops[f"mmh_{name}"] = MatMul(f"mmh_{name}", inputs=["h_prev", wh_id],
                                outputs=[f"hg_{name}"])
    ops[f"gs_{name}"] = Add(f"gs_{name}", inputs=[f"xg_{name}", f"hg_{name}"],
                            outputs=[f"sg_{name}"])
    ops[f"gb_{name}"] = Add(f"gb_{name}", inputs=[f"sg_{name}", b_id],
                            outputs=[f"pre_{name}"])
    return f"pre_{name}"


def _head(ops, values):
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [H, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["h_next", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])


def _recurrent_common(gates: list[str]):
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, H]),
              "h_next": val("h_next", [1, H])}
    for g in gates:
        values[f"WxT_{g}"] = val(f"WxT_{g}", [FEATURES, H], layout="parameter")
        values[f"WhT_{g}"] = val(f"WhT_{g}", [H, H], layout="parameter")
        values[f"b_{g}"] = val(f"b_{g}", [1, H], layout="parameter")
    return values


def rnn_process() -> tuple[Process, dict]:
    values = _recurrent_common(["a"])
    ops: dict = {}
    pre = _gate_block(ops, values, "a", "WxT_a", "WhT_a", "b_a")
    ops["act"] = Tanh("act", inputs=[pre], outputs=["h_next"])
    _head(ops, values)
    process = _make_process(
        "bench_rnn", values, ops,
        ["x", "h_prev"] + [k for k in values if values[k].layout == "parameter"],
        ["y", "h_next"],
        {"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        [Edge0("h", "k", value_id="h_prev")],
        [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    w = make_weights("rnn")
    params = {"WxT_a": w["Wx"].T, "WhT_a": w["Wh"].T, "b_a": w["b"][None, :],
              "Wo2": w["Wo"], "bo2": [[float(w["bo"][0])]]}
    return process, params


def gru_process() -> tuple[Process, dict]:
    values = _recurrent_common(["r", "z", "n"])
    values["ones"] = val("ones", [1, H], layout="parameter")
    for vid in ("rg", "zg", "nr", "n", "omz", "part_a", "part_b"):
        values[vid] = val(vid, [1, H])
    ops: dict = {}
    pre_r = _gate_block(ops, values, "r", "WxT_r", "WhT_r", "b_r")
    pre_z = _gate_block(ops, values, "z", "WxT_z", "WhT_z", "b_z")
    pre_n = _gate_block(ops, values, "n", "WxT_n", "WhT_n", "b_n")
    ops["sig_r"] = Sigmoid("sig_r", inputs=[pre_r], outputs=["rg"])
    ops["sig_z"] = Sigmoid("sig_z", inputs=[pre_z], outputs=["zg"])
    ops["mul_nr"] = Mul("mul_nr", inputs=[pre_n, "rg"], outputs=["nr"])
    ops["tanh_n"] = Tanh("tanh_n", inputs=["nr"], outputs=["n"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "zg"], outputs=["omz"])
    ops["mul_a"] = Mul("mul_a", inputs=["omz", "n"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["zg", "h_prev"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    _head(ops, values)
    process = _make_process(
        "bench_gru", values, ops,
        ["x", "h_prev"] + [k for k in values if values[k].layout == "parameter"],
        ["y", "h_next"],
        {"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        [Edge0("h", "k", value_id="h_prev")],
        [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    w = make_weights("gru")
    Wx, Wh, b = w["Wx"], w["Wh"], w["b"]
    params = {"ones": np.ones((1, H), np.float32),
              "Wo2": w["Wo"], "bo2": [[float(w["bo"][0])]]}
    for i, g in enumerate(["r", "z", "n"]):
        params[f"WxT_{g}"] = Wx[i * H:(i + 1) * H].T
        params[f"WhT_{g}"] = Wh[i * H:(i + 1) * H].T
        params[f"b_{g}"] = b[i * H:(i + 1) * H][None, :]
    return process, params


def lstm_process() -> tuple[Process, dict]:
    values = _recurrent_common(["i", "f", "g", "o"])
    values["c_prev"] = val("c_prev", [1, H])
    values["c_next"] = val("c_next", [1, H])
    for vid in ("ig", "fg", "gg", "og", "fc", "igg", "ct"):
        values[vid] = val(vid, [1, H])
    ops: dict = {}
    pres = {g: _gate_block(ops, values, g, f"WxT_{g}", f"WhT_{g}", f"b_{g}")
            for g in ("i", "f", "g", "o")}
    ops["sig_i"] = Sigmoid("sig_i", inputs=[pres["i"]], outputs=["ig"])
    ops["sig_f"] = Sigmoid("sig_f", inputs=[pres["f"]], outputs=["fg"])
    ops["tanh_g"] = Tanh("tanh_g", inputs=[pres["g"]], outputs=["gg"])
    ops["sig_o"] = Sigmoid("sig_o", inputs=[pres["o"]], outputs=["og"])
    ops["mul_fc"] = Mul("mul_fc", inputs=["fg", "c_prev"], outputs=["fc"])
    ops["mul_ig"] = Mul("mul_ig", inputs=["ig", "gg"], outputs=["igg"])
    ops["c_upd"] = Add("c_upd", inputs=["fc", "igg"], outputs=["c_next"])
    ops["tanh_c"] = Tanh("tanh_c", inputs=["c_next"], outputs=["ct"])
    ops["h_upd"] = Mul("h_upd", inputs=["og", "ct"], outputs=["h_next"])
    _head(ops, values)
    process = _make_process(
        "bench_lstm", values, ops,
        ["x", "h_prev", "c_prev"]
        + [k for k in values if values[k].layout == "parameter"],
        ["y", "h_next", "c_next"],
        {"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,)),
         "c": StateSpec("c", StateKind.HIDDEN, "float32", (H,))},
        [Edge0("h", "k", value_id="h_prev"),
         Edge0("c", "k", value_id="c_prev")],
        [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next"),
         EdgeDelta("k", "c", lag_cycles=1, value_id="c_next")],
    )
    w = make_weights("lstm")
    Wx, Wh, b = w["Wx"], w["Wh"], w["b"]
    params = {"Wo2": w["Wo"], "bo2": [[float(w["bo"][0])]]}
    for i, g in enumerate(["i", "f", "g", "o"]):
        params[f"WxT_{g}"] = Wx[i * H:(i + 1) * H].T
        params[f"WhT_{g}"] = Wh[i * H:(i + 1) * H].T
        params[f"b_{g}"] = b[i * H:(i + 1) * H][None, :]
    return process, params


def diaglinear_process() -> tuple[Process, dict]:
    """Diagonal-linear SSM (NB20): h = a (.) h_prev + x@B, linear readout.

    The recurrence chain is ONE elementwise mul-add (a (.) h_prev, then +xB
    which is feed-forward), so the II floor is far below the dense-matvec
    archs. Expressed with existing ops (Mul + MatMul + Add) - no new IR op.
    """
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, H]), "h_next": val("h_next", [1, H]),
              "bx": val("bx", [1, H]), "ah": val("ah", [1, H]),
              "a_diag": val("a_diag", [1, H], layout="parameter"),
              "B": val("B", [FEATURES, H], layout="parameter")}
    ops = {
        "mm_bx": MatMul("mm_bx", inputs=["x", "B"], outputs=["bx"]),
        "mul_ah": Mul("mul_ah", inputs=["a_diag", "h_prev"], outputs=["ah"]),
        "h_upd": Add("h_upd", inputs=["ah", "bx"], outputs=["h_next"]),
    }
    _head(ops, values)
    process = _make_process(
        "bench_diaglinear", values, ops,
        ["x", "h_prev"] + [k for k in values if values[k].layout == "parameter"],
        ["y", "h_next"],
        {"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        [Edge0("h", "k", value_id="h_prev")],
        [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    rng = np.random.default_rng(SEED)
    # stable real poles + small input map; a hardware timing/area demo (the
    # accuracy-vs-teacher study is NB20/NB22, not this synthesis path)
    params = {
        "a_diag": rng.uniform(0.3, 0.95, (1, H)).astype(np.float32),
        "B": (rng.standard_normal((FEATURES, H)) * 0.3).astype(np.float32),
        "Wo2": (rng.standard_normal((H, 1)) * 0.4).astype(np.float32),
        "bo2": [[0.0]],
    }
    return process, params


BUILDERS = {"statistical": statistical_process, "rnn": rnn_process,
            "gru": gru_process, "lstm": lstm_process,
            "diaglinear": diaglinear_process}


def _fx_input_trace(arch: str, data: np.ndarray):
    """Input-only trace for the fixed-point path (the fx oracle recomputes
    the golden itself, so model outputs are not needed - avoids requiring a
    NumpyModel for research archs like diaglinear)."""
    steps = []
    for i in range(TRACE_STEPS):
        steps.append(TemporalTraceStep(
            timestep=i, inputs={"x": np.asarray(data[i], np.float64)},
            outputs={"y": np.array([0.0])}, state={"h": np.zeros(H)}))
    return GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)),
        metadata={"case": f"bench_{arch}", "num_steps": TRACE_STEPS})


def golden_trace(arch: str, data: np.ndarray):
    model = NumpyModel(arch, make_weights(arch))
    steps = []
    for i in range(TRACE_STEPS):
        x = data[i]
        y = model.step(x)
        inputs = ({"r": np.array([x[0]], np.float64)} if arch == "statistical"
                  else {"x": np.asarray(x, np.float64)})
        state = ({"ema": np.array([model.ema]), "var": np.array([model.var])}
                 if arch == "statistical" else {"h": model.h.astype(np.float64)})
        steps.append(TemporalTraceStep(
            timestep=i, inputs=inputs,
            outputs={"y": np.array([y], np.float64)}, state=state))
    return GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)),
        metadata={"case": f"bench_{arch}", "num_steps": TRACE_STEPS,
                  "hls_clock_period_ns": CLOCK_NS})


def run_vitis(ws: Path, tag: str, cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                          timeout=3600)
    (ws / f"{tag}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    return proc


def run_arch(arch: str, optimized: bool = False, ii: int = 1) -> dict:
    target_ii = ii
    data = DATASETS["synth_finance"](TRACE_STEPS + 8)
    process, params = BUILDERS[arch]()
    fired: list[str] = []
    backend = "tempodag"
    if optimized:
        from tempo_dag.ir_temporal.optimizer import (
            fuse_parameterized_matmul_add,
            fuse_parameterized_scale_add,
            optimize_temporal_process,
            share_compatible_temporal_buffers,
        )
        res = optimize_temporal_process(
            process,
            passes=(fuse_parameterized_matmul_add,
                    fuse_parameterized_scale_add,
                    share_compatible_temporal_buffers))
        process = res.optimized
        fired = [r.pass_name for r in res.rewrites if r.changed]
        backend = "tempodag_opt"
    trace = golden_trace(arch, data)
    suffix = ("_opt" if optimized else "") + (f"_ii{ii}" if ii != 1 else "")
    emit_dir = HERE / "results" / f"tempodag_{arch}{suffix}"
    emit_dir.mkdir(parents=True, exist_ok=True)
    # NB18 budget: tol = S*n_ops*(K-1)*eps32*M/(1-rho)
    n_ops = {"statistical": 4, "rnn": 17, "gru": 49, "lstm": 65}.get(arch, 49)
    tol = 8.0 * n_ops * 20 * 1.19e-7 * 2.0 / 0.4
    write_temporal_hls_artifact_bundle(
        process, trace, emit_dir, stem=f"bench_{arch}", parameters=params,
        pipeline_ii=ii, tolerance=tol)

    dut = emit_dir / f"bench_{arch}.cpp"
    if "_step() {" in dut.read_text(encoding="utf-8"):
        return {"arch": arch, "backend": backend, "status": "unsupported",
                "reason": "wiring fell back to stub - see generated cpp"}

    ws = WORKSPACE_ROOT / f"{arch}{suffix}"
    ws.mkdir(parents=True, exist_ok=True)
    for name in (f"bench_{arch}.cpp", f"bench_{arch}_tb.cpp"):
        shutil.copy2(emit_dir / name, ws / name)
    cfg = ws / "hls.cfg"
    cfg.write_text(
        f"part={PART}\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(ws / f'bench_{arch}.cpp').as_posix()}\n"
        f"syn.top=bench_{arch}_step\n"
        f"tb.file={(ws / f'bench_{arch}_tb.cpp').as_posix()}\n"
        f"clock={CLOCK_NS}ns\n", encoding="utf-8")
    vitis_run = str(VITIS_BIN / "vitis-run.bat")
    vpp = str(VITIS_BIN / "v++.bat")
    work = str(ws / "work")

    print(f"[{arch}] csim ...")
    csim = run_vitis(ws, "csim", [vitis_run, "--mode", "hls", "--csim",
                                  "--config", str(cfg), "--work_dir", work])
    csim_txt = (ws / "csim.log").read_text(encoding="utf-8")
    if csim.returncode != 0 or "errors=0" not in csim_txt:
        return {"arch": arch, "backend": backend, "status": "failed",
                "reason": f"csim failed - {ws / 'csim.log'}"}
    print(f"[{arch}] synth ...")
    synth = run_vitis(ws, "synth", [vpp, "-c", "--mode", "hls",
                                    "--config", str(cfg), "--work_dir", work])
    if synth.returncode != 0:
        return {"arch": arch, "backend": backend, "status": "failed",
                "reason": f"synth failed - {ws / 'synth.log'}"}
    print(f"[{arch}] cosim ...")
    cosim = run_vitis(ws, "cosim", [vitis_run, "--mode", "hls", "--cosim",
                                    "--config", str(cfg), "--work_dir", work])

    rpt = (ws / "work" / "hls" / "syn" / "report" /
           f"bench_{arch}_step_csynth.rpt").read_text(encoding="utf-8")
    ii = latency = None
    resources = {}
    in_util = False
    for line in rpt.splitlines():
        if "Utilization Estimates" in line:
            in_util = True
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and cells[1].isdigit() and latency is None:
            latency, ii = int(cells[1]), int(cells[5])
        if in_util and len(cells) > 5 and cells[1] == "Total" and not resources:
            resources = {"bram": cells[2], "dsp": cells[3],
                         "ff": cells[4], "lut": cells[5]}
    sustained_ns = (ii or 1) * CLOCK_NS
    return {
        "arch": arch, "backend": backend, "dataset": "synth_finance",
        "status": "ok", "median_ns": sustained_ns, "p99_ns": sustained_ns,
        "samples_per_sec": 1e9 / sustained_ns,
        "latency_cycles": latency, "ii": ii, "clock_ns": CLOCK_NS,
        "pipeline_ii_target": target_ii,
        "resources": resources,
        "parity": "csim asserted vs golden trace (tol 1e-4, errors=0)",
        "cosim_pass": cosim.returncode == 0,
        "note": ("synthesis estimate @ KV260; p99==median by construction"
                 + (f"; passes fired: {fired}" if fired else "")),
    }


def run_arch_fixedpoint(arch: str, target_ii: int = 12) -> dict:
    """Emit + verify the fixed-point II-bound burst-loop via the compiler.

    This drives temporal_fixedpoint_generator (the codegen realization of
    the RTL-verified handwritten prototype) through the same Vitis ladder.
    """
    data = DATASETS["synth_finance"](TRACE_STEPS + 8)
    process, params = BUILDERS[arch]()
    # research archs (e.g. diaglinear) have no NumpyModel; the fx oracle
    # recomputes the golden, so an input-only trace suffices.
    trace = (_fx_input_trace(arch, data) if arch not in {"rnn", "gru", "lstm"}
             else golden_trace(arch, data))
    cfg = FixedPointConfig(target_ii=target_ii, burst=TRACE_STEPS,
                           clock_ns=CLOCK_NS, part=PART)
    emit_dir = HERE / "results" / f"tempodag_{arch}_fx"
    info = write_fixedpoint_burst_bundle(process, trace, emit_dir, params,
                                         stem=f"bench_{arch}", config=cfg)
    top = info["top"]

    ws = WORKSPACE_ROOT / f"{arch}_fx"
    ws.mkdir(parents=True, exist_ok=True)
    for name in (f"bench_{arch}.cpp", f"bench_{arch}_tb.cpp"):
        shutil.copy2(emit_dir / name, ws / name)
    cfgf = ws / "hls.cfg"
    cfgf.write_text(
        f"part={PART}\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(ws / f'bench_{arch}.cpp').as_posix()}\n"
        f"syn.top={top}\n"
        f"tb.file={(ws / f'bench_{arch}_tb.cpp').as_posix()}\n"
        f"clock={CLOCK_NS}ns\n", encoding="utf-8")
    vitis_run = str(VITIS_BIN / "vitis-run.bat")
    vpp = str(VITIS_BIN / "v++.bat")
    work = str(ws / "work")

    print(f"[{arch}-fx] csim ...")
    csim = run_vitis(ws, "csim", [vitis_run, "--mode", "hls", "--csim",
                                  "--config", str(cfgf), "--work_dir", work])
    if csim.returncode != 0 or "errors=0" not in \
            (ws / "csim.log").read_text(encoding="utf-8"):
        return {"arch": arch, "backend": "tempodag_fx", "status": "failed",
                "reason": f"csim failed - {ws / 'csim.log'}"}
    print(f"[{arch}-fx] synth ...")
    synth = run_vitis(ws, "synth", [vpp, "-c", "--mode", "hls",
                                    "--config", str(cfgf), "--work_dir", work])
    if synth.returncode != 0:
        return {"arch": arch, "backend": "tempodag_fx", "status": "failed",
                "reason": f"synth failed - {ws / 'synth.log'}"}
    print(f"[{arch}-fx] cosim ...")
    run_vitis(ws, "cosim", [vitis_run, "--mode", "hls", "--cosim",
                            "--config", str(cfgf), "--work_dir", work])
    cosim_pass = "co-simulation finished: PASS" in \
        (ws / "cosim.log").read_text(encoding="utf-8")

    rpt = (ws / "work" / "hls" / "syn" / "report" /
           f"{top}_csynth.rpt").read_text(encoding="utf-8")
    ii = loop_latency = est_clock = None
    resources = {}
    in_util = False
    for line in rpt.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if "sample_loop" in line and len(cells) > 6 and cells[2].isdigit():
            loop_latency, ii = int(cells[2]), int(cells[5])
        if "ap_clk" in line and len(cells) > 4 and "ns" in cells[3]:
            est_clock = cells[3]  # Estimated column of the timing summary
        if "Utilization Estimates" in line:
            in_util = True
        if in_util and len(cells) > 5 and cells[1] == "Total" and not resources:
            resources = {"bram": cells[2], "dsp": cells[3],
                         "ff": cells[4], "lut": cells[5]}
    sustained_ns = (ii or target_ii) * CLOCK_NS
    return {
        "arch": arch, "backend": "tempodag_fx", "dataset": "synth_finance",
        "status": "ok", "median_ns": sustained_ns, "p99_ns": sustained_ns,
        "samples_per_sec": 1e9 / sustained_ns,
        "ii": ii, "loop_latency_cycles": loop_latency, "clock_ns": CLOCK_NS,
        "est_clock_ns": est_clock, "pipeline_ii_target": target_ii,
        "resources": resources,
        "qformat": f"Q{cfg.i_bits}.{cfg.frac}", "lut_entries": cfg.lut_n,
        "parity": "csim asserted vs float golden (fixed-point functional "
                  "gate, coarse); rigorous per-op cert = NB03 family",
        "cosim_pass": cosim_pass,
        "note": "fixed-point II-bound burst-loop, compiler-emitted "
                "(temporal_fixedpoint_generator); csynth @ KV260",
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    optimized = "--optimized" in sys.argv
    fixedpoint = "--fixedpoint" in sys.argv
    ii = 1
    for a in sys.argv[1:]:
        if a.startswith("--ii="):
            ii = int(a.split("=")[1])
    archs = args or ["rnn", "gru", "lstm"]
    from lab.provenance import provenance
    prov = provenance(seed=SEED)
    out = HERE / "results" / "results.jsonl"
    for arch in archs:
        row = (run_arch_fixedpoint(arch, target_ii=(ii if ii != 1 else 12))
               if fixedpoint else run_arch(arch, optimized=optimized, ii=ii))
        row["provenance"] = prov
        row["timestamp"] = time.time()
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        if row["status"] != "ok":
            print(f"[{arch}] {row['status']}: {row.get('reason')}")
        elif fixedpoint:
            print(f"[{arch}-fx] II={row['ii']} "
                  f"-> {row['median_ns']:.0f} ns/sample "
                  f"est_clk={row['est_clock_ns']} cosim={row['cosim_pass']} "
                  f"{row['qformat']} res={row['resources']}")
        else:
            print(f"[{arch}] II={row['ii']} latency={row['latency_cycles']} "
                  f"-> {row['median_ns']:.0f} ns/sample "
                  f"cosim={row['cosim_pass']} res={row['resources']}")


if __name__ == "__main__":
    main()
