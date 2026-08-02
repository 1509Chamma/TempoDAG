"""Cost-model validation campaign: predictions first, synthesis second.

The predictive cost model so far rests on four anchors (RNN/GRU/LSTM/diag-SSM
at H=16). This campaign widens the evidence in two directions:

  1. CELL VARIANTS at H=16 -- MGU and UGRNN (2-gate cells), IndRNN (elementwise
     recurrence + activation: a predicted *intermediate* II class the binary
     model doesn't cover), and reset-before GRU (identical parameter count to
     the anchor GRU but a longer state chain -- the sharpest possible test of
     "cost tracks state structure, not model size").
  2. HIDDEN-SIZE SWEEP -- RNN/GRU/LSTM at H in {8, 32}, diagonal SSM at
     H in {8, 32, 64}, testing the refined scaling laws:
        II_hat(matmul-in-loop)  = 8 + ceil(log2 H)      (12 at H=16 anchors)
        II_hat(elementwise)     = 4                      (H-independent)
        DSP_hat                 = k_mac * weighted MACs  (k_mac from RNN@16)

Protocol (stated before any tool runs; predictions are written to the results
file BEFORE synthesis so they cannot be tuned afterwards):

  - Each design synthesizes with PIPELINE II target = its predicted II.
    achieved == target is consistent-with (feasibility at the predicted floor);
    achieved > target refutes. One probe run (RNN@16, target 8) tests tightness
    from below: the raw chain sum says the RNN floor may sit under the II=12
    the anchors were only ever *targeted* at.
  - DSP is always a held-out prediction: the single coefficient k_mac is
    calibrated on the RNN@16 anchor and never refitted.
  - Q6.12 numerics, 512-entry LUT, 64-sample burst, 5 ns clock, KV260 part --
    identical to the anchor runs. Weights are seeded and scaled ~1/sqrt(H) so
    larger H stays inside Q6.12 range.

Usage:
    python research/cost_model_validation.py --predict-only     # table, no Vitis
    python research/cost_model_validation.py --suite round1     # full campaign
    python research/cost_model_validation.py --only mgu16       # one entry
    python research/cost_model_validation.py --report           # table from results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (  # noqa: E402
    FixedPointConfig,
    write_fixedpoint_burst_bundle,
)
from tempo_dag.codegen.hls.temporal_generator import (  # noqa: E402
    _topological_ops,
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
    ReLU,
    Sigmoid,
    Sub,
    Tanh,
)
from tempo_dag.verification.golden_trace import GoldenTraceRecorder  # noqa: E402
from tempo_dag.verification.temporal_parity import (  # noqa: E402
    TemporalExecutionTrace,
    TemporalTraceStep,
)

SEED = 20260731
FEATURES = 4
TRACE_STEPS = 64
CLOCK_NS = 5.0
PART = "xck26-sfvc784-2LV-c"
WORKSPACE_ROOT = Path(os.environ.get("TEMPODAG_CMV_WORKSPACE", "C:/tmp/tempodag_cmv"))
VITIS_BIN = Path(os.environ.get("VITIS_BIN", "C:/AMDDesignTools/2026.1/Vitis/bin"))
RESULTS_DIR = HERE / "results"
RESULTS_FILE = RESULTS_DIR / "cost_model_validation.jsonl"

# Anchor measurements (H=16, Q6.12, RTL-cosim-verified; docs/benchmarks.md).
ANCHORS = {
    "rnn16": dict(II=12, DSP=289),
    "gru16": dict(II=12, DSP=871),
    "lstm16": dict(II=12, DSP=1153),
    "diag16": dict(II=4, DSP=87),
}

# Per-op loop-chain latency table (cycles). One global table, never per-arch.
# MatMul latency is 1 (parallel multiplies) + ceil(log2 K) (balanced adder
# tree over the K-wide reduction).
OP_LAT = {"Mul": 1, "Add": 1, "Sub": 1, "Div": 1, "Tanh": 2, "Sigmoid": 2,
          "ReLU": 1}


# --------------------------------------------------------------------------
# Builders (parameterized by H; same op vocabulary the anchor designs used)
# --------------------------------------------------------------------------

def val(vid, shape, layout=None):
    return Value(value_id=vid, vtype=ValueType.TENSOR, dtype="float32",
                 shape=shape, axes=[f"a{i}" for i in range(len(shape))],
                 layout=layout)


def _proc(pid, values, ops, gin, gout, states, e0, ed):
    g = Graph(values=values, ops=ops, graph_inputs=gin, graph_outputs=gout)
    p = Process(process_id=pid, kernels={"k": Kernel("k", graph=g)},
                states=states, edge0=e0, edge_delta=ed)
    p.validate()
    return p


def _rng(tag):
    # zlib.crc32, not hash(): Python string hashing is salted per process,
    # which would silently unpin the seeds across runs.
    import zlib
    return np.random.default_rng((SEED * 1000003 + zlib.crc32(tag.encode())) % (2**32))


def _params_of(values):
    return [k for k, v in values.items() if v.layout == "parameter"]


def _w(rng, shape, scale):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _stream_f(process):
    """Input feature width, read from the process itself."""
    return process.kernels["k"].graph.values["x"].shape[-1]


def _head(ops, values, h):
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [h, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["h_next", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])


def _gate(ops, values, name, h, reset_input="h_prev"):
    """x@WxT + <reset_input>@WhT + b -> pre_<name>."""
    for vid in (f"xg_{name}", f"hg_{name}", f"sg_{name}", f"pre_{name}"):
        values[vid] = val(vid, [1, h])
    ops[f"mmx_{name}"] = MatMul(f"mmx_{name}", inputs=["x", f"WxT_{name}"],
                                outputs=[f"xg_{name}"])
    ops[f"mmh_{name}"] = MatMul(f"mmh_{name}", inputs=[reset_input, f"WhT_{name}"],
                                outputs=[f"hg_{name}"])
    ops[f"gs_{name}"] = Add(f"gs_{name}", inputs=[f"xg_{name}", f"hg_{name}"],
                            outputs=[f"sg_{name}"])
    ops[f"gb_{name}"] = Add(f"gb_{name}", inputs=[f"sg_{name}", f"b_{name}"],
                            outputs=[f"pre_{name}"])
    return f"pre_{name}"


def _common(gates, h, f=FEATURES):
    values = {"x": val("x", [1, f]),
              "h_prev": val("h_prev", [1, h]),
              "h_next": val("h_next", [1, h])}
    for g in gates:
        values[f"WxT_{g}"] = val(f"WxT_{g}", [f, h], layout="parameter")
        values[f"WhT_{g}"] = val(f"WhT_{g}", [h, h], layout="parameter")
        values[f"b_{g}"] = val(f"b_{g}", [1, h], layout="parameter")
    return values


def _gate_params(rng, gates, h, f=FEATURES):
    sx, sh = 0.4 / math.sqrt(max(f / FEATURES, 1)), 0.9 / math.sqrt(h)
    p = {}
    for g in gates:
        p[f"WxT_{g}"] = _w(rng, (f, h), sx)
        p[f"WhT_{g}"] = _w(rng, (h, h), sh)
        p[f"b_{g}"] = _w(rng, (1, h), 0.1)
    p["Wo2"] = _w(rng, (h, 1), 0.4)
    p["bo2"] = [[0.0]]
    return p


def _hidden_state(h, extra=()):
    states = {"h": StateSpec("h", StateKind.HIDDEN, "float32", (h,))}
    e0 = [Edge0("h", "k", value_id="h_prev")]
    ed = [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")]
    for sid in extra:
        states[sid] = StateSpec(sid, StateKind.HIDDEN, "float32", (h,))
        e0.append(Edge0(sid, "k", value_id=f"{sid}_prev"))
        ed.append(EdgeDelta("k", sid, lag_cycles=1, value_id=f"{sid}_next"))
    return states, e0, ed


def build_rnn(h, f=FEATURES):
    values = _common(["a"], h, f)
    ops = {}
    pre = _gate(ops, values, "a", h)
    ops["act"] = Tanh("act", inputs=[pre], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_rnn{h}f{f}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    return p, {**_gate_params(_rng(f"rnn{h}f{f}"), ["a"], h, f)}


def build_gru(h, reset_before=False, f=FEATURES):
    gates = ["r", "z", "n"]
    values = _common(gates, h, f)
    values["ones"] = val("ones", [1, h], layout="parameter")
    for vid in ("rg", "zg", "n", "omz", "part_a", "part_b"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre_r = _gate(ops, values, "r", h)
    pre_z = _gate(ops, values, "z", h)
    ops["sig_r"] = Sigmoid("sig_r", inputs=[pre_r], outputs=["rg"])
    ops["sig_z"] = Sigmoid("sig_z", inputs=[pre_z], outputs=["zg"])
    if reset_before:
        # TF-style: the reset gate multiplies h BEFORE the candidate matmul,
        # so sigmoid -> mul -> matmul -> ... all sit in series on the loop.
        values["rh"] = val("rh", [1, h])
        ops["mul_rh"] = Mul("mul_rh", inputs=["rg", "h_prev"], outputs=["rh"])
        pre_n = _gate(ops, values, "n", h, reset_input="rh")
        ops["tanh_n"] = Tanh("tanh_n", inputs=[pre_n], outputs=["n"])
    else:
        # PyTorch-style (the anchor): reset applies after the matmul.
        values["nr"] = val("nr", [1, h])
        pre_n = _gate(ops, values, "n", h)
        ops["mul_nr"] = Mul("mul_nr", inputs=[pre_n, "rg"], outputs=["nr"])
        ops["tanh_n"] = Tanh("tanh_n", inputs=["nr"], outputs=["n"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "zg"], outputs=["omz"])
    ops["mul_a"] = Mul("mul_a", inputs=["omz", "n"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["zg", "h_prev"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    tag = f"gru{'rb' if reset_before else ''}{h}f{f}"
    p = _proc(f"cmv_{tag}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    params = _gate_params(_rng(tag), gates, h, f)
    params["ones"] = np.ones((1, h), np.float32)
    return p, params


def build_lstm(h):
    gates = ["i", "f", "g", "o"]
    values = _common(gates, h)
    values["c_prev"] = val("c_prev", [1, h])
    values["c_next"] = val("c_next", [1, h])
    for vid in ("ig", "fg", "gg", "og", "fc", "igg", "ct"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pres = {g: _gate(ops, values, g, h) for g in gates}
    ops["sig_i"] = Sigmoid("sig_i", inputs=[pres["i"]], outputs=["ig"])
    ops["sig_f"] = Sigmoid("sig_f", inputs=[pres["f"]], outputs=["fg"])
    ops["tanh_g"] = Tanh("tanh_g", inputs=[pres["g"]], outputs=["gg"])
    ops["sig_o"] = Sigmoid("sig_o", inputs=[pres["o"]], outputs=["og"])
    ops["mul_fc"] = Mul("mul_fc", inputs=["fg", "c_prev"], outputs=["fc"])
    ops["mul_ig"] = Mul("mul_ig", inputs=["ig", "gg"], outputs=["igg"])
    ops["c_upd"] = Add("c_upd", inputs=["fc", "igg"], outputs=["c_next"])
    ops["tanh_c"] = Tanh("tanh_c", inputs=["c_next"], outputs=["ct"])
    ops["h_upd"] = Mul("h_upd", inputs=["og", "ct"], outputs=["h_next"])
    _head(ops, values, h)
    states = {"h": StateSpec("h", StateKind.HIDDEN, "float32", (h,)),
              "c": StateSpec("c", StateKind.HIDDEN, "float32", (h,))}
    e0 = [Edge0("h", "k", value_id="h_prev"), Edge0("c", "k", value_id="c_prev")]
    ed = [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next"),
          EdgeDelta("k", "c", lag_cycles=1, value_id="c_next")]
    p = _proc(f"cmv_lstm{h}", values, ops,
              ["x", "h_prev", "c_prev"]
              + [k for k, v in values.items() if v.layout == "parameter"],
              ["y", "h_next", "c_next"], states, e0, ed)
    return p, _gate_params(_rng(f"lstm{h}"), gates, h)


def build_diag(h):
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "bx": val("bx", [1, h]), "ah": val("ah", [1, h]),
              "a_diag": val("a_diag", [1, h], layout="parameter"),
              "B": val("B", [FEATURES, h], layout="parameter")}
    ops = {
        "mm_bx": MatMul("mm_bx", inputs=["x", "B"], outputs=["bx"]),
        "mul_ah": Mul("mul_ah", inputs=["a_diag", "h_prev"], outputs=["ah"]),
        "h_upd": Add("h_upd", inputs=["ah", "bx"], outputs=["h_next"]),
    }
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_diag{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"diag{h}")
    params = {"a_diag": rng.uniform(0.3, 0.95, (1, h)).astype(np.float32),
              "B": _w(rng, (FEATURES, h), 0.3),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


def build_mgu(h):
    """Minimal Gated Unit (Zhou et al. 2016): one forget gate + candidate.
    f = sig(pre_f); htil = tanh(Wh x + Uh(f (.) h) + b); h' = (1-f)(.)h + f(.)htil
    Two gates -> the G=2 point the anchor set lacks."""
    gates = ["f", "n"]
    values = _common(gates, h)
    values["ones"] = val("ones", [1, h], layout="parameter")
    for vid in ("fg", "fh", "n", "omf", "part_a", "part_b"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre_f = _gate(ops, values, "f", h)
    ops["sig_f"] = Sigmoid("sig_f", inputs=[pre_f], outputs=["fg"])
    ops["mul_fh"] = Mul("mul_fh", inputs=["fg", "h_prev"], outputs=["fh"])
    pre_n = _gate(ops, values, "n", h, reset_input="fh")
    ops["tanh_n"] = Tanh("tanh_n", inputs=[pre_n], outputs=["n"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "fg"], outputs=["omf"])
    ops["mul_a"] = Mul("mul_a", inputs=["omf", "h_prev"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["fg", "n"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_mgu{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    params = _gate_params(_rng(f"mgu{h}"), gates, h)
    params["ones"] = np.ones((1, h), np.float32)
    return p, params


def build_ugrnn(h):
    """UGRNN (Collins et al. 2017): update gate + candidate, no reset.
    g = sig(pre_g); c = tanh(pre_c); h' = g(.)h + (1-g)(.)c  -> G=2, and the
    candidate matmul reads h directly (no gating before or after it)."""
    gates = ["g", "c"]
    values = _common(gates, h)
    values["ones"] = val("ones", [1, h], layout="parameter")
    for vid in ("gg", "cc", "omg", "part_a", "part_b"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre_g = _gate(ops, values, "g", h)
    pre_c = _gate(ops, values, "c", h)
    ops["sig_g"] = Sigmoid("sig_g", inputs=[pre_g], outputs=["gg"])
    ops["tanh_c"] = Tanh("tanh_c", inputs=[pre_c], outputs=["cc"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "gg"], outputs=["omg"])
    ops["mul_a"] = Mul("mul_a", inputs=["gg", "h_prev"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["omg", "cc"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_ugrnn{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    params = _gate_params(_rng(f"ugrnn{h}"), gates, h)
    params["ones"] = np.ones((1, h), np.float32)
    return p, params


def build_indrnn(h):
    """IndRNN (Li et al. 2018): h' = tanh(u (.) h + W x + b).
    The recurrence is ELEMENTWISE but passes through an activation -- an
    intermediate II class between the SSM (no activation) and the dense cells
    (matmul in loop). The binary state-matmul bit cannot express this point;
    the chain-sum model predicts it. That is exactly why it is in the suite."""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "u": val("u", [1, h], layout="parameter"),
              "WxT": val("WxT", [FEATURES, h], layout="parameter"),
              "b": val("b", [1, h], layout="parameter"),
              "uh": val("uh", [1, h]), "wx": val("wx", [1, h]),
              "s1": val("s1", [1, h]), "pre": val("pre", [1, h])}
    ops = {
        "mul_uh": Mul("mul_uh", inputs=["u", "h_prev"], outputs=["uh"]),
        "mm_wx": MatMul("mm_wx", inputs=["x", "WxT"], outputs=["wx"]),
        "s_add": Add("s_add", inputs=["uh", "wx"], outputs=["s1"]),
        "b_add": Add("b_add", inputs=["s1", "b"], outputs=["pre"]),
        "act": Tanh("act", inputs=["pre"], outputs=["h_next"]),
    }
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_indrnn{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"indrnn{h}")
    params = {"u": rng.uniform(0.3, 0.95, (1, h)).astype(np.float32),
              "WxT": _w(rng, (FEATURES, h), 0.4), "b": _w(rng, (1, h), 0.1),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


# --------------------------------------------------------------------------
# Published-cell zoo: architectures from the literature, expressible in the
# emitter's op vocabulary. Each docstring carries the citation; each design
# gets its II class and DSP predicted from the IR like everything else.
# The interesting split: cells DESIGNED for parallel/fast execution
# (SRU, QRNN, minGRU) should land in the shallow classes - the cost model
# explains structurally why they are fast.
# --------------------------------------------------------------------------

def build_janet(h):
    """JANET (van der Westhuizen & Lasenby 2018): forget-gate-only LSTM.
    f = sig(pre_f); ctil = tanh(pre_c); h' = f(.)h + (1-f)(.)ctil"""
    gates = ["f", "c"]
    values = _common(gates, h)
    values["ones"] = val("ones", [1, h], layout="parameter")
    for vid in ("fg", "ct", "omf", "part_a", "part_b"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre_f = _gate(ops, values, "f", h)
    pre_c = _gate(ops, values, "c", h)
    ops["sig_f"] = Sigmoid("sig_f", inputs=[pre_f], outputs=["fg"])
    ops["tanh_c"] = Tanh("tanh_c", inputs=[pre_c], outputs=["ct"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "fg"], outputs=["omf"])
    ops["mul_a"] = Mul("mul_a", inputs=["fg", "h_prev"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["omf", "ct"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_janet{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    params = _gate_params(_rng(f"janet{h}"), gates, h)
    params["ones"] = np.ones((1, h), np.float32)
    return p, params


def build_sru(h):
    """SRU (Lei et al. 2018), the light recurrence: the cell state sees only
    ELEMENTWISE weights (v (.) c), by design, so it parallelizes.
    f = sig(Wf x + vf(.)c + bf); r = sig(Wr x + vr(.)c + br)
    c' = f(.)c + (1-f)(.)(W x); h = r(.)c' + (1-r)(.)(Wh x)"""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h])}
    ops = {}
    for g in ("f", "r"):
        for vid, shape in ((f"Wx_{g}", [FEATURES, h]), (f"v_{g}", [1, h]),
                           (f"b_{g}", [1, h])):
            values[vid] = val(vid, shape, layout="parameter")
        for vid in (f"xp_{g}", f"vc_{g}", f"s_{g}", f"pre_{g}", f"{g}g"):
            values[vid] = val(vid, [1, h])
        ops[f"mmx_{g}"] = MatMul(f"mmx_{g}", inputs=["x", f"Wx_{g}"],
                                 outputs=[f"xp_{g}"])
        ops[f"vc_{g}"] = Mul(f"vc_{g}", inputs=[f"v_{g}", "h_prev"],
                             outputs=[f"vc_{g}"])
        ops[f"s_{g}"] = Add(f"s_{g}", inputs=[f"xp_{g}", f"vc_{g}"],
                            outputs=[f"s_{g}"])
        ops[f"pb_{g}"] = Add(f"pb_{g}", inputs=[f"s_{g}", f"b_{g}"],
                             outputs=[f"pre_{g}"])
        ops[f"sig_{g}"] = Sigmoid(f"sig_{g}", inputs=[f"pre_{g}"],
                                  outputs=[f"{g}g"])
    for vid, shape in (("Wx_z", [FEATURES, h]), ("Wx_hw", [FEATURES, h]),
                       ("ones", [1, h])):
        values[vid] = val(vid, shape, layout="parameter")
    for vid in ("zx", "hwx", "omf", "fa", "fb", "omr", "ra", "rb"):
        values[vid] = val(vid, [1, h])
    ops["mmx_z"] = MatMul("mmx_z", inputs=["x", "Wx_z"], outputs=["zx"])
    ops["mmx_hw"] = MatMul("mmx_hw", inputs=["x", "Wx_hw"], outputs=["hwx"])
    ops["omf"] = Sub("omf", inputs=["ones", "fg"], outputs=["omf"])
    ops["fa"] = Mul("fa", inputs=["fg", "h_prev"], outputs=["fa"])
    ops["fb"] = Mul("fb", inputs=["omf", "zx"], outputs=["fb"])
    ops["c_upd"] = Add("c_upd", inputs=["fa", "fb"], outputs=["h_next"])
    ops["omr"] = Sub("omr", inputs=["ones", "rg"], outputs=["omr"])
    ops["ra"] = Mul("ra", inputs=["rg", "h_next"], outputs=["ra"])
    ops["rb"] = Mul("rb", inputs=["omr", "hwx"], outputs=["rb"])
    values["hout"] = val("hout", [1, h])
    ops["h_out"] = Add("h_out", inputs=["ra", "rb"], outputs=["hout"])
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [h, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["hout", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_sru{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"sru{h}")
    params = {"ones": np.ones((1, h), np.float32),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    for g in ("f", "r"):
        params[f"Wx_{g}"] = _w(rng, (FEATURES, h), 0.4)
        params[f"v_{g}"] = rng.uniform(0.2, 0.8, (1, h)).astype(np.float32)
        params[f"b_{g}"] = _w(rng, (1, h), 0.1)
    params["Wx_z"] = _w(rng, (FEATURES, h), 0.4)
    params["Wx_hw"] = _w(rng, (FEATURES, h), 0.4)
    return p, params


def build_qrnn(h):
    """QRNN f-pooling (Bradbury et al. 2016), conv window k=2. The gates come
    from a convolution over (x_t, x_{t-1}) - entirely OFF the state cycle;
    x_{t-1} is carried as a delay state. The pooling recurrence itself is
    pure elementwise, which is exactly why the paper calls it fast."""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "xp_prev": val("xp_prev", [1, FEATURES]),
              "xp_next": val("xp_next", [1, FEATURES]),
              "zeroF": val("zeroF", [1, FEATURES], layout="parameter"),
              "ones": val("ones", [1, h], layout="parameter")}
    for g in ("z", "f"):
        values[f"W1_{g}"] = val(f"W1_{g}", [FEATURES, h], layout="parameter")
        values[f"W2_{g}"] = val(f"W2_{g}", [FEATURES, h], layout="parameter")
        for vid in (f"a_{g}", f"b_{g}v", f"pre_{g}"):
            values[vid] = val(vid, [1, h])
    for vid in ("zg", "fg", "omf", "pa", "pb"):
        values[vid] = val(vid, [1, h])
    ops = {
        "xp_upd": Add("xp_upd", inputs=["x", "zeroF"], outputs=["xp_next"]),
    }
    for g in ("z", "f"):
        ops[f"mm1_{g}"] = MatMul(f"mm1_{g}", inputs=["x", f"W1_{g}"],
                                 outputs=[f"a_{g}"])
        ops[f"mm2_{g}"] = MatMul(f"mm2_{g}", inputs=["xp_prev", f"W2_{g}"],
                                 outputs=[f"b_{g}v"])
        ops[f"cadd_{g}"] = Add(f"cadd_{g}", inputs=[f"a_{g}", f"b_{g}v"],
                               outputs=[f"pre_{g}"])
    ops["tanh_z"] = Tanh("tanh_z", inputs=["pre_z"], outputs=["zg"])
    ops["sig_f"] = Sigmoid("sig_f", inputs=["pre_f"], outputs=["fg"])
    ops["omf"] = Sub("omf", inputs=["ones", "fg"], outputs=["omf"])
    ops["pa"] = Mul("pa", inputs=["fg", "h_prev"], outputs=["pa"])
    ops["pb"] = Mul("pb", inputs=["omf", "zg"], outputs=["pb"])
    ops["h_upd"] = Add("h_upd", inputs=["pa", "pb"], outputs=["h_next"])
    _head(ops, values, h)
    states = {"h": StateSpec("h", StateKind.HIDDEN, "float32", (h,)),
              "xp": StateSpec("xp", StateKind.HIDDEN, "float32", (FEATURES,))}
    e0 = [Edge0("h", "k", value_id="h_prev"), Edge0("xp", "k", value_id="xp_prev")]
    ed = [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next"),
          EdgeDelta("k", "xp", lag_cycles=1, value_id="xp_next")]
    p = _proc(f"cmv_qrnn{h}", values, ops,
              ["x", "h_prev", "xp_prev"] + _params_of(values),
              ["y", "h_next", "xp_next"], states, e0, ed)
    rng = _rng(f"qrnn{h}")
    params = {"ones": np.ones((1, h), np.float32),
              "zeroF": np.zeros((1, FEATURES), np.float32),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    for g in ("z", "f"):
        params[f"W1_{g}"] = _w(rng, (FEATURES, h), 0.4)
        params[f"W2_{g}"] = _w(rng, (FEATURES, h), 0.4)
    return p, params


def build_mingru(h):
    """minGRU (Feng et al. 2024, "Were RNNs All We Needed?"): gate and
    candidate depend on x ONLY, so the recurrence is pure elementwise and
    parallel-scannable - the 2024 argument, visible as graph structure.
    z = sig(Wz x); htil = Wh x; h' = (1-z)(.)h + z(.)htil"""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "Wz": val("Wz", [FEATURES, h], layout="parameter"),
              "Wh": val("Wh", [FEATURES, h], layout="parameter"),
              "ones": val("ones", [1, h], layout="parameter")}
    for vid in ("zx", "zg", "ht", "omz", "pa", "pb"):
        values[vid] = val(vid, [1, h])
    ops = {
        "mm_z": MatMul("mm_z", inputs=["x", "Wz"], outputs=["zx"]),
        "sig_z": Sigmoid("sig_z", inputs=["zx"], outputs=["zg"]),
        "mm_h": MatMul("mm_h", inputs=["x", "Wh"], outputs=["ht"]),
        "omz": Sub("omz", inputs=["ones", "zg"], outputs=["omz"]),
        "pa": Mul("pa", inputs=["omz", "h_prev"], outputs=["pa"]),
        "pb": Mul("pb", inputs=["zg", "ht"], outputs=["pb"]),
        "h_upd": Add("h_upd", inputs=["pa", "pb"], outputs=["h_next"]),
    }
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_mingru{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"mingru{h}")
    params = {"ones": np.ones((1, h), np.float32),
              "Wz": _w(rng, (FEATURES, h), 0.4), "Wh": _w(rng, (FEATURES, h), 0.4),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


def build_fastgrnn(h):
    """FastGRNN (Kusupati et al. 2018, Microsoft EdgeML): shared pre-activation
    pre = Wx + Uh; z = sig(pre); htil = tanh(pre);
    h' = (zeta(1-z) + nu)(.)htil + z(.)h. Built FOR edge devices - the cost
    model should place it in the same class as GRU (matmul in loop)."""
    values = _common(["s"], h)
    values["ones"] = val("ones", [1, h], layout="parameter")
    values["zeta"] = val("zeta", [1, h], layout="parameter")
    values["nu"] = val("nu", [1, h], layout="parameter")
    for vid in ("zg", "ht", "omz", "zscaled", "gain", "pa", "pb"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre = _gate(ops, values, "s", h)
    ops["sig_z"] = Sigmoid("sig_z", inputs=[pre], outputs=["zg"])
    ops["tanh_h"] = Tanh("tanh_h", inputs=[pre], outputs=["ht"])
    ops["omz"] = Sub("omz", inputs=["ones", "zg"], outputs=["omz"])
    ops["zscaled"] = Mul("zscaled", inputs=["zeta", "omz"], outputs=["zscaled"])
    ops["gain"] = Add("gain", inputs=["zscaled", "nu"], outputs=["gain"])
    ops["pa"] = Mul("pa", inputs=["gain", "ht"], outputs=["pa"])
    ops["pb"] = Mul("pb", inputs=["zg", "h_prev"], outputs=["pb"])
    ops["h_upd"] = Add("h_upd", inputs=["pa", "pb"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_fastgrnn{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"fastgrnn{h}")
    params = _gate_params(rng, ["s"], h)
    params["ones"] = np.ones((1, h), np.float32)
    params["zeta"] = np.full((1, h), 1.0, np.float32)
    params["nu"] = np.full((1, h), 0.05, np.float32)
    return p, params


def build_ligru(h):
    """LiGRU (Ravanelli et al. 2018): GRU minus the reset gate, ReLU
    candidate (batch-norm folds into the weights at inference).
    z = sig(pre_z); htil = relu(pre_h); h' = z(.)h + (1-z)(.)htil"""
    gates = ["z", "n"]
    values = _common(gates, h)
    values["ones"] = val("ones", [1, h], layout="parameter")
    for vid in ("zg", "ht", "omz", "pa", "pb"):
        values[vid] = val(vid, [1, h])
    ops = {}
    pre_z = _gate(ops, values, "z", h)
    pre_n = _gate(ops, values, "n", h)
    ops["sig_z"] = Sigmoid("sig_z", inputs=[pre_z], outputs=["zg"])
    ops["relu_n"] = ReLU("relu_n", inputs=[pre_n], outputs=["ht"])
    ops["omz"] = Sub("omz", inputs=["ones", "zg"], outputs=["omz"])
    ops["pa"] = Mul("pa", inputs=["zg", "h_prev"], outputs=["pa"])
    ops["pb"] = Mul("pb", inputs=["omz", "ht"], outputs=["pb"])
    ops["h_upd"] = Add("h_upd", inputs=["pa", "pb"], outputs=["h_next"])
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_ligru{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    params = _gate_params(_rng(f"ligru{h}"), gates, h)
    params["ones"] = np.ones((1, h), np.float32)
    # ReLU keeps positives unbounded; damp weights a little so Q6.12 range
    # is safe over 64 steps.
    for k in list(params):
        if k.startswith("WhT_"):
            params[k] = params[k] * 0.6
    return p, params


def build_gru2l(h):
    """Two stacked GRU layers (H=h each). The cross-layer edge is
    feed-forward, so DEPTH must not move II: both layer cycles are the
    standard matmul-in-loop chain. DSP should be ~2x one layer + head."""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "g_prev": val("g_prev", [1, h]), "g_next": val("g_next", [1, h]),
              "ones": val("ones", [1, h], layout="parameter")}
    ops = {}
    params = {"ones": np.ones((1, h), np.float32)}
    rng = _rng(f"gru2l{h}")

    def gru_layer(prefix, x_vid, hprev_vid, hnext_vid, in_w):
        gates = [f"{prefix}r", f"{prefix}z", f"{prefix}n"]
        for g in gates:
            values[f"WxT_{g}"] = val(f"WxT_{g}", [in_w, h], layout="parameter")
            values[f"WhT_{g}"] = val(f"WhT_{g}", [h, h], layout="parameter")
            values[f"b_{g}"] = val(f"b_{g}", [1, h], layout="parameter")
            params[f"WxT_{g}"] = _w(rng, (in_w, h), 0.4)
            params[f"WhT_{g}"] = _w(rng, (h, h), 0.9 / math.sqrt(h))
            params[f"b_{g}"] = _w(rng, (1, h), 0.1)
            for vid in (f"xg_{g}", f"hg_{g}", f"sg_{g}", f"pre_{g}"):
                values[vid] = val(vid, [1, h])
            ops[f"mmx_{g}"] = MatMul(f"mmx_{g}", inputs=[x_vid, f"WxT_{g}"],
                                     outputs=[f"xg_{g}"])
            ops[f"mmh_{g}"] = MatMul(f"mmh_{g}", inputs=[hprev_vid, f"WhT_{g}"],
                                     outputs=[f"hg_{g}"])
            ops[f"gs_{g}"] = Add(f"gs_{g}", inputs=[f"xg_{g}", f"hg_{g}"],
                                 outputs=[f"sg_{g}"])
            ops[f"gb_{g}"] = Add(f"gb_{g}", inputs=[f"sg_{g}", f"b_{g}"],
                                 outputs=[f"pre_{g}"])
        for vid in (f"{prefix}rg", f"{prefix}zg", f"{prefix}nr", f"{prefix}nn",
                    f"{prefix}omz", f"{prefix}pa", f"{prefix}pb"):
            values[vid] = val(vid, [1, h])
        ops[f"sig_{prefix}r"] = Sigmoid(f"sig_{prefix}r",
                                        inputs=[f"pre_{prefix}r"],
                                        outputs=[f"{prefix}rg"])
        ops[f"sig_{prefix}z"] = Sigmoid(f"sig_{prefix}z",
                                        inputs=[f"pre_{prefix}z"],
                                        outputs=[f"{prefix}zg"])
        ops[f"mul_{prefix}nr"] = Mul(f"mul_{prefix}nr",
                                     inputs=[f"pre_{prefix}n", f"{prefix}rg"],
                                     outputs=[f"{prefix}nr"])
        ops[f"tanh_{prefix}n"] = Tanh(f"tanh_{prefix}n",
                                      inputs=[f"{prefix}nr"],
                                      outputs=[f"{prefix}nn"])
        ops[f"omz_{prefix}"] = Sub(f"omz_{prefix}", inputs=["ones", f"{prefix}zg"],
                                   outputs=[f"{prefix}omz"])
        ops[f"pa_{prefix}"] = Mul(f"pa_{prefix}",
                                  inputs=[f"{prefix}omz", f"{prefix}nn"],
                                  outputs=[f"{prefix}pa"])
        ops[f"pb_{prefix}"] = Mul(f"pb_{prefix}",
                                  inputs=[f"{prefix}zg", hprev_vid],
                                  outputs=[f"{prefix}pb"])
        ops[f"upd_{prefix}"] = Add(f"upd_{prefix}",
                                   inputs=[f"{prefix}pa", f"{prefix}pb"],
                                   outputs=[hnext_vid])

    gru_layer("l1", "x", "h_prev", "h_next", FEATURES)
    gru_layer("l2", "h_next", "g_prev", "g_next", h)
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [h, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["g_next", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])
    params["Wo2"] = _w(rng, (h, 1), 0.4)
    params["bo2"] = [[0.0]]
    states = {"h": StateSpec("h", StateKind.HIDDEN, "float32", (h,)),
              "g": StateSpec("g", StateKind.HIDDEN, "float32", (h,))}
    e0 = [Edge0("h", "k", value_id="h_prev"), Edge0("g", "k", value_id="g_prev")]
    ed = [EdgeDelta("k", "h", lag_cycles=1, value_id="h_next"),
          EdgeDelta("k", "g", lag_cycles=1, value_id="g_next")]
    p = _proc(f"cmv_gru2l{h}", values, ops,
              ["x", "h_prev", "g_prev"] + _params_of(values),
              ["y", "h_next", "g_next"], states, e0, ed)
    return p, params


def build_tcn(h):
    """Two-layer causal TCN (temporal convolutional network; Bai et al.
    2018), kernel 2, dilations 1 and 2, ReLU. A TCN is FEED-FORWARD over a
    window of past values: the window lives in delay-line states (pure
    copies), so NO arithmetic sits on any delay-edge cycle. The feed-forward
    corollary of the II bound predicts II = 1 - the generalization test
    beyond recurrent cells."""
    values = {"x": val("x", [1, FEATURES]),
              # delay-line states: previous input, and layer-1 output at
              # t-1 and t-2 (dilation-2 tap)
              "x1_prev": val("x1_prev", [1, FEATURES]),
              "x1_next": val("x1_next", [1, FEATURES]),
              "h1d1_prev": val("h1d1_prev", [1, h]),
              "h1d1_next": val("h1d1_next", [1, h]),
              "h1d2_prev": val("h1d2_prev", [1, h]),
              "h1d2_next": val("h1d2_next", [1, h]),
              "zeroF": val("zeroF", [1, FEATURES], layout="parameter"),
              "zeroH": val("zeroH", [1, h], layout="parameter")}
    for vid, shape in (("W1a", [FEATURES, h]), ("W1b", [FEATURES, h]),
                       ("b1", [1, h]), ("W2a", [h, h]), ("W2b", [h, h]),
                       ("b2", [1, h])):
        values[vid] = val(vid, shape, layout="parameter")
    for vid in ("c1a", "c1b", "s1", "pre1", "h1",
                "c2a", "c2b", "s2", "pre2", "h2"):
        values[vid] = val(vid, [1, h])
    ops = {
        # layer 1: causal conv k=2, d=1 over (x_t, x_{t-1})
        "mm1a": MatMul("mm1a", inputs=["x", "W1a"], outputs=["c1a"]),
        "mm1b": MatMul("mm1b", inputs=["x1_prev", "W1b"], outputs=["c1b"]),
        "s1": Add("s1", inputs=["c1a", "c1b"], outputs=["s1"]),
        "b1": Add("b1", inputs=["s1", "b1"], outputs=["pre1"]),
        "relu1": ReLU("relu1", inputs=["pre1"], outputs=["h1"]),
        # layer 2: causal conv k=2, d=2 over (h1_t, h1_{t-2})
        "mm2a": MatMul("mm2a", inputs=["h1", "W2a"], outputs=["c2a"]),
        "mm2b": MatMul("mm2b", inputs=["h1d2_prev", "W2b"], outputs=["c2b"]),
        "s2": Add("s2", inputs=["c2a", "c2b"], outputs=["s2"]),
        "b2": Add("b2", inputs=["s2", "b2"], outputs=["pre2"]),
        "relu2": ReLU("relu2", inputs=["pre2"], outputs=["h2"]),
        # delay-line shifts (pure copies; the only writes to state)
        "x1_upd": Add("x1_upd", inputs=["x", "zeroF"], outputs=["x1_next"]),
        "d1_upd": Add("d1_upd", inputs=["h1", "zeroH"], outputs=["h1d1_next"]),
        "d2_upd": Add("d2_upd", inputs=["h1d1_prev", "zeroH"],
                      outputs=["h1d2_next"]),
    }
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [h, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["h2", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])
    g = Graph(values=values, ops=ops,
              graph_inputs=["x", "x1_prev", "h1d1_prev", "h1d2_prev"]
              + _params_of(values),
              graph_outputs=["y", "x1_next", "h1d1_next", "h1d2_next"])
    states = {
        "x1": StateSpec("x1", StateKind.HIDDEN, "float32", (FEATURES,)),
        "h1d1": StateSpec("h1d1", StateKind.HIDDEN, "float32", (h,)),
        "h1d2": StateSpec("h1d2", StateKind.HIDDEN, "float32", (h,)),
    }
    p = Process(process_id=f"cmv_tcn{h}",
                kernels={"k": Kernel("k", graph=g)}, states=states,
                edge0=[Edge0("x1", "k", value_id="x1_prev"),
                       Edge0("h1d1", "k", value_id="h1d1_prev"),
                       Edge0("h1d2", "k", value_id="h1d2_prev")],
                edge_delta=[EdgeDelta("k", "x1", lag_cycles=1,
                                      value_id="x1_next"),
                            EdgeDelta("k", "h1d1", lag_cycles=1,
                                      value_id="h1d1_next"),
                            EdgeDelta("k", "h1d2", lag_cycles=1,
                                      value_id="h1d2_next")])
    p.validate()
    rng = _rng(f"tcn{h}")
    params = {"zeroF": np.zeros((1, FEATURES), np.float32),
              "zeroH": np.zeros((1, h), np.float32),
              "W1a": _w(rng, (FEATURES, h), 0.4), "W1b": _w(rng, (FEATURES, h), 0.4),
              "b1": _w(rng, (1, h), 0.1),
              "W2a": _w(rng, (h, h), 0.25), "W2b": _w(rng, (h, h), 0.25),
              "b2": _w(rng, (1, h), 0.1),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


def build_gatedssm(h):
    """Input-gated diagonal SSM - the selective-SSM (Mamba-family)
    recurrence form: the decay is a function of the INPUT, never of the
    state, so the state path stays elementwise. This is the structural
    reason the Mamba family is fast, visible as graph shape:
    a_t = sig(Wa x + ba);  h' = a_t (.) h + (1 - a_t) (.) (Wz x)."""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "Wa": val("Wa", [FEATURES, h], layout="parameter"),
              "ba": val("ba", [1, h], layout="parameter"),
              "Wz": val("Wz", [FEATURES, h], layout="parameter"),
              "ones": val("ones", [1, h], layout="parameter")}
    for vid in ("pa", "pab", "ag", "zx", "oma", "sa", "sb"):
        values[vid] = val(vid, [1, h])
    ops = {
        "mm_a": MatMul("mm_a", inputs=["x", "Wa"], outputs=["pa"]),
        "ba_add": Add("ba_add", inputs=["pa", "ba"], outputs=["pab"]),
        "sig_a": Sigmoid("sig_a", inputs=["pab"], outputs=["ag"]),
        "mm_z": MatMul("mm_z", inputs=["x", "Wz"], outputs=["zx"]),
        "oma": Sub("oma", inputs=["ones", "ag"], outputs=["oma"]),
        "sa": Mul("sa", inputs=["ag", "h_prev"], outputs=["sa"]),
        "sb": Mul("sb", inputs=["oma", "zx"], outputs=["sb"]),
        "h_upd": Add("h_upd", inputs=["sa", "sb"], outputs=["h_next"]),
    }
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_gatedssm{h}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"gatedssm{h}")
    params = {"ones": np.ones((1, h), np.float32),
              "Wa": _w(rng, (FEATURES, h), 0.4), "ba": _w(rng, (1, h), 0.5),
              "Wz": _w(rng, (FEATURES, h), 0.4),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


def build_rwkv(h):
    """RWKV-style time-mixing recurrence (bounded-kernel surrogate):
    two coupled elementwise states - a weighted-value numerator and a
    weight denominator - with a divide readout.
      kappa = sig(Wk x); v = Wv x
      num' = d (.) num + kappa (.) v;   den' = d (.) den + kappa
      out = num' / den'
    The exponential kernel of full RWKV is replaced by a bounded sigmoid
    kernel so Q6.12 range is safe; the recurrence STRUCTURE - and hence the
    predicted class - is the same. den initializes at 1 (never zero)."""
    values = {"x": val("x", [1, FEATURES]),
              "num_prev": val("num_prev", [1, h]),
              "num_next": val("num_next", [1, h]),
              "den_prev": val("den_prev", [1, h]),
              "den_next": val("den_next", [1, h]),
              "d": val("d", [1, h], layout="parameter"),
              "Wk": val("Wk", [FEATURES, h], layout="parameter"),
              "Wv": val("Wv", [FEATURES, h], layout="parameter")}
    for vid in ("pk", "kap", "v", "kv", "dnum", "dden", "outv"):
        values[vid] = val(vid, [1, h])
    ops = {
        "mm_k": MatMul("mm_k", inputs=["x", "Wk"], outputs=["pk"]),
        "sig_k": Sigmoid("sig_k", inputs=["pk"], outputs=["kap"]),
        "mm_v": MatMul("mm_v", inputs=["x", "Wv"], outputs=["v"]),
        "kv": Mul("kv", inputs=["kap", "v"], outputs=["kv"]),
        "dnum": Mul("dnum", inputs=["d", "num_prev"], outputs=["dnum"]),
        "num_upd": Add("num_upd", inputs=["dnum", "kv"], outputs=["num_next"]),
        "dden": Mul("dden", inputs=["d", "den_prev"], outputs=["dden"]),
        "den_upd": Add("den_upd", inputs=["dden", "kap"], outputs=["den_next"]),
        "wkv": Div("wkv", inputs=["num_next", "den_next"], outputs=["outv"]),
    }
    values["y0"] = val("y0", [1, 1])
    values["y"] = val("y", [1, 1])
    values["Wo2"] = val("Wo2", [h, 1], layout="parameter")
    values["bo2"] = val("bo2", [1, 1], layout="parameter")
    ops["mm_out"] = MatMul("mm_out", inputs=["outv", "Wo2"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo2"], outputs=["y"])
    g = Graph(values=values, ops=ops,
              graph_inputs=["x", "num_prev", "den_prev"] + _params_of(values),
              graph_outputs=["y", "num_next", "den_next"])
    states = {
        "num": StateSpec("num", StateKind.HIDDEN, "float32", (h,)),
        "den": StateSpec("den", StateKind.HIDDEN, "float32", (h,),
                         metadata={"initializer": [1.0] * h}),
    }
    p = Process(process_id=f"cmv_rwkv{h}",
                kernels={"k": Kernel("k", graph=g)}, states=states,
                edge0=[Edge0("num", "k", value_id="num_prev"),
                       Edge0("den", "k", value_id="den_prev")],
                edge_delta=[EdgeDelta("k", "num", lag_cycles=1,
                                      value_id="num_next"),
                            EdgeDelta("k", "den", lag_cycles=1,
                                      value_id="den_next")])
    p.validate()
    rng = _rng(f"rwkv{h}")
    params = {"d": rng.uniform(0.3, 0.9, (1, h)).astype(np.float32),
              "Wk": _w(rng, (FEATURES, h), 0.4),
              "Wv": _w(rng, (FEATURES, h), 0.3),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


def build_chain(d, h=8):
    """Pathological depth-sweep cell: EXACTLY d cascaded elementwise stages
    on the state cycle (each stage: a_i (.) prev + Bx), so the recurrence
    chain is 2d ops by construction while model size and I/O stay fixed.
    Per-stage fixed-point truncation makes the affine cascade algebraically
    unfoldable, so the tool cannot collapse the chain. Pre-registered
    hypothesis: achieved II tracks the chain sum (II = 2d) - latency
    growing with recurrence DEPTH alone, the cost model's central variable
    isolated from every other knob."""
    values = {"x": val("x", [1, FEATURES]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "B": val("B", [FEATURES, h], layout="parameter"),
              "bx": val("bx", [1, h])}
    ops = {"mm_bx": MatMul("mm_bx", inputs=["x", "B"], outputs=["bx"])}
    prev = "h_prev"
    for i in range(d):
        values[f"a{i}"] = val(f"a{i}", [1, h], layout="parameter")
        values[f"m{i}"] = val(f"m{i}", [1, h])
        out = "h_next" if i == d - 1 else f"s{i}"
        if out != "h_next":
            values[out] = val(out, [1, h])
        ops[f"mul{i}"] = Mul(f"mul{i}", inputs=[f"a{i}", prev],
                             outputs=[f"m{i}"])
        ops[f"add{i}"] = Add(f"add{i}", inputs=[f"m{i}", "bx"],
                             outputs=[out])
        prev = out
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_chain{d}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"chain{d}")
    params = {"B": _w(rng, (FEATURES, h), 0.3),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    for i in range(d):
        # decays well under 1 so the d-fold cascade stays in Q6.12 range
        params[f"a{i}"] = rng.uniform(0.3, 0.8, (1, h)).astype(np.float32)
    return p, params


def build_blockdiag(k, h=16):
    """Theorem L on silicon: a k-step BLOCK of the diagonal SSM as one
    affine update. Because the per-step maps are state-affine, k steps
    compose exactly: h_{t+k} = (a^k) (.) h_t + [x_t..x_{t+k-1}] @ Bstar,
    where Bstar stacks a^{k-1-j} (.) B per lag. The block is itself an
    elementwise-class cell (II = 4 per BLOCK), so the amortized cost is
    4/k cycles PER SAMPLE - at k = 8 that is HALF a cycle per sample,
    below the one-sample-per-cycle barrier. Pre-registered demonstration
    that class-L architectures have no fundamental per-sample floor: the
    'floor' moves wherever the blocking factor puts it."""
    fblk = k * FEATURES
    values = {"x": val("x", [1, fblk]),
              "h_prev": val("h_prev", [1, h]), "h_next": val("h_next", [1, h]),
              "bx": val("bx", [1, h]), "ah": val("ah", [1, h]),
              "astar": val("astar", [1, h], layout="parameter"),
              "Bstar": val("Bstar", [fblk, h], layout="parameter")}
    ops = {
        "mm_bx": MatMul("mm_bx", inputs=["x", "Bstar"], outputs=["bx"]),
        "mul_ah": Mul("mul_ah", inputs=["astar", "h_prev"], outputs=["ah"]),
        "h_upd": Add("h_upd", inputs=["ah", "bx"], outputs=["h_next"]),
    }
    _head(ops, values, h)
    states, e0, ed = _hidden_state(h)
    p = _proc(f"cmv_blockdiag{k}", values, ops,
              ["x", "h_prev"] + _params_of(values),
              ["y", "h_next"], states, e0, ed)
    rng = _rng(f"blockdiag{k}")
    a = rng.uniform(0.3, 0.9, (1, h)).astype(np.float64)
    B = (rng.standard_normal((FEATURES, h)) * 0.3).astype(np.float64)
    Bstar = np.vstack([B * (a ** (k - 1 - j)) for j in range(k)])
    params = {"astar": (a ** k).astype(np.float32),
              "Bstar": Bstar.astype(np.float32),
              "Wo2": _w(rng, (h, 1), 0.4), "bo2": [[0.0]]}
    return p, params


# --------------------------------------------------------------------------
# Static predictions, computed from the built IR (not asserted by hand)
# --------------------------------------------------------------------------

def _op_latency(op, values):
    if op.op_type == "MatMul":
        k = values[op.inputs[0]].shape[-1]
        return 1 + math.ceil(math.log2(max(k, 2)))
    return OP_LAT.get(op.op_type, 1)


def loop_chain(process):
    """Binding recurrence chain: the longest op-latency path around an
    actual delay-edge CYCLE. A read->write path between two states counts
    only if the states feed each other both ways (otherwise it is
    feed-forward - e.g. layer 1 -> layer 2 in a stacked cell - and must
    not bound II). Returns (cycles, [op names of the binding leg])."""
    g = process.kernels["k"].graph
    producers = {}
    for name, op in g.ops.items():
        for out in op.outputs:
            producers[out] = name

    read_state = {e.value_id: e.source for e in process.edge0}
    write_state = {e.value_id: e.target for e in process.edge_delta}

    def longest_from(src_vid, w_vid):
        """Longest (cycles, ops) path from read value src_vid to w_vid."""
        memo = {}

        def rec(vid):
            if vid == src_vid:
                return 0, []
            if vid in read_state:      # other reads are step inputs, not paths
                return None
            if vid in memo:
                return memo[vid]
            if vid not in producers:
                return None
            op = g.ops[producers[vid]]
            best = None
            for inp in op.inputs:
                sub = rec(inp)
                if sub is None:
                    continue
                cand = (sub[0] + _op_latency(op, g.values),
                        sub[1] + [producers[vid]])
                if best is None or cand[0] > best[0]:
                    best = cand
            memo[vid] = best
            return best

        return rec(w_vid)

    # d[(A, B)] = longest chain from A's read to B's write (per state pair)
    d = {}
    for r_vid, a in read_state.items():
        for w_vid, b in write_state.items():
            path = longest_from(r_vid, w_vid)
            if path is not None and (
                    (a, b) not in d or path[0] > d[(a, b)][0]):
                d[(a, b)] = path

    # Lag of each state's delay edge (the theorem divides by the SUM of
    # lags around the cycle, not the number of edges).
    lag = {e.target: e.lag_cycles for e in process.edge_delta}

    best = (0, [])
    states = {e.target for e in process.edge_delta}
    for a in states:
        if (a, a) in d:
            bound = -(-d[(a, a)][0] // lag[a])
            if bound > best[0]:
                best = (bound, d[(a, a)][1])
    for a in states:
        for b in states:
            if a >= b or (a, b) not in d or (b, a) not in d:
                continue
            # two-state cycle: L = both legs, Lambda = both edges' lags;
            # the class comes from the longer leg
            total = -(-(d[(a, b)][0] + d[(b, a)][0]) // (lag[a] + lag[b]))
            leg = max(d[(a, b)], d[(b, a)], key=lambda p: p[0])
            if total > best[0]:
                best = (total, leg[1])
    return best


def mac_count(process):
    """Weighted MAC count: matmul K*N products + elementwise-Mul lanes.
    This is the DSP proxy; k_mac is calibrated on RNN@16 only."""
    g = process.kernels["k"].graph
    total = 0
    for op in g.ops.values():
        if op.op_type == "MatMul":
            k = g.values[op.inputs[0]].shape[-1]
            n = g.values[op.inputs[1]].shape[-1]
            total += k * n
        elif op.op_type == "Mul":
            total += int(np.prod(g.values[op.outputs[0]].shape))
    return total


def streaming_class(process):
    """Algebraic classification of the ARCHITECTURE (not the
    implementation): is the state-update map affine in the state?

    - "FF": no state feedback at all - trivially parallel.
    - "L-affine": every state-write is an affine function of the state
      (adds/subs freely; multiplication and matmul only against
      state-independent operands; no activation applied to a
      state-dependent value on the update path). Such per-step maps live
      in the monoid of affine maps, so k-step blocks compose in O(log k)
      depth - the amortized streaming depth vanishes (scan/blocking
      theorem). The architecture has NO fundamental streaming-latency
      floor; any measured floor is an implementation choice.
    - "N-nonlinear": the update multiplies state-dependent values together
      or passes a state-dependent value through a nonlinearity. In the
      polynomial-gate circuit model, composed state degree grows
      exponentially, so depth grows linearly in steps (Kung-1976-style
      degree argument): an irreducible sequential core per sample.
    """
    g = process.kernels["k"].graph
    read_of = {e.source: e.value_id for e in process.edge0}
    write_of = {e.target: e.value_id for e in process.edge_delta}
    producers = {}
    for name, op in g.ops.items():
        for out in op.outputs:
            producers[out] = name

    def reaches(src_vid, dst_vid):
        seen = set()
        stack = [dst_vid]
        while stack:
            v = stack.pop()
            if v == src_vid:
                return True
            if v in seen or v not in producers:
                continue
            seen.add(v)
            stack.extend(g.ops[producers[v]].inputs)
        return False

    # A state is CYCLIC if its read reaches its own write, possibly through
    # other states across timesteps (transitive closure over the state
    # graph). Acyclic states are finite-memory delay lines: their influence
    # washes out, so they act as extended inputs - nonlinearity on them
    # composes over a bounded window and cannot grow degree unboundedly.
    sids = [s for s in write_of if s in read_of]
    adj = {a: {b for b in sids if reaches(read_of[a], write_of[b])}
           for a in sids}
    closure = {a: set(adj[a]) for a in sids}
    for _ in sids:
        for a in sids:
            closure[a] |= {c for b in closure[a] for c in closure[b]}
    cyclic = {s for s in sids if s in closure[s]}
    if not cyclic:
        return "FF"
    reads = {read_of[s] for s in cyclic}
    writes = {write_of[s] for s in cyclic}
    # state-dependent closure + affinity, in one topological sweep
    dep: dict = {v: True for v in reads}
    affine: dict = {v: True for v in reads}
    order = _topological_ops(g)
    nonlinear = False
    for op_id in order:
        op = g.ops[op_id]
        out = op.outputs[0]
        in_dep = [i for i in op.inputs if dep.get(i, False)]
        out_dep = bool(in_dep)
        dep[out] = out_dep
        if not out_dep:
            affine[out] = True
            continue
        in_affine = all(affine.get(i, True) for i in in_dep)
        if op.op_type in ("Add", "Sub"):
            affine[out] = in_affine
        elif op.op_type in ("Mul", "MatMul"):
            if len(in_dep) >= 2:
                affine[out] = False       # state x state product
            else:
                affine[out] = in_affine   # scaling by a state-free operand
        elif op.op_type == "Div":
            # affine only when the state enters through the numerator
            affine[out] = (in_affine
                           and not dep.get(op.inputs[1], False))
        else:                              # activation on state-dependent
            affine[out] = False
    for w in writes:
        if dep.get(w, False) and not affine.get(w, True):
            nonlinear = True
    if not any(dep.get(w, False) for w in writes):
        return "FF"
    return "N-nonlinear" if nonlinear else "L-affine"


def predict(tag, process):
    chain_cycles, chain_ops = loop_chain(process)
    g = process.kernels["k"].graph
    chain_matmuls = sum(1 for op in chain_ops
                        if g.ops[op].op_type == "MatMul" and op != "mm_out")
    has_act = any(g.ops[op].op_type in ("Tanh", "Sigmoid") for op in chain_ops)
    h = max(s.shape[0] for s in process.states.values())
    # Structural classes, ordered by loop depth. A graph with NO delay-edge
    # cycle has no iteration bound at all - II is limited only by resources,
    # so the prediction is 1 (the feed-forward corollary of the II theorem).
    if chain_cycles == 0:
        h0 = max(s.shape[0] for s in process.states.values())
        return {"tag": tag, "H": int(h0), "ii_class": "feed-forward",
                "ii_hat": 1, "chain_cycles": 0, "chain_ops": [],
                "macs": int(mac_count(process))}
    if chain_matmuls >= 2:
        ii_hat = chain_cycles                          # serial-gated matmul
        ii_class = "serial-gated"
    elif chain_matmuls == 1:
        ii_hat = 8 + math.ceil(math.log2(h))           # 12 at the H=16 anchors
        ii_class = "matmul-in-loop"
    elif has_act:
        ii_hat = max(chain_cycles, 4)                  # elementwise + LUT act
        ii_class = "elementwise+act"
    elif any(g.ops[op].op_type == "ReLU" for op in chain_ops):
        ii_hat = max(chain_cycles, 4)
        ii_class = "elementwise+act"
    elif chain_cycles > 4:
        # deep pure-elementwise chains: pre-registered law = the chain sum
        # itself (the depth-sweep designs test this directly)
        ii_hat = chain_cycles
        ii_class = "elementwise-deep"
    else:
        ii_hat = 4                                     # anchor-calibrated
        ii_class = "elementwise"
    macs = mac_count(process)
    # The model states its own uncertainty: anchored classes carry direct
    # calibration; chain-sum classes are extrapolations; very wide single
    # pipelined loops risk the measured scheduler-cost cliff.
    confidence = ("anchored" if ii_class in
                  ("elementwise", "matmul-in-loop", "feed-forward")
                  else "extrapolated")
    return {"tag": tag, "H": h, "ii_class": ii_class, "ii_hat": int(ii_hat),
            "chain_cycles": int(chain_cycles), "chain_ops": chain_ops,
            "macs": int(macs), "confidence": confidence,
            "cliff_risk": bool(macs > 1400),
            "streaming_class": streaming_class(process)}


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------

def _entry(tag, builder, cosim=True, target_override=None):
    return dict(tag=tag, builder=builder, cosim=cosim,
                target_override=target_override)


SUITE = {
    # H-sweep of the anchor families (H=16 anchors reused, not re-run)
    "rnn8":    _entry("rnn8", lambda: build_rnn(8)),
    "rnn32":   _entry("rnn32", lambda: build_rnn(32)),
    "gru8":    _entry("gru8", lambda: build_gru(8)),
    "gru32":   _entry("gru32", lambda: build_gru(32), cosim=False),
    "lstm8":   _entry("lstm8", lambda: build_lstm(8)),
    "diag8":   _entry("diag8", lambda: build_diag(8)),
    "diag32":  _entry("diag32", lambda: build_diag(32)),
    "diag64":  _entry("diag64", lambda: build_diag(64), cosim=False),
    # Cell variants at H=16
    "mgu16":   _entry("mgu16", lambda: build_mgu(16)),
    "ugrnn16": _entry("ugrnn16", lambda: build_ugrnn(16)),
    "indrnn16": _entry("indrnn16", lambda: build_indrnn(16)),
    "grurb16": _entry("grurb16", lambda: build_gru(16, reset_before=True)),
    # Tightness probe: anchor RNN with the target set BELOW its class II.
    # If the floor is really 12, Vitis must relax 8 -> 12; if it closes at
    # <=9 the raw chain sum was right and the class model needs refining.
    "rnn16probe8": _entry("rnn16probe8", lambda: build_rnn(16),
                          target_override=8),
    # Anchor rebuilds at H=16 under THIS harness (fresh draws, fixed
    # literals) - closes the loop on the original benchmark anchors.
    "rnn16":   _entry("rnn16", lambda: build_rnn(16)),
    "gru16":   _entry("gru16", lambda: build_gru(16)),
    "lstm16":  _entry("lstm16", lambda: build_lstm(16)),
    "diag16":  _entry("diag16", lambda: build_diag(16)),
    # F-sweep (input feature width) at H=16. The model's claim is sharp:
    # W.x sits OFF the recurrence loop, so II must NOT move with F while
    # DSP grows by k_mac * (added input MACs). H-sweep moved II; this must not.
    "gruf8":   _entry("gruf8", lambda: build_gru(16, f=8)),
    "gruf16":  _entry("gruf16", lambda: build_gru(16, f=16)),
    "rnnf16":  _entry("rnnf16", lambda: build_rnn(16, f=16)),
    # Widest H point for the log2 law (predicted II = 8 + 6 = 14).
    "rnn64":   _entry("rnn64", lambda: build_rnn(64), cosim=False),
    # Published-cell zoo (H=16). Cells designed for speed (SRU, QRNN,
    # minGRU) should land in shallow classes; classic gated cells
    # (JANET, FastGRNN, LiGRU) in the matmul-in-loop class; the stacked
    # GRU tests depth-invariance of II.
    "janet16":    _entry("janet16", lambda: build_janet(16)),
    "sru16":      _entry("sru16", lambda: build_sru(16)),
    "qrnn16":     _entry("qrnn16", lambda: build_qrnn(16)),
    "mingru16":   _entry("mingru16", lambda: build_mingru(16)),
    "fastgrnn16": _entry("fastgrnn16", lambda: build_fastgrnn(16)),
    "ligru16":    _entry("ligru16", lambda: build_ligru(16)),
    "gru2l16":    _entry("gru2l16", lambda: build_gru2l(16), cosim=False),
    # Floor probes: target II=1 and let the scheduler relax to the true
    # dependence-bound floor. The rnn16probe8 result (targeted 8, achieved
    # 8) showed achieved-at-target proves achievability, not tightness -
    # these probes measure the floors directly.
    "rnn16floor":    _entry("rnn16floor", lambda: build_rnn(16),
                            cosim=False, target_override=1),
    "gru16floor":    _entry("gru16floor", lambda: build_gru(16),
                            cosim=False, target_override=1),
    "diag16floor":   _entry("diag16floor", lambda: build_diag(16),
                            cosim=False, target_override=1),
    "mingru16floor": _entry("mingru16floor", lambda: build_mingru(16),
                            cosim=False, target_override=1),
    "sru16floor":    _entry("sru16floor", lambda: build_sru(16),
                            cosim=False, target_override=1),
    "indrnn16floor": _entry("indrnn16floor", lambda: build_indrnn(16),
                            cosim=False, target_override=1),
    "mgu16floor":    _entry("mgu16floor", lambda: build_mgu(16),
                            cosim=False, target_override=1),
    # SUPPLEMENTARY (non-protocol) probes for the scheduler-cost cliff:
    # gruf16 and gru32 time out at their predicted targets even on a quiet
    # machine. Do they close at slightly relaxed targets? If yes, the cliff
    # is modulo-scheduling tightness, not design size.
    "gruf16relax14": _entry("gruf16relax14", lambda: build_gru(16, f=16),
                            cosim=False, target_override=14),
    "gru32relax16":  _entry("gru32relax16", lambda: build_gru(32),
                            cosim=False, target_override=16),
    # Feed-forward generalization: a temporal CNN. No delay-edge cycle ->
    # the II theorem's feed-forward corollary predicts II = 1.
    "tcn16":   _entry("tcn16", lambda: build_tcn(16)),
    # Modern constant-state architectures (the post-2021 recurrence
    # renaissance): input-gated diagonal SSM (selective/Mamba form) and an
    # RWKV-style num/den recurrence. Both keep the state path elementwise
    # BY DESIGN - the cost model should place both in the fast class.
    "gatedssm16": _entry("gatedssm16", lambda: build_gatedssm(16)),
    "rwkv16":     _entry("rwkv16", lambda: build_rwkv(16)),
    # Depth sweep: recurrence depth isolated from every other knob.
    # Pre-registered: achieved II tracks the chain sum (2d).
    **{f"chain{d}": _entry(f"chain{d}",
                           (lambda dd: lambda: build_chain(dd))(d),
                           cosim=(d in (1, 4, 8)))
       for d in range(1, 9)},
    # Theorem L on silicon: blocked class-L cells; II=4 per k-sample block
    # -> amortized 4/k cycles per sample (0.5 at k=8).
    "blockdiag4": _entry("blockdiag4", lambda: build_blockdiag(4)),
    "blockdiag8": _entry("blockdiag8", lambda: build_blockdiag(8)),
}


def input_trace(process, tag):
    h_shapes = {sid: s.shape for sid, s in process.states.items()}
    rng = _rng(f"data_{tag}")
    f = _stream_f(process)
    data = (rng.standard_normal((TRACE_STEPS, f)) * 0.5).astype(np.float64)
    steps = []
    for i in range(TRACE_STEPS):
        steps.append(TemporalTraceStep(
            timestep=i, inputs={"x": data[i]},
            outputs={"y": np.array([0.0])},
            state={sid: np.zeros(shape) for sid, shape in h_shapes.items()}))
    return GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)),
        metadata={"case": tag, "num_steps": TRACE_STEPS})


def run_vitis(ws, tag, cmd, timeout=5400):
    """Run a Vitis stage with a timeout that kills the whole process TREE.

    subprocess.run's own timeout only kills the direct child; Vitis spawns
    grandchildren that inherit the output pipes, so communicate() blocks
    forever after the kill (measured: a wedged scheduler survived 9.5 h
    past its timeout). taskkill /T takes the tree down.
    """
    proc = subprocess.Popen(cmd, cwd=ws, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            out, err = "", "timeout: process tree killed"
        (ws / f"{tag}.log").write_text(
            (out or "") + "\n--- stderr ---\n" + (err or "")
            + f"\n--- KILLED after {timeout}s (tree kill) ---",
            encoding="utf-8")
        raise
    (ws / f"{tag}.log").write_text(
        (out or "") + "\n--- stderr ---\n" + (err or ""), encoding="utf-8")
    return proc


def parse_report(rpt_text):
    ii = loop_latency = est_clock = None
    resources = {}
    in_util = False
    for line in rpt_text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if "sample_loop" in line and len(cells) > 6 and cells[2].isdigit():
            loop_latency, ii = int(cells[2]), int(cells[5])
        if "ap_clk" in line and len(cells) > 4 and "ns" in cells[3]:
            est_clock = cells[3]
        if "Utilization Estimates" in line:
            in_util = True
        if in_util and len(cells) > 5 and cells[1] == "Total" and not resources:
            resources = {"bram": cells[2], "dsp": cells[3],
                         "ff": cells[4], "lut": cells[5]}
    return ii, loop_latency, est_clock, resources


def run_one(entry, k_mac, do_cosim=None, csim_only=False):
    tag = entry["tag"]
    process, params = entry["builder"]()
    pred = predict(tag, process)
    target_ii = entry["target_override"] or pred["ii_hat"]
    pred["target_ii"] = target_ii
    pred["dsp_hat"] = round(k_mac * pred["macs"])
    record = {"kind": "prediction", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              **{k: v for k, v in pred.items()}}
    RESULTS_DIR.mkdir(exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{tag}] predicted: II={pred['ii_hat']} ({pred['ii_class']}, "
          f"chain={pred['chain_cycles']}), DSP={pred['dsp_hat']} "
          f"({pred['macs']} MACs); target II={target_ii}")

    trace = input_trace(process, tag)
    cfg = FixedPointConfig(target_ii=target_ii, burst=TRACE_STEPS,
                           clock_ns=CLOCK_NS, part=PART)
    emit_dir = RESULTS_DIR / f"cmv_{tag}"
    info = write_fixedpoint_burst_bundle(process, trace, emit_dir, params,
                                         stem=f"cmv_{tag}", config=cfg)
    top = info["top"]

    ws = WORKSPACE_ROOT / tag
    ws.mkdir(parents=True, exist_ok=True)
    for name in (f"cmv_{tag}.cpp", f"cmv_{tag}_tb.cpp"):
        shutil.copy2(emit_dir / name, ws / name)
    cfgf = ws / "hls.cfg"
    cfgf.write_text(
        f"part={PART}\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(ws / f'cmv_{tag}.cpp').as_posix()}\n"
        f"syn.top={top}\n"
        f"tb.file={(ws / f'cmv_{tag}_tb.cpp').as_posix()}\n"
        f"clock={CLOCK_NS}ns\n", encoding="utf-8")
    vitis_run = str(VITIS_BIN / "vitis-run.bat")
    vpp = str(VITIS_BIN / "v++.bat")
    work = str(ws / "work")

    result = {"kind": "measurement", "tag": tag, "H": pred["H"],
              "target_ii": target_ii,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print(f"[{tag}] csim ...")
    csim = run_vitis(ws, "csim", [vitis_run, "--mode", "hls", "--csim",
                                  "--config", str(cfgf), "--work_dir", work])
    csim_ok = (csim.returncode == 0 and
               "errors=0" in (ws / "csim.log").read_text(encoding="utf-8"))
    result["csim"] = "pass" if csim_ok else "FAIL"
    # A csim numerics failure does not gate the quantities this campaign
    # measures -- achieved II, clock, and resources come from synthesis.
    # Record honestly and continue (unless this is a numerics-only pass).
    if not csim_ok:
        print(f"[{tag}] csim FAILED (recorded) - see {ws / 'csim.log'}")
    if csim_only:
        result["kind"] = "numerics"
        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[{tag}] numerics: csim={result['csim']}")
        return result

    print(f"[{tag}] synth ...")
    synth = run_vitis(ws, "synth", [vpp, "-c", "--mode", "hls",
                                    "--config", str(cfgf), "--work_dir", work])
    if synth.returncode != 0:
        result["synth"] = "FAIL"
        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[{tag}] synth FAILED - see {ws / 'synth.log'}")
        return result

    rpt = (ws / "work" / "hls" / "syn" / "report" / f"{top}_csynth.rpt")
    ii, loop_lat, est_clock, resources = parse_report(
        rpt.read_text(encoding="utf-8"))
    result.update({"synth": "ok", "ii_measured": ii,
                   "loop_latency": loop_lat, "est_clock": est_clock,
                   "resources": resources})

    cosim_flag = entry["cosim"] if do_cosim is None else do_cosim
    if cosim_flag:
        print(f"[{tag}] cosim ...")
        run_vitis(ws, "cosim", [vitis_run, "--mode", "hls", "--cosim",
                                "--config", str(cfgf), "--work_dir", work])
        result["cosim"] = ("PASS" if "co-simulation finished: PASS" in
                           (ws / "cosim.log").read_text(encoding="utf-8")
                           else "FAIL")

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[{tag}] measured: II={ii} (target {target_ii}), "
          f"DSP={resources.get('dsp')}, clk={est_clock}, "
          f"cosim={result.get('cosim', 'skipped')}")
    return result


def emit_lean_certificates(out_path):
    """Generate machine-checkable certificates: for every design whose
    binding recurrence is a single-state lag-1 cycle, emit a Lean file
    stating the analysis' bound as an instance of the verified iteration
    bound. `lake build` in proofs/ then re-verifies the Python analysis'
    arithmetic under the declared latency table."""
    lines = [
        "/- GENERATED by research/cost_model_validation.py --emit-lean-certs.",
        "   Each certificate instantiates the verified iteration bound with",
        "   a design's binding cycle as found by the compiler's analysis.",
        "   The bounds are relative to the analysis' declared latency table",
        "   (see OP_LAT); the physical-delay refinement is documented in",
        "   docs/cost-model-validation.md. -/",
        "import IterationBound",
        "",
    ]
    emitted = []
    for tag, entry in SUITE.items():
        if "floor" in tag or "relax" in tag or "probe" in tag:
            continue
        process, _ = entry["builder"]()
        chain_cycles, chain_ops = loop_chain(process)
        if chain_cycles == 0 or not chain_ops:
            continue
        g = process.kernels["k"].graph
        lats = [_op_latency(g.ops[op], g.values) for op in chain_ops]
        if sum(lats) != chain_cycles:
            continue  # multi-state cycle: bound already lag-divided; skip
        steps = ", ".join(f"⟨{la}, 0⟩" for la in lats) + ", ⟨0, 1⟩"
        lines += [
            f"namespace cert_{tag}",
            f"/-- {tag}: cycle {' -> '.join(chain_ops)} (closing lag-1 delay"
            " edge). -/",
            f"def cycle : List Step := [{steps}]",
            f"example {{II a : Int}} (h : Chain II a cycle a) :"
            f" {chain_cycles} ≤ II := by",
            "  have hb := iteration_bound h",
            "  simp [cycle, totalLat, totalLag] at hb",
            "  omega",
            f"end cert_{tag}",
            "",
        ]
        emitted.append(tag)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"emitted {len(emitted)} certificates -> {out_path}")
    print("  " + ", ".join(emitted))


def explain(tag):
    """Interpretability: print WHY the prediction is what it is - the
    binding recurrence path op by op, the class law applied, and the
    resource account. Everything shown is what the model actually used."""
    process, _ = SUITE[tag]["builder"]()
    pred = predict(tag, process)
    g = process.kernels["k"].graph
    print(f"{tag}: predicted II = {pred['ii_hat']}  "
          f"({pred['ii_class']}, confidence: {pred['confidence']})")
    sc = pred["streaming_class"]
    meaning = {
        "FF": "no cyclic state - trivially parallel, no latency floor",
        "L-affine": "state-affine update - NO fundamental streaming floor "
                    "(blocking theorem); any floor is implementation choice",
        "N-nonlinear": "state-nonlinear update - irreducible sequential "
                       "core per sample (degree bound)",
    }[sc]
    print(f"  streaming class: {sc} - {meaning}")
    print("    (docs/streaming-latency-classes.md; engines machine-checked"
          " in proofs/StreamingClasses.lean)")
    if pred["chain_ops"]:
        print("  binding recurrence path (state read -> state write):")
        for op in pred["chain_ops"]:
            o = g.ops[op]
            print(f"    {o.op_type:<9} {op:<12} "
                  f"+{_op_latency(o, g.values)} cycle(s)")
        print(f"    = chain {pred['chain_cycles']} cycles"
              f" -> class law gives II = {pred['ii_hat']}")
    else:
        print("  no delay-edge cycle: feed-forward, II bounded only by"
              " resources (theorem's corollary)")
    k_mac = calibrate_k_mac()
    print(f"  resources: {pred['macs']} weighted MACs x {k_mac:.2f} DSP/MAC"
          f" = {round(k_mac * pred['macs'])} DSP predicted"
          f"{'  [cliff risk: wide single loop]' if pred['cliff_risk'] else ''}")
    print(f"  per-sample latency at 5 ns clock: {pred['ii_hat'] * 5} ns; "
          f"proved floor: II*Lambda >= chain (proofs/), "
          f"area wall: DSP >= ceil(MACs/II) = "
          f"{-(-pred['macs'] // max(pred['ii_hat'], 1))}")


def budget_select(dsp_budget):
    """Hardware-aware architecture selection v0: rank every suite design by
    predicted per-sample latency subject to a DSP budget, from the cost
    model alone - no synthesis in the loop. Measured designs are marked."""
    k_mac = calibrate_k_mac()
    meas = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["kind"] == "measurement" and r.get("ii_measured"):
                meas[r["tag"]] = r
    rows = []
    for tag, entry in SUITE.items():
        if "floor" in tag or "relax" in tag or "probe" in tag:
            continue
        p, _ = entry["builder"]()
        pr = predict(tag, p)
        dsp = round(k_mac * pr["macs"])
        if dsp <= dsp_budget:
            rows.append((pr["ii_hat"] * 5, dsp, tag, pr, tag in meas))
    rows.sort()
    print(f"designs predicted to fit {dsp_budget} DSPs, fastest first:")
    print(f"{'ns/sample':>10}{'DSP':>6}  {'design':<14}{'class':<18}"
          f"{'confidence':<13}status")
    for ns, dsp, tag, pr, verified in rows:
        status = "measured+verified" if verified else "prediction only"
        risk = " (cliff risk)" if pr["cliff_risk"] else ""
        print(f"{ns:>10}{dsp:>6}  {tag:<14}{pr['ii_class']:<18}"
              f"{pr['confidence']:<13}{status}{risk}")


def calibrate_k_mac():
    """DSP-per-MAC from the RNN@16 anchor only (289 DSP)."""
    p, _ = build_rnn(16)
    return ANCHORS["rnn16"]["DSP"] / mac_count(p)


def print_predictions(k_mac):
    print(f"k_mac (DSP per weighted MAC, RNN@16 calibration): {k_mac:.4f}\n")
    print(f"{'tag':<12}{'H':>4}{'class':>18}{'chain':>7}{'II_hat':>8}"
          f"{'MACs':>7}{'DSP_hat':>9}")
    for entry in SUITE.values():
        p, _ = entry["builder"]()
        pr = predict(entry["tag"], p)
        print(f"{pr['tag']:<12}{pr['H']:>4}{pr['ii_class']:>18}"
              f"{pr['chain_cycles']:>7}{pr['ii_hat']:>8}{pr['macs']:>7}"
              f"{round(k_mac * pr['macs']):>9}")


def print_report():
    if not RESULTS_FILE.exists():
        print("no results yet")
        return
    preds, meas, num = {}, {}, {}
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["kind"] == "prediction":
            preds[r["tag"]] = r
        elif r["kind"] == "numerics":
            num[r["tag"]] = r
        else:
            # keep the latest row that actually carries synthesis numbers
            if r.get("ii_measured") is not None or r["tag"] not in meas:
                meas[r["tag"]] = r
    print(f"{'tag':<12}{'II_hat':>7}{'II_meas':>8}{'match':>7}"
          f"{'DSP_hat':>9}{'DSP_meas':>9}{'err':>8}  csim(fixed)  cosim")
    for tag, pr in preds.items():
        m = meas.get(tag, {})
        ii_m = m.get("ii_measured")
        dsp_m = m.get("resources", {}).get("dsp")
        match = "-" if ii_m is None else ("OK" if ii_m == pr["ii_hat"] else "MISS")
        err = "-"
        if dsp_m and str(dsp_m).isdigit():
            err = f"{100 * (pr['dsp_hat'] - int(dsp_m)) / int(dsp_m):+.1f}%"
        nv = num.get(tag, {}).get("csim", "-")
        print(f"{tag:<12}{pr['ii_hat']:>7}{str(ii_m):>8}{match:>7}"
              f"{pr['dsp_hat']:>9}{str(dsp_m):>9}{err:>8}  {nv:>11}  "
              f"{m.get('cosim', m.get('csim', '-'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["round1"], default=None)
    ap.add_argument("--only", default=None, help="run one suite entry by tag")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-cosim", action="store_true")
    ap.add_argument("--csim-only", action="store_true",
                    help="numerics verification: emit + csim, skip synth/cosim")
    ap.add_argument("--emit-lean-certs", action="store_true",
                    help="write proofs/Certificates.lean from the analysis")
    ap.add_argument("--explain", default=None, metavar="TAG",
                    help="print WHY a design's prediction is what it is")
    ap.add_argument("--budget", type=int, default=None, metavar="DSP",
                    help="rank designs predicted to fit a DSP budget")
    ap.add_argument("--tags", default=None,
                    help="comma-separated suite tags to run")
    args = ap.parse_args()

    k_mac = calibrate_k_mac()
    if args.explain:
        explain(args.explain)
        return
    if args.budget is not None:
        budget_select(args.budget)
        return
    if args.emit_lean_certs:
        emit_lean_certificates(REPO / "proofs" / "Certificates.lean")
        return
    if args.report:
        print_report()
        return
    if args.predict_only:
        print_predictions(k_mac)
        return

    if args.only:
        tags = [args.only]
    elif args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = list(SUITE)
    print(f"campaign: {len(tags)} designs -> {RESULTS_FILE}")
    for tag in tags:
        try:
            run_one(SUITE[tag], k_mac,
                    do_cosim=False if args.no_cosim else None,
                    csim_only=args.csim_only)
        except Exception as exc:  # keep the campaign going; log the casualty
            with RESULTS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "measurement", "tag": tag,
                                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "error": str(exc)[:500]}) + "\n")
            print(f"[{tag}] ERROR: {exc}")
    print_report()


if __name__ == "__main__":
    main()
