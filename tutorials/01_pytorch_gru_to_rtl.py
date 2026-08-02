# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Tutorial 1 — from a trained PyTorch GRU to RTL simulation
#
# This notebook walks the full deployment path once, end to end:
#
# 1. train a small GRU forecaster in PyTorch,
# 2. pull its trained weights out and build the equivalent TempoDAG
#    temporal process (the compiler's representation of a streaming model),
# 3. check — numerically, not on faith — that the process computes exactly
#    what the trained model computes,
# 4. emit the fixed-point HLS design, and
# 5. (if Vitis is installed) push it through C simulation, synthesis, and
#    RTL co-simulation.
#
# Steps 1–4 run anywhere in a few seconds; step 5 needs AMD's toolchain and
# takes a few minutes. Nothing here assumes you have read the compiler
# internals — where a detail matters, it is explained in place.
#
# One warning worth giving up front, because it is the single most common
# way to get a "working" deployment that silently computes the wrong thing:
# **every framework orders its gate weights differently.** PyTorch stores
# GRU gates as (reset, update, new); ONNX stores (update, reset, hidden);
# Keras stores (update, reset, hidden) but transposed. And PyTorch applies
# the reset gate only to the *hidden-side* term of the candidate, keeping
# two separate bias vectors. This tutorial builds the PyTorch-exact cell and
# then proves the numbers match, so layout mistakes cannot hide.

# %%
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(REPO / "src"))

torch.manual_seed(7)
rng = np.random.default_rng(7)

H = 16  # hidden size
F = 4  # input features
OUT_DIR = (
    Path(__file__).resolve().parent / "output"
    if "__file__" in dir()
    else Path.cwd() / "output"
)

# %% [markdown]
# ## 1. Train a small forecaster
#
# The task is deliberately simple: predict the next value of a noisy
# multi-frequency signal from four lagged features. The point of this
# notebook is the deployment path, not the model — any GRU you already
# have will do, as long as you can reach its `state_dict`.


# %%
def make_stream(n):
    t = np.arange(n)
    sig = np.sin(0.07 * t) + 0.5 * np.sin(0.23 * t) + 0.1 * rng.standard_normal(n)
    x = np.stack([np.roll(sig, k) for k in range(1, F + 1)], axis=1)
    return x[F:].astype(np.float32), sig[F:].astype(np.float32)


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(F, H, batch_first=True)
        self.head = nn.Linear(H, 1)

    def forward(self, x, h0=None):
        out, hn = self.gru(x, h0)
        return self.head(out), hn


x_train, y_train = make_stream(2200)
model = Forecaster()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
xb = torch.from_numpy(x_train[None, :2000])
yb = torch.from_numpy(y_train[None, :2000, None])
for epoch in range(60):
    opt.zero_grad()
    pred, _ = model(xb)
    loss = ((pred - yb) ** 2).mean()
    loss.backward()
    opt.step()
print(f"final training MSE: {loss.item():.5f}")
model.eval()

# %% [markdown]
# ## 2. Unpack the trained weights
#
# PyTorch keeps all three gates stacked in single matrices:
# `weight_ih_l0` is `[3H, F]` with rows ordered **(r, z, n)**, and likewise
# `weight_hh_l0` is `[3H, H]`. There are *four* bias vectors — input-side
# and hidden-side for each gate — and for the candidate gate `n` the two
# must stay separate, because the reset gate multiplies only the
# hidden-side term:
#
#     r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
#     z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
#     n = tanh(  W_in x + b_in + r * (W_hn h + b_hn) )
#     h' = (1 - z) * n + z * h
#
# For r and z the two biases just add, so we merge them into one vector.

# %%
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}


def torch_gru_weights(sd):
    Wi, Wh = sd["gru.weight_ih_l0"], sd["gru.weight_hh_l0"]
    bi, bh = sd["gru.bias_ih_l0"], sd["gru.bias_hh_l0"]
    g = {}
    for i, name in enumerate(("r", "z", "n")):  # PyTorch gate order
        sl = slice(i * H, (i + 1) * H)
        g[name] = dict(Wx=Wi[sl], Wh=Wh[sl], bx=bi[sl], bh=bh[sl])
    return g


gates = torch_gru_weights(sd)
Wo, bo = sd["head.weight"], sd["head.bias"]
print({k: {kk: vv.shape for kk, vv in v.items()} for k, v in gates.items()})

# %% [markdown]
# ## 3. Build the temporal process
#
# TempoDAG represents a streaming model as a *process*: an acyclic compute
# graph for one timestep, plus explicitly declared state and a delay edge
# that says "the `h_next` written this step is the `h_prev` read next
# step". That delay edge is the only legal way to close a cycle — which is
# exactly how hardware feedback works (combinational logic is acyclic;
# loops close through registers).
#
# The construction below is verbose on purpose: every op you see becomes a
# concrete piece of arithmetic in the generated hardware, so there is
# value in seeing the cell written out once, gate by gate.

# %%
from tempo_dag.ir.graph import Graph
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
)
from tempo_dag.ops.builtins import Add, MatMul, Mul, Sigmoid, Sub, Tanh


def val(vid, shape, layout=None):
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"a{i}" for i in range(len(shape))],
        layout=layout,
    )


values = {
    "x": val("x", [1, F]),
    "h_prev": val("h_prev", [1, H]),
    "h_next": val("h_next", [1, H]),
    "ones": val("ones", [1, H], layout="parameter"),
    "y0": val("y0", [1, 1]),
    "y": val("y", [1, 1]),
    "WoT": val("WoT", [H, 1], layout="parameter"),
    "bo": val("bo", [1, 1], layout="parameter"),
}
ops = {}
params = {
    "ones": np.ones((1, H), np.float32),
    "WoT": Wo.T.copy(),
    "bo": bo[None, :].copy(),
}

# r and z gates: x@WxT + h@WhT + b  ->  sigmoid
for g in ("r", "z"):
    for vid in (f"xg_{g}", f"hg_{g}", f"sg_{g}", f"pre_{g}", f"{g}g"):
        values[vid] = val(vid, [1, H])
    values[f"WxT_{g}"] = val(f"WxT_{g}", [F, H], layout="parameter")
    values[f"WhT_{g}"] = val(f"WhT_{g}", [H, H], layout="parameter")
    values[f"b_{g}"] = val(f"b_{g}", [1, H], layout="parameter")
    ops[f"mmx_{g}"] = MatMul(f"mmx_{g}", inputs=["x", f"WxT_{g}"], outputs=[f"xg_{g}"])
    ops[f"mmh_{g}"] = MatMul(
        f"mmh_{g}", inputs=["h_prev", f"WhT_{g}"], outputs=[f"hg_{g}"]
    )
    ops[f"gs_{g}"] = Add(f"gs_{g}", inputs=[f"xg_{g}", f"hg_{g}"], outputs=[f"sg_{g}"])
    ops[f"gb_{g}"] = Add(f"gb_{g}", inputs=[f"sg_{g}", f"b_{g}"], outputs=[f"pre_{g}"])
    ops[f"sig_{g}"] = Sigmoid(f"sig_{g}", inputs=[f"pre_{g}"], outputs=[f"{g}g"])
    params[f"WxT_{g}"] = gates[g]["Wx"].T.copy()
    params[f"WhT_{g}"] = gates[g]["Wh"].T.copy()
    params[f"b_{g}"] = (gates[g]["bx"] + gates[g]["bh"])[None, :].copy()

# candidate gate, PyTorch-exact: n = tanh(xg_n + b_in + r * (hg_n + b_hn))
for vid in ("xg_n", "xgb_n", "hg_n", "hgb_n", "nr", "pre_n", "n"):
    values[vid] = val(vid, [1, H])
for vid, shape in (
    ("WxT_n", [F, H]),
    ("WhT_n", [H, H]),
    ("bx_n", [1, H]),
    ("bh_n", [1, H]),
):
    values[vid] = val(vid, shape, layout="parameter")
ops["mmx_n"] = MatMul("mmx_n", inputs=["x", "WxT_n"], outputs=["xg_n"])
ops["bx_add"] = Add("bx_add", inputs=["xg_n", "bx_n"], outputs=["xgb_n"])
ops["mmh_n"] = MatMul("mmh_n", inputs=["h_prev", "WhT_n"], outputs=["hg_n"])
ops["bh_add"] = Add("bh_add", inputs=["hg_n", "bh_n"], outputs=["hgb_n"])
ops["mul_nr"] = Mul("mul_nr", inputs=["rg", "hgb_n"], outputs=["nr"])
ops["pre_add"] = Add("pre_add", inputs=["xgb_n", "nr"], outputs=["pre_n"])
ops["tanh_n"] = Tanh("tanh_n", inputs=["pre_n"], outputs=["n"])
params["WxT_n"] = gates["n"]["Wx"].T.copy()
params["WhT_n"] = gates["n"]["Wh"].T.copy()
params["bx_n"] = gates["n"]["bx"][None, :].copy()
params["bh_n"] = gates["n"]["bh"][None, :].copy()

# state blend + linear head
for vid in ("omz", "part_a", "part_b"):
    values[vid] = val(vid, [1, H])
ops["one_minus"] = Sub("one_minus", inputs=["ones", "zg"], outputs=["omz"])
ops["mul_a"] = Mul("mul_a", inputs=["omz", "n"], outputs=["part_a"])
ops["mul_b"] = Mul("mul_b", inputs=["zg", "h_prev"], outputs=["part_b"])
ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
ops["mm_out"] = MatMul("mm_out", inputs=["h_next", "WoT"], outputs=["y0"])
ops["add_out"] = Add("add_out", inputs=["y0", "bo"], outputs=["y"])

graph = Graph(
    values=values,
    ops=ops,
    graph_inputs=["x", "h_prev"]
    + [k for k, v in values.items() if v.layout == "parameter"],
    graph_outputs=["y", "h_next"],
)
process = Process(
    process_id="tutorial_gru",
    kernels={"k": Kernel("k", graph=graph)},
    states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
    edge0=[Edge0("h", "k", value_id="h_prev")],
    edge_delta=[EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
)
process.validate()
print(
    "process validates: acyclic within a timestep, "
    "recurrence only through the delay edge"
)

# %% [markdown]
# ## 4. Prove the process computes what the trained model computes
#
# Before generating any hardware it is worth proving that the graph we
# just built *is* the trained model — this is where a gate-order or bias
# mistake would surface. The interpreter below is twenty lines: evaluate
# ops in dependency order, feed `h_next` back as `h_prev`, repeat. If the
# graph and the weight unpacking are right, it must track PyTorch to
# float32 round-off.

# %%
SEMANTICS = {
    "MatMul": lambda a, b: a @ b,
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "Tanh": np.tanh,
}


def run_process(process, params, xs):
    g = process.kernels["k"].graph
    h = np.zeros((1, H), np.float32)
    ys = []
    for x in xs:
        env = {
            "x": x[None, :].astype(np.float32),
            "h_prev": h,
            **{k: np.asarray(v, np.float32) for k, v in params.items()},
        }
        pending = dict(g.ops)
        while pending:
            for name, op in list(pending.items()):
                if all(i in env for i in op.inputs):
                    args = [env[i] for i in op.inputs]
                    fn = SEMANTICS[op.op_type]
                    env[op.outputs[0]] = fn(*args)
                    del pending[name]
        h = env["h_next"]
        ys.append(float(env["y"][0, 0]))
    return np.array(ys)


x_test = x_train[2000:2064]
with torch.no_grad():
    y_torch, _ = model(torch.from_numpy(x_test[None]))
y_torch = y_torch.numpy().ravel()
y_ir = run_process(process, params, x_test)
gap = np.abs(y_torch - y_ir).max()
print(f"max |torch - IR| over 64 streaming steps: {gap:.2e}")
assert gap < 1e-5, "the process does NOT match the trained model"

# %% [markdown]
# The gap is float round-off, so the process and the trained network are
# the same function. Everything from here on is mechanical.
#
# ## 5. Emit the fixed-point hardware design
#
# The emitter takes the process, the trained parameters, and a short input
# trace, and produces two C++ files: the design (an `ap_fixed` datapath
# with the weights baked in as constants and tanh/sigmoid as small lookup
# tables, wrapped in a pipelined sample loop) and an asserting testbench.
# The testbench's expected outputs come from a fixed-point oracle that
# mirrors the hardware's rounding exactly, so a pass means "the hardware
# matches its specification to the last rounding unit", not "close
# enough".

# %%
from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (
    FixedPointConfig,
    write_fixedpoint_burst_bundle,
)
from tempo_dag.verification.golden_trace import GoldenTraceRecorder
from tempo_dag.verification.temporal_parity import (
    TemporalExecutionTrace,
    TemporalTraceStep,
)

steps = [
    TemporalTraceStep(
        timestep=i,
        inputs={"x": x_test[i].astype(np.float64)},
        outputs={"y": np.array([0.0])},
        state={"h": np.zeros(H)},
    )
    for i in range(64)
]
trace = GoldenTraceRecorder().record(
    TemporalExecutionTrace(tuple(steps)),
    metadata={"case": "tutorial_gru", "num_steps": 64},
)

cfg = FixedPointConfig(target_ii=12, burst=64, clock_ns=5.0, part="xck26-sfvc784-2LV-c")
info = write_fixedpoint_burst_bundle(
    process, trace, OUT_DIR, params, stem="tutorial_gru", config=cfg
)
print(f"emitted: {OUT_DIR / 'tutorial_gru.cpp'}")
print(f"top function: {info['top']}")

# %% [markdown]
# Two things are worth finding in the generated file. First, the pipeline
# pragma on the sample loop — this is the line that turns "one 440-cycle
# step at a time" into "a new sample every 12 cycles":

# %%
src = (OUT_DIR / "tutorial_gru.cpp").read_text(encoding="utf-8")
for line in src.splitlines():
    if "PIPELINE" in line or "sample_loop" in line:
        print(line.rstrip())

# %% [markdown]
# Second, the datapath type and the activation tables — `ap_fixed<18,6>`
# arithmetic and a 512-entry tanh ROM indexed by an integer bit-slice of
# the pre-activation. Cheap, fast, and exactly mirrored by the oracle that
# generated the testbench's expected values.
#
# ## 6. RTL simulation (needs Vitis)
#
# The last step hands the design to AMD's toolchain: C simulation asserts
# the design against the oracle, synthesis reports the achieved initiation
# interval / clock / resources, and C/RTL co-simulation runs the actual
# generated circuit against the same assertions. Set `VITIS_BIN` if your
# install lives elsewhere, and flip `RUN_VITIS = True` (it is off by
# default so the notebook runs everywhere).

# %%
import os
import shutil
import subprocess

RUN_VITIS = os.environ.get("TUTORIAL_RUN_VITIS", "0") == "1"
VITIS_BIN = Path(os.environ.get("VITIS_BIN", "C:/AMDDesignTools/2026.1/Vitis/bin"))

if RUN_VITIS and VITIS_BIN.exists():
    ws = Path(os.environ.get("TEMPODAG_WORKSPACE", "C:/tmp/tempodag_tutorial"))
    ws.mkdir(parents=True, exist_ok=True)
    for name in ("tutorial_gru.cpp", "tutorial_gru_tb.cpp"):
        shutil.copy2(OUT_DIR / name, ws / name)
    (ws / "hls.cfg").write_text(
        f"part={cfg.part}\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(ws / 'tutorial_gru.cpp').as_posix()}\n"
        f"syn.top={info['top']}\n"
        f"tb.file={(ws / 'tutorial_gru_tb.cpp').as_posix()}\n"
        f"clock={cfg.clock_ns}ns\n",
        encoding="utf-8",
    )
    for stage, cmd in [
        (
            "csim",
            [
                str(VITIS_BIN / "vitis-run.bat"),
                "--mode",
                "hls",
                "--csim",
                "--config",
                str(ws / "hls.cfg"),
                "--work_dir",
                str(ws / "work"),
            ],
        ),
        (
            "synth",
            [
                str(VITIS_BIN / "v++.bat"),
                "-c",
                "--mode",
                "hls",
                "--config",
                str(ws / "hls.cfg"),
                "--work_dir",
                str(ws / "work"),
            ],
        ),
        (
            "cosim",
            [
                str(VITIS_BIN / "vitis-run.bat"),
                "--mode",
                "hls",
                "--cosim",
                "--config",
                str(ws / "hls.cfg"),
                "--work_dir",
                str(ws / "work"),
            ],
        ),
    ]:
        print(f"{stage} ...")
        r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=3600)
        print(f"  -> exit {r.returncode}")
    rpt = ws / "work" / "hls" / "syn" / "report" / f"{info['top']}_csynth.rpt"
    if rpt.exists():
        for line in rpt.read_text(encoding="utf-8").splitlines():
            if "sample_loop" in line or ("ap_clk" in line and "ns" in line):
                print(line.rstrip())
else:
    print(
        "Vitis stage skipped. To run it: set TUTORIAL_RUN_VITIS=1 "
        "(and VITIS_BIN if needed) and re-run this cell."
    )
    print(
        "Expected result at this configuration: achieved II = 12, "
        "estimated clock ~3.6 ns, C/RTL co-simulation PASS "
        "-> 60 ns per streaming sample."
    )

# %% [markdown]
# ## 7. The same weights from TensorFlow or ONNX
#
# Everything after step 3 is frontend-independent — the *only* thing that
# changes is how you unpack the gate matrices. The two functions below do
# it for Keras and ONNX; note the different gate orders, which are the
# whole trap:
#
# | frontend | gate order | layout |
# |---|---|---|
# | PyTorch  | r, z, n | `weight_ih_l0[3H, F]`, reset applied to hidden term only |
# | ONNX GRU | z, r, h | `W[1, 3H, F]`, `linear_before_reset` controls the reset form |
# | Keras    | z, r, h | `kernel[F, 3H]` (transposed relative to PyTorch) |


# %%
def onnx_gru_weights(onnx_path):
    """Gate dict from an ONNX file (e.g. torch.onnx.export of the model)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    Wname = next(
        n
        for n in init
        if init[n].ndim == 3 and init[n].shape[1] == 3 * H and init[n].shape[2] == F
    )
    Rname = next(
        n
        for n in init
        if init[n].ndim == 3 and init[n].shape[1] == 3 * H and init[n].shape[2] == H
    )
    W, R = init[Wname][0], init[Rname][0]
    g = {}
    for i, name in enumerate(("z", "r", "n")):  # ONNX gate order: z, r, h
        sl = slice(i * H, (i + 1) * H)
        g[name] = dict(Wx=W[sl], Wh=R[sl])
    return g


def keras_gru_weights(keras_gru_layer):
    """Gate dict from a tf.keras GRU layer (order z, r, h; transposed)."""
    kernel, recurrent, bias = keras_gru_layer.get_weights()
    g = {}
    for i, name in enumerate(("z", "r", "n")):
        sl = slice(i * H, (i + 1) * H)
        g[name] = dict(
            Wx=kernel[:, sl].T,
            Wh=recurrent[:, sl].T,
            bx=bias[0][sl] if bias.ndim == 2 else bias[sl],
            bh=bias[1][sl] if bias.ndim == 2 else np.zeros(H),
        )
    return g


# Demonstrate the ONNX route by exporting the torch model and re-reading it:
onnx_path = OUT_DIR / "tutorial_gru.onnx"
try:
    # dynamo=False selects the stable TorchScript exporter (the default
    # dynamo path also prints non-ASCII progress glyphs that trip Windows
    # consoles).
    torch.onnx.export(model.gru, (torch.zeros(1, 1, F),), str(onnx_path), dynamo=False)
    g_onnx = onnx_gru_weights(onnx_path)
    drift = max(np.abs(g_onnx[k]["Wx"] - gates[k]["Wx"]).max() for k in ("r", "z", "n"))
    print(f"ONNX round-trip: gate matrices match torch to {drift:.1e}")
except Exception as exc:  # older torch exporters vary; the mapping is the point
    print(f"ONNX export skipped on this setup: {exc}")

# %% [markdown]
# ## Where to go from here
#
# - Swap in your own trained GRU: only steps 1–2 change.
# - A different cell (LSTM, minimal-gated, a diagonal state-space model)
#   means a different step-3 graph; the builders in
#   `experiments/benchmark/tempodag_backend.py` and
#   `research/cost_model_validation.py` are working examples of every
#   family the compiler currently emits.
# - `research/walkthrough/` explains *why* the emitted design is fast
#   (you pay for the recurrence loop, not the whole step) and shows the
#   accuracy of a trained model surviving the fixed-point deploy.
