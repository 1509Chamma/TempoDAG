# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Tutorial 2 — adding a custom operator
#
# The built-in operator set (matmul, elementwise arithmetic, tanh/sigmoid
# LUTs, ReLU) covers the standard cells, but research means wanting something
# the library doesn't have. This tutorial adds a new elementwise operator —
# LeakyReLU — end to end: the IR node, the hardware implementation, and the
# verification semantics, all registered through one object.
#
# The design principle worth internalizing before the code: **an operator is
# not "added" until the verification oracle knows its exact arithmetic.** The
# emitter and the oracle extend together through `CustomFixedPointOp`, so the
# oracle-relative certificate — the thing that makes every number in this
# repository checkable — covers custom logic exactly like the built-ins. If
# the C function and the Python semantics disagree by even one least
# significant bit, C-simulation fails loudly rather than shipping a silent
# mismatch.

# %%
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(REPO / "src"))

H, F = 8, 4
OUT_DIR = (
    (Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
    / "output"
    / "custom_op"
)

# %% [markdown]
# ## 1. The IR node
#
# An operator class gives the compiler shape/type checking for the new node.
# For an elementwise unary op the base class does the work — subclassing and
# naming it is enough.

# %%
from tempo_dag.ops.builtins import Add, MatMul, UnaryElementwiseOperator


class LeakyReLU(UnaryElementwiseOperator):
    """max(x, 0.25*x). Alpha is 0.25 deliberately: a power of two, so the
    fixed-point multiply is an exact shift and the semantics stay crisp."""

    OP_TYPE = "LeakyReLU"


# %% [markdown]
# ## 2. The hardware + verification pair
#
# `CustomFixedPointOp` carries the two implementations that must agree:
#
# - `c_body` — the C function the emitter places in the design. It can use
#   the design's `fx` (the Q-format word) and `acc_t` (the wide accumulator)
#   types. Here: negative inputs are multiplied by 0.25 and truncated back
#   to `fx` (the cast drops low bits toward negative infinity, `ap_fixed`'s
#   default), positives pass through.
# - `semantics` — the same function in NumPy, on the same grid, with the
#   same truncation. `np.floor(v * 0.25 * 4096) / 4096` is exactly the
#   C cast's behaviour for Q6.12.

# %%
from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (
    CustomFixedPointOp,
    FixedPointConfig,
    write_fixedpoint_burst_bundle,
)

FRAC_SCALE = 1 << 12  # Q6.12


def leaky_relu_semantics(v):
    v = np.asarray(v, np.float64)
    neg = np.floor(v * 0.25 * FRAC_SCALE) / FRAC_SCALE
    return np.where(v < 0, neg, v)


LEAKY = CustomFixedPointOp(
    c_name="leaky_relu_fx",
    c_body=(
        "static fx leaky_relu_fx(acc_t x) {\n"
        "#pragma HLS INLINE\n"
        "  const fx quarter = 0.25;\n"
        "  return (x < 0) ? (fx)(x * quarter) : (fx)x;\n"
        "}"
    ),
    semantics=leaky_relu_semantics,
)

# %% [markdown]
# ## 3. Use it in a cell
#
# A minimal IndRNN-style cell with the new activation on the recurrence
# path: `h' = leaky_relu(u ⊙ h + W x + b)`. Everything below is the standard
# process-building pattern from tutorial 1 — the only new thing is the
# `LeakyReLU` node.

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
from tempo_dag.ops.builtins import Mul


def val(vid, shape, layout=None):
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"a{i}" for i in range(len(shape))],
        layout=layout,
    )


rng = np.random.default_rng(2)
values = {
    "x": val("x", [1, F]),
    "h_prev": val("h_prev", [1, H]),
    "h_next": val("h_next", [1, H]),
    "u": val("u", [1, H], layout="parameter"),
    "WxT": val("WxT", [F, H], layout="parameter"),
    "b": val("b", [1, H], layout="parameter"),
    "uh": val("uh", [1, H]),
    "wx": val("wx", [1, H]),
    "s1": val("s1", [1, H]),
    "pre": val("pre", [1, H]),
    "y0": val("y0", [1, 1]),
    "y": val("y", [1, 1]),
    "WoT": val("WoT", [H, 1], layout="parameter"),
    "bo": val("bo", [1, 1], layout="parameter"),
}
ops = {
    "mul_uh": Mul("mul_uh", inputs=["u", "h_prev"], outputs=["uh"]),
    "mm_wx": MatMul("mm_wx", inputs=["x", "WxT"], outputs=["wx"]),
    "s_add": Add("s_add", inputs=["uh", "wx"], outputs=["s1"]),
    "b_add": Add("b_add", inputs=["s1", "b"], outputs=["pre"]),
    "act": LeakyReLU("act", inputs=["pre"], outputs=["h_next"]),
    "mm_out": MatMul("mm_out", inputs=["h_next", "WoT"], outputs=["y0"]),
    "add_out": Add("add_out", inputs=["y0", "bo"], outputs=["y"]),
}
graph = Graph(
    values=values,
    ops=ops,
    graph_inputs=["x", "h_prev"]
    + [k for k, v in values.items() if v.layout == "parameter"],
    graph_outputs=["y", "h_next"],
)
process = Process(
    process_id="custom_cell",
    kernels={"k": Kernel("k", graph=graph)},
    states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
    edge0=[Edge0("h", "k", value_id="h_prev")],
    edge_delta=[EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
)
process.validate()
params = {
    "u": rng.uniform(0.3, 0.9, (1, H)).astype(np.float32),
    "WxT": (rng.standard_normal((F, H)) * 0.4).astype(np.float32),
    "b": (rng.standard_normal((1, H)) * 0.1).astype(np.float32),
    "WoT": (rng.standard_normal((H, 1)) * 0.4).astype(np.float32),
    "bo": [[0.0]],
}
print(
    "process validates; recurrence cycle:",
    "u*h -> add -> add -> LeakyReLU -> h  (elementwise + activation class)",
)

# %% [markdown]
# ## 4. Emit — the custom op rides the whole pipeline
#
# Registering `{"LeakyReLU": LEAKY}` on the config is the only change from
# the standard flow. The emitter places `leaky_relu_fx` in the design's
# prelude and calls it per lane; the oracle uses `semantics` to produce the
# testbench's expected values. The generated testbench therefore *asserts
# the custom op's exact arithmetic* — run C-simulation and any divergence
# between the two implementations shows up as a nonzero error count.

# %%
from tempo_dag.verification.golden_trace import GoldenTraceRecorder
from tempo_dag.verification.temporal_parity import (
    TemporalExecutionTrace,
    TemporalTraceStep,
)

xs = (rng.standard_normal((64, F)) * 0.5).astype(np.float64)
steps = [
    TemporalTraceStep(
        timestep=i,
        inputs={"x": xs[i]},
        outputs={"y": np.array([0.0])},
        state={"h": np.zeros(H)},
    )
    for i in range(64)
]
trace = GoldenTraceRecorder().record(
    TemporalExecutionTrace(tuple(steps)), metadata={"case": "custom_cell"}
)

cfg = FixedPointConfig(target_ii=5, burst=64, custom_ops={"LeakyReLU": LEAKY})
info = write_fixedpoint_burst_bundle(
    process, trace, OUT_DIR, params, stem="custom_cell", config=cfg
)
src = (OUT_DIR / "custom_cell.cpp").read_text(encoding="utf-8")
assert "leaky_relu_fx" in src
print(f"emitted {info['top']}; custom function present in the design:")
for line in src.splitlines():
    if "leaky_relu_fx" in line:
        print(" ", line.strip())

# %% [markdown]
# ## 5. Verify (needs Vitis)
#
# As in tutorial 1, the hardware stages are optional and off by default.
# With Vitis installed, set `TUTORIAL_RUN_VITIS=1` and re-run: C-simulation
# asserts every output against the oracle (`errors=0` = the C body and the
# NumPy semantics agree on every tested lane and timestep), and synthesis
# reports the achieved II — this cell's recurrence is elementwise + an
# activation, the same structural class the cost model predicts at II≈5.

# %%
import os
import shutil
import subprocess

RUN_VITIS = os.environ.get("TUTORIAL_RUN_VITIS", "0") == "1"
VITIS_BIN = Path(os.environ.get("VITIS_BIN", "C:/AMDDesignTools/2026.1/Vitis/bin"))
if RUN_VITIS and VITIS_BIN.exists():
    ws = Path(os.environ.get("TEMPODAG_WORKSPACE", "C:/tmp/tempodag_tutorial2"))
    ws.mkdir(parents=True, exist_ok=True)
    for name in ("custom_cell.cpp", "custom_cell_tb.cpp"):
        shutil.copy2(OUT_DIR / name, ws / name)
    (ws / "hls.cfg").write_text(
        f"part={cfg.part}\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(ws / 'custom_cell.cpp').as_posix()}\n"
        f"syn.top={info['top']}\n"
        f"tb.file={(ws / 'custom_cell_tb.cpp').as_posix()}\n"
        f"clock={cfg.clock_ns}ns\n",
        encoding="utf-8",
    )
    r = subprocess.run(
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
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    ok = "errors=0" in r.stdout
    print(f"csim errors=0: {ok}")
else:
    print("Vitis stage skipped (set TUTORIAL_RUN_VITIS=1 to run it).")
    print("Expected: csim errors=0 - the C body and the NumPy semantics agree.")

# %% [markdown]
# ## What to take away
#
# - **One object, two implementations, one gate.** A custom operator is a C
#   body plus its exact NumPy mirror; the testbench enforces their
#   agreement, so extending the compiler never weakens the verification
#   story.
# - **The cost model applies automatically.** The scheduling analysis reads
#   graph structure, not an op whitelist — a custom activation on the
#   recurrence path lands in the elementwise+activation class with its II
#   predicted like any built-in.
# - **Non-elementwise ops** (new reductions, table lookups with state,
#   multi-input kernels) need an emitter branch of their own — the
#   `MatMul` case in `temporal_fixedpoint_generator.py` is the template to
#   follow, and the same rule applies: no emitter change without its oracle
#   mirror.
