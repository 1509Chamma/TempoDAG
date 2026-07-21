"""Temporal architecture zoo (validated in NB05) as reusable Process builders."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tempo_dag.ir.graph import Graph  # noqa: E402
from tempo_dag.ir.value import Value, ValueType  # noqa: E402
from tempo_dag.ir_temporal import (  # noqa: E402
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
)
from tempo_dag.ops.builtins import Add, Div, MatMul, Mul, Sub, Tanh  # noqa: E402
from tempo_dag.ops.temporal_builtins import Delay, RollingMean, RollingVar  # noqa: E402


def t(vid: str, shape: list[int], layout: str | None = None) -> Value:
    return Value(value_id=vid, vtype=ValueType.TENSOR, dtype="float32",
                 shape=shape, axes=[f"a{i}" for i in range(len(shape))], layout=layout)


def make_process(pid: str, graph: Graph, buffers: dict | None = None) -> Process:
    buffers = buffers or {}
    p = Process(process_id=pid, kernels={"k": Kernel("k", graph=graph)},
                buffers=buffers,
                edge0=[Edge0(b, "k") for b in buffers],
                edge_delta=[EdgeDelta("k", b, lag_cycles=1) for b in buffers])
    p.validate()
    return p


def zscore_head() -> Process:
    W = 32
    values = {"x": t("x", [1]), "mu": t("mu", [1]), "var": t("var", [1]),
              "xc": t("xc", [1]), "z": t("z", [1]),
              "scale": t("scale", [1], layout="parameter"),
              "bias": t("bias", [1], layout="parameter"),
              "zs": t("zs", [1]), "pre": t("pre", [1]), "y": t("y", [1])}
    ops = {"rmean": RollingMean("rmean", inputs=["x"], outputs=["mu"],
                                attrs={"window_size": W, "buffer_id": "win_mu"}),
           "rvar": RollingVar("rvar", inputs=["x"], outputs=["var"],
                              attrs={"window_size": W, "buffer_id": "win_var"}),
           "sub": Sub("sub", inputs=["x", "mu"], outputs=["xc"]),
           "div": Div("div", inputs=["xc", "var"], outputs=["z"]),
           "mul": Mul("mul", inputs=["z", "scale"], outputs=["zs"]),
           "add": Add("add", inputs=["zs", "bias"], outputs=["pre"]),
           "act": Tanh("act", inputs=["pre"], outputs=["y"])}
    g = Graph(values=values, ops=ops, graph_inputs=["x", "scale", "bias"],
              graph_outputs=["y"])
    bufs = {b: BufferSpec(b, "float32", (1,), depth=W) for b in ("win_mu", "win_var")}
    return make_process("zscore_head", g, bufs)


def multi_horizon() -> Process:
    values = {"x": t("x", [1])}
    ops, outs = {}, []
    for W in (8, 32, 128):
        values[f"m{W}"] = t(f"m{W}", [1])
        ops[f"rm{W}"] = RollingMean(f"rm{W}", inputs=["x"], outputs=[f"m{W}"],
                                    attrs={"window_size": W, "buffer_id": f"win{W}"})
        outs.append(f"m{W}")
    g = Graph(values=values, ops=ops, graph_inputs=["x"], graph_outputs=outs)
    bufs = {f"win{W}": BufferSpec(f"win{W}", "float32", (1,), depth=W)
            for W in (8, 32, 128)}
    return make_process("multi_horizon", g, bufs)


def delay_chain() -> Process:
    values = {"x": t("x", [1]), "d1": t("d1", [1]), "d2": t("d2", [1]),
              "d3": t("d3", [1])}
    ops = {"del1": Delay("del1", inputs=["x"], outputs=["d1"],
                         attrs={"lag_cycles": 1, "buffer_id": "b1"}),
           "del2": Delay("del2", inputs=["d1"], outputs=["d2"],
                         attrs={"lag_cycles": 2, "buffer_id": "b2"}),
           "del3": Delay("del3", inputs=["d2"], outputs=["d3"],
                         attrs={"lag_cycles": 4, "buffer_id": "b3"})}
    g = Graph(values=values, ops=ops, graph_inputs=["x"], graph_outputs=["d3"])
    bufs = {f"b{i}": BufferSpec(f"b{i}", "float32", (1,), depth=d)
            for i, d in ((1, 1), (2, 2), (3, 4))}
    return make_process("delay_chain", g, bufs)


def matmul_head() -> Process:
    H = 8
    values = {"x": t("x", [1, H]), "Wm": t("Wm", [H, H], layout="parameter"),
              "xw": t("xw", [1, H]), "b": t("b", [1, H], layout="parameter"),
              "pre": t("pre", [1, H]), "y": t("y", [1, H])}
    ops = {"mm": MatMul("mm", inputs=["x", "Wm"], outputs=["xw"]),
           "add": Add("add", inputs=["xw", "b"], outputs=["pre"]),
           "act": Tanh("act", inputs=["pre"], outputs=["y"])}
    g = Graph(values=values, ops=ops, graph_inputs=["x", "Wm", "b"],
              graph_outputs=["y"])
    return make_process("matmul_head", g)


def dup_features() -> Process:
    """Duplicated rolling stats — the shape mechanical ONNX lowering emits."""
    values = {"x": t("x", [1]), "ma": t("ma", [1]), "mb": t("mb", [1]),
              "y": t("y", [1])}
    ops = {"rma": RollingMean("rma", inputs=["x"], outputs=["ma"],
                              attrs={"window_size": 32, "buffer_id": "wa"}),
           "rmb": RollingMean("rmb", inputs=["x"], outputs=["mb"],
                              attrs={"window_size": 32, "buffer_id": "wb"}),
           "add": Add("add", inputs=["ma", "mb"], outputs=["y"])}
    g = Graph(values=values, ops=ops, graph_inputs=["x"], graph_outputs=["y"])
    bufs = {b: BufferSpec(b, "float32", (1,), depth=32) for b in ("wa", "wb")}
    return make_process("dup_features", g, bufs)


ZOO = (zscore_head, multi_horizon, delay_chain, matmul_head)

__all__ = ["ZOO", "delay_chain", "dup_features", "make_process", "matmul_head",
           "multi_horizon", "t", "zscore_head"]
