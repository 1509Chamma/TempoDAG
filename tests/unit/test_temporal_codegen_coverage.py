"""Coverage for the fixed-point burst-loop emitter's full op palette
(custom ops, Sub/Mul/Div, Sigmoid/ReLU, non-power-of-two matvec padding)
and the temporal demo's __main__ entry point."""

import json
import runpy

import numpy as np
import pytest

from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (
    CustomFixedPointOp,
    FixedPointConfig,
    _FixedPointUnsupported,
    render_fixedpoint_burst_artifact,
)
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
from tempo_dag.ops.builtins import Add, Div, MatMul, Mul, ReLU, Sigmoid, Sub, Tanh
from tempo_dag.verification.golden_trace import GoldenTraceRecorder
from tempo_dag.verification.temporal_parity import (
    TemporalExecutionTrace,
    TemporalTraceStep,
)

F = 3  # non-power-of-two reduction depth exercises the adder-tree padding
H = 2


class Square(Tanh):
    """Unary op wired through FixedPointConfig.custom_ops in the tests."""

    OP_TYPE = "Square"


def _val(vid, shape, layout=None):
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"a{i}" for i in range(len(shape))],
        layout=layout,
    )


def _full_palette_process():
    values = {
        "x": _val("x", [1, F]),
        "h_prev": _val("h_prev", [1, H]),
        "h_next": _val("h_next", [1, H]),
        "xg": _val("xg", [1, H]),
        "hg": _val("hg", [1, H]),
        "s_add": _val("s_add", [1, H]),
        "s_sub": _val("s_sub", [1, H]),
        "s_mul": _val("s_mul", [1, H]),
        "s_div": _val("s_div", [1, H]),
        "s_sig": _val("s_sig", [1, H]),
        "s_rel": _val("s_rel", [1, H]),
        "s_sq": _val("s_sq", [1, H]),
        "y0": _val("y0", [1, 1]),
        "y": _val("y", [1, 1]),
        "w2": _val("w2", [H, H]),
        "WxT": _val("WxT", [F, H], "parameter"),
        "WhT": _val("WhT", [H, H], "parameter"),
        "b": _val("b", [1, H], "parameter"),
        "c": _val("c", [1, H], "parameter"),
        "d": _val("d", [1, H], "parameter"),
        "Wo": _val("Wo", [H, 1], "parameter"),
        "bo": _val("bo", [1, 1], "parameter"),
        "Wa": _val("Wa", [H, H], "parameter"),
        "Wb": _val("Wb", [H, H], "parameter"),
    }
    ops = {
        "mmx": MatMul("mmx", inputs=["x", "WxT"], outputs=["xg"]),
        "mmh": MatMul("mmh", inputs=["h_prev", "WhT"], outputs=["hg"]),
        "add": Add("add", inputs=["xg", "hg"], outputs=["s_add"]),
        "sub": Sub("sub", inputs=["s_add", "b"], outputs=["s_sub"]),
        "mul": Mul("mul", inputs=["s_sub", "c"], outputs=["s_mul"]),
        "dvd": Div("dvd", inputs=["s_mul", "d"], outputs=["s_div"]),
        "sig": Sigmoid("sig", inputs=["s_div"], outputs=["s_sig"]),
        "rel": ReLU("rel", inputs=["s_sig"], outputs=["s_rel"]),
        "sqr": Square("sqr", inputs=["s_rel"], outputs=["s_sq"]),
        "act": Tanh("act", inputs=["s_sq"], outputs=["h_next"]),
        "mm_out": MatMul("mm_out", inputs=["h_next", "Wo"], outputs=["y0"]),
        "add_out": Add("add_out", inputs=["y0", "bo"], outputs=["y"]),
        # 2-D intermediate: its output is skipped by the declaration pass
        "wadd": Add("wadd", inputs=["Wa", "Wb"], outputs=["w2"]),
    }
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=[
            "x",
            "h_prev",
            "WxT",
            "WhT",
            "b",
            "c",
            "d",
            "Wo",
            "bo",
            "Wa",
            "Wb",
        ],
        graph_outputs=["y", "h_next"],
    )
    process = Process(
        process_id="fxcov",
        kernels={"k": Kernel("k", graph=graph)},
        states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        edge0=[Edge0("h", "k", value_id="h_prev")],
        edge_delta=[EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    params = {
        "WxT": np.full((F, H), 0.1, np.float32),
        "WhT": np.full((H, H), 0.1, np.float32),
        "b": np.full((1, H), 0.05, np.float32),
        "c": np.full((1, H), 0.5, np.float32),
        "d": np.ones((1, H), np.float32),
        "Wo": np.full((H, 1), 0.2, np.float32),
        "bo": np.zeros((1, 1), np.float32),
        "Wa": np.full((H, H), 0.25, np.float32),
        "Wb": np.full((H, H), 0.125, np.float32),
    }
    return process, params


def _trace(num_steps):
    steps = [
        TemporalTraceStep(
            timestep=i,
            inputs={"x": np.array([0.1 * i, 0.2, -0.1], np.float64)},
            outputs={"y": np.array([0.0], np.float64)},
            state={"h": np.zeros(H)},
        )
        for i in range(num_steps)
    ]
    return GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)),
        metadata={"case": "fxcov", "num_steps": num_steps},
    )


def _config(burst=2):
    return FixedPointConfig(
        burst=burst,
        lut_n=64,
        custom_ops={
            "Square": CustomFixedPointOp(
                c_name="square_fx",
                c_body=(
                    "static fx square_fx(acc_t x) {\n"
                    "#pragma HLS INLINE\n"
                    "  return (fx)(x * x);\n"
                    "}"
                ),
                semantics=lambda a: np.square(np.asarray(a, np.float64)),
            )
        },
    )


def test_fixedpoint_requires_exactly_one_kernel():
    with pytest.raises(_FixedPointUnsupported, match="exactly one kernel"):
        render_fixedpoint_burst_artifact(
            Process(process_id="empty"), _trace(1), {}, FixedPointConfig(burst=1)
        )


def test_fixedpoint_full_palette_emission_and_oracle():
    process, params = _full_palette_process()
    art = render_fixedpoint_burst_artifact(process, _trace(2), params, _config())
    dut = art.dut_hls

    assert art.top_name == "fxcov_stream"
    # custom op body is emitted into the prelude and invoked per lane
    assert "static fx square_fx(acc_t x)" in dut
    assert "square_fx((acc_t)" in dut
    # LUT-backed sigmoid and fixed-point ReLU calls
    assert "sig_lut((acc_t)" in dut
    assert "relu_fx((acc_t)" in dut
    # Sub/Mul/Div elementwise lanes
    assert "- (acc_t)b[i]" in dut
    assert "* (acc_t)c[i]" in dut
    assert "/ (acc_t)d[i]" in dut
    # K=3 matvec pads partial products up to the power-of-two tree width
    assert "for (int k = 3; k < 4; ++k) {" in dut
    # 2-D intermediate value is not declared as a flat lane array
    assert "fx w2[" not in dut

    # the oracle ran every op type and produced one golden value per step
    tb = art.testbench_hls
    assert "static const double golden[2]" in tb
    assert "fxcov_stream" in tb


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_temporal_demo_main_entrypoint(capsys):
    runpy.run_module("tempo_dag.examples.temporal_demo", run_name="__main__")
    payload = json.loads(capsys.readouterr().out)
    assert payload["process_id"] == "temporal_demo"
    assert payload["validation_passed"] is True
