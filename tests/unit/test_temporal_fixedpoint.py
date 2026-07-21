"""Contract tests for the fixed-point II-bound burst-loop emitter.

Locks the structural shape that RTL-cosim-verified in the prototype so a
regression in the emitter is caught without a Vitis run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (
    FixedPointConfig,
    _FixedPointUnsupported,
    render_fixedpoint_burst_artifact,
    write_fixedpoint_burst_bundle,
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
from tempo_dag.ops.builtins import Add, MatMul, Sqrt, Tanh
from tempo_dag.verification.golden_trace import GoldenTraceRecorder
from tempo_dag.verification.temporal_parity import (
    TemporalExecutionTrace,
    TemporalTraceStep,
)

H = F = 2


def _val(vid, shape, layout=None):
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"a{i}" for i in range(len(shape))],
        layout=layout,
    )


def _rnn_process():
    values = {
        "x": _val("x", [1, F]),
        "h_prev": _val("h_prev", [1, H]),
        "h_next": _val("h_next", [1, H]),
        "xg": _val("xg", [1, H]),
        "hg": _val("hg", [1, H]),
        "sg": _val("sg", [1, H]),
        "pre": _val("pre", [1, H]),
        "y0": _val("y0", [1, 1]),
        "y": _val("y", [1, 1]),
        "WxT": _val("WxT", [F, H], "parameter"),
        "WhT": _val("WhT", [H, H], "parameter"),
        "b": _val("b", [1, H], "parameter"),
        "Wo2": _val("Wo2", [H, 1], "parameter"),
        "bo2": _val("bo2", [1, 1], "parameter"),
    }
    ops = {
        "mmx": MatMul("mmx", inputs=["x", "WxT"], outputs=["xg"]),
        "mmh": MatMul("mmh", inputs=["h_prev", "WhT"], outputs=["hg"]),
        "gs": Add("gs", inputs=["xg", "hg"], outputs=["sg"]),
        "gb": Add("gb", inputs=["sg", "b"], outputs=["pre"]),
        "act": Tanh("act", inputs=["pre"], outputs=["h_next"]),
        "mm_out": MatMul("mm_out", inputs=["h_next", "Wo2"], outputs=["y0"]),
        "add_out": Add("add_out", inputs=["y0", "bo2"], outputs=["y"]),
    }
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=["x", "h_prev", "WxT", "WhT", "b", "Wo2", "bo2"],
        graph_outputs=["y", "h_next"],
    )
    process = Process(
        process_id="rnn2",
        kernels={"k": Kernel("k", graph=graph)},
        states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        edge0=[Edge0("h", "k", value_id="h_prev")],
        edge_delta=[EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    process.validate()
    params = {
        "WxT": np.full((F, H), 0.1, np.float32),
        "WhT": np.full((H, H), 0.1, np.float32),
        "b": np.zeros((1, H), np.float32),
        "Wo2": np.full((H, 1), 0.2, np.float32),
        "bo2": np.zeros((1, 1), np.float32),
    }
    return process, params


def _trace(n):
    steps = [
        TemporalTraceStep(
            timestep=i,
            inputs={"x": np.array([0.1 * i, 0.2], np.float64)},
            outputs={"y": np.array([0.01 * i], np.float64)},
            state={"h": np.zeros(H)},
        )
        for i in range(n)
    ]
    return GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)), metadata={"case": "rnn2", "num_steps": n}
    )


def test_fixedpoint_emits_burst_loop_contract():
    process, params = _rnn_process()
    cfg = FixedPointConfig(burst=4, lut_n=64)
    art = render_fixedpoint_burst_artifact(process, _trace(4), params, cfg)
    dut = art.dut_hls

    assert art.top_name == "rnn2_stream"
    # datapath types + LUT activations (no float transcendentals)
    assert "typedef ap_fixed<18, 6> fx;" in dut
    assert "static const fx TANH_LUT[64]" in dut
    assert "ap_uint<" in dut and "TANH_LUT[idx]" in dut
    assert "std::tanh" not in dut and "std::exp" not in dut
    # pipelined burst loop at the configured II
    assert "sample_loop: for (int t = 0; t < 4; ++t)" in dut
    assert "#pragma HLS PIPELINE II=12" in dut
    # baked weights, not ports
    assert "static const fx WxT[2][2]" in dut
    assert "y_out[" in dut  # output port renamed to avoid the value 'y'
    # state as a static ROM read/written each iteration, partitioned
    assert "static fx h__state[2]" in dut
    assert "#pragma HLS ARRAY_PARTITION variable=h__state complete" in dut
    # matvec: partial products + balanced adder tree
    assert "mmx_mv:" in dut and "_p[k] = _p[k] + _p[k + s];" in dut


def test_fixedpoint_testbench_asserts_golden():
    process, params = _rnn_process()
    art = render_fixedpoint_burst_artifact(
        process, _trace(4), params, FixedPointConfig(burst=4, lut_n=64)
    )
    tb = art.testbench_hls
    assert "extern void rnn2_stream(const fx x[4][2], fx y[4]);" in tb
    assert "static const double golden[4]" in tb
    assert "++errors" in tb and "return errors == 0 ? 0 : 1;" in tb


def test_fixedpoint_rejects_unbaked_params():
    process, params = _rnn_process()
    params.pop("WhT")
    with pytest.raises(_FixedPointUnsupported):
        render_fixedpoint_burst_artifact(
            process, _trace(4), params, FixedPointConfig(burst=4)
        )


def test_fixedpoint_rejects_unsupported_op():
    values = {
        "x": _val("x", [1, F]),
        "h_prev": _val("h_prev", [1, H]),
        "h_next": _val("h_next", [1, H]),
        "xg": _val("xg", [1, H]),
        "y0": _val("y0", [1, 1]),
        "y": _val("y", [1, 1]),
        "WxT": _val("WxT", [F, H], "parameter"),
        "Wo2": _val("Wo2", [H, 1], "parameter"),
    }
    ops = {
        "mmx": MatMul("mmx", inputs=["x", "WxT"], outputs=["xg"]),
        # Sqrt is a real op that the fixed-point emitter does not support
        "bad": Sqrt("bad", inputs=["xg"], outputs=["h_next"]),
        "mm_out": MatMul("mm_out", inputs=["h_next", "Wo2"], outputs=["y0"]),
    }
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=["x", "h_prev", "WxT", "Wo2"],
        graph_outputs=["y0", "h_next"],
    )
    process = Process(
        process_id="bad1",
        kernels={"k": Kernel("k", graph=graph)},
        states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
        edge0=[Edge0("h", "k", value_id="h_prev")],
        edge_delta=[EdgeDelta("k", "h", lag_cycles=1, value_id="h_next")],
    )
    with pytest.raises(_FixedPointUnsupported):
        render_fixedpoint_burst_artifact(
            process,
            _trace(2),
            {"WxT": np.zeros((F, H), np.float32), "Wo2": np.zeros((H, 1), np.float32)},
            FixedPointConfig(burst=2),
        )


def test_write_fixedpoint_burst_bundle_writes_files(tmp_path):
    process, params = _rnn_process()
    info = write_fixedpoint_burst_bundle(
        process,
        _trace(4),
        tmp_path,
        params,
        stem="bench_rnn",
        config=FixedPointConfig(burst=4, lut_n=64),
    )
    assert info["top"] == "rnn2_stream"
    dut, tb = tmp_path / "bench_rnn.cpp", tmp_path / "bench_rnn_tb.cpp"
    assert dut.exists() and tb.exists()
    assert "rnn2_stream" in dut.read_text()
    assert "return errors == 0 ? 0 : 1;" in tb.read_text()
