"""Coverage tests for uncovered emission paths in the temporal HLS generator.

Targets: multi-kernel rejection, flat-index helpers, cyclic kernel graphs,
state read/write emission (scalar and vector), state-binding error branches,
fused elementwise runs, FusedMatMulAdd emission, the MAC-count pipeline
guardrail, and both testbench variants (legacy scaffold and wired
array-stream ports).
"""

from __future__ import annotations

import numpy as np
import pytest

from tempo_dag.codegen.hls.temporal_generator import (
    TemporalArtifactKind,
    _flat_index_expr,
    _state_bindings,
    _step_wiring,
    _topological_ops,
    _WiringUnsupported,
    load_and_render_temporal_artifact,
    render_temporal_process_hls,
    render_temporal_testbench,
    write_temporal_hls_artifact_bundle,
)
from tempo_dag.examples import rolling_window_process
from tempo_dag.ir.graph import Graph
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    FusedMatMulAdd,
    Kernel,
    Process,
    StateKind,
    StateSpec,
)
from tempo_dag.ops.builtins import Add, MatMul, Sigmoid, Softmax, Tanh
from tempo_dag.ops.temporal_builtins import RollingMean
from tempo_dag.verification.golden_trace import TRACE_SCHEMA_VERSION, GoldenTrace
from tempo_dag.verification.temporal_parity import TemporalTraceStep


def _tensor(
    value_id: str,
    shape: list[int],
    axes: list[str] | None = None,
    layout: str | None = None,
) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=list(shape),
        axes=list(axes) if axes else [f"axis_{idx}" for idx in range(len(shape))],
        layout=layout,
    )


def _empty_graph() -> Graph:
    return Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])


def _empty_kernel(kernel_id: str) -> Kernel:
    return Kernel(kernel_id=kernel_id, graph=_empty_graph())


# ---------------------------------------------------------------------------
# Top-level renderer guards
# ---------------------------------------------------------------------------


def test_buffer_declarations_half_header_and_wiring_fallback() -> None:
    process = Process(
        process_id="buf_proc",
        kernels={"bk": _empty_kernel("bk")},
        buffers={
            "line": BufferSpec("line", "float32", (1,), depth=4),
            "win": BufferSpec("win", "float16", (4, 2), depth=8, axes=("a", "b")),
        },
        edge0=[Edge0("line", "bk", value_id="x_prev")],
        edge_delta=[EdgeDelta("bk", "line", lag_cycles=3, value_id="x")],
    )

    rendered = render_temporal_process_hls(process)

    # Single-element buffers flatten to [depth]; larger ones keep their shape.
    assert "static float line[4];" in rendered
    assert "static half win[8][4][2];" in rendered
    assert "#include <hls_half.h>" in rendered
    assert "// edge_delta: bk -> line lag=3" in rendered
    # The empty kernel graph cannot be wired: fall back to the scaffold.
    assert "#pragma HLS DATAFLOW" in rendered
    assert "// wiring unavailable:" in rendered


def test_render_process_rejects_multiple_kernels() -> None:
    process = Process(
        process_id="two_kernel_proc",
        kernels={
            "k1": _empty_kernel("k1"),
            "k2": _empty_kernel("k2"),
        },
    )

    with pytest.raises(ValueError, match="exactly one kernel"):
        render_temporal_process_hls(process)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_flat_index_expr_rank1_rank2_and_reject() -> None:
    assert _flat_index_expr("v", [4], "i") == "v[i]"
    assert _flat_index_expr("v", [1, 4], "i") == "v[0][i]"
    with pytest.raises(_WiringUnsupported, match="unsupported for element access"):
        _flat_index_expr("v", [2, 3], "i")


def test_topological_ops_rejects_operator_cycle() -> None:
    graph = Graph(
        values={
            "v1": _tensor("v1", [4]),
            "v2": _tensor("v2", [4]),
        },
        ops={
            "cyc_a": Tanh(op_id="cyc_a", inputs=["v2"], outputs=["v1"]),
            "cyc_b": Tanh(op_id="cyc_b", inputs=["v1"], outputs=["v2"]),
        },
        graph_inputs=[],
        graph_outputs=[],
    )

    with pytest.raises(_WiringUnsupported, match="operator cycle"):
        _topological_ops(graph)


# ---------------------------------------------------------------------------
# State bindings: error branches
# ---------------------------------------------------------------------------


def _state_process(
    state: StateSpec,
    edge0: list[Edge0],
    edge_delta: list[EdgeDelta],
) -> Process:
    kernel = _empty_kernel("cell")
    return Process(
        process_id="state_err_proc",
        kernels={kernel.kernel_id: kernel},
        states={state.state_id: state},
        edge0=edge0,
        edge_delta=edge_delta,
    )


def test_state_binding_requires_read_and_write() -> None:
    state = StateSpec("s", StateKind.HIDDEN, "float32", (1,))
    process = _state_process(
        state,
        edge0=[Edge0("s", "cell", value_id="s_prev")],
        edge_delta=[],
    )

    kernel = process.kernels["cell"]
    with pytest.raises(_WiringUnsupported, match="needs both a read Edge0"):
        _state_bindings(process, kernel)


def test_state_binding_rejects_initializer_length_mismatch() -> None:
    state = StateSpec(
        "s",
        StateKind.HIDDEN,
        "float32",
        (3,),
        metadata={"initializer": [1.0, 2.0]},
    )
    process = _state_process(
        state,
        edge0=[Edge0("s", "cell", value_id="s_prev")],
        edge_delta=[EdgeDelta("cell", "s", lag_cycles=1, value_id="s_next")],
    )

    kernel = process.kernels["cell"]
    with pytest.raises(_WiringUnsupported, match="initializer length 2"):
        _state_bindings(process, kernel)


def test_state_binding_rejects_nonzero_scalar_init_for_vector_state() -> None:
    state = StateSpec(
        "s",
        StateKind.HIDDEN,
        "float32",
        (3,),
        metadata={"initializer": 0.5},
    )
    process = _state_process(
        state,
        edge0=[Edge0("s", "cell", value_id="s_prev")],
        edge_delta=[EdgeDelta("cell", "s", lag_cycles=1, value_id="s_next")],
    )

    kernel = process.kernels["cell"]
    with pytest.raises(_WiringUnsupported, match="non-zero scalar initializer"):
        _state_bindings(process, kernel)


# ---------------------------------------------------------------------------
# Step wiring: error branches
# ---------------------------------------------------------------------------


def test_step_wiring_rejects_unsupported_op_type() -> None:
    graph = Graph(
        values={
            "x": _tensor("x", [2, 3]),
            "y": _tensor("y", [2, 3]),
        },
        ops={
            "soft0": Softmax(
                op_id="soft0",
                inputs=["x"],
                outputs=["y"],
                attrs={"axis": 1},
            ),
        },
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    kernel = Kernel(kernel_id="cell", graph=graph)
    process = Process(process_id="soft_proc", kernels={"cell": kernel})

    with pytest.raises(_WiringUnsupported, match="unsupported op type 'Softmax'"):
        _step_wiring(process, kernel)


def test_step_wiring_rejects_scalar_where_array_expected() -> None:
    # Degenerate graph: 'x' is both a graph input consumed only by
    # RollingMean (so it is classified scalar) and the output of an
    # elementwise op, which requires a 1-D array expression.
    graph = Graph(
        values={
            "p": _tensor("p", [1], layout="parameter"),
            "x": _tensor("x", [1]),
            "rm_out": _tensor("rm_out", [1]),
        },
        ops={
            "t0": Tanh(op_id="t0", inputs=["p"], outputs=["x"]),
            "rm": RollingMean(
                op_id="rm",
                inputs=["x"],
                outputs=["rm_out"],
                attrs={"window_size": 4},
            ),
        },
        graph_inputs=["x", "p"],
        graph_outputs=["rm_out"],
    )
    kernel = Kernel(kernel_id="cell", graph=graph)
    process = Process(process_id="scalar_clash_proc", kernels={"cell": kernel})

    with pytest.raises(_WiringUnsupported, match="scalar where 1-D array expected"):
        _step_wiring(process, kernel)


def test_step_interface_promotes_referenced_initializers_to_params() -> None:
    # 'w' is referenced but neither produced nor a graph input: it must be
    # treated as a parameter port (the ONNX-initializer case).
    graph = Graph(
        values={
            "x": _tensor("x", [4]),
            "w": _tensor("w", [4]),
            "y": _tensor("y", [4]),
        },
        ops={
            "addp": Add(op_id="addp", inputs=["x", "w"], outputs=["y"]),
        },
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    kernel = Kernel(kernel_id="cell", graph=graph)
    process = Process(process_id="init_param_proc", kernels={"cell": kernel})

    ports, _, prelude = _step_wiring(process, kernel)

    assert "const float w[4]" in ports
    assert prelude == []


# ---------------------------------------------------------------------------
# State read/write emission
# ---------------------------------------------------------------------------


def _scalar_state_process() -> Process:
    graph = Graph(
        values={
            "x": _tensor("x", [1]),
            "h_prev": _tensor("h_prev", [1]),
            "h_sum": _tensor("h_sum", [1]),
            "y": _tensor("y", [1]),
        },
        ops={
            "acc": Add(op_id="acc", inputs=["x", "h_prev"], outputs=["h_sum"]),
            "act": Tanh(op_id="act", inputs=["h_sum"], outputs=["y"]),
        },
        graph_inputs=["x", "h_prev"],
        graph_outputs=["h_sum", "y"],
    )
    kernel = Kernel(kernel_id="cell", graph=graph)
    return Process(
        process_id="scal_state_proc",
        kernels={"cell": kernel},
        states={
            "h": StateSpec(
                "h",
                StateKind.HIDDEN,
                "float32",
                (1,),
                metadata={"initializer": [0.5]},
            ),
            # No edges reference this state: bindings must skip it.
            "unused": StateSpec("unused", StateKind.HIDDEN, "float32", (2,)),
        },
        edge0=[Edge0("h", "cell", value_id="h_prev")],
        edge_delta=[EdgeDelta("cell", "h", lag_cycles=1, value_id="h_sum")],
    )


def test_scalar_state_read_write_and_fused_run_emission() -> None:
    rendered = render_temporal_process_hls(_scalar_state_process())

    # Scalar state: static local, read into value, written back at the end.
    assert "static float h__state = 0.5f;" in rendered
    assert "float h_prev[1];" in rendered
    assert "h_prev[0] = h__state;" in rendered
    assert "h__state = h_sum[0];" in rendered
    # The unbound state contributes nothing.
    assert "unused__state" not in rendered
    # Both elementwise ops fuse into one pipelined loop.
    assert "fused_ew_0_loop:" in rendered
    assert "for (int i = 0; i < 1; ++i) {" in rendered
    assert "const float t_h_sum = (x[i] + h_prev[i]);" in rendered
    assert "const float t_y = std::tanh(t_h_sum);" in rendered
    assert "h_sum[i] = t_h_sum;" in rendered
    assert "y[i] = t_y;" in rendered
    assert "void scal_state_proc_step(const float x[1], float y[1]) {" in rendered


def _vector_state_process() -> Process:
    axes = ["axis_0"]
    graph = Graph(
        values={
            "x": _tensor("x", [8], axes),
            "h1_prev": _tensor("h1_prev", [8], axes),
            "h2_prev": _tensor("h2_prev", [8], axes),
            "h1_next": _tensor("h1_next", [8], axes),
            "h2_next": _tensor("h2_next", [8], axes),
            "y_out": _tensor("y_out", [8], axes),
        },
        ops={
            "op_a": Add(op_id="op_a", inputs=["x", "h1_prev"], outputs=["h1_next"]),
            "op_b": Add(
                op_id="op_b",
                inputs=["h1_next", "h2_prev"],
                outputs=["h2_next"],
            ),
            "op_c": Tanh(op_id="op_c", inputs=["h2_next"], outputs=["y_out"]),
        },
        graph_inputs=["x", "h1_prev", "h2_prev"],
        graph_outputs=["h1_next", "h2_next", "y_out"],
    )
    kernel = Kernel(kernel_id="vcell", graph=graph)
    return Process(
        process_id="vec_state_proc",
        kernels={"vcell": kernel},
        states={
            "h1": StateSpec("h1", StateKind.HIDDEN, "float32", (8,)),
            "h2": StateSpec(
                "h2",
                StateKind.HIDDEN,
                "float32",
                (8,),
                metadata={"initializer": [0.125] * 8},
            ),
        },
        edge0=[
            Edge0("h1", "vcell", value_id="h1_prev"),
            Edge0("h2", "vcell", value_id="h2_prev"),
        ],
        edge_delta=[
            EdgeDelta("vcell", "h1", lag_cycles=1, value_id="h1_next"),
            EdgeDelta("vcell", "h2", lag_cycles=1, value_id="h2_next"),
        ],
    )


def test_vector_state_read_write_loops_and_partition_pragmas() -> None:
    rendered = render_temporal_process_hls(_vector_state_process())

    # Zero-initialized vector state.
    assert "static float h1__state[8] = {0};" in rendered
    assert "#pragma HLS ARRAY_PARTITION variable=h1__state complete" in rendered
    # Non-zero vector initializer expands element by element.
    assert "static float h2__state[8] = {0.125f, 0.125f," in rendered
    # Read loops copy state into kernel values, with partition pragmas on
    # the 8-element read arrays.
    assert "h1_read_loop:" in rendered
    assert "h1_prev[i] = h1__state[i];" in rendered
    assert "#pragma HLS ARRAY_PARTITION variable=h1_prev complete dim=0" in rendered
    assert "h2_read_loop:" in rendered
    # Write-back loops.
    assert "h1_write_loop:" in rendered
    assert "h1__state[i] = h1_next[i];" in rendered
    assert "h2_write_loop:" in rendered
    assert "h2__state[i] = h2_next[i];" in rendered


# ---------------------------------------------------------------------------
# Fused elementwise runs
# ---------------------------------------------------------------------------


def _mixed_size_process() -> Process:
    graph = Graph(
        values={
            "x4": _tensor("x4", [4], ["axis_0"]),
            "t4": _tensor("t4", [4], ["axis_0"]),
            "u4": _tensor("u4", [4], ["axis_0"]),
            "c2": _tensor("c2", [2], ["axis_0"], layout="parameter"),
            "d2": _tensor("d2", [2], ["axis_0"]),
        },
        ops={
            "e1": Tanh(op_id="e1", inputs=["x4"], outputs=["t4"]),
            "e2": Sigmoid(op_id="e2", inputs=["t4"], outputs=["u4"]),
            "e3": Tanh(op_id="e3", inputs=["c2"], outputs=["d2"]),
        },
        graph_inputs=["x4", "c2"],
        graph_outputs=["u4", "d2"],
    )
    kernel = Kernel(kernel_id="mix_cell", graph=graph)
    return Process(process_id="mix_proc", kernels={"mix_cell": kernel})


def test_fused_run_flushes_when_element_count_changes() -> None:
    rendered = render_temporal_process_hls(_mixed_size_process())

    # e1 + e2 fuse into one 4-element loop.
    assert "fused_ew_0_loop:" in rendered
    assert "for (int i = 0; i < 4; ++i) {" in rendered
    assert "const float t_t4 = std::tanh(x4[i]);" in rendered
    assert "const float t_u4 = (1.0f / (1.0f + std::exp(-(t_t4))));" in rendered
    assert "u4[i] = t_u4;" in rendered
    # t4 is only consumed inside the run: no array store is emitted.
    assert "t4[i] = t_t4;" not in rendered
    # e3 has a different element count, so it is emitted standalone.
    assert "e3_kernel(c2, d2);" in rendered
    # The un-baked parameter stays a top-level port.
    assert "const float c2[2]" in rendered


# ---------------------------------------------------------------------------
# MatMul-family emission
# ---------------------------------------------------------------------------


def _fused_matmul_add_process() -> Process:
    graph = Graph(
        values={
            "lhs": _tensor("lhs", [1, 3], ["rows", "inner"]),
            "rhs": _tensor("rhs", [3, 4], ["inner", "cols"], layout="parameter"),
            "bias": _tensor("bias", [1, 4], ["rows", "cols"], layout="parameter"),
            "out": _tensor("out", [1, 4], ["rows", "cols"]),
        },
        ops={
            "fma0": FusedMatMulAdd(
                op_id="fma0",
                inputs=["lhs", "rhs", "bias"],
                outputs=["out"],
            ),
        },
        graph_inputs=["lhs", "rhs", "bias"],
        graph_outputs=["out"],
    )
    kernel = Kernel(kernel_id="fma_cell", graph=graph)
    return Process(process_id="fma_proc", kernels={"fma_cell": kernel})


def test_fused_matmul_add_call_emission() -> None:
    rendered = render_temporal_process_hls(_fused_matmul_add_process())

    # Bias is rank-2 [1, N], so it is passed as its first row.
    assert "fma0_kernel(lhs, rhs, bias[0], out);" in rendered
    # 12 MACs stays below the guardrail threshold.
    assert "guardrail" not in rendered


def _large_matmul_process() -> Process:
    graph = Graph(
        values={
            "x": _tensor("x", [16, 16], ["rows", "inner"]),
            "w": _tensor("w", [16, 16], ["inner", "cols"], layout="parameter"),
            "y": _tensor("y", [16, 16], ["rows", "cols"]),
        },
        ops={
            "mm0": MatMul(op_id="mm0", inputs=["x", "w"], outputs=["y"]),
        },
        graph_inputs=["x", "w"],
        graph_outputs=["y"],
    )
    kernel = Kernel(kernel_id="mm_cell", graph=graph)
    return Process(process_id="mm_proc", kernels={"mm_cell": kernel})


def test_matmul_guardrail_suppresses_top_level_pipeline() -> None:
    rendered = render_temporal_process_hls(_large_matmul_process())

    assert "mm0_kernel(x, w, y);" in rendered
    assert (
        "// guardrail: top-level PIPELINE suppressed (4096 MACs); "
        "per-kernel pipelining applies"
    ) in rendered


# ---------------------------------------------------------------------------
# RollingMean scalar-stream emission
# ---------------------------------------------------------------------------


def _rolling_mean_process() -> Process:
    graph = Graph(
        values={
            "x": _tensor("x", [1]),
            "rm_out": _tensor("rm_out", [1]),
        },
        ops={
            "rm0": RollingMean(
                op_id="rm0",
                inputs=["x"],
                outputs=["rm_out"],
                attrs={"window_size": 4, "buffer_id": "rm_buf"},
            ),
        },
        graph_inputs=["x"],
        graph_outputs=["rm_out"],
    )
    kernel = Kernel(kernel_id="rm_cell", graph=graph)
    return Process(
        process_id="rm_proc",
        kernels={"rm_cell": kernel},
        buffers={"rm_buf": BufferSpec("rm_buf", "float32", (1,), depth=4)},
    )


def test_rolling_mean_scalar_stream_emission() -> None:
    rendered = render_temporal_process_hls(_rolling_mean_process())

    # The stream input is consumed only by RollingMean, so it stays scalar.
    assert "void rm_proc_step(const float x, float rm_out[1]) {" in rendered
    assert "rm0_kernel<float, 4>(x, rm_buf, rm_out[0]);" in rendered


# ---------------------------------------------------------------------------
# Testbench rendering
# ---------------------------------------------------------------------------


def _trace(steps: tuple[TemporalTraceStep, ...]) -> GoldenTrace:
    return GoldenTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        metadata={},
        steps=steps,
    )


def test_testbench_legacy_scaffold_for_unwired_process() -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.0])},
        outputs={"y": np.array([2.0])},
        state={"h": np.array([0.5])},
    )

    rendered = render_temporal_testbench(rolling_window_process(), _trace((step,)))

    assert "extern void reference_rolling_window_step();" in rendered
    assert "// timestep 0" in rendered
    assert "// input x = [1.0]" in rendered
    assert "// expected output y = [2.0]" in rendered
    assert "// expected state h = [0.5]" in rendered
    assert "  reference_rolling_window_step();" in rendered
    assert "Temporal testbench complete" in rendered


def _vector_add_process() -> Process:
    graph = Graph(
        values={
            "x": _tensor("x", [4], ["axis_0"]),
            "c": _tensor("c", [4], ["axis_0"], layout="parameter"),
            "y": _tensor("y", [4], ["axis_0"]),
        },
        ops={
            "add1": Add(op_id="add1", inputs=["x", "c"], outputs=["y"]),
        },
        graph_inputs=["x", "c"],
        graph_outputs=["y"],
    )
    kernel = Kernel(kernel_id="vadd_cell", graph=graph)
    return Process(process_id="vec_proc", kernels={"vadd_cell": kernel})


def test_testbench_drives_array_stream_input_with_baked_params() -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.0, 2.0, 3.0, 4.0])},
        outputs={"y": np.array([2.0, 3.0, 4.0, 5.0])},
        state={},
    )
    parameters = {"c": np.array([1.0, 1.0, 1.0, 1.0])}

    rendered = render_temporal_testbench(
        _vector_add_process(),
        _trace((step,)),
        parameters=parameters,
    )

    # Baked parameter 'c' is neither declared nor passed by the testbench.
    assert "extern void vec_proc_step(const float x[4], float y[4]);" in rendered
    assert "\n  float x[4];" in rendered
    # Array stream input is written element by element before each call.
    assert "  x[0] = 1.0f;" in rendered
    assert "  x[3] = 4.0f;" in rendered
    assert "  vec_proc_step(x, y);" in rendered
    # Parameter values were supplied, so outputs are asserted.
    assert "MISMATCH t=0" in rendered
    assert "return errors == 0 ? 0 : 1;" in rendered


def test_testbench_scalar_stream_passes_input_literal() -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.5])},
        outputs={"rm_out": np.array([0.375])},
        state={},
    )

    rendered = render_temporal_testbench(_rolling_mean_process(), _trace((step,)))

    # Scalar stream inputs are passed as literals, not via a staging array.
    assert "rm_proc_step(1.5f, rm_out);" in rendered


def test_testbench_unbaked_params_default_to_zero_without_values() -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.0, 2.0, 3.0, 4.0])},
        outputs={"y": np.array([1.0, 2.0, 3.0, 4.0])},
        state={},
    )

    rendered = render_temporal_testbench(_vector_add_process(), _trace((step,)))

    # No parameter values: 'c' stays a port, declared with zero contents,
    # and outputs are not asserted.
    assert "float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};" in rendered
    assert "// parameter values not supplied: outputs not asserted" in rendered
    assert "vec_proc_step(x, c, y);" in rendered
    assert "MISMATCH" not in rendered


# ---------------------------------------------------------------------------
# Artifact bundle writing and loading
# ---------------------------------------------------------------------------


def test_write_bundle_and_load_render_round_trip(tmp_path) -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.0, 2.0, 3.0, 4.0])},
        outputs={"y": np.array([2.0, 3.0, 4.0, 5.0])},
        state={},
    )
    trace = _trace((step,))
    process = _vector_add_process()

    manifest = write_temporal_hls_artifact_bundle(
        process,
        trace,
        tmp_path,
        stem="vec",
        parameters={"c": np.ones(4)},
    )

    assert manifest.process_id == "vec_proc"
    files = {artifact.kind: artifact.path for artifact in manifest.files}
    assert set(files) == set(TemporalArtifactKind)
    for path in files.values():
        assert path.is_file()

    artifact = load_and_render_temporal_artifact(
        process,
        files[TemporalArtifactKind.GOLDEN_TRACE_JSON],
    )
    assert "void vec_proc_step(" in artifact.process_hls
    assert "vec_proc_step(x, c, y);" in artifact.testbench_hls
