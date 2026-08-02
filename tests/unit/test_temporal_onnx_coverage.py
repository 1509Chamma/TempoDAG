"""Coverage-focused tests for tempo_dag.parsers.temporal_onnx."""

from __future__ import annotations

import onnx
import onnx.helper as helper
import pytest
from onnx import TensorProto

from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import TemporalIRValidationError
from tempo_dag.parsers.onnx.parser import ONNXParser
from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    _coerce_attr_int,
    _conv_attr_as_int,
    _materialize_missing_values,
    build_demo_temporal_onnx_model,
)


def _add_then_relu_graph():
    add = helper.make_node("Add", ["lhs", "rhs"], ["add_out"], name="add_node")
    relu = helper.make_node("Relu", ["add_out"], ["y"], name="relu_node")
    graph = helper.make_graph(
        [add, relu],
        "add_materialize",
        [
            helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    return ONNXParser().parse_model(helper.make_model(graph))


def test_report_to_dict_serializes_all_sections() -> None:
    result = TemporalONNXParser().parse_model(
        build_demo_temporal_onnx_model(), process_id="report_process"
    )

    payload = result.report.to_dict()

    assert payload["process_id"] == "report_process"
    assert payload["lowered_ops"] == list(result.report.lowered_ops)
    assert payload["states"] == list(result.report.states)
    assert payload["buffers"] == ["rolling_mean_node_buffer"]
    assert isinstance(payload["detected_patterns"], list)


def test_extra_op_mapping_merges_with_defaults() -> None:
    parser = TemporalONNXParser(
        extra_op_mapping={"CustomOp": "ReLU", "Loop": "Delay"},
    )

    assert parser.onnx_parser.op_mapping["CustomOp"] == "ReLU"
    # Caller-provided entries override the defaults.
    assert parser.onnx_parser.op_mapping["Loop"] == "Delay"
    # Untouched defaults remain in place.
    assert parser.onnx_parser.op_mapping["Scan"] == "ScanCell"


def test_parse_loads_model_from_path(tmp_path) -> None:
    model_path = tmp_path / "demo.onnx"
    onnx.save(build_demo_temporal_onnx_model(), str(model_path))

    result = TemporalONNXParser().parse(str(model_path), process_id="from_file")

    assert result.process.process_id == "from_file"
    assert "kernel_main" in result.process.kernels


def test_pattern_with_undeclared_state_input_is_skipped_then_rejected() -> None:
    # The LSTM initial state input is referenced by the node but never declared
    # in the model, so the pattern lowering skips buffer creation for it and
    # final process validation rejects the dangling reference.
    lstm = helper.make_node(
        "LSTM",
        ["x", "w", "r", "b", "seq_lens", "initial_h"],
        ["y"],
        name="lstm_node",
        hidden_size=2,
    )
    graph = helper.make_graph(
        [lstm],
        "lstm_missing_state",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 8, 3]),
            helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 8, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 1, 1, 2])],
    )
    model = helper.make_model(graph, producer_name="lstm_missing_state")

    with pytest.raises(TemporalIRValidationError, match="graph is invalid"):
        TemporalONNXParser().parse_model(model)


def test_materialize_conv_output_with_list_padding_attr() -> None:
    conv = helper.make_node(
        "Conv",
        ["stream_in", "conv_weight"],
        ["conv_out"],
        name="conv_node",
        padding=[1, 1],
    )
    relu = helper.make_node("Relu", ["conv_out"], ["y"], name="relu_node")
    graph = helper.make_graph(
        [conv, relu],
        "conv_list_padding",
        [helper.make_tensor_value_info("stream_in", TensorProto.FLOAT, [1, 1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8])],
        initializer=[
            helper.make_tensor(
                "conv_weight",
                TensorProto.FLOAT,
                [1, 1, 3],
                [0.25, 0.5, 0.25],
            )
        ],
    )
    ir_graph = ONNXParser().parse_model(helper.make_model(graph))
    assert "conv_out" not in ir_graph.values

    _materialize_missing_values(ir_graph)

    # stride defaults to 1, padding taken from the list-valued attribute:
    # (8 + 2*1 - (3 - 1) - 1) // 1 + 1 == 8
    assert ir_graph.values["conv_out"].shape == [1, 1, 8]
    assert ir_graph.values["conv_out"].producer_op_id == "conv_node"


def test_materialize_add_output_follows_tensor_lhs() -> None:
    ir_graph = _add_then_relu_graph()
    assert "add_out" not in ir_graph.values

    _materialize_missing_values(ir_graph)

    assert ir_graph.values["add_out"].shape == [2, 3]
    assert ir_graph.values["add_out"].axes == ir_graph.values["lhs"].axes


def test_materialize_add_output_follows_rhs_when_lhs_scalar() -> None:
    ir_graph = _add_then_relu_graph()
    ir_graph.values["lhs"] = Value(
        value_id="lhs",
        vtype=ValueType.SCALAR,
        dtype="float32",
        shape=[],
        axes=[],
    )

    _materialize_missing_values(ir_graph)

    assert ir_graph.values["add_out"].shape == [2, 3]
    assert ir_graph.values["add_out"].axes == ir_graph.values["rhs"].axes


def test_materialize_generic_op_output_follows_first_input() -> None:
    sigmoid = helper.make_node("Sigmoid", ["x"], ["sig_out"], name="sig_node")
    relu = helper.make_node("Relu", ["sig_out"], ["y"], name="relu_node")
    graph = helper.make_graph(
        [sigmoid, relu],
        "generic_materialize",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
    )
    ir_graph = ONNXParser().parse_model(helper.make_model(graph))
    assert "sig_out" not in ir_graph.values

    _materialize_missing_values(ir_graph)

    assert ir_graph.values["sig_out"].shape == [2, 2]
    assert ir_graph.values["sig_out"].axes == ir_graph.values["x"].axes


def test_scan_pattern_with_declared_state_creates_buffer_and_edges() -> None:
    body = helper.make_graph(
        [helper.make_node("Add", ["state_in", "x_t"], ["state_out"], name="body_add")],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1])],
    )
    scan = helper.make_node(
        "Scan",
        ["state_init", "scan_input"],
        ["state_final", "scan_output"],
        name="scan_node",
        num_scan_inputs=1,
        body=body,
    )
    graph = helper.make_graph(
        [scan],
        "scan_graph",
        [
            helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("scan_input", TensorProto.FLOAT, [4, 1]),
        ],
        [helper.make_tensor_value_info("scan_output", TensorProto.FLOAT, [4, 1])],
    )
    model = helper.make_model(graph, producer_name="scan_state_buffer")

    result = TemporalONNXParser().parse_model(model, process_id="scan_cov")

    buffer_id = "scan_node_state_init_buffer"
    assert buffer_id in result.process.buffers
    spec = result.process.buffers[buffer_id]
    assert spec.metadata == {"pattern": "Scan", "state_input": "state_init"}
    assert spec.depth == 1
    assert "state_init" in result.report.states
    assert any(edge.source == buffer_id for edge in result.process.edge0)
    assert any(
        edge.target == buffer_id and edge.lag_cycles == 1
        for edge in result.process.edge_delta
    )


def test_conv_attr_as_int_prefers_plural_list_then_default() -> None:
    plural = _conv_attr_as_int(
        {"strides": [2, 2]}, singular="stride", plural="strides", default=1
    )
    fallback = _conv_attr_as_int({}, singular="stride", plural="strides", default=7)

    assert plural == 2
    assert fallback == 7


def test_coerce_attr_int_handles_float_and_rejects_strings() -> None:
    assert _coerce_attr_int(2.0, default=0) == 2

    with pytest.raises(TypeError, match="expected numeric attribute"):
        _coerce_attr_int("bad", default=0)
