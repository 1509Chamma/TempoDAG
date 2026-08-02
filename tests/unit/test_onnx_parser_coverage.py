"""Coverage-focused tests for tempo_dag.parsers.onnx.parser."""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as helper
import pytest
from onnx import AttributeProto, TensorProto

from tempo_dag.ir.registry import get_default_registry
from tempo_dag.parsers.onnx.parser import ONNXParser, _loop_state_inputs


def _single_node_model(node, inputs, outputs, initializer=None):
    graph = helper.make_graph(
        [node],
        "coverage_graph",
        inputs,
        outputs,
        initializer=initializer or [],
    )
    return helper.make_model(graph, producer_name="coverage")


def test_register_op_mapping_enables_custom_operator() -> None:
    parser = ONNXParser()
    parser.register_op_mapping("MyCustomRelu", "ReLU")

    node = helper.make_node("MyCustomRelu", ["X"], ["Y"], name="custom_node")
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    )

    ir_graph = parser.parse_model(model)

    assert ir_graph.ops["custom_node"].op_type == "ReLU"


def test_registry_fallback_resolves_case_insensitive_op() -> None:
    parser = ONNXParser(registry=get_default_registry())

    node = helper.make_node("matmul", ["X", "W"], ["Y"], name="mm_node")
    model = _single_node_model(
        node,
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
    )

    ir_graph = parser.parse_model(model)

    assert ir_graph.ops["mm_node"].op_type == "MatMul"


def test_conv_dilations_flattened_to_scalar_attr() -> None:
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        name="conv_node",
        strides=[2],
        pads=[1, 1],
        dilations=[2],
    )
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4])],
        initializer=[
            helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3], [0.25, 0.5, 0.25])
        ],
    )

    ir_graph = ONNXParser().parse_model(model)

    attrs = ir_graph.ops["conv_node"].attrs
    assert attrs["stride"] == 2
    assert attrs["padding"] == 1
    assert attrs["dilation"] == 2


def test_gemm_without_bias_lowers_to_single_matmul() -> None:
    node = helper.make_node("Gemm", ["a", "b"], ["y"], name="gemm_node")
    model = _single_node_model(
        node,
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
    )

    ir_graph = ONNXParser().parse_model(model)

    assert ir_graph.ops["gemm_node"].op_type == "MatMul"
    assert "gemm_node_matmul" not in ir_graph.ops
    assert ir_graph.ops["gemm_node"].inputs == ["a", "b"]
    assert ir_graph.ops["gemm_node"].outputs == ["y"]


def test_lstm_hidden_size_inferred_from_recurrence_weights() -> None:
    node = helper.make_node("LSTM", ["x", "w", "r"], ["y"], name="lstm_node")
    model = _single_node_model(
        node,
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [5, 1, 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 8, 3]),
            helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 8, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [5, 1, 1, 2])],
    )

    ir_graph = ONNXParser().parse_model(model)

    assert ir_graph.ops["lstm_node"].attrs["hidden_size"] == 2


def test_detects_lstm_recurrent_pattern_with_state_inputs() -> None:
    node = helper.make_node(
        "LSTM",
        ["x", "w", "r", "b", "seq_lens", "h0", "c0"],
        ["y"],
    )
    model = _single_node_model(node, [], [])

    patterns = ONNXParser().detect_temporal_patterns(model)

    assert len(patterns) == 1
    assert patterns[0].op_type == "LSTM"
    assert patterns[0].node_name == "LSTM_0"
    assert patterns[0].stateful_inputs == ("h0", "c0")
    assert patterns[0].body_node_count == 0


def test_tensor_attribute_converted_to_numpy_array() -> None:
    tensor = helper.make_tensor("t", TensorProto.FLOAT, [2], [1.0, 2.0])
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu_t", tensor_attr=tensor)
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    )

    ir_graph = ONNXParser().parse_model(model)

    np.testing.assert_array_equal(
        ir_graph.ops["relu_t"].attrs["tensor_attr"],
        np.array([1.0, 2.0], dtype=np.float32),
    )


def test_loop_without_body_reports_zero_body_nodes() -> None:
    node = helper.make_node(
        "Loop",
        ["trip_count", "cond", "loop_state"],
        ["final_state"],
        name="loop_node",
    )
    model = _single_node_model(node, [], [])

    patterns = ONNXParser().detect_temporal_patterns(model)

    assert patterns[0].body_node_count == 0
    assert patterns[0].stateful_inputs == ("loop_state",)


def test_scan_without_num_scan_inputs_treats_all_inputs_as_state() -> None:
    node = helper.make_node("Scan", ["s0", "seq"], ["out"], name="scan_node")
    model = _single_node_model(node, [], [])

    patterns = ONNXParser().detect_temporal_patterns(model)

    assert patterns[0].stateful_inputs == ("s0", "seq")
    assert patterns[0].body_node_count == 0


def test_loop_state_inputs_returns_empty_for_other_ops() -> None:
    node = helper.make_node("Add", ["a", "b"], ["c"], name="add_node")

    assert _loop_state_inputs(node) == ()


def test_unresolvable_operator_raises_even_with_registry() -> None:
    parser = ONNXParser(registry=get_default_registry())
    node = helper.make_node("CompletelyUnknownOp", ["X"], ["Y"], name="unknown")
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )

    with pytest.raises(ValueError, match="Unsupported ONNX operator"):
        parser.parse_model(model)


def test_parse_reads_model_from_disk(tmp_path) -> None:
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu_node")
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    )
    model_path = tmp_path / "relu.onnx"
    onnx.save(model, str(model_path))

    ir_graph = ONNXParser().parse(str(model_path))

    assert ir_graph.ops["relu_node"].op_type == "ReLU"


def test_gemm_with_bias_lowers_to_matmul_plus_add() -> None:
    node = helper.make_node("Gemm", ["a", "b", "c"], ["y"], name="gemm_node")
    model = _single_node_model(
        node,
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
    )

    ir_graph = ONNXParser().parse_model(model)

    assert ir_graph.ops["gemm_node_matmul"].op_type == "MatMul"
    assert ir_graph.ops["gemm_node"].op_type == "Add"
    assert ir_graph.ops["gemm_node"].inputs == ["gemm_node_matmul_out", "c"]
    assert ir_graph.values["gemm_node_matmul_out"].shape == [2, 4]


def test_dynamic_dimension_defaults_to_one() -> None:
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu_node")
    dynamic_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [])
    dynamic_input.type.tensor_type.shape.dim.add().dim_param = "batch"
    model = _single_node_model(
        node,
        [dynamic_input],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )

    ir_graph = ONNXParser().parse_model(model)

    assert ir_graph.values["X"].shape == [1]


def test_scalar_and_sequence_attribute_types() -> None:
    node = helper.make_node(
        "Relu",
        ["X"],
        ["Y"],
        name="attr_node",
        f_val=1.5,
        s_val=b"text",
        floats_val=[1.0, 2.0],
        strings_val=[b"a", b"b"],
    )
    empty_attr = AttributeProto()
    empty_attr.name = "empty_attr"
    empty_attr.type = AttributeProto.TENSOR
    node.attribute.extend([empty_attr])
    model = _single_node_model(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )

    attrs = ONNXParser().parse_model(model).ops["attr_node"].attrs

    assert attrs["f_val"] == 1.5
    assert attrs["s_val"] == "text"
    assert attrs["floats_val"] == [1.0, 2.0]
    assert attrs["strings_val"] == ["a", "b"]
    assert attrs["empty_attr"] is None


def test_scan_with_body_counts_nodes_and_reads_num_scan_inputs() -> None:
    body = helper.make_graph(
        [helper.make_node("Add", ["state_in", "x_t"], ["state_out"], name="body_add")],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1])],
    )
    node = helper.make_node(
        "Scan",
        ["state_init", "sequence"],
        ["state_final", "scan_out"],
        name="scan_node",
        num_scan_inputs=1,
        body=body,
    )
    model = _single_node_model(node, [], [])

    patterns = ONNXParser().detect_temporal_patterns(model)

    assert patterns[0].body_node_count == 1
    assert patterns[0].stateful_inputs == ("state_init",)
