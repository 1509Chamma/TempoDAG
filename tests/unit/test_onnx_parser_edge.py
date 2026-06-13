import onnx.helper as helper
import pytest
from onnx import AttributeProto, TensorProto

from tempo_dag.parsers.onnx.parser import ONNXParser


@pytest.fixture
def parser():
    return ONNXParser()


def test_unsupported_operator(parser):
    node = helper.make_node("UnsupportedOp", ["X"], ["Y"], name="n1")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    with pytest.raises(ValueError, match="Unsupported ONNX operator: UnsupportedOp"):
        parser.parse_model(model)


def test_dynamic_shapes(parser):
    # Create an input with dynamic shape
    node = helper.make_node("Relu", ["X"], ["Y"])

    # Create dimension that doesn't have dim_value (dynamic)
    dynamic_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [])
    # Hack to add dynamic dim
    dynamic_tensor.type.tensor_type.shape.dim.add().dim_param = "batch"

    graph = helper.make_graph(
        [node],
        "g",
        [dynamic_tensor],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)

    ir_graph = parser.parse_model(model)
    # The parser replaces dynamic dims with 1
    assert ir_graph.values["X"].shape == [1]


def test_empty_model(parser):
    graph = helper.make_graph([], "empty_graph", [], [])
    model = helper.make_model(graph)
    ir_graph = parser.parse_model(model)
    assert len(ir_graph.ops) == 0
    assert len(ir_graph.values) == 0


def test_model_with_only_constants(parser):
    # initializer only
    tensor = helper.make_tensor("const_val", TensorProto.FLOAT, [2], [1.0, 2.0])
    graph = helper.make_graph([], "const_graph", [], [], initializer=[tensor])
    model = helper.make_model(graph)

    ir_graph = parser.parse_model(model)
    assert "const_val" in ir_graph.values
    assert len(ir_graph.ops) == 0
    assert ir_graph.values["const_val"].shape == [2]


def test_incorrect_attribute_types(parser):
    # Create an attribute with an unhandled type (like a sparse tensor
    # without the others set)
    # We can just manually construct an AttributeProto that has none
    # of the covered fields
    attr = AttributeProto()
    attr.name = "weird_attr"
    attr.type = AttributeProto.TENSOR  # but we don't set .t

    node = helper.make_node("Relu", ["X"], ["Y"])
    node.attribute.extend([attr])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)

    ir_graph = parser.parse_model(model)
    # The default op_id for Relu_0 is depends on the number of ops if not named.
    # In parse_model: op_id = node.name or f"{op_type}_{len(ops)}"
    # For Relu, op_type is "ReLU" (from mapping)
    assert ir_graph.ops["ReLU_0"].attrs["weird_attr"] is None


def test_all_attribute_types(parser):
    # Test f, i, s, floats, ints, strings
    node = helper.make_node(
        "Relu",
        ["X"],
        ["Y"],
        f_val=1.5,
        i_val=42,
        s_val=b"test",
        floats_val=[1.5, 2.5],
        ints_val=[1, 2],
        strings_val=[b"a", b"b"],
    )
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    ir_graph = parser.parse_model(model)

    attrs = ir_graph.ops["ReLU_0"].attrs
    assert attrs["f_val"] == 1.5
    assert attrs["i_val"] == 42
    assert attrs["s_val"] == "test"
    assert attrs["floats_val"] == [1.5, 2.5]
    assert attrs["ints_val"] == [1, 2]
    assert attrs["strings_val"] == ["a", "b"]
