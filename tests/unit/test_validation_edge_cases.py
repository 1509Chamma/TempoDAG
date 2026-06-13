import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.validation import (
    GraphValidationError,
    OperatorValidationError,
    ValueValidationError,
    validate_graph,
    validate_operators,
    validate_values,
)
from tempo_dag.ir.value import Value, ValueType


class DummyOp(Operator):
    OP_TYPE = "Dummy"

    def validate(self, values):
        pass

    def estimate_fpga_cost(self, values):
        return FPGACost(latency_cycles=1)

    def hls_template_path(self):
        return ""

    def hls_context(self, values):
        return {}


def make_val(
    vid,
    shape=None,
    axes=None,
    producer=None,
    dtype="float32",
    vtype=ValueType.TENSOR,
    quant=None,
):
    if shape is None:
        shape = [1]
    if axes is None:
        axes = ["N"]
    return Value(
        value_id=vid,
        vtype=vtype,
        dtype=dtype,
        shape=shape,
        axes=axes,
        producer_op_id=producer,
        quant=quant,
    )


def test_value_zero_dimensional():
    """Zero-dimensional tensors (shape = []) should be valid."""
    val = make_val("v1", shape=[], axes=[])
    graph = Graph({"v1": val}, {}, ["v1"], ["v1"])
    validate_values(graph)


def test_value_single_element():
    """Single-element tensors (shape = [1]) should be valid."""
    val = make_val("v1", shape=[1], axes=["N"])
    graph = Graph({"v1": val}, {}, ["v1"], ["v1"])
    validate_values(graph)


@pytest.mark.parametrize(
    "shape,axes,err_match",
    [
        ([0], ["N"], "positive integer"),
        ([-1], ["N"], "positive integer"),
        ([1.5], ["N"], "positive integer"),
        ([1], ["N", "C"], r"axes length \(2\) must match shape length \(1\)"),
    ],
)
def test_value_invalid_shape_axes(shape, axes, err_match):
    """Value shape and axes should be validated properly."""
    val = make_val("v1", shape=shape, axes=axes)
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(ValueValidationError, match=err_match):
        validate_values(graph)


def test_value_missing_shape():
    """Value shape must be a list."""
    # Manually bypass make_val to test raw Value validation
    val = Value("v1", ValueType.TENSOR, "float32", shape=None, axes=["N"])  # type: ignore
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(ValueValidationError, match="shape must be a list"):
        validate_values(graph)


def test_value_missing_axes():
    """Value axes must be a list."""
    val = Value("v1", ValueType.TENSOR, "float32", shape=[1], axes=None)  # type: ignore
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(ValueValidationError, match="axes must be a list"):
        validate_values(graph)


@pytest.mark.parametrize(
    "dtype,err_match",
    [
        (None, "valid string dtype"),
        ("", "valid string dtype"),
        (123, "valid string dtype"),
    ],
)
def test_value_invalid_dtype(dtype, err_match):
    """Value should fail on missing or invalid dtype."""
    val = make_val("v1", dtype=dtype)
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(ValueValidationError, match=err_match):
        validate_values(graph)


def test_value_invalid_vtype():
    """Value should fail with invalid vtype."""
    val = make_val("v1", vtype="InvalidType")
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(ValueValidationError, match="invalid vtype"):
        validate_values(graph)


def test_value_invalid_quant():
    """Value quantization should be a dict if present."""
    val = make_val("v1", quant="not_a_dict")
    graph = Graph({"v1": val}, {}, ["v1"], [])
    with pytest.raises(
        ValueValidationError, match="quantization config must be a dictionary"
    ):
        validate_values(graph)


def test_graph_empty():
    """Empty graph should pass validation."""
    graph = Graph({}, {}, [], [])
    validate_graph(graph)


def test_graph_duplicate_value_id():
    """Graphs with duplicate value IDs should fail."""
    graph = Graph({"v1": make_val("v1")}, {}, ["v1"], [])

    # The validator now checks list(graph.values.keys()) and uses Counter.
    # We can simulate duplicates by providing a list with duplicates to keys().
    class DuplicateDict(dict):
        def keys(self):  # type: ignore
            return ["v1", "v1"]

    graph.values = DuplicateDict({"v1": make_val("v1")})  # type: ignore
    with pytest.raises(GraphValidationError, match="Duplicate value_id found: v1"):
        validate_graph(graph)


def test_graph_duplicate_op_id():
    """Graphs with duplicate operator IDs should fail."""
    graph = Graph({"v1": make_val("v1")}, {}, ["v1"], [])

    class DuplicateOpDict(dict):
        def keys(self):  # type: ignore
            return ["op1", "op1"]

    graph.ops = DuplicateOpDict(  # type: ignore
        {"op1": DummyOp(op_id="op1", inputs=["v1"], outputs=["v1"])}
    )
    with pytest.raises(GraphValidationError, match="Duplicate op_id found: op1"):
        validate_graph(graph)


def test_graph_input_not_in_values():
    """Graph input should fail if not registered in values."""
    graph = Graph({"v1": make_val("v1")}, {}, ["v2"], [])
    with pytest.raises(GraphValidationError, match="Graph input 'v2' not found"):
        validate_graph(graph)


def test_graph_output_not_in_values():
    """Graph output should fail if not registered in values."""
    graph = Graph({"v1": make_val("v1")}, {}, [], ["v2"])
    with pytest.raises(GraphValidationError, match="Graph output 'v2' not found"):
        validate_graph(graph)


def test_op_missing_output_in_values():
    """Operator output should fail if not registered in values."""
    op = DummyOp(op_id="op1", inputs=["v1"], outputs=["missing"])
    graph = Graph({"v1": make_val("v1")}, {"op1": op}, ["v1"], [])
    with pytest.raises(
        GraphValidationError, match="references non-existent output 'missing'"
    ):
        validate_graph(graph)


def test_value_producer_not_in_ops():
    """Value with producer_op_id must point to existing operator."""
    val = make_val("v1", producer="missing_op")
    graph = Graph({"v1": val}, {}, [], ["v1"])
    with pytest.raises(
        GraphValidationError,
        match="references non-existent producer_op_id 'missing_op'",
    ):
        validate_graph(graph)


def test_value_claims_producer_but_not_in_outputs():
    """Value claims to be produced by an op, but is not in op.outputs."""
    val = make_val("v1", producer="op1")
    op = DummyOp(op_id="op1", inputs=[], outputs=["v2"])
    val2 = make_val("v2", producer="op1")
    graph = Graph({"v1": val, "v2": val2}, {"op1": op}, [], ["v1"])
    with pytest.raises(
        GraphValidationError,
        match="claims to be produced by 'op1', but is not in its outputs",
    ):
        validate_graph(graph)


def test_op_no_outputs():
    """Operator must have at least one output."""
    op = DummyOp(op_id="op1", inputs=["v1"], outputs=[])
    graph = Graph({"v1": make_val("v1")}, {"op1": op}, ["v1"], [])
    with pytest.raises(OperatorValidationError, match="Operator 'op1' has no outputs"):
        validate_operators(graph)
