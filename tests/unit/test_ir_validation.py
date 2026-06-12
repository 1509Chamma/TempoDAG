from unittest.mock import MagicMock

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.validation import (
    GraphValidationError,
    IRValidationError,
    OperatorValidationError,
    TopologyValidationError,
    ValueValidationError,
    validate_ir,
)
from tempo_dag.ir.value import Value, ValueType


# Mocked Operator for testing
class MockOp(Operator):
    OP_TYPE = "MockOp"

    def validate(self, values):
        pass

    def estimate_fpga_cost(self, values):
        return FPGACost(latency_cycles=1, lut=10, dsp=5, bram=1)

    def hls_template_path(self):
        return ""

    def hls_context(self, values):
        return {}


def create_simple_value(vid, producer=None):
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1, 10],
        axes=["N", "C"],
        producer_op_id=producer,
    )


def test_validate_valid_graph():
    v1 = create_simple_value("v1")
    v2 = create_simple_value("v2", producer="op1")
    op1 = MockOp(op_id="op1", inputs=["v1"], outputs=["v2"])

    graph = Graph(
        values={"v1": v1, "v2": v2},
        ops={"op1": op1},
        graph_inputs=["v1"],
        graph_outputs=["v2"],
    )

    # Should not raise
    validate_ir(graph)


def test_validate_disconnected_subgraph():
    v1 = create_simple_value("v1")
    v2 = create_simple_value("v2", producer="op1")
    # Disconnected part
    v_unreachable = create_simple_value("v_unreachable")  # No producer, not in inputs
    vdisconnected = create_simple_value("v_disconnected", producer="op_disconnected")
    op_disconnected = MockOp(
        op_id="op_disconnected", inputs=["v_unreachable"], outputs=["v_disconnected"]
    )

    graph = Graph(
        values={
            "v1": v1,
            "v2": v2,
            "v_disconnected": vdisconnected,
            "v_unreachable": v_unreachable,
        },
        ops={
            "op1": MockOp(op_id="op1", inputs=["v1"], outputs=["v2"]),
            "op_disconnected": op_disconnected,
        },
        graph_inputs=["v1"],
        graph_outputs=["v2"],
    )

    with pytest.raises(TopologyValidationError, match="unreachable from inputs"):
        validate_ir(graph)


def test_validate_cyclic_graph():
    v1 = create_simple_value("v1", producer="op2")
    v2 = create_simple_value("v2", producer="op1")
    op1 = MockOp(op_id="op1", inputs=["v1"], outputs=["v2"])
    op2 = MockOp(op_id="op2", inputs=["v2"], outputs=["v1"])

    graph = Graph(
        values={"v1": v1, "v2": v2},
        ops={"op1": op1, "op2": op2},
        graph_inputs=["v1"],  # It has a cycle but also an input?
        graph_outputs=["v2"],
    )

    with pytest.raises(TopologyValidationError, match="cycle"):
        validate_ir(graph)


def test_validate_missing_value_reference():
    v1 = create_simple_value("v1")
    op1 = MockOp(op_id="op1", inputs=["v1", "missing_val"], outputs=["v2"])
    v2 = create_simple_value("v2", producer="op1")

    graph = Graph(
        values={"v1": v1, "v2": v2},  # missing_val is not here
        ops={"op1": op1},
        graph_inputs=["v1"],
        graph_outputs=["v2"],
    )

    with pytest.raises(GraphValidationError, match="references non-existent input"):
        validate_ir(graph)


def test_validate_unreachable_output():
    v1 = create_simple_value("v1")
    v2 = create_simple_value("v2", producer="op1")
    v_out = create_simple_value("v_out")  # Not connected to anything

    graph = Graph(
        values={"v1": v1, "v2": v2, "v_out": v_out},
        ops={"op1": MockOp(op_id="op1", inputs=["v1"], outputs=["v2"])},
        graph_inputs=["v1"],
        graph_outputs=["v_out"],
    )

    with pytest.raises(TopologyValidationError, match="unreachable from inputs"):
        validate_ir(graph)


def test_validate_fpga_constraints():
    v1 = create_simple_value("v1")
    v2 = create_simple_value("v2", producer="op1")
    op1 = MockOp(op_id="op1", inputs=["v1"], outputs=["v2"])

    graph = Graph(
        values={"v1": v1, "v2": v2},
        ops={"op1": op1},
        graph_inputs=["v1"],
        graph_outputs=["v2"],
    )

    device = MagicMock()
    device.name = "TestDevice"
    device.resources.luts = 5  # MockOp needs 10
    device.resources.dsps = 100
    device.resources.bram_36k = 100

    with pytest.raises(IRValidationError, match="Insufficient LUTs"):
        validate_ir(graph, device=device)


def test_validate_value_invalid_shape():
    val = Value(
        value_id="v1",
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1, -10],  # Negative dimension
        axes=["N", "C"],
    )

    graph = Graph(values={"v1": val}, ops={}, graph_inputs=["v1"], graph_outputs=[])

    with pytest.raises(
        ValueValidationError, match="shape must contain positive integers"
    ):
        validate_ir(graph)


def test_validate_operator_internal_failure():
    class FailingOp(MockOp):
        def validate(self, values):
            raise ValueError("Something is wrong inside")

    v1 = create_simple_value("v1")
    v2 = create_simple_value("v2", producer="op1")
    op1 = FailingOp(op_id="op1", inputs=["v1"], outputs=["v2"])

    graph = Graph(
        values={"v1": v1, "v2": v2},
        ops={"op1": op1},
        graph_inputs=["v1"],
        graph_outputs=["v2"],
    )

    with pytest.raises(OperatorValidationError, match="Internal validation failed"):
        validate_ir(graph)


def test_custom_operator_extensibility():
    """Test that a custom operator can implement its own validation logic."""

    class RangeOp(MockOp):
        OP_TYPE = "Range"

        def validate(self, values):
            # Custom rule: output shape must be 1D
            for out_id in self.outputs:
                if len(values[out_id].shape) != 1:
                    raise ValueError("RangeOp output must be 1D")

    v1 = create_simple_value("v1")  # shape [1, 10]
    v_out = create_simple_value("v_out", producer="op1")  # shape [1, 10]
    op1 = RangeOp(op_id="op1", inputs=["v1"], outputs=["v_out"])

    graph = Graph(
        values={"v1": v1, "v_out": v_out},
        ops={"op1": op1},
        graph_inputs=["v1"],
        graph_outputs=["v_out"],
    )

    with pytest.raises(OperatorValidationError, match="RangeOp output must be 1D"):
        validate_ir(graph)

