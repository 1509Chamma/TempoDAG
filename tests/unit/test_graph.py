from collections.abc import Mapping
from typing import cast

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType


class PassThroughOperator(Operator):
    OP_TYPE = "PassThrough"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=1)

    def hls_template_path(self) -> str:
        return "pass_through.cpp"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"input": self.inputs[0], "output": self.outputs[0]}


class ShiftOperator(Operator):
    OP_TYPE = "Shift"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=2, ff=4)

    def hls_template_path(self) -> str:
        return "shift.cpp"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"shift": self.attrs.get("amount", 0)}


def build_tensor_value(value_id):
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1, 8],
        axes=["batch", "time"],
    )


def test_graph_accepts_operator_instances_and_preserves_serialization_shape():
    x_value = build_tensor_value("x")
    y_value = build_tensor_value("y")
    op = PassThroughOperator(op_id="pass_0", inputs=["x"], outputs=["y"])

    graph = Graph(
        values={"x": x_value, "y": y_value},
        ops={"pass_0": op},
        graph_inputs=["x"],
        graph_outputs=["y"],
    )

    assert graph.to_dict() == {
        "values": {
            "x": x_value.to_dict(),
            "y": y_value.to_dict(),
        },
        "ops": {
            "pass_0": {
                "op_id": "pass_0",
                "op_type": "PassThrough",
                "inputs": ["x"],
                "outputs": ["y"],
                "attrs": {},
                "name": None,
                "source_span": None,
            }
        },
        "graph_inputs": ["x"],
        "graph_outputs": ["y"],
        "states": {},
    }


def test_graph_create_operator_uses_registry_and_stores_result():
    registry = OperatorRegistry()
    registry.register(ShiftOperator)

    graph = Graph(
        values={"x": build_tensor_value("x"), "y": build_tensor_value("y")},
        ops={},
        graph_inputs=["x"],
        graph_outputs=["y"],
        registry=registry,
    )

    operator = graph.create_operator(
        "Shift",
        op_id="shift_0",
        inputs=["x"],
        outputs=["y"],
        attrs={"amount": 3},
        name="time_shift",
    )

    assert operator is graph.ops["shift_0"]
    assert operator.to_dict() == {
        "op_id": "shift_0",
        "op_type": "Shift",
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {"amount": 3},
        "name": "time_shift",
        "source_span": None,
    }


def test_graph_rejects_non_operator_instances():
    invalid_ops = cast(dict[str, Operator], {"bad_0": object()})

    with pytest.raises(TypeError, match="ops must contain Operator instances"):
        Graph(values={}, ops=invalid_ops, graph_inputs=[], graph_outputs=[])


def test_graph_rejects_mismatched_operator_dictionary_keys():
    operator = PassThroughOperator(op_id="pass_0", inputs=["x"], outputs=["y"])

    with pytest.raises(
        ValueError,
        match="operator key 'wrong_key' does not match operator.op_id 'pass_0'",
    ):
        Graph(
            values={},
            ops={"wrong_key": operator},
            graph_inputs=[],
            graph_outputs=[],
        )

