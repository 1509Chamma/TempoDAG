import math
from collections.abc import Mapping
from typing import cast

from tempo_dag.codegen.hls.generator import render_operator_hls
from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError, Operator
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import register_builtin_operators


def make_tensor(value_id, shape, axes=None, dtype="float32"):
    if axes is None:
        axes = [f"axis_{idx}" for idx in range(len(shape))]
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype=dtype,
        shape=list(shape),
        axes=list(axes),
    )


class CustomScaleOperator(Operator):
    OP_TYPE = "CustomScale"

    def validate(self, values: Mapping[str, Value]) -> None:
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise InvalidOperatorInstanceError(
                "CustomScale expects exactly 1 input and 1 output"
            )
        if "scale" not in self.attrs or not isinstance(self.attrs["scale"], int):
            raise InvalidOperatorInstanceError(
                "CustomScale requires integer 'scale' in attrs"
            )

        input_value = values[self.inputs[0]]
        output_value = values[self.outputs[0]]
        if input_value.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                "CustomScale expects tensor input values"
            )
        if output_value.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                "CustomScale expects tensor output values"
            )
        if input_value.shape != output_value.shape:
            raise InvalidOperatorInstanceError(
                "CustomScale requires input and output shapes to match"
            )
        if input_value.axes != output_value.axes:
            raise InvalidOperatorInstanceError(
                "CustomScale requires input and output axes to match"
            )
        if input_value.dtype != output_value.dtype:
            raise InvalidOperatorInstanceError(
                "CustomScale requires input and output dtypes to match"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = math.prod(values[self.outputs[0]].shape)
        return FPGACost(
            latency_cycles=work,
            initiation_interval=1,
            dsp=1,
            lut=work,
            ff=work,
            metadata={"heuristic": "custom_scale"},
        )

    def hls_template_path(self) -> str:
        return "templates/custom_scale.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "scale": self.attrs["scale"],
        }


def test_operator_registry_flow_serializes_graph_estimates_cost_and_renders_hls():
    registry = OperatorRegistry()
    register_builtin_operators(registry)
    registry.register(CustomScaleOperator)

    values = {
        "x": make_tensor("x", [1, 4], ["batch", "feature"]),
        "bias": make_tensor("bias", [1, 4], ["batch", "feature"]),
        "sum_out": make_tensor("sum_out", [1, 4], ["batch", "feature"]),
        "gain": make_tensor("gain", [1, 4], ["batch", "feature"]),
        "scaled_out": make_tensor("scaled_out", [1, 4], ["batch", "feature"]),
        "custom_out": make_tensor("custom_out", [1, 4], ["batch", "feature"]),
    }
    graph = Graph(
        values=values,
        ops={},
        graph_inputs=["x", "bias", "gain"],
        graph_outputs=["scaled_out"],
        registry=registry,
    )

    add_operator = graph.create_operator(
        "Add",
        op_id="add_0",
        inputs=["x", "bias"],
        outputs=["sum_out"],
    )
    mul_operator = graph.create_operator(
        "Mul",
        op_id="mul_0",
        inputs=["sum_out", "gain"],
        outputs=["scaled_out"],
    )

    serialized = graph.to_dict()
    serialized_ops = cast(dict[str, dict[str, object]], serialized["ops"])
    assert serialized["graph_inputs"] == ["x", "bias", "gain"]
    assert serialized["graph_outputs"] == ["scaled_out"]
    assert serialized_ops["add_0"]["op_type"] == "Add"
    assert serialized_ops["mul_0"]["inputs"] == ["sum_out", "gain"]

    assert add_operator.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=4,
        initiation_interval=1,
        lut=4,
        ff=4,
        metadata={"heuristic": "binary_elementwise"},
    )
    assert mul_operator.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=5,
        initiation_interval=1,
        dsp=4,
        lut=4,
        ff=4,
        metadata={"heuristic": "binary_mul"},
    )

    builtin_hls = render_operator_hls(add_operator, values)
    assert "Operator: Add" in builtin_hls
    assert "Kernel: add_0_kernel" in builtin_hls

    custom_operator = registry.create(
        "CustomScale",
        op_id="custom_scale_0",
        inputs=["scaled_out"],
        outputs=["custom_out"],
        attrs={"scale": 3},
    )
    assert custom_operator.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=4,
        initiation_interval=1,
        dsp=1,
        lut=4,
        ff=4,
        metadata={"heuristic": "custom_scale"},
    )

    custom_hls = render_operator_hls(custom_operator, values)
    assert "Custom scale operator CustomScale" in custom_hls
    assert "scale=3" in custom_hls
