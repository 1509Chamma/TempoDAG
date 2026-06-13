import re
from collections.abc import Mapping
from typing import Any, cast

import pytest

from tempo_dag.ir.op import (
    FPGACost,
    InvalidOperatorDefinitionError,
    InvalidOperatorInstanceError,
    Operator,
)
from tempo_dag.ir.value import Value


class DummyOperator(Operator):
    OP_TYPE = "Dummy"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=4, initiation_interval=1, dsp=1, lut=8, ff=16)

    def hls_template_path(self) -> str:
        return "dummy.cpp"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"op_id": self.op_id, "inputs": self.inputs, "outputs": self.outputs}


def test_operator_to_dict_preserves_existing_schema():
    op = DummyOperator(
        op_id="op_0",
        inputs=["lhs", "rhs"],
        outputs=["out"],
        attrs={"axis": 1},
        name="accumulate",
        source_span="model.py:10",
    )

    assert op.to_dict() == {
        "op_id": "op_0",
        "op_type": "Dummy",
        "inputs": ["lhs", "rhs"],
        "outputs": ["out"],
        "attrs": {"axis": 1},
        "name": "accumulate",
        "source_span": "model.py:10",
    }


def test_operator_is_abstract():
    abstract_operator = cast(Any, Operator)

    with pytest.raises(TypeError):
        abstract_operator(op_id="op_0", inputs=["x"], outputs=["y"])


def test_concrete_operator_requires_non_empty_op_type():
    with pytest.raises(InvalidOperatorDefinitionError):

        class MissingOpTypeOperator(Operator):
            def validate(self, values: Mapping[str, Value]) -> None:
                return None

            def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
                return FPGACost(latency_cycles=1)

            def hls_template_path(self) -> str:
                return "missing.cpp"

            def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
                return {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"op_id": "", "inputs": ["x"], "outputs": ["y"]},
            "op_id must be a non-empty string",
        ),
        (
            {"op_id": "op_0", "inputs": "x", "outputs": ["y"]},
            "inputs must be a sequence of non-empty strings",
        ),
        (
            {"op_id": "op_0", "inputs": ["x"], "outputs": [1]},
            "outputs[0] must be a non-empty string",
        ),
        (
            {"op_id": "op_0", "inputs": ["x"], "outputs": ["y"], "attrs": []},
            "attrs must be a dictionary when set",
        ),
    ],
)
def test_invalid_operator_instance_data_raises_clear_error(kwargs, message):
    with pytest.raises(InvalidOperatorInstanceError, match=re.escape(message)):
        DummyOperator(**kwargs)


def test_fpga_cost_rejects_negative_values():
    with pytest.raises(ValueError, match="latency_cycles must be non-negative"):
        FPGACost(latency_cycles=-1)
