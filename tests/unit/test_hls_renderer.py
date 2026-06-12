import re
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tempo_dag.codegen.hls.generator import (
    HLSTemplateNotFoundError,
    HLSTemplateRenderError,
    render_operator_hls,
    resolve_hls_template_path,
)
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import Add


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


class CustomTemplateOperator(Operator):
    OP_TYPE = "CustomTemplate"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=1)

    def hls_template_path(self) -> str:
        return "templates/custom_template.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"op_id": self.op_id, "op_type": self.op_type, "scale": 7}


class MissingTemplateOperator(Operator):
    OP_TYPE = "MissingTemplate"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=1)

    def hls_template_path(self) -> str:
        return "templates/not_here.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"op_id": self.op_id, "op_type": self.op_type}


class MissingContextOperator(Operator):
    OP_TYPE = "MissingContext"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=1)

    def hls_template_path(self) -> str:
        return "templates/missing_context.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"op_id": self.op_id, "op_type": self.op_type}


def test_resolve_hls_template_path_finds_repo_managed_builtin_template():
    operator = Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])

    template_path = resolve_hls_template_path(operator)

    assert template_path == (Path.cwd() / "hls" / "operators" / "add.cpp.tpl").resolve()


def test_render_operator_hls_renders_builtin_template():
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_tensor("rhs", [2, 3], ["batch", "feature"]),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    operator = Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])

    rendered = render_operator_hls(operator, values)

    assert "Operator: Add" in rendered
    assert "Kernel: add_0_kernel" in rendered
    assert "Inputs: ['lhs', 'rhs']" in rendered


def test_render_operator_hls_resolves_custom_template_relative_to_module():
    operator = CustomTemplateOperator(
        op_id="custom_0",
        inputs=["x"],
        outputs=["y"],
    )

    rendered = render_operator_hls(operator, values={})

    assert "Custom operator CustomTemplate" in rendered
    assert "scale=7" in rendered


def test_render_operator_hls_raises_for_missing_template_file():
    operator = MissingTemplateOperator(
        op_id="missing_0",
        inputs=["x"],
        outputs=["y"],
    )

    with pytest.raises(
        HLSTemplateNotFoundError,
        match=re.escape("not_here.cpp.tpl"),
    ):
        render_operator_hls(operator, values={})


def test_render_operator_hls_raises_for_missing_context_keys():
    operator = MissingContextOperator(
        op_id="missing_ctx_0",
        inputs=["x"],
        outputs=["y"],
    )

    with pytest.raises(
        HLSTemplateRenderError,
        match="missing context value 'required_value'",
    ):
        render_operator_hls(operator, values={})


def test_resolve_hls_template_absolute_path_missing():
    class AbsoluteMissingOp(Operator):
        OP_TYPE = "AbsMiss"

        def validate(self, values):
            pass

        def estimate_fpga_cost(self, values):
            return FPGACost(latency_cycles=1)

        def hls_template_path(self):
            return "C:/non_existent_absolute_path_12345/template.cpp.tpl"

        def hls_context(self, values):
            return {}

    with pytest.raises(
        HLSTemplateNotFoundError,
        match="(does not exist|could not be resolved) for operator",
    ):
        resolve_hls_template_path(AbsoluteMissingOp(op_id="op1", inputs=[], outputs=[]))


def test_render_operator_hls_malformed_syntax():
    class BadOp(Operator):
        OP_TYPE = "BadOp"

        def validate(self, values):
            pass

        def estimate_fpga_cost(self, values):
            return FPGACost(latency_cycles=1)

        def hls_template_path(self):
            return "C:/bad.cpp.tpl"

        def hls_context(self, values):
            return {}

    mock_path_obj = MagicMock()
    mock_path_obj.read_text.return_value = "Hello ${unmatched"

    with (
        pytest.raises(HLSTemplateRenderError, match="invalid for operator"),
        patch(
            "tempo_dag.codegen.hls.generator.resolve_hls_template_path",
            return_value=mock_path_obj,
        ),
    ):
        render_operator_hls(BadOp(op_id="bad1", inputs=[], outputs=[]), {})


def test_render_operator_hls_empty_context():
    class EmptyCtxOp(Operator):
        OP_TYPE = "EmptyCtx"

        def validate(self, values):
            pass

        def estimate_fpga_cost(self, values):
            return FPGACost(latency_cycles=1)

        def hls_template_path(self):
            return "C:/empty.cpp.tpl"

        def hls_context(self, values):
            return {}

    mock_path_obj = MagicMock()
    mock_path_obj.read_text.return_value = "No variables here"

    with patch(
        "tempo_dag.codegen.hls.generator.resolve_hls_template_path",
        return_value=mock_path_obj,
    ):
        res = render_operator_hls(EmptyCtxOp(op_id="empty1", inputs=[], outputs=[]), {})
    assert res == "No variables here"

