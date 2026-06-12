import pytest

from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError
from tempo_dag.ir.registry import get_default_registry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import Add, Concat, Conv1D, MatMul, Sum


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


def make_scalar(value_id, dtype="float32"):
    return Value(
        value_id=value_id,
        vtype=ValueType.SCALAR,
        dtype=dtype,
        shape=[],
        axes=[],
    )


def test_default_registry_creates_builtin_operator_instances():
    operator = get_default_registry().create(
        "Add",
        op_id="add_0",
        inputs=["lhs", "rhs"],
        outputs=["out"],
    )

    assert isinstance(operator, Add)
    assert operator.hls_template_path() == "hls/operators/add.cpp.tpl"


def test_add_validation_and_cost_for_matching_tensor_shapes():
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_tensor("rhs", [2, 3], ["batch", "feature"]),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    operator = Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])

    operator.validate(values)
    cost = operator.estimate_fpga_cost(values)

    assert cost == FPGACost(
        latency_cycles=6,
        initiation_interval=1,
        lut=6,
        ff=6,
        metadata={"heuristic": "binary_elementwise"},
    )
    assert operator.hls_context(values)["input_shapes"] == [[2, 3], [2, 3]]


def test_add_rejects_incompatible_tensor_shapes_without_scalar_broadcasting():
    values = {
        "lhs": make_tensor("lhs", [2, 3]),
        "rhs": make_tensor("rhs", [2, 4]),
        "out": make_tensor("out", [2, 3]),
    }
    operator = Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])

    with pytest.raises(
        InvalidOperatorInstanceError,
        match="Add requires matching tensor shapes or scalar broadcasting",
    ):
        operator.validate(values)


def test_add_allows_scalar_broadcasting():
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_scalar("rhs"),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    operator = Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])

    operator.validate(values)


def test_matmul_validation_and_cost_are_shape_aware():
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["rows", "inner"]),
        "rhs": make_tensor("rhs", [3, 4], ["inner", "cols"]),
        "out": make_tensor("out", [2, 4], ["rows", "cols"]),
    }
    operator = MatMul(op_id="matmul_0", inputs=["lhs", "rhs"], outputs=["out"])

    operator.validate(values)
    assert operator.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=24,
        initiation_interval=1,
        dsp=3,
        lut=8,
        ff=8,
        metadata={"heuristic": "matmul"},
    )


def test_matmul_rejects_inner_dimension_mismatch():
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["rows", "inner"]),
        "rhs": make_tensor("rhs", [5, 4], ["inner", "cols"]),
        "out": make_tensor("out", [2, 4], ["rows", "cols"]),
    }
    operator = MatMul(op_id="matmul_0", inputs=["lhs", "rhs"], outputs=["out"])

    with pytest.raises(
        InvalidOperatorInstanceError,
        match="MatMul requires lhs.shape\\[1\\] == rhs.shape\\[0\\], got 3 and 5",
    ):
        operator.validate(values)


def test_concat_and_reduction_validate_expected_output_shapes():
    concat_values = {
        "a": make_tensor("a", [2, 3], ["batch", "feature"]),
        "b": make_tensor("b", [2, 2], ["batch", "feature"]),
        "out": make_tensor("out", [2, 5], ["batch", "feature"]),
    }
    concat = Concat(
        op_id="concat_0",
        inputs=["a", "b"],
        outputs=["out"],
        attrs={"axis": 1},
    )
    concat.validate(concat_values)

    reduction_values = {
        "x": make_tensor("x", [2, 3, 4], ["batch", "time", "feature"]),
        "out": make_tensor("out", [2, 4], ["batch", "feature"]),
    }
    reduction = Sum(
        op_id="sum_0",
        inputs=["x"],
        outputs=["out"],
        attrs={"axis": 1},
    )
    reduction.validate(reduction_values)
    assert reduction.estimate_fpga_cost(reduction_values) == FPGACost(
        latency_cycles=24,
        initiation_interval=1,
        lut=24,
        ff=12,
        metadata={"heuristic": "reduction"},
    )


def test_conv1d_validation_and_cost_follow_output_length_formula():
    values = {
        "x": make_tensor("x", [1, 2, 8], ["batch", "channel", "time"]),
        "w": make_tensor("w", [4, 2, 3], ["out_channel", "in_channel", "kernel"]),
        "y": make_tensor("y", [1, 4, 8], ["batch", "channel", "time"]),
    }
    operator = Conv1D(
        op_id="conv_0",
        inputs=["x", "w"],
        outputs=["y"],
        attrs={"stride": 1, "padding": 1, "dilation": 1},
    )

    operator.validate(values)
    assert operator.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=192,
        initiation_interval=1,
        dsp=6,
        bram=1,
        lut=32,
        ff=32,
        metadata={"heuristic": "conv1d"},
    )


def test_conv1d_rejects_incorrect_output_shape():
    values = {
        "x": make_tensor("x", [1, 2, 8], ["batch", "channel", "time"]),
        "w": make_tensor("w", [4, 2, 3], ["out_channel", "in_channel", "kernel"]),
        "y": make_tensor("y", [1, 4, 7], ["batch", "channel", "time"]),
    }
    operator = Conv1D(
        op_id="conv_0",
        inputs=["x", "w"],
        outputs=["y"],
        attrs={"stride": 1, "padding": 1, "dilation": 1},
    )

    with pytest.raises(
        InvalidOperatorInstanceError,
        match="Conv1D expects output shape \\[1, 4, 8\\], got \\[1, 4, 7\\]",
    ):
        operator.validate(values)

