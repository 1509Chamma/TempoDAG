"""Coverage-focused tests for tempo_dag.ops.builtins.

Exercises validation, cost-estimation, and HLS-context paths for every
built-in operator, including the error branches the main suites skip.
"""

import pytest

from tempo_dag.ir.op import InvalidOperatorInstanceError
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import (
    BUILTIN_OPERATOR_TYPES,
    BUILTIN_OPERATORS,
    LSTM,
    Add,
    Concat,
    Conv1D,
    Div,
    LayerNorm,
    MatMul,
    Mean,
    Mul,
    Pad,
    Reshape,
    Shift,
    Sigmoid,
    Slice,
    Softmax,
    Sub,
    Sum,
    Transpose,
    _cpp_bool,
    _cpp_dtype,
    _is_int_sequence,
    _shape_product,
    _snake_case,
    register_builtin_operators,
)


def make_tensor(vid, shape, axes=None, dtype="float32"):
    if axes is None:
        axes = [f"axis_{i}" for i in range(len(shape))]
    return Value(
        value_id=vid,
        vtype=ValueType.TENSOR,
        dtype=dtype,
        shape=list(shape),
        axes=list(axes),
    )


def make_scalar(vid, dtype="float32"):
    return Value(value_id=vid, vtype=ValueType.SCALAR, dtype=dtype, shape=[], axes=[])


def make_state(vid, shape=None, dtype="float32"):
    shape = list(shape or [])
    return Value(
        value_id=vid,
        vtype=ValueType.STATE,
        dtype=dtype,
        shape=shape,
        axes=[f"axis_{i}" for i in range(len(shape))],
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_snake_case_converts_camel_case_names():
    assert _snake_case("Add") == "add"
    assert _snake_case("LayerNorm") == "layer_norm"


def test_cpp_dtype_and_bool_helpers():
    assert _cpp_dtype("float32") == "float"
    assert _cpp_dtype("float64") == "double"
    assert _cpp_dtype("int32") == "std::int32_t"
    assert _cpp_dtype("bfloat16") == "bfloat16"
    assert _cpp_bool(True) == "true"
    assert _cpp_bool(False) == "false"


def test_is_int_sequence_rejects_strings_and_mixed_items():
    assert _is_int_sequence([1, 2, 3])
    assert not _is_int_sequence("123")
    assert not _is_int_sequence([1, "2"])
    assert not _is_int_sequence(7)


def test_shape_product_handles_empty_and_rejects_non_positive_dims():
    assert _shape_product([]) == 1
    assert _shape_product([2, 3, 4]) == 24
    with pytest.raises(InvalidOperatorInstanceError, match="must be positive"):
        _shape_product([2, 0])


# ---------------------------------------------------------------------------
# Shared BuiltinOperator helpers
# ---------------------------------------------------------------------------


def test_input_count_mismatch_reports_single_expected_count():
    op = Add(op_id="a", inputs=["x"], outputs=["out"])
    with pytest.raises(
        InvalidOperatorInstanceError, match="Add expects 2 inputs, got 1"
    ):
        op.validate({})


def test_output_count_mismatch_is_rejected():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["o1", "o2"])
    with pytest.raises(
        InvalidOperatorInstanceError, match="Add expects 1 outputs, got 2"
    ):
        op.validate({})


def test_input_count_mismatch_reports_expected_choices():
    op = Conv1D(op_id="c", inputs=["x"], outputs=["y"])
    with pytest.raises(
        InvalidOperatorInstanceError, match="Conv1D expects one of 2, 3 inputs, got 1"
    ):
        op.validate({})


def test_unknown_input_and_output_values_are_reported():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    with pytest.raises(InvalidOperatorInstanceError, match="unknown input value 'x'"):
        op.validate({})

    values = {"x": make_tensor("x", [2]), "y": make_tensor("y", [2])}
    with pytest.raises(
        InvalidOperatorInstanceError, match="unknown output value 'out'"
    ):
        op.validate(values)


def test_matmul_rejects_scalar_input():
    op = MatMul(op_id="m", inputs=["a", "b"], outputs=["c"])
    values = {
        "a": make_scalar("a"),
        "b": make_tensor("b", [2, 2]),
        "c": make_tensor("c", [2, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError,
        match=r"expects input\[0\] to be a tensor, got scalar",
    ):
        op.validate(values)


def test_binary_rejects_state_operand():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    values = {
        "x": make_state("x", [2]),
        "y": make_tensor("y", [2]),
        "out": make_tensor("out", [2]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="to be a scalar or tensor"):
        op.validate(values)


def test_require_same_dtype_allows_empty_and_rejects_mixed():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    assert op._require_same_dtype([]) is None

    values = {
        "x": make_tensor("x", [2]),
        "y": make_tensor("y", [2], dtype="int32"),
        "out": make_tensor("out", [2]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="share dtype float32"):
        op.validate(values)


def test_resolve_axis_type_and_range_checks():
    op = Softmax(op_id="s", inputs=["x"], outputs=["out"], attrs={"axis": 5})
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    with pytest.raises(
        InvalidOperatorInstanceError, match="axis 5 is out of range for rank 2"
    ):
        op.validate(values)

    with pytest.raises(
        InvalidOperatorInstanceError, match="requires 'axis' to be an integer"
    ):
        op._resolve_axis("last", 2)


def test_attr_type_validation_errors():
    concat = Concat(op_id="c", inputs=["a", "b"], outputs=["out"])
    values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 2]),
        "out": make_tensor("out", [4, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="requires 'axis' to be of type int"
    ):
        concat.validate(values)

    softmax = Softmax(op_id="s", inputs=["x"], outputs=["out"], attrs={"axis": "last"})
    sm_values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    with pytest.raises(
        InvalidOperatorInstanceError, match="requires 'axis' to be of type int"
    ):
        softmax.validate(sm_values)

    reduce_sum = Sum(op_id="r", inputs=["x"], outputs=["out"], attrs={"keepdims": 1})
    red_values = {"x": make_tensor("x", [2, 3]), "out": make_scalar("out")}
    with pytest.raises(
        InvalidOperatorInstanceError, match="requires 'keepdims' to be of type bool"
    ):
        reduce_sum.validate(red_values)


def test_int_sequence_attr_validation():
    transpose = Transpose(op_id="t", inputs=["x"], outputs=["out"])
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [3, 2])}
    with pytest.raises(InvalidOperatorInstanceError, match="requires 'perm' in attrs"):
        transpose.validate(values)

    transpose.attrs["perm"] = "10"
    with pytest.raises(
        InvalidOperatorInstanceError,
        match="requires 'perm' to be a sequence of integers",
    ):
        transpose.validate(values)


def test_match_output_shape_axes_dtype_and_vtype_checks():
    def build(out_value):
        op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
        values = {
            "x": make_tensor("x", [2, 3], ["batch", "feature"]),
            "y": make_tensor("y", [2, 3], ["batch", "feature"]),
            "out": out_value,
        }
        return op, values

    op, values = build(make_tensor("out", [3, 2], ["batch", "feature"]))
    with pytest.raises(InvalidOperatorInstanceError, match="expects output shape"):
        op.validate(values)

    op, values = build(make_tensor("out", [2, 3], ["rows", "cols"]))
    with pytest.raises(InvalidOperatorInstanceError, match="expects output axes"):
        op.validate(values)

    op, values = build(make_tensor("out", [2, 3], ["batch", "feature"], dtype="int32"))
    with pytest.raises(
        InvalidOperatorInstanceError, match="expects output dtype float32"
    ):
        op.validate(values)

    state_out = Value(
        value_id="out",
        vtype=ValueType.STATE,
        dtype="float32",
        shape=[2, 3],
        axes=["batch", "feature"],
    )
    op, values = build(state_out)
    with pytest.raises(
        InvalidOperatorInstanceError, match="expects output value type tensor"
    ):
        op.validate(values)


# ---------------------------------------------------------------------------
# Elementwise operators
# ---------------------------------------------------------------------------


def test_binary_scalar_scalar_produces_scalar_output():
    op = Sub(op_id="s", inputs=["x", "y"], outputs=["out"])
    values = {
        "x": make_scalar("x"),
        "y": make_scalar("y"),
        "out": make_scalar("out"),
    }
    op.validate(values)
    assert op.estimate_fpga_cost(values).latency_cycles == 1

    ctx = op.hls_context(values)
    assert ctx["has_scalar_lhs"] == "true"
    assert ctx["has_scalar_rhs"] == "true"
    assert ctx["input_0_size"] == 1


def test_binary_scalar_lhs_broadcasts_to_tensor_output():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    values = {
        "x": make_scalar("x"),
        "y": make_tensor("y", [2, 2], ["r", "c"]),
        "out": make_tensor("out", [2, 2], ["r", "c"]),
    }
    op.validate(values)
    ctx = op.hls_context(values)
    assert ctx["has_scalar_lhs"] == "true"
    assert ctx["has_scalar_rhs"] == "false"


def test_binary_scalar_rhs_broadcasts_to_tensor_output():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    values = {
        "x": make_tensor("x", [2, 2], ["r", "c"]),
        "y": make_scalar("y"),
        "out": make_tensor("out", [2, 2], ["r", "c"]),
    }
    op.validate(values)
    assert op.hls_context(values)["has_scalar_rhs"] == "true"


def test_binary_rejects_mismatched_tensor_shapes():
    op = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    values = {
        "x": make_tensor("x", [2, 2]),
        "y": make_tensor("y", [3, 3]),
        "out": make_tensor("out", [2, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError,
        match="matching tensor shapes or scalar broadcasting",
    ):
        op.validate(values)


def test_unary_scalar_and_tensor_paths():
    op = Sigmoid(op_id="s", inputs=["x"], outputs=["out"])

    scalar_values = {"x": make_scalar("x"), "out": make_scalar("out")}
    op.validate(scalar_values)
    assert op.estimate_fpga_cost(scalar_values).latency_cycles == 1

    tensor_values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    op.validate(tensor_values)
    cost = op.estimate_fpga_cost(tensor_values)
    assert cost.latency_cycles == 6
    assert cost.metadata == {"heuristic": "unary_elementwise"}


def test_mul_and_div_cost_heuristics():
    values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 2]),
        "c": make_tensor("c", [2, 2]),
    }
    mul = Mul(op_id="m", inputs=["a", "b"], outputs=["c"])
    mul_cost = mul.estimate_fpga_cost(values)
    assert mul_cost.latency_cycles == 5
    assert mul_cost.dsp == 4
    assert mul_cost.metadata == {"heuristic": "binary_mul"}

    div = Div(op_id="d", inputs=["a", "b"], outputs=["c"])
    div_cost = div.estimate_fpga_cost(values)
    assert div_cost.latency_cycles == 16
    assert div_cost.lut == 8
    assert div_cost.metadata == {"heuristic": "binary_div"}


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_reduction_without_axis_reduces_all_dims_to_scalar():
    op = Mean(op_id="m", inputs=["x"], outputs=["out"])
    values = {"x": make_tensor("x", [2, 3]), "out": make_scalar("out")}
    op.validate(values)


def test_reduction_axis_sequence_and_keepdims():
    op = Sum(op_id="s", inputs=["x"], outputs=["out"], attrs={"axis": [2, 1]})
    values = {
        "x": make_tensor("x", [2, 3, 4], ["b", "t", "f"]),
        "out": make_tensor("out", [2], ["b"]),
    }
    op.validate(values)

    keep = Sum(
        op_id="k", inputs=["x"], outputs=["out"], attrs={"axis": 1, "keepdims": True}
    )
    keep_values = {
        "x": make_tensor("x", [2, 3], ["b", "f"]),
        "out": make_tensor("out", [2, 1], ["b", "f"]),
    }
    keep.validate(keep_values)


def test_reduction_axis_error_branches():
    op = Sum(op_id="s", inputs=["x"], outputs=["out"], attrs={"axis": [0, -2]})
    values = {"x": make_tensor("x", [2, 3]), "out": make_scalar("out")}
    with pytest.raises(InvalidOperatorInstanceError, match="unique reduction axes"):
        op.validate(values)

    op.attrs["axis"] = "all"
    with pytest.raises(
        InvalidOperatorInstanceError, match="integer or sequence of integers"
    ):
        op.validate(values)


def test_reduction_cost_and_hls_context_suffix_and_non_suffix():
    op = Sum(op_id="s", inputs=["x"], outputs=["out"], attrs={"axis": 1})
    values = {
        "x": make_tensor("x", [2, 3], ["b", "f"]),
        "out": make_tensor("out", [2], ["b"]),
    }
    op.validate(values)
    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 6
    assert cost.metadata == {"heuristic": "reduction"}

    ctx = op.hls_context(values)
    assert ctx["input_size"] == 6
    assert ctx["output_size"] == 2
    assert ctx["reduction_size"] == 3

    bad = Sum(op_id="b", inputs=["x"], outputs=["out"], attrs={"axis": 0})
    bad_values = {
        "x": make_tensor("x", [2, 3], ["b", "f"]),
        "out": make_tensor("out", [3], ["f"]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="contiguous suffix reductions"
    ):
        bad.hls_context(bad_values)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def test_softmax_validate_cost_and_hls_context():
    op = Softmax(op_id="s", inputs=["x"], outputs=["out"])
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 12
    assert cost.metadata == {"heuristic": "softmax"}

    ctx = op.hls_context(values)
    assert ctx["axis_size"] == 3
    assert ctx["outer_size"] == 2
    assert ctx["inner_size"] == 1


def test_softmax_rejects_scalar_input():
    op = Softmax(op_id="s", inputs=["x"], outputs=["out"])
    values = {"x": make_scalar("x"), "out": make_scalar("out")}
    with pytest.raises(InvalidOperatorInstanceError, match="to be a tensor"):
        op.validate(values)


# ---------------------------------------------------------------------------
# MatMul
# ---------------------------------------------------------------------------


def test_matmul_validate_cost_and_hls_context():
    values = {
        "a": make_tensor("a", [2, 3], ["rows", "inner"]),
        "b": make_tensor("b", [3, 4], ["inner", "cols"]),
        "c": make_tensor("c", [2, 4], ["rows", "cols"]),
    }
    op = MatMul(op_id="m", inputs=["a", "b"], outputs=["c"])
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 24
    assert cost.dsp == 3

    ctx = op.hls_context(values)
    assert ctx["m_dim"] == 2
    assert ctx["k_dim"] == 3
    assert ctx["n_dim"] == 4
    assert ctx["k_dim_pow2"] == 4


def test_matmul_shape_error_branches():
    op = MatMul(op_id="m", inputs=["a", "b"], outputs=["c"])
    rank_values = {
        "a": make_tensor("a", [2, 3, 4]),
        "b": make_tensor("b", [3, 4]),
        "c": make_tensor("c", [2, 4]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="rank-2 tensor inputs only"):
        op.validate(rank_values)

    mismatch = {
        "a": make_tensor("a", [2, 3]),
        "b": make_tensor("b", [5, 4]),
        "c": make_tensor("c", [2, 4]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match=r"lhs.shape\[1\] == rhs.shape\[0\]"
    ):
        op.validate(mismatch)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_transpose_validate_cost_and_hls_context():
    op = Transpose(op_id="t", inputs=["x"], outputs=["out"], attrs={"perm": [1, 0]})
    values = {
        "x": make_tensor("x", [2, 3], ["rows", "cols"]),
        "out": make_tensor("out", [3, 2], ["cols", "rows"]),
    }
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 6
    assert cost.bram == 1

    ctx = op.hls_context(values)
    assert ctx["rows"] == 2
    assert ctx["cols"] == 3


def test_transpose_error_branches():
    op = Transpose(op_id="t", inputs=["x"], outputs=["out"], attrs={"perm": [1, 0, 2]})
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [3, 2])}
    with pytest.raises(InvalidOperatorInstanceError, match="requires 'perm' length 2"):
        op.validate(values)

    op.attrs["perm"] = [0, 5]
    with pytest.raises(InvalidOperatorInstanceError, match="out of range"):
        op.validate(values)

    op.attrs["perm"] = [0, 0]
    with pytest.raises(InvalidOperatorInstanceError, match="permutation of axes"):
        op.validate(values)


def test_transpose_hls_context_requires_rank2_swap():
    op = Transpose(op_id="t", inputs=["x"], outputs=["out"], attrs={"perm": [0, 1]})
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    with pytest.raises(InvalidOperatorInstanceError, match="rank-2"):
        op.hls_context(values)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


def test_reshape_validate_cost_and_hls_context():
    op = Reshape(op_id="r", inputs=["x"], outputs=["out"], attrs={"shape": [6]})
    values = {
        "x": make_tensor("x", [2, 3], ["b", "f"]),
        "out": make_tensor("out", [6], ["flat"]),
    }
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 1
    assert cost.metadata == {"heuristic": "reshape"}
    assert op.hls_context(values)["num_elements"] == 6


def test_reshape_shape_attr_error_branches():
    op = Reshape(op_id="r", inputs=["x"], outputs=["out"], attrs={"shape": [5]})
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [5])}
    with pytest.raises(InvalidOperatorInstanceError, match="preserve element count"):
        op.validate(values)

    op.attrs["shape"] = [0, 6]
    with pytest.raises(InvalidOperatorInstanceError, match="positive integers"):
        op.validate(values)


def test_reshape_output_error_branches():
    op = Reshape(op_id="r", inputs=["x"], outputs=["out"], attrs={"shape": [6]})
    x = make_tensor("x", [2, 3])

    with pytest.raises(
        InvalidOperatorInstanceError, match=r"expects output shape \[6\]"
    ):
        op.validate({"x": x, "out": make_tensor("out", [3, 2])})

    bad_axes = Value(
        value_id="out",
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[6],
        axes=["a", "b"],
    )
    with pytest.raises(
        InvalidOperatorInstanceError, match="axes length to match output rank"
    ):
        op.validate({"x": x, "out": bad_axes})

    with pytest.raises(
        InvalidOperatorInstanceError, match="tensor output with dtype float32"
    ):
        op.validate({"x": x, "out": make_tensor("out", [6], ["flat"], dtype="int32")})

    state_out = Value(
        value_id="out",
        vtype=ValueType.STATE,
        dtype="float32",
        shape=[6],
        axes=["flat"],
    )
    with pytest.raises(
        InvalidOperatorInstanceError, match="tensor output with dtype float32"
    ):
        op.validate({"x": x, "out": state_out})


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def test_concat_validate_cost_and_hls_context():
    op = Concat(op_id="c", inputs=["a", "b"], outputs=["out"], attrs={"axis": 1})
    values = {
        "a": make_tensor("a", [2, 3], ["b", "f"]),
        "b": make_tensor("b", [2, 2], ["b", "f"]),
        "out": make_tensor("out", [2, 5], ["b", "f"]),
    }
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 10
    assert cost.bram == 1

    ctx = op.hls_context(values)
    assert ctx["num_inputs"] == 2
    assert ctx["input_sizes_csv"] == "6, 4"
    assert ctx["output_size"] == 10


def test_concat_error_branches():
    op = Concat(op_id="c", inputs=["a", "b"], outputs=["out"], attrs={"axis": 0})
    rank_values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 2, 2]),
        "out": make_tensor("out", [4, 2]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="same rank"):
        op.validate(rank_values)

    dim_values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 3]),
        "out": make_tensor("out", [4, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="non-concatenated dimensions"
    ):
        op.validate(dim_values)

    axes_values = {
        "a": make_tensor("a", [2, 2], ["r", "c"]),
        "b": make_tensor("b", [2, 2], ["x", "y"]),
        "out": make_tensor("out", [4, 2], ["r", "c"]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="matching axes"):
        op.validate(axes_values)


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def test_slice_validate_cost_and_hls_context():
    op = Slice(
        op_id="s",
        inputs=["x"],
        outputs=["out"],
        attrs={"axis": 0, "start": 2, "end": 8, "step": 2},
    )
    values = {
        "x": make_tensor("x", [10], ["t"]),
        "out": make_tensor("out", [3], ["t"]),
    }
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 3
    assert cost.metadata == {"heuristic": "slice"}

    ctx = op.hls_context(values)
    assert ctx["start"] == 2
    assert ctx["end"] == 8
    assert ctx["step"] == 2
    assert ctx["output_size"] == 3


def test_slice_error_branches():
    op = Slice(
        op_id="s",
        inputs=["x"],
        outputs=["out"],
        attrs={"axis": 0, "start": 0, "end": 2, "step": 0},
    )
    values = {"x": make_tensor("x", [4]), "out": make_tensor("out", [2])}
    with pytest.raises(
        InvalidOperatorInstanceError, match="'step' to be a positive integer"
    ):
        op.validate(values)

    op.attrs["step"] = 1
    op.attrs["end"] = 9
    with pytest.raises(
        InvalidOperatorInstanceError, match="start < end <= input dimension"
    ):
        op.validate(values)


def test_slice_hls_context_requires_rank1_axis0():
    op = Slice(
        op_id="s",
        inputs=["x"],
        outputs=["out"],
        attrs={"axis": 1, "start": 0, "end": 2},
    )
    values = {"x": make_tensor("x", [2, 4]), "out": make_tensor("out", [2, 2])}
    with pytest.raises(InvalidOperatorInstanceError, match="rank-1 slices"):
        op.hls_context(values)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


def test_layer_norm_validate_cost_and_hls_context():
    op = LayerNorm(op_id="ln", inputs=["x"], outputs=["out"])
    values = {"x": make_tensor("x", [2, 4]), "out": make_tensor("out", [2, 4])}
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 16
    assert cost.dsp == 2
    assert cost.metadata == {"heuristic": "layer_norm"}

    ctx = op.hls_context(values)
    assert ctx["normalized_size"] == 4
    assert ctx["outer_size"] == 2
    assert ctx["epsilon"] == pytest.approx(1e-5)

    custom = LayerNorm(
        op_id="ln2", inputs=["x"], outputs=["out"], attrs={"epsilon": 1e-3}
    )
    assert custom.hls_context(values)["epsilon"] == pytest.approx(1e-3)


def test_layer_norm_rejects_scalar_input():
    op = LayerNorm(op_id="ln", inputs=["x"], outputs=["out"])
    values = {"x": make_scalar("x"), "out": make_scalar("out")}
    with pytest.raises(InvalidOperatorInstanceError, match="to be a tensor"):
        op.validate(values)


# ---------------------------------------------------------------------------
# Conv1D
# ---------------------------------------------------------------------------


def conv1d_values(bias=None):
    values = {
        "x": make_tensor("x", [1, 2, 8], ["b", "c", "t"]),
        "w": make_tensor("w", [4, 2, 3], ["o", "i", "k"]),
        "y": make_tensor("y", [1, 4, 6], ["b", "c", "t"]),
    }
    if bias is not None:
        values["bias"] = bias
    return values


def test_conv1d_validates_bias_variants():
    op = Conv1D(op_id="cv", inputs=["x", "w", "bias"], outputs=["y"])
    op.validate(conv1d_values(bias=make_tensor("bias", [4], ["c"])))
    op.validate(conv1d_values(bias=make_scalar("bias")))

    with pytest.raises(
        InvalidOperatorInstanceError, match="bias to be scalar or tensor"
    ):
        op.validate(conv1d_values(bias=make_state("bias")))

    with pytest.raises(
        InvalidOperatorInstanceError, match="bias shape to match output channels"
    ):
        op.validate(conv1d_values(bias=make_tensor("bias", [3], ["c"])))


def test_conv1d_rank_and_channel_error_branches():
    op = Conv1D(op_id="cv", inputs=["x", "w"], outputs=["y"])

    rank_values = conv1d_values()
    rank_values["w"] = make_tensor("w", [4, 2], ["o", "i"])
    with pytest.raises(
        InvalidOperatorInstanceError, match="rank-3 input and weight tensors"
    ):
        op.validate(rank_values)

    channel_values = conv1d_values()
    channel_values["w"] = make_tensor("w", [4, 3, 3], ["o", "i", "k"])
    with pytest.raises(
        InvalidOperatorInstanceError,
        match="input channels to match weight channels",
    ):
        op.validate(channel_values)


def test_conv1d_attr_and_dtype_error_branches():
    op = Conv1D(op_id="cv", inputs=["x", "w"], outputs=["y"])

    dtype_values = conv1d_values()
    dtype_values["w"] = make_tensor("w", [4, 2, 3], ["o", "i", "k"], dtype="int32")
    with pytest.raises(InvalidOperatorInstanceError, match="matching dtypes"):
        op.validate(dtype_values)

    neg = Conv1D(op_id="cv2", inputs=["x", "w"], outputs=["y"], attrs={"padding": -1})
    with pytest.raises(
        InvalidOperatorInstanceError, match="'padding' to be a non-negative"
    ):
        neg.validate(conv1d_values())

    zero_stride = Conv1D(
        op_id="cv3", inputs=["x", "w"], outputs=["y"], attrs={"stride": 0}
    )
    with pytest.raises(
        InvalidOperatorInstanceError, match="'stride' and 'dilation' to be positive"
    ):
        zero_stride.validate(conv1d_values())

    wide_values = {
        "x": make_tensor("x", [1, 2, 2]),
        "w": make_tensor("w", [4, 2, 5]),
        "y": make_tensor("y", [1, 4, 1]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="invalid kernel/padding/dilation"
    ):
        op.validate(wide_values)


def test_conv1d_validate_and_cost_without_bias():
    op = Conv1D(op_id="cv", inputs=["x", "w"], outputs=["y"])
    values = conv1d_values()
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 144  # 1 * 4 * 6 * 2 * 3
    assert cost.dsp == 6
    assert cost.bram == 1
    assert cost.metadata == {"heuristic": "conv1d"}


def test_conv1d_hls_context_bias_variants():
    no_bias = Conv1D(op_id="cv", inputs=["x", "w"], outputs=["y"])
    ctx = no_bias.hls_context(conv1d_values())
    assert ctx["has_bias"] == "false"
    assert ctx["bias_parameter"] == ""
    assert ctx["bias_init"] == "(float)0"
    assert ctx["output_length"] == 6
    assert ctx["kernel_width"] == 3

    with_bias = Conv1D(op_id="cv2", inputs=["x", "w", "bias"], outputs=["y"])
    scalar_ctx = with_bias.hls_context(conv1d_values(bias=make_scalar("bias")))
    assert scalar_ctx["bias_init"] == "bias[0]"
    assert "bias[1]" in scalar_ctx["bias_parameter"]

    channel_ctx = with_bias.hls_context(
        conv1d_values(bias=make_tensor("bias", [4], ["c"]))
    )
    assert channel_ctx["bias_init"] == "bias[out_channel]"
    assert "bias[4]" in channel_ctx["bias_parameter"]

    with pytest.raises(
        InvalidOperatorInstanceError, match="HLS template supports scalar"
    ):
        with_bias.hls_context(conv1d_values(bias=make_tensor("bias", [2, 2])))


# ---------------------------------------------------------------------------
# Pad
# ---------------------------------------------------------------------------


def test_pad_validate_cost_and_hls_context():
    op = Pad(op_id="p", inputs=["x"], outputs=["out"], attrs={"pads": [1, 2]})
    values = {
        "x": make_tensor("x", [4], ["t"]),
        "out": make_tensor("out", [7], ["t"]),
    }
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 7
    assert cost.metadata == {"heuristic": "pad"}

    ctx = op.hls_context(values)
    assert ctx["input_size"] == 4
    assert ctx["output_size"] == 7
    assert ctx["pad_before"] == 1
    assert ctx["pad_after"] == 2


def test_pad_error_branches():
    op = Pad(op_id="p", inputs=["x"], outputs=["out"], attrs={"pads": [1]})
    values = {"x": make_tensor("x", [4]), "out": make_tensor("out", [7])}
    with pytest.raises(InvalidOperatorInstanceError, match="requires 'pads' length 2"):
        op.validate(values)

    op.attrs["pads"] = [-1, 2]
    with pytest.raises(InvalidOperatorInstanceError, match="non-negative integers"):
        op.validate(values)


def test_pad_hls_context_requires_rank1():
    op = Pad(op_id="p", inputs=["x"], outputs=["out"], attrs={"pads": [1, 1, 1, 1]})
    values = {"x": make_tensor("x", [2, 2]), "out": make_tensor("out", [4, 4])}
    with pytest.raises(InvalidOperatorInstanceError, match="rank-1 padding"):
        op.hls_context(values)


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


def test_shift_validate_cost_and_hls_context():
    op = Shift(
        op_id="sh", inputs=["x"], outputs=["out"], attrs={"axis": 0, "amount": -1}
    )
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    op.validate(values)

    cost = op.estimate_fpga_cost(values)
    assert cost.latency_cycles == 6
    assert cost.bram == 1
    assert cost.metadata == {"heuristic": "shift"}

    ctx = op.hls_context(values)
    assert ctx["amount"] == -1
    assert ctx["output_size"] == 6


def test_shift_rejects_zero_amount():
    op = Shift(
        op_id="sh", inputs=["x"], outputs=["out"], attrs={"axis": 0, "amount": 0}
    )
    values = {"x": make_tensor("x", [2, 3]), "out": make_tensor("out", [2, 3])}
    with pytest.raises(InvalidOperatorInstanceError, match="non-zero 'amount'"):
        op.validate(values)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def make_lstm_values(
    *,
    seq_len=5,
    batch=2,
    input_size=4,
    hidden_size=3,
    num_directions=1,
    with_bias=False,
):
    values = {
        "x": make_tensor("x", [seq_len, batch, input_size]),
        "w": make_tensor("w", [num_directions, 4 * hidden_size, input_size]),
        "r": make_tensor("r", [num_directions, 4 * hidden_size, hidden_size]),
        "y": make_tensor("y", [seq_len, num_directions, batch, hidden_size]),
    }
    if with_bias:
        values["b"] = make_tensor("b", [num_directions, 8 * hidden_size])
    return values


def test_lstm_validates_forward_with_bias_and_bidirectional():
    op = LSTM(
        op_id="l", inputs=["x", "w", "r", "b"], outputs=["y"], attrs={"hidden_size": 3}
    )
    op.validate(make_lstm_values(with_bias=True))

    bi = LSTM(
        op_id="l2",
        inputs=["x", "w", "r"],
        outputs=["y"],
        attrs={"hidden_size": 3, "direction": "bidirectional"},
    )
    bi.validate(make_lstm_values(num_directions=2))


def test_lstm_allows_zero_outputs():
    op = LSTM(op_id="l", inputs=["x", "w", "r"], outputs=[], attrs={"hidden_size": 3})
    op.validate(make_lstm_values())


def test_lstm_shape_error_branches():
    op = LSTM(
        op_id="l", inputs=["x", "w", "r"], outputs=["y"], attrs={"hidden_size": 3}
    )

    bad_x = make_lstm_values()
    bad_x["x"] = make_tensor("x", [5, 4])
    with pytest.raises(InvalidOperatorInstanceError, match="rank-3 input X"):
        op.validate(bad_x)

    bad_w = make_lstm_values()
    bad_w["w"] = make_tensor("w", [1, 8, 4])
    with pytest.raises(InvalidOperatorInstanceError, match="weight W shape"):
        op.validate(bad_w)

    bad_r = make_lstm_values()
    bad_r["r"] = make_tensor("r", [1, 12, 5])
    with pytest.raises(InvalidOperatorInstanceError, match="recurrence R shape"):
        op.validate(bad_r)

    with_bias = LSTM(
        op_id="l2", inputs=["x", "w", "r", "b"], outputs=["y"], attrs={"hidden_size": 3}
    )
    bad_bias = make_lstm_values(with_bias=True)
    bad_bias["b"] = make_tensor("b", [1, 12])
    with pytest.raises(InvalidOperatorInstanceError, match="bias B shape"):
        with_bias.validate(bad_bias)

    bad_y = make_lstm_values()
    bad_y["y"] = make_tensor("y", [5, 1, 2, 4])
    with pytest.raises(InvalidOperatorInstanceError, match="output Y shape"):
        op.validate(bad_y)


def test_lstm_cost_heuristic():
    op = LSTM(
        op_id="l", inputs=["x", "w", "r"], outputs=["y"], attrs={"hidden_size": 3}
    )
    cost = op.estimate_fpga_cost(make_lstm_values())
    # work = 5 * 2 * 1 * 4 * (4 + 3) * 3 = 840 -> latency 840 // 4
    assert cost.latency_cycles == 210
    assert cost.dsp == 4
    assert cost.lut == 100
    assert cost.metadata == {"heuristic": "lstm"}


def test_lstm_hls_context_without_and_with_bias():
    op = LSTM(
        op_id="l", inputs=["x", "w", "r"], outputs=["y"], attrs={"hidden_size": 3}
    )
    ctx = op.hls_context(make_lstm_values())
    assert ctx["seq_len"] == 5
    assert ctx["batch"] == 2
    assert ctx["input_size"] == 4
    assert ctx["hidden_size"] == 3
    assert ctx["num_directions"] == 1
    assert ctx["has_bias"] == "false"
    assert ctx["reverse_direction"] == "false"
    assert ctx["gate_i_bias"] == "(float)0"
    assert ctx["bias_parameter"] == ""

    biased = LSTM(
        op_id="l2",
        inputs=["x", "w", "r", "b"],
        outputs=["y"],
        attrs={"hidden_size": 3, "direction": "reverse"},
    )
    biased_ctx = biased.hls_context(make_lstm_values(with_bias=True))
    assert biased_ctx["has_bias"] == "true"
    assert biased_ctx["reverse_direction"] == "true"
    assert "b[direction][hidden_idx]" in biased_ctx["gate_i_bias"]
    assert "b[1][24]" in biased_ctx["bias_parameter"]


def test_lstm_hls_context_error_branches():
    bad_direction = LSTM(
        op_id="l",
        inputs=["x", "w", "r"],
        outputs=["y"],
        attrs={"hidden_size": 3, "direction": "sideways"},
    )
    with pytest.raises(InvalidOperatorInstanceError, match="direction to be forward"):
        bad_direction.hls_context(make_lstm_values())

    extra_values = make_lstm_values(with_bias=True)
    extra_values["p"] = make_tensor("p", [1])
    too_many = LSTM(
        op_id="l2",
        inputs=["x", "w", "r", "b", "p"],
        outputs=["y"],
        attrs={"hidden_size": 3},
    )
    with pytest.raises(InvalidOperatorInstanceError, match="optional B inputs only"):
        too_many.hls_context(extra_values)

    no_output = LSTM(
        op_id="l3", inputs=["x", "w", "r"], outputs=[], attrs={"hidden_size": 3}
    )
    with pytest.raises(InvalidOperatorInstanceError, match="emits the Y output only"):
        no_output.hls_context(make_lstm_values())


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_register_builtin_operators_populates_registry():
    registry = OperatorRegistry()
    register_builtin_operators(registry)
    assert registry.list_registered() == sorted(BUILTIN_OPERATOR_TYPES)
    assert len(BUILTIN_OPERATORS) == len(BUILTIN_OPERATOR_TYPES)
    assert "LSTM" in BUILTIN_OPERATOR_TYPES


def test_hls_template_paths_follow_snake_case_convention():
    add = Add(op_id="a", inputs=["x", "y"], outputs=["out"])
    assert add.hls_template_path() == "hls/operators/add.cpp.tpl"

    layer_norm = LayerNorm(op_id="ln", inputs=["x"], outputs=["out"])
    assert layer_norm.hls_template_path() == "hls/operators/layer_norm.cpp.tpl"
