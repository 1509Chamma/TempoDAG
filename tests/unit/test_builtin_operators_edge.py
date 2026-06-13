import pytest

from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import (
    LSTM,
    Add,
    Concat,
    Conv1D,
    Div,
    InvalidOperatorInstanceError,
    MatMul,
    Mul,
    Pad,
    Reshape,
    Slice,
    Softmax,
    Sum,
    Transpose,
)


def make_tensor(vid, shape, axes=None, dtype="float32", vtype=ValueType.TENSOR):
    if axes is None:
        axes = [f"dim_{i}" for i in range(len(shape))]
    return Value(value_id=vid, vtype=vtype, dtype=dtype, shape=shape, axes=axes)


def test_matmul_validation_edge_cases():
    # Rank not 2
    op = MatMul(op_id="m1", inputs=["a", "b"], outputs=["c"])
    values = {
        "a": make_tensor("a", [1, 2, 3]),
        "b": make_tensor("b", [3, 4]),
        "c": make_tensor("c", [1, 4]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="supports rank-2 tensor inputs only"
    ):
        op.validate(values)

    # Mismatched inner dimensions
    values = {
        "a": make_tensor("a", [2, 3]),
        "b": make_tensor("b", [4, 5]),
        "c": make_tensor("c", [2, 5]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError,
        match=r"requires lhs.shape\[1\] == rhs.shape\[0\]",
    ):
        op.validate(values)


def test_transpose_validation_edge_cases():
    op = Transpose(op_id="t1", inputs=["a"], outputs=["b"], attrs={"perm": [1, 0]})
    # Invalid perm length
    values = {"a": make_tensor("a", [1, 2, 3]), "b": make_tensor("b", [2, 1, 3])}
    with pytest.raises(InvalidOperatorInstanceError, match="requires 'perm' length 3"):
        op.validate(values)

    # Not a permutation
    op = Transpose(op_id="t1", inputs=["a"], outputs=["b"], attrs={"perm": [1, 1]})
    values = {"a": make_tensor("a", [2, 2]), "b": make_tensor("b", [2, 2])}
    with pytest.raises(InvalidOperatorInstanceError, match="permutation of axes"):
        op.validate(values)


def test_reshape_validation_edge_cases():
    op = Reshape(op_id="r1", inputs=["a"], outputs=["b"], attrs={"shape": [2, 3]})
    # Mismatched element count
    values = {"a": make_tensor("a", [1, 4]), "b": make_tensor("b", [2, 3])}
    with pytest.raises(InvalidOperatorInstanceError, match="preserve element count"):
        op.validate(values)

    # Negative dimension in target shape
    op = Reshape(op_id="r1", inputs=["a"], outputs=["b"], attrs={"shape": [2, -1]})
    values = {"a": make_tensor("a", [4]), "b": make_tensor("b", [2, -1])}
    with pytest.raises(InvalidOperatorInstanceError, match="contain positive integers"):
        op.validate(values)


def test_concat_validation_edge_cases():
    op = Concat(op_id="c1", inputs=["a", "b"], outputs=["out"], attrs={"axis": 0})
    # Mismatched rank
    values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 2, 2]),
        "out": make_tensor("out", [4, 2]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="same rank"):
        op.validate(values)

    # Mismatched non-concat dimensions
    values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 3]),
        "out": make_tensor("out", [4, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="non-concatenated dimensions to match"
    ):
        op.validate(values)


def test_slice_validation_edge_cases():
    op = Slice(
        op_id="s1", inputs=["a"], outputs=["b"], attrs={"axis": 0, "start": 0, "end": 5}
    )
    # End > dimension
    values = {"a": make_tensor("a", [3]), "b": make_tensor("b", [3])}
    with pytest.raises(
        InvalidOperatorInstanceError, match="start < end <= input dimension"
    ):
        op.validate(values)

    # Negative step
    op = Slice(
        op_id="s1",
        inputs=["a"],
        outputs=["b"],
        attrs={"axis": 0, "start": 0, "end": 2, "step": -1},
    )
    with pytest.raises(InvalidOperatorInstanceError, match="positive integer"):
        op.validate(values)


def test_conv1d_validation_edge_cases():
    # Rank not 3
    op = Conv1D(op_id="cv1", inputs=["in", "w"], outputs=["out"])
    values = {
        "in": make_tensor("in", [1, 16, 10]),
        "w": make_tensor("w", [32, 16]),
        "out": make_tensor("out", [1, 32, 10]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="rank-3"):
        op.validate(values)

    # Input channels mismatch
    values = {
        "in": make_tensor("in", [1, 16, 10]),
        "w": make_tensor("w", [32, 8, 3]),
        "out": make_tensor("out", [1, 32, 8]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError, match="input channels to match weight channels"
    ):
        op.validate(values)


def test_lstm_validation_edge_cases():
    op = LSTM(
        op_id="l1", inputs=["x", "w", "r"], outputs=["y"], attrs={"hidden_size": 128}
    )
    # Input X rank not 3
    values = {
        "x": make_tensor("x", [10, 128]),
        "w": make_tensor("w", [1, 512, 128]),
        "r": make_tensor("r", [1, 512, 128]),
        "y": make_tensor("y", [10, 1, 1, 128]),
    }
    with pytest.raises(InvalidOperatorInstanceError, match="rank-3 input X"):
        op.validate(values)

    # Weight W shape mismatch
    values = {
        "x": make_tensor("x", [10, 1, 64]),
        "w": make_tensor(
            "w", [1, 512, 64]
        ),  # should be 4*hidden_size = 512, but hidden_size=128, so it matches. Wait.
        "r": make_tensor("r", [1, 512, 128]),
        "y": make_tensor("y", [10, 1, 1, 128]),
    }
    # Let's change hidden_size in attrs to 64
    op.attrs["hidden_size"] = 64  # 4*64=256
    with pytest.raises(InvalidOperatorInstanceError, match="expects weight W shape"):
        op.validate(values)


def test_pad_validation_edge_cases():
    op = Pad(op_id="p1", inputs=["in"], outputs=["out"], attrs={"pads": [1, 1]})
    # pads length mismatch (rank 2 tensor needs 4 pads)
    values = {"in": make_tensor("in", [10, 10]), "out": make_tensor("out", [12, 12])}
    with pytest.raises(InvalidOperatorInstanceError, match="requires 'pads' length 4"):
        op.validate(values)


def test_reduction_axes_edge_cases():
    op = Sum(op_id="s1", inputs=["in"], outputs=["out"], attrs={"axis": [0, 0]})
    # Duplicate reduction axes
    values = {"in": make_tensor("in", [10, 10]), "out": make_tensor("out", [])}
    with pytest.raises(InvalidOperatorInstanceError, match="unique reduction axes"):
        op.validate(values)

    # Invalid axis type
    op.attrs["axis"] = "zero"
    with pytest.raises(
        InvalidOperatorInstanceError, match="integer or sequence of integers"
    ):
        op.validate(values)


def test_binary_mismatch():
    op = Add(op_id="a1", inputs=["lhs", "rhs"], outputs=["out"])
    values = {
        "lhs": make_tensor("lhs", [2, 2]),
        "rhs": make_tensor("rhs", [3, 3]),
        "out": make_tensor("out", [2, 2]),
    }
    with pytest.raises(
        InvalidOperatorInstanceError,
        match="matching tensor shapes or scalar broadcasting",
    ):
        op.validate(values)


def test_builtin_op_hls_context():
    op = Add(op_id="a1", inputs=["lhs", "rhs"], outputs=["out"])
    values = {
        "lhs": make_tensor("lhs", [2, 2]),
        "rhs": make_tensor("rhs", [2, 2]),
        "out": make_tensor("out", [2, 2]),
    }
    ctx = op.hls_context(values)
    assert ctx["op_id"] == "a1"
    assert ctx["input_shapes"] == [[2, 2], [2, 2]]
    assert ctx["output_shapes"] == [[2, 2]]


def test_built_in_cost_estimation():
    # Mul cost
    op_mul = Mul(op_id="m1", inputs=["a", "b"], outputs=["c"])
    values = {
        "a": make_tensor("a", [2, 2]),
        "b": make_tensor("b", [2, 2]),
        "c": make_tensor("c", [2, 2]),
    }
    cost = op_mul.estimate_fpga_cost(values)
    assert cost.dsp == 4

    # Div cost
    op_div = Div(op_id="d1", inputs=["a", "b"], outputs=["c"])
    cost = op_div.estimate_fpga_cost(values)
    assert cost.latency_cycles == 16

    # Softmax cost
    op_soft = Softmax(op_id="s1", inputs=["a"], outputs=["b"])
    cost = op_soft.estimate_fpga_cost(values)
    assert cost.latency_cycles == 8

    # Transpose cost
    op_trans = Transpose(
        op_id="t1", inputs=["a"], outputs=["b"], attrs={"perm": [1, 0]}
    )
    cost = op_trans.estimate_fpga_cost(values)
    assert cost.bram == 1
