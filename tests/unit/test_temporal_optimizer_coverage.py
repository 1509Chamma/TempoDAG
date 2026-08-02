"""Coverage-focused tests for `tempo_dag.ir_temporal.optimizer`.

Exercises the guard branches, fused-operator validation errors, and HLS hook
methods that the standard demo processes never trigger: malformed fused
operator instances, declined fusions (wrong producer, runtime bias, shared
intermediates, non-activation consumers), rewrite legality guards, and the
activation expression table.
"""

from copy import deepcopy

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import InvalidOperatorInstanceError, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Clock,
    Edge0,
    EdgeDelta,
    FusedConv1DAdd,
    FusedConv1DAddActivation,
    FusedMatMulAdd,
    FusedMatMulAddActivation,
    FusedScaleAdd,
    FusedScaleAddActivation,
    Kernel,
    Process,
    StateKind,
    StateSpec,
    TemporalOptimizationError,
    fuse_parameterized_conv1d_add,
    fuse_parameterized_matmul_add,
    fuse_parameterized_scale_add,
    optimize_temporal_process,
    share_compatible_temporal_buffers,
    validate_temporal_rewrite,
)
from tempo_dag.ir_temporal.optimizer import _delta
from tempo_dag.ops.builtins import Add, Conv1D, MatMul, Mul, ReLU


def tensor(
    value_id: str,
    shape: list[int],
    *,
    dtype: str = "float32",
    layout: str | None = None,
    quant: dict[str, float | int | str | None] | None = None,
) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype=dtype,
        shape=shape,
        axes=[f"axis_{idx}" for idx in range(len(shape))],
        layout=layout,
        quant=quant,
    )


def scalar(value_id: str, *, dtype: str = "float32") -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.SCALAR,
        dtype=dtype,
        shape=[],
        axes=[],
    )


def matmul_values(**overrides: Value) -> dict[str, Value]:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4]),
        "bias": tensor("bias", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    values.update(overrides)
    return values


def conv_values(**overrides: Value) -> dict[str, Value]:
    values = {
        "x": tensor("x", [1, 2, 8]),
        "w": tensor("w", [3, 2, 3]),
        "bias": tensor("bias", [1, 3, 6]),
        "out": tensor("out", [1, 3, 6]),
    }
    values.update(overrides)
    return values


def scale_values(**overrides: Value) -> dict[str, Value]:
    values = {
        "x": tensor("x", [2, 4]),
        "scale": tensor("scale", [2, 4]),
        "bias": tensor("bias", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    values.update(overrides)
    return values


def matmul_op(
    attrs: dict[str, object] | None = None,
    *,
    cls: type[Operator] = FusedMatMulAdd,
) -> Operator:
    return cls("fused", inputs=["x", "w", "bias"], outputs=["out"], attrs=attrs)


def conv_op(
    attrs: dict[str, object] | None = None,
    *,
    cls: type[Operator] = FusedConv1DAdd,
) -> Operator:
    return cls("fused", inputs=["x", "w", "bias"], outputs=["out"], attrs=attrs)


def scale_op(
    attrs: dict[str, object] | None = None,
    *,
    cls: type[Operator] = FusedScaleAdd,
) -> Operator:
    return cls("fused", inputs=["x", "scale", "bias"], outputs=["out"], attrs=attrs)


# ---------------------------------------------------------------------------
# FusedMatMulAdd / FusedMatMulAddActivation operator contract
# ---------------------------------------------------------------------------


def test_fused_matmul_add_rejects_wrong_input_count() -> None:
    op = FusedMatMulAdd("fused", inputs=["x", "w"], outputs=["out"])
    with pytest.raises(InvalidOperatorInstanceError, match="expects 3 inputs"):
        op.validate({})


def test_fused_matmul_add_rejects_wrong_output_count() -> None:
    op = FusedMatMulAdd(
        "fused",
        inputs=["x", "w", "bias"],
        outputs=["out", "extra"],
    )
    with pytest.raises(InvalidOperatorInstanceError, match="expects 1 output"):
        op.validate({})


def test_fused_matmul_add_rejects_unknown_value() -> None:
    values = matmul_values()
    del values["w"]

    with pytest.raises(InvalidOperatorInstanceError, match="unknown value 'w'"):
        matmul_op().validate(values)


def test_fused_matmul_add_rejects_non_tensor_bias() -> None:
    values = matmul_values(bias=scalar("bias"))

    with pytest.raises(InvalidOperatorInstanceError, match="bias to be a tensor"):
        matmul_op().validate(values)


def test_fused_matmul_add_rejects_dtype_mismatch() -> None:
    values = matmul_values(w=tensor("w", [3, 4], dtype="float64"))

    with pytest.raises(InvalidOperatorInstanceError, match="share dtype"):
        matmul_op().validate(values)


def test_fused_matmul_add_rejects_non_rank2_inputs() -> None:
    values = matmul_values(x=tensor("x", [1, 2, 3]))

    with pytest.raises(InvalidOperatorInstanceError, match="rank-2"):
        matmul_op().validate(values)


def test_fused_matmul_add_rejects_inner_dim_mismatch() -> None:
    values = matmul_values(w=tensor("w", [5, 4]))

    with pytest.raises(InvalidOperatorInstanceError, match="requires lhs"):
        matmul_op().validate(values)


def test_fused_matmul_add_rejects_bias_shape_mismatch() -> None:
    values = matmul_values(bias=tensor("bias", [2, 5]))

    with pytest.raises(InvalidOperatorInstanceError, match="match matmul shape"):
        matmul_op().validate(values)


def test_fused_matmul_add_hls_hooks() -> None:
    op = matmul_op()

    context = op.hls_context(matmul_values())

    assert op.hls_template_path() == "hls/operators/fused_mat_mul_add.cpp.tpl"
    assert context["cpp_dtype"] == "float"
    assert context["m_dim"] == 2
    assert context["k_dim"] == 3
    assert context["n_dim"] == 4
    assert context["output_0_size"] == 8


def test_fused_matmul_add_activation_requires_supported_activation() -> None:
    op = matmul_op({"activation": "Softmax"}, cls=FusedMatMulAddActivation)

    with pytest.raises(InvalidOperatorInstanceError, match="supported activation"):
        op.validate(matmul_values())


@pytest.mark.parametrize(
    ("activation", "fragment"),
    [
        ("ReLU", "acc > (float)0"),
        ("GELU", "std::tanh"),
    ],
)
def test_fused_matmul_add_activation_hls_hooks(activation: str, fragment: str) -> None:
    op = matmul_op({"activation": activation}, cls=FusedMatMulAddActivation)

    context = op.hls_context(matmul_values())

    assert op.hls_template_path() == (
        "hls/operators/fused_mat_mul_add_activation.cpp.tpl"
    )
    assert context["activation"] == activation
    assert fragment in str(context["activation_expression"])


def test_activation_expression_rejects_unknown_activation() -> None:
    op = matmul_op({"activation": "Swish"}, cls=FusedMatMulAddActivation)

    with pytest.raises(InvalidOperatorInstanceError, match="unsupported fused"):
        op.hls_context(matmul_values())


# ---------------------------------------------------------------------------
# FusedConv1DAdd / FusedConv1DAddActivation operator contract
# ---------------------------------------------------------------------------


def test_fused_conv1d_add_rejects_wrong_input_count() -> None:
    op = FusedConv1DAdd("fused", inputs=["x", "w"], outputs=["out"])
    with pytest.raises(InvalidOperatorInstanceError, match="expects 3 inputs"):
        op.validate({})


def test_fused_conv1d_add_rejects_wrong_output_count() -> None:
    op = FusedConv1DAdd(
        "fused",
        inputs=["x", "w", "bias"],
        outputs=["out", "extra"],
    )
    with pytest.raises(InvalidOperatorInstanceError, match="expects 1 output"):
        op.validate({})


def test_fused_conv1d_add_rejects_non_tensor_bias() -> None:
    values = conv_values(bias=scalar("bias"))

    with pytest.raises(InvalidOperatorInstanceError, match="bias to be a tensor"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_dtype_mismatch() -> None:
    values = conv_values(w=tensor("w", [3, 2, 3], dtype="float64"))

    with pytest.raises(InvalidOperatorInstanceError, match="share dtype"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_bias_shape_mismatch() -> None:
    values = conv_values(bias=tensor("bias", [1, 3, 5]))

    with pytest.raises(InvalidOperatorInstanceError, match="match Conv1D shape"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_non_rank3_input() -> None:
    values = conv_values(x=tensor("x", [2, 8]))

    with pytest.raises(InvalidOperatorInstanceError, match="rank-3"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_channel_mismatch() -> None:
    values = conv_values(w=tensor("w", [3, 4, 3]))

    with pytest.raises(InvalidOperatorInstanceError, match="input channels"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_nonpositive_stride() -> None:
    with pytest.raises(InvalidOperatorInstanceError, match="positive stride"):
        conv_op({"stride": 0}).validate(conv_values())


def test_fused_conv1d_add_rejects_invalid_geometry() -> None:
    values = conv_values(x=tensor("x", [1, 2, 2]), w=tensor("w", [3, 2, 5]))

    with pytest.raises(InvalidOperatorInstanceError, match="invalid kernel"):
        conv_op().validate(values)


def test_fused_conv1d_add_rejects_non_integer_stride() -> None:
    with pytest.raises(InvalidOperatorInstanceError, match="stride must be an int"):
        conv_op({"stride": "two"}).validate(conv_values())


def test_fused_conv1d_add_hls_hooks() -> None:
    op = conv_op()

    context = op.hls_context(conv_values())

    assert op.hls_template_path() == "hls/operators/fused_conv1_d_add.cpp.tpl"
    assert context["cpp_dtype"] == "float"
    assert context["batch"] == 1
    assert context["in_channels"] == 2
    assert context["input_length"] == 8
    assert context["out_channels"] == 3
    assert context["kernel_width"] == 3
    assert context["output_length"] == 6
    assert context["stride"] == 1
    assert context["padding"] == 0
    assert context["dilation"] == 1


def test_fused_conv1d_add_activation_requires_supported_activation() -> None:
    op = conv_op({"activation": "Softmax"}, cls=FusedConv1DAddActivation)

    with pytest.raises(InvalidOperatorInstanceError, match="supported activation"):
        op.validate(conv_values())


def test_fused_conv1d_add_activation_hls_hooks() -> None:
    op = conv_op({"activation": "Tanh"}, cls=FusedConv1DAddActivation)

    context = op.hls_context(conv_values())

    assert op.hls_template_path() == (
        "hls/operators/fused_conv1_d_add_activation.cpp.tpl"
    )
    assert context["activation"] == "Tanh"
    assert context["activation_expression"] == "std::tanh(acc)"


# ---------------------------------------------------------------------------
# FusedScaleAdd / FusedScaleAddActivation operator contract
# ---------------------------------------------------------------------------


def test_fused_scale_add_rejects_wrong_input_count() -> None:
    op = FusedScaleAdd("fused", inputs=["x", "scale"], outputs=["out"])
    with pytest.raises(InvalidOperatorInstanceError, match="expects 3 inputs"):
        op.validate({})


def test_fused_scale_add_rejects_wrong_output_count() -> None:
    op = FusedScaleAdd(
        "fused",
        inputs=["x", "scale", "bias"],
        outputs=["out", "extra"],
    )
    with pytest.raises(InvalidOperatorInstanceError, match="expects 1 output"):
        op.validate({})


def test_fused_scale_add_rejects_non_tensor_input() -> None:
    values = scale_values(x=scalar("x"))

    with pytest.raises(InvalidOperatorInstanceError, match="input to be a tensor"):
        scale_op().validate(values)


def test_fused_scale_add_rejects_non_tensor_output() -> None:
    values = scale_values(out=scalar("out"))

    with pytest.raises(InvalidOperatorInstanceError, match="output to be a tensor"):
        scale_op().validate(values)


def test_fused_scale_add_rejects_input_output_shape_mismatch() -> None:
    values = scale_values(out=tensor("out", [4, 2]))

    with pytest.raises(InvalidOperatorInstanceError, match="shapes to match"):
        scale_op().validate(values)


def test_fused_scale_add_rejects_dtype_mismatch() -> None:
    values = scale_values(scale=tensor("scale", [2, 4], dtype="float64"))

    with pytest.raises(InvalidOperatorInstanceError, match="share dtype"):
        scale_op().validate(values)


def test_fused_scale_add_rejects_wrong_shaped_scale() -> None:
    values = scale_values(scale=tensor("scale", [4]))

    with pytest.raises(InvalidOperatorInstanceError, match="scalar or output-shaped"):
        scale_op().validate(values)


def test_fused_scale_add_accepts_scalar_parameters() -> None:
    op = scale_op()
    values = scale_values(scale=scalar("scale"), bias=scalar("bias"))

    op.validate(values)
    context = op.hls_context(values)

    assert op.hls_template_path() == "hls/operators/fused_scale_add.cpp.tpl"
    assert context["output_0_size"] == 8
    assert context["has_scalar_scale"] == "true"
    assert context["has_scalar_bias"] == "true"


def test_fused_scale_add_activation_requires_supported_activation() -> None:
    op = scale_op({"activation": "Softmax"}, cls=FusedScaleAddActivation)

    with pytest.raises(InvalidOperatorInstanceError, match="supported activation"):
        op.validate(scale_values())


def test_fused_scale_add_activation_hls_hooks() -> None:
    op = scale_op({"activation": "Sigmoid"}, cls=FusedScaleAddActivation)

    context = op.hls_context(scale_values())

    assert op.hls_template_path() == (
        "hls/operators/fused_scale_add_activation.cpp.tpl"
    )
    assert context["activation"] == "Sigmoid"
    assert "std::exp(-acc)" in str(context["activation_expression"])


def test_fused_operator_cost_estimates() -> None:
    matmul_cost = matmul_op().estimate_fpga_cost(matmul_values())
    matmul_act = matmul_op(
        {"activation": "Tanh"},
        cls=FusedMatMulAddActivation,
    ).estimate_fpga_cost(matmul_values())
    conv_cost = conv_op().estimate_fpga_cost(conv_values())
    conv_act = conv_op(
        {"activation": "ReLU"},
        cls=FusedConv1DAddActivation,
    ).estimate_fpga_cost(conv_values())
    scale_cost = scale_op().estimate_fpga_cost(scale_values())
    scale_act = scale_op(
        {"activation": "Sigmoid"},
        cls=FusedScaleAddActivation,
    ).estimate_fpga_cost(scale_values())

    assert matmul_cost.metadata["heuristic"] == "fused_matmul_add"
    assert matmul_act.latency_cycles > matmul_cost.latency_cycles
    assert conv_cost.metadata["heuristic"] == "fused_conv1d_add"
    assert conv_act.latency_cycles == conv_cost.latency_cycles + 1
    assert scale_cost.metadata["heuristic"] == "fused_scale_add"
    assert scale_act.latency_cycles > scale_cost.latency_cycles


# ---------------------------------------------------------------------------
# Fusion pass guard branches
# ---------------------------------------------------------------------------


def matmul_process(
    *,
    tap: bool = False,
    post_consumer: bool = False,
    parameter_bias: bool = True,
    shared_add_output: bool = False,
) -> Process:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4], layout="parameter"),
        "bias": tensor(
            "bias",
            [2, 4],
            quant={"role": "parameter"} if parameter_bias else None,
        ),
        "z": tensor("z", [2, 4]),
        "biased": tensor("biased", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    ops: dict[str, Operator] = {
        "matmul": MatMul("matmul", inputs=["x", "w"], outputs=["z"]),
        "add": Add("add", inputs=["z", "bias"], outputs=["biased"]),
    }
    graph_inputs = ["x", "w", "bias"]
    graph_outputs = ["out"]
    if post_consumer:
        values["y"] = tensor("y", [2, 4])
        ops["post"] = Mul("post", inputs=["biased", "y"], outputs=["out"])
        graph_inputs.append("y")
    else:
        ops["relu"] = ReLU("relu", inputs=["biased"], outputs=["out"])
    if tap:
        values["tap"] = tensor("tap", [2, 4])
        ops["tap"] = ReLU("tap", inputs=["z"], outputs=["tap"])
        graph_outputs.append("tap")
    if shared_add_output:
        values["skip"] = tensor("skip", [2, 4])
        ops["skip"] = Add("skip", inputs=["biased", "bias"], outputs=["skip"])
        graph_outputs.append("skip")
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
    )
    return Process(
        process_id="matmul_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def conv_process(*, tap: bool = False, parameter_bias: bool = True) -> Process:
    values = {
        "x": tensor("x", [1, 2, 8]),
        "w": tensor("w", [3, 2, 3], layout="parameter"),
        "bias": tensor(
            "bias",
            [1, 3, 6],
            quant={"role": "parameter"} if parameter_bias else None,
        ),
        "conv_out": tensor("conv_out", [1, 3, 6]),
        "biased": tensor("biased", [1, 3, 6]),
        "out": tensor("out", [1, 3, 6]),
    }
    ops: dict[str, Operator] = {
        "conv": Conv1D("conv", inputs=["x", "w"], outputs=["conv_out"]),
        "add": Add("add", inputs=["conv_out", "bias"], outputs=["biased"]),
        "relu": ReLU("relu", inputs=["biased"], outputs=["out"]),
    }
    graph_outputs = ["out"]
    if tap:
        values["tap"] = tensor("tap", [1, 3, 6])
        ops["tap"] = ReLU("tap", inputs=["conv_out"], outputs=["tap"])
        graph_outputs.append("tap")
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=["x", "w", "bias"],
        graph_outputs=graph_outputs,
    )
    return Process(
        process_id="conv_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def scale_process(
    *,
    parameter_bias: bool = True,
    parameter_scale: bool = True,
    tap: bool = False,
) -> Process:
    values = {
        "x": tensor("x", [2, 4]),
        "scale": tensor(
            "scale",
            [2, 4],
            quant={"role": "parameter"} if parameter_scale else None,
        ),
        "bias": tensor("bias", [2, 4], layout="parameter" if parameter_bias else None),
        "scaled": tensor("scaled", [2, 4]),
        "biased": tensor("biased", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    ops: dict[str, Operator] = {
        "mul": Mul("mul", inputs=["x", "scale"], outputs=["scaled"]),
        "add": Add("add", inputs=["scaled", "bias"], outputs=["biased"]),
        "relu": ReLU("relu", inputs=["biased"], outputs=["out"]),
    }
    graph_outputs = ["out"]
    if tap:
        values["tap"] = tensor("tap", [2, 4])
        ops["tap"] = ReLU("tap", inputs=["scaled"], outputs=["tap"])
        graph_outputs.append("tap")
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=["x", "scale", "bias"],
        graph_outputs=graph_outputs,
    )
    return Process(
        process_id="scale_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def test_matmul_activation_chain_fuses() -> None:
    result = optimize_temporal_process(
        matmul_process(),
        passes=(fuse_parameterized_matmul_add,),
    )
    ops = result.optimized.kernels["kernel"].graph.ops
    payload = result.to_dict()

    assert result.changed is True
    assert set(ops) == {"matmul_add_relu_fused"}
    assert ops["matmul_add_relu_fused"].op_type == "FusedMatMulAddActivation"
    delta = payload["graph_only_delta"]
    assert isinstance(delta, dict)
    assert delta["estimated_latency_cycles"] < 0


def test_conv_activation_chain_fuses() -> None:
    result = optimize_temporal_process(
        conv_process(),
        passes=(fuse_parameterized_conv1d_add,),
    )
    ops = result.optimized.kernels["kernel"].graph.ops

    assert result.changed is True
    assert set(ops) == {"conv_add_relu_fused"}
    assert ops["conv_add_relu_fused"].op_type == "FusedConv1DAddActivation"


def test_scale_activation_chain_fuses() -> None:
    result = optimize_temporal_process(
        scale_process(),
        passes=(fuse_parameterized_scale_add,),
    )
    ops = result.optimized.kernels["kernel"].graph.ops

    assert result.changed is True
    assert set(ops) == {"mul_add_relu_fused"}
    assert ops["mul_add_relu_fused"].op_type == "FusedScaleAddActivation"


def test_passes_skip_add_without_matching_producer() -> None:
    """Each fusion pass declines an Add fed by a different producer type."""

    conv_on_matmul = optimize_temporal_process(
        matmul_process(),
        passes=(fuse_parameterized_conv1d_add,),
    )
    scale_on_matmul = optimize_temporal_process(
        matmul_process(),
        passes=(fuse_parameterized_scale_add,),
    )
    matmul_on_scale = optimize_temporal_process(
        scale_process(),
        passes=(fuse_parameterized_matmul_add,),
    )

    assert conv_on_matmul.changed is False
    assert scale_on_matmul.changed is False
    assert matmul_on_scale.changed is False


def test_matmul_pass_declines_shared_intermediate() -> None:
    result = optimize_temporal_process(
        matmul_process(tap=True),
        passes=(fuse_parameterized_matmul_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "matmul",
        "add",
        "relu",
        "tap",
    }


def test_conv_pass_declines_shared_intermediate() -> None:
    result = optimize_temporal_process(
        conv_process(tap=True),
        passes=(fuse_parameterized_conv1d_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "conv",
        "add",
        "relu",
        "tap",
    }


def test_scale_pass_declines_shared_intermediate() -> None:
    result = optimize_temporal_process(
        scale_process(tap=True),
        passes=(fuse_parameterized_scale_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "mul",
        "add",
        "relu",
        "tap",
    }


def test_scale_pass_declines_runtime_bias() -> None:
    result = optimize_temporal_process(
        scale_process(parameter_bias=False),
        passes=(fuse_parameterized_scale_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "mul",
        "add",
        "relu",
    }


def test_scale_pass_declines_runtime_scale() -> None:
    result = optimize_temporal_process(
        scale_process(parameter_scale=False),
        passes=(fuse_parameterized_scale_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "mul",
        "add",
        "relu",
    }


def test_matmul_pass_declines_runtime_bias() -> None:
    result = optimize_temporal_process(
        matmul_process(parameter_bias=False),
        passes=(fuse_parameterized_matmul_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "matmul",
        "add",
        "relu",
    }


def test_conv_pass_declines_runtime_bias() -> None:
    result = optimize_temporal_process(
        conv_process(parameter_bias=False),
        passes=(fuse_parameterized_conv1d_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "conv",
        "add",
        "relu",
    }


def test_matmul_pass_keeps_activation_when_add_output_shared() -> None:
    result = optimize_temporal_process(
        matmul_process(shared_add_output=True),
        passes=(fuse_parameterized_matmul_add,),
    )
    ops = result.optimized.kernels["kernel"].graph.ops

    assert result.changed is True
    assert set(ops) == {"matmul_add_fused", "relu", "skip"}
    assert ops["matmul_add_fused"].op_type == "FusedMatMulAdd"


def test_matmul_pass_fuses_without_activation_for_non_activation_consumer() -> None:
    result = optimize_temporal_process(
        matmul_process(post_consumer=True),
        passes=(fuse_parameterized_matmul_add,),
    )
    graph = result.optimized.kernels["kernel"].graph

    assert result.changed is True
    assert set(graph.ops) == {"matmul_add_fused", "post"}
    assert graph.ops["matmul_add_fused"].op_type == "FusedMatMulAdd"
    assert graph.ops["matmul_add_fused"].outputs == ["biased"]
    assert "biased" in graph.values


def test_buffer_sharing_annotates_compatible_groups() -> None:
    process = Process(
        process_id="buffer_demo",
        buffers={
            "a": BufferSpec("a", dtype="float32", shape=(4,), depth=8),
            "b": BufferSpec("b", dtype="float32", shape=(4,), depth=8),
            "solo": BufferSpec("solo", dtype="float32", shape=(4,), depth=16),
        },
    )

    result = optimize_temporal_process(
        process,
        passes=(share_compatible_temporal_buffers,),
    )
    buffers = result.optimized.buffers

    assert result.changed is True
    assert buffers["a"].metadata["physical_buffer_id"] == "a"
    assert buffers["b"].metadata["physical_buffer_id"] == "a"
    assert buffers["a"].metadata["shared_buffer_group"] == ["a", "b"]
    assert "physical_buffer_id" not in buffers["solo"].metadata


# ---------------------------------------------------------------------------
# validate_temporal_rewrite legality guards
# ---------------------------------------------------------------------------


def guard_process(
    *, with_edge0: bool = False, with_edge_delta: bool = False
) -> Process:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4], layout="parameter"),
        "unused_param": tensor("unused_param", [1], layout="parameter"),
        "out": tensor("out", [2, 4]),
    }
    graph = Graph(
        values=values,
        ops={"matmul": MatMul("matmul", inputs=["x", "w"], outputs=["out"])},
        graph_inputs=["x", "w", "unused_param"],
        graph_outputs=["out"],
    )
    process = Process(
        process_id="guard_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(4,),
            )
        },
        buffers={"window": BufferSpec("window", dtype="float32", shape=(4,), depth=8)},
    )
    if with_edge0:
        process.edge0 = [Edge0("kernel", "hidden")]
    if with_edge_delta:
        process.edge_delta = [EdgeDelta("kernel", "hidden", lag_cycles=1)]
    return process


def test_rewrite_rejects_process_id_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.process_id = "renamed"

    with pytest.raises(TemporalOptimizationError, match="process_id"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_clock_identifier_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.clocks["aux"] = Clock("aux")

    with pytest.raises(TemporalOptimizationError, match="clock identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_state_identifier_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.states.pop("hidden")

    with pytest.raises(TemporalOptimizationError, match="state identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_buffer_identifier_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.buffers.pop("window")

    with pytest.raises(TemporalOptimizationError, match="buffer identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_kernel_identifier_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.kernels["extra"] = Kernel(
        "extra",
        graph=deepcopy(original.kernels["kernel"].graph),
    )

    with pytest.raises(TemporalOptimizationError, match="kernel identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_same_timestep_edge_changes() -> None:
    original = guard_process(with_edge0=True)
    optimized = deepcopy(original)
    optimized.edge0 = []

    with pytest.raises(TemporalOptimizationError, match="same-timestep edges"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_delayed_edge_changes() -> None:
    original = guard_process(with_edge_delta=True)
    optimized = deepcopy(original)
    optimized.edge_delta = []

    with pytest.raises(TemporalOptimizationError, match="delayed temporal edges"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_graph_output_changes() -> None:
    original = matmul_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.graph_outputs = ["biased"]

    with pytest.raises(TemporalOptimizationError, match="graph outputs"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_parameter_removal() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    graph = optimized.kernels["kernel"].graph
    graph.values.pop("unused_param")
    graph.graph_inputs.remove("unused_param")

    with pytest.raises(TemporalOptimizationError, match="parameter identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_parameter_metadata_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.values["unused_param"].axes = ["changed"]

    with pytest.raises(TemporalOptimizationError, match="parameter dtype"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_surviving_value_metadata_changes() -> None:
    original = guard_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.values["out"].quant = {"bit_width": 8}

    with pytest.raises(TemporalOptimizationError, match="value dtype"):
        validate_temporal_rewrite(original, optimized)


def test_delta_helper_handles_non_numeric_summaries() -> None:
    assert _delta(3, 5) == 2
    assert _delta("n/a", 5) is None
