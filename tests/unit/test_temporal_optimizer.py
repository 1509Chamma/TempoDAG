from copy import deepcopy
from typing import cast

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    EdgeDelta,
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
from tempo_dag.ops.builtins import Add, Conv1D, MatMul, Mul, ReLU


def tensor(
    value_id: str,
    shape: list[int],
    *,
    layout: str | None = None,
    quant: dict[str, float | int | str | None] | None = None,
) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"axis_{idx}" for idx in range(len(shape))],
        layout=layout,
        quant=quant,
    )


def optimizer_process() -> Process:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4], layout="parameter"),
        "bias": tensor("bias", [2, 4], quant={"role": "parameter"}),
        "unused_param": tensor("unused_param", [1], layout="parameter"),
        "z": tensor("z", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    graph = Graph(
        values=values,
        ops={
            "matmul": MatMul("matmul", inputs=["x", "w"], outputs=["z"]),
            "add": Add("add", inputs=["z", "bias"], outputs=["out"]),
        },
        graph_inputs=["x", "w", "bias", "unused_param"],
        graph_outputs=["out"],
    )
    return Process(
        process_id="optimizer_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(4,),
            )
        },
        edge_delta=[EdgeDelta("kernel", "hidden", lag_cycles=1, value_id="h_next")],
    )


def activation_process(*, extra_consumer: bool = False) -> Process:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4], layout="parameter"),
        "bias": tensor("bias", [2, 4], quant={"role": "parameter"}),
        "z": tensor("z", [2, 4]),
        "biased": tensor("biased", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    ops: dict[str, Operator] = {
        "matmul": MatMul("matmul", inputs=["x", "w"], outputs=["z"]),
        "add": Add("add", inputs=["z", "bias"], outputs=["biased"]),
        "relu": ReLU("relu", inputs=["biased"], outputs=["out"]),
    }
    graph_outputs = ["out"]
    if extra_consumer:
        values["skip"] = tensor("skip", [2, 4])
        ops["skip_add"] = Add("skip_add", inputs=["biased", "bias"], outputs=["skip"])
        graph_outputs.append("skip")
    graph = Graph(
        values=values,
        ops=ops,
        graph_inputs=["x", "w", "bias"],
        graph_outputs=graph_outputs,
    )
    return Process(
        process_id="activation_optimizer_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def conv_activation_process(*, parameter_bias: bool = True) -> Process:
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
    graph = Graph(
        values=values,
        ops={
            "conv": Conv1D("conv", inputs=["x", "w"], outputs=["conv_out"]),
            "add": Add("add", inputs=["conv_out", "bias"], outputs=["biased"]),
            "relu": ReLU("relu", inputs=["biased"], outputs=["out"]),
        },
        graph_inputs=["x", "w", "bias"],
        graph_outputs=["out"],
    )
    return Process(
        process_id="conv_optimizer_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def scale_activation_process(*, parameter_scale: bool = True) -> Process:
    values = {
        "x": tensor("x", [2, 4]),
        "scale": tensor(
            "scale",
            [2, 4],
            quant={"role": "parameter"} if parameter_scale else None,
        ),
        "bias": tensor("bias", [2, 4], layout="parameter"),
        "scaled": tensor("scaled", [2, 4]),
        "biased": tensor("biased", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    graph = Graph(
        values=values,
        ops={
            "scale": Mul("scale", inputs=["x", "scale"], outputs=["scaled"]),
            "add": Add("add", inputs=["scaled", "bias"], outputs=["biased"]),
            "relu": ReLU("relu", inputs=["biased"], outputs=["out"]),
        },
        graph_inputs=["x", "scale", "bias"],
        graph_outputs=["out"],
    )
    return Process(
        process_id="scale_optimizer_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )


def buffer_sharing_process() -> Process:
    return Process(
        process_id="buffer_sharing_demo",
        buffers={
            "returns_window": BufferSpec(
                "returns_window",
                dtype="float32",
                shape=(4,),
                depth=16,
                axes=("feature",),
            ),
            "vol_window": BufferSpec(
                "vol_window",
                dtype="float32",
                shape=(4,),
                depth=16,
                axes=("feature",),
            ),
            "long_window": BufferSpec(
                "long_window",
                dtype="float32",
                shape=(4,),
                depth=32,
                axes=("feature",),
            ),
        },
    )


def test_identity_optimizer_returns_before_after_reports() -> None:
    process = optimizer_process()

    result = optimize_temporal_process(process)
    payload = result.to_dict()

    assert result.changed is False
    assert result.optimized.to_dict() == process.to_dict()
    assert payload["process_id"] == "optimizer_demo"
    graph_only_delta = cast(dict[str, object], payload["graph_only_delta"])
    assert graph_only_delta["estimated_latency_cycles"] == 0
    assert (
        result.baseline_report_before.to_dict()
        == result.baseline_report_after.to_dict()
    )


def test_optimizer_records_pass_changes() -> None:
    def add_metadata(process: Process) -> Process:
        process.metadata["optimized"] = True
        return process

    result = optimize_temporal_process(optimizer_process(), passes=(add_metadata,))

    assert result.changed is True
    assert result.rewrites[0].pass_name == "add_metadata"
    assert result.optimized.metadata["optimized"] is True


def test_optimizer_fuses_parameterized_matmul_add_chain() -> None:
    process = optimizer_process()

    result = optimize_temporal_process(
        process,
        passes=(fuse_parameterized_matmul_add,),
    )
    fused_ops = result.optimized.kernels["kernel"].graph.ops
    fused_op = next(iter(fused_ops.values()))
    before = result.baseline_report_before.summary
    after = result.baseline_report_after.summary
    graph_only_delta = cast(dict[str, object], result.to_dict()["graph_only_delta"])

    assert result.changed is True
    assert set(fused_ops) == {"matmul_add_fused"}
    assert fused_op.op_type == "FusedMatMulAdd"
    assert fused_op.inputs == ["x", "w", "bias"]
    assert fused_op.outputs == ["out"]
    assert fused_op.attrs["fused_ops"] == ["matmul", "add"]
    assert "z" not in result.optimized.kernels["kernel"].graph.values
    assert result.optimized.kernels["kernel"].graph.values["w"].layout == "parameter"
    assert result.optimized.kernels["kernel"].graph.values["bias"].quant == {
        "role": "parameter"
    }
    assert cast(int, after["estimated_latency_cycles"]) < cast(
        int,
        before["estimated_latency_cycles"],
    )
    assert cast(int, graph_only_delta["estimated_latency_cycles"]) < 0
    assert cast(int, graph_only_delta["traffic_elements_per_timestep"]) < 0


def test_optimizer_does_not_fuse_runtime_add_inputs() -> None:
    process = optimizer_process()
    process.kernels["kernel"].graph.values["bias"].quant = None

    result = optimize_temporal_process(
        process,
        passes=(fuse_parameterized_matmul_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {"matmul", "add"}


def test_optimizer_fuses_matmul_add_activation_chain() -> None:
    result = optimize_temporal_process(
        activation_process(),
        passes=(fuse_parameterized_matmul_add,),
    )
    fused_ops = result.optimized.kernels["kernel"].graph.ops
    fused_op = next(iter(fused_ops.values()))
    graph_only_delta = cast(dict[str, object], result.to_dict()["graph_only_delta"])

    assert result.changed is True
    assert set(fused_ops) == {"matmul_add_relu_fused"}
    assert fused_op.op_type == "FusedMatMulAddActivation"
    assert fused_op.inputs == ["x", "w", "bias"]
    assert fused_op.outputs == ["out"]
    assert fused_op.attrs["activation"] == "ReLU"
    assert fused_op.attrs["fused_ops"] == ["matmul", "add", "relu"]
    assert "z" not in result.optimized.kernels["kernel"].graph.values
    assert "biased" not in result.optimized.kernels["kernel"].graph.values
    assert cast(int, graph_only_delta["estimated_latency_cycles"]) < 0
    assert cast(int, graph_only_delta["traffic_elements_per_timestep"]) < 0


def test_optimizer_skips_activation_with_multiple_add_consumers() -> None:
    result = optimize_temporal_process(
        activation_process(extra_consumer=True),
        passes=(fuse_parameterized_matmul_add,),
    )

    assert result.changed is True
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "matmul_add_fused",
        "relu",
        "skip_add",
    }


def test_optimizer_fuses_conv1d_add_activation_chain() -> None:
    result = optimize_temporal_process(
        conv_activation_process(),
        passes=(fuse_parameterized_conv1d_add,),
    )
    fused_ops = result.optimized.kernels["kernel"].graph.ops
    fused_op = next(iter(fused_ops.values()))
    graph_only_delta = cast(dict[str, object], result.to_dict()["graph_only_delta"])

    assert result.changed is True
    assert set(fused_ops) == {"conv_add_relu_fused"}
    assert fused_op.op_type == "FusedConv1DAddActivation"
    assert fused_op.inputs == ["x", "w", "bias"]
    assert fused_op.outputs == ["out"]
    assert fused_op.attrs["activation"] == "ReLU"
    assert fused_op.attrs["fused_ops"] == ["conv", "add", "relu"]
    assert "conv_out" not in result.optimized.kernels["kernel"].graph.values
    assert "biased" not in result.optimized.kernels["kernel"].graph.values
    assert result.optimized.kernels["kernel"].graph.values["w"].layout == "parameter"
    assert cast(int, graph_only_delta["estimated_latency_cycles"]) < 0
    assert cast(int, graph_only_delta["traffic_elements_per_timestep"]) < 0


def test_optimizer_does_not_fuse_conv1d_runtime_bias() -> None:
    result = optimize_temporal_process(
        conv_activation_process(parameter_bias=False),
        passes=(fuse_parameterized_conv1d_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "conv",
        "add",
        "relu",
    }


def test_optimizer_fuses_scale_add_activation_chain() -> None:
    result = optimize_temporal_process(
        scale_activation_process(),
        passes=(fuse_parameterized_scale_add,),
    )
    fused_ops = result.optimized.kernels["kernel"].graph.ops
    fused_op = next(iter(fused_ops.values()))
    graph_only_delta = cast(dict[str, object], result.to_dict()["graph_only_delta"])

    assert result.changed is True
    assert set(fused_ops) == {"scale_add_relu_fused"}
    assert fused_op.op_type == "FusedScaleAddActivation"
    assert fused_op.inputs == ["x", "scale", "bias"]
    assert fused_op.outputs == ["out"]
    assert fused_op.attrs["activation"] == "ReLU"
    assert fused_op.attrs["fused_ops"] == ["scale", "add", "relu"]
    assert "scaled" not in result.optimized.kernels["kernel"].graph.values
    assert "biased" not in result.optimized.kernels["kernel"].graph.values
    assert result.optimized.kernels["kernel"].graph.values["scale"].quant == {
        "role": "parameter"
    }
    assert result.optimized.kernels["kernel"].graph.values["bias"].layout == (
        "parameter"
    )
    assert cast(int, graph_only_delta["estimated_latency_cycles"]) < 0
    assert cast(int, graph_only_delta["traffic_elements_per_timestep"]) < 0


def test_optimizer_does_not_fuse_runtime_scale() -> None:
    result = optimize_temporal_process(
        scale_activation_process(parameter_scale=False),
        passes=(fuse_parameterized_scale_add,),
    )

    assert result.changed is False
    assert set(result.optimized.kernels["kernel"].graph.ops) == {
        "scale",
        "add",
        "relu",
    }


def test_optimizer_annotates_compatible_buffer_sharing() -> None:
    result = optimize_temporal_process(
        buffer_sharing_process(),
        passes=(share_compatible_temporal_buffers,),
    )
    buffers = result.optimized.buffers

    assert result.changed is True
    assert buffers["returns_window"].metadata["physical_buffer_id"] == (
        "returns_window"
    )
    assert buffers["vol_window"].metadata["physical_buffer_id"] == "returns_window"
    assert buffers["returns_window"].metadata["shared_buffer_group"] == [
        "returns_window",
        "vol_window",
    ]
    assert "physical_buffer_id" not in buffers["long_window"].metadata


def test_rewrite_rejects_process_identity_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    optimized.process_id = "other"

    with pytest.raises(TemporalOptimizationError, match="process_id"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_temporal_edge_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    optimized.edge_delta = []

    with pytest.raises(TemporalOptimizationError, match="delayed temporal edges"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_graph_output_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.graph_outputs = ["z"]

    with pytest.raises(TemporalOptimizationError, match="graph outputs"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_parameter_identity_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.values.pop("unused_param")
    optimized.kernels["kernel"].graph.graph_inputs.remove("unused_param")

    with pytest.raises(TemporalOptimizationError, match="parameter identifiers"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_parameter_metadata_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    optimized.kernels["kernel"].graph.values["unused_param"].axes = ["changed"]

    with pytest.raises(TemporalOptimizationError, match="parameter dtype"):
        validate_temporal_rewrite(original, optimized)


def test_rewrite_rejects_surviving_value_quantization_changes() -> None:
    original = optimizer_process()
    optimized = deepcopy(original)
    original.kernels["kernel"].graph.values["out"].quant = {
        "bit_width": 16,
        "integer_bits": 8,
        "fractional_bits": 8,
    }
    optimized.kernels["kernel"].graph.values["out"].quant = {
        "bit_width": 8,
        "integer_bits": 4,
        "fractional_bits": 4,
    }

    with pytest.raises(TemporalOptimizationError, match="value dtype"):
        validate_temporal_rewrite(original, optimized)
