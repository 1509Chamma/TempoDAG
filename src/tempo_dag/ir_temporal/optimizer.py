from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal.process import BufferSpec, Edge0, EdgeDelta, Kernel, Process
from tempo_dag.ir_temporal.report import (
    TemporalBaselineReport,
    derive_temporal_baseline_report,
)
from tempo_dag.ir_temporal.schedule import derive_temporal_schedule

TemporalRewritePass = Callable[[Process], Process]
STATELESS_CHAIN_FUSION_ATTR = "tempo_dag_fused_stateless_chain"
_SUPPORTED_FUSED_ACTIVATIONS = frozenset({"ReLU", "Tanh", "Sigmoid", "GELU"})


class TemporalOptimizationError(ValueError):
    """Raised when a temporal graph rewrite violates optimizer legality."""


class FusedMatMulAdd(Operator):
    """Logical fused MatMul plus parameter-bias Add for schedule optimization."""

    OP_TYPE = "FusedMatMulAdd"

    def validate(self, values: Mapping[str, Value]) -> None:
        if len(self.inputs) != 3:
            raise InvalidOperatorInstanceError("FusedMatMulAdd expects 3 inputs")
        if len(self.outputs) != 1:
            raise InvalidOperatorInstanceError("FusedMatMulAdd expects 1 output")
        lhs = _lookup_value(values, self.inputs[0], self.op_type)
        rhs = _lookup_value(values, self.inputs[1], self.op_type)
        bias = _lookup_value(values, self.inputs[2], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        for label, value in (
            ("lhs", lhs),
            ("rhs", rhs),
            ("bias", bias),
            ("output", output),
        ):
            if value.vtype is not ValueType.TENSOR:
                raise InvalidOperatorInstanceError(
                    f"FusedMatMulAdd expects {label} to be a tensor"
                )
        if (
            lhs.dtype != rhs.dtype
            or lhs.dtype != bias.dtype
            or lhs.dtype != output.dtype
        ):
            raise InvalidOperatorInstanceError(
                "FusedMatMulAdd requires all values to share dtype"
            )
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise InvalidOperatorInstanceError(
                "FusedMatMulAdd currently supports rank-2 matmul inputs"
            )
        if lhs.shape[1] != rhs.shape[0]:
            raise InvalidOperatorInstanceError(
                "FusedMatMulAdd requires lhs.shape[1] == rhs.shape[0]"
            )
        expected_shape = [lhs.shape[0], rhs.shape[1]]
        if bias.shape != expected_shape or output.shape != expected_shape:
            raise InvalidOperatorInstanceError(
                "FusedMatMulAdd requires bias and output to match matmul shape"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        lhs = _lookup_value(values, self.inputs[0], self.op_type)
        rhs = _lookup_value(values, self.inputs[1], self.op_type)
        m_dim, k_dim = lhs.shape
        _, n_dim = rhs.shape
        work = m_dim * n_dim * k_dim
        output_work = m_dim * n_dim
        return FPGACost(
            latency_cycles=max(1, work + 1),
            initiation_interval=1,
            dsp=max(1, min(work, k_dim)),
            lut=max(1, output_work),
            ff=max(1, output_work),
            metadata={"heuristic": "fused_matmul_add"},
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_mat_mul_add.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        lhs = _lookup_value(values, self.inputs[0], self.op_type)
        rhs = _lookup_value(values, self.inputs[1], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
            "cpp_dtype": {
                "float32": "float",
                "float64": "double",
            }.get(output.dtype, output.dtype),
            "m_dim": lhs.shape[0],
            "k_dim": lhs.shape[1],
            "n_dim": rhs.shape[1],
            "output_0_size": output.shape[0] * output.shape[1],
        }


class FusedMatMulAddActivation(FusedMatMulAdd):
    """Logical fused MatMul plus parameter-bias Add plus activation."""

    OP_TYPE = "FusedMatMulAddActivation"

    def validate(self, values: Mapping[str, Value]) -> None:
        super().validate(values)
        activation = self.attrs.get("activation")
        if activation not in _SUPPORTED_FUSED_ACTIVATIONS:
            raise InvalidOperatorInstanceError(
                "FusedMatMulAddActivation requires a supported activation"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        lhs = _lookup_value(values, self.inputs[0], self.op_type)
        rhs = _lookup_value(values, self.inputs[1], self.op_type)
        m_dim, k_dim = lhs.shape
        _, n_dim = rhs.shape
        work = m_dim * n_dim * k_dim
        output_work = m_dim * n_dim
        activation = self.attrs["activation"]
        activation_cost = 1 if activation == "ReLU" else max(1, output_work // 2)
        return FPGACost(
            latency_cycles=max(1, work + 1 + activation_cost),
            initiation_interval=1,
            dsp=max(1, min(work, k_dim)),
            lut=max(1, output_work * 2),
            ff=max(1, output_work),
            metadata={
                "heuristic": "fused_matmul_add_activation",
                "activation": activation,
            },
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_mat_mul_add_activation.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        context["activation"] = self.attrs["activation"]
        context["activation_expression"] = _activation_expression(
            str(self.attrs["activation"]),
            "acc",
            str(context["cpp_dtype"]),
        )
        return context


class FusedConv1DAdd(Operator):
    """Logical fused Conv1D plus parameter-bias Add for schedule optimization."""

    OP_TYPE = "FusedConv1DAdd"

    def validate(self, values: Mapping[str, Value]) -> None:
        if len(self.inputs) != 3:
            raise InvalidOperatorInstanceError("FusedConv1DAdd expects 3 inputs")
        if len(self.outputs) != 1:
            raise InvalidOperatorInstanceError("FusedConv1DAdd expects 1 output")
        input_value = _lookup_value(values, self.inputs[0], self.op_type)
        weight = _lookup_value(values, self.inputs[1], self.op_type)
        bias = _lookup_value(values, self.inputs[2], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        for label, value in (
            ("input", input_value),
            ("weight", weight),
            ("bias", bias),
            ("output", output),
        ):
            if value.vtype is not ValueType.TENSOR:
                raise InvalidOperatorInstanceError(
                    f"FusedConv1DAdd expects {label} to be a tensor"
                )
        if (
            input_value.dtype != weight.dtype
            or input_value.dtype != bias.dtype
            or input_value.dtype != output.dtype
        ):
            raise InvalidOperatorInstanceError(
                "FusedConv1DAdd requires all values to share dtype"
            )
        expected_shape = _conv1d_output_shape(input_value, weight, self.attrs)
        if bias.shape != expected_shape or output.shape != expected_shape:
            raise InvalidOperatorInstanceError(
                "FusedConv1DAdd requires bias and output to match Conv1D shape"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = _lookup_value(values, self.inputs[0], self.op_type)
        weight = _lookup_value(values, self.inputs[1], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        batch = input_value.shape[0]
        out_channels, in_channels, kernel_width = weight.shape
        output_length = output.shape[2]
        work = batch * out_channels * output_length * in_channels * kernel_width
        output_work = batch * out_channels * output_length
        return FPGACost(
            latency_cycles=max(1, work + 1),
            initiation_interval=1,
            dsp=max(1, in_channels * kernel_width),
            bram=max(1, (out_channels * kernel_width + 31) // 32),
            lut=max(1, output_work),
            ff=max(1, output_work),
            metadata={"heuristic": "fused_conv1d_add"},
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_conv1_d_add.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        input_value = _lookup_value(values, self.inputs[0], self.op_type)
        weight = _lookup_value(values, self.inputs[1], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
            "cpp_dtype": {"float32": "float", "float64": "double"}.get(
                output.dtype,
                output.dtype,
            ),
            "batch": input_value.shape[0],
            "in_channels": input_value.shape[1],
            "input_length": input_value.shape[2],
            "out_channels": weight.shape[0],
            "kernel_width": weight.shape[2],
            "output_length": output.shape[2],
            "stride": self.attrs.get("stride", 1),
            "padding": self.attrs.get("padding", 0),
            "dilation": self.attrs.get("dilation", 1),
        }


class FusedConv1DAddActivation(FusedConv1DAdd):
    """Logical fused Conv1D plus parameter-bias Add plus activation."""

    OP_TYPE = "FusedConv1DAddActivation"

    def validate(self, values: Mapping[str, Value]) -> None:
        super().validate(values)
        activation = self.attrs.get("activation")
        if activation not in _SUPPORTED_FUSED_ACTIVATIONS:
            raise InvalidOperatorInstanceError(
                "FusedConv1DAddActivation requires a supported activation"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        base = super().estimate_fpga_cost(values)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        output_work = output.shape[0] * output.shape[1] * output.shape[2]
        activation = self.attrs["activation"]
        activation_cost = 1 if activation == "ReLU" else max(1, output_work // 2)
        return FPGACost(
            latency_cycles=base.latency_cycles + activation_cost,
            initiation_interval=base.initiation_interval,
            dsp=base.dsp,
            bram=base.bram,
            lut=base.lut + max(1, output_work),
            ff=base.ff,
            metadata={
                "heuristic": "fused_conv1d_add_activation",
                "activation": activation,
            },
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_conv1_d_add_activation.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        context["activation"] = self.attrs["activation"]
        context["activation_expression"] = _activation_expression(
            str(self.attrs["activation"]),
            "acc",
            str(context["cpp_dtype"]),
        )
        return context


class FusedScaleAdd(Operator):
    """Logical fused elementwise parameter scale plus bias Add."""

    OP_TYPE = "FusedScaleAdd"

    def validate(self, values: Mapping[str, Value]) -> None:
        if len(self.inputs) != 3:
            raise InvalidOperatorInstanceError("FusedScaleAdd expects 3 inputs")
        if len(self.outputs) != 1:
            raise InvalidOperatorInstanceError("FusedScaleAdd expects 1 output")
        input_value = _lookup_value(values, self.inputs[0], self.op_type)
        scale = _lookup_value(values, self.inputs[1], self.op_type)
        bias = _lookup_value(values, self.inputs[2], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        if input_value.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                "FusedScaleAdd expects input to be a tensor"
            )
        if output.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                "FusedScaleAdd expects output to be a tensor"
            )
        if input_value.shape != output.shape:
            raise InvalidOperatorInstanceError(
                "FusedScaleAdd requires input and output shapes to match"
            )
        if (
            input_value.dtype != scale.dtype
            or input_value.dtype != bias.dtype
            or input_value.dtype != output.dtype
        ):
            raise InvalidOperatorInstanceError(
                "FusedScaleAdd requires all values to share dtype"
            )
        for label, value in (("scale", scale), ("bias", bias)):
            if not _matches_tensor_or_scalar(value, output.shape):
                raise InvalidOperatorInstanceError(
                    f"FusedScaleAdd requires {label} to be scalar or output-shaped"
                )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        output = _lookup_value(values, self.outputs[0], self.op_type)
        output_work = _shape_product(output.shape)
        return FPGACost(
            latency_cycles=max(1, output_work + 1),
            initiation_interval=1,
            dsp=max(1, output_work),
            lut=max(1, output_work),
            ff=max(1, output_work),
            metadata={"heuristic": "fused_scale_add"},
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_scale_add.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        scale = _lookup_value(values, self.inputs[1], self.op_type)
        bias = _lookup_value(values, self.inputs[2], self.op_type)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
            "cpp_dtype": {"float32": "float", "float64": "double"}.get(
                output.dtype,
                output.dtype,
            ),
            "output_0_size": _shape_product(output.shape),
            "has_scalar_scale": str(scale.vtype is ValueType.SCALAR).lower(),
            "has_scalar_bias": str(bias.vtype is ValueType.SCALAR).lower(),
        }


class FusedScaleAddActivation(FusedScaleAdd):
    """Logical fused elementwise parameter scale plus bias plus activation."""

    OP_TYPE = "FusedScaleAddActivation"

    def validate(self, values: Mapping[str, Value]) -> None:
        super().validate(values)
        activation = self.attrs.get("activation")
        if activation not in _SUPPORTED_FUSED_ACTIVATIONS:
            raise InvalidOperatorInstanceError(
                "FusedScaleAddActivation requires a supported activation"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        base = super().estimate_fpga_cost(values)
        output = _lookup_value(values, self.outputs[0], self.op_type)
        output_work = _shape_product(output.shape)
        activation = self.attrs["activation"]
        activation_cost = 1 if activation == "ReLU" else max(1, output_work // 2)
        return FPGACost(
            latency_cycles=base.latency_cycles + activation_cost,
            initiation_interval=base.initiation_interval,
            dsp=base.dsp,
            bram=base.bram,
            lut=base.lut + max(1, output_work),
            ff=base.ff,
            metadata={
                "heuristic": "fused_scale_add_activation",
                "activation": activation,
            },
        )

    def hls_template_path(self) -> str:
        return "hls/operators/fused_scale_add_activation.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        context["activation"] = self.attrs["activation"]
        context["activation_expression"] = _activation_expression(
            str(self.attrs["activation"]),
            "acc",
            str(context["cpp_dtype"]),
        )
        return context


@dataclass(frozen=True)
class TemporalRewriteRecord:
    """One optimizer pass application."""

    pass_name: str
    changed: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "pass_name": self.pass_name,
            "changed": self.changed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TemporalOptimizationResult:
    """Optimized process plus before/after graph-only report data."""

    original: Process
    optimized: Process
    rewrites: tuple[TemporalRewriteRecord, ...]
    baseline_report_before: TemporalBaselineReport
    baseline_report_after: TemporalBaselineReport

    @property
    def changed(self) -> bool:
        return any(record.changed for record in self.rewrites)

    def to_dict(self) -> dict[str, object]:
        before_summary = self.baseline_report_before.summary
        after_summary = self.baseline_report_after.summary
        return {
            "process_id": self.optimized.process_id,
            "changed": self.changed,
            "rewrites": [record.to_dict() for record in self.rewrites],
            "before": self.baseline_report_before.to_dict(),
            "after": self.baseline_report_after.to_dict(),
            "graph_only_delta": {
                "estimated_latency_cycles": _delta(
                    before_summary["estimated_latency_cycles"],
                    after_summary["estimated_latency_cycles"],
                ),
                "estimated_initiation_interval": _delta(
                    before_summary["estimated_initiation_interval"],
                    after_summary["estimated_initiation_interval"],
                ),
                "traffic_elements_per_timestep": _delta(
                    self.baseline_report_before.traffic_summary[
                        "total_elements_per_timestep"
                    ],
                    self.baseline_report_after.traffic_summary[
                        "total_elements_per_timestep"
                    ],
                ),
            },
        }


def optimize_temporal_process(
    process: Process,
    passes: tuple[TemporalRewritePass, ...] = (),
) -> TemporalOptimizationResult:
    """Apply legal temporal graph rewrite passes and report graph-only deltas."""

    original = deepcopy(process)
    original.validate()
    current = deepcopy(process)
    records: list[TemporalRewriteRecord] = []

    for rewrite_pass in passes:
        before_payload = current.to_dict()
        candidate = rewrite_pass(deepcopy(current))
        validate_temporal_rewrite(current, candidate)
        changed = candidate.to_dict() != before_payload
        records.append(
            TemporalRewriteRecord(
                pass_name=getattr(
                    rewrite_pass,
                    "__name__",
                    rewrite_pass.__class__.__name__,
                ),
                changed=changed,
            )
        )
        current = candidate

    before_schedule = derive_temporal_schedule(original)
    after_schedule = derive_temporal_schedule(current)
    return TemporalOptimizationResult(
        original=original,
        optimized=current,
        rewrites=tuple(records),
        baseline_report_before=derive_temporal_baseline_report(
            original,
            before_schedule,
        ),
        baseline_report_after=derive_temporal_baseline_report(
            current,
            after_schedule,
        ),
    )


def fuse_parameterized_matmul_add(process: Process) -> Process:
    """Fuse safe `MatMul -> Add(parameter bias)[ -> Activation]` chains."""

    optimized = deepcopy(process)
    for kernel_id, kernel in list(optimized.kernels.items()):
        fused_kernel = _fuse_kernel_parameterized_matmul_add(kernel)
        optimized.kernels[kernel_id] = fused_kernel
    return optimized


def fuse_parameterized_conv1d_add(process: Process) -> Process:
    """Fuse safe `Conv1D -> Add(parameter bias)[ -> Activation]` chains."""

    optimized = deepcopy(process)
    for kernel_id, kernel in list(optimized.kernels.items()):
        fused_kernel = _fuse_kernel_parameterized_conv1d_add(kernel)
        optimized.kernels[kernel_id] = fused_kernel
    return optimized


def fuse_parameterized_scale_add(process: Process) -> Process:
    """Fuse safe `Mul(parameter scale) -> Add(parameter bias)[ -> Activation]`."""

    optimized = deepcopy(process)
    for kernel_id, kernel in list(optimized.kernels.items()):
        fused_kernel = _fuse_kernel_parameterized_scale_add(kernel)
        optimized.kernels[kernel_id] = fused_kernel
    return optimized


def share_compatible_temporal_buffers(process: Process) -> Process:
    """Annotate compatible temporal buffers that may share physical storage."""

    optimized = deepcopy(process)
    groups: dict[tuple[object, ...], list[str]] = {}
    for buffer_id, buffer in sorted(optimized.buffers.items()):
        groups.setdefault(_buffer_sharing_signature(buffer), []).append(buffer_id)

    for group in groups.values():
        if len(group) < 2:
            continue
        physical_buffer_id = group[0]
        for buffer_id in group:
            buffer = optimized.buffers[buffer_id]
            metadata = dict(buffer.metadata)
            metadata["physical_buffer_id"] = physical_buffer_id
            metadata["shared_buffer_group"] = list(group)
            optimized.buffers[buffer_id] = BufferSpec(
                buffer_id=buffer.buffer_id,
                dtype=buffer.dtype,
                shape=buffer.shape,
                depth=buffer.depth,
                axes=buffer.axes,
                clock_id=buffer.clock_id,
                metadata=metadata,
            )
    return optimized


def validate_temporal_rewrite(original: Process, optimized: Process) -> None:
    """Validate invariants every temporal graph optimizer pass must preserve."""

    original.validate()
    optimized.validate()
    if optimized.process_id != original.process_id:
        raise TemporalOptimizationError("rewrite must preserve process_id")
    if set(optimized.clocks) != set(original.clocks):
        raise TemporalOptimizationError("rewrite must preserve clock identifiers")
    if set(optimized.states) != set(original.states):
        raise TemporalOptimizationError("rewrite must preserve state identifiers")
    if set(optimized.buffers) != set(original.buffers):
        raise TemporalOptimizationError("rewrite must preserve buffer identifiers")
    if set(optimized.kernels) != set(original.kernels):
        raise TemporalOptimizationError("rewrite must preserve kernel identifiers")
    if _edge0_signature(optimized.edge0) != _edge0_signature(original.edge0):
        raise TemporalOptimizationError("rewrite must preserve same-timestep edges")
    if _edge_delta_signature(optimized.edge_delta) != _edge_delta_signature(
        original.edge_delta
    ):
        raise TemporalOptimizationError("rewrite must preserve delayed temporal edges")

    for kernel_id, original_kernel in original.kernels.items():
        optimized_kernel = optimized.kernels[kernel_id]
        if optimized_kernel.graph.graph_outputs != original_kernel.graph.graph_outputs:
            raise TemporalOptimizationError("rewrite must preserve graph outputs")
        _validate_parameter_identity(
            original_kernel.graph.values,
            optimized_kernel.graph.values,
        )
        _validate_preserved_value_metadata(
            original_kernel.graph.values,
            optimized_kernel.graph.values,
        )


def _fuse_kernel_parameterized_conv1d_add(kernel: Kernel) -> Kernel:
    graph = kernel.graph
    producers = _value_producers(graph)
    consumers = _value_consumers(graph)
    ops = dict(graph.ops)
    values = dict(graph.values)

    for add_id, add_op in sorted(graph.ops.items()):
        if (
            add_op.op_type != "Add"
            or len(add_op.inputs) != 2
            or len(add_op.outputs) != 1
        ):
            continue
        conv_input = None
        bias_input = None
        conv_id = None
        for input_id in add_op.inputs:
            producer_id = producers.get(input_id)
            producer = graph.ops.get(producer_id or "")
            if producer is not None and producer.op_type == "Conv1D":
                conv_input = input_id
                conv_id = producer_id
            else:
                bias_input = input_id
        if conv_input is None or bias_input is None or conv_id is None:
            continue
        if not _is_parameter_value(values[bias_input]):
            continue
        if consumers[conv_input] != [add_id]:
            continue

        conv_op = graph.ops[conv_id]
        add_output = add_op.outputs[0]
        activation_id = _single_supported_activation_consumer(
            graph,
            consumers,
            add_output,
        )
        activation_op = graph.ops[activation_id] if activation_id is not None else None
        fused_id_parts = [conv_id, add_id]
        output_id = add_output
        removed_intermediates = [conv_input]
        fused_cls: type[Operator] = FusedConv1DAdd
        attrs: dict[str, object] = {
            **dict(conv_op.attrs),
            STATELESS_CHAIN_FUSION_ATTR: True,
            "fused_ops": [conv_id, add_id],
            "removed_intermediate": conv_input,
            "removed_intermediates": list(removed_intermediates),
        }
        if activation_op is not None:
            fused_id_parts.append(activation_id or "")
            output_id = activation_op.outputs[0]
            removed_intermediates.append(add_output)
            fused_cls = FusedConv1DAddActivation
            attrs["activation"] = activation_op.op_type
            attrs["fused_ops"] = [conv_id, add_id, activation_id]
            attrs["removed_intermediates"] = list(removed_intermediates)

        fused_id = "_".join(fused_id_parts) + "_fused"
        ops[fused_id] = fused_cls(
            fused_id,
            inputs=[conv_op.inputs[0], conv_op.inputs[1], bias_input],
            outputs=[output_id],
            attrs=attrs,
        )
        del ops[conv_id]
        del ops[add_id]
        if activation_id is not None:
            del ops[activation_id]
        for intermediate in removed_intermediates:
            if (
                intermediate not in graph.graph_outputs
                and intermediate not in graph.graph_inputs
            ):
                values.pop(intermediate, None)
        break

    if ops == graph.ops and values == graph.values:
        return kernel
    return Kernel(
        kernel.kernel_id,
        graph=graph.__class__(
            values=values,
            ops=ops,
            graph_inputs=list(graph.graph_inputs),
            graph_outputs=list(graph.graph_outputs),
            states=dict(graph.states),
            registry=graph.registry,
        ),
        clock_id=kernel.clock_id,
    )


def _fuse_kernel_parameterized_scale_add(kernel: Kernel) -> Kernel:
    graph = kernel.graph
    producers = _value_producers(graph)
    consumers = _value_consumers(graph)
    ops = dict(graph.ops)
    values = dict(graph.values)

    for add_id, add_op in sorted(graph.ops.items()):
        if (
            add_op.op_type != "Add"
            or len(add_op.inputs) != 2
            or len(add_op.outputs) != 1
        ):
            continue
        mul_input = None
        bias_input = None
        mul_id = None
        for input_id in add_op.inputs:
            producer_id = producers.get(input_id)
            producer = graph.ops.get(producer_id or "")
            if producer is not None and producer.op_type == "Mul":
                mul_input = input_id
                mul_id = producer_id
            else:
                bias_input = input_id
        if mul_input is None or bias_input is None or mul_id is None:
            continue
        if not _is_parameter_value(values[bias_input]):
            continue
        if consumers[mul_input] != [add_id]:
            continue

        mul_op = graph.ops[mul_id]
        runtime_input = None
        scale_input = None
        for input_id in mul_op.inputs:
            if _is_parameter_value(values[input_id]):
                scale_input = input_id
            else:
                runtime_input = input_id
        if runtime_input is None or scale_input is None:
            continue
        if consumers[mul_input] != [add_id]:
            continue

        add_output = add_op.outputs[0]
        activation_id = _single_supported_activation_consumer(
            graph,
            consumers,
            add_output,
        )
        activation_op = graph.ops[activation_id] if activation_id is not None else None
        fused_id_parts = [mul_id, add_id]
        output_id = add_output
        removed_intermediates = [mul_input]
        fused_cls: type[Operator] = FusedScaleAdd
        attrs: dict[str, object] = {
            STATELESS_CHAIN_FUSION_ATTR: True,
            "fused_ops": [mul_id, add_id],
            "removed_intermediate": mul_input,
            "removed_intermediates": list(removed_intermediates),
        }
        if activation_op is not None:
            fused_id_parts.append(activation_id or "")
            output_id = activation_op.outputs[0]
            removed_intermediates.append(add_output)
            fused_cls = FusedScaleAddActivation
            attrs["activation"] = activation_op.op_type
            attrs["fused_ops"] = [mul_id, add_id, activation_id]
            attrs["removed_intermediates"] = list(removed_intermediates)

        fused_id = "_".join(fused_id_parts) + "_fused"
        ops[fused_id] = fused_cls(
            fused_id,
            inputs=[runtime_input, scale_input, bias_input],
            outputs=[output_id],
            attrs=attrs,
        )
        del ops[mul_id]
        del ops[add_id]
        if activation_id is not None:
            del ops[activation_id]
        for intermediate in removed_intermediates:
            if (
                intermediate not in graph.graph_outputs
                and intermediate not in graph.graph_inputs
            ):
                values.pop(intermediate, None)
        break

    if ops == graph.ops and values == graph.values:
        return kernel
    return Kernel(
        kernel.kernel_id,
        graph=graph.__class__(
            values=values,
            ops=ops,
            graph_inputs=list(graph.graph_inputs),
            graph_outputs=list(graph.graph_outputs),
            states=dict(graph.states),
            registry=graph.registry,
        ),
        clock_id=kernel.clock_id,
    )


def _fuse_kernel_parameterized_matmul_add(kernel: Kernel) -> Kernel:
    graph = kernel.graph
    producers = _value_producers(graph)
    consumers = _value_consumers(graph)
    ops = dict(graph.ops)
    values = dict(graph.values)

    for add_id, add_op in sorted(graph.ops.items()):
        if (
            add_op.op_type != "Add"
            or len(add_op.inputs) != 2
            or len(add_op.outputs) != 1
        ):
            continue
        matmul_input = None
        bias_input = None
        matmul_id = None
        for input_id in add_op.inputs:
            producer_id = producers.get(input_id)
            producer = graph.ops.get(producer_id or "")
            if producer is not None and producer.op_type == "MatMul":
                matmul_input = input_id
                matmul_id = producer_id
            else:
                bias_input = input_id
        if matmul_input is None or bias_input is None or matmul_id is None:
            continue
        if not _is_parameter_value(values[bias_input]):
            continue
        if consumers[matmul_input] != [add_id]:
            continue

        matmul_op = graph.ops[matmul_id]
        add_output = add_op.outputs[0]
        activation_id = _single_supported_activation_consumer(
            graph,
            consumers,
            add_output,
        )
        activation_op = graph.ops[activation_id] if activation_id is not None else None
        fused_id_parts = [matmul_id, add_id]
        output_id = add_output
        removed_intermediates = [matmul_input]
        fused_cls: type[Operator] = FusedMatMulAdd
        attrs: dict[str, object] = {
            STATELESS_CHAIN_FUSION_ATTR: True,
            "fused_ops": [matmul_id, add_id],
            "removed_intermediate": matmul_input,
            "removed_intermediates": list(removed_intermediates),
        }
        if activation_op is not None:
            fused_id_parts.append(activation_id or "")
            output_id = activation_op.outputs[0]
            removed_intermediates.append(add_output)
            fused_cls = FusedMatMulAddActivation
            attrs["activation"] = activation_op.op_type
            attrs["fused_ops"] = [matmul_id, add_id, activation_id]
            attrs["removed_intermediates"] = list(removed_intermediates)

        fused_id = "_".join(fused_id_parts) + "_fused"
        ops[fused_id] = fused_cls(
            fused_id,
            inputs=[matmul_op.inputs[0], matmul_op.inputs[1], bias_input],
            outputs=[output_id],
            attrs=attrs,
        )
        del ops[matmul_id]
        del ops[add_id]
        if activation_id is not None:
            del ops[activation_id]
        for intermediate in removed_intermediates:
            if (
                intermediate not in graph.graph_outputs
                and intermediate not in graph.graph_inputs
            ):
                values.pop(intermediate, None)
        break

    if ops == graph.ops and values == graph.values:
        return kernel
    return Kernel(
        kernel.kernel_id,
        graph=graph.__class__(
            values=values,
            ops=ops,
            graph_inputs=list(graph.graph_inputs),
            graph_outputs=list(graph.graph_outputs),
            states=dict(graph.states),
            registry=graph.registry,
        ),
        clock_id=kernel.clock_id,
    )


def _single_supported_activation_consumer(
    graph: Graph,
    consumers: dict[str, list[str]],
    value_id: str,
) -> str | None:
    value_consumers = consumers.get(value_id, [])
    if len(value_consumers) != 1:
        return None
    consumer_id = value_consumers[0]
    consumer = graph.ops[consumer_id]
    if (
        consumer.op_type in _SUPPORTED_FUSED_ACTIVATIONS
        and len(consumer.inputs) == 1
        and len(consumer.outputs) == 1
    ):
        return consumer_id
    return None


def _validate_parameter_identity(
    original_values: dict[str, Value],
    optimized_values: dict[str, Value],
) -> None:
    original_parameters = {
        value_id: value
        for value_id, value in original_values.items()
        if _is_parameter_value(value)
    }
    optimized_parameters = {
        value_id: value
        for value_id, value in optimized_values.items()
        if _is_parameter_value(value)
    }
    if set(optimized_parameters) != set(original_parameters):
        raise TemporalOptimizationError("rewrite must preserve parameter identifiers")
    for value_id, original_value in original_parameters.items():
        optimized_value = optimized_parameters[value_id]
        if optimized_value.to_dict() != original_value.to_dict():
            raise TemporalOptimizationError(
                "rewrite must preserve parameter dtype, shape, and metadata"
            )


def _validate_preserved_value_metadata(
    original_values: dict[str, Value],
    optimized_values: dict[str, Value],
) -> None:
    for value_id, original_value in original_values.items():
        optimized_value = optimized_values.get(value_id)
        if optimized_value is None:
            continue
        if optimized_value.to_dict() != original_value.to_dict():
            raise TemporalOptimizationError(
                "rewrite must preserve value dtype, shape, and metadata"
            )


def _buffer_sharing_signature(buffer: BufferSpec) -> tuple[object, ...]:
    metadata = {
        key: value
        for key, value in buffer.metadata.items()
        if key not in {"physical_buffer_id", "shared_buffer_group"}
    }
    return (
        buffer.dtype,
        buffer.shape,
        buffer.depth,
        buffer.axes,
        buffer.clock_id,
        tuple(sorted(metadata.items())),
    )


def _edge0_signature(edges: list[Edge0]) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(sorted((edge.source, edge.target, edge.value_id) for edge in edges))


def _edge_delta_signature(
    edges: list[EdgeDelta],
) -> tuple[tuple[str, str, int, str | None], ...]:
    return tuple(
        sorted(
            (edge.source, edge.target, edge.lag_cycles, edge.value_id) for edge in edges
        )
    )


def _is_parameter_value(value: Value) -> bool:
    return value.layout == "parameter" or (
        isinstance(value.quant, dict) and value.quant.get("role") == "parameter"
    )


def _value_producers(graph: Graph) -> dict[str, str]:
    return {
        output_id: op_id
        for op_id, operator in graph.ops.items()
        for output_id in operator.outputs
    }


def _value_consumers(graph: Graph) -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for op_id, operator in graph.ops.items():
        for input_id in operator.inputs:
            consumers.setdefault(input_id, []).append(op_id)
    return consumers


def _lookup_value(values: Mapping[str, Value], value_id: str, op_type: str) -> Value:
    try:
        return values[value_id]
    except KeyError as exc:
        raise InvalidOperatorInstanceError(
            f"{op_type} references unknown value '{value_id}'"
        ) from exc


def _matches_tensor_or_scalar(value: Value, shape: list[int]) -> bool:
    if value.vtype is ValueType.SCALAR:
        return True
    return value.vtype is ValueType.TENSOR and value.shape == shape


def _shape_product(shape: list[int]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


def _activation_expression(activation: str, value_name: str, cpp_dtype: str) -> str:
    zero = f"({cpp_dtype})0"
    one = f"({cpp_dtype})1"
    half = f"({cpp_dtype})0.5"
    alpha = f"({cpp_dtype})0.7978845608028654"
    beta = f"({cpp_dtype})0.044715"
    if activation == "ReLU":
        return f"{value_name} > {zero} ? {value_name} : {zero}"
    if activation == "Tanh":
        return f"std::tanh({value_name})"
    if activation == "Sigmoid":
        return f"{one} / ({one} + std::exp(-{value_name}))"
    if activation == "GELU":
        cubic = f"{value_name} * {value_name} * {value_name}"
        inner = f"{alpha} * ({value_name} + {beta} * {cubic})"
        return f"{half} * {value_name} * ({one} + std::tanh({inner}))"
    raise InvalidOperatorInstanceError(f"unsupported fused activation '{activation}'")


def _conv1d_output_shape(
    input_value: Value,
    weight: Value,
    attrs: Mapping[str, object],
) -> list[int]:
    if len(input_value.shape) != 3 or len(weight.shape) != 3:
        raise InvalidOperatorInstanceError(
            "FusedConv1DAdd requires rank-3 input and weight tensors"
        )
    if input_value.shape[1] != weight.shape[1]:
        raise InvalidOperatorInstanceError(
            "FusedConv1DAdd requires input channels to match weight channels"
        )
    stride = _int_attr(attrs, "stride", 1)
    padding = _int_attr(attrs, "padding", 0)
    dilation = _int_attr(attrs, "dilation", 1)
    if stride <= 0 or dilation <= 0 or padding < 0:
        raise InvalidOperatorInstanceError(
            "FusedConv1DAdd requires positive stride/dilation and non-negative padding"
        )
    batch, _, input_length = input_value.shape
    out_channels, _, kernel_width = weight.shape
    numerator = input_length + (2 * padding) - (dilation * (kernel_width - 1)) - 1
    if numerator < 0:
        raise InvalidOperatorInstanceError(
            "FusedConv1DAdd has invalid kernel/padding/dilation"
        )
    output_length = numerator // stride + 1
    if output_length <= 0:
        raise InvalidOperatorInstanceError("FusedConv1DAdd output length must be > 0")
    return [batch, out_channels, output_length]


def _int_attr(attrs: Mapping[str, object], name: str, default: int) -> int:
    value = attrs.get(name, default)
    if not isinstance(value, int):
        raise InvalidOperatorInstanceError(f"{name} must be an integer")
    return value


def _delta(before: object, after: object) -> object:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


__all__ = [
    "TemporalOptimizationError",
    "TemporalOptimizationResult",
    "TemporalRewritePass",
    "TemporalRewriteRecord",
    "FusedConv1DAdd",
    "FusedConv1DAddActivation",
    "FusedMatMulAdd",
    "FusedMatMulAddActivation",
    "FusedScaleAdd",
    "FusedScaleAddActivation",
    "fuse_parameterized_conv1d_add",
    "fuse_parameterized_matmul_add",
    "fuse_parameterized_scale_add",
    "optimize_temporal_process",
    "share_compatible_temporal_buffers",
    "validate_temporal_rewrite",
]
