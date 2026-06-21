from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal.process import Edge0, EdgeDelta, Kernel, Process
from tempo_dag.ir_temporal.report import (
    TemporalBaselineReport,
    derive_temporal_baseline_report,
)
from tempo_dag.ir_temporal.schedule import derive_temporal_schedule

TemporalRewritePass = Callable[[Process], Process]
STATELESS_CHAIN_FUSION_ATTR = "tempo_dag_fused_stateless_chain"


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
    """Fuse safe `MatMul -> Add(parameter bias)` chains inside kernels."""

    optimized = deepcopy(process)
    for kernel_id, kernel in list(optimized.kernels.items()):
        fused_kernel = _fuse_kernel_parameterized_matmul_add(kernel)
        optimized.kernels[kernel_id] = fused_kernel
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
        fused_id = f"{matmul_id}_{add_id}_fused"
        output_id = add_op.outputs[0]
        ops[fused_id] = FusedMatMulAdd(
            fused_id,
            inputs=[matmul_op.inputs[0], matmul_op.inputs[1], bias_input],
            outputs=[output_id],
            attrs={
                STATELESS_CHAIN_FUSION_ATTR: True,
                "fused_ops": [matmul_id, add_id],
                "removed_intermediate": matmul_input,
            },
        )
        del ops[matmul_id]
        del ops[add_id]
        if (
            matmul_input not in graph.graph_outputs
            and matmul_input not in graph.graph_inputs
        ):
            values.pop(matmul_input, None)
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


def _delta(before: object, after: object) -> object:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


__all__ = [
    "TemporalOptimizationError",
    "TemporalOptimizationResult",
    "TemporalRewritePass",
    "TemporalRewriteRecord",
    "FusedMatMulAdd",
    "fuse_parameterized_matmul_add",
    "optimize_temporal_process",
    "validate_temporal_rewrite",
]
