from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from tempo_dag.ir.value import Value
from tempo_dag.ir_temporal.process import Edge0, EdgeDelta, Process
from tempo_dag.ir_temporal.report import (
    TemporalBaselineReport,
    derive_temporal_baseline_report,
)
from tempo_dag.ir_temporal.schedule import derive_temporal_schedule

TemporalRewritePass = Callable[[Process], Process]


class TemporalOptimizationError(ValueError):
    """Raised when a temporal graph rewrite violates optimizer legality."""


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


def _delta(before: object, after: object) -> object:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


__all__ = [
    "TemporalOptimizationError",
    "TemporalOptimizationResult",
    "TemporalRewritePass",
    "TemporalRewriteRecord",
    "optimize_temporal_process",
    "validate_temporal_rewrite",
]
