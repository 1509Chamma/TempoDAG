from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tempo_dag.ir_temporal.process import Kernel, Process
from tempo_dag.ir_temporal.schedule import (
    ScheduleEdge,
    ScheduleEdgeKind,
    ScheduleNode,
    ScheduleNodeKind,
    TemporalSchedule,
    derive_temporal_schedule,
)


@dataclass(frozen=True)
class TemporalBaselineReport:
    """Human- and machine-readable baseline schedule/cost report."""

    process_id: str
    summary: dict[str, object]
    node_table: tuple[dict[str, object], ...]
    edge_table: tuple[dict[str, object], ...]
    buffer_table: tuple[dict[str, object], ...]
    resource_summary: dict[str, object]
    traffic_summary: dict[str, object]
    directive_plan: tuple[dict[str, object], ...]
    baseline_comparison: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "summary": dict(self.summary),
            "node_table": [dict(row) for row in self.node_table],
            "edge_table": [dict(row) for row in self.edge_table],
            "buffer_table": [dict(row) for row in self.buffer_table],
            "resource_summary": dict(self.resource_summary),
            "traffic_summary": dict(self.traffic_summary),
            "directive_plan": [dict(row) for row in self.directive_plan],
            "baseline_comparison": dict(self.baseline_comparison),
        }


def derive_temporal_baseline_report(
    process: Process,
    schedule: TemporalSchedule | None = None,
    *,
    trace_metadata: dict[str, object] | None = None,
) -> TemporalBaselineReport:
    """Derive conservative baseline report tables from a temporal schedule."""

    if schedule is None:
        schedule = derive_temporal_schedule(process)

    node_table = tuple(_node_row(node) for node in schedule.nodes)
    edge_table = tuple(_edge_row(process, edge) for edge in schedule.edges)
    buffer_table = tuple(
        _buffer_row(process, edge) for edge in schedule.edges if _is_storage_edge(edge)
    )
    resource_summary = _resource_summary(schedule.nodes)
    traffic_summary = _traffic_summary(edge_table)
    directive_plan = tuple(_directive_plan(schedule))
    baseline_comparison = _baseline_comparison(schedule, trace_metadata or {})
    summary: dict[str, object] = {
        "estimated_latency_cycles": schedule.estimated_latency_cycles,
        "estimated_initiation_interval": schedule.estimated_initiation_interval,
        "estimated_events_per_cycle": 1.0 / schedule.estimated_initiation_interval,
        "num_nodes": len(schedule.nodes),
        "num_edges": len(schedule.edges),
        "num_buffers": len(process.buffers),
        "num_states": len(process.states),
        "num_kernels": len(process.kernels),
    }

    return TemporalBaselineReport(
        process_id=process.process_id,
        summary=summary,
        node_table=node_table,
        edge_table=edge_table,
        buffer_table=buffer_table,
        resource_summary=resource_summary,
        traffic_summary=traffic_summary,
        directive_plan=directive_plan,
        baseline_comparison=baseline_comparison,
    )


def _node_row(node: ScheduleNode) -> dict[str, object]:
    row: dict[str, object] = {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "phase": node.phase,
        "latency_cycles": node.latency_cycles,
        "initiation_interval": node.initiation_interval,
    }
    for key, value in _prefixed_resource_metadata(node.metadata or {}).items():
        row[key] = value
    if node.kind is ScheduleNodeKind.OPERATOR:
        row["op_type"] = (node.metadata or {}).get("op_type", "unknown")
    return row


def _edge_row(process: Process, edge: ScheduleEdge) -> dict[str, object]:
    value_metadata = _value_metadata(process, edge)
    elements = value_metadata["elements_per_timestep"]
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value,
        "source": edge.source,
        "target": edge.target,
        "value_id": edge.value_id,
        "phase": edge.phase,
        "latency_cycles": edge.latency_cycles,
        "storage_kind": edge.storage_kind.value if edge.storage_kind else None,
        "dtype": value_metadata["dtype"],
        "shape": value_metadata["shape"],
        "elements_per_timestep": elements,
        "traffic_elements_per_timestep": elements,
    }


def _buffer_row(process: Process, edge: ScheduleEdge) -> dict[str, object]:
    value_metadata = _value_metadata(process, edge)
    depth = max(1, edge.latency_cycles)
    if edge.target in process.buffers:
        depth = process.buffers[edge.target].depth
    elif edge.source in process.buffers:
        depth = process.buffers[edge.source].depth
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value,
        "storage_kind": edge.storage_kind.value if edge.storage_kind else None,
        "depth": depth,
        "dtype": value_metadata["dtype"],
        "shape": value_metadata["shape"],
        "elements": value_metadata["elements_per_timestep"],
    }


def _resource_summary(nodes: tuple[ScheduleNode, ...]) -> dict[str, object]:
    totals = {"dsp": 0, "bram": 0, "lut": 0, "ff": 0, "uram": 0}
    for node in nodes:
        if node.kind is not ScheduleNodeKind.OPERATOR:
            continue
        metadata = node.metadata or {}
        for key in totals:
            value = metadata.get(key, 0)
            if isinstance(value, int | float):
                totals[key] += int(value)
    return {
        "resource_model": "coarse_operator_sum",
        **totals,
    }


def _traffic_summary(edge_table: tuple[dict[str, object], ...]) -> dict[str, object]:
    total = 0
    by_kind: dict[str, int] = {}
    for row in edge_table:
        kind = str(row["kind"])
        elements_value = row["traffic_elements_per_timestep"]
        elements = elements_value if isinstance(elements_value, int) else 0
        total += elements
        by_kind[kind] = by_kind.get(kind, 0) + elements
    return {
        "traffic_model": "value_elements_per_timestep",
        "total_elements_per_timestep": total,
        "by_edge_kind": by_kind,
    }


def _directive_plan(schedule: TemporalSchedule) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = [
        {
            "target": schedule.process_id,
            "directive": "DATAFLOW",
            "setting": "enabled",
            "reason": "Baseline process-level task overlap.",
        }
    ]
    for node in schedule.nodes:
        if node.kind is ScheduleNodeKind.OPERATOR:
            plan.append(
                {
                    "target": node.node_id,
                    "directive": "PIPELINE",
                    "setting": f"II={node.initiation_interval}",
                    "reason": "Safe operator-level pipeline target from cost model.",
                }
            )
        elif node.kind in {ScheduleNodeKind.BUFFER, ScheduleNodeKind.STATE}:
            plan.append(
                {
                    "target": node.node_id,
                    "directive": "BIND_STORAGE",
                    "setting": "default_temporal_storage",
                    "reason": (
                        "Storage kind is selected by temporal execution contract."
                    ),
                }
            )
    for edge in schedule.edges:
        if edge.kind in {
            ScheduleEdgeKind.STREAM,
            ScheduleEdgeKind.GRAPH_INPUT,
            ScheduleEdgeKind.GRAPH_OUTPUT,
        }:
            plan.append(
                {
                    "target": edge.edge_id,
                    "directive": "STREAM",
                    "setting": "depth=2",
                    "reason": "Conservative FIFO depth for baseline channel ABI.",
                }
            )
    return plan


def _baseline_comparison(
    schedule: TemporalSchedule,
    metadata: dict[str, object],
) -> dict[str, object]:
    latency_ns = _first_float(
        metadata,
        (
            "python_latency_ns_per_step",
            "software_latency_ns_per_step",
            "baseline_latency_ns_per_step",
        ),
    )
    clock_period_ns = _first_float(metadata, ("hls_clock_period_ns",))
    if clock_period_ns is None:
        clock_period_ns = 5.0
    estimated_hls_latency_ns = schedule.estimated_latency_cycles * clock_period_ns
    comparison: dict[str, object] = {
        "baseline_source": "trace_metadata",
        "python_latency_ns_per_step": latency_ns,
        "hls_clock_period_ns": clock_period_ns,
        "estimated_hls_latency_ns_per_step": estimated_hls_latency_ns,
    }
    if latency_ns is not None and estimated_hls_latency_ns > 0:
        comparison["estimated_speedup_vs_python"] = (
            latency_ns / estimated_hls_latency_ns
        )
    else:
        comparison["estimated_speedup_vs_python"] = None
    return comparison


def _value_metadata(process: Process, edge: ScheduleEdge) -> dict[str, object]:
    if edge.value_id is not None:
        for kernel in _candidate_kernels(process, edge):
            value = kernel.graph.values.get(edge.value_id)
            if value is not None:
                return _shape_metadata(value.dtype, value.shape)
    for component_id in (edge.source, edge.target):
        if component_id in process.states:
            state = process.states[component_id]
            return _shape_metadata(state.dtype, state.shape)
        if component_id in process.buffers:
            buffer = process.buffers[component_id]
            return _shape_metadata(buffer.dtype, buffer.shape)
    return _shape_metadata("unknown", ())


def _candidate_kernels(process: Process, edge: ScheduleEdge) -> list[Kernel]:
    candidates: list[Kernel] = []
    for endpoint in (edge.source, edge.target):
        kernel_id = endpoint.split(".", maxsplit=1)[0]
        kernel = process.kernels.get(kernel_id)
        if kernel is not None and kernel not in candidates:
            candidates.append(kernel)
    for kernel in process.kernels.values():
        if kernel not in candidates:
            candidates.append(kernel)
    return candidates


def _shape_metadata(
    dtype: str,
    shape: tuple[int, ...] | list[int],
) -> dict[str, object]:
    normalized_shape = tuple(int(dim) for dim in shape)
    elements = int(prod(normalized_shape)) if normalized_shape else 1
    return {
        "dtype": dtype,
        "shape": list(normalized_shape),
        "elements_per_timestep": elements,
    }


def _prefixed_resource_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        f"resource_{key}": metadata.get(key, 0)
        for key in ("dsp", "bram", "lut", "ff", "uram")
        if key in metadata
    }


def _is_storage_edge(edge: ScheduleEdge) -> bool:
    return edge.kind in {
        ScheduleEdgeKind.STATE_READ,
        ScheduleEdgeKind.STATE_WRITE,
        ScheduleEdgeKind.BUFFER_READ,
        ScheduleEdgeKind.BUFFER_WRITE,
        ScheduleEdgeKind.TEMPORAL_DELAY,
    }


def _first_float(
    metadata: dict[str, object],
    keys: tuple[str, ...],
    *,
    default: float | None = None,
) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int | float):
            return float(value)
    return default


__all__ = [
    "TemporalBaselineReport",
    "derive_temporal_baseline_report",
]
