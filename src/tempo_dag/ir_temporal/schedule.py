from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from tempo_dag.ir.op import FPGACost
from tempo_dag.ir_temporal.contract import (
    TemporalExecutionContract,
    TemporalStorageKind,
    derive_temporal_execution_contract,
)
from tempo_dag.ir_temporal.process import Edge0, EdgeDelta, Kernel, Process


class ScheduleNodeKind(Enum):
    """Component categories visible to the process-level scheduler."""

    KERNEL = "kernel"
    OPERATOR = "operator"
    STATE = "state"
    BUFFER = "buffer"


class ScheduleEdgeKind(Enum):
    """ABI role assigned to one data/control dependency."""

    STREAM = "stream"
    GRAPH_INPUT = "graph_input"
    GRAPH_OUTPUT = "graph_output"
    PARAMETER_BLOCK = "parameter_block"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    BUFFER_READ = "buffer_read"
    BUFFER_WRITE = "buffer_write"
    TEMPORAL_DELAY = "temporal_delay"


@dataclass(frozen=True)
class ScheduleNode:
    """One scheduled component or operator."""

    node_id: str
    kind: ScheduleNodeKind
    phase: int
    latency_cycles: int = 0
    initiation_interval: int = 1
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "phase": self.phase,
            "latency_cycles": self.latency_cycles,
            "initiation_interval": self.initiation_interval,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ScheduleEdge:
    """One scheduled ABI edge between components, ops, or external ports."""

    edge_id: str
    kind: ScheduleEdgeKind
    source: str
    target: str
    value_id: str | None = None
    storage_kind: TemporalStorageKind | None = None
    latency_cycles: int = 0
    phase: int = 0
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "edge_id": self.edge_id,
            "kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "value_id": self.value_id,
            "latency_cycles": self.latency_cycles,
            "phase": self.phase,
            "metadata": dict(self.metadata or {}),
        }
        if self.storage_kind is not None:
            payload["storage_kind"] = self.storage_kind.value
        return payload


@dataclass(frozen=True)
class TemporalSchedule:
    """Baseline task-level schedule and ABI classification for a process."""

    process_id: str
    nodes: tuple[ScheduleNode, ...]
    edges: tuple[ScheduleEdge, ...]
    estimated_latency_cycles: int
    estimated_initiation_interval: int

    def to_dict(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "estimated_latency_cycles": self.estimated_latency_cycles,
            "estimated_initiation_interval": self.estimated_initiation_interval,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def derive_temporal_schedule(
    process: Process,
    contract: TemporalExecutionContract | None = None,
) -> TemporalSchedule:
    """Derive a conservative task-level schedule and channel ABI report."""

    if contract is None:
        contract = derive_temporal_execution_contract(process)
    component_phases = _component_phases(process)
    nodes: list[ScheduleNode] = []
    edges: list[ScheduleEdge] = []

    for state_id in sorted(process.states):
        nodes.append(
            ScheduleNode(
                node_id=state_id,
                kind=ScheduleNodeKind.STATE,
                phase=component_phases.get(state_id, 0),
            )
        )
    for buffer_id, buffer in sorted(process.buffers.items()):
        nodes.append(
            ScheduleNode(
                node_id=buffer_id,
                kind=ScheduleNodeKind.BUFFER,
                phase=component_phases.get(buffer_id, 0),
                latency_cycles=buffer.depth,
                metadata={"depth": buffer.depth},
            )
        )
    for kernel_id, kernel in sorted(process.kernels.items()):
        kernel_cost = _kernel_cost(kernel)
        nodes.append(
            ScheduleNode(
                node_id=kernel_id,
                kind=ScheduleNodeKind.KERNEL,
                phase=component_phases.get(kernel_id, 0),
                latency_cycles=kernel_cost.latency_cycles,
                initiation_interval=kernel_cost.initiation_interval,
                metadata={"clock_id": kernel.clock_id},
            )
        )
        operator_nodes, operator_edges = _kernel_schedule(
            kernel,
            component_phases[kernel_id],
        )
        nodes.extend(operator_nodes)
        edges.extend(operator_edges)

    buffer_storage = {
        mapping.component_id: mapping.storage_kind
        for mapping in contract.buffer_storage
    }
    temporal_storage = {
        mapping.component_id: mapping.storage_kind
        for mapping in contract.edge_delta_storage
    }
    for edge in process.edge0:
        edges.append(
            _schedule_edge0(
                process,
                edge,
                phase=component_phases.get(edge.target, 0),
                buffer_storage=buffer_storage,
            )
        )
    for edge in process.edge_delta:
        edge_id = _edge_delta_id(edge)
        edges.append(
            ScheduleEdge(
                edge_id=edge_id,
                kind=ScheduleEdgeKind.TEMPORAL_DELAY,
                source=edge.source,
                target=edge.target,
                value_id=edge.value_id,
                storage_kind=temporal_storage.get(edge_id),
                latency_cycles=edge.lag_cycles,
                phase=component_phases.get(edge.target, 0),
                metadata={"lag_cycles": edge.lag_cycles},
            )
        )

    estimated_latency = sum(
        node.latency_cycles for node in nodes if node.kind is ScheduleNodeKind.KERNEL
    )
    estimated_latency += max((edge.latency_cycles for edge in edges), default=0)
    estimated_ii = max((node.initiation_interval for node in nodes), default=1)
    return TemporalSchedule(
        process_id=process.process_id,
        nodes=tuple(sorted(nodes, key=lambda node: (node.phase, node.node_id))),
        edges=tuple(sorted(edges, key=lambda edge: (edge.phase, edge.edge_id))),
        estimated_latency_cycles=max(1, estimated_latency),
        estimated_initiation_interval=max(1, estimated_ii),
    )


def _component_phases(process: Process) -> dict[str, int]:
    component_ids = process.component_ids()
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {component_id: 0 for component_id in component_ids}
    for edge in process.edge0:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    phases = {node: 0 for node in ready}
    while ready:
        node = ready.popleft()
        for target in sorted(adjacency[node]):
            phases[target] = max(phases.get(target, 0), phases[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return phases


def _kernel_schedule(
    kernel: Kernel,
    kernel_phase: int,
) -> tuple[list[ScheduleNode], list[ScheduleEdge]]:
    graph = kernel.graph
    op_phases = _operator_phases(kernel)
    value_producers = _value_producers(kernel)
    nodes: list[ScheduleNode] = []
    edges: list[ScheduleEdge] = []

    for op_id, operator in sorted(graph.ops.items()):
        cost = operator.estimate_fpga_cost(graph.values)
        phase = kernel_phase + op_phases[op_id]
        nodes.append(
            ScheduleNode(
                node_id=f"{kernel.kernel_id}.{op_id}",
                kind=ScheduleNodeKind.OPERATOR,
                phase=phase,
                latency_cycles=cost.latency_cycles,
                initiation_interval=cost.initiation_interval,
                metadata={
                    "kernel_id": kernel.kernel_id,
                    "op_type": operator.op_type,
                    "dsp": cost.dsp,
                    "bram": cost.bram,
                    "lut": cost.lut,
                    "ff": cost.ff,
                },
            )
        )
        for input_id in operator.inputs:
            producer = value_producers.get(input_id)
            if producer is not None:
                edges.append(
                    ScheduleEdge(
                        edge_id=f"{kernel.kernel_id}.{producer}->{op_id}:{input_id}",
                        kind=ScheduleEdgeKind.STREAM,
                        source=f"{kernel.kernel_id}.{producer}",
                        target=f"{kernel.kernel_id}.{op_id}",
                        value_id=input_id,
                        storage_kind=TemporalStorageKind.WIRE,
                        phase=phase,
                    )
                )
            elif input_id in graph.graph_inputs:
                value = graph.values[input_id]
                if _is_parameter_value(value):
                    edges.append(
                        ScheduleEdge(
                            edge_id=f"{kernel.kernel_id}.param->{op_id}:{input_id}",
                            kind=ScheduleEdgeKind.PARAMETER_BLOCK,
                            source=f"{kernel.kernel_id}.param",
                            target=f"{kernel.kernel_id}.{op_id}",
                            value_id=input_id,
                            storage_kind=TemporalStorageKind.RAM,
                            phase=phase,
                        )
                    )
                else:
                    edges.append(
                        ScheduleEdge(
                            edge_id=f"{kernel.kernel_id}.input->{op_id}:{input_id}",
                            kind=ScheduleEdgeKind.GRAPH_INPUT,
                            source=f"{kernel.kernel_id}.input",
                            target=f"{kernel.kernel_id}.{op_id}",
                            value_id=input_id,
                            phase=phase,
                        )
                    )
            else:
                edges.append(
                    ScheduleEdge(
                        edge_id=f"{kernel.kernel_id}.param->{op_id}:{input_id}",
                        kind=ScheduleEdgeKind.PARAMETER_BLOCK,
                        source=f"{kernel.kernel_id}.param",
                        target=f"{kernel.kernel_id}.{op_id}",
                        value_id=input_id,
                        storage_kind=TemporalStorageKind.RAM,
                        phase=phase,
                    )
                )

    for output_id in graph.graph_outputs:
        producer = value_producers.get(output_id)
        if producer is not None:
            edges.append(
                ScheduleEdge(
                    edge_id=f"{kernel.kernel_id}.{producer}->output:{output_id}",
                    kind=ScheduleEdgeKind.GRAPH_OUTPUT,
                    source=f"{kernel.kernel_id}.{producer}",
                    target=f"{kernel.kernel_id}.output",
                    value_id=output_id,
                    phase=kernel_phase + op_phases[producer],
                )
            )
    return nodes, edges


def _operator_phases(kernel: Kernel) -> dict[str, int]:
    graph = kernel.graph
    value_producers = _value_producers(kernel)
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {op_id: 0 for op_id in graph.ops}
    for op_id, operator in graph.ops.items():
        for input_id in operator.inputs:
            producer = value_producers.get(input_id)
            if producer is not None and producer != op_id:
                adjacency[producer].append(op_id)
                indegree[op_id] += 1

    ready = deque(sorted(op_id for op_id, degree in indegree.items() if degree == 0))
    phases = {op_id: 0 for op_id in ready}
    while ready:
        op_id = ready.popleft()
        for target in sorted(adjacency[op_id]):
            phases[target] = max(phases.get(target, 0), phases[op_id] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return phases


def _value_producers(kernel: Kernel) -> dict[str, str]:
    producers = {}
    for op_id, operator in kernel.graph.ops.items():
        for output_id in operator.outputs:
            producers[output_id] = op_id
    return producers


def _is_parameter_value(value: object) -> bool:
    layout = getattr(value, "layout", None)
    quant = getattr(value, "quant", None)
    return layout == "parameter" or (
        isinstance(quant, dict) and quant.get("role") == "parameter"
    )


def _kernel_cost(kernel: Kernel) -> FPGACost:
    costs = [
        operator.estimate_fpga_cost(kernel.graph.values)
        for operator in kernel.graph.ops.values()
    ]
    if not costs:
        return FPGACost(latency_cycles=1)
    return FPGACost(
        latency_cycles=sum(cost.latency_cycles for cost in costs),
        initiation_interval=max(cost.initiation_interval for cost in costs),
        dsp=sum(cost.dsp for cost in costs),
        bram=sum(cost.bram for cost in costs),
        lut=sum(cost.lut for cost in costs),
        ff=sum(cost.ff for cost in costs),
    )


def _schedule_edge0(
    process: Process,
    edge: Edge0,
    *,
    phase: int,
    buffer_storage: dict[str, TemporalStorageKind],
) -> ScheduleEdge:
    kind = ScheduleEdgeKind.STREAM
    storage_kind = TemporalStorageKind.WIRE
    if edge.source in process.states and edge.target in process.kernels:
        kind = ScheduleEdgeKind.STATE_READ
        storage_kind = TemporalStorageKind.REGISTER
    elif edge.source in process.kernels and edge.target in process.states:
        kind = ScheduleEdgeKind.STATE_WRITE
        storage_kind = TemporalStorageKind.REGISTER
    elif edge.source in process.buffers and edge.target in process.kernels:
        kind = ScheduleEdgeKind.BUFFER_READ
        storage_kind = buffer_storage.get(edge.source, TemporalStorageKind.RING_BUFFER)
    elif edge.source in process.kernels and edge.target in process.buffers:
        kind = ScheduleEdgeKind.BUFFER_WRITE
        storage_kind = buffer_storage.get(edge.target, TemporalStorageKind.RING_BUFFER)
    return ScheduleEdge(
        edge_id=f"{edge.source}->{edge.target}:{edge.value_id or 'value'}",
        kind=kind,
        source=edge.source,
        target=edge.target,
        value_id=edge.value_id,
        storage_kind=storage_kind,
        phase=phase,
    )


def _edge_delta_id(edge: EdgeDelta) -> str:
    value_suffix = f":{edge.value_id}" if edge.value_id else ""
    return f"{edge.source}->{edge.target}@{edge.lag_cycles}{value_suffix}"
