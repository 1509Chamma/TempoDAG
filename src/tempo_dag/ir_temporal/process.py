from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.validation import IRValidationError, validate_ir


class TemporalIRValidationError(ValueError):
    """Raised when a temporal IR process violates structural invariants."""


class StateKind(Enum):
    HIDDEN = "hidden_state"
    ROLLING_BUFFER = "rolling_buffer"
    RUNNING_STAT = "running_stat"


@dataclass(frozen=True)
class Clock:
    """Logical clock domain for a temporal process."""

    clock_id: str
    period: int = 1
    unit: str = "cycle"

    def to_dict(self) -> dict[str, object]:
        return {
            "clock_id": self.clock_id,
            "period": self.period,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class Kernel:
    """A same-timestep acyclic compute region backed by the existing IR graph."""

    kernel_id: str
    graph: Graph
    clock_id: str = "main"

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "clock_id": self.clock_id,
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True)
class StateSpec:
    """Persistent value carried across timesteps."""

    state_id: str
    kind: StateKind
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...] = ()
    clock_id: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "kind": self.kind.value,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "clock_id": self.clock_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BufferSpec:
    """Bounded history storage such as a delay line or rolling window."""

    buffer_id: str
    dtype: str
    shape: tuple[int, ...]
    depth: int
    axes: tuple[str, ...] = ()
    clock_id: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "buffer_id": self.buffer_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "depth": self.depth,
            "axes": list(self.axes),
            "clock_id": self.clock_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Edge0:
    """Same-timestep dependency between temporal process components."""

    source: str
    target: str
    value_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "value_id": self.value_id,
        }


@dataclass(frozen=True)
class EdgeDelta:
    """Positive-lag dependency that crosses one or more timesteps."""

    source: str
    target: str
    lag_cycles: int
    value_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "lag_cycles": self.lag_cycles,
            "value_id": self.value_id,
        }


@dataclass
class Process:
    """Top-level container for a streaming temporal computation."""

    process_id: str
    clocks: dict[str, Clock] = field(default_factory=lambda: {"main": Clock("main")})
    kernels: dict[str, Kernel] = field(default_factory=dict)
    states: dict[str, StateSpec] = field(default_factory=dict)
    buffers: dict[str, BufferSpec] = field(default_factory=dict)
    edge0: list[Edge0] = field(default_factory=list)
    edge_delta: list[EdgeDelta] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_temporal_process(self)

    def component_ids(self) -> set[str]:
        return set(self.kernels) | set(self.states) | set(self.buffers)

    def to_dict(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "clocks": {
                clock_id: clock.to_dict()
                for clock_id, clock in sorted(self.clocks.items())
            },
            "kernels": {
                kernel_id: kernel.to_dict()
                for kernel_id, kernel in sorted(self.kernels.items())
            },
            "states": {
                state_id: state.to_dict()
                for state_id, state in sorted(self.states.items())
            },
            "buffers": {
                buffer_id: buffer.to_dict()
                for buffer_id, buffer in sorted(self.buffers.items())
            },
            "edge0": [edge.to_dict() for edge in self.edge0],
            "edge_delta": [edge.to_dict() for edge in self.edge_delta],
            "metadata": dict(self.metadata),
        }


def validate_temporal_process(process: Process) -> None:
    _validate_process_id(process.process_id)
    _validate_unique_component_ids(process)
    _validate_clocks(process)
    _validate_component_keys(process)
    _validate_component_clocks(process)
    _validate_kernels(process.kernels.values())
    _validate_state_specs(process.states.values())
    _validate_buffer_specs(process.buffers.values())
    _validate_edges(process)
    _validate_same_timestep_is_dag(process.edge0, process.component_ids())


def _validate_process_id(process_id: str) -> None:
    if not process_id:
        raise TemporalIRValidationError("process_id must be non-empty")


def _validate_unique_component_ids(process: Process) -> None:
    all_ids = list(process.kernels) + list(process.states) + list(process.buffers)
    if len(all_ids) != len(set(all_ids)):
        raise TemporalIRValidationError(
            "kernel, state, and buffer identifiers must be globally unique"
        )


def _validate_component_keys(process: Process) -> None:
    for kernel_id, kernel in process.kernels.items():
        if kernel.kernel_id != kernel_id:
            raise TemporalIRValidationError(
                f"kernel key '{kernel_id}' does not match kernel_id "
                f"'{kernel.kernel_id}'"
            )
    for state_id, state in process.states.items():
        if state.state_id != state_id:
            raise TemporalIRValidationError(
                f"state key '{state_id}' does not match state_id '{state.state_id}'"
            )
    for buffer_id, buffer in process.buffers.items():
        if buffer.buffer_id != buffer_id:
            raise TemporalIRValidationError(
                f"buffer key '{buffer_id}' does not match buffer_id "
                f"'{buffer.buffer_id}'"
            )


def _validate_clocks(process: Process) -> None:
    if not process.clocks:
        raise TemporalIRValidationError("process must define at least one clock")

    for clock_id, clock in process.clocks.items():
        if clock.clock_id != clock_id:
            raise TemporalIRValidationError(
                f"clock key '{clock_id}' does not match clock_id '{clock.clock_id}'"
            )
        if not clock.clock_id:
            raise TemporalIRValidationError("clock_id must be non-empty")
        if clock.period < 1:
            raise TemporalIRValidationError(f"clock '{clock_id}' period must be >= 1")


def _validate_component_clocks(process: Process) -> None:
    known_clocks = set(process.clocks)
    clocked_components = [
        *process.kernels.values(),
        *process.states.values(),
        *process.buffers.values(),
    ]
    for component in clocked_components:
        clock_id = component.clock_id
        if clock_id not in known_clocks:
            raise TemporalIRValidationError(
                f"component references unknown clock '{clock_id}'"
            )


def _validate_kernels(kernels: Iterable[Kernel]) -> None:
    for kernel in kernels:
        if not kernel.kernel_id:
            raise TemporalIRValidationError("kernel_id must be non-empty")
        try:
            validate_ir(kernel.graph)
        except IRValidationError as exc:
            raise TemporalIRValidationError(
                f"kernel '{kernel.kernel_id}' graph is invalid: {exc}"
            ) from exc


def _validate_state_specs(states: Iterable[StateSpec]) -> None:
    for state in states:
        if not state.state_id:
            raise TemporalIRValidationError("state_id must be non-empty")
        if len(state.shape) != len(state.axes) and state.axes:
            raise TemporalIRValidationError(
                f"state '{state.state_id}' shape and axes lengths differ"
            )
        if any(dim < 1 for dim in state.shape):
            raise TemporalIRValidationError(
                f"state '{state.state_id}' shape dimensions must be >= 1"
            )


def _validate_buffer_specs(buffers: Iterable[BufferSpec]) -> None:
    for buffer in buffers:
        if not buffer.buffer_id:
            raise TemporalIRValidationError("buffer_id must be non-empty")
        if buffer.depth < 1:
            raise TemporalIRValidationError(
                f"buffer '{buffer.buffer_id}' depth must be >= 1"
            )
        if len(buffer.shape) != len(buffer.axes) and buffer.axes:
            raise TemporalIRValidationError(
                f"buffer '{buffer.buffer_id}' shape and axes lengths differ"
            )
        if any(dim < 1 for dim in buffer.shape):
            raise TemporalIRValidationError(
                f"buffer '{buffer.buffer_id}' shape dimensions must be >= 1"
            )


def _validate_edges(process: Process) -> None:
    known_components = process.component_ids()
    if not known_components and (process.edge0 or process.edge_delta):
        raise TemporalIRValidationError("edges require at least one component")

    for edge in process.edge0:
        _validate_endpoint(edge.source, known_components, "edge0 source")
        _validate_endpoint(edge.target, known_components, "edge0 target")

    for edge in process.edge_delta:
        _validate_endpoint(edge.source, known_components, "edge_delta source")
        _validate_endpoint(edge.target, known_components, "edge_delta target")
        if edge.lag_cycles < 1:
            raise TemporalIRValidationError(
                "edge_delta lag_cycles must be a positive integer"
            )


def _validate_endpoint(endpoint: str, known_components: set[str], label: str) -> None:
    if endpoint not in known_components:
        raise TemporalIRValidationError(f"{label} references unknown '{endpoint}'")


def _validate_same_timestep_is_dag(
    edges: list[Edge0],
    component_ids: set[str],
) -> None:
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {component_id: 0 for component_id in component_ids}

    for edge in edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    ready = deque(
        component_id for component_id, degree in indegree.items() if degree == 0
    )
    visited = 0

    while ready:
        component_id = ready.popleft()
        visited += 1
        for target in adjacency[component_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if visited != len(indegree):
        raise TemporalIRValidationError(
            "same-timestep edge0 dependencies must form a DAG; "
            "use edge_delta for temporal cycles"
        )
