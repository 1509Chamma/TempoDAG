from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tempo_dag.ir_temporal.process import BufferSpec, EdgeDelta, Process


class ResetPolicy(Enum):
    """Reset initialization policy for persistent temporal storage."""

    ZERO = "zero"
    METADATA_INITIALIZER = "metadata_initializer"


class TemporalStorageKind(Enum):
    """Conservative HLS storage choices for temporal dependencies."""

    WIRE = "wire"
    REGISTER = "register"
    SHIFT_REGISTER = "shift_register"
    FIFO = "fifo"
    RING_BUFFER = "ring_buffer"
    RAM = "ram"


@dataclass(frozen=True)
class TemporalStorageMapping:
    """Suggested storage for one temporal component or dependency."""

    component_id: str
    storage_kind: TemporalStorageKind
    latency_cycles: int = 0


@dataclass(frozen=True)
class TemporalExecutionContract:
    """Machine-readable summary of the temporal execution contract."""

    process_id: str
    reset_policy: ResetPolicy
    warmup_timesteps: int
    flush_cycles: int
    edge_delta_storage: tuple[TemporalStorageMapping, ...]
    buffer_storage: tuple[TemporalStorageMapping, ...]

    @property
    def has_warmup(self) -> bool:
        return self.warmup_timesteps > 0


def derive_temporal_execution_contract(process: Process) -> TemporalExecutionContract:
    """Validate a process and derive conservative execution metadata."""

    process.validate()
    return TemporalExecutionContract(
        process_id=process.process_id,
        reset_policy=_reset_policy(process),
        warmup_timesteps=max(
            _max_edge_delta_lag(process.edge_delta),
            _max_buffer_warmup(process),
        ),
        flush_cycles=_flush_cycles(process),
        edge_delta_storage=tuple(
            TemporalStorageMapping(
                component_id=_edge_delta_id(edge),
                storage_kind=_storage_for_edge_delta(edge),
                latency_cycles=edge.lag_cycles,
            )
            for edge in process.edge_delta
        ),
        buffer_storage=tuple(
            TemporalStorageMapping(
                component_id=buffer.buffer_id,
                storage_kind=_storage_for_buffer(buffer),
                latency_cycles=buffer.depth,
            )
            for buffer in process.buffers.values()
        ),
    )


def _reset_policy(process: Process) -> ResetPolicy:
    has_initializer = any(
        "initializer" in component.metadata
        for component in (*process.states.values(), *process.buffers.values())
    )
    if has_initializer:
        return ResetPolicy.METADATA_INITIALIZER
    return ResetPolicy.ZERO


def _max_edge_delta_lag(edges: list[EdgeDelta]) -> int:
    return max((edge.lag_cycles for edge in edges), default=0)


def _max_buffer_warmup(process: Process) -> int:
    return max((buffer.depth - 1 for buffer in process.buffers.values()), default=0)


def _flush_cycles(process: Process) -> int:
    value = process.metadata.get("flush_cycles", 0)
    if not isinstance(value, int) or value < 0:
        raise ValueError("process metadata 'flush_cycles' must be a non-negative int")
    return value


def _edge_delta_id(edge: EdgeDelta) -> str:
    value_suffix = f":{edge.value_id}" if edge.value_id else ""
    return f"{edge.source}->{edge.target}@{edge.lag_cycles}{value_suffix}"


def _storage_for_edge_delta(edge: EdgeDelta) -> TemporalStorageKind:
    if edge.lag_cycles == 1:
        return TemporalStorageKind.REGISTER
    if edge.lag_cycles <= 64:
        return TemporalStorageKind.SHIFT_REGISTER
    if edge.lag_cycles <= 1024:
        return TemporalStorageKind.FIFO
    return TemporalStorageKind.RAM


def _storage_for_buffer(buffer: BufferSpec) -> TemporalStorageKind:
    if buffer.depth <= 2:
        return TemporalStorageKind.SHIFT_REGISTER
    if buffer.depth <= 1024:
        return TemporalStorageKind.RING_BUFFER
    return TemporalStorageKind.RAM
