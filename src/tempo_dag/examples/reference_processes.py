from __future__ import annotations

from tempo_dag.ir.graph import Graph
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
)


def delay_line_process() -> Process:
    """Reference graph for a fixed positive-lag temporal dependency."""

    kernel = _empty_kernel("delay_kernel")
    buffer = BufferSpec(
        buffer_id="delay_buffer",
        dtype="float32",
        shape=(1,),
        depth=4,
    )
    return Process(
        process_id="reference_delay_line",
        kernels={kernel.kernel_id: kernel},
        buffers={buffer.buffer_id: buffer},
        edge0=[Edge0("delay_buffer", "delay_kernel", value_id="x_t_minus_3")],
        edge_delta=[
            EdgeDelta("delay_kernel", "delay_buffer", lag_cycles=3, value_id="x")
        ],
    )


def recurrent_state_process() -> Process:
    """Reference graph for legal recurrence through positive-lag state."""

    kernel = _empty_kernel("cell")
    hidden = StateSpec(
        state_id="hidden",
        kind=StateKind.HIDDEN,
        dtype="float32",
        shape=(8,),
    )
    return Process(
        process_id="reference_recurrent_state",
        kernels={kernel.kernel_id: kernel},
        states={hidden.state_id: hidden},
        edge0=[Edge0("hidden", "cell", value_id="hidden_prev")],
        edge_delta=[EdgeDelta("cell", "hidden", lag_cycles=1, value_id="hidden_next")],
    )


def rolling_window_process() -> Process:
    """Reference graph for bounded window history and same-timestep compute."""

    kernel = _empty_kernel("feature_kernel")
    window = BufferSpec(
        buffer_id="rolling_window",
        dtype="float32",
        shape=(4,),
        depth=8,
        axes=("channel",),
    )
    return Process(
        process_id="reference_rolling_window",
        kernels={kernel.kernel_id: kernel},
        buffers={window.buffer_id: window},
        edge0=[Edge0("rolling_window", "feature_kernel", value_id="window")],
        edge_delta=[
            EdgeDelta("feature_kernel", "rolling_window", lag_cycles=1, value_id="x")
        ],
    )


def initialized_state_process() -> Process:
    """Reference graph for metadata-driven reset initialization."""

    kernel = _empty_kernel("stateful_kernel")
    state = StateSpec(
        state_id="running_mean",
        kind=StateKind.RUNNING_STAT,
        dtype="float32",
        shape=(1,),
        metadata={"initializer": [0.5]},
    )
    return Process(
        process_id="reference_initialized_state",
        kernels={kernel.kernel_id: kernel},
        states={state.state_id: state},
        edge0=[Edge0("running_mean", "stateful_kernel", value_id="mean_prev")],
        edge_delta=[
            EdgeDelta(
                "stateful_kernel",
                "running_mean",
                lag_cycles=1,
                value_id="mean_next",
            )
        ],
    )


def temporal_reference_processes() -> tuple[Process, ...]:
    """Return the canonical M1 reference temporal processes."""

    return (
        delay_line_process(),
        recurrent_state_process(),
        rolling_window_process(),
        initialized_state_process(),
    )


def _empty_kernel(kernel_id: str) -> Kernel:
    return Kernel(
        kernel_id=kernel_id,
        graph=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[]),
    )
