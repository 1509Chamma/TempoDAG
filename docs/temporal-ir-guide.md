# Temporal IR Guide

## Purpose

The temporal IR layer represents streaming computations that evolve across
timesteps. It sits above the existing `tempo_dag.ir.Graph` layer rather than
replacing it.

Use `tempo_dag.ir.Graph` for a same-timestep tensor/dataflow region. Use
`tempo_dag.ir_temporal.Process` when a model needs persistent state, bounded
history, or feedback across timesteps.

## Core Concepts

### Process

`Process` is the top-level temporal container. It owns clocks, kernels, state,
buffers, and edges.

The initial implementation targets the single-clock case through the default
`main` clock, while keeping the data model ready for multiple clocks later.

### Kernel

`Kernel` wraps a regular `tempo_dag.ir.Graph`. Kernels are intended to describe
acyclic same-timestep computation, such as a linear projection, convolution, or
operator chain evaluated for the current input sample.

### State

`StateSpec` describes persistent values carried across timesteps. Current state
kinds are:

- `hidden_state` for recurrent cells such as RNN/GRU/LSTM state.
- `rolling_buffer` for windowed state attached to rolling operators.
- `running_stat` for accumulators such as rolling mean or variance.

### Buffer

`BufferSpec` describes bounded history storage. This covers delay lines, ring
buffers, and causal rolling windows. Buffers include an explicit `depth` so
later scheduling and HLS generation can reason about storage size.

### Edge0

`Edge0` is a same-timestep dependency. All `Edge0` dependencies inside a process
must form a DAG. If a cycle exists at the same timestep, validation fails.

### EdgeDelta

`EdgeDelta` is a positive-lag temporal dependency. Use it to represent feedback
from a previous timestep, such as `hidden[t - 1] -> cell[t]`.

`lag_cycles` must be at least `1`.

## Invariant

Same-timestep cycles are illegal. Feedback cycles must cross time through
`EdgeDelta`.

This keeps each timestep schedulable as an acyclic region while still allowing
streaming state and recurrence.

## Example

```python
from tempo_dag.ir.graph import Graph
from tempo_dag.ir_temporal import Edge0, EdgeDelta, Kernel, Process, StateKind, StateSpec

cell_graph = Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])

process = Process(
    process_id="gru_cell",
    kernels={"cell": Kernel(kernel_id="cell", graph=cell_graph)},
    states={
        "hidden": StateSpec(
            state_id="hidden",
            kind=StateKind.HIDDEN,
            dtype="float32",
            shape=(64,),
        )
    },
    edge0=[Edge0("hidden", "cell")],
    edge_delta=[EdgeDelta("cell", "hidden", lag_cycles=1)],
)

process.validate()
```

## Current Scope

This first implementation is structural. It validates process consistency and
the key temporal DAG invariant. Operator lowering, temporal ONNX parsing,
state-aware quantization, scheduling, and HLS generation build on this layer in
later roadmap steps.
