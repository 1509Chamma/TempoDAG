# Temporal Execution And HLS Contract

This document is the implementation contract for TempoDAG temporal processes.
It defines how a `Process` executes over time and how temporal dependencies
lower into FPGA-oriented HLS storage and interfaces.

The contract is intentionally conservative. Optimizers and backends may improve
the schedule or storage layout, but they must preserve these observable
semantics.

## Scope

This contract applies to `tempo_dag.ir_temporal.Process` and the first Vitis
HLS-oriented backend. It covers:

- Reset, warm-up, steady-state, flush, and stream termination behavior.
- State and buffer read/write ordering.
- `Edge0` and `EdgeDelta` execution meaning.
- Legal lowering of temporal dependencies to HLS storage.
- The minimum generated HLS block interface expected by later backends.

## Temporal Process Model

A `Process` is evaluated as a sequence of timesteps:

```text
t = 0, 1, 2, ...
```

At each timestep, the process consumes zero or more stream inputs, reads
persistent state and bounded history, evaluates one or more same-timestep
kernels, produces outputs, and commits state or buffer updates for future
timesteps.

The initial implementation assumes a single logical clock named `main`.
Multiple clocks are represented in the IR but are not yet part of the execution
contract.

## Component Semantics

### Kernel

A `Kernel` wraps an acyclic same-timestep `tempo_dag.ir.Graph`.

At timestep `t`, every kernel observes values for timestep `t` plus any
explicitly connected state or buffer views. A kernel must not observe writes
that are committed later in the same timestep unless an explicit `Edge0`
dependency defines that same-timestep value flow.

### State

`StateSpec` represents persistent values such as hidden state, running
statistics, or recurrent cell state.

At timestep `t`:

1. State reads observe the committed state from before timestep `t` begins.
2. Kernels may compute next-state values during timestep `t`.
3. State writes commit after all same-timestep computation for `t` completes.
4. The committed value becomes visible to timestep `t + 1` and later.

This two-phase read/commit behavior prevents accidental same-timestep feedback
loops through state.

### Buffer

`BufferSpec` represents bounded history such as a delay line, ring buffer, or
rolling window.

At timestep `t`:

1. Buffer reads observe samples committed before timestep `t`.
2. New samples are appended or written during timestep `t`.
3. Buffer pointer and storage updates commit after same-timestep computation.
4. The new history is visible starting at timestep `t + 1`.

Buffers with insufficient history during warm-up must use the documented
initialization policy in metadata or the backend default of zero-fill.

## Edge Semantics

### `Edge0`

`Edge0(source, target)` is a same-timestep dependency:

```text
source[t] -> target[t]
```

All `Edge0` dependencies in a process must form a DAG. This makes every
timestep schedulable without requiring unbounded combinational feedback.

Legal HLS lowerings include:

- C++ local variables for scalar or small tensor values.
- Wires or direct function-call arguments in generated code.
- HLS streams inside a `DATAFLOW` region when producer and consumer are
  separate processes or functions.

`Edge0` must not be lowered to persistent storage unless the storage is purely
an implementation detail and does not add visible latency.

### `EdgeDelta`

`EdgeDelta(source, target, lag_cycles=n)` is a positive-lag dependency:

```text
source[t - n] -> target[t]
```

`lag_cycles` must be at least `1`. Any feedback cycle in the full temporal graph
must include at least one `EdgeDelta`.

Legal HLS lowerings include:

- Registers for lag-one scalar state.
- Shift registers for short fixed delays.
- FIFOs for stream-aligned delays.
- Ring buffers for rolling windows or bounded history.
- BRAM, URAM, or external memory for larger histories.

The chosen storage must preserve ordering, lag, dtype, shape, and reset
behavior.

## Execution Phases

### Reset

Reset initializes every state and buffer before timestep `0`.

Default reset policy:

- Numeric state values reset to zero unless metadata provides an initializer.
- Buffers reset to zero-filled history unless metadata provides an initializer.
- Write pointers reset to the first logical position.

Generated HLS must expose reset behavior either as an explicit reset argument,
an initialization function, or documented static initialization in the
testbench flow.

### Warm-Up

Warm-up covers timesteps where delayed dependencies or rolling windows do not
yet have enough real history.

During warm-up:

- `EdgeDelta` reads beyond available history use the reset/initial history.
- Buffers report valid history according to their metadata or zero-fill policy.
- Outputs may be marked warm-up in reports if they depend on incomplete
  history.

Warm-up must be deterministic and reproducible in golden traces.

### Steady State

Steady state begins once every delay, buffer, and rolling window has enough
committed history to satisfy its maximum lag or depth.

Optimization metrics such as initiation interval and throughput should be
reported primarily for steady state.

### Flush

Flush drains any internally buffered output after the final input timestep.

The first backend may report `flush_cycles = 0` for purely combinational or
state-update-only pipelines. Future backends with HLS streams or multi-stage
dataflow must report flush cycles explicitly.

### Stream Termination

The process terminates after:

1. All input timesteps have been consumed.
2. All required state and buffer commits have completed.
3. All flush outputs have been produced.

Reports and testbenches must state the number of input timesteps, warm-up
timesteps, flush cycles, and checked output timesteps.

## HLS Block Interface

The baseline generated HLS block for a temporal process should expose a stable
step-style interface:

```cpp
void process_step(/* stream inputs, stream outputs, reset/control */);
```

Minimum interface requirements:

- One call represents one logical timestep unless a generated report says
  otherwise.
- Reset behavior is explicit in the generated testbench or function interface.
- Persistent state and buffer storage are owned by the generated block unless a
  backend explicitly emits external state ports.
- The generated testbench must replay the same golden trace used by the
  fixed-point oracle.

Backends may later emit wider streaming interfaces, `hls::stream` ports,
AXI-style ports, or external memory interfaces, but those forms must preserve
the same logical timestep contract.

## Optimizer Obligations

Every optimizer pass must preserve:

- `Edge0` acyclicity within each timestep.
- Positive lag for every temporal feedback path.
- State read-before-commit behavior.
- Buffer read-before-update behavior.
- Dtype, shape, quantization metadata, and parameter identity.
- Reset, warm-up, steady-state, flush, and termination observability.

Any optimization that changes latency must update the generated schedule and
report so parity checking uses the correct timestep alignment.

## Reference Processes

The canonical M1 reference processes live in
`tempo_dag.examples.reference_processes`:

- `delay_line_process()` exercises a fixed positive-lag dependency.
- `recurrent_state_process()` exercises legal recurrence through lag-one state.
- `rolling_window_process()` exercises buffer warm-up and ring-buffer storage.
- `initialized_state_process()` exercises metadata-driven reset behavior.

These examples are intentionally small so scheduler, optimizer, HLS, and
verification tests can share the same semantic fixtures.
