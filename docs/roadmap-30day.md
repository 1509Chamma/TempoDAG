# TempoDAG 30-Day Roadmap & Implementation Checklist

**Vision**: Build a credible temporal IR compiler foundation with streaming state management and fixed-point verification that can compile a hybrid time-series pipeline (rolling statistics + TCN/GRU) into parity-verified Vitis HLS.

**Goal for Month 1**: Core temporal IR scaffolding + initial operator bridge + verification layer entry point.

---

## Week 1: Temporal IR Scaffolding & Architecture

### 1.1 Define Temporal IR Data Structures
- [ ] **Process**: Streaming component with one or more logical clocks
  - [ ] Single-clock entry point (common case)
  - [ ] Multi-clock support planned but deferred
- [ ] **Kernel**: Acyclic same-timestep tensor/dataflow region (reuse current IR)
- [ ] **State**: Persistent structured values
  - [ ] HiddenState (for RNN/GRU)
  - [ ] RollingBuffer (for windows/delays)
  - [ ] RunningStat (for rolling mean/var)
- [ ] **Buffer**: Bounded history (delay lines, ring buffers, rolling windows)
- [ ] **Edge0**: Same-timestep dependency (reuse current edge model)
- [ ] **EdgeDelta**: Positive-lag temporal dependency (new)
- [ ] Enforce invariant: same-timestep edges form DAG; cycles must use EdgeDelta

**Deliverable**: `src/tempo_dag/ir_temporal/` module with typed dataclasses and validation

### 1.2 Update Documentation to Reflect Temporal Vision
- [ ] Update `docs/architecture.md` to describe temporal layer above current kernel layer
- [ ] Add `docs/temporal-ir-guide.md`: temporal concepts and invariants
- [ ] Update `docs/development.md` with temporal operator patterns
- [ ] Review and update `docs/roadmap.md` to align with stage-based model progression

**Deliverable**: Developer-ready documentation for temporal IR mental model

### 1.3 Rename & Reorganize Package
- [ ] Verify all imports use `tempo_dag` ✓ (already done in redesign)
- [ ] Update documentation to promote TempoDAG brand (already done)
- [ ] Add platform-vision.md to docs ✓ (doing now)

**Deliverable**: Consistent branding across codebase and docs

---

## Week 2: Temporal Operator Bridge & State Lowering

### 2.1 Create Temporal Operator Interface
- [ ] Define `TemporalOperator` base class (wraps current `Operator`)
- [ ] Add `temporal_metadata()` method for state threading hints
- [ ] Implement for built-in ops: Add, MatMul (no state), Delay (stateful)
- [ ] Pattern detection: identify scan/loop structure in ONNX

**Deliverable**: `src/tempo_dag/ops/temporal_builtins.py` with Delay, RollingWindow, ScanCell

### 2.2 Implement Delay & RollingBuffer Operators
- [ ] `Delay(value, lag_cycles)`: output is input from N timesteps ago
- [ ] `RollingWindow(buffer, window_size)`: causal sliding window view
- [ ] `RollingMean`, `RollingVar`: streaming statistics
- [ ] Attach fixed-point range metadata to each

**Deliverable**: Working delay and windowing operators with tests

### 2.3 Extend Quantization Config for Temporal State
- [ ] Add `StateQuantSpec` dataclass (dtype, scale, overflow_policy)
- [ ] Update `NumericalParityConfig` to handle state across timesteps
- [ ] Create sample fixed-point profiles for rolling statistics

**Deliverable**: Quantization config extends to state values

---

## Week 3: Verification Foundation & Fixed-Point Oracle

### 3.1 Extend Numerical Parity to Temporal
- [ ] Create `TemporalParityAdapter` base class
- [ ] Implement `StreamingPyTorchAdapter`: run PyTorch model timestep-by-timestep
- [ ] Build `FixedPointOracle`: Python fixed-point reference implementation
- [ ] Trace per-timestep outputs and state

**Deliverable**: `src/tempo_dag/verification/temporal_parity.py`

### 3.2 Golden Trace Format & Serialization
- [ ] Define JSON schema for golden traces (timestep, inputs, state, outputs)
- [ ] Implement `GoldenTraceRecorder` (captures reference runs)
- [ ] Implement `GoldenTraceValidator` (compares hardware against trace)
- [ ] Add utilities for trace diffing and error reporting

**Deliverable**: `src/tempo_dag/verification/golden_trace.py`

### 3.3 Bit-Exact Verification Test Suite
- [ ] Write 5 verification test cases:
  - [ ] Simple delay chain (Delay → Delay → output)
  - [ ] Rolling mean over 4-timestep window
  - [ ] Running accumulator (stateful MAC)
  - [ ] Small GRU cell (state + non-linearity)
  - [ ] Hybrid: rolling stat + linear output
- [ ] Each test: PyTorch reference → temporal IR → fixed-point oracle → golden trace

**Deliverable**: `tests/verification/test_temporal_parity.py` with 5 golden traces

---

## Week 4: MVP Graph Compilation & Demo

### 4.1 Temporal Graph Lowering (ONNX → Temporal IR)
- [ ] Extend ONNX parser to detect scan patterns
- [ ] Pattern matching: rolling windows, state threading, recurrence
- [ ] Build `TemporalGraph` from ONNX with state and delay edges
- [ ] Validation: check DAG property on same-timestep edges

**Deliverable**: `src/tempo_dag/parsers/temporal_onnx.py`

### 4.2 Operator-Level HLS Generation
- [ ] Generate HLS for Delay (shift register or FIFO)
- [ ] Generate HLS for RollingWindow (ring buffer indexing)
- [ ] Generate HLS for RollingMean, RollingVar (accumulators + normalization)
- [ ] Attach testbench generation (golden trace → HLS stimulus)

**Deliverable**: HLS templates and renderer for temporal operators

### 4.3 End-to-End Demo Pipeline
- [ ] Compile: rolling-mean + causal Conv1D → temporal IR
- [ ] Quantize: fixed-point specs for state and activations
- [ ] Verify: golden trace from PyTorch reference
- [ ] Generate: Vitis HLS C++ with testbench
- [ ] Report: per-timestep error, state divergence, overflow checks

**Deliverable**: `examples/temporal_demo.py` with report

### 4.4 Documentation & Developer Guide
- [ ] Tutorial: "First Temporal Model" (rolling stats + conv)
- [ ] API reference for temporal IR and operators
- [ ] Verification guide: how to write golden traces
- [ ] HLS code generation walkthrough

**Deliverable**: `docs/temporal-quickstart.md`

---

## Cross-Cutting Deliverables (Throughout)

### Quality & Testing
- [ ] Unit tests for all temporal IR classes (validators, edge construction)
- [ ] Integration tests: ONNX parse → temporal IR → HLS → golden trace
- [ ] Linting with ruff (should pass already)
- [ ] Coverage: aim for >80% on temporal modules

### Documentation Quality
- [ ] Update all architecture diagrams to show temporal layer
- [ ] Add sequence diagrams for temporal scheduling
- [ ] Keep README in sync with implementation progress
- [ ] Inline docstrings for all temporal IR classes

### Git & Collaboration
- [ ] Weekly commits with clear messages linking to roadmap sections
- [ ] Maintain redesign branch in sync with changes
- [ ] PR template includes "Relates to roadmap section X.Y"

---

## Success Criteria (30-Day Checkpoint)

✅ **Temporal IR is implemented and validated**
- Process, Kernel, State, Buffer, Edge0, EdgeDelta classes exist
- DAG property enforced on same-timestep edges
- Full test coverage with realistic temporal graphs

✅ **First temporal operators work end-to-end**
- Delay, RollingWindow, RollingMean compile to HLS
- Fixed-point specs attached and verified
- 5 golden traces demonstrate parity

✅ **Verification foundation is in place**
- Fixed-point oracle computes correct references
- Golden traces recorded from PyTorch
- Bit-exact parity achieved on simple models

✅ **MVP demo is credible**
- Rolling statistics + Conv1D + linear head compiles
- Vitis HLS testbench generated
- Per-timestep error report shows <1 ULP drift

✅ **Developer experience is smooth**
- Quickstart guide gets someone to first compile in <30 min
- Temporal IR mental model is clear from docs
- Roadmap is transparent and up-to-date

---

## Stage 2+ Sketch (Post-30-Day)

Once MVP is solid, prioritize in this order:

1. **GRU Support** (Weeks 5-8)
   - Gate functions, state reset, scan integration
   - First neural flagship model

2. **Graph-Level Optimization** (Weeks 9-12)
   - Temporal scheduling (initiation interval, buffer depth)
   - State placement heuristics
   - Fusion of stateless chains

3. **Precision Automation** (Weeks 13-16)
   - Range analysis for fixed-point bit widths
   - Saturation/clipping diagnostics
   - Long-horizon drift prediction

4. **Attention & KV Cache** (Weeks 17-20)
   - Selective attention patterns
   - KV cache state management
   - Memory-aware buffer placement

---

## Key Decisions for Implementation

1. **Python-first for MVP**: Keep temporal IR in Python; MLIR migration is future work.
2. **ONNX focus**: Prioritize ONNX import; PyTorch FX/export as secondary path.
3. **Single-clock MVP**: Defer multi-clock support; assume single logical clock for first models.
4. **Exact optimization later**: Use greedy/heuristic scheduling for MVP; exact CP-SAT solver as next iteration.
5. **Fixed-point before floating**: Verification ladder starts with fixed-point oracle; float parity is post-MVP.

---

## Research Alignment

This roadmap synthesizes and operationalizes the key findings:

- **Finding 1** (Temporal DAG): Implemented in Week 1 with Edge0/EdgeDelta invariant.
- **Finding 2** (Stateful first-class): State/Buffer/Cache in Week 1-2; RollingBuffer/RollingMean in Week 2.
- **Finding 3** (Streaming optimization): Scheduling heuristics in Stage 2; initiation interval as primary metric.
- **Finding 4** (Fixed-point correctness): Fixed-point oracle and golden traces in Week 3; bit-exact gate in MVP.
- **Finding 5** (Prioritize streaming/DSP/GRU): MVP is rolling stats + conv (DSP-like); GRU in Stage 2.

---

## Blockers & Risks

- **Blocker**: If ONNX Scan semantics unclear → add raw PyTorch imperative DSL path
- **Risk**: Fixed-point oracle divergence from hardware → use HLS C simulation as bridge
- **Risk**: State placement heuristic is suboptimal → add constraint-based solver in Stage 2
- **Risk**: GRU lowering more complex than expected → scope to linear/additive first, defer non-linearity

---

## Budget Allocation (Rough)

- **IR & Architecture**: 25% (Week 1-2)
- **Verification & Traces**: 30% (Week 3)
- **MVP Compilation & Demo**: 35% (Week 4)
- **Docs & Testing**: 10% (throughout)
