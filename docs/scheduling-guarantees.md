# Scheduling guarantees

The empirical results in [cost-model-validation.md](cost-model-validation.md)
are instances of properties that can be stated and proved. This page gives
the formal core: what the temporal IR guarantees by construction, which
classical theorem the cost model rests on, and exactly what is proved versus
measured. The intellectual honesty matters: the central bound is a classical
result from the dataflow-scheduling literature, restated for this IR — the
contribution is an IR whose legality contract makes the bound statically
computable on ML models, a compiler that constructively achieves it, and the
measured validation.

**Why existing dataflow frameworks don't already do this.** The IR's
ancestry is synchronous dataflow (Lee & Messerschmitt 1987), and the bound
is the classical iteration bound — but no SDF toolchain compiles a trained
PyTorch GRU into verified fixed-point hardware with a pre-stated initiation
interval. What is missing from those frameworks, and present here: machine-
learning operator semantics (matmul/gate/activation nodes with shape and
dtype checking), typed state kinds (hidden state vs. rolling buffers vs.
running statistics) under a legality contract, fixed-point legality
reasoning with oracle-relative certificates, ingestion from PyTorch/ONNX/
Keras, and the closed loop from static analysis through code generation to
RTL-level verification — with the analysis itself certified against the
theorem ([proofs/](../proofs/)). The claim is the system, not the graph
formalism.

## Definitions

A **temporal process** is a finite set of operations and values forming a
graph with two edge kinds: same-timestep edges, and **delay edges**, each
with an integer lag ℓ ≥ 1 meaning "the value written at timestep t is read
at timestep t + ℓ". The IR's **legality contract**, enforced by
`Process.validate()`, is:

> Restricted to same-timestep edges, the graph is acyclic. Every cycle
> passes through at least one delay edge.

Each operation `v` has a latency `lat(v) ≥ 0` in clock cycles. A **schedule**
assigns operation `v` at timestep `t` a start cycle `s(v, t)` respecting all
dependencies; a schedule has **initiation interval II** if it is periodic
with `s(v, t+1) = s(v, t) + II`. For a cycle `C`, write `L(C)` for the sum of
operation latencies around `C` and `Λ(C)` for the sum of delay-edge lags
around `C`.

## Theorem (iteration bound)

> For any temporal process and any valid periodic schedule,
>
> **II ≥ max over cycles C of ⌈ L(C) / Λ(C) ⌉.**

*Proof.* Fix a cycle `C` and follow its dependency chain across timesteps:
starting from any operation on `C` at timestep `t`, traversing `C` once ends
at the same operation at timestep `t + Λ(C)`, and the chain's operations
take at least `L(C)` cycles in sequence. Traversing `C` k times gives a
dependency chain of length `k·L(C)` between timesteps `t` and `t + k·Λ(C)`.
A periodic schedule separates those two timesteps' start cycles by
`k·Λ(C)·II`, so `k·Λ(C)·II ≥ k·L(C)` for all k, hence `II ≥ L(C)/Λ(C)`;
integrality of II gives the ceiling. ∎

This is the classical *iteration bound* of the dataflow-scheduling
literature (Reiter 1968; Renfors & Neuvo 1981; textbook treatment in Parhi,
*VLSI Digital Signal Processing Systems*). The theorem — **and its converse**:
any II meeting every cycle's bound admits a valid schedule, so the bound is
exactly the frontier of the possible — is **machine-checked in Lean 4 in
both directions**, along with per-design certificates instantiating it for
every design in the validation suite; see [proofs/](../proofs/). What the temporal IR adds is that
the bound is **statically computable on machine-learning models**: because
the legality contract forces every cycle through typed delay edges, the
binding cycle is found by a graph search over state read→write paths — the
`loop_chain` analysis in `research/cost_model_validation.py` — with no
scheduling, synthesis, or simulation in the loop.

## Corollary 1 — structural invariance

> Any change to the process that neither adds a cycle nor alters an existing
> cycle's operations leaves the iteration bound unchanged.

Immediate from the theorem: the bound depends only on the set of cycles.
This single statement is the formal content of three measured results:

- **Input-width invariance** — widening `x` grows only the off-cycle input
  projections (measured: F = 4 → 16 at constant II).
- **Depth invariance** — stacking layers adds cross-layer edges that are
  feed-forward, never cyclic (measured: 2-layer GRU at the 1-layer II).
- **Context-window independence** — history carried in acyclic buffers or
  compressed into state adds no cycle, so per-sample cost is flat in
  window length (measured against a window-unrolling toolflow whose cost
  grows linearly).

## Corollary 2 — feed-forward networks

> A process with no delay-edge cycle has iteration bound 1: II is limited
> only by resource constraints, never by dependencies.

Temporal CNNs are the canonical case: a causal dilated convolution reads a
window of past values, which the IR carries as delay-line states written by
pure copies — no arithmetic on any cycle, in fact no cycle at all. The
tcn16 design in the validation suite is this corollary's test article, with
II = 1 as its pre-registered prediction.

## Corollary 3 — composition

> If two processes are connected only by feed-forward edges (the output of
> one feeding the input of the other), the composite's iteration bound is
> the maximum of the components' bounds.

Immediate from Corollary 1: composition adds no cycle, so the composite's
cycle set is the union of the components'. This is what licenses building
pipelines — feature extraction → model → decision logic, or stacked model
layers — without re-deriving anything: the slowest stage's recurrence sets
the streaming rate, and everything else overlaps. (Measured instance: the
2-layer GRU at the 1-layer II.)

## Theorem 2 — the resource bound (the other axis)

> A periodic schedule executing M operations of one kind per sample at
> initiation interval II requires at least ⌈M / II⌉ functional units of
> that kind.

*Proof.* Each unit completes at most one such operation per cycle, so II
cycles complete at most II·U; feasibility forces II·U ≥ M. ∎
(Machine-checked: [proofs/Resources.lean](../proofs/Resources.lean).
Empirical comparison across the suite: the resource section of the
[validation campaign](cost-model-validation.md).) Together the two theorems
wall off both axes of the design space: dependence bounds the rate,
throughput bounds the area, and every implementation lives between them.

## Theorem 3 — error stability under contraction

> If each step scales the carried fixed-point error by ρ ≥ 0 and adds at
> most q, then any B with ρ·B + q ≤ B bounds the error at every horizon —
> for ρ < 1, the classical steady state q/(1 − ρ).

*Proof.* Induction: e(t+1) ≤ ρ·e(t) + q ≤ ρ·B + q ≤ B. ∎ (Machine-checked,
including the division-free steady-state characterization:
[proofs/ErrorBounds.lean](../proofs/ErrorBounds.lean). The contraction
factors of trained models are measured quantities; the empirical
counterpart is [accuracy retention](accuracy-retention.md).)

## Property 3 — C-slow interleaving is exact (machine-checked)

> Under ANY interleaving of independent streams through one datapath, each
> stream's final state equals the plain sequential fold of its own inputs —
> bit-exact to running alone.

Machine-checked for arbitrary event orders (strictly more general than
round-robin) in [proofs/CSlow.lean](../proofs/CSlow.lean): stream state
components are disjoint, so events commute across streams while each
stream's own order is preserved. The hardware-level premise N ≥ II — each
stream's feedback ready before its next turn — is the scheduling fact
covered by Theorem 1. (Measured: aggregate II = 1 at N = II, the ~200M
samples/s figure in [benchmarks](benchmarks.md).)

## Remark — the theorem in physical time

The floor probes (validation campaign) showed the conserved quantity is
the recurrence cycle's combinational delay in nanoseconds, with cycle
counts as its projection onto a clock choice. No new mathematics is needed
to cover this: the machine-checked bound is stated over integers with
unspecified units, so instantiating latencies in picoseconds (any time
quantum) yields the delay-form bound II·T ≥ D(C)/Λ(C) from the same
theorem. Cycles and nanoseconds are two unit choices for one proved
statement.

## Property 4 — hoisting is exact

> Moving a state-independent subgraph (e.g. the input projections W·x) off
> the recurrence cycle preserves every computed value bit-for-bit.

*Argument.* Dataflow semantics: a value depends only on its operands, not on
where its operation is scheduled. Hoisting reorders computation but changes
no operation, operand, or arithmetic order within any value's computation. ∎
(Verified numerically to 0 ulp over 256 samples in walkthrough 2.)

## What is proved vs. measured — the honest boundary

- **Proved**: the lower bound (theorem), its invariance, feed-forward and
  composition corollaries, the exactness of C-slow and hoisting — and the
  **converse**: in the abstract scheduling model, any II meeting every
  cycle's bound admits a valid schedule (machine-checked construction in
  [proofs/Achievability.lean](../proofs/Achievability.lean)). The bound is
  therefore exactly the frontier of the model's possibilities.
- **Model vs. toolchain**: that a *particular HLS tool* reaches the bound
  within a practical time budget remains empirical. The emitter reaches
  the predicted II on 23 of 26 synthesized designs; the exceptions are a
  measured scheduler-cost cliff (documented in the
  [validation campaign](cost-model-validation.md)) — a statement about
  synthesis cost, which the existence proof deliberately does not cover.
- **Measured**: the physical-delay refinement (the floor probes' result that
  the conserved quantity is the cycle's combinational delay in nanoseconds,
  with cycle counts as its projection onto a clock period) is an empirical
  characterization of the tool flow, not a theorem.
