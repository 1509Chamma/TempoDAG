# The streaming-latency classes of temporal architectures

The [scheduling guarantees](scheduling-guarantees.md) bound what a given
*implementation* of a model can do. This page asks the deeper question:
what streaming latency is intrinsic to the *architecture itself* — to the
mathematics of its state update, before any implementation is chosen? The
answer is a dichotomy, provable in a standard circuit model, that
classifies temporal neural architectures the way complexity theory
classifies problems. The compiler applies the classification automatically
(`--explain` reports it), and the measured physical floors of the
[validation campaign](cost-model-validation.md) are its silicon shadow.

**Honest lineage first.** The negative half descends from classical
parallel-evaluation lower bounds (Kung, JACM 1976: nonlinear recurrences
admit no unbounded parallel speedup); the positive half is the scan/prefix
tradition (Ladner & Fischer 1980; Blelloch 1990). The contribution here is
the *classification*: stating both halves as one dichotomy over modern
neural state-update maps, deciding the class mechanically from a compiler
IR, machine-checking the positive engine and the degree lemma
([proofs/StreamingClasses.lean](../proofs/StreamingClasses.lean)), and
grounding each class in measured hardware.

## The model

Inputs arrive one per step. An implementation may use arbitrary parallel
hardware and a bounded pipeline lag: it may fall k steps behind and
process inputs in blocks of k. Its **amortized streaming depth** is the
critical-path depth needed per k-step block, divided by k, as k grows —
the fundamental "sequential work per sample" no parallelism can remove.
Write `f_x = f(·, x)` for the one-step state map at input x, and
`F_k = f_{x_k} ∘ … ∘ f_{x_1}` for the k-step block map.

## Theorem L — state-affine architectures have no latency floor

> If every per-step map `f_x` is **affine in the state**
> (`h ↦ A(x)·h + b(x)`), then `F_k` is computable in `O(log k)` depth, so
> the amortized streaming depth tends to **zero**.

*Proof sketch.* Affine maps form a monoid under composition with a
bounded-size representation `(A, b)` and bounded-depth composition
(`(A₂,b₂)∘(A₁,b₁) = (A₂A₁, A₂b₁+b₂)`). A balanced binary tree composes the
k per-step actions in ⌈log₂ k⌉ rounds; one application to the carried
state finishes the block. ∎ — The balanced-fold engine (associativity ⇒
tree-rebracketing with a per-round halving bound) is machine-checked in
`StreamingClasses.lean`.

Crucially, "affine in the state" permits arbitrary nonlinearity in the
**input**: gates like `a_t = σ(W x_t)` keep the map state-affine. That is
exactly the design move of the modern architecture generation.

## Theorem N — multiplicative state feedback has an irreducible core

> In circuits over a commutative ring with fan-in-2 {+, ×} gates and
> arbitrary constants, any circuit computing a `F_k` whose state-degree is
> `2^k` (e.g. any update containing a product of two state-dependent
> quantities, iterated) requires depth ≥ k. The amortized streaming depth
> is bounded below by a **constant per step**.

*Proof sketch.* A depth-D circuit computes a polynomial of degree ≤ 2^D
in the state variable (degree at most doubles per level — the machine-
checked degree lemma). The composed map has state-degree 2^k; agreeing
with it as a function over an infinite ring forces equal polynomials,
hence 2^D ≥ 2^k, i.e. D ≥ k. ∎ — **Machine-checked end to end**
([proofs/PolyKernel.lean](../proofs/PolyKernel.lean)): circuits denote
computable polynomials, the roots bound forces coefficient agreement, and
`depth_lower_bound` closes the chain — depth ≥ k for any circuit computing
the composed squaring map.

**Scope, fenced deliberately — read this before quoting the theorem.**
Every lower-bound statement here holds *within the polynomial-gate circuit
model* ({+, ×} gates with constants over a commutative ring) and concerns
*exact* evaluation of the composed state map. Three things the theorem
does NOT claim: (1) universality across computational models — rational
gates, table lookups, or other bases are outside the model, which is why
the model is named in the statement; (2) anything about ε-approximation —
approximating the trajectory arbitrarily well with a shallower parallel
circuit is a different (open) question, and parallel-in-time methods such
as Newton iterations live exactly there, additionally requiring future
inputs that streaming does not have; (3) analytic gates — real cells use
σ/tanh, the statement transfers to their polynomial surrogates, and the
analytic extension is a conjecture, not a claim. Finite-memory state
(delay lines) also escapes legitimately: nonlinearity over a bounded
window composes boundedly — the FF class below.

## The classification — decided by the compiler

The classifier inspects which states are **cyclic** (their read reaches
their own write, possibly through other states); acyclic delay-line state
acts as extended input. It then checks whether every cyclic-state update
is affine in the cyclic state.

**What "architecture" means here — the representation question, answered
precisely.** The classifier judges a *realization* (the IR graph), and the
two directions have different strengths, stated as such. An **L verdict is
constructive and certain**: the exhibited affine structure is itself the
input to the blocking rewrite, so the latency floor demonstrably vanishes
for this realization — no representation-dependence caveat needed. An
**N verdict is conservative**: it says this realization exposes no affine
structure, not that no exact finite-state affine realization of the same
function exists. Function-level hardness is established separately, by
reduction to the canonical family (Theorem N applies to the composed
squaring *function*, regardless of how it is realized in the model); an
obfuscated realization of a genuinely affine function would be classified
N by the compiler and simply miss an available optimization — a soundness
asymmetry, never an unsoundness. One observed invariance is worth
recording: the compiler's own rewrites preserve the class — hoisting never
touches cycle operations, and blocking maps affine realizations to affine
realizations (the monoid closure that powers Theorem L; the blocked cells
in the suite classify L).

| class | meaning | architectures (this suite) | measured physical floor |
|---|---|---|---|
| **FF** | no cyclic state — nonlinearity over a finite window only | temporal CNN | none (II=1, 5 ns/sample) |
| **L-affine** | cyclic updates affine in state; no intrinsic floor (Theorem L) | diagonal SSM, selective-SSM form, minGRU, QRNN, RWKV-form | ~4 ns loop delay — an implementation choice, and **measurably removed**: the blocked SSM streams at 0.5 cycles/sample amortized (2.5 ns/sample, co-simulation verified) |
| **N-nonlinear** | state × state products or gates reading the state; irreducible sequential core (Theorem N) | RNN, GRU, LSTM, JANET, FastGRNN, LiGRU, MGU, UGRNN, IndRNN, SRU | 10–16.6 ns loop delay — the intrinsic core made physical |

**Theorem L, demonstrated.** The blocking construction is not
hypothetical: composing k = 8 steps of the diagonal SSM into one exact
affine block yields a design with II = 4 per block of eight samples — an
**amortized initiation cost of 0.5 cycles per sample**, verified by C/RTL
co-simulation through the standard flow (`blockdiag8` in the
[validation campaign](cost-model-validation.md)). Worded carefully: this
is sustained throughput, not latency — each block still takes its cycles,
and each output waits for its block. What it demonstrates is that the
per-sample *initiation* cost of a class-L architecture is not floored at
one cycle, or at any constant: it is 4/k, an implementation dial. No
class-N architecture can ever receive this rewrite; that impossibility is
Theorem N.

Three readings worth pausing on:

1. **The modern architecture renaissance is this dichotomy, discovered
   empirically.** Mamba-family SSMs, minGRU, QRNN, RWKV keep their gates
   off the state path — placing themselves in class L — which is *why*
   they parallelize. The classification derives from the update's algebra
   what their designers reached by engineering.
2. **SRU sits in class N although it was designed for speed**: its gate
   reads the state (`v ⊙ c` inside a sigmoid). The classifier catches
   this, and the silicon agrees — its measured loop delay (15.1 ns) is
   N-class, not L-class.
3. **Implementation ≠ architecture.** The depth-sweep cells (chains of
   affine stages) show a perfect II staircase in this compiler — yet they
   classify L-affine: their staircase is a property of the current
   emitter, removable in principle by blocking. The iteration-bound
   theorems govern implementations; this page's dichotomy governs
   architectures. Both are true; they answer different questions.

## What this buys

For a model designer: a mechanical answer to "does my architecture have a
fundamental streaming-latency floor, or only an implementation one?" —
with the proof obligations split exactly along that line. For the
compiler: class L architectures are candidates for blocking rewrites that
no class-N architecture can ever receive; the classifier gates that
optimization soundly. For the theory: this is complexity theory made
*executable* — theorem → classifier → prediction → synthesized hardware →
verification, one pipeline, so latency claims about architectures become
falsifiable mathematics rather than benchmark folklore.
