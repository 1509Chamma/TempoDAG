# Cost-model validation

The compiler's central claim is that the temporal IR *predicts* the
synthesized hardware before any FPGA tool runs: the initiation interval
(**II** — how many clock cycles pass between accepting consecutive samples,
which sets the time per prediction) from the structure of the state-feedback
cycle, and DSP usage (the FPGA's multiplier budget) from a weighted count of
the model's multiplications. This page is the evidence for that claim:
**26 designs** covering a hidden-size sweep, an input-width sweep, a depth
test, and seven recurrent cells from the published literature — plus floor
probes that reveal what the cost model's invariant physically is.
(Background on why II is the quantity that matters:
[the explainer](explainer.md).)

**Protocol.** Every prediction is computed from the IR and written to the
results log *before* its synthesis runs, so nothing can be tuned after the
fact. The II model has four structural classes read off the recurrence
chain; the DSP model is a single coefficient (0.86 DSP per weighted MAC)
calibrated on one anchor (RNN, H=16) and never refit. Each design
synthesizes at its predicted II (Vitis HLS 2026.1, KV260 part, 5 ns clock,
Q6.12); C-simulation asserts numerics against a fixed-point oracle;
co-simulation verifies the RTL. Design tags read as cell + hidden size
(`gru32` = a GRU with H=32) with `f` marking input width (`gruf8` = GRU,
16 hidden, 8 inputs). Any row can be reproduced with
`python research/cost_model_validation.py --only <tag>`; the raw
prediction/measurement log is
`research/results/cost_model_validation.jsonl`.

## The four II classes

| class | what is on the state cycle | II law |
|---|---|---|
| feed-forward | nothing — no delay-edge cycle at all | 1 (resource-bound only) |
| elementwise | multiply + add only | 4 |
| elementwise + activation | elementwise + a LUT activation | raw chain sum (5–8) |
| matmul-in-loop | a hidden-size matmul + activation + blend | 8 + log2(H) |
| serial-gated | a gate that must resolve *before* the state matmul | chain sum − 1 |

The class laws are instances of a provable scheduling bound — the iteration
bound and its corollaries are stated and proved in
[scheduling-guarantees.md](scheduling-guarantees.md).

## Main results

Bold = measured equals prediction. DSP error is prediction vs. measurement.

| design | H | class | II pred | II meas | DSP pred | DSP meas | err | cosim |
|---|---:|---|---:|---:|---:|---:|---:|---|
| tcn16 (temporal CNN) | 16 | feed-forward | 1 | **1** | 564 | 604 | -6.9% | PASS |
| gatedssm16 (selective-SSM form) | 16 | elementwise | 4 | **4** | 151 | 162 | -6.8% | PASS |
| rwkv16 (RWKV-style num/den) | 16 | elementwise | 4 | **4** | 165 | 180 | -8.3% | PASS |
| diag8 | 8 | elementwise | 4 | **4** | 41 | 45 | -8.9% | PASS |
| diag16 | 16 | elementwise | 4 | **4** | 83 | 90 | -7.8% | PASS |
| diag32 | 32 | elementwise | 4 | **4** | 165 | 183 | -9.8% | PASS |
| diag64 | 64 | elementwise | 4 | **4** | 330 | 366 | -9.8% | csim pass |
| qrnn16 | 16 | elementwise | 4 | **4** | 261 | 286 | -8.7% | PASS |
| mingru16 | 16 | elementwise | 4 | **4** | 151 | 166 | -9.0% | PASS |
| indrnn16 | 16 | elementwise+act | 5 | **5** | 83 | 90 | -7.8% | csim pass |
| sru16 | 16 | elementwise+act | 8 | **8** | 317 | 352 | -9.9% | PASS |
| rnn8 | 8 | matmul-in-loop | 11 | **11** | 89 | 100 | -11.0% | PASS |
| gru8 | 8 | matmul-in-loop | 11 | **11** | 275 | 308 | -10.7% | PASS |
| lstm8 | 8 | matmul-in-loop | 11 | **11** | 358 | 390 | -8.2% | csim pass |
| rnn16 | 16 | matmul-in-loop | 12 | **12** | 289 | 313 | -7.7% | PASS |
| gru16 | 16 | matmul-in-loop | 12 | **12** | 881 | 951 | -7.4% | PASS |
| lstm16 | 16 | matmul-in-loop | 12 | **12** | 1156 | 1230 | -6.0% | PASS |
| janet16 | 16 | matmul-in-loop | 12 | **12** | 592 | 637 | -7.1% | PASS |
| fastgrnn16 | 16 | matmul-in-loop | 12 | **12** | 330 | 340 | -2.9% | PASS |
| ligru16 | 16 | matmul-in-loop | 12 | **12** | 592 | 614 | -3.6% | PASS |
| ugrnn16 | 16 | matmul-in-loop | 12 | **12** | 592 | 642 | -7.8% | csim pass |
| gru2l16 (2-layer) | 16 | matmul-in-loop | 12 | **12** | 2243 | 2422 | -7.4% | csim pass |
| gruf8 (F=8) | 16 | matmul-in-loop | 12 | **12** | 1046 | 1123 | -6.9% | PASS |
| rnnf16 (F=16) | 16 | matmul-in-loop | 12 | **12** | 454 | 489 | -7.2% | PASS |
| rnn32 | 32 | matmul-in-loop | 13 | **13** | 1018 | 1075 | -5.3% | PASS |
| mgu16 | 16 | serial-gated | 21 | 20 | 606 | 640 | -5.3% | csim pass |
| grurb16 | 16 | serial-gated | 21 | 20 | 881 | 951 | -7.4% | csim pass |
| gruf16 (F=16) | 16 | matmul-in-loop | 12 | timeout¹ | 1376 | 1450² | -5.1%² | csim pass² |
| gru32 | 32 | matmul-in-loop | 13 | timeout¹ | 3083 | — | — | — |
| rnn64 | 64 | matmul-in-loop | 14 | timeout¹ | 3798 | — | — | — |

¹ synthesis exceeded a 90-minute scheduling budget — see "the scheduler-cost
cliff" below. ² measured at a relaxed target (II=14), which did complete —
see below.

**Score: 26 of 29 designs measured; 23 match their predicted II exactly.**
The architecture set spans 1997 (LSTM) to 2024 (minGRU, the selective-SSM
recurrence form, an RWKV-style recurrence) — the modern constant-state
generation lands in the fast class precisely because its designers kept
weights off the state path, which is the structural property the model
reads.
The feed-forward row is the generalization test: a two-layer causal dilated
temporal CNN (Bai et al. 2018) whose window lives in delay-line states —
no cycle, so the bound vanishes and the design streams at one sample per
clock (5 ns/sample), co-simulation verified.
The three unmeasured designs all belong to one family whose synthesis
exceeds a time budget (explained under "the scheduler-cost cliff"); the
three rows that measured differently from their prediction each carry a
lesson rather than noise:

- **The serial-gated law is chain − 1.** Both serial-gated cells (MGU and
  reset-before GRU) measured exactly one cycle under the raw chain sum,
  independently — the state-write register absorbs one operation.
- **Achieved-at-target proves achievability, not tightness.** One probe
  deliberately targeted II=8 on the RNN — below its class prediction of
  12 — and the scheduler *achieved 8*. Cycle-count targets are operating
  points, not floors, which is what the floor probes below measure
  directly.

## The published-cell zoo

Seven cells from the literature, each landing in the class its structure
dictates: **QRNN** (Bradbury et al. 2016) and **minGRU** (Feng et al. 2024)
at II=4, **SRU** (Lei et al. 2018) at II=8, **JANET** (van der Westhuizen &
Lasenby 2018), **FastGRNN** (Kusupati et al. 2018), and **LiGRU**
(Ravanelli et al. 2018) at II=12, and **MGU** (Zhou et al. 2016) in the
serial-gated class. The split is the interesting part: the cells that were
*designed* for fast or parallel execution are exactly the ones whose gates
avoid the state path — the cost model doesn't just measure that they are
fast, it identifies the structural reason.

## Invariance results

- **Hidden size H moves II logarithmically**: 11 / 12 / 13 measured at
  H = 8 / 16 / 32 (the 8 + log2 H law, confirmed in both directions), while
  the elementwise class stays flat at 4 from H=8 to H=64.
- **Input width F does not move II**: F = 4 → 16 at constant II=12 (RNN,
  exact), F = 4 → 8 for GRU (exact); GRU at F=16 is bounded at II ≤ 14 by
  the scheduling budget.
- **Depth does not move II**: a 2-layer stacked GRU holds II=12, because the
  cross-layer edge is feed-forward, not a cycle — at 2.3× the DSP.
- **Context length does not enter the design at all** (see
  [benchmarks](benchmarks.md): flat 60 ns against a window-based tool's
  linear growth).

## Floor probes: the invariant is time, not cycles

Targeting II=1 and letting the scheduler relax finds each design's true
floor — and the answer reframes the model. The scheduler reached II=1–2
everywhere, but the *estimated clock* absorbed the recurrence depth:

| design | floor II | est. clock | per-sample time |
|---|---:|---:|---:|
| diag16 | 1 | 4.04 ns | **4.0 ns** |
| mingru16 | 1 | 4.04 ns | **4.0 ns** |
| indrnn16 | 1 | 10.09 ns | 10.1 ns |
| rnn16 | 1 | 12.88 ns | 12.9 ns |
| sru16 | 1 | 15.13 ns | 15.1 ns |
| gru16 | 2 | 16.65 ns | 33.3 ns |
| mgu16 | 2 | 16.63 ns | 33.3 ns |

The conserved quantity is the **physical delay of the state-dependence
path** — ~4 ns for the elementwise class, ~10–15 ns with a matmul or
activation in the loop, ~16.6 ns for gated cells. The cycle-class model is
the projection of this invariant onto a 5 ns clock; the class ordering is
identical in both views. Practical consequences: the compiler has a measured
(II × clock) trade-off surface per class, and the elementwise class reaches
**4 ns/sample** single-stream. Caveat: floor-probe clocks are C-synthesis
estimates, and long combinational paths are precisely what place-and-route
punishes; the pipelined 5 ns operating points remain the RTL-verified ones.

## The scheduler-cost cliff (an honest boundary)

Three designs exceeded a 90-minute scheduling budget at their predicted
targets on an otherwise idle machine: gruf16, gru32, rnn64. Two behaviors
separate cleanly:

- **Tightness-limited**: gruf16 *does* complete at a slightly relaxed
  target (II=14, ~85 minutes, clock closes, DSP within the usual band). Its
  minimum II is therefore ≤14 and undetermined at 12 — scheduling cost, not
  hardware, is the limit.
- **Size-limited**: gru32 and rnn64 (1024-wide × 3 and 4096-wide reduction
  trees) do not complete at any tried target. The fully-unrolled
  balanced-tree emission scales through ~1,200-MAC single-layer loops and a
  2,608-MAC two-layer design, but not to these shapes; wide dense cells
  need a partially-shared matmul form. This is the emitter's current
  practical boundary, stated as measured.

## Resource bounds: the other axis

Latency is only half a cost model. The dual limit — machine-checked in
[proofs/Resources.lean](../proofs/Resources.lean) — is that a design
executing M multiply-accumulates per sample at initiation interval II
needs at least ⌈M / II⌉ multiplier units: each unit finishes at most one
operation per cycle. Comparing every measured design against its bound:

| class (II) | designs | DSP / bound | reading |
|---|---|---:|---|
| feed-forward (1) | tcn16 | **0.9×** | at the frontier; slightly under the DSP count because ~8% of multiplies map to LUT fabric rather than DSP slices |
| elementwise (4) | diag8–64, qrnn, mingru, gatedssm, rwkv | 3.7–3.8× | |
| elementwise+act (5–8) | indrnn, sru | 4.5× / 7.7× | |
| matmul-in-loop (11–13) | rnn/gru/lstm family | 10.0–11.7× | |
| serial-gated (20) | mgu, grurb | 17.8–18.3× | |

The pattern is exact enough to state as a law: **measured DSP ≈ 0.92 · M,
independent of II** — the fully-unrolled emitter allocates one multiplier
per operation and therefore sits a factor ≈ II above the proved frontier.
That is a deliberate trade (unrolled datapaths make schedules trivial and
enable C-slow stream sharing), and it quantifies the headroom precisely: a
resource-shared emitter mode could approach the ⌈M/II⌉ wall — e.g. a GRU
at H=16 in ~86 DSPs instead of 951 at the same 60 ns/sample — which is the
measured, bounded scope of that future work.

## Systematic DSP bias

Every measured design under-predicts DSP by 2.9–11% (median ≈ −7.5%). The
model counts matmul and elementwise-multiply MACs only; the uncounted
remainder (output head, state blending, control) is a small, consistent
overhead. The bias is reported as-is rather than refit away — the model's
value is that a single anchor calibrates the entire table, and a reader
applying it should simply expect predictions to run a few percent low.

**Where the 0.86 DSP-per-MAC coefficient comes from.** A Q6.12 multiply
(18-bit operands) fits a single DSP slice, so the natural coefficient is
1.0. It measures below 1.0 because the emitter bakes weights as constants,
and the synthesis tool implements multiplication by "cheap" constants (few
significant bits) in LUT fabric instead of DSP slices — the same absorption
that puts the II=1 temporal CNN slightly under its multiplier-unit bound.
The coefficient is therefore a property of the constant-baking emission
style, with a testable consequence stated rather than assumed: designs with
runtime-loaded weights should measure near 1.0 DSP per MAC.
