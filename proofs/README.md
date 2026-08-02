# Machine-checked proofs

This folder contains mathematical proofs verified by the
[Lean 4](https://lean-lang.org/) proof assistant — a program that checks
every logical step of a proof mechanically, so the results below do not
depend on trusting anyone's algebra. Everything builds with Lean's core
library alone (no extra dependencies) via one command:

```bash
cd proofs
lake build      # checks every theorem and all 29 design certificates
```

## What is proved, in plain terms

**1. The speed limit is real** (`IterationBound.lean`). A streaming model's
hardware contains feedback loops — places where this step's result feeds
into the next step's computation. The theorem says: however cleverly a
schedule is arranged, the time between accepting consecutive samples can
never be smaller than the work inside a feedback loop divided by how many
steps that loop spans. In short: *you can parallelize everything except
waiting for your own previous answer.* This holds for every loop of a whole
design at once, not just a chosen one.

**2. The speed limit is exactly achievable** (`Achievability.lean`). The
converse, and the harder half: whenever a proposed rate respects every
loop's limit, a schedule running at that rate *actually exists* — the proof
builds one. So the limit is not a pessimistic estimate; it is precisely the
frontier between possible and impossible. (The construction: repeatedly
"relax" each dependency, Bellman-Ford style; the key step shows a schedule
conflict would require a trip around some loop to gain time, which the
hypothesis forbids — any over-long dependency chain must revisit a node,
and the revisited loop can be cut out without losing anything.)

**3. The hardware-size limit is real too** (`Resources.lean`). Speed has a
price in silicon: doing M multiplications per sample at a given rate needs
at least M-divided-by-the-interval multiplier units — each unit can only
finish one per clock tick. Together with result 1, both walls of the
design space are proved: how fast a design may run, and how small it may
be at that speed. (The measured comparison across all designs is in the
[validation campaign](../docs/cost-model-validation.md) — the current
generator deliberately sits above this wall to keep schedules simple, and
the proof quantifies exactly how much could be reclaimed.)

**4. Rounding errors don't snowball** (`ErrorBounds.lean`). Fixed-point
hardware rounds a little at every step; the worry is that a stream running
forever accumulates unbounded error. The theorem: if the model's state
update shrinks carried error (as trained, stable models do), the total
error stays under a fixed ceiling forever — the classical q/(1−ρ) bound,
proved without ever dividing.

**5. Sharing the datapath changes nothing** (`CSlow.lean`). One physical
circuit can serve many independent streams by taking turns. The theorem:
under *any* turn order, every stream ends in exactly the state it would
have reached on its own private hardware — bit for bit. This is what makes
the ~200M samples/s aggregate figure a matter of arithmetic rather than
approximation.

**6. Architectures themselves split into two worlds**
(`StreamingClasses.lean`). Some networks' state updates are affine — for
those, combining k steps takes only ~log k sequential layers (the
balanced-fold theorem), so no fundamental per-sample latency floor exists.
Networks that multiply state-dependent values cannot escape: circuit depth
bounds polynomial degree (the degree lemma), and their composed updates
grow degree exponentially, forcing depth to grow with every step. The
compiler decides which world an architecture lives in automatically — see
[the classification](../docs/streaming-latency-classes.md).

**7. The compiler's own predictions check out** (`Certificates.lean`).
For each of the 29 designs in the
[validation campaign](../docs/cost-model-validation.md), the compiler's
analysis finds the feedback loop that limits its speed. Those 29 loops are
exported into this folder automatically, and each one's predicted limit is
re-derived *inside Lean* as an instance of theorem 1. If the compiler's
analysis ever mis-identifies a loop or mis-adds its latencies, this build
fails. Regenerate after changing the analysis with:

```bash
python research/cost_model_validation.py --emit-lean-certs
```

## What this does and does not certify

Lean verifies the *mathematics*: the speed limit, its exact achievability,
and the arithmetic of the compiler's per-design predictions under the
analysis' declared latency table. It does not model AMD's synthesis tools
or the physical chip — those are validated empirically, by C/RTL
co-simulation and place-and-route, in the
[benchmarks](../docs/benchmarks.md) and the
[validation campaign](../docs/cost-model-validation.md). The two halves
meet in the middle: the proofs say no schedule can beat the bound, and the
measurements show the generated hardware reaches it.

The toolchain is pinned (`lean-toolchain`, Lean 4.32.2); `elan` fetches it
automatically on first build.
