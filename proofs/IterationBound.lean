/-
The iteration bound for temporal processes, machine-checked.

This file verifies the theorem stated in docs/scheduling-guarantees.md:
around any cycle of a temporal process, a periodic schedule with initiation
interval II satisfies  II * (total lag) ≥ (total latency).

Encoding. A cycle is a list of steps; step `⟨lat, lag⟩` is one dependency
edge on the cycle, carrying the producing operation's latency `lat` and the
delay-edge lag `lag` it crosses (0 for a same-timestep edge — the legality
contract guarantees at least one step of every cycle has lag ≥ 1). A
periodic schedule assigns each node on the cycle a start offset; with
period II, the dependency constraint between consecutive offsets `a`
(producer) and `b` (consumer) is

    a + lat ≤ b + II * lag

because the consumer runs `lag` iterations later, and each iteration shifts
every start by II. `Chain II a l c` says offsets can be threaded along the
step list `l` from `a` to `c` with every constraint satisfied; a CYCLE is a
chain that returns to its starting offset. Summing the constraints
telescopes the offsets away — that is `chain_bound` — and closing the cycle
yields the bound.

What this does and does not certify: it verifies the mathematics of the
bound — the statement any periodic schedule must obey. It does not model
the HLS scheduler, the emitter, or the FPGA toolchain; those are validated
empirically (docs/cost-model-validation.md).

Check with:  lean IterationBound.lean   (Lean 4, core only — no Mathlib)
-/

/-- One dependency step along a cycle: the producing operation's latency,
and the lag of the delay edge the dependency crosses (0 if same-timestep). -/
structure Step where
  lat : Int
  lag : Int

/-- Total operation latency around a step list (L in the paper statement). -/
def totalLat : List Step → Int
  | [] => 0
  | st :: l => st.lat + totalLat l

/-- Total delay-edge lag around a step list (Λ in the paper statement). -/
def totalLag : List Step → Int
  | [] => 0
  | st :: l => st.lag + totalLag l

/-- Offsets threaded along a step list under the periodic-schedule
constraint: `Chain II a l c` holds when start offsets exist along `l`,
beginning at `a` and ending at `c`, with every consecutive pair satisfying
`a + lat ≤ b + II * lag`. -/
inductive Chain (II : Int) : Int → List Step → Int → Prop
  | nil (a : Int) : Chain II a [] a
  | cons {a b c : Int} {st : Step} {l : List Step} :
      a + st.lat ≤ b + II * st.lag →
      Chain II b l c →
      Chain II a (st :: l) c

/-- Chain inequality: accumulated latency is dominated by the end offset
plus II-weighted accumulated lag. Proof: induction; each step composes its
constraint with the tail's inequality. -/
theorem chain_bound {II a c : Int} {l : List Step}
    (h : Chain II a l c) : a + totalLat l ≤ c + II * totalLag l := by
  induction h with
  | nil a => simp [totalLat, totalLag]
  | cons hs _ ih =>
      simp only [totalLat, totalLag] at *
      rw [Int.mul_add]
      omega

/-- **The iteration bound.** Around any closed cycle (a chain returning to
its starting offset), `II * Λ ≥ L`: the initiation interval times the total
lag dominates the total latency. The starting offset cancels by
telescoping. -/
theorem iteration_bound {II a : Int} {l : List Step}
    (h : Chain II a l a) : totalLat l ≤ II * totalLag l := by
  have hb := chain_bound h
  omega

/-- Strict form, giving the ceiling: any `k` with `k * Λ < L` satisfies
`k < II` — i.e. `II ≥ ⌈L / Λ⌉` when `Λ > 0`. -/
theorem iteration_bound_strict {II a k : Int} {l : List Step}
    (h : Chain II a l a) (hpos : 0 < totalLag l)
    (hk : k * totalLag l < totalLat l) : k < II := by
  have hb := iteration_bound h
  have hlt : k * totalLag l < II * totalLag l := Int.lt_of_lt_of_le hk hb
  exact Int.lt_of_mul_lt_mul_right hlt (Int.le_of_lt hpos)

/-! ## Graph-level form

The chain theorems above take the cycle as given. The statements below
quantify over a whole process: a graph is a list of edges over an arbitrary
node type, a periodic schedule assigns every node a start offset satisfying
each edge's constraint, and the bound then holds for every walk that
returns to its starting node — i.e. every cycle of the graph, not a cycle
someone chose. -/

/-- A dependency edge of a temporal process: producer, consumer, the
producer's latency, and the delay-edge lag (0 for same-timestep edges). -/
structure GEdge (V : Type) where
  src : V
  dst : V
  lat : Int
  lag : Int

/-- A periodic schedule with period `II`: start offsets for every node such
that every edge's constraint holds. -/
def ValidSchedule {V : Type} (II : Int) (s : V → Int)
    (E : List (GEdge V)) : Prop :=
  ∀ e ∈ E, s e.src + e.lat ≤ s e.dst + II * e.lag

/-- A walk through the edge list, tracking endpoints. -/
inductive Walk {V : Type} (E : List (GEdge V)) : V → V → List (GEdge V) → Prop
  | nil (v : V) : Walk E v v []
  | cons {u w : V} {e : GEdge V} {l : List (GEdge V)} :
      e ∈ E → e.src = u → Walk E e.dst w l → Walk E u w (e :: l)

def walkLat {V : Type} : List (GEdge V) → Int
  | [] => 0
  | e :: l => e.lat + walkLat l

def walkLag {V : Type} : List (GEdge V) → Int
  | [] => 0
  | e :: l => e.lag + walkLag l

theorem walk_bound {V : Type} {II : Int} {s : V → Int} {E : List (GEdge V)}
    (hs : ValidSchedule II s E) :
    ∀ {u w : V} {l : List (GEdge V)}, Walk E u w l →
      s u + walkLat l ≤ s w + II * walkLag l := by
  intro u w l hw
  induction hw with
  | nil v => simp [walkLat, walkLag]
  | cons he hsrc _ ih =>
      have hc := hs _ he
      rw [hsrc] at hc
      simp only [walkLat, walkLag]
      rw [Int.mul_add]
      omega

/-- **The iteration bound, graph form.** For any valid periodic schedule of
a process and ANY cycle of its graph, `II * Λ ≥ L`. -/
theorem iteration_bound_graph {V : Type} {II : Int} {s : V → Int}
    {E : List (GEdge V)} {v : V} {l : List (GEdge V)}
    (hs : ValidSchedule II s E) (hw : Walk E v v l) :
    walkLat l ≤ II * walkLag l := by
  have hb := walk_bound hs hw
  omega

/-- Instance: a single-step cycle with a 12-cycle loop latency through a
lag-1 delay edge (the matmul-in-loop class at H = 16) forces II ≥ 12. -/
example {II a : Int} (h : Chain II a [⟨12, 1⟩] a) : 12 ≤ II := by
  have hb := iteration_bound h
  simp [totalLat, totalLag] at hb
  omega

/-- Instance: a two-state cycle (the LSTM's h↔c coupling) — legs of
latency 11 and 3 through two lag-1 delay edges forces II ≥ 7 = ⌈14/2⌉. -/
example {II a : Int} (h : Chain II a [⟨11, 1⟩, ⟨3, 1⟩] a) : 7 ≤ II := by
  have hb := iteration_bound h
  simp [totalLat, totalLag] at hb
  omega
