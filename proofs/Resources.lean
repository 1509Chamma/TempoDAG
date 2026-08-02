/-
The throughput-resource bound, machine-checked.

The iteration bound limits how OFTEN a design can accept samples; this file
proves the dual limit on how much HARDWARE that rate requires. Model: a
periodic schedule with initiation interval II issues work in II phases;
each phase, every functional unit of a given kind completes at most one
operation of that kind, so a phase with U units completes at most U
operations. The theorem: the operations completed per sample cannot exceed
II · U — equivalently, executing M operations per sample requires
U ≥ ⌈M / II⌉ units.

Combined with the iteration bound, both axes of the design space carry
proved limits: II is bounded below by the recurrence (dependence), and
units are bounded below by M / II (throughput). The area-latency Pareto
frontier of docs/cost-model-validation.md lives between these two walls.

Scope note (stated honestly): "unit" is abstract. Vitis maps some
multiplications to DSP slices and others to LUT fabric, so measured DSP
counts can sit below the multiplier-unit bound while total multiplier
capacity cannot — the empirical comparison in the docs keeps that
distinction explicit.
-/
import IterationBound

set_option linter.unusedSectionVars false

namespace Resources

/-- Work completed in a period, phase by phase: `phases` lists, one per
cycle of the II window, each holding the operations that finish in that
cycle on the tracked unit kind. -/
def totalOps (phases : List (List α)) : Nat :=
  (phases.map List.length).sum

/-- **The resource bound.** If every phase completes at most `U`
operations (one per unit), a period of `II` phases completes at most
`II * U` operations. -/
theorem resource_bound {α : Type} (phases : List (List α)) (U : Nat)
    (hphase : ∀ p ∈ phases, p.length ≤ U) :
    totalOps phases ≤ phases.length * U := by
  induction phases with
  | nil => simp [totalOps]
  | cons p rest ih =>
      have hp := hphase p (List.mem_cons_self ..)
      have hrest : ∀ q ∈ rest, q.length ≤ U := fun q hq =>
        hphase q (List.mem_cons_of_mem _ hq)
      have := ih hrest
      simp only [totalOps, List.map_cons, List.sum_cons] at *
      calc p.length + (rest.map List.length).sum
          ≤ U + rest.length * U := Nat.add_le_add hp this
        _ = (rest.length + 1) * U := by
            rw [Nat.succ_mul, Nat.add_comm]
        _ = (p :: rest).length * U := by simp

/-- Contrapositive, the form used against a design: `M` operations per
sample cannot be completed by `U` units in `II` phases when `II * U < M`.
Hence any feasible design satisfies `U ≥ ⌈M / II⌉`. -/
theorem infeasible_of_undersized {α : Type} (phases : List (List α))
    (U : Nat) (hphase : ∀ p ∈ phases, p.length ≤ U)
    (hM : phases.length * U < totalOps phases) : False :=
  Nat.lt_irrefl _ (Nat.lt_of_lt_of_le hM (resource_bound phases U hphase))

end Resources
