/-
C-slow interleaving is bit-exact, machine-checked.

C-slowing shares one physical datapath among N independent streams by
issuing them round-robin. The hardware claim (docs/benchmarks.md,
~200M samples/s aggregate) rests on a semantic fact: interleaving changes
only the ORDER in which independent streams' steps execute, and since each
stream reads and writes only its own state, every stream computes exactly
what it would have computed running alone.

The model here: a step function `f : S → I → S`, a state vector indexed by
stream id, and an arbitrary event list of (stream, input) pairs — strictly
more general than round-robin, so the theorem covers any interleaving.
The theorem: after executing the events, each stream's component equals a
plain fold of `f` over that stream's own inputs in their original order.
Bit-exactness is equality of the resulting states — nothing is
approximated.

The hardware-level premise this rests on — N ≥ II so that a stream's
feedback value is ready before its next turn — is a scheduling fact
(covered by the iteration-bound machinery), not a semantic one; the
semantics proved here is what makes the shared datapath's answers exact.
-/
import IterationBound

set_option linter.unusedSectionVars false

namespace CSlow

variable {V S I : Type} [DecidableEq V]

/-- Update one component of a state vector. -/
def upd (σ : V → S) (i : V) (s : S) : V → S :=
  fun j => if j = i then s else σ j

/-- Execute an interleaved event list: each event advances exactly the
stream it names. -/
def exec (f : S → I → S) (σ : V → S) : List (V × I) → (V → S)
  | [] => σ
  | (i, x) :: evs => exec f (upd σ i (f (σ i) x)) evs

/-- A stream's own inputs, in order, from the event list. -/
def inputsOf (i : V) : List (V × I) → List I
  | [] => []
  | (j, x) :: evs => if j = i then x :: inputsOf i evs else inputsOf i evs

/-- **C-slow exactness.** Under any interleaving of independent streams,
each stream's final state is the plain sequential fold of its own inputs —
exactly the state it reaches running alone on a dedicated datapath. -/
theorem exec_eq_foldl (f : S → I → S) (evs : List (V × I))
    (σ : V → S) (i : V) :
    exec f σ evs i = List.foldl f (σ i) (inputsOf i evs) := by
  induction evs generalizing σ with
  | nil => rfl
  | cons ev evs ih =>
      obtain ⟨j, x⟩ := ev
      by_cases hji : j = i
      · subst hji
        simp only [exec, inputsOf, List.foldl_cons]
        rw [ih]
        simp [upd]
      · simp only [exec, inputsOf, if_neg hji]
        rw [ih]
        have : upd σ j (f (σ j) x) i = σ i := by
          simp [upd]
          intro h
          exact absurd h.symm hji
        rw [this]

end CSlow
