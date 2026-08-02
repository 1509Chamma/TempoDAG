/-
Temporal error accumulation, machine-checked.

Fixed-point deployment introduces a bounded per-step rounding error `q`.
The question that decides whether streaming inference is safe is whether
those errors ACCUMULATE over time. This file proves the stability law for
contractive state updates: if each step scales the carried error by
ρ (with 0 ≤ ρ) and adds at most q of fresh error, then any B satisfying
ρ·B + q ≤ B is an invariant bound — the error never exceeds B at any
horizon. For ρ < 1 the smallest such B is the classical steady state
q / (1 − ρ), reached here without division: B qualifies exactly when
(1 − ρ)·B ≥ q.

Everything is stated over integers — read them as fixed-point-scaled
quantities (error and q in LSBs, ρ as a scaled fraction after clearing
denominators), which matches how the compiler's oracle actually measures.
The empirical counterpart lives in docs/accuracy-retention.md; the
contraction factors themselves are properties of trained models and are
measured, not assumed.
-/
import IterationBound

set_option linter.unusedSectionVars false

namespace ErrorBounds

/-- **Error stability under contraction.** If the error sequence starts
within `B`, each step obeys `e (t+1) ≤ ρ · e t + q` with `ρ ≥ 0`, and `B`
absorbs one step (`ρ·B + q ≤ B`), then the error stays within `B`
forever — rounding does not accumulate past the steady bound. -/
theorem contraction_bound {ρ q B : Int} (e : Nat → Int)
    (hρ : 0 ≤ ρ) (h0 : e 0 ≤ B)
    (hstep : ∀ t, e (t + 1) ≤ ρ * e t + q)
    (hB : ρ * B + q ≤ B) : ∀ t, e t ≤ B := by
  intro t
  induction t with
  | zero => exact h0
  | succ t ih =>
      have h1 : ρ * e t ≤ ρ * B := Int.mul_le_mul_of_nonneg_left ih hρ
      have h2 := hstep t
      omega

/-- The steady-state form: `B` absorbs a step exactly when
`(1 − ρ) · B ≥ q` — the division-free statement of `B ≥ q / (1 − ρ)`. -/
theorem absorbs_iff {ρ q B : Int} : ρ * B + q ≤ B ↔ q ≤ (1 - ρ) * B := by
  have hexp : (1 - ρ) * B = B - ρ * B := by
    rw [Int.sub_mul, Int.one_mul]
  constructor
  · intro h; omega
  · intro h; omega

end ErrorBounds
