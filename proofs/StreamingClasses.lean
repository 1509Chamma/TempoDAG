/-
The streaming-latency dichotomy: machine-checked engines.

docs/streaming-latency-classes.md classifies temporal architectures by the
algebra of their state update. This file verifies the two engines the
dichotomy runs on:

  1. THE POSITIVE ENGINE (Theorem L): for any associative composition with
     identity, k elements combine to their ordered product in ⌈log₂ k⌉
     pairing rounds — each round one parallel layer. Instantiated at the
     monoid of affine state maps, this is why state-affine architectures
     have vanishing amortized streaming depth: `blocked_fold` proves the
     rounds compute exactly the sequential fold.

  2. THE DEGREE LEMMA (Theorem N's circuit half): an arithmetic expression
     of depth D has polynomial degree at most 2^D. Since k-fold composition
     of a state-squaring update has degree 2^k, matching it forces depth ≥
     k. The remaining (unformalized, classical) kernel is the
     function-to-polynomial identification — a nonzero polynomial has
     finitely many roots, so agreement on an infinite ring forces
     syntactic degree agreement. That step is documented in the paper
     proof and deliberately NOT claimed here; this file is sorry-free.

Together: affine feedback parallelizes at logarithmic depth; multiplicative
feedback provably cannot beat constant depth per step in the polynomial-
gate model.
-/
import IterationBound

set_option linter.unusedSectionVars false

namespace StreamingClasses

/-! ### The balanced-fold engine -/

variable {M : Type}

/-- Ordered product (sequential fold) of a list of monoid elements. -/
def prodList (op : M → M → M) (e : M) : List M → M
  | [] => e
  | a :: l => op a (prodList op e l)

/-- One parallel round: combine adjacent pairs. -/
def pairUp (op : M → M → M) : List M → List M
  | a :: b :: t => op a b :: pairUp op t
  | l => l

/-- Pairing preserves the ordered product (associativity alone). -/
theorem pairUp_prod (op : M → M → M) (e : M)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c)) :
    ∀ l, prodList op e (pairUp op l) = prodList op e l
  | [] => rfl
  | [_] => rfl
  | a :: b :: t => by
      simp only [pairUp, prodList]
      rw [pairUp_prod op e hassoc t, hassoc]

/-- Each round at least halves the list (exactly: ceiling halving). -/
theorem pairUp_length (op : M → M → M) :
    ∀ l : List M, (pairUp op l).length = (l.length + 1) / 2
  | [] => by simp [pairUp]
  | [_] => by simp [pairUp]
  | _ :: _ :: t => by
      simp only [pairUp, List.length_cons]
      rw [pairUp_length op t]
      omega

/-- Pairing never empties a nonempty list. -/
theorem pairUp_ne_nil (op : M → M → M) :
    ∀ {l : List M}, l ≠ [] → pairUp op l ≠ []
  | [], h => absurd rfl h
  | [_], _ => by simp [pairUp]
  | _ :: _ :: _, _ => by simp [pairUp]

/-- r parallel rounds of pairing. -/
def rounds (op : M → M → M) : Nat → List M → List M
  | 0, l => l
  | r + 1, l => rounds op r (pairUp op l)

theorem rounds_prod (op : M → M → M) (e : M)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c)) :
    ∀ (r : Nat) (l : List M),
      prodList op e (rounds op r l) = prodList op e l
  | 0, _ => rfl
  | r + 1, l => by
      simp only [rounds]
      rw [rounds_prod op e hassoc r (pairUp op l), pairUp_prod op e hassoc]

theorem rounds_ne_nil (op : M → M → M) :
    ∀ (r : Nat) {l : List M}, l ≠ [] → rounds op r l ≠ []
  | 0, _, h => h
  | r + 1, _, h => rounds_ne_nil op r (pairUp_ne_nil op h)

theorem rounds_length (op : M → M → M) :
    ∀ (r : Nat) (l : List M), l.length ≤ 2 ^ r →
      (rounds op r l).length ≤ 1
  | 0, l, h => by simpa [rounds] using h
  | r + 1, l, h => by
      simp only [rounds]
      refine rounds_length op r (pairUp op l) ?_
      rw [pairUp_length op]
      have hp : 2 ^ (r + 1) = 2 * 2 ^ r := by
        rw [Nat.pow_succ, Nat.mul_comm]
      omega

/-- **The balanced-fold theorem (Theorem L's engine).** With an
associative operation and two-sided identity, ⌈log₂ k⌉ parallel pairing
rounds reduce any nonempty list of k elements to exactly its ordered
product. Instantiated at affine state maps, a k-step block of a
state-affine recurrence composes in logarithmic depth — the vanishing
amortized streaming depth of class L. -/
theorem blocked_fold (op : M → M → M) (e : M)
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c))
    (hid : ∀ a, op a e = a)
    (r : Nat) (l : List M) (hne : l ≠ []) (hlen : l.length ≤ 2 ^ r) :
    rounds op r l = [prodList op e l] := by
  have h1 := rounds_length op r l hlen
  have h2 := rounds_ne_nil op r hne
  have h3 := rounds_prod op e hassoc r l
  match hres : rounds op r l with
  | [] => exact absurd hres h2
  | [a] =>
      rw [hres] at h3
      simp only [prodList] at h3
      rw [hid a] at h3
      rw [h3]
  | _ :: _ :: _ =>
      rw [hres] at h1
      simp at h1

/-! ### The degree lemma (Theorem N's circuit half) -/

/-- Arithmetic expressions in one state variable with ring constants. -/
inductive Expr where
  | var : Expr
  | const : Int → Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Circuit depth. -/
def depth : Expr → Nat
  | .var => 0
  | .const _ => 0
  | .add a b => 1 + max (depth a) (depth b)
  | .mul a b => 1 + max (depth a) (depth b)

/-- Polynomial degree in the state variable. -/
def deg : Expr → Nat
  | .var => 1
  | .const _ => 0
  | .add a b => max (deg a) (deg b)
  | .mul a b => deg a + deg b

private theorem two_pow_mono {m n : Nat} (h : m ≤ n) : 2 ^ m ≤ 2 ^ n :=
  Nat.pow_le_pow_right (by omega) h

/-- **Degree ≤ 2^depth.** Depth-D circuits compute polynomials of degree
at most 2^D in the state. Since the k-fold composition of a state map
containing a state × state product has degree 2^k, any circuit computing
it needs depth ≥ k — the irreducible sequential core of class N. (The
remaining classical step — function agreement over an infinite ring forces
polynomial agreement — is the documented unformalized kernel.) -/
theorem deg_le_two_pow_depth : ∀ e : Expr, deg e ≤ 2 ^ depth e
  | .var => by simp [deg, depth]
  | .const c => by simp [deg, depth]
  | .add a b => by
      have iha := deg_le_two_pow_depth a
      have ihb := deg_le_two_pow_depth b
      have ha := two_pow_mono (Nat.le_max_left (depth a) (depth b))
      have hb := two_pow_mono (Nat.le_max_right (depth a) (depth b))
      simp only [deg, depth]
      have hp : 2 ^ (1 + max (depth a) (depth b))
          = 2 * 2 ^ max (depth a) (depth b) := by
        rw [Nat.add_comm, Nat.pow_succ, Nat.mul_comm]
      omega
  | .mul a b => by
      have iha := deg_le_two_pow_depth b
      have ihb := deg_le_two_pow_depth a
      have ha := two_pow_mono (Nat.le_max_left (depth a) (depth b))
      have hb := two_pow_mono (Nat.le_max_right (depth a) (depth b))
      have iha' := deg_le_two_pow_depth a
      have ihb' := deg_le_two_pow_depth b
      simp only [deg, depth]
      have hp : 2 ^ (1 + max (depth a) (depth b))
          = 2 * 2 ^ max (depth a) (depth b) := by
        rw [Nat.add_comm, Nat.pow_succ, Nat.mul_comm]
      omega

end StreamingClasses
