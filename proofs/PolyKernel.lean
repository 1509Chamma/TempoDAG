/-
Polynomial semantics for circuit expressions — Theorem N, closed.

This file completes the streaming-latency dichotomy's negative half,
sorry-free, in three layers:

  Layer 1: every expression denotes a polynomial with computable
  coefficients (`peval (coeffs e) x = evalE e x`), via verified
  coefficient-list addition, scaling, and multiplication.

  Layer 2: the Horner-remainder factorization p(x) = (x−a)·q(x) + p(a),
  the roots bound (vanishing on more distinct points than length forces
  vanishing everywhere), and coefficient extraction (function-zero forces
  coefficient-zero).

  Layer 3: length accounting (coefficient count ≤ 2^depth + 1) assembling
  into `depth_lower_bound`: ANY circuit computing the k-fold composed
  squaring map x ↦ x^(2^k) — the canonical multiplicative state feedback,
  iterated — has depth ≥ k. Multiplicative state feedback provably costs
  depth linear in steps; state-affine feedback (StreamingClasses.lean's
  balanced fold) provably does not.
-/
import StreamingClasses

set_option linter.unusedSectionVars false

namespace PolyKernel

open StreamingClasses

/-- Semantic evaluation of a circuit expression at a state value. -/
def evalE : Expr → Int → Int
  | .var, x => x
  | .const c, _ => c
  | .add a b, x => evalE a x + evalE b x
  | .mul a b, x => evalE a x * evalE b x

/-- Polynomial as a coefficient list (index = power), Horner evaluation. -/
def peval : List Int → Int → Int
  | [], _ => 0
  | c :: p, x => c + x * peval p x

def padd : List Int → List Int → List Int
  | [], q => q
  | p, [] => p
  | a :: p, b :: q => (a + b) :: padd p q

theorem peval_padd : ∀ (p q : List Int) (x : Int),
    peval (padd p q) x = peval p x + peval q x
  | [], q, x => by simp [padd, peval]
  | _ :: _, [], x => by simp [padd, peval]
  | a :: p, b :: q, x => by
      simp only [padd, peval]
      rw [peval_padd p q x, Int.mul_add]
      omega

def pscale (c : Int) : List Int → List Int
  | [] => []
  | a :: p => c * a :: pscale c p

theorem peval_pscale : ∀ (c : Int) (p : List Int) (x : Int),
    peval (pscale c p) x = c * peval p x
  | c, [], x => by simp [pscale, peval]
  | c, a :: p, x => by
      simp only [pscale, peval]
      rw [peval_pscale c p x, Int.mul_add, Int.mul_left_comm]

def pmul : List Int → List Int → List Int
  | [], _ => []
  | a :: p, q => padd (pscale a q) (0 :: pmul p q)

theorem peval_pmul : ∀ (p q : List Int) (x : Int),
    peval (pmul p q) x = peval p x * peval q x
  | [], q, x => by simp [pmul, peval]
  | a :: p, q, x => by
      simp only [pmul]
      rw [peval_padd, peval_pscale]
      simp only [peval]
      rw [peval_pmul p q x, Int.add_mul, ← Int.mul_assoc]
      omega

/-- The coefficient extraction: circuits to polynomials, computably. -/
def coeffs : Expr → List Int
  | .var => [0, 1]
  | .const c => [c]
  | .add a b => padd (coeffs a) (coeffs b)
  | .mul a b => pmul (coeffs a) (coeffs b)

/-- **Every expression denotes its coefficient polynomial.** The semantic
bridge: circuit evaluation and Horner evaluation of the extracted
coefficients agree at every point. -/
theorem peval_coeffs : ∀ (e : Expr) (x : Int),
    peval (coeffs e) x = evalE e x
  | .var, x => by simp [coeffs, peval, evalE]
  | .const c, x => by simp [coeffs, peval, evalE]
  | .add a b, x => by
      simp only [coeffs, evalE]
      rw [peval_padd, peval_coeffs a x, peval_coeffs b x]
  | .mul a b, x => by
      simp only [coeffs, evalE]
      rw [peval_pmul, peval_coeffs a x, peval_coeffs b x]

/-! ### Layer 2: Horner division, the roots bound, and Theorem N's close -/

/-- Polynomial subtraction. -/
def psub (p q : List Int) : List Int := padd p (pscale (-1) q)

theorem peval_psub (p q : List Int) (x : Int) :
    peval (psub p q) x = peval p x - peval q x := by
  unfold psub
  rw [peval_padd, peval_pscale]
  omega

/-- Horner quotient: `p(x) = (x − a) · (hq p a)(x) + p(a)`. The quotient of
a length-(n+1) polynomial is length n. -/
def hq : List Int → Int → List Int
  | [], _ => []
  | [_], _ => []
  | _ :: p', a => peval p' a :: hq p' a

theorem horner_rem : ∀ (p : List Int) (a x : Int),
    peval p x = (x - a) * peval (hq p a) x + peval p a
  | [], a, x => by simp [hq, peval]
  | [c], a, x => by simp [hq, peval]
  | c :: d :: p', a, x => by
      have ih := horner_rem (d :: p') a x
      simp only [hq, peval] at *
      rw [ih]
      -- expand both sides over the atoms
      rw [Int.mul_add, Int.mul_add, Int.mul_add]
      have h1 : x * ((x - a) * peval (hq (d :: p') a) x)
          = (x - a) * (x * peval (hq (d :: p') a) x) := by
        rw [← Int.mul_assoc, ← Int.mul_assoc, Int.mul_comm x (x - a)]
      have h2 : (x - a) * (d + a * peval p' a)
          = x * (d + a * peval p' a) - a * (d + a * peval p' a) := by
        rw [Int.sub_mul]
      have h3 : x * (d + a * peval p' a)
          = x * d + x * (a * peval p' a) := by rw [Int.mul_add]
      have h4 : a * (d + a * peval p' a)
          = a * d + a * (a * peval p' a) := by rw [Int.mul_add]
      omega

theorem hq_length : ∀ (c : Int) (p' : List Int) (a : Int),
    (hq (c :: p') a).length = p'.length
  | _, [], _ => rfl
  | _, d :: p'', a => by
      simp only [hq, List.length_cons]
      rw [hq_length d p'' a]

/-- All-zero coefficient lists evaluate to zero. -/
theorem peval_zero : ∀ {p : List Int}, (∀ c ∈ p, c = 0) →
    ∀ x, peval p x = 0
  | [], _, x => rfl
  | c :: p, h, x => by
      have hc := h c (List.mem_cons_self ..)
      have ht := peval_zero (fun d hd => h d (List.mem_cons_of_mem _ hd)) x
      simp only [peval, hc, ht]
      omega

/-- **The roots bound.** A polynomial vanishing on more distinct points
than its length vanishes as a function everywhere. -/
theorem vanish_of_roots : ∀ (p : List Int) (xs : List Int),
    xs.Nodup → p.length ≤ xs.length →
    (∀ x ∈ xs, peval p x = 0) → ∀ y, peval p y = 0
  | [], _, _, _, _, _ => rfl
  | c :: p', xs, hnd, hlen, hz, y => by
      cases xs with
      | nil => simp at hlen
      | cons a xs' =>
          have hpa : peval (c :: p') a = 0 :=
            hz a (List.mem_cons_self ..)
          have hnd' := List.nodup_cons.mp hnd
          -- the quotient vanishes on the remaining points
          have hq_zero : ∀ x ∈ xs', peval (hq (c :: p') a) x = 0 := by
            intro x hx
            have hne : x ≠ a := fun hEq => hnd'.1 (hEq ▸ hx)
            have hpx := hz x (List.mem_cons_of_mem _ hx)
            have hrem := horner_rem (c :: p') a x
            rw [hpa] at hrem
            rw [hpx] at hrem
            have hfac : (x - a) * peval (hq (c :: p') a) x = 0 := by omega
            rcases Int.mul_eq_zero.mp hfac with h0 | h0
            · exact absurd (by omega : x = a) hne
            · exact h0
          have hlen' : (hq (c :: p') a).length ≤ xs'.length := by
            rw [hq_length]
            simp only [List.length_cons] at hlen
            omega
          have hvan := vanish_of_roots (hq (c :: p') a) xs'
            hnd'.2 hlen' hq_zero
          have hrem := horner_rem (c :: p') a y
          rw [hpa, hvan y] at hrem
          simpa using hrem

/-- 1..n as distinct nonzero points. -/
private def pts (n : Nat) : List Int :=
  (List.range n).map (fun (i : Nat) => (i : Int) + 1)

private theorem nodup_map_of_inj {α β : Type} (f : α → β)
    (hf : ∀ a b, f a = f b → a = b) :
    ∀ l : List α, l.Nodup → (l.map f).Nodup
  | [], _ => by simp
  | a :: l, h => by
      have hc := List.nodup_cons.mp h
      refine List.nodup_cons.mpr ⟨?_, nodup_map_of_inj f hf l hc.2⟩
      intro hmem
      obtain ⟨b, hb, hfb⟩ := List.mem_map.mp hmem
      exact hc.1 (hf b a hfb ▸ hb)

private theorem pts_nodup (n : Nat) : (pts n).Nodup := by
  refine nodup_map_of_inj _ ?_ _ (List.nodup_range)
  intro a b h
  omega

private theorem pts_length (n : Nat) : (pts n).length = n := by
  simp [pts]

private theorem pts_pos {n : Nat} {x : Int} (hx : x ∈ pts n) : x ≠ 0 := by
  simp only [pts, List.mem_map, List.mem_range] at hx
  obtain ⟨i, _, rfl⟩ := hx
  omega

/-- **Function-zero forces coefficient-zero.** -/
theorem coeffs_zero_of_vanish : ∀ (p : List Int),
    (∀ x, peval p x = 0) → ∀ c ∈ p, c = 0
  | [], _, c, hc => absurd hc (by simp)
  | c0 :: p', hv, c, hc => by
      have hc0 : c0 = 0 := by
        have := hv 0
        simpa [peval] using this
      have hp' : ∀ y, peval p' y = 0 := by
        -- p' vanishes on the nonzero points 1..(len p' + 1), enough for
        -- the roots bound
        refine vanish_of_roots p' (pts (p'.length + 1)) (pts_nodup _)
          (by rw [pts_length]; omega) ?_
        intro x hx
        have hx0 : x ≠ 0 := pts_pos hx
        have := hv x
        simp only [peval, hc0] at this
        have hfac : x * peval p' x = 0 := by omega
        rcases Int.mul_eq_zero.mp hfac with h | h
        · exact absurd h hx0
        · exact h
      rcases List.mem_cons.mp hc with rfl | hmem
      · exact hc0
      · exact coeffs_zero_of_vanish p' hp' c hmem

/-! ### Layer 3: assembly — the depth lower bound -/

theorem pscale_length (c : Int) : ∀ p : List Int,
    (pscale c p).length = p.length
  | [] => rfl
  | _ :: p => by simp [pscale, pscale_length c p]

theorem padd_length : ∀ p q : List Int,
    (padd p q).length = max p.length q.length
  | [], q => by simp [padd]
  | _ :: _, [] => by simp [padd]
  | a :: p, b :: q => by
      simp only [padd, List.length_cons]
      rw [padd_length p q]
      omega

theorem padd_getLast? : ∀ (p q : List Int), q.length < p.length →
    (padd p q).getLast? = p.getLast?
  | [], q, h => by simp at h
  | _ :: _, [], _ => by simp [padd]
  | a :: p, b :: q, h => by
      simp only [List.length_cons] at h
      have hlt : q.length < p.length := by omega
      have hpne : p ≠ [] := by
        intro hp; rw [hp] at hlt; simp at hlt
      have hne : padd p q ≠ [] := by
        intro hc
        have := padd_length p q
        rw [hc] at this
        simp at this
        omega
      simp only [padd]
      rw [List.getLast?_cons_of_ne_nil hne,
          List.getLast?_cons_of_ne_nil hpne]
      exact padd_getLast? p q hlt

theorem mem_of_getLast? : ∀ (l : List Int) (x : Int),
    l.getLast? = some x → x ∈ l
  | [], x, h => by simp at h
  | [a], x, h => by
      simp at h
      simp [h]
  | a :: b :: l, x, h => by
      have hne : b :: l ≠ [] := by simp
      rw [List.getLast?_cons_of_ne_nil hne] at h
      exact List.mem_cons_of_mem _ (mem_of_getLast? (b :: l) x h)

/-- The monomial x^n as a coefficient list. -/
def monomial (n : Nat) : List Int := List.replicate n 0 ++ [1]

theorem monomial_length (n : Nat) : (monomial n).length = n + 1 := by
  simp [monomial]

theorem monomial_getLast? (n : Nat) : (monomial n).getLast? = some 1 := by
  simp [monomial]

theorem peval_monomial : ∀ (n : Nat) (x : Int),
    peval (monomial n) x = x ^ n
  | 0, x => by simp [monomial, peval]
  | n + 1, x => by
      have : monomial (n + 1) = 0 :: monomial n := by
        simp [monomial, List.replicate_succ]
      rw [this]
      simp only [peval]
      rw [peval_monomial n x]
      have hp : x ^ (n + 1) = x ^ n * x := Int.pow_succ x n
      rw [hp, Int.mul_comm (x ^ n) x]
      omega

theorem padd_ne_nil_left : ∀ p q : List Int, p ≠ [] → padd p q ≠ []
  | [], _, h => absurd rfl h
  | _ :: _, [], _ => by simp [padd]
  | _ :: _, _ :: _, _ => by simp [padd]

theorem padd_ne_nil_right : ∀ p q : List Int, q ≠ [] → padd p q ≠ []
  | [], _, h => by simpa [padd] using h
  | _ :: _, [], h => absurd rfl h
  | _ :: _, _ :: _, _ => by simp [padd]

theorem coeffs_ne_nil : ∀ e : Expr, coeffs e ≠ []
  | .var => by simp [coeffs]
  | .const c => by simp [coeffs]
  | .add a b => by
      simp only [coeffs]
      exact padd_ne_nil_left _ _ (coeffs_ne_nil a)
  | .mul a b => by
      simp only [coeffs]
      cases hca : coeffs a with
      | nil => exact absurd hca (coeffs_ne_nil a)
      | cons c p =>
          simp only [pmul]
          exact padd_ne_nil_right _ _ (by simp)

theorem pmul_length : ∀ (p q : List Int), q ≠ [] →
    (pmul p q).length ≤ p.length + q.length - 1
  | [], q, _ => by simp [pmul]
  | a :: p, q, hq => by
      simp only [pmul]
      rw [padd_length, pscale_length]
      have hql : 1 ≤ q.length := by
        cases q with
        | nil => exact absurd rfl hq
        | cons _ _ => simp
      have := pmul_length p q hq
      simp only [List.length_cons]
      omega

private theorem two_pow_mono {m n : Nat} (h : m ≤ n) : 2 ^ m ≤ 2 ^ n :=
  Nat.pow_le_pow_right (by omega) h

/-- Depth bounds coefficient count: `len (coeffs e) ≤ 2^depth + 1`. -/
theorem coeffs_length : ∀ e : Expr, (coeffs e).length ≤ 2 ^ depth e + 1
  | .var => by simp [coeffs, depth]
  | .const c => by simp [coeffs, depth]
  | .add a b => by
      have iha := coeffs_length a
      have ihb := coeffs_length b
      have ha := two_pow_mono
        (Nat.le_max_left (depth a) (depth b))
      have hb := two_pow_mono
        (Nat.le_max_right (depth a) (depth b))
      simp only [coeffs, depth]
      rw [padd_length]
      have hp : 2 ^ (1 + max (depth a) (depth b))
          = 2 * 2 ^ max (depth a) (depth b) := by
        rw [Nat.add_comm, Nat.pow_succ, Nat.mul_comm]
      omega
  | .mul a b => by
      have iha := coeffs_length a
      have ihb := coeffs_length b
      have ha := two_pow_mono
        (Nat.le_max_left (depth a) (depth b))
      have hb := two_pow_mono
        (Nat.le_max_right (depth a) (depth b))
      have hml := pmul_length (coeffs a) (coeffs b) (coeffs_ne_nil b)
      have hbn : 1 ≤ (coeffs b).length := by
        cases hcb : coeffs b with
        | nil => exact absurd hcb (coeffs_ne_nil b)
        | cons _ _ => simp
      simp only [coeffs, depth]
      have hp : 2 ^ (1 + max (depth a) (depth b))
          = 2 * 2 ^ max (depth a) (depth b) := by
        rw [Nat.add_comm, Nat.pow_succ, Nat.mul_comm]
      omega

/-- **The depth lower bound (Theorem N, closed).** Any circuit computing
the k-fold composed squaring map `x ↦ x^(2^k)` — the canonical
multiplicative state feedback iterated k steps — has depth at least k.
Chain: circuits denote polynomials (layer 1); a circuit agreeing with the
monomial everywhere has a coefficient list matching it coefficient-wise
(layer 2: roots bound); the monomial's top coefficient forces the list
past length 2^k; depth bounds coefficient count (layer 3). -/
theorem depth_lower_bound {e : Expr} {k : Nat}
    (h : ∀ x : Int, evalE e x = x ^ (2 ^ k)) : k ≤ depth e := by
  -- the difference monomial − coeffs(e) vanishes everywhere
  have hvan : ∀ x, peval (psub (monomial (2 ^ k)) (coeffs e)) x = 0 := by
    intro x
    rw [peval_psub, peval_monomial, peval_coeffs, h x]
    omega
  have hzero := coeffs_zero_of_vanish _ hvan
  -- so coeffs e must reach the monomial's top entry
  rcases Nat.lt_or_ge (depth e) k with hdlt | hge
  · exfalso
    have hclen := coeffs_length e
    have hpow : 2 ^ depth e < 2 ^ k := by
      have h2 := two_pow_mono (hdlt : depth e + 1 ≤ k)
      have h3 : 2 ^ (depth e + 1) = 2 * 2 ^ depth e := by
        rw [Nat.pow_succ, Nat.mul_comm]
      have h4 : 0 < 2 ^ depth e := Nat.pow_pos (by omega)
      omega
    have hshort :
        (pscale (-1) (coeffs e)).length < (monomial (2 ^ k)).length := by
      rw [pscale_length, monomial_length]
      omega
    have hlast : (psub (monomial (2 ^ k)) (coeffs e)).getLast? = some 1 := by
      unfold psub
      rw [padd_getLast? _ _ hshort, monomial_getLast?]
    have hone := mem_of_getLast? _ _ hlast
    have := hzero 1 hone
    omega
  · exact hge

end PolyKernel
