/-
Achievability of the iteration bound — COMPLETE (no sorry).

`IterationBound.lean` proves no periodic schedule can beat the cycle bound.
This file proves the converse: if II satisfies every cycle's bound, a valid
periodic schedule EXISTS — so the bound is not merely a limit but exactly
the frontier of the possible.

The construction is longest-walk potentials over the reduced weights
w(e) = lat(e) − II·lag(e), computed by Bellman-Ford value iteration
(`bestUpTo`). The proof chain:

  * `bestUpTo` is characterized by walks in both directions: soundness
    (`walk_le_bestUpTo`) and witness (`bestUpTo_witness`).
  * A walk longer than the node pool repeats a node (`nodup_length_le`,
    `not_nodup_decomp`); the enclosed cycle has reduced weight ≤ 0 under
    the no-positive-cycle hypothesis, so stripping it gives a strictly
    shorter, no-lighter walk (`strip_step`) — hence every walk's weight is
    matched within length |nodesOf E| (`exists_short_walk`).
  * Therefore the iteration stabilizes at K = |nodesOf E|
    (`stabilizes_of_noPosCycle`), and a stable round satisfies every edge
    constraint (`valid_of_stabilizes`), giving `achievability`:

        NoPosCycle II E → ∃ s, ValidSchedule II s E
-/
import IterationBound

set_option linter.unusedSectionVars false

namespace Achievability

variable {V : Type} [DecidableEq V]

/-- Integer max, defined locally so every lemma reduces by `split <;> omega`
without relying on library lemma names. -/
def imax (a b : Int) : Int := if a ≤ b then b else a

theorem le_imax_l (a b : Int) : a ≤ imax a b := by
  unfold imax; split <;> omega

theorem le_imax_r (a b : Int) : b ≤ imax a b := by
  unfold imax; split <;> omega

/-- Reduced edge weight at period II. -/
def rw (II : Int) (e : GEdge V) : Int := e.lat - II * e.lag

/-- Reduced weight of a walk. -/
def walkRW (II : Int) : List (GEdge V) → Int
  | [] => 0
  | e :: l => rw II e + walkRW II l

/-- Reduced walk weight is exactly `L − II·Λ`: the no-positive-cycle
condition `walkRW ≤ 0` is the iteration bound's inequality rearranged. -/
theorem walkRW_eq (II : Int) (l : List (GEdge V)) :
    walkRW II l = walkLat l - II * walkLag l := by
  induction l with
  | nil => simp [walkRW, walkLat, walkLag]
  | cons e l ih =>
      simp only [walkRW, walkLat, walkLag, rw]
      rw [Int.mul_add]
      omega

/-- No cycle has positive reduced weight — i.e. II satisfies every cycle's
iteration bound. -/
def NoPosCycle (II : Int) (E : List (GEdge V)) : Prop :=
  ∀ v (l : List (GEdge V)), Walk E v v l → walkRW II l ≤ 0

/-- One relaxation sweep folded over a candidate edge list. -/
def relaxStep (II : Int) (f : V → Int) (acc : Int) (e : GEdge V) : Int :=
  imax acc (f e.src + rw II e)

theorem foldl_relax_ge_init (II : Int) (f : V → Int)
    (l : List (GEdge V)) (init : Int) :
    init ≤ l.foldl (relaxStep II f) init := by
  induction l generalizing init with
  | nil => exact Int.le_refl init
  | cons e l ih =>
      have h1 : init ≤ relaxStep II f init e := le_imax_l _ _
      exact Int.le_trans h1 (ih _)

theorem foldl_relax_ge_mem (II : Int) (f : V → Int)
    (l : List (GEdge V)) (init : Int) {e : GEdge V} (he : e ∈ l) :
    f e.src + rw II e ≤ l.foldl (relaxStep II f) init := by
  induction l generalizing init with
  | nil => cases he
  | cons a l ih =>
      rcases List.mem_cons.mp he with rfl | hm
      · exact Int.le_trans (le_imax_r _ _) (foldl_relax_ge_init _ _ _ _)
      · exact ih _ hm

/-- Best reduced walk weight into `v` over walks of length ≤ k, computed by
value iteration. The empty walk contributes 0 at every stage. -/
def bestUpTo (II : Int) (E : List (GEdge V)) : Nat → V → Int
  | 0, _ => 0
  | k + 1, v =>
      let s := bestUpTo II E k
      (E.filter (fun e => e.dst == v)).foldl (relaxStep II s) (s v)

theorem bestUpTo_le_succ (II : Int) (E : List (GEdge V)) (k : Nat) (v : V) :
    bestUpTo II E k v ≤ bestUpTo II E (k + 1) v := by
  simp only [bestUpTo]
  exact foldl_relax_ge_init _ _ _ _

/-- Each edge relaxes against the next round: the value at the source plus
the edge's reduced weight never exceeds the next round's value at the
destination. -/
theorem bestUpTo_succ_edge {II : Int} {E : List (GEdge V)} {k : Nat}
    {e : GEdge V} (he : e ∈ E) :
    bestUpTo II E k e.src + rw II e ≤ bestUpTo II E (k + 1) e.dst := by
  have hf : e ∈ E.filter (fun e' => e'.dst == e.dst) := by
    simp [List.mem_filter, he]
  simp only [bestUpTo]
  exact foldl_relax_ge_mem _ _ _ _ hf

/-- The iteration has stabilized after K rounds. -/
def Stabilizes (II : Int) (E : List (GEdge V)) (K : Nat) : Prop :=
  ∀ v, bestUpTo II E (K + 1) v = bestUpTo II E K v

/-- **Reduction (proved): a stable round is a valid schedule.** If the
value iteration stabilizes at K, then `bestUpTo II E K` satisfies every
edge constraint of a periodic schedule with period II. -/
theorem valid_of_stabilizes {II : Int} {E : List (GEdge V)} {K : Nat}
    (h : Stabilizes II E K) : ValidSchedule II (bestUpTo II E K) E := by
  intro e he
  have h1 := bestUpTo_succ_edge (II := II) (k := K) he
  rw [h e.dst] at h1
  unfold rw at h1
  omega

/-- Achievability, conditional on stabilization (proved). -/
theorem achievable_of_stabilizes {II : Int} {E : List (GEdge V)} {K : Nat}
    (h : Stabilizes II E K) : ∃ s : V → Int, ValidSchedule II s E :=
  ⟨bestUpTo II E K, valid_of_stabilizes h⟩

/-! ### Characterizing the iteration: `bestUpTo` is the best walk weight

Two directions. Soundness: every walk of length ≤ k into `v` weighs at
most `bestUpTo k v`. Witness: the value is 0 (the empty walk) or attained
by an actual walk of length ≤ k. Together they let walk surgery reason
about the iteration's values. -/

theorem bestUpTo_nonneg (II : Int) (E : List (GEdge V)) (k : Nat) (v : V) :
    0 ≤ bestUpTo II E k v := by
  induction k generalizing v with
  | zero => simp [bestUpTo]
  | succ k ih =>
      have h := bestUpTo_le_succ II E k v
      exact Int.le_trans (ih v) h

/-- Walks compose. -/
theorem walk_append {E : List (GEdge V)} {u m w : V}
    {l1 l2 : List (GEdge V)} (h1 : Walk E u m l1) (h2 : Walk E m w l2) :
    Walk E u w (l1 ++ l2) := by
  induction h1 with
  | nil v => simpa using h2
  | cons he hsrc _ ih => exact Walk.cons he hsrc (ih h2)

/-- Walks split at any list decomposition, exposing the midpoint. -/
theorem walk_split {E : List (GEdge V)} {u w : V} {l1 l2 : List (GEdge V)}
    (h : Walk E u w (l1 ++ l2)) :
    ∃ m, Walk E u m l1 ∧ Walk E m w l2 := by
  induction l1 generalizing u with
  | nil => exact ⟨u, Walk.nil u, by simpa using h⟩
  | cons e l1 ih =>
      cases h with
      | cons he hsrc htail =>
          obtain ⟨m, hm1, hm2⟩ := ih htail
          exact ⟨m, Walk.cons he hsrc hm1, hm2⟩

/-- Reduced weight is additive over concatenation. -/
theorem walkRW_append (II : Int) (l1 l2 : List (GEdge V)) :
    walkRW II (l1 ++ l2) = walkRW II l1 + walkRW II l2 := by
  induction l1 with
  | nil => simp [walkRW]
  | cons e l1 ih => simp [walkRW, ih]; omega

/-- Inversion for single-edge walks. -/
theorem walk_singleton {E : List (GEdge V)} {u w : V} {e : GEdge V}
    (h : Walk E u w [e]) : e ∈ E ∧ e.src = u ∧ e.dst = w := by
  cases h with
  | cons he hsrc htail =>
      cases htail
      exact ⟨he, hsrc, rfl⟩

/-- A `relaxStep` fold lands on its initial value or on a contribution of
some member. -/
theorem foldl_relax_attained (II : Int) (f : V → Int)
    (l : List (GEdge V)) (init : Int) :
    l.foldl (relaxStep II f) init = init ∨
      ∃ e ∈ l, l.foldl (relaxStep II f) init = f e.src + rw II e := by
  induction l generalizing init with
  | nil => exact Or.inl rfl
  | cons a l ih =>
      rcases ih (relaxStep II f init a) with h | ⟨e, hm, he⟩
      · by_cases hc : init ≤ f a.src + rw II a
        · refine Or.inr ⟨a, by simp, ?_⟩
          rw [List.foldl_cons, h]
          simp only [relaxStep, imax]
          rw [if_pos hc]
        · refine Or.inl ?_
          rw [List.foldl_cons, h]
          simp only [relaxStep, imax]
          rw [if_neg hc]
      · exact Or.inr ⟨e, List.mem_cons_of_mem _ hm, he⟩

/-- Witness: each `bestUpTo` value is 0 or attained by a walk of bounded
length. -/
theorem bestUpTo_witness (II : Int) (E : List (GEdge V)) (k : Nat) (v : V) :
    bestUpTo II E k v = 0 ∨
      ∃ u l, Walk E u v l ∧ l.length ≤ k ∧
        walkRW II l = bestUpTo II E k v := by
  induction k generalizing v with
  | zero => exact Or.inl rfl
  | succ k ih =>
      have hunf : bestUpTo II E (k + 1) v =
          (E.filter (fun e => e.dst == v)).foldl
            (relaxStep II (bestUpTo II E k)) (bestUpTo II E k v) := by
        simp [bestUpTo]
      rcases foldl_relax_attained II (bestUpTo II E k)
          (E.filter (fun e => e.dst == v)) (bestUpTo II E k v) with h | ⟨e, hm, he⟩
      · rcases ih v with h0 | ⟨u, l, hw, hlen, hval⟩
        · exact Or.inl (by rw [hunf, h, h0])
        · exact Or.inr ⟨u, l, hw, Nat.le_succ_of_le hlen,
            by rw [hunf, h, hval]⟩
      · have hmem := List.mem_filter.mp hm
        have hdst : e.dst = v := by
          have := hmem.2
          simpa using this
        rcases ih e.src with h0 | ⟨u, l, hw, hlen, hval⟩
        · -- the walk is the single edge e, from e.src
          refine Or.inr ⟨e.src, [e], ?_, by simp, ?_⟩
          · subst hdst
            exact Walk.cons hmem.1 rfl (Walk.nil _)
          · rw [hunf, he, h0]
            simp [walkRW]
        · -- extend the witness walk into e.src by the edge e
          refine Or.inr ⟨u, l ++ [e], ?_, ?_, ?_⟩
          · subst hdst
            exact walk_append hw (Walk.cons hmem.1 rfl (Walk.nil _))
          · simpa using Nat.succ_le_succ hlen
          · rw [walkRW_append, hunf, he, hval]
            simp [walkRW]

/-- Soundness: every walk of length ≤ k into `v` weighs at most
`bestUpTo k v`. Proved by induction on k, peeling the walk's LAST edge. -/
theorem walk_le_bestUpTo (II : Int) (E : List (GEdge V)) :
    ∀ (k : Nat) {u v : V} {l : List (GEdge V)},
      Walk E u v l → l.length ≤ k → walkRW II l ≤ bestUpTo II E k v := by
  intro k
  induction k with
  | zero =>
      intro u v l hw hlen
      have : l = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.mp hlen)
      subst this
      cases hw
      simp [walkRW, bestUpTo]
  | succ k ih =>
      intro u v l hw hlen
      rcases List.eq_nil_or_concat l with rfl | ⟨l', e, rfl⟩
      · cases hw
        simp only [walkRW]
        exact bestUpTo_nonneg II E (k + 1) _
      · rw [List.concat_eq_append] at hw hlen ⊢
        obtain ⟨m, hw1, hw2⟩ := walk_split hw
        obtain ⟨heE, hsrc, hdst⟩ := walk_singleton hw2
        have hlen' : l'.length ≤ k := by
          simp at hlen
          omega
        have h1 : walkRW II l' ≤ bestUpTo II E k m := ih hw1 hlen'
        have h2 : bestUpTo II E k e.src + rw II e ≤
            bestUpTo II E (k + 1) e.dst := bestUpTo_succ_edge heE
        rw [hsrc] at h2
        rw [hdst] at h2
        rw [walkRW_append]
        simp only [walkRW]
        omega

/-! ### The pigeonhole / cycle-stripping layer

A walk longer than the node pool revisits a node; the enclosed cycle has
reduced weight ≤ 0 under `NoPosCycle`; stripping it yields a strictly
shorter, no-lighter walk. Iterating bounds every walk's weight by a walk
of length ≤ |nodesOf E|, which pins the value iteration at that depth. -/

/-- Every node occurrence in the edge list (sources then destinations). -/
def nodesOf (E : List (GEdge V)) : List V :=
  E.map GEdge.src ++ E.map GEdge.dst

/-- Every edge of a walk belongs to the edge list. -/
theorem walk_edges_mem {E : List (GEdge V)} {u v : V} {l : List (GEdge V)}
    (h : Walk E u v l) : ∀ e ∈ l, e ∈ E := by
  induction h with
  | nil => intro e he; cases he
  | cons hmem _ _ ih =>
      intro e he
      rcases List.mem_cons.mp he with rfl | hm
      · exact hmem
      · exact ih e hm

/-- A nonempty walk starts at its first edge's source. -/
theorem walk_head {E : List (GEdge V)} {u v : V} {e : GEdge V}
    {l : List (GEdge V)} (h : Walk E u v (e :: l)) : e.src = u ∧ e ∈ E := by
  cases h with
  | cons hmem hsrc _ => exact ⟨hsrc, hmem⟩

/-- A walk ending in edge `e` ends at `e.dst`. -/
theorem walk_end_dst {E : List (GEdge V)} {u v : V} {e : GEdge V}
    {l : List (GEdge V)} (h : Walk E u v (l ++ [e])) : v = e.dst := by
  obtain ⟨m, _, h2⟩ := walk_split h
  obtain ⟨_, _, hdst⟩ := walk_singleton h2
  exact hdst.symm

/-- Pigeonhole, list form: a duplicate-free list drawn from a pool is no
longer than the pool. -/
theorem nodup_length_le {l P : List V} (hnd : l.Nodup)
    (hsub : ∀ x ∈ l, x ∈ P) : l.length ≤ P.length := by
  induction l generalizing P with
  | nil => exact Nat.zero_le _
  | cons a l ih =>
      have hcons := List.nodup_cons.mp hnd
      have haP : a ∈ P := hsub a (List.mem_cons_self ..)
      obtain ⟨s, t, rfl⟩ := List.append_of_mem haP
      have hsub' : ∀ x ∈ l, x ∈ s ++ t := by
        intro x hx
        have hxP := hsub x (List.mem_cons_of_mem _ hx)
        have hne : x ≠ a := fun hEq => hcons.1 (hEq ▸ hx)
        rcases List.mem_append.mp hxP with hs | hst
        · exact List.mem_append.mpr (Or.inl hs)
        · rcases List.mem_cons.mp hst with rfl | ht
          · exact absurd rfl hne
          · exact List.mem_append.mpr (Or.inr ht)
      have := ih hcons.2 hsub'
      simp at this ⊢
      omega

/-- A list with a repetition decomposes around the repeated element. -/
theorem not_nodup_decomp {l : List V} (h : ¬ l.Nodup) :
    ∃ x p q r, l = p ++ x :: q ++ x :: r := by
  induction l with
  | nil => exact absurd List.nodup_nil h
  | cons a l ih =>
      by_cases ha : a ∈ l
      · obtain ⟨q, r, rfl⟩ := List.append_of_mem ha
        exact ⟨a, [], q, r, rfl⟩
      · have hl : ¬ l.Nodup := by
          intro hnd
          exact h (List.nodup_cons.mpr ⟨ha, hnd⟩)
        obtain ⟨x, p, q, r, rfl⟩ := ih hl
        exact ⟨x, a :: p, q, r, rfl⟩

/-- A destination-list decomposition lifts to an edge-list decomposition. -/
theorem map_dst_split {l : List (GEdge V)} {q r : List V} {x : V}
    (h : l.map GEdge.dst = q ++ x :: r) :
    ∃ l1 e l2, l = l1 ++ e :: l2 ∧ l1.map GEdge.dst = q ∧
      e.dst = x ∧ l2.map GEdge.dst = r := by
  induction q generalizing l with
  | nil =>
      cases l with
      | nil => simp at h
      | cons e l2 =>
          simp at h
          exact ⟨[], e, l2, rfl, rfl, h.1, h.2⟩
  | cons y q ih =>
      cases l with
      | nil => simp at h
      | cons e l' =>
          simp at h
          obtain ⟨l1, e', l2, rfl, hq, hx, hr⟩ := ih h.2
          exact ⟨e :: l1, e', l2, rfl, by simp [h.1, hq], hx, hr⟩

/-- One stripping step: a walk longer than the node pool contains a
removable cycle — there is a strictly shorter walk, between the same
endpoints, that weighs no less. -/
theorem strip_step {II : Int} {E : List (GEdge V)} (hnp : NoPosCycle II E)
    {u v : V} {l : List (GEdge V)} (hw : Walk E u v l)
    (hlong : (nodesOf E).length < l.length) :
    ∃ l', Walk E u v l' ∧ l'.length < l.length ∧
      walkRW II l ≤ walkRW II l' := by
  -- the visited-node list is too long for the pool, so it repeats
  have hne : l ≠ [] := by
    intro hnil; rw [hnil] at hlong; simp at hlong
  have hsub : ∀ x ∈ u :: l.map GEdge.dst, x ∈ nodesOf E := by
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hd
    · cases l with
      | nil => exact absurd rfl hne
      | cons e t =>
          obtain ⟨hsrc, heE⟩ := walk_head hw
          exact List.mem_append.mpr
            (Or.inl (hsrc ▸ List.mem_map_of_mem heE))
    · obtain ⟨e, hel, rfl⟩ := List.mem_map.mp hd
      exact List.mem_append.mpr
        (Or.inr (List.mem_map_of_mem (walk_edges_mem hw e hel)))
  have hdup : ¬ (u :: l.map GEdge.dst).Nodup := by
    intro hnd
    have := nodup_length_le hnd hsub
    simp at this
    omega
  obtain ⟨x, p, q, r, hdecomp⟩ := not_nodup_decomp hdup
  cases p with
  | nil =>
      -- the start node u recurs: the prefix up to that recurrence is a cycle
      simp at hdecomp
      obtain ⟨rfl, hmap⟩ := hdecomp
      obtain ⟨l1, e, l2, rfl, _, hedst, _⟩ := map_dst_split hmap
      have hassoc : l1 ++ e :: l2 = (l1 ++ [e]) ++ l2 := by simp
      rw [hassoc] at hw
      obtain ⟨m, hcyc, hrest⟩ := walk_split hw
      have hm : m = u := by
        have := walk_end_dst hcyc
        rw [this, hedst]
      subst hm
      have hcw := hnp _ _ hcyc
      simp only [walkRW_append] at hcw
      refine ⟨l2, hrest, ?_, ?_⟩
      · simp; omega
      · simp only [hassoc, walkRW_append]
        omega
  | cons w p' =>
      -- an interior node x recurs: the segment between its occurrences
      -- is a cycle
      simp at hdecomp
      obtain ⟨_, hmap⟩ := hdecomp
      obtain ⟨l1, e1, l2', rfl, _, he1, hmap2⟩ := map_dst_split hmap
      obtain ⟨l2, e2, l3, rfl, _, he2, _⟩ := map_dst_split hmap2
      have hassoc : l1 ++ e1 :: (l2 ++ e2 :: l3)
          = (l1 ++ [e1]) ++ ((l2 ++ [e2]) ++ l3) := by simp
      rw [hassoc] at hw
      obtain ⟨m1, hA, hrest1⟩ := walk_split hw
      obtain ⟨m2, hB, hC⟩ := walk_split hrest1
      have hm1 : m1 = x := by rw [walk_end_dst hA, he1]
      have hm2 : m2 = x := by rw [walk_end_dst hB, he2]
      subst hm1; subst hm2
      have hcw := hnp _ _ hB
      simp only [walkRW_append] at hcw
      refine ⟨(l1 ++ [e1]) ++ l3, walk_append hA hC, ?_, ?_⟩
      · simp; omega
      · simp only [hassoc, walkRW_append]
        omega

/-- Every walk's weight is matched by a walk of length ≤ |nodesOf E|
between the same endpoints (fuel-based strip iteration). -/
theorem exists_short_walk {II : Int} {E : List (GEdge V)}
    (hnp : NoPosCycle II E) {u v : V} {l : List (GEdge V)}
    (hw : Walk E u v l) :
    ∃ l', Walk E u v l' ∧ l'.length ≤ (nodesOf E).length ∧
      walkRW II l ≤ walkRW II l' := by
  suffices h : ∀ (fuel : Nat) {u v : V} {l : List (GEdge V)},
      l.length ≤ fuel → Walk E u v l →
      ∃ l', Walk E u v l' ∧ l'.length ≤ (nodesOf E).length ∧
        walkRW II l ≤ walkRW II l' from h l.length (Nat.le_refl _) hw
  intro fuel
  induction fuel with
  | zero =>
      intro u v l hlen hw
      have : l = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.mp hlen)
      subst this
      exact ⟨[], hw, Nat.zero_le _, Int.le_refl _⟩
  | succ fuel ih =>
      intro u v l hlen hw
      by_cases hshort : l.length ≤ (nodesOf E).length
      · exact ⟨l, hw, hshort, Int.le_refl _⟩
      · have hlong : (nodesOf E).length < l.length := Nat.lt_of_not_le hshort
        obtain ⟨l'', hw'', hlt, hle⟩ := strip_step hnp hw hlong
        have hlen'' : l''.length ≤ fuel := by omega
        obtain ⟨l', hw', hshort', hle'⟩ := ih hlen'' hw''
        exact ⟨l', hw', hshort', Int.le_trans hle hle'⟩

/-- **Stabilization (proved).** Under the no-positive-cycle hypothesis the
value iteration is stable at K = |nodesOf E|: a witness for round K+1
either is short enough for round K already, or strips to one that is. -/
theorem stabilizes_of_noPosCycle {II : Int} {E : List (GEdge V)}
    (h : NoPosCycle II E) : ∃ K, Stabilizes II E K := by
  refine ⟨(nodesOf E).length, fun v => Int.le_antisymm ?_ ?_⟩
  · -- round K+1 ≤ round K
    rcases bestUpTo_witness II E ((nodesOf E).length + 1) v with h0 |
        ⟨u, l, hw, _, hval⟩
    · rw [h0]; exact bestUpTo_nonneg ..
    · obtain ⟨l', hw', hshort, hle⟩ := exists_short_walk h hw
      have := walk_le_bestUpTo II E (nodesOf E).length hw' hshort
      omega
  · exact bestUpTo_le_succ ..

/-- The target statement (conditional on the remaining work above):
any II meeting every cycle's bound admits a valid periodic schedule. -/
theorem achievability {II : Int} {E : List (GEdge V)}
    (h : NoPosCycle II E) : ∃ s : V → Int, ValidSchedule II s E := by
  obtain ⟨K, hK⟩ := stabilizes_of_noPosCycle h
  exact achievable_of_stabilizes hK

end Achievability
