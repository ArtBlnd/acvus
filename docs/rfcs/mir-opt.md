# MIR control shape and optimization

What shape MIR gives a loop and a branch, and what the optimizer may do to a
body. A terminator carries the control shape the source wrote, so no pass
rediscovers it. Every pass here keeps the program's result bit for bit and
declines where it cannot show that. A loop's facts are analyses; a pass may
normalize a loop the same way on every target, and the shape that depends on
the target is the lowerer's.

## RFC-0057: a `for` loop is one terminator that is its own condition

Status: Accepted

**A terminator keeps the control shape the source wrote.** A general `Jump`
remains only for an exit that shape does not have (`break`, `continue`,
`return`). Information a terminator drops is information a later pass has to
rediscover. `Switch` (RFC-0051), `For` and `Diamond` (RFC-0063) are the
instances of this rule.

1. **Syntax.** Four heads and no other:

   ```
   for x in &v { … }        // v: Vec<T> or Array<T, N>;  x: &T
   for x in &mut v { … }    // v: Vec<T> or Array<T, N>;  x: &mut T
   for x in a { … }         // a: Array<T, N>, consumed;   x: T
   for i in lo..hi { … }    // lo, hi: one integer width;  i: that width
   ```

   `lo..hi` is a `for` head, not a value: there is no `Range` type. `lo > hi`
   runs zero times. An iterator is not a `for` source: it is an extern's
   value and is consumed by `while let Some(x) = next(&mut it)`.

2. **MIR: the loop is one terminator.** The header block ends in
   `Terminator::For { source, body, body_args, exit, exit_trip, exit_args }`, with
   `source` one of `Slice(s)`, `SliceMut(s)`, `Array(a)` or `Range { at, hi }`.
   The terminator is the condition. No instruction writes `index < len`,
   `i = i + 1` or the element read. The terminator fills the body block's
   leading parameters (`elem`, `index`; for a `Range` the element is the
   counter, so there is one) and advances. The latch is
   `jump header(carried…)`, and `continue` is that jump. `break` jumps to
   `exit` with the carried values. A `Slice` source is the `AsSlice` of the
   container, taken once before the header. The borrow lives from there to
   `exit`, so the loans refuse a shape write to the container inside the
   loop (RFC-0064). `SliceMut` is the exclusive form. `Array` moves element
   `index` out on each iteration and leaves the array empty.

3. **What the terminator makes free.**
   - Bounds: the element read is `IndexUnchecked`, because the terminator is
     the bound. RFC-0047 rule 7's interval domain is not consulted.
   - Loop recognition: the header is the block that ends in `For`, and the
     induction variable is the terminator's `index`. No back-edge search or
     pattern match finds them (`analysis::loops::for_headers`).
   - Independence: the iterations run apart when the header carries no value
     (`carried` is empty) and every element write goes through `elem` of a
     `SliceMut` source. This is asked of the terminator and the body once.
     Splitting the loop on that answer is RFC-0066's.
   - The machine: a loop whose body rejoins is one region operation `For<S>`.
     The counter is a local of the operation, the bound is read once above
     the loop, and one chain runs per iteration.

4. **`break` and `continue`.** They are admitted inside `for` and `while` and
   target the innermost loop only, with no labels. The lowering's scope stack
   emits the drops of every scope between the statement and the loop body on
   that edge. A branch whose arm leaves the loop does not rejoin, so it is not
   a `Diamond`. It is an `Escape`, an operation of the body whose arm ends in
   the verdict the enclosing region reads (RFC-0052 rule 3). The loop stays one
   region (`For<S, Escapes>`), and a loop with no such branch is the same
   region it would otherwise be, with no compare added.

5. **Aliasing.** `for x in &mut v` holds the container exclusively for the
   loop, and the terminator carries that borrow. Naming `v` in the body is a
   loan error, and two `for` loops over `&mut v` cannot nest.

6. **An `Array` head's release is the MIR's, on every edge that leaves the
   loop.** The terminator takes element `index` out of its slot and leaves
   the slot holding nothing, so the array owns exactly the elements the
   counter has not reached. The terminator reads the array, so it is live
   wherever a path returns to the header, and `drop_insertion` places its
   `Drop` on each edge from there to where no path does: the terminator's
   exit, a `break`, a `?` and a `return` alike (RFC-0048). That `Drop`
   releases the storage and the elements not taken. An element taken was
   moved to the binding and is released by its own scope, and its slot
   releases nothing. The machine does not release the array and must not.
   Leaving early is therefore admitted at every element type, as Rust's
   `IntoIter` releases what it has not yielded: the slots record the
   counter, so one release covers the suffix without the MIR naming it.

7. **A `break` that jumps past the exit's drop block runs as joints.** Where
   `drop_insertion` gives the exit edge a block of its own, the lowering lays
   that block between the terminator and the body, and the `break` jumps past
   it to the continuation. The verdict "leave, then run the region's
   successor" would then run drops the arm has already run, so the loop is
   not collapsed into a region. The header is then the terminator as written:
   `ForAt<S>` compares the counter to the bound and continues to the body or
   the exit. The counter lives in a frame register: `ForStart<S>` on the
   entering edge sets it, and `ForStep<S>` advances it on the latch and on
   every `continue`. The terminator is the counter's only reader, so liveness
   reports it as a use there, or slot assignment would reuse the register.

8. **`return e` is the third exit edge.** It leaves the enclosing body from
   any depth. A loop holding one takes rule 7's path for the same reason a
   `break` does: the return's drop block stands between the terminator and the
   body.

9. **The exit edge defines the trip count where a pass reads it.** With
   `exit_trip: Defined` the terminator fills the exit block's first parameter
   with the number of times the body ran, a `u64`: `max(hi − at, 0)` for a
   range and the source's length for a slice or an array, the count RFC-0066
   rule 2 states. `exit_args` follow it. Only a pass that reads the count sets
   it, so every other loop keeps the exit edge it was lowered with. The
   validator refuses a count that is not a `u64`, and an exit block another
   edge also enters, which would define the parameter a second way. The
   machine takes the count from its counter as the test fails: the counter
   less `at` for a range, the counter itself otherwise.

**Why.** An index loop makes every pass rediscover the traversal the source
stated: the back edge, the bound proof, the induction variable, the borrow's
extent, iteration independence. The machine would also run a condition chain
and a counter move per iteration that the terminator removes.
An exit value `base + trip·step` (RFC-0066 rule 7) needs the count after the
loop, and the terminator is what knows it, as it knows the body's element.
**Cost.** One terminator arm in every reader of `Terminator`. The enum is
closed, so the compiler enumerates the sites. There are two loop forms:
`while` remains for a loop whose condition is not a traversal. Every reader
that pairs the exit edge's arguments with its target's parameters skips the
count where the edge defines one.
**Rejected.**
- Lowering `for` to `while` with an index — every pass rediscovers the
  traversal, and the machine runs the condition chain.
- A `Range` value type — a range that exists only as a loop head has no other
  reader, and a value would need a representation and a crossing.
- `for` over an iterator — `next` is an extern call per element and cannot be
  a terminator's condition.
- Labeled `break` — a second scope mechanism the block design does not have.
- Refusing `break`, `?` and `return` in a loop over an array of owners —
  the elements not taken have a release (rule 6), and a refusal decided at
  the jump was skipped wherever the element was still open there.
- A release that takes the counter as its start — the slots already record
  it, and a second record of which elements left beside the array's own
  would be two answers to one question.
- Tail duplication for `break` (copying the loop's tail into each arm so every
  branch rejoins) — 2^k code for k exits.
- A length instruction for the trip count — it answers a slice, and a range's
  `max` would still need a branch the exit block grows.
- A count found per source (an array's `N`, a branch for a range, nothing for
  a slice) — three rules for one question, and a slice's length is no MIR
  value.
- The count as a value the terminator defines rather than a parameter — a
  definition on one edge of a two-edge terminator is a kind of definition
  every dominance check would have to learn.

## RFC-0063: an `if` whose arms rejoin is a `Diamond` terminator

Status: Accepted

1. **The terminator.** A two-way branch whose arms both reach one join ends
   its block in
   `Terminator::Diamond { cond, then, then_args, else, else_args, join }`.
   Each arm's last block jumps to `join`, and an `if`-expression's value is
   the join's parameter. The lowering emits `Diamond` for `if` (statement and
   expression), `&&`, `||`, and a payload pattern's test (`Some(1) = e`,
   which is `&&` written by hand). The criterion is reachability of `join`
   from each arm over the instructions the arms occupy (`ir::reaches`). The
   lowering asks it of the arms it has just written, and `validate` asks it
   of a whole body. An arm that leaves a loop, returns (`?`), or ends in `!`
   (`Diverge`, RFC-0038) does not rejoin. A `break` of a loop written
   *inside* an arm leaves that loop and not the branch, so the branch still
   rejoins.

2. **`JumpIf` remains for the shapes that are not a diamond.** These are the
   `while` test (its exit is the loop's), a `match` arm's guard and arm chain
   (the else is the next arm), `?`, and an arm that does not rejoin. The
   machine runs a non-rejoining arm inside a loop as an `Escape`
   (RFC-0057 rule 4).

3. **What the terminator makes free.** `prepare` builds the `Diamond<C>`
   region from the terminator and the block table. The arms' extents come
   from the table and the join from the terminator, with no forward scan.
   `is_closed` stays, because collapsing a range into one operation still
   requires that no jump from outside names a block inside it. `Select`
   (RFC-0052) is asked of the terminator. An arm that ends in a branch naming
   the same join is `ArmRegion::Branch`, so an `if`/`else if`/`else` chain of
   any length is one region, and such an arm is never a `Select`. Both labels
   of a `Diamond` are present, or `validate` refuses the terminator.

4. **Passes.** Every consumer of `JumpIf` has a `Diamond` arm with the same
   edge semantics (two successors, arguments per edge) plus `join`. A pass
   that rewrites edges rewrites `join` alongside. A pass that treats the two
   alike says so through one helper (`ir::two_way`).

5. **A pass that dissolves a join maintains it.** Where `sroa`'s threading
   scatters the paths through a join, the branch is demoted to `JumpIf`
   (`cfg::demote_diamond`), and the demotion is recorded in
   `MirBody::demoted_diamonds`. `forward::collapse` maps `join` through the
   forwarders it removes. `optimize::rejoin` runs once, after every other
   pass and before `validate`. It restores a recorded demotion whose arms
   meet again (`ir::meets_again`, the lowering's own criterion). `validate`
   refuses a `Diamond` whose arm misses its join (`DiamondArmMissesJoin`) and
   a demoted branch that meets again (`DemotedDiamondMeetsAgain`). A pass
   that decides a branch on a constant removes the label from
   `demoted_diamonds` (RFC-0071).

**Why.** Rebuilding the diamond from a `JumpIf` by scanning block order made
region recognition depend on the lowering's emission order. A different
order ran the branch as joints, which was a performance cliff that no test
stated. The lowering knows the join, so it writes it.
**Cost.** There are two terminators for a two-way branch until `while` and
the guard get their own. Every edge-rewriting pass carries `join`, and
`rejoin` exists because a demotion and the arms' return happen in different
passes.
**Rejected.**
- `JumpIf` with an optional `join` field — a field the lowering sets and no
  pass decides on is RFC-0057's removed `merge_of`, and a terminator that
  *is* the shape has no optional part.
- Recognizing the diamond in a MIR pass (a side table) — the same
  rediscovery one stage earlier, when the lowering already knows.
- Keeping a demoted diamond's shape as a `JumpIf` field for `rejoin` — a
  shape a pass decides. The demotion record is the lowering's fact kept
  beside the body.

## RFC-0055: a binary operation on two constants is the constant

Status: Accepted

**A binary operation on two constants is replaced by the constant.** The
pass runs after SSA construction and before dead-code elimination, so the
operands it orphans die with the rest of the dead code.

The value is the one the machine computes, at the operand's width. `+`, `-`
and `*` wrap. A shift takes its amount modulo the width. A comparison reads
both operands at the width's own signedness. A float comparison compares bit
patterns (RFC-0037).

Nothing is folded where the machine would panic (a zero divisor, or a
quotient that leaves the width). The operation stands, and the program
panics where it did. A float operation whose result is NaN is not folded,
because a folded NaN would carry the compiler's bit pattern where the
machine's belongs.

Two constants that are not adjacent still join when one associative and
commutative operator at one integer width separates them and the
intermediate has exactly one use: `(x + 1) + 2` becomes `x + 3`. The
intermediate becomes the joined constant where it stood, so no new
definition is placed.

**The optimizer introduces no operator the chain recognizer does not
claim.** A fold replaces an operation with its constant, or moves a constant
from one operand of one operator to another. The operators a body holds
after the fold are the ones the source wrote.

The fold's value and the machine's are one claim, and a test executes it.
Each edge the surface can write (a wrapping width, the widest constant, a
negative dividend, a NaN) runs once with foldable operands and once with
operands from the page, and the two must produce the same register word.

**Why.** Scalar replacement (RFC-0050 rule 11) turns a field read into a constant, and
nothing after it computes `1 + 2`. Refusing wherever the machine would panic
is how the fold is guaranteed never to change a result, rather than argued.
**Cost.** One more pass over every body. Each rewrite removes an operation
and writes none, so the repetition ends. `i % 2 == 0` still runs a division.
**Rejected.**
- Strength reduction to a mask or shift (`x % 2^k`, unsigned `x / 2^k`, a
  signed `%` compared to zero) — built and correct, and slower. `prepare`
  fuses the five arithmetic operators into one chain, and a mask splits
  `i % 2 == 0` into two dispatches, which cost more than the division
  removed. It is re-admitted when the chain alphabet holds the bitwise and
  shift operators and `enum match` and `collatz while` do not regress.
- Folding through a chain's leaves — couples the fold to the recognizer's
  shape language. The fold runs first and leaves whole instructions.
- A general algebraic simplifier (`x - x`, reassociation across operators,
  distribution) — each rule is its own correctness argument, and together
  they are a rewrite engine with a termination proof to maintain. A further
  rule earns its place by a bench that names it, as `collatz` named the
  integer identities of RFC-0083.

## RFC-0056: a loop's `i * k + x` becomes its own counter

Status: Accepted

**A loop is a back edge, and one module says so.** An edge `tail -> head`
whose head dominates its tail names a natural loop. Its body is the head
together with every block that reaches the tail without passing the head.
`analysis::loops` owns back edges, headers, latches, bodies, the loop depth
`code_motion` holds every hoist to, and the definition of a value that is
the same on every iteration. A second answer to any of these is a defect. A
`for` is a natural loop here too (RFC-0057).

**An induction variable is a header parameter plus an invariant.** `i` is one
when the latch sends the header `i + c` and `c` is the same on every
iteration. This holds for `while` loops as well as `for` loops.
`analysis::affine` owns this and `i * k + x` (RFC-0066 rule 4), and the pass
reads the rule that made each value affine. Nothing deeper is examined: a
counter rewritten through memory, a step that is itself an induction
variable, and a derived variable of a derived variable are outside the
pattern.

**A value is the same on every iteration when it is defined outside the loop
or is a word constant.** The first case is what the hoist produces. The
second is the case the hoist cannot reach: control equivalence keeps a
`Const` inside the body, yet it reads nothing and writes the same word every
time. Such a constant is re-emitted above the header, not moved.

**Only a counter read in an `InOrder` join is reduced** (RFC-0066 rule
7). A counter that feeds a pure stage or an unordered join is canonicalized
instead, so each iteration computes `i * k + x` from its own `i`. A derived
counter would carry that value from the previous iteration and order what
reads it. An `InOrder` join runs in order already, so the reduction costs
it nothing.

**`i * k + x`, where only that sum reads the product, becomes a header
parameter.** It is started in the preheader and advanced in the latch. The
conditions are that the loop has one entering edge from a block ending in a
plain jump to the header, and one back edge, and that the multiplication
stands in a block that dominates the latch. There is one derived variable per
distinct `(i, k, x)`. A loop the pass declines is left exactly as it was.

**A reduction must lower the body's operation count.** `i * k + x` goes from
four operations to three, counting `i`'s own increment. A bare `i * k` goes
from three to three plus a carried parameter, so it is not reduced. What
decides this is the count, not the syntax.

**Only wrapping integer arithmetic is reduced.** Modulo `2^width`,
multiplication distributes over addition exactly, and neither operation can
raise, so the value and the iteration that raises are unchanged. In floating
point the accumulated form is a different number, and on mandelbrot's grid
34 of 20000 pixels reach a different escape count. A multiplication that does
not run on every iteration is not reduced.

The pass runs after `code_motion` and before `reorder`. The hoist is what
puts `k` and `x` above the header and leaves the preheader a block of its
own. The reorder schedules within a block and must see the instructions this
pass adds.

**Why.** Two passes asking what a loop is, with two answers, is the failure
the shared definition prevents. The reduction must not change a result.
**Cost.** A derived variable costs one header parameter, preheader
arithmetic, and one latch addition. The count rule gives up the bare form,
and the one-product pattern gives up a product read by two invariant sums.
**Rejected.**
- Reducing inside the prepared chains — the chain already fuses `i * k + x`,
  and the dominator tree that says an expression runs on every iteration is
  gone by then.
- A general scalar evolution — a lattice fixpoint with its own correctness
  burden. The pattern here is two shapes deep, and each shape is checkable by
  reading it.
- Reducing a multiplication under a branch — correct, and slower whenever the
  branch is not taken.
- The float form behind a tolerance — nothing in the language says the caller
  accepts a different number. Reopening it needs a measurement that shows no
  result changing.

## RFC-0061: a store nothing reads is dead

Status: Accepted

1. **The reader decides.** An `Assign` whose target is a `Var` or `Param`
   with an empty path is a conditional root. `dce` keeps it when an
   instruction already marked live reads that slot on some path out of the
   store, before the next whole store into the slot or the body's end. A read
   is what `loans` counts as one: a `Take`, a `Ref` and every use through it,
   a field read, a loan to a call, or a loan the terminator carries out. A
   store's reader may itself be dead, so the mark phase alternates between
   tracing operands and pulling in the stores that newly live instructions
   read, until one fixpoint. A store through a reference or into a path
   stays an unconditional root.

2. **The old occupant's release does not move.** An assign releases what the
   slot held (RFC-0045), so a store is removed only where the slot holds
   nothing on every path into it. This is decided by a forward may-hold
   analysis: at entry every parameter and capture holds a value, a whole
   store fills a slot, and a take of a move-only slot empties it. A
   word-typed slot is the exception, because its assign releases nothing
   (RFC-0048 rule 4).

3. **The stored value is released once, by `drop_insertion`.** A pure
   producer of the removed store's value goes with it in the same backward
   walk, so the value is never made. A root producer (a `Fetch`, an opaque
   call) stands, and `drop_insertion`, which runs after `dce`, releases its
   result where it is made. `dce` emits no drop (RFC-0048).

**Why.** A value stored and never read outlived the body, together with every
producer and borrow that fed it.
**Cost.** Two analyses per body: the may-hold fixpoint, and a forward walk
per candidate store.
**Rejected.**
- `dce` emitting the orphaned value's drop — two places would decide one
  release.
- Leaving RFC-0060's residue to this rule — the inlined residue names a value
  with no definition, which the passes before `dce` would read and
  `validate` refuses.

## RFC-0060: a small pure closure called where it was made is its body

Status: Accepted

**A call whose callee is one `MakeClosure` of the same body is spliced when
the closure is small, pure and called nowhere else.** All four conditions
must hold:

- the closure's body takes no `Order`, which is the checker's verdict, read
  and not recomputed;
- it holds at most `INLINE_MAX_INSTS` (8) instructions;
- it is one block, with its trailing `Return` the only instruction that
  starts or ends one;
- every reader of the closure is one of those calls.

The reader condition is asked first. A closure that is also stored, returned
or passed stays a closure, and its calls stay calls, because an inlined copy
beside a live closure is two bodies for one. The reader walk follows the
closure through its variable and the shared references taken of that
variable, and refuses anything else.

**A capture becomes the local it was.** There are three shapes and no fourth:

- the argument already *is* the reference the body reads, because the
  captured name was itself a capture and no `&&T` exists (RFC-0029); it is
  substituted as it is;
- the body reads the capture only as its word copy (RFC-0018); the copy is
  the argument;
- otherwise the caller moves the captured value into one local and lends it,
  where `MakeClosure` stood, once however many calls read it.

Any other capture type refuses the closure. No `&mut` capture exists: a
capture is typed as a shared reference, and the checker refuses assignment
to a captured name.

**The inliner removes what is left of an inlined closure.** That is the
`MakeClosure`, its variable, the references taken of it, and its drop. `dce`
cannot, because the residue would name a value with no definition through
every pass before it (RFC-0061).

A `Callee::Direct` to a local function is spliced whatever its size, unless
its body makes a closure. Such a callee is not spliced, because a closure
body belongs to its own module's namespace, and a spliced `MakeClosure`
would name a body the caller's module does not have. An inlined closure's
`MirBody` stays in the module unreached.

**Why.** A small closure called in a loop pays a full indirect call and an
operand-space build for a body of two instructions.
**Cost.** The bound 8 is a count, not a measurement. Raising or lowering it
changes only which closures qualify.
**Rejected.**
- Splicing a closure that is also stored or passed — two bodies for one value.
- Splicing a direct callee that makes a closure — needs a closure-label
  namespace the two modules share, which is a separate decision.

## RFC-0066: a loop is analyzed and normalized in MIR, and the lowerer decides its shape

Status: Proposed

A loop's facts are analyses over MIR that every reader shares:
`analysis::loops` for the nest and the trip count, `analysis::affine` for the
values that advance by a fixed step, and `analysis::carried` for what one
iteration hands the next. A pass may rewrite MIR into a normal form that holds
on every target. The shape of a loop that depends on the target, the runtime
or the actual `n` is the lowerer's, and no MIR pass writes it.

1. **Normalization is the optimizer's; the target's shape is the
   lowerer's.** A MIR pass may rewrite a loop when the result is the same
   program on every target: IV canonicalization, collapsing a rectangular
   nest, separating a merge's join from the body. Unrolling, tiling and
   blocking, a chunk size, and whether to split a loop at all depend on the
   target, the runtime or `n`, and they are the lowerer's.

2. **The nest.** Every natural loop (RFC-0056) has its parent, the smallest
   other loop that contains its header, and its children. Its kind is
   `For { source }` when its header ends in a `for` terminator and `While`
   otherwise. Its trip count is a term: `max(hi − at, 0)` for a range and
   `len(source)` for a slice or an array, read off the terminator
   (RFC-0057). A `while`'s is unknown. A loop is rectangular in its parent
   when its trip count is known and every value the term reads is invariant
   in the parent.

3. **The term.** A term is a constant, a value invariant in the loop, a
   source's length, or `+`, `−`, `×` or `max` of terms. It denotes an
   integer. It is kept as found and evaluated where the lowerer knows the
   atoms, never simplified. A value is invariant in a loop when it is
   defined outside the loop, or is a word `Const`, which reads nothing and
   writes the same word each time (RFC-0056).

4. **Affine values.** A value `v` is affine in a loop when
   `v = base + k·step` over the iteration number `k`, with `base` and `step`
   terms of rule 3. A `for`'s counter is affine from its terminator: a
   range's element is `{at, 1}` and a slice's or an array's index is
   `{0, 1}`. A header parameter entered with `b` whose back edges all send
   `p + c`, `c` invariant, is `{b, c}`. `a·v` and `v + b` of an affine `v`,
   `a` and `b` invariant, are affine. Only integers are affine, because
   wrapping `+` and `*` are exact (RFC-0037). The analysis is one loop
   deep.

5. **Carried state.** Each header parameter is exactly one of `Iv`, affine
   by rule 4; `Merge { op, exact }`, whose back edges send `p ⊕ x` where the
   body reads `p` only as that operand and `p ⊕ x` only on the back edges;
   or `Recurrence`, anything else. Integer `+` and `*` are exact merges.
   Float `+` and `*` are inexact merges. A call of an extern that declares
   itself associative is an exact merge (RFC-0082), which is how `min` and
   `max` over integers are recognized. `&&` and `||` are exact operations,
   but they reach MIR as a short-circuit `Diamond`, and their recognition
   is open.

6. **Order is per target, not per loop.** What a loop changes and how the
   change is ordered is stated per target by the stages of RFC-0089: pure
   work runs apart, and each join is `Disjoint`, `AnyOrder` or `InOrder`.
   A carried value is a target like any storage. A loop whose every target
   joins `InOrder` with no law runs in order; that is a cost, not a
   different form. RFC-0057 rule 3's question is a loop whose only target
   is its `&mut` source's element, joined `Disjoint`.

7. **One normalization per induction variable.** IV canonicalization
   rewrites an `Iv` into `base + k·step` in the body, `k` read off the
   counter, and `base + trip·step` where it is read after the loop, from
   the count the exit edge defines (RFC-0057 rule 9), whenever the `Iv`
   feeds a join that is not `InOrder` or a pure stage. Strength reduction
   (RFC-0056) applies to an `Iv` whose readers all sit in an `InOrder`
   join, which runs in order anyway. The choice is per variable, not per
   loop. A `while` is declined, since it states no count; a later exact
   recognizer makes it a `for`, and this pass applies to that unchanged.

8. **A cost table is the backend's, supplied from outside.** The embedder
   gives the lowerer one table for the backend it runs. The table has a row
   per operation kind the backend can emit (each `Op` family at its
   register forms, a spawn, a join, a frame bind) in the table's own unit.
   How a backend produces its table is its own. Every cost is a sum of
   rows. A table records ranges, and the lowerer compares against the
   conservative end. With no table, every loop runs in place. A loop's cost
   is its body's rows times its trip count, a lower bound. Costs are
   computed once, by the lowerer, and no MIR pass reads them.

9. **Regions.** A region is a stage of a `For` (RFC-0089) evaluated as
   one unit, without crossing a jump the analysis cannot see through. A
   jump's target must be inside it. A loop whose stages are not regions is
   not divided.

10. **The lowerer reads each join's order, and the executor decides how
    to wait.**
    - Split, the pure stages run in chunks, bounded in how many are in
      flight. An `InOrder` join takes the chunks' results in chunk order;
      an `AnyOrder` join takes them as they arrive; a `Disjoint` join lets
      each chunk write its own range. A law lets a chunk combine before the
      join.
    - In a body typed `Sync`, the split is one synchronous call to the
      embedder's executor, `run_ordered` or `run_unordered`, and the
      executor decides how to wait: inline, on a pool, or otherwise. The
      body stays `Sync` (RFC-0046 rule 1). In a body that already suspends,
      the chunks are spawned and awaited, in order or as they arrive.
    - Running in place is always admitted, and is the only choice with no
      table (rule 8). Where the count is known only at run time, one
      compare ahead of the loop, against a threshold folded from the table,
      chooses between the split chain and the in-place chain.
    - A chunk does not split again: one level, by structure.

**Why.** A normal form that holds on every target is the same program
everywhere, so writing it into MIR decides nothing a target could decide
better. A shape chosen for a target is a guess about the lowerer, which
knows the target, the runtime and the actual `n`. Strength is read from the
kind of the carried state because the kind says whether iterations can be
reordered, and a count says only what reordering costs. The author writes
the loop, and the system finds the parallelism from facts the checker
already establishes.
**Cost.** Three analyses built per body and read by every loop pass. A
table per backend and a family of split operations (chunk, spawn, join
through a merge). A loop's normalization is chosen by its strength, so
a change in the analysis moves a loop between two passes. RFC-0057's and
RFC-0064's analyses become inputs whose promises must stay stable.
**Rejected.**
- Unrolling, tiling, blocking or permutation as MIR passes — a lowerer's
  guess written into the program's meaning, and nothing undoes it when the
  guess is wrong for a target.
- Strength decided by the number of carried values — a count is a cost, and
  a loop with one recurrence is ordered where a loop with ten sums is not.
- Constant costs in source — true on one machine on one day.
- A table the runtime measures at first compilation and caches — it ties
  the compiler to one machine's timing at one moment and adds a cache to
  invalidate. The backend knows its costs, and the embedder that chose the
  backend supplies them.
- Scalar evolution alone — it answers the trip count and nothing when that
  fails.
- Leaving an `Iv` read after the loop carried — that loop keeps the
  dependence the normal form removes, which then holds per variable.
- A `while` trip count derived from its recurrence — the door to general
  scalar evolution. A `while` is promoted to `for` only by a recognizer that
  is exact, or it stays undivided.
- An explicit `par for` — the facts the split needs are the checker's. Where
  they are not established, the author is told why, not asked to assert them.

**Open.** How `&&` and `||` are recognized as merges. Whether
rule 9's jump-boundary conditions reduce to effect boundaries alone. Whether
the machine offers an explicitly reassociable float reduction, which is a
language decision.

## RFC-0089: a `for` is a chain of stages, each pure or a join over the storage it changes, and each join states its order

Status: Proposed

An iteration reads, computes, and changes storage. `For` states which of its
instructions only compute and which touch what the loop changes, and how
the touching is ordered. Pure work runs apart; a join runs in the order it
states. How either runs is the lowerer's (RFC-0066 rule 10).

1. **The terminator.** `Terminator::For { source, stages, exit, exit_trip,
   exit_args }`, with `source`, `exit`, `exit_trip` and `exit_args` as
   RFC-0057 states them.
   - `stages` lists `Stage::Pure { entry }` and `Stage::Join { entry,
     targets, order, law }` in body order. The body block's parameters are
     the element and the counter. Each stage's last block jumps to the next
     stage's entry, and the last one is the one latch, which `continue`s
     join.
   - The stages run one after another as written. Every pass that reads
     edges reads the body as a sequential program, and running in place is
     the loop as written: the chain is the body of the region RFC-0057
     rule 3 builds.
   - A value reaches a later stage by dominance.

2. **A target is storage the loop changes.** A target is a storage live at
   the header that an instruction of the body writes or lends `&mut`. A
   context is a storage (RFC-0025). A slot the body defines and drops
   inside an iteration is not a target. The element of a `&mut` source is
   a target whose writes land at the counter's slot.

3. **A pure stage changes no target.** No instruction of a pure stage
   writes a target, lends one `&mut`, carries an ordered effect, or leaves
   the loop. It reads the element, the counter, values from outside the
   loop and values earlier stages defined, through shared borrows.
   `validate` refuses a pure stage that breaks this, from stage membership
   and `analysis::loans`; nothing else about purity is assumed.

4. **A join is the smallest slice that touches a target.** It starts at
   each instruction that reads a target's state, such as a `pop`, and holds
   everything whose value depends on that state up to the instructions
   that write the target: the target's dependence cycle.
   - Instructions on one target keep their written order unless a
     declaration relates them, since two `&mut` users of one storage are
     opaque to each other.
   - A heavy call inside a join is split into spawn and evaluation
     (RFC-0046 rule 3), so the join can issue it and wait apart.
   - Work that depends on a target's state and writes no target leaves the
     cycle as a pure stage after the join.
   - A body that needs a target's state before it can compute makes that
     target's join a producer placed before the pure stage that reads it.
     A pair of target operations declared inverse is promoted to one
     carried value before the loop and written back after it (RFC-0082).

5. **Order is one of three.**
   - `Disjoint`: each iteration writes only the element at its counter.
   - `AnyOrder`: every operation of the join commutes (RFC-0013), stands in
     an `anyorder` region (RFC-0007), storage writes included, or joins
     through a commutative law.
   - `InOrder`: anything else. A float law is `InOrder` by default:
     joining in arrival order changes the rounding.

   `validate` holds the mark to the operations' declarations.

6. **A law is specialization, not permission.** A join whose target's
   update is a monoid action names it: `Op`, `Call` (an associative extern,
   lifted over `Option` when it states no identity), `Fold` (RFC-0082
   rule 3) or `Order` (RFC-0007 rule 7). The lowerer may combine inside a
   chunk and join the partials. A join without a law still runs in its
   order.

7. **Exits and traps.** A loop is left only from the header or from an
   `InOrder` join. A run apart reports the trap least in the order
   (iteration, stage), and a trap releases nothing (RFC-0048 rule 8). The
   lowerer runs an `Array` source in place until the release of elements
   scattered over chunks exists.

8. **Who writes it.** A pass after IV canonicalization, which is decided
   per target rather than per loop (RFC-0066 rule 7), and after every pass
   that moves or merges the body's instructions, writes the stages. It
   duplicates nothing: an instruction two joins share joins them. A loop
   the pass cannot split is one `InOrder` join over everything it
   touches, and runs as written.

**Why.** Pure work and changes to storage are different facts, and only the
second has an order. Stating them apart makes independence a structural
check (rule 3) and leaves one question to the lowerer, how each join is
ordered. A recurrence is a join that serializes; that is a cost the form
states, not a shape it refuses.
**Cost.** A terminator arm in every reader. A validator that restates rule
3 and rule 5. The stage pass and IV canonicalization per target.
**Rejected.**
- Parts joined by a law, with writes left in the body — writes in a body
  made independence a proof about which element each write reaches, and a
  law became the price of running apart.
- A two-valued order — an element store is neither ordered nor a free
  join; it is disjoint by the counter.
- Join operations marked in place in the body — the order would be read
  per instruction by every pass.
- Arms that fork and all run — every pass reads successors as
  alternatives.

**Open.** Each runs as `InOrder` today; the tiers weigh ease against reach.
- First, after the form: a law through a nested loop; a store at an
  injective affine index of the counter as `Disjoint`; named commutation
  sets in place of `commutes: bool`.
- After the executor's ordered and unordered runs: a join split by key; a
  `Stream` source for `while` loops.
- When a use asks: speculative exits; a scan law, for a body that reads a
  partial; an action law for heavy work inside a target's cycle.

## RFC-0081: a `while` that counts by one to an invariant bound is a range `for`

Status: Proposed

`let i = b; while i < n { …; i = i + 1; }` is the traversal `b..n` written
without the terminator that states it. A pass recognizes exactly that form
and gives its header the `For` terminator (RFC-0057 rule 2), so every reader
that asks a terminator for a traversal reads this loop unchanged: the nest's
trip count, IV canonicalization, the region and the lowerer's split
(RFC-0066). This is the exact recognizer RFC-0066 admits for promoting a
`while`.

1. **The form.** A `While` loop (RFC-0066 rule 2) is converted when all of
   these hold, and is left exactly as it was otherwise:
   - one block enters it, and its header ends in a two-way branch on `c`,
     to a body block inside the loop when `c` holds and to an exit block
     outside it when it does not;
   - `c` is `i < n` or `n > i`, computed in the header;
   - `i` is a header parameter that every entering edge sends `b` and every
     back edge sends `i + 1`, with `1` the integer one (RFC-0066 rule 4);
   - `n` is invariant in the loop (RFC-0066 rule 3), or the header
     computes it from such values in the steps rule 3 admits;
   - `i`, `b` and `n` have one integer type. A range admits every width
     (RFC-0057 rule 1), so no width is declined and nothing is cast;
   - no block of the loop but the header has an edge out of it or returns,
     so a `break`, a `return` or a branch into a block that diverges
     declines;
   - the body block and the exit block each have one predecessor, the
     header;
   - when the pass writes the bound above the header, the entering block
     ends in a jump to the header, so what it writes runs once per entry
     and on no other path.

2. **The terminator alone changes.** The header's branch becomes
   `For { source: Range { at: b, hi: n } }` with the branch's two edges and
   their arguments, and the body block gains a fresh counter as its leading
   parameter. `i` stays a header parameter advanced by the body's own
   `i + 1`, and stays an induction variable (RFC-0066 rule 5). No body
   instruction changes, and nothing reads the new counter. The comparison
   loses its reader and `dce` sweeps it, and with it `i` and its `i + 1`
   when the comparison was their only reader. The pass computes no exit
   value: the exit edge carries `i` as it did, and what `i` is after the
   loop is IV canonicalization's (RFC-0066 rule 7). A bound the header
   computes is computed again at the end of the entering block from the
   same operands, because the machine reads a range's bounds on the
   entering edge (RFC-0057 rule 7). For a word constant, as `while i < 10`
   lowers, that is RFC-0056's re-emission of a word. The header's own
   computation loses its reader with the comparison.

3. **A computed bound is the header's first visit, moved to the entry.**
   The header runs on every entry before any body block. A step computed
   from word literals, values defined outside the loop and earlier steps
   gives the same value, and raises the same trap, on every visit, when
   the step is deterministic:
   - `+`, `-`, `*`, `/` and `%` at an integer width. The machine computes
     each from its two words at the width (RFC-0037): `+`, `-` and `*`
     wrap; at every width `/` panics with `attempt to divide by zero` on a
     zero divisor and `%` with `attempt to calculate the remainder with a
     divisor of zero`; at a signed width `MIN / -1` panics with `attempt
     to divide with overflow` and `MIN % -1` with `attempt to calculate the
     remainder with overflow`; an unsigned width has no other failure.
   - a call of an extern whose declared effect is `pure` and touches no
     context, the author's promise (RFC-0080 rule 3). Each argument is a
     shared reference defined outside the loop, which the borrow check
     keeps unwritten while the loop runs, since the header reads it on
     every visit (RFC-0064), or a scalar. A `&mut` argument is declined,
     since the call writes through it; so is a reference the header makes
     (`while i < v.len()`, `&v` made on each visit) or a step computes,
     neither being defined outside, and any other argument, which the
     header moves.

   Evaluating the steps once, in the header's order, at the end of the
   entering block is exact when moving them ahead of the header's other
   instructions changes nothing observable. A `/`, a `%` and a call can
   trap, since `pure` does not say that a call returns: `unwrap` is `pure`
   and panics. So when the bound holds one, no instruction before its last
   such step in the header, other than a step of the bound, may have an
   effect or trap; only a constant, a reference, a cast and a binary
   operation that is not an integer `/` or `%` qualify. Then the entry
   raises exactly the trap the first header visit raised, on an entry that
   runs the body zero times too, and a bound with no trapping step moves
   freely. One rule covers `/`, `%` and a `pure` call; which of them trap
   and when does not enter it.

4. **The two loops are one program at every entry.** Both start at `b` on
   each entry, so the `k`-th visit of the header holds `b + k` in both. The
   range runs its body while its counter is below `n`, by the comparison
   `i < n` makes at the same width, and advances by one, wrapping as
   `i + 1` does (RFC-0037). `n ≤ b` runs zero times in both. The advance
   runs only after the counter compared below `n`, so it never wraps, and
   `n` at the width's maximum ends both loops at `n`.

5. **Placement.** The pass runs after the SSA construction, which makes
   `i` a header parameter, and after the fold, which settles a constant
   bound. It runs before `dce`, which sweeps the comparison, and so before
   `code_motion`, `lsr` and IV canonicalization, each of which then sees the
   loop as a `for`. A bound the header computes is not hoisted yet at that
   point, which is why rule 2 computes it again rather than finding it
   above the header.

**Why.** A counted `while` is common, and a `for` is what every loop pass
and the lowerer read a traversal from. Rewriting the terminator alone keeps
the rewrite checkable by reading it: the body and the value after the loop
are the ones the source wrote.
**Cost.** Every other form stays a `while`: `i <= n`, a step of two, a
bound that calls through a reference the header makes, such as
`while i < v.len()`, a bound whose trapping step follows an effect or
another trap, and a loop with a `break`. A computed bound costs its instructions above the header, once
per entry, and a literal bound one constant. Until IV canonicalization
replaces `i` with the counter, a body that reads `i` carries both. Every converted loop loses its head's comparison, and in the
attention kernel each loop that holds another prepares one more back-edge
move than its `while` did.
**Rejected.**
- Converting `i <= n`, or a step other than one, by computing a bound —
  `n + 1` wraps at the width's maximum, and a step `s` needs the rounded-up
  quotient of `n − b` by `s`, which is the arithmetic of the scalar
  evolution RFC-0066 rejects.
- Converting with a computed exit value — the value after the loop would
  be the pass's arithmetic instead of the body's own `i + 1`, and deriving
  it from the trip count is IV canonicalization's.
- Declining `n > i` — it is `i < n` on integers, and the spelling would
  decide the optimization.
- Running after `code_motion`, which would have hoisted the bound — the
  hoist moves no call, so `v.len()` gains nothing, and it moves the
  header's `/` by control equivalence, so which bound is promoted would be
  that pass's rule instead of this one's. The pass would also need a
  `dce` after it to sweep the comparison, and `code_motion` would see a
  `while` where it now sees a `for`.
- A `no_panic` declaration for promotion — the header's first visit
  already raises the trap the entry raises; a later reader, hoisting from a
  body, may add it.

## RFC-0083: a pure operation computed on every path to it is the value computed first

Status: Proposed

One pass, `optimize::gvn`, numbers the values of a body by one walk of its
dominator tree. It carries a scoped table from an operation and the numbers
of its operands to the value that first computed it. Before the lookup it
simplifies the integer identities below.

1. **What is numbered.** An SSA value is defined once and never written, so
   an operation that reads only its operands' words and writes only its
   destination gives the same value wherever the same operands reach it.
   - `Const` of a word type: an integer, a float, a `bool` or a `char`,
     keyed by its type and its bits. A float is keyed by its bit pattern,
     so `0.0` and `-0.0` are two constants. A constant equal to a
     dominating one takes that one's number, so operations over the two
     are one entry, but its instruction is never replaced. Writing a
     constant costs nothing, while a constant read from a block above is a
     value the machine carries there: where it fills a loop header's
     parameter it is copied in by a move.
   - `BinOp` at any operand type: the machine computes it from the two
     words alone (RFC-0037). A division or remainder that traps traps at
     the dominating copy first, so the copy it replaces is never reached
     with another outcome.
   - `Cast`: total, and a function of its operand's word (RFC-0049).
   - A read of a scalar part of an aggregate held by value: `FieldGet` and
     `ObjectGet` of an object, `TupleIndex` of a tuple, `ArrayIndex` of an
     array. The aggregate is an SSA value and nothing writes it. A part that
     is not a scalar is not numbered, because two reads made one would be
     two owners of it (RFC-0018).

   Nothing else is numbered: a `Take`, a `Ref`, an `Index` and every other
   read of storage, any `UnaryOp`, whose `Deref` reads storage and whose
   `-` and `!` no measured program repeats, every call and `Spawn` and
   `Eval` whatever its effect, `ConstStr`, and a constant of a `String`, a
   list or `()`. An operation with an operand named as a storage by a
   `Ref`, a `Take` or an `Assign`, and a constant written to such a name,
   are not numbered either, since a write through the storage changes what
   the name holds. A block parameter, a body parameter and every value an
   unnumbered operation defines are their own number.

2. **Dominance.** The walk visits the dominator tree in preorder. An entry a
   block makes is visible to the rest of that block and to every block it
   dominates, and is forgotten when the walk leaves the block's subtree. A
   value is replaced only by an entry's value, which a dominating block or
   an earlier instruction of its own block defined, so the replacement
   dominates every use it takes over. Two sibling branches never share an
   entry. A block the entry does not reach is not walked.

3. **Simplification.** Before the lookup, an integer `+`, `-` or `*` whose
   operand is a numbered constant is simplified: `x + 0`, `0 + x`, `x - 0`,
   `x * 1` and `1 * x` are `x`, and `x * 0` and `0 * x` are the `0` the
   operation reads. At width `w` wrapping arithmetic is arithmetic modulo
   `2^w` (RFC-0037), where `0` is the additive identity and the absorbing
   element of multiplication, and `1` is the multiplicative identity, at
   every width and signedness. None of the three operations traps. An
   integer `min(x, x)` or `max(x, x)` whose operands have one number is `x`
   (RFC-0088 rule 1). No float operation
   is simplified: `-0.0 + 0.0` is `0.0`, `inf * 0.0` is NaN, and a
   signaling NaN times `1.0` comes out quiet.

4. **Commutativity.** The operands of a commutative operation are ordered by
   number in its key, so `a + b` and `b + a` are one entry. The instruction
   keeps the order it was written in. At an integer type `+`, `*`, `==`,
   `!=`, `&`, `|`, `^`, `min` and `max` commute, at `bool` `==`, `!=` and `^`, and at a
   float or a `char` `==` and `!=`. A float `+` and `*` are not ordered:
   given two NaN operands the machine returns the payload of one of them,
   chosen by operand order.

5. **Replacement.** Every use of a replaced value, in instructions and
   terminators, reads its replacement, which has the same type. The pass
   removes nothing. What it leaves unread is swept by a `dce` that runs
   right after it. A `Const` is not replaced (rule 1).

6. **Placement.** The pass runs after `lsr`, and so after IV
   canonicalization (RFC-0066 rule 7). Both write arithmetic that rule 3
   simplifies: the canonical `base + k · step`, and a reduction's start and
   step products (RFC-0056). It runs before `forward` and `reorder`. No
   `dce` followed the loop passes before, so one is added after this pass.
   It also sweeps what the header arguments IV canonicalization removes
   leave unread, which that pass swept itself before, and an induction
   variable that `lsr` replaced and that nothing reads but its own
   advance.

**Why.** IV canonicalization writes `base + (counter − at) · step`, which
is `0 + (c − 0) * 1` for a loop that starts at 0 and counts by 1, and equals
`c`. The fold needs two constants, and no pass merged equal values, so
`collatz`'s loop body went from one operation to three. The same
redundancy follows inlining and folding. With this pass the body is `i % 2
== 0` and nothing else.
**Cost.** A dominator tree, one walk and one hash table per body, and one
more `dce`: 0.002 to 0.3 ms of optimization per program on the bench
kernels and examples. No loop of the attention and mandelbrot kernels
gains or loses a move.
**Rejected.**
- A heavier value numbering now — congruent block parameters, a
  partition-based numbering, or partial redundancy elimination. Each is a
  fixpoint with its own correctness argument, and every redundancy measured
  so far lies within dominance. One of them may replace this pass.
- A special case in IV canonicalization that skips `- 0`, `* 1` and `0 +`
  — the same redundancy follows inlining and folding, and two passes would
  each decide one identity.
- Float identities — none is exact bit for bit, as rule 3 states.
- Replacing a constant by an equal dominating one — it saves an
  instruction that costs nothing and makes the machine carry the value
  down to its use. Mandelbrot's pixel loop left with two moves where it
  had none.
- Numbering storage reads and calls — a storage read needs to know that no
  write reaches it in between, and a call's declared effect does not say
  that it returns (RFC-0081 rule 3).

## RFC-0088: a `for` whose body does nothing is a jump to its exit

Status: Proposed

IV canonicalization computes a loop's induction variables from its
counter (RFC-0066 rule 7), and value numbering and the `dce` after it sweep
what that leaves unread (RFC-0083). A loop whose only work was advancing
those variables is then a `for` whose body only jumps back and whose header
carries nothing, and it still runs `max(hi − at, 0)` iterations. One pass,
`optimize::empty_loop`, removes it, which needs the trip count computed
without the loop, and so needs an integer maximum the MIR did not have.

1. **Integer `min` and `max` are MIR operations.** `Min` and `Max` give the
   lesser and the greater of two integers of one width, compared at that
   width's own signedness. Both are total at every width: neither wraps nor
   traps. The validator admits them only at an integer type, with both
   operands and the result of that type. `fold` folds them over two
   constants (RFC-0055). Value numbering orders their operands at an
   integer type and simplifies `min(x, x)` and `max(x, x)` to `x`
   (RFC-0083 rules 3 and 4). The machine runs them at every width.

2. **They are the MIR's own.** The MIR's `BinOp` is its own enum: the
   source's operators and these two. The lowering reaches it through a
   conversion from the parser's `BinOp`, which has no variant for either
   and yields neither. That the source cannot write `min` or `max` is a
   fact of the two types, not a convention every pass keeps.

3. **What qualifies.** A natural loop (RFC-0056) is removed when all
   of these hold:
   - its header ends in `For`, has no parameter, and holds no instruction;
   - every other block of the loop holds no instruction and ends in a jump
     to a block of the loop, so the loop has no edge out but the header's
     exit, and no block of it returns or diverges;
   - its source is a range or an array.

   Every other loop stays exactly as written: one whose body holds any
   instruction, whose header carries any value, or whose body leaves it.

4. **The rewrite.** The header's `For` becomes a jump to its exit, with the
   exit's arguments, and the body's blocks, which no path reaches any more,
   are pruned. Where the exit edge defines the trip count (RFC-0057 rule 9),
   the count is computed at the end of the header and is the jump's first
   argument:
   - for `at..hi` at width `w`, `(max(hi, at) as u64) − (at as u64)`, with no
     cast where `w` is `u64`;
   - for an array, its length as a `u64` constant, read off its type.

   A `dce` follows each removal. It sweeps the bounds the removed terminator
   read, so a loop whose body held only the removed one qualifies in turn.
   An array is no longer moved into a loop, so the `Drop` that
   `drop_insertion` places releases every element at once (RFC-0057 rule 6)
   instead of each element in the body.

5. **The count is exact.** The loop's count is `max(hi − at, 0)` over the
   integers (RFC-0057 rule 9), which lies in `[0, 2^64)` at every width up to
   64. `max` compares at `w`'s signedness, so `max(hi, at) − at` is that count
   over the integers. A cast to `u64` keeps its operand modulo `2^64`, and a
   `u64` subtraction wraps modulo `2^64` (RFC-0037), so the difference of the
   two casts is the count exactly. `max(hi − at, 0)` at `w` is not: at `i8`,
   `-100..100` has `hi − at` wrap to `-56`, and at `u8`, `5..3` has it wrap
   to `254`.

6. **A slice is declined.** The MIR has no instruction that reads a slice's
   length: the `For` terminator is its only reader. A slice `for` whose body
   does nothing stays a loop.

7. **Placement.** The pass runs after the `dce` that follows value
   numbering, and before `forward`. Until IV canonicalization, value
   numbering and that `dce` have run, the body still holds the arithmetic
   they remove, so no loop qualifies earlier. `forward` then collapses a
   header the removal left holding nothing but its jump.

**Why.** A body that holds no instruction writes nothing, calls nothing and
defines nothing a later block reads, and a header that holds none and
carries nothing runs nothing on each test. Running the loop is only
counting, and the count is one `max`, two casts and a subtraction.
**Cost.** A dominator tree and the natural loops per removal and one more
`dce` for each loop removed. A slice loop whose body does nothing still
runs its iterations.
**Rejected.**
- `Min` and `Max` in the parser's `BinOp` — its type would admit two
  operations the parser never produces, and only a convention would keep
  them out of the source.
- An instruction that reads a slice's length, now — it is a new operation
  on a borrowed value for the machine and every pass, for one decline.
- The count `max(hi − at, 0)` at the range's width — it wraps, as rule 5
  shows.
- The count in a wider type — the MIR has no integer wider than 64 bits, and
  no 64-bit type holds every difference of two `i64` or two `u64` values.
