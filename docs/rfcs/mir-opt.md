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
   `Terminator::For { source, body, body_args, exit, exit_args }`, with
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

**Why.** An index loop makes every pass rediscover the traversal the source
stated: the back edge, the bound proof, the induction variable, the borrow's
extent, iteration independence. The machine would also run a condition chain
and a counter move per iteration that the terminator removes.
**Cost.** One terminator arm in every reader of `Terminator`. The enum is
closed, so the compiler enumerates the sites. There are two loop forms:
`while` remains for a loop whose condition is not a traversal.
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
- A general algebraic simplifier (`x * 1`, `x - x`, reassociation across
  operators, distribution) — each rule is its own correctness argument, and
  together they are a rewrite engine with a termination proof to maintain. A
  further rule earns its place by a bench that names it.

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

**Only a strong loop is reduced** (RFC-0066 rule 7). A weak loop's
iterations may run in any order, and its counters are canonicalized instead,
so each iteration computes `i * k + x` from its own `i`. A derived counter
would carry that value from the previous iteration and order the loop. A
strong loop is ordered already, so the reduction costs it nothing.

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
   Float `+` and `*` are inexact merges. `&&`, `||`, `min` and `max` are
   exact operations, but they reach MIR as a short-circuit `Diamond` and as
   extension calls, and their recognition is open.

6. **Weak and strong, by kind.** A loop is weak when every carried
   parameter is an `Iv` or a `Merge`, no instruction of the body carries an
   `Order` (RFC-0013, RFC-0046), and every storage the body writes is its
   `SliceMut` source's, reached through the element (RFC-0057 rule 3).
   Otherwise it is strong. The kind of the carried state decides, never its
   count: how many merges a loop carries is a cost, and cost is the
   lowerer's. An inexact merge is weak and marked inexact, and whether to
   split one is the lowerer's, by its reassociation policy. A merge through
   storage, such as `v.push(x)` in a loop, is a write and strong; what kind
   of merge such an operation is, ordered or not, is declared by the extern,
   not discovered here. A loop left from anywhere but its header, by a
   `break` or a `return`, is strong: the iterations after the one that
   leaves never run. RFC-0057 rule 3's question
   is a weak loop that carries nothing.

7. **One normalization per loop.** Strength reduction (RFC-0056) applies to
   strong loops, which run in order anyway. IV canonicalization applies to
   weak loops, and keeps each iteration computed from its own counter. A
   loop gets one or the other.

8. **A cost table is measured, not written.** At first compilation on a
   target, the runtime measures every operation kind it can emit (each `Op`
   family at its register forms, a spawn, a join, a frame bind) and caches
   the table per target. The cache is invalidated when the runtime binary
   changes. Every later cost is a sum of rows in the table's own unit. The
   table records ranges, and the lowerer compares against the conservative
   end. A target with no table runs every loop on one thread. A loop's cost
   is its body's rows times its trip count, a lower bound.

9. **Regions.** A region is a weak loop's body evaluated as one unit without
   crossing a jump the analysis cannot see through. A jump's target must be
   inside it. It carries the entry and exit values of every variable
   crossing its boundary, its cost per iteration, and its merges with their
   exactness. A loop whose body is not one region is not divided.

10. **The lowerer decides.** `prepare` chooses between a split region
    (chunks as `Heavy` spawns joined through the merges), a region
    evaluated in place, and a region unrolled at the operation level. The
    choice is one comparison of sums of rule 8's table. Where the count is
    known only at run time, the lowerer inserts one dispatch point ahead of
    the loop. It reads `n`, compares against a threshold folded from the
    table at a few concrete points, and continues into the split chain
    (cold, laid out of line) or the in-place chain (hot). A loop never
    worth splitting costs one compare. The lowerer splits nested regions at
    one level, and the level is a comparison over rule 8's table.

**Why.** A normal form that holds on every target is the same program
everywhere, so writing it into MIR decides nothing a target could decide
better. A shape chosen for a target is a guess about the lowerer, which
knows the target, the runtime and the actual `n`. Strength is read from the
kind of the carried state because the kind says whether iterations can be
reordered, and a count says only what reordering costs. The author writes
the loop, and the system finds the parallelism from facts the checker
already establishes.
**Cost.** Three analyses built per body and read by every loop pass. A
measurement step per target and a family of split operations (chunk, spawn,
join through a merge). A loop's normalization is chosen by its strength, so
a change in the analysis moves a loop between two passes. RFC-0057's and
RFC-0064's analyses become inputs whose promises must stay stable.
**Rejected.**
- Unrolling, tiling, blocking or permutation as MIR passes — a lowerer's
  guess written into the program's meaning, and nothing undoes it when the
  guess is wrong for a target.
- Strength decided by the number of carried values — a count is a cost, and
  a loop with one recurrence is ordered where a loop with ten sums is not.
- Constant costs in source — true on one machine on one day.
- Scalar evolution alone — it answers the trip count and nothing when that
  fails.
- A `while` trip count derived from its recurrence — the door to general
  scalar evolution. A `while` is promoted to `for` only by a recognizer that
  is exact, or it stays undivided.
- An explicit `par for` — the facts the split needs are the checker's. Where
  they are not established, the author is told why, not asked to assert them.

**Open.** How `&&`, `||`, `min` and `max` are recognized as merges. Whether
rule 9's jump-boundary conditions reduce to effect boundaries alone. Whether
the machine offers an explicitly reassociable float reduction, which is a
language decision. What a failed join drops and in what order, which is
answered in rule 10's operation family.
