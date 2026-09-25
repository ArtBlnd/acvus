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
   `Terminator::For { source, stages, exit, exit_trip, exit_args }`, with
   `source` one of `Slice(s)`, `SliceMut(s)`, `Array(a)` or `Range { at, hi }`.
   The terminator is the condition. No instruction writes `index < len`,
   `i = i + 1` or the element read. The terminator fills the leading
   parameters of the body block, the first of `stages`, (`elem`, `index`; for a `Range` the element is the
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
   region it would otherwise be, with no compare added. The loop's own exit
   edge is a part (`exit`) the region runs only where its test fails, so it
   cannot overwrite what a `break` moved.

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

The value is the one the machine computes, at the operand's width
(RFC-0037). A trapping `+`, `-` or `*`, the source's, gives its exact
result, and a trapping shift takes an amount below the width; a wrapping
one, which only a pass writes, wraps and takes a shift's amount modulo the
width. A comparison reads both operands at the width's own signedness. A
float comparison compares bit patterns.

Nothing is folded where the machine would panic: a trapping operation whose
exact result does not fit the width, a trapping shift by the width or more,
a zero divisor, or a quotient that leaves the width. The operation stands,
and the program panics where it did. A float operation whose result is NaN is not folded,
because a folded NaN would carry the compiler's bit pattern where the
machine's belongs.

Two constants that are not adjacent still join when one associative and
commutative operator of one kind at one integer width separates them and
the intermediate has exactly one use: `(x + 1) + 2` becomes `x + 3`. The
intermediate becomes the joined constant where it stood, so no new
definition is placed. A trapping join keeps the trap (RFC-0037 rule 3):
it happens only where the joined constant fits, where `(x op a) op b`
and `x op (a op b)` trap on the same `x` with the same text, which is a
`+` of two constants of one sign and a `*` of two at least one, and where
the two stand in one block with nothing between them that can end the
run, so the trap moves to the outer operation's place on exactly the runs
that reached the inner one. `(x + 3) + -4` and `(x * -1) * -1` are left
as written.

**The optimizer introduces no operator the chain recognizer does not
claim.** A fold replaces an operation with its constant, or moves a constant
from one operand of one operator to another. The operators a body holds
after the fold are the ones the source wrote.

The fold's value and the machine's are one claim, and a test executes it.
Each edge the surface can write (an overflowing width, the widest constant,
a negative dividend, a NaN) runs once with foldable operands and once with
operands from the page, and the two must produce the same register word or
the same trap.

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

**Only a counter expression read in one `InOrder` join is reduced**
(RFC-0066 rule 7), after the stages are written; its step joins that join.
Elsewhere each iteration computes `i * k + x` from its own `i`. A derived
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
multiplication distributes over addition exactly. The start, the step and
the latch's advance are the pass's own operations, so they wrap: the start
runs before the first iteration and the advance after the last one, on
values the program never computed, and a trap there would end a run the
program defines. The program's `i * k + x` traps where either operation
overflows, and a reduction keeps those traps (RFC-0037 rule 3): a check of
each at its place costs what the reduction sheds, so by the count rule a
trapping product is reduced only where word bounds, `k` and `x` put both
operations inside the width on every iteration. A pass's own wrapping
`i * k + x` is reduced as well. In floating point the
accumulated form is a different number, and on mandelbrot's grid 34 of 20000
pixels reach a different escape count. A multiplication that does not run
on every iteration is not reduced.

The pass runs after the stages are written (RFC-0089 rule 6) and before
`bce`. `code_motion` has put `k` and `x` above the header and left the
preheader a block of its own. No value numbering follows, so the pass writes
its start and step already folded where their operands are constants: a
start `0 * k + x` is `x`.

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
iteration hands the next.

1. **Normalization is the optimizer's; the target's shape is the
   lowerer's.** A MIR pass may rewrite a loop when the result is the same
   program on every target: IV canonicalization, collapsing a rectangular
   nest, separating a merge's join from the body, moving an exit that
   leaves a nested loop's parent into the nested loop's own exit with a
   verdict the parent branches on after it. Unrolling, tiling and
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
   source's length, a storage's length on the loop's entry, or `+`, `−`, `×` or `max` of terms. It denotes an
   integer. It is kept as found and evaluated where the lowerer knows the
   atoms, never simplified. A value is invariant in a loop when it is
   defined outside the loop, is a word `Const`, which reads nothing and
   writes the same word each time (RFC-0056), or is an operation of
   invariant operands reading nothing the loop writes, left where it
   stands.

4. **Affine values.** A value `v` is affine in a loop when
   `v = base + k·step` over the iteration number `k`, with `base` and `step`
   terms of rule 3. A `for`'s counter is affine from its terminator: a
   range's element is `{at, 1}` and a slice's or an array's index is
   `{0, 1}`. A header parameter entered with `b` whose back edges all send
   `p + c`, `c` invariant, is `{b, c}`. `a·v`, `v + b`, `v − b` and `b − v`
   of an affine `v`, `a` and `b` invariant, are affine. So is `len(s)` read
   before a call every iteration makes exactly once, when that call's
   postcondition states `len(s) = old(len(s)) + c` and nothing else in the
   loop changes `s`'s length: `{len(s) on entry, c}` (RFC-0082 rule 6). Only integers are affine: `+` and
   `*` of either kind are exact modulo `2^width` on every run past them
   (RFC-0037 rule 3). The analysis is one loop deep.

5. **Carried state.** Each header parameter is exactly one of `Iv`, affine
   by rule 4, or state, anything else. A state's law, when it has one, is
   what RFC-0093 reads in `analysis::loop_deps`, the one reader of
   laws; `analysis::carried` classifies and reads none.

6. **Order is per target, not per loop.** What a loop changes and how the
   change is ordered is stated per target by the stages of RFC-0089: pure
   work runs apart, and each join is `Disjoint`, `AnyOrder` or `InOrder`.
   A carried value is a target like any storage. A loop whose every target
   joins `InOrder` with no law runs in order; that is a cost, not a
   different form. RFC-0057 rule 3's question is a loop whose only target
   is its `&mut` source's element, joined `Disjoint`.

7. **One normalization per induction variable.** IV canonicalization
   rewrites every `Iv` of a `for` that anything besides its own step reads
   into `base + k·step` in the body, `k` read off the counter, and
   `base + trip·step` where it is read after the loop, from the count the
   exit edge defines (RFC-0057 rule 9). Its arithmetic wraps (RFC-0037
   rule 3): its operands are right only modulo `2^width`. Its trapping
   step stays a `Check` over the rewritten value, carrying nothing; a
   step that cannot trap, a range's own counter or one word bounds keep
   in the width, needs none. A rewritten `Iv`
   is carried no longer, so it is no target and orders nothing. The choice is per
   variable and reads nothing the stage pass decides. A header parameter
   every back edge sends `f(e)`, `e` the element the iteration read at
   the counter of a slice or array source and `f` pure work on `e` and
   invariants, holds `f` of the element at `k − 1` from the second
   iteration on and its entry value on the first; the pass reads it so,
   and it carries nothing. A call rule 4 makes `{len(s) on entry, c}` is
   read as that term from `len(s)` read above the header, and removed,
   where the call cannot trap (RFC-0048 rule 8). After
   the stages are written, strength reduction (RFC-0056) reduces a counter
   expression whose readers all sit in one `InOrder` join, and its step
   joins that join. A `while` is declined, since it states no count; a later
   exact recognizer makes it a `for`, and this pass applies to that
   unchanged.

8. **Cost is one scalar per loop, from a static table.** The unit is a
   tick. A backend holds one table, a row per operation family (arithmetic,
   compare, load, store, allocation, local call, extern call, spawn, merge,
   a chunk's dispatch, one buffered element), fixed in the backend. An
   extern may state its own weight (`cost = N`, or `heavy`); one that
   states none weighs its family's row. A loop's work `W` is the least
   weight one iteration of the stages that run apart, its free and
   `Disjoint` ones, can take: summed over their blocks, the lighter arm of
   a branch, an inner loop at its least trip count, zero where that is
   unknown or the loop can leave early; an operation with no row weighs
   nothing. Its overhead `O` is one chunk's dispatch (a spawn where the
   body suspends), one merge, and one buffered element per boundary. The loop splits when `n · W > K · O`, where `n` is the
   trip count, known on entry, and `K` is a high constant, so a wrong
   estimate errs toward running in place. `⌈K · O / W⌉` folds into one
   constant, and the run makes one compare. With no table, or no stage
   that runs apart, a loop runs in place. No MIR pass reads a cost, and none is
   measured at run time: another dialect's lowering decides statically.

9. **Regions.** A region is a stage of a `For` (RFC-0089) evaluated as
   one unit, without crossing a jump the analysis cannot see through. A
   jump's target must be inside it. A loop whose stages are not regions is
   not divided.

10. **The lowerer runs the stages.** RFC-0092 states how.

**Why.** A normal form is the same program on every target; a shape
chosen for one is a guess about the lowerer, which knows the target, the
runtime and `n`. Order is read from the
operations' declarations, which say whether iterations can be reordered;
a count says only what reordering costs.
**Cost.** Three analyses built per body and read by every loop pass. A
table per backend and a family of split operations. A change in who reads
an induction variable moves it between two forms. RFC-0057's and
RFC-0064's analyses become inputs whose promises must stay stable.
**Rejected.**
- Unrolling, tiling, blocking or permutation as MIR passes — a lowerer's
  guess written into the program, which nothing undoes.
- A table the runtime measures and caches — one machine's timing, and a
  cache to invalidate.
- Scalar evolution alone — nothing when the trip count fails.
- Leaving an `Iv` read after the loop carried — it keeps the dependence
  the normal form removes.
- A `while` trip count derived from its recurrence — the door to general
  scalar evolution. A `while` is promoted to `for` only by a recognizer that
  is exact (RFC-0081, RFC-0094), or it is a pull loop (RFC-0089 rule 1), or
  it stays undivided.
- An explicit `par for` — the facts the split needs are the checker's.
- A reassociable float reduction — it fixes a rounding every later lowering
  would have to keep.

**Open.** Whether rule 9's jump-boundary conditions reduce to effect
boundaries alone.

## RFC-0089: a `for` is a chain of stages no dependence cycle crosses, and what each stage is the IR already says

Status: Proposed

An iteration waits for another only through a token the IR already holds.
`For` writes where the body is cut and nothing else; which stage serializes
what, in which order and by which law, is read from the tokens and the
operations' declarations. How a stage runs is the lowerer's (RFC-0092).

1. **The terminator.** `Terminator::For { source, stages, exit, exit_trip,
   exit_args }`, with `source`, `exit`, `exit_trip` and `exit_args` as
   RFC-0057 states them, and `stages` the entry blocks of the body's stages
   in body order. The body block is the first, and its parameters are the
   element and the counter. Each stage's last block jumps to the next
   stage's entry, and the last one is the one latch, which `continue`s
   join. Every pass that reads edges reads the body as a sequential
   program; running in place is the loop as written, the chain being the
   body of the region RFC-0057 rule 3 builds. A value reaches a later
   stage by dominance.
   A pull loop — a `while` whose header calls an extern `next(&mut it)`,
   tests its `Option` and leaves only there — ends its header in
   `Terminator::While { cond, stages, exit, exit_args }`, its body
   `stages.body()` as a `For`'s: the header,
   its call and its test are the first stage and the control token's
   cycle, in order (rule 5), and the payload reaches later stages by
   dominance. Its count is unknown on entry, so its cost (RFC-0066 rule 8)
   is in place until a declaration bounds its pulls.

2. **Tokens.** One iteration orders another only through:
   - an `Order` value (RFC-0007);
   - a header parameter, which carries a value to the next iteration;
   - a storage the body writes or lends `&mut`, a context among them
     (RFC-0025); the `&mut` source's element is written at its counter's
     slot;
   - the control token, the fact that an iteration exists: every
     iteration has it from the start when every exit is the header's, and
     otherwise receives it from the stage its predecessor could leave from.

   A slot an iteration defines and drops within itself carries nothing to
   the next one and is no token; nor is a header parameter every back
   edge sends one constant: past the first iteration it holds that
   constant.

3. **No cycle crosses a boundary.** A dependence cycle through a token lies
   inside one stage. A stage that holds one is that token's join; a stage
   that holds none is free. `validate` checks this, rule 5 and rule 1's
   shape from the IR alone.

4. **What a stage is, is read.** `analysis::loop_deps` is the one place
   that computes, per stage, its cycles and, per cycle, its tokens, order
   and law; the validator, the passes and the lowerer read it.
   - Order is `Disjoint` when every place the cycle writes in its storage,
     and every place the loop reads there, lies under one path component
     that is `a·k + b` in the counter `k` (RFC-0066 rule 4), one `a ≠ 0` at
     every access and bases invariant in the loop that differ by constants,
     none a nonzero multiple of `a`: the terms are exact, so two iterations
     never touch one place. The `&mut`
     source's element is `a = 1, b = 0`. An extern call reaches only the
     places its declaration states, and all a reference argument lends it
     when it states none; `AnyOrder` when every operation in it commutes (RFC-0013), is
     joined by a `merge` (RFC-0007 rule 7), or combines through a
     commutative law; `InOrder` otherwise. A float law is `InOrder`:
     joining in arrival order changes the rounding.
   - A law is specialization, not permission: `Op`, `Call` (an associative
     extern, lifted over `Option` when it states no identity), `Fold`
     (RFC-0082 rule 3) or `Order` (RFC-0007 rule 7). The lowerer may
     combine inside a chunk and join the partials, in chunk order when the
     law does not commute; a cycle without one still runs in its order.
   - A law is read from what the cycle computes, not from how it is
     spelled, by the readings of RFC-0093.

5. **Exits and effects.** An exit other than the header's is the control
   token's cycle: the stage it leaves from passes the control token to the
   next iteration, in order, and every cycle in that stage joins it, so an
   exiting stage is never free and never out of order. An operation with
   an effect runs only once its iteration holds the control token, so no
   effect is issued ahead of an exit; one without an effect may run ahead
   and be discarded when it finishes on every run: it holds no `while`
   and no call of a local function, a `for` in it finishes when its body
   does, and every extern it calls states `returns` (RFC-0082 rule 8).
   `analysis::raise::finishes` is that test, and the removal of an unused
   call reads it too (RFC-0048 rule 8) with what the call graph shows of
   each local function; run-ahead reads it knowing no callee, since rule 3
   checks this rule from one body. A
   trap it raises in an iteration past the exit is discarded with it.
   Such a loop's trip count is only a bound, and its cost (RFC-0066 rule
   8) says `n ≤` that bound: work past the exit is spent and discarded.
   A run apart reports the trap least in the order (iteration, stage), and
   a trap releases nothing (RFC-0048 rule 8). The lowerer runs an `Array`
   source in place until the release of elements scattered over chunks
   exists.

6. **Who cuts.** A pass after IV canonicalization (RFC-0066 rule 7) and
   after every pass that moves or merges the body's instructions writes
   the boundaries, before strength reduction. It keeps each cycle whole,
   puts work the cycle does not read after it and work it reads before it,
   and duplicates nothing: an instruction two cycles share joins them. The
   terminator names no token, so a pass that removes a `merge` or a phi
   leaves a stage with no cycle, free, and cutting again merges the
   boundaries left empty. A loop the pass cannot cut is one stage.

**Why.** A token is a value, so every pass keeps it right while it moves,
merges and deletes instructions; a mark on the loop is a second statement
of the same fact, which a pass can leave stale. Where the body is cut is
the one fact tokens do not state, and recomputing it is the lowerer's
whole burden, so the IR writes it. What follows is that a pass that adds a
`merge` where order does not matter, or removes an order edge, frees
stages without touching them, and validation checks one invariant.
**Cost.** A terminator arm in every reader. One analysis every loop reader
consults. The cutting pass, rerun after a pass that frees a stage.
**Rejected.**
- Order, law and targets written on the terminator — a second statement
  of what the tokens say, which a pass can leave stale, and a terminator
  that names a token keeps its phi alive against elimination.
- No boundaries, every cut computed by the lowerer — the lowerer re-solves
  every loop, and no pass can rely on a cut.
- Parts joined by a law, with writes left in the body — independence
  became a proof about which element each write reaches.
- Arms that fork and all run — every pass reads successors as
  alternatives.

**Open.** Each runs as `InOrder` today; the tiers weigh ease against reach.
- First: named commutation sets in place of `commutes: bool`.
- After the executor's pipeline: a cycle split by key; a `Stream` source
  for `while` loops.
- When a use asks: speculative exits; an action law for heavy work inside
  a cycle. (A scan law is RFC-0093 rule 8.)

## RFC-0093: a cycle's law is read from what it computes

Status: Proposed

`analysis::loop_deps` reads a cycle's law (RFC-0089 rule 4) from the
operations the cycle holds, not from how the script spells them. Each
reading below names what it recognizes; a cycle no reading covers has no
law and runs in its order.

1. **Through what a law is read.** Through a branch whose other arm leaves the token as it was
   (the law's identity stands in), through a compare and select of
   integers, which is their `min` or `max` by the order's definition,
   through a storage the cycle loads, combines and stores whole, and
   through a nested loop: a token that enters one as a header
   parameter's entry value and that its exit hands back, whose cycle
   there has law `L` and nothing else there reads, is combined through
   `L` with the nested loop's run from `L`'s identity. A branch is also
   a switch; an assignment is read as the chain of all the iteration's
   assignments to the token.
2. **Bool.** `||`, `&&` and `!=` over `Bool` are the language's laws, with
   identities `false`, `true`, `false`; an arm that fixes the token to
   one of them reads as that law.
3. **Several tokens.** A cycle of several tokens whose steps read no other token has the
   product of their laws, `AnyOrder` when each commutes.
4. **Option.** A switch on an `Option` token whose `None` arm sends `Some(y)` and
   whose `Some(b)` arm sends `Some(b ⊕ y)` is `⊕` lifted over `Option`.
5. **Last, first, extremum, reset.** An arm that sends a token a value reading none of it, the other arm
   leaving it, is `last`; a compare and select by a strict order that
   also carries other tokens is the left-biased maximum or minimum;
   both are associative, not commutative, `InOrder`. `first` is `last`
   guarded by a `||` token the arm sets. An arm taken only at `k = 0`
   that sends a value reading none of the token resets it: its law's
   join takes no entry value.
6. **An inverse pair.** A storage every access of which in an iteration
   is `f(s)` giving an `Option`, its payload taken by the language's own
   unwrap or `Some` match, then `g(s, x)` of that payload, `f` stating
   `inverse = g` (RFC-0082 rule 3), holds that payload as one cell: a
   pass reads `f(s)` above the header, the iterations read the cell, and
   `g` follows the exit when the cell holds a value. The storage is then
   no token; an unwrap's trap stays at the first iteration that reaches
   it.
7. **A sentinel.** A token compared with a constant that no write in the
   loop can send, as the interval domain proves, is its own `||` guard
   for rule 5's `first`.
8. **A scan.** A cycle whose token has a law by the readings above and
   whose values are read outside the cycle (a partial: the token's value
   before or after an iteration's update, read by a later stage, a
   push or a compare) has that law as a scan: the facts say `scan` beside
   the law. The readers take the partial as a value the cycle hands them,
   in order; a reader that writes the token is in the cycle, not a
   reader. A read of the storage at the previous iteration's affine place
   (`v[i - 1]` where the cycle stores `v[i]`) is a read of the previous
   partial. A token updated as `a·y + x`, with `a` and `x` read from the
   iteration and not from `y`, has the law of affine maps, which compose
   as `(a₁·a₂, a₂·x₁ + x₂)` with identity `(1, 0)`, at an integer width;
   a float one stays in order.

**Why.** A law stated on the loop would be a second statement of what the
operations already say; read from them, it follows every pass that
rewrites them.
**Cost.** Each reading is a recognizer in `loop_deps`, with a test that
fails when it is removed. A scan costs its lowerer a second pass over
each chunk (RFC-0092 rule 5).
**Rejected.**
- Keeping the readings in RFC-0089 — they grow with every law, and RFC-0089
  states what a stage is.

## RFC-0094: a `while` whose exit is a linear bound on one counter of a constant step is a range `for`

Status: Proposed

RFC-0081 converts the `while` that counts by one up to `n`. The same
conversion is exact for a few more spellings of a counted loop, where no
bound needs rounding beyond one constant division and nothing can leave
the width. Every condition of RFC-0081 rule 1 other than the comparison and
the step holds; its rules 2 and 3 apply unchanged.

1. **Up to and including.** `c` is `i <= n` and the interval domain
   (RFC-0047 rule 7) shows `n` below the width's maximum: the range is
   `b..n + 1`, the `+ 1` the pass's own and exact.
2. **Counting down by one.** `c` is `i > n` (or `n < i`) and every back edge
   sends `i − 1`: the header becomes `For { source: Range { at: n, hi: b } }`,
   which counts the iterations and computes no bound. The loop's `i` stays
   its own header parameter; on the `k`-th visit it is `b − k`, which IV
   canonicalization reads from the counter.
3. **A constant step.** Every back edge sends `i + s` for a word constant
   `s ≠ 1` and `c` compares `i` against `n` in the direction `s` moves:
   the trip count is `⌈(n − b) / |s|⌉` where `i` has not already passed
   `n`, else zero, computed above the header from the difference at the
   width read unsigned and one constant division, so it neither traps nor
   wraps. The header becomes a range over that count; `i` is read from the
   counter as `b + k·s`.
4. **An offset compare.** `c` is `i + c₀ < n` for a word constant `c₀`:
   the range is `b + c₀ .. n`, whose start is the header's first visit's
   own step moved to the entry (RFC-0081 rule 3).
5. **A borrow in the header.** A `Ref` the header makes of a slot that no
   instruction of the loop writes or lends `&mut`, and that no call of the
   loop reaches through another loan (RFC-0082 rule 7), is a deterministic
   step of RFC-0081 rule 3. The header's own shared borrow passed to its
   bound's call (`v.len()`) is a read of the slot.
6. **A back edge whose arguments decide the test leaves.** Where the
   header's condition is a chain of `&&` and `||` over tests whose own
   steps are the header's (RFC-0081 rule 3), and a back edge sends a
   header parameter a constant under which the chain is false whatever
   the other tests give (`found = true` for `… && !found`), that edge,
   followed through blocks that only pass their parameters on, goes
   to the exit instead of the header, carrying for each of the exit's
   arguments what the header would have sent it on that visit. It does
   so only where every instruction the header runs on that visit before
   the deciding test is one an unused removal may drop (RFC-0048 rule 8):
   it has no effect, cannot trap, and returns. A call qualifies only by
   RFC-0082 rule 9's `total`. The skipped visit then computes nothing
   observable, and the exit sees the values it saw. A parameter every
   entering edge sends that constant's opposite, and every remaining back
   edge that opposite or the parameter itself, is invariant, so its test
   folds and the rest of the chain is
   the condition rules 1 to 5 read.
7. **An exit from the body.** RFC-0081 rule 1 declines a loop with an edge
   out of a block other than the header. That condition is lifted for an
   edge to a block outside the loop, the edge rule 6 makes included. The
   converted `For` keeps the edge and its arguments, and its exit carries
   no trip count (`ExitTrip::Absent`). RFC-0081 rule 4's argument is per
   visit: the `k`-th visit holds `b + k` in both loops, so a body edge
   taken on that visit leaves both with the same values. The count the
   range states is then an upper bound, and the cost reads it as one
   (RFC-0066 rule 8, `n ≤`). The converted loop keeps the region its
   `while` ran as (RFC-0057 rule 3): an edge out of the body leaves the
   region and continues where it went, as the exit does.

**Why.** Each form is a counted loop the script wrote without the `for`
that states it; the conversion is exact because the bound is a word the
pass can compute with no rounding it cannot state and no overflow.
**Cost.** Three recognizer cases and one computed trip count. An edge
rule 6 moves skips one header visit's pure steps, which the removal rule
already admits dropping.
**Rejected.**
- A step that is not a word constant — the division's divisor and sign
  are unknown, which is scalar evolution.
- A quadratic or otherwise non-linear exit condition — its count is a
  root, not a quotient.
- A normal form that reads a flag by its name or its spelling
  (`found`, `done`) — rule 6 reads a constant an edge sends and the test it
  decides, whatever the slot is called.

## RFC-0092: a lowerer runs a loop's stages as a token pipeline, and the executor decides how to wait

Status: Proposed

The stages of RFC-0089 and the cost of RFC-0066 rule 8 are what a lowerer
reads to run a loop apart. No lowerer builds this yet; every loop runs in
place. This is the contract a lowerer is held to.

1. **Chunks pass the stages in order.** A split loop is cut into chunks of
   iterations, bounded in flight, which pass the stages in order: a free
   stage runs a chunk on arrival; a cycle's stage admits one holding its
   token, in chunk order under `InOrder`, one at a time under `AnyOrder`,
   none under `Disjoint`. The control token passes with the exiting
   stage's.
2. **A law combines inside a chunk.** With a law a chunk combines first and
   the stage joins the partial, in chunk order when the law does not
   commute. Where the program's operation traps (RFC-0037 rule 3), a
   partial combined from the identity may overflow where the program's
   own order does not; the lowerer combines partials with the wrapping
   operation and traps where the program would.
3. **The executor decides how to wait.** In a `Sync` body the split is one
   synchronous executor call, which decides how to wait (RFC-0046 rule 1);
   a suspending body spawns and awaits chunks.
4. **One level.** Running in place is always admitted; chunks never split
   again.
5. **A scan rescans.** A cycle whose law is a scan (RFC-0093 rule 8) runs
   in two passes: each chunk combines its updates from the law's
   identity; the chunks' totals are combined in chunk order from the
   entry value into each chunk's offset; each chunk then runs its cycle
   again from its offset, handing its readers the program's own partials.
   The second pass computes every partial the program computes, with the
   program's operations, so it traps exactly where the program does; the
   first pass and the offsets combine with the wrapping operation (rule 2).

**Why.** The stages say what may run apart and the cost what it is worth;
how chunks move is the lowerer's, and stating it apart from the analyses
keeps each RFC to what its readers use.
**Cost.** A family of split operations in each backend that splits.
**Rejected.**
- Keeping the pipeline in RFC-0066 — it mixes the analyses with a
  lowerer's contract no pass reads.

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
     declines, except an edge RFC-0094 rule 7 admits;
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
   computes moves to the end of the entering block, because the machine
   reads a range's bounds on the entering edge (RFC-0057 rule 7) and a
   `for` header holds no instruction. A header instruction that is not a
   step of the bound ran on every visit, the last one included, so it
   moves to the head of the body block and to the head of the exit block,
   in the header's order: it runs exactly as often, before the same body
   and the same exit. The comparison it preceded is pure, so running it
   after the range's own test changes nothing observable.

3. **A computed bound is the header's first visit, moved to the entry.**
   The header runs on every entry before any body block. A step computed
   from word literals, values defined outside the loop and earlier steps
   gives the same value, and raises the same trap, on every visit, when
   the step is deterministic:
   - `+`, `-`, `*`, `/` and `%` at an integer width. The machine computes
     each from its two words at the width (RFC-0037): the program's `+`,
     `-` and `*` trap with Rust's texts where the exact result leaves the
     width, and a pass's wrap; at every width
     `/` panics with `attempt to divide by zero` on a zero divisor and `%`
     with `attempt to calculate the remainder with a divisor of zero`; at a
     signed width `MIN / -1` panics with `attempt to divide with overflow`
     and `MIN % -1` with `attempt to calculate the remainder with
     overflow`; an unsigned width has no other failure.
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
   instructions changes nothing observable. The entering block ends in the
   jump to the header, so the moved steps run on exactly the paths the
   first visit ran on (RFC-0048 rule 8), and every later visit repeated
   them. A trapping `+`, `-` or `*`, a `/`, a `%`
   and a call can trap, since `pure` does not say that a call returns:
   `unwrap` is `pure` and panics. So when the bound holds one, no
   instruction before its last such step in the header, other than a step
   of the bound, may trap; only a constant, a reference, a cast, a `merge`
   and an arithmetic operation that cannot trap qualify. An effect there does not decline: a trap is not
   ordered with effects (RFC-0048 rule 8). Then the entry raises exactly
   the trap the first header visit raised, on an entry that runs the body
   zero times too, and a bound with no trapping step moves freely. One rule
   covers the program's arithmetic, `/`, `%` and a `pure` call; which of
   them trap and when does not enter it.

4. **The two loops are one program at every entry.** Both start at `b` on
   each entry, so the `k`-th visit of the header holds `b + k` in both. The
   range runs its body while its counter is below `n`, by the comparison
   `i < n` makes at the same width, and advances by one. `n ≤ b` runs zero
   times in both. Each advance runs only after its counter compared below
   `n`, so the range's never wraps and the program's `i + 1` never traps
   (RFC-0037 rule 3), and `n` at the width's maximum ends both loops at
   `n`.

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
**Cost.** A form neither this rule nor RFC-0094 states stays a `while`: a
bound whose trapping step follows another trap, and a flag whose deciding
edge skips a header step that may trap or not return. A computed bound costs its instructions above the header, once
per entry, and a literal bound one constant. Until IV canonicalization
replaces `i` with the counter, a body that reads `i` carries both. Every converted loop loses its head's comparison, and in the
attention kernel each loop that holds another prepares one more back-edge
move than its `while` did.
**Rejected.**
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
  body, may add it. RFC-0082 rule 9's `total` is one, read by the removal
  of an unused call and not by promotion.

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
     words alone (RFC-0037). A division, a remainder or a trapping `+`, `-`
     or `*` that traps traps at the dominating copy first, so the copy it
     replaces is never reached with another outcome. The key is the whole
     operation, its kind included, so a trapping and a wrapping `+` over
     the same operands are two entries.
   - `Cast`: total, and a function of its operand's word (RFC-0049).
   - A read of a scalar part of an aggregate held by value: `FieldGet` and
     `ObjectGet` of an object, `TupleIndex` of a tuple, `ArrayIndex` of an
     array. The aggregate is an SSA value and nothing writes it. A part that
     is not a scalar is not numbered, because two reads made one would be
     two owners of it (RFC-0018).

   Nothing else is numbered: a `Take`, a `Ref`, an `Index` and every other
   read of storage, any `UnaryOp`, whose `-` and `!` no measured program
   repeats, every call and `Spawn` and
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
   operation reads. At width `w` arithmetic is arithmetic modulo `2^w`,
   where `0` is the additive identity and the absorbing element of
   multiplication, and `1` is the multiplicative identity, at every width
   and signedness. Each identity holds at either kind (RFC-0037 rule 3):
   none of the results leaves the width, so a trapping one never traps
   there. An integer `min(x, x)` or `max(x, x)` whose operands have one
   number is `x` (RFC-0088 rule 1). No float operation is simplified: `-0.0 + 0.0` is `0.0`, `inf * 0.0` is NaN, and a
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

6. **Placement.** The pass runs after IV canonicalization (RFC-0066 rule
   7), whose canonical `base + k · step` rule 3 simplifies, and before
   `forward`, `reorder` and the stages, since it merges values. Strength
   reduction runs after the stages and folds its own start and step
   (RFC-0056). A `dce` follows this pass; it sweeps what the header
   arguments IV canonicalization removes leave unread.

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
   - every other block of the loop ends in a jump to a block of the loop,
     so the loop has no edge out but the header's exit, and no block of it
     returns or diverges;
   - those blocks hold no instruction but constants, integer casts,
     wrapping `+`, `-` and `*`, and `Check`s of a `+` whose checked value
     advances by the added step on each iteration (rule 8);
   - its source is a range or an array.

   Every other loop stays exactly as written: one whose body holds any
   other instruction or a check rule 8 does not close, whose header carries
   any value, or whose body leaves it.

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
   over the integers. A cast to `u64` keeps its operand modulo `2^64`, and
   the subtraction is the pass's own, so it wraps modulo `2^64` (RFC-0037
   rule 3): the difference of the two casts is the count exactly, and at
   `i8`, `-5..3` passes through a wrap on the way. `max(hi − at, 0)` at `w`
   is not: at `i8`, `-100..100` has `hi − at` leave the width at `200`, and
   at `u8`, `5..3` has it leave it at `-2`; a trapping `−` would trap on a
   loop the program runs, and a wrapping one gives `-56` and `254`.

6. **A slice is declined.** The MIR has no instruction that reads a slice's
   length: the `For` terminator is its only reader. A slice `for` whose body
   does nothing stays a loop.

7. **Placement.** The pass runs after the `dce` that follows value
   numbering, and before `forward`. Until IV canonicalization, value
   numbering and that `dce` have run, the body still holds the arithmetic
   they remove, so no loop qualifies earlier. `forward` then collapses a
   header the removal left holding nothing but its jump.

8. **A step's check is closed at the header.** IV canonicalization leaves
   a `Check` of each step that can overflow (RFC-0066 rule 7), so the loop
   of `for i in 0..@n { s = s + 3 } s` holds that check and the arithmetic
   from the counter it reads. A check `v + c` qualifies where `v` is a value
   of the body that gains `c` modulo `2^w` each time the counter gains one,
   read off the constants, casts of the counter and wrapping `+`, `-` and
   `*` the body holds. It then becomes, at the end of the header,
   `CheckSteps { from: v₀, step: c, count: n }`: `v₀` is the body's
   computation of `v` copied with the counter at its first value, and `n`
   the count of rule 4. `CheckSteps` ends the run with `+`'s trap where
   `v₀ + n·c`, computed exactly, does not fit the width.
   - **Exact.** `v₀` is a value of the width. On a run whose checks below
     iteration `k` passed, `v₀ + k·c` fits, so `v` there is that integer
     and the check at `k` traps exactly where `v₀ + (k+1)·c` does not fit.
     That is monotone in `k` and `v₀` fits, so some iteration below `n`
     traps exactly where `v₀ + n·c` does not fit.
   - **The same trap.** Nothing else in the body can end the run or be
     observed, so no other trap or effect precedes the check inside the
     loop, and every qualifying check traps with `+`'s text: which of two
     traps first does not show. The trap is at the loop's place.
   - The machine computes `v₀ + n·c` in `i128`; a product that leaves
     `i128` has magnitude at least `2^127`, and no width holds the sum.

**Why.** A body that holds no instruction writes nothing, calls nothing and
defines nothing a later block reads, and a header that holds none and
carries nothing runs nothing on each test. Running the loop is only
counting, and the count is one `max`, two casts and a subtraction. A body
of step checks is counting too: its one observable is the trap, and rule 8
keeps that trap.
**Cost.** A dominator tree and the natural loops per removal and one more
`dce` for each loop removed. A slice loop whose body does nothing still
runs its iterations.
**Rejected.**
- `Min` and `Max` in the parser's `BinOp` — its type would admit two
  operations the parser never produces, and only a convention would keep
  them out of the source.
- An instruction that reads a slice's length, now — it is a new operation
  on a borrowed value for the machine and every pass, for one decline.
- The count `max(hi − at, 0)` at the range's width — it leaves the width,
  as rule 5 shows.
- The count in a wider type — the MIR has no integer wider than 64 bits, and
  no 64-bit type holds every difference of two `i64` or two `u64` values.
- Keeping a loop whose body is a step's check — it runs `n` iterations of
  arithmetic to find a trap one exact sum decides.
- The closed check as MIR arithmetic — the MIR has no integer wider than 64
  bits, and a sign-dispatched division bounding `n` is several operations
  and branches for what one instruction states.
- Assuming a loop with no effect ends, so any such loop may go — the
  analysis that finds "no effect" is the one that errs, and a wrong answer
  deletes a loop that does not end; such loops are rare, so it gains little.
  Only a `for`, which states its count, is removed, and an unused call of a
  local function holding a `while` stays (RFC-0048 rule 8).
