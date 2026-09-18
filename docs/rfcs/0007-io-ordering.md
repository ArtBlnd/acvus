# RFC-0007: IO ordering

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

The intent behind an IO call is known only to the script author and is not
inferred. IO calls execute in source order by default.

A block, written `anyorder { ... }`, declares that order is irrelevant
inside it. The rule applies to the
region as a whole, including every iteration of a loop inside the block. The
block boundary is a join: outside it, source order resumes. The block's name
expresses order-irrelevance, not an execution strategy; whether the interpreter
runs the block in parallel, concurrently, or sequentially is not part of the
declaration and every choice is sound.

A wrong declaration is a wrong program. The interpreter may verify a
declaration by running the block under an adversarial schedule and reporting a
difference in result.

An ExternFn is assumed to have side effects. Purity is an explicit opt-in by
the ExternFn author.

An iterator is always lazy and move-only. There is no memoized variant.

The IR keeps a low-level dependency relation. The block is surface syntax and
lowers to that relation.

Order is a value. The IR has a type `Order` that no script can name. A call
whose effect is not Pure takes an `Order` and yields one; a Pure call knows
nothing of it. Everything in the IR is simultaneous until a dependency says
otherwise, and an `Order` value is that dependency. Sequential code is a
chain: each effectful call takes the value the previous one yielded. A local
function or lambda whose effect is not Pure takes an `Order` first and
yields one last; its declared type does not show this, the lowering adds it.

One instruction joins orders, as phi joins values: `merge(o..) -> o`. It is
associative and commutative, and to an executor it means "after all of
these". The `anyorder` block lowers to a fan-out and one merge: every
effectful call inside takes the block's entry value, and the exit is
`merge` of everything they yielded. Nothing else changes; the binding that
sequential code makes at each call moves to the block's exit. A loop inside
the block accumulates through a loop phi on `Order` and a `merge` per
iteration, so a dynamic number of calls needs no runtime bookkeeping.

With a call split into spawn and evaluation, the spawn takes the `Order`
and the evaluation yields it. A value is evaluated where it is used; when
it is used matters to the order only through the `Order` chain.

An executor holds no `Order` values. Once dependencies fix a schedule, the
values have done their work. What an executor tracks is which calls have
not yet reached the merge that awaits them.

Code motion moves an instruction only between control-equivalent blocks:
the destination dominates the source and the source post-dominates the
destination. The two then execute under exactly the same condition, so the
move changes nothing the program can observe — no raise on a path that did
not reach the instruction, no work on a path that did not need it. The rule
is one criterion and holds for every instruction, not a list of the kinds
that are exempt; whether an instruction can fail is not asked, because
control equivalence already fixes the set of paths it runs on. A spawn is
held to a stricter rule, stated under Not built: it does not move at all,
because moving it also moves when the work starts.

Equivalence says the two blocks execute under the same condition, not that
they execute equally often. A loop's exit post-dominates its header, so a
destination may be control-equivalent to its source and still sit inside a
loop the source has left. Every destination is therefore also held to the
source's loop depth, the number of natural loops containing the block: a
move never lands deeper in the loop nest than it started. This bound is on
the destination itself and so applies to the borrow below as well.

One instruction carries a different criterion, because nothing it does
reaches a path: a shared borrow of a variable or a parameter with no path
under it writes one register with the address of another, reads nothing,
allocates nothing and cannot raise. It moves to any dominator, and what
bounds it is the borrow instead. `check_borrows` runs before code motion,
so a borrow moved above a loop holds its loan through iterations the
checker saw it outside of; the move is taken only when no block the borrow
would newly span — the blocks the destination dominates that still reach
the source — writes that storage. A `&mut`, a borrow through a reference
and a borrow with a path under it do not move at all: the first takes a
loan that conflicts with every other, the second is a memory op that stays
in order with the other ops through that reference, and the third walks
into the value, which a path that would not have reached it can find in a
shape the walk does not expect.

Two such borrows of one storage that end up in the same block are one. A
block is a straight line, so the question needs no dominance and no
reachability: the second is the first unless something between them takes
the storage exclusively — a write, or a `&mut` borrow, which writes nothing
but holds the storage for as long as the reference it makes lives. The
merge runs after the move, because it is the move that brings the pair into
one block: every borrow a loop rebuilt each iteration meets the one already
standing above the loop.

## Rationale

Whether two IO calls may be reordered depends on what the author means by
them, and the ExternFn author cannot know that; a conservative type
declaration only restates ignorance. Putting the declaration where the intent
is keeps the declaration honest.

A region rule is one rule. A per-call or per-pair relation grows quadratically
and would not be written. A sequential default with an explicit opening is
sound; the reverse default is not.

Assuming side effects by default is the sound direction for the fact the
ExternFn author does know. The author declares the fact (this function does
IO) and the script author declares the intent (order does not matter here);
each states only what they know.

A memoized iterator would need to know that its pipeline has no side effects,
which this layer cannot derive; a lazy pull is correct without that knowledge.

Carrying order as an SSA value lets every existing dependency-driven pass
respect it without learning a new concept; independence comes only from the
block, so there is no name to collide on. A fan-out is the only place an
`Order` value is used twice, and only the lowering of a block makes one, so
the value needs no linearity of its own.

A join as a value, rather than a pair of instructions that open and close a
region, leaves nothing to keep well-formed. A pair is a bracket: two
brackets that cross have no inside, and every pass that moves or copies
code would have to preserve the nesting. A `merge` is only a dependency;
two overlapping joins are two merges over shared inputs, a plain DAG, and
the only well-formedness is SSA dominance.

The name says what is declared. "Any order" is the whole statement; a name
for an execution strategy would say more than the author knows.

The shape has prior art. XLA threads a `token` through side-effecting
operations and joins tokens with `AfterAll`; JAX types each primitive's
effect as ordered or unordered and threads tokens in its jaxpr; PyTorch's
compiled path threads effect tokens through `with_effects` and sinks them
before code generation. In each, whether order matters is a property of the
operation. Here it is also a declaration the script author makes at the
place where the intent lives, which none of them has.

## Not built

- No automatic inference of IO ordering from types, identities, or
  consistency declarations. The intent is not in the types.
- No "parallel by default" mode. Its default is unsound.
- No memoized or shareable iterator.
- No `Order` value in the surface language. The author declares a region;
  the compiler wires it.
- No linear or region-typed `Order`. One type; the lowering alone decides
  where a value fans out.
- No hoisting of a call's issue above a branch, with one exception. The
  call is issued where the spawn stands; moving it to a dominator would
  issue it on a path that never reaches it, and `Order` says "after", not
  "only if". The exception is a commutative call whose block
  post-dominates the block of the call it follows: every path through
  that block reaches it, so issuing it there speculates nothing, and it
  joins that call's run. A pass may still reorder within a block and move
  the wait toward the use.

## Consequences

- An ExternFn with no purity declaration is treated as effectful by every
  consumer of its type.
- A script with no block runs its IO in source order.
- Passes that reorder instructions respect an `Order` operand as they
  respect any other operand; calls are not barriers in themselves.
- `merge` is a value instruction, not control flow: it may sit anywhere its
  operands are available.
- Cloning an iterator is a type error, never a runtime copy.

## Open questions

- Whether a block may also declare an order among its own sub-blocks, or
  nesting is the only composition.
