# RFC-0007: IO ordering

Status: Accepted
Date: 2026-09-07
Supersedes: none

## Ruling

### Accepted

The intent behind an IO call is known only to the script author and is not
inferred. IO calls execute in source order by default.

A block declares that order is irrelevant inside it. The rule applies to the
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

### Proposed

Ordering is carried as an SSA value. An IO call consumes an ordering token and
produces one. Sequential code threads the token through each call; inside a
block every call consumes the block's entry token, and a join at the block exit
merges the produced tokens into one. With the call split into spawn and
evaluation, the spawn consumes the token and the evaluation produces it.

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
block, so there is no name to collide on.

## Not built

- No automatic inference of IO ordering from types, identities, or
  consistency declarations. The intent is not in the types.
- No "parallel by default" mode. Its default is unsound.
- No memoized or shareable iterator.

## Consequences

- An ExternFn with no purity declaration is treated as effectful by every
  consumer of its type.
- A script with no block runs its IO in source order.
- Passes that reorder instructions must respect the ordering token as they
  respect any other operand.
- Cloning an iterator is a type error, never a runtime copy.

## Open questions

- The surface name of the block.
- Whether a callback passed to an ExternFn contributes its own purity to the
  caller, or the caller is simply effectful.
- Whether an ExternFn may declare the contexts it reads, and where that
  declaration lives.
