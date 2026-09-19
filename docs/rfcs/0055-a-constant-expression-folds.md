# RFC-0055: a constant expression folds

Status: Accepted — 2026-09-19
Extends: RFC-0037 (an integer operation is the Rust operator at its
width), RFC-0044 (a body is prepared once — the arithmetic chain),
RFC-0052 (an operation is a struct the machine calls once), RFC-0053 (an
aggregate that does not escape never exists — it is what turns a field
read into a constant)

## Problem

The MIR reaches the machine with arithmetic on operands the compiler
already knows. Two shapes cost an operation each, per iteration, in
bodies that are otherwise tight.

`shapes`'s `field read` is `acc = acc + p.x + p.y` over an object whose
fields are `1` and `2`. RFC-0053 replaces the object with its
components, so the loop body arrives as two adds on two constants:

```
r6 = r3 + 1
r8 = r6 + 2
```

Nothing after that pass computes `1 + 2`. The machine adds twice, every
iteration, to add three.

`accum`'s `collatz while` and `shapes`'s `enum match` both branch on
`i % 2 == 0`. The machine runs an integer division for a question that
reads one bit. LLVM at `-O1` turns the same source into a mask; the
distance between the two is a `div` instruction per iteration, and a
`div` is the most expensive integer operation the hardware has.

Neither shape is a source-level mistake a programmer should be asked to
avoid. `p.x + p.y` is how the object is meant to be read, and `i % 2 ==
0` is how evenness is meant to be written.

## Decision

**A binary operation on two constants is the constant.** One rule runs on
a body after SSA construction and before dead-code elimination, so the
operands it leaves behind die with the rest of the dead code. The second
shape above, `i % 2 == 0`, is not addressed: see Rejected.

The value is the value the machine would have computed, at the operand's
width: `+`, `-` and `*` wrap, a shift takes its amount modulo the width,
a comparison reads both operands at the width's own signedness, and a
float comparison compares bit patterns.

Where the machine would **panic instead of producing a value** — a zero
divisor, or a quotient that leaves the width — nothing is folded. The
operation stands and the program panics exactly where it did. A fold
never changes a result, and refusing is how that is guaranteed rather
than argued.

Two constants that are not adjacent still join when one associative and
commutative operator at one integer width separates them and the
intermediate has exactly one use: `(x + 1) + 2` is `x + 3`. The
intermediate becomes the joined constant where it stood, so the value
that carried it still dominates every reader, and no new definition has
to be placed.

**The rule introduces no operator the body did not already hold.** It
replaces a binary operation with its constant, or moves a constant from
one operand of one operator to another. That is not a stylistic
restriction; it is what keeps the fold from paying more than it saves,
and the measurement that establishes it is the first Rejected entry
below.

## What it costs

**One more pass over every body.** It reads the body's constants, use
counts and definitions, applies the first rewrite it finds, and repeats.
Each rewrite removes a binary operation and writes none, so the
repetition ends.

**`i % 2 == 0` still runs a division.** `collatz` and `enum match` pay an
`idiv` per iteration for a question that reads one bit, which is the
second problem stated above and which this RFC does not solve. Why it is
left standing is the first Rejected entry.

## Rejected

**Strength reduction to a bitwise or shift operator** — `x % 2^k` to
`x & (2^k - 1)`, unsigned `x / 2^k` to `x >> k`, and the signed `%` whose
every reader compares it to zero. It was built to this RFC's first draft,
it was correct, and it was measured: **`enum match` +19 %** (13.1 → 15.7
ns/iteration at 1M, base and new spreads non-overlapping across three
alternating pinned reps) and **`collatz while` +2.3 %** (8.8 → 9.0).

The cause is not the mask. `prepare::arith_of` claims the five arithmetic
operators, so `i % 2 == 0` was one fused `Chain2<i64>` and the reduction
made it two dispatches, `BitAnd<i64>` and `Eq<i64>`. One dispatch on this
machine costs more than the integer division it removed.

The general form is the rule the Decision states: **the optimizer
introduces no operator the recognizer does not claim.** It is not about
signedness and not about `%`; a shift or a mask splits a chain wherever
one is introduced.

This is re-admitted when the chain alphabet holds the bitwise and shift
operators — an `acvus-interpreter` item, queued beside `Chain4`. The two
numbers above are its test: the reduction returns when `enum match` and
`collatz` do not regress against them.

**Folding through a chain's leaves.** A chain reads its constants as
leaves, and a rule that rewrote the tree the recognizer is about to
match would have to know that recognizer's shape language. The fold runs
first and leaves whole instructions behind; what the recognizer then
sees is an ordinary body with fewer operations in it.

**A general algebraic simplifier.** `x * 1`, `x + 0`, `x - x`,
reassociation across mixed operators, and distribution are each a rule
whose correctness argument is its own, and together they are a rewrite
engine with a termination proof to maintain. The two rules here were
chosen because a bench named them and because each is one line of
arithmetic to check. A third rule earns its place the same way.

**A constant-propagating branch folder.** A comparison on two constants
becomes a constant, and a branch on a constant condition is still a
branch. Deleting the untaken arm is control-flow work with its own
effect on block parameters and drop insertion, and nothing has measured
it.

## Consequences

- The value a fold produces and the value the machine produces are one
  claim, and it is executed: every edge the surface language can write —
  a wrapping width, the widest constant, a negative dividend, a NaN — is
  a test that runs the same program twice, once with the operands the
  pass can read and once with them arriving through the page, and
  requires the same register word. An operation added to the machine's
  arithmetic without its counterpart in the fold breaks that test.
- The surface grammar has no shift and no bitwise operator, so no body
  reaching the machine holds one today. A pass that introduced one would
  be introducing an operator no source can write, and would own both the
  agreement of that operator's meaning and its cost at the recognizer.
  The fold introduces none, so the operators a body holds after it are
  the operators the source wrote.
- A float operation whose result is NaN is not folded, because `==` on
  floats compares bit patterns and a folded NaN would carry the
  compiler's pattern where the machine's belongs.
