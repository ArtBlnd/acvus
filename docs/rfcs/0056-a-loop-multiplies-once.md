# RFC-0056: a loop multiplies once

Status: Accepted — owner and coordinator, 2026-09-19
Extends: RFC-0037 (integer widths — `+`, `-`, `*` wrap; only `/` and `%`
raise), RFC-0044 (a body is prepared once — the chain a loop body
becomes), RFC-0007 (an operation that can raise runs only where the
program wrote it)

## Problem

A loop that computes `i * k` or `i * k + x` from its own counter, with
`k` and `x` the same on every iteration, redoes a multiplication the
previous iteration already almost had. The classic answer is to give the
expression its own counter: start it at `i0 * k + x` and add `c * k` once
per iteration, where `c` is the step of `i`.

The machine has no loop analysis to build that on. It has a dominator
tree, and `code_motion` reads back edges off it to ask how deep in the
loop nest a block sits; there is nothing that names a loop's header, its
latch or its body, and nothing that names the values one iteration shares
with the next. Two passes asking the same question of the same control
flow with two answers is the failure this RFC is written to prevent.

## Decision

**A loop is a back edge and nothing more, and one definition says so.**
An edge `tail -> head` whose head dominates its tail names a natural
loop, whose body is the head together with every block reaching the tail
without passing the head. `code_motion`'s loop depth and this pass's
header, latches and body are the same definition, read from one place.

**An induction variable is a header parameter plus an invariant.** `i` is
one when the latch sends the header `i + c` and `c` is the same on every
iteration. Nothing deeper is examined: a counter rewritten through
memory, a step that is itself an induction variable, and a derived
variable of a derived variable are all outside the pattern.

**A value is the same on every iteration when it is defined outside the
loop, or when it is a word constant.** The first is what `code_motion`'s
hoist produces, and it is the whole of the term for a value the hoist can
move. The second is the case the hoist cannot reach and this pass needs:
control equivalence holds a `Const` inside a loop body, because a body
does not post-dominate its preheader, while the instruction reads nothing
and writes the same word every time it runs. A literal step `1` and a
literal factor `3` both stand inside the body, so without this case the
pass reaches nothing at all. The price is named: such a constant is
re-emitted above the header rather than moved.

**`i * k + x`, where nothing but that sum reads the product, becomes a
header parameter**, started in the preheader and advanced in the latch,
when the loop has one entering edge from a block that ends in a plain jump
to the header, one back edge, and the multiplication stands in a block
that dominates the latch. One derived variable per distinct `(i, k, x)`.

**A reduction must lower the body's operation count, and that is what
selects the form.** `i * k + x` sheds the multiplication and the sum and
gains the latch's addition — four operations become three, counting `i`'s
own increment. A bare `i * k` sheds one and gains one: three become three,
and a header parameter is added to carry between them. A change with no
gain is not a reduction, so the bare form is not reduced. The count is the
rule, not the syntax: a bare product read by several consumers is still
the bare form, because SSA computes it once however many values read it,
while a product read by two invariant sums would let the count fall and is
excluded instead by the pattern, which matches one product and the one sum
that reads it.

**The multiplication is reduced only where the result is the same
number.** `+` and `*` wrap at the operand's width, and wrapping
arithmetic is arithmetic modulo `2^width`, where multiplication
distributes over addition exactly: `(i0 + n*c) * k` and `i0*k + n*(c*k)`
are the same integer, always. Neither operation can raise, so no trap
moves and none is introduced. In floating point they are different
numbers, and the difference is measured, not assumed: on mandelbrot's
200×100 grid at 200 iterations, `cx` accumulated by repeated addition of
`3.0 / w` differs from `-2.0 + 3.0 * px / w` by up to 6.4e-15, `cy` by up
to 1.7e-15, and **34 of the 20000 pixels reach a different escape
count**, moving the grid's total from 946598 to 943023. A changed result
is a wrong pass, so the float form is refused.

## What it costs

One pass, one file, and one analysis module that `code_motion` now reads
its loop depth from.

Per reduction: one header parameter, two multiplications and one addition
in the preheader, one addition in the latch, and one constant re-emitted
above the header for each word literal among the step, the factor and the
offset. Against that, the body loses one multiplication and one addition.

The count rule costs the bare form. A loop that writes `i * k` and nothing
more keeps its multiplication, and the loop that would gain from having it
reduced — one where the product is read by two invariant sums, so that two
derived variables replace three operations — is not reached either,
because the pattern is one product and one sum.

The dominance condition costs the conditional cases. A multiplication
under an `if` is left alone, because a derived variable must be advanced
on every iteration and the original was not computed on every iteration.

The float refusal costs mandelbrot, whose pixel loop is the one place in
the benches where an accumulated coordinate would pay.

## Rejected

**Strength reduction through the prepared chains.** RFC-0044 fuses a
body's arithmetic into a tree the machine walks once, so `i * k + x` is
already one operation by the time the chain exists. Reducing there would
mean rewriting a chain's leaves after the shape is fixed, and the
dominator tree — which is what says an expression runs on every iteration
— is gone by then. The reduction belongs where the control flow is still
visible.

**A general scalar evolution.** Classifying every value in a loop as an
affine function of every counter would reach a derived variable of a
derived variable, a step that is itself an induction variable, and loops
whose bound is computed. It is also a fixpoint over a lattice, with its
own correctness burden, for a language whose lowering emits `while` and
`while let` and nothing else. The pattern here is two shapes deep and
each one is checkable by reading it.

**Reducing a multiplication that does not run on every iteration.** It is
correct — the arithmetic wraps either way — and it is slower whenever the
branch is not taken. Correct and slower is not a reason to do it.

**Applying the float form behind a tolerance.** A tolerance is a claim
that the caller does not care which number comes back. Nothing in the
language says that, and mandelbrot's escape count is an integer that
changes.

## Consequences

- One module owns natural loops: back edges, a loop's header, latches and
  body, the loop depth `code_motion` holds every hoist to, and the
  definition of a value that is the same on every iteration. A second
  answer to any of these is a defect.
- The pass runs after the hoist, which is what puts a loop's invariants
  above its header and leaves the entering block ending in a jump of its
  own, and before the scheduler, which orders instructions within a block
  and must see the ones this pass adds.
- A loop the pass declines — two back edges, two entering edges, an
  entering block that does not end in a plain jump to the header — is
  left exactly as it was. Declining is always available and never wrong.
- Integer arithmetic reduced by this pass carries the bits the
  multiplication carried, past the width's maximum included; a program's
  value and the iteration on which it raises are both unchanged. The
  tests that hold this run the reduced and the unreduced form of one
  computation in one program and read their difference.
- Floating-point multiplication by an induction variable is not reduced.
  Reopening it requires a measurement that shows no result changing, not
  an argument that the error is small. The refusal rests on the grid
  measured above: 6.4e-15 of drift in `cx`, and 34 of the 20000 pixels
  reaching a different escape count.
- **No loop in the bench set reaches this pass today**, and the pass lands
  anyway. Mandelbrot's loops multiply `px.to_float()`, not `px`: an extern
  call stands between the counter and the multiplication, and the product
  is a float besides. Attention multiplies two loaded floats
  (`@query[i] * key[i]`) and never an induction variable at all — its
  `@keys[t]` and `key[i]` are two one-level indexings, not a `t * d`. So
  the pass applies zero reductions across the bench set, for two reasons
  that are both about the programs. RFC-0049's `as` cast is what would put
  mandelbrot's product in reach, and the `for` loop is what will write
  counters in the shape this pass matches. Until then the pass stands on
  its listing snapshots and its value tests, and its first measured
  speedup arrives with one of those two.
