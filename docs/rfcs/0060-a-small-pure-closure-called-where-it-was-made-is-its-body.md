# RFC-0060: a small pure closure called where it was made is its body

Status: Accepted — 2026-09-20
Extends: RFC-0018 (references are types — a capture of word type is a copy
the closure owns and a capture of any other type is a borrow of one),
RFC-0044 (a body is prepared once), RFC-0052 (an operation is a struct the
machine calls once — §7 Consequences, the operand space a call builds)

## Problem

`accum`'s `call while` is

```
let step = |x| -> x + 1; let i = 0; while i < @n { i = step(i); } i
```

and cost **6.8–6.9 ns/iteration** at `5770a5de`, against **2.7–3.4** for
`int while`, which writes more arithmetic inline. The body reached is two
MIR instructions. Everything above them is the call: `Callable::call_in`
through a vtable and the operand space `chain_value` builds.

`graph/inliner.rs` splices every `Callee::Direct` to a local function with
no size bound. A `Callee::Indirect` did not move. The pass held a
devirtualizer, but the callee value at the call is `ref &step` — a
reference to the variable's slot — and never the `MakeClosure` result, so
it never fired. `acvus-mir-test/tests/inline.rs`'s
`inline_devirt_known_closure` was named and commented as if it did, and
its snapshot showed `r2 = call r1(5 (r3))`.

## Decision

**A call whose callee is one `MakeClosure` of the same body is that
closure's body, when the closure is small, pure and called nowhere else.**
Four conditions, all of them or none:

- the closure's body takes no `Order` — the checker's verdict, read and
  not recomputed;
- it holds at most `INLINE_MAX_INSTS` instructions;
- it is one block: its trailing `Return` is the only instruction that
  starts or ends one;
- every reader of the closure is one of those calls.

`INLINE_MAX_INSTS` is **8**. The number is a count, not a measurement: no
experiment separates 8 from 4 or from 16, and the rule is written so that
raising or lowering it changes nothing but which closures qualify.

**The reader condition is asked before the size.** A closure that is also
stored, returned or passed stays a closure and its calls stay calls: an
inlined copy beside a live closure is two bodies for one. The reader walk
follows the closure through the variable it is bound to and the shared
references taken of that variable, and refuses on anything else — a
`Take` of the slot, a second `Assign` to it, a block argument, an
argument position.

**A capture becomes the local it was.** The closure owns what it captures
and its body reads each capture through a reference into that storage
(RFC-0018). Spliced into a caller, that reference has to name storage of
the caller, and there are three shapes and no fourth:

- the argument already *is* that reference, where the captured name was
  itself a capture and no `&&T` exists (RFC-0029) — it is substituted as
  it is;
- the body reads the capture as nothing but the word copy RFC-0018 gives
  it — the copy is the argument, and no storage is made;
- otherwise the caller moves the captured value into one local and lends
  it, where `MakeClosure` stood. It is made there and not at each call
  because a value moves once, however many calls read it.

Anything else — a capture register whose type is neither the argument's
nor a shared reference to it — refuses the closure.

**Nothing is left of an inlined closure.** The `MakeClosure`, the variable
it was bound to, the references taken of that variable and its drop are
the closure's existence; with every call spliced they go with it, in the
inliner. `optimize::dce` will not do it: `dce::is_root` makes every
`Assign` unconditionally live, so a `MakeClosure` whose only reader is
`assign f = r0` outlives every call to it.

## What it costs

Three alternating pinned reps per side, `cargo bench --no-run --workspace`
binaries, `taskset -c 6`, n = 1 000 000, base `5770a5de`:

| case | base | new |
|---|---|---|
| `call while` | 6.9, 6.8, 6.8 | 1.8, 1.8, 2.0 |
| `bf call` | 20.0, 20.4, 20.3 | 17.0, 16.8, 16.6 |
| `bf table` | 16.8, 16.8, 16.6 | 16.8, 17.6, 17.8 |
| `int while` | 3.1, 3.4, 2.8 | 3.4, 2.7, 2.4 |
| `grade while` | 10.5, 10.5, 10.5 | 10.5, 10.5, 10.6 |
| `map cap \| sum` | 8.4, 8.4, 8.3 | 8.3, 8.6, 8.4 |

`call while` falls below `int while` because what is left of it is one
add and one compare per iteration where `int while` has two adds and two
carried values.

`bf call` reaches `bf table`, which is the floor: once `|x, d| -> x + d`
is spliced into the `Inc` and `Dec` arms, `bf call` **is** `bf table`, and
what the 17 ns measures is the brainfuck interpreter — the `Switch`
dispatch, the tape index, the jump table. The one `CallIndirect` left in
the listing is `build_jumps(code)`, called once before the loop.

`map cap | sum`, `int while` and `grade while` do not move. `map cap |
sum`'s closure cannot: it is an argument an extern stage holds, not a
`Callee::Indirect` in MIR.

`bf table` holds no closure in its loop and reads 16.6–16.8 on the base
binary against 16.8–17.8 on the new one. Code layout in a binary whose
inliner changed is the only account available for that, and no probe has
been run; it is the width inside which `bf call` reaching `bf table` is
read.

## Not built

**A `&mut` capture.** The Decision's third capture shape says "shared
reference", and that is exhaustive rather than partial: no `&mut` capture
exists to write a rule for. `lower.rs` builds every non-reference capture
type as `Ty::Ref(Mutability::Shared, ..)`, and the checker refuses
assignment to a captured name — `acc = acc + x` inside a lambda that
captured `acc` is "cannot assign to `acc`: it is captured by the lambda,
not bound in it". Where the captured name already holds a `&mut T`, the
capture register is that reference and the first shape substitutes it.

**A bound on the named-function path.** `Callee::Direct` to a local
function is still spliced whatever its size. Whether an unbounded rule is
right there is a separate question with a separate measurement, and this
RFC does not touch it.

**Removing the closure body from the module.** An inlined closure's
`MirBody` stays in `MirModule.closures` with nothing reaching it. It costs
one prepared body per closure at compile time and nothing at run time, and
deleting it is a reachability pass nobody has needed.

## Consequences

- A test that reads a `CallIndirect` or a `LayArg` from a machine listing
  needs a callee this rule refuses. Two already did:
  `acvus-interpreter-test/tests/sync_call_is_an_operation.rs` and
  `.../laid_argument_drop.rs` each carried `|x| -> x + 1`-sized closures as
  the witness for a rule of RFC-0052, and each now carries an `if`, with
  the obligation written where the source is.
- The number of times a captured `Large` is released does not depend on
  whether its closure was inlined. The two forms are one variable apart in
  `acvus-interpreter-test/tests/inlined_closure_capture.rs`.
- A closure that captures a closure collapses at both levels, because the
  inner call's callee reaches the outer capture's local through the
  reference the third capture shape made.

## Open questions

- `dce::is_root` treats every `Assign` as observable, so dead storage of
  any kind survives it, not only a closure's. Whether a store to a slot no
  reader names is dead is dce's question and is unanswered.
- `INLINE_MAX_INSTS` has no measurement behind it. What would settle it is
  a bench whose closure sits near the bound.
