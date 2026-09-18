# RFC-0049: `expr as T` is Rust's `as`, and inside a chain it is a leaf

Status: Accepted — owner and coordinator, 2026-09-19
Extends: RFC-0037 (integers have a width), RFC-0044 (a body is prepared
once — the arithmetic chain), RFC-0052 (an operation is a struct the
machine calls once), RFC-0055 (a constant expression folds)

## Problem

RFC-0037 gave the language eight integer widths and left conversion out:
"No conversion between widths in the language; `to_int` still reads every
width into an `i64`." What the language had instead were two extern
signatures, `core::to_float` and `core::to_int`, and they cost three
things.

**A call per conversion.** `accum`'s `float while` is
`acc = acc + i.to_float()`, and `to_float` is a `CallExtern1`: an argument
window, a `takes` mask, an indirect call, one dispatch, every iteration,
to widen an `i64` the machine already holds in a register.

**A conversion the recognizer cannot see through.** `prepare`'s chain
recognizer claims arithmetic and stops at a call, so mandelbrot's
`px.to_float() * k` hid the induction variable behind an opaque
instruction. The pixel loop's body prepared to eighteen operations, four
of them calls into `acvus-ext`.

**Conversions that did not exist.** `to_int` returns an `i64` and has no
`u64` instance, so no expression could produce a `u64` — and `a[i]` takes
one (RFC-0047). The `logs` bench had to write `plen / plen` to obtain the
`u64` one.

## Decision

1. **Syntax.** `expr as T`, where `T` is one of RFC-0037's eight integer
   widths or `f64`. Postfix, left-associative, binding tighter than every
   binary operator and looser than the unary ones — exactly where Rust
   binds it, so `1.0 + 1 as f64` is `1.0 + (1 as f64)` and `-x as u8` is
   `(-x) as u8`.

2. **Semantics are Rust's `as`, exactly.** Integer to integer truncates or
   sign-extends by width; integer to `f64` rounds to nearest; `f64` to an
   integer saturates at the width's ends and maps `NaN` to zero, which is
   Rust's behaviour since 1.45. `f64` to `f64` is the identity. The
   conversion is total: no `as` traps, so no pass has to keep one on the
   path the program wrote.

3. **A cast is a leaf, never a node.** Inside an arithmetic chain a cast
   into the chain's own type is absorbed into the *read* of the leaf below
   it — `chain::LeafRead::Cast` in the plan the chain already carries — and
   contributes no operator node. `Chain{1,2,3}<T>` therefore stays one `T`
   per chain, and the chain family's instance count is what RFC-0052 left
   it: 351 per entry. Outside a chain a cast is one operation,
   `cast::Cast<From, To, Src, Dst>`: a word in, a word out, a `WORD` store,
   a `Place` like any other word operation.

4. **The numeric externs go.** `core::to_float` is deleted, signature and
   all eight instances. `core::to_int` keeps exactly one instance, `&Bool`
   to `i64`, which is the one conversion `as` does not do; its seven
   numeric instances are deleted. Every script, test and bench that called
   them is written with `as`.

5. **The fold applies the same semantics.** `optimize::fold` rewrites a
   cast whose source is a constant into that constant, computing it with
   the same Rust `as` expression `ops::cast`'s instance for the pair runs
   (RFC-0055). Where the source is not a constant — including every `NaN`,
   which the float fold refuses to produce — the cast stands and the
   machine computes it.

**The MIR instruction is new.** `InstKind::Cast { dst, src, to: NumTy }`.
The `CastKind` already in `ir.rs` is not it and was not reused: that names
a pure ExternFn performing a declared coercion between user-defined types
(RFC-0016), which is a call, not a width. RFC-0012's identity parameters
have no run-time cast at all — "the runtime sees none of it" — so there
was nothing there to reuse either. The instruction carries no source type,
because `src`'s own type is the one every reader already holds and a
second copy could disagree with it.

## What it costs

**One branch per chain, and it was measured twice.** The chain's plan
gained `Reads`, which is `Own` for a chain that absorbed no cast and
`Leafwise` otherwise, and `tree{1,2,3}` matches it once before reading its
leaves.

The first draft put a `LeafRead` on each leaf and matched per leaf. On
three alternating pinned reps that cost `range | sum` **1.7 → 1.9
ns/iteration**, `call while` **10.0 → 10.8** and mandelbrot **9.1 → 10.4**
(+11 %) — loops with no cast in them, paying for a conversion they do not
do. Hoisting the test to one per chain returned mandelbrot to +2 %. What
remained was the frameless-chain path, where the chain *is* the body and
is reached through a function pointer, so the plan read is the whole cost:
`call while` stayed at 10.0 → 10.5 (+5 %). That path settles the question
at preparation instead — `chain_eval` picks the evaluator from `plan.reads`
— which costs more `ExprFn` monomorphisations and no `dyn Op` instances at
all. `call while` came back to +2 %.

Making the leaf's type a parameter of the chain operation would have been
zero-cost for both, and would have multiplied the `dyn Op` family RFC-0052
cut from 1404 to 351. It is not paid.

**324 cast instances.** Nine source types times nine target types times
the four `Place` combinations `at_unary!` reaches. Only the pairs a
program writes are called; the count is code size, not dispatch. The whole
bench set instantiates one, `Cast<u64, f64, Slot, Slot>`.

**Casts a chain cannot absorb still dispatch.** A cast whose result feeds a
call argument, a store, or an operator at another type is an operation of
its own. `(d as f64).sqrt()` in the attention script is the one in the
bench set: `sqrt` is a call, so the cast stands beside it.

**What it bought.** `accum`'s `float while` is **3.8 → 2.8 ns/iteration**
(−26 %, seven alternating pinned reps, minimum and median agreeing), and
its loop body fell from `CallExtern1, AddF64, Add` to `Chain1<f64>, Add`.
Mandelbrot's pixel-loop body fell from eighteen operations to twelve and
its time moved +1.1 % / +2.3 %, inside the spread: its four conversions
are per pixel, not per escape step. Every other `accum` case is within
±3 %, and attention within ±1.2 %.

## Rejected

**`as` as another extern signature.** It is what the language had. A
signature per target width is eight more names, each still a
`CallExtern1`, each still opaque to the recognizer — the three costs in
Problem, renamed. The whole value of this RFC is that the conversion stops
being a call.

**`float` and `int` as spellings.** RFC-0037 named the types `i8`…`i64`,
`u8`…`u64` and `f64`, and a second set of names for two of them would make
`as int` and `as i64` two ways to write one thing. `1 as float` is
refused, with the list of names that are accepted.

**A cast as a chain *node*.** A node is an operator the tree applies to two
subtrees; a cast applies to one and changes its type. Making it a node
would have meant a chain whose interior is not all at one type, which is
the assumption `Chain{1,2,3}<T>` is built on — so it would have meant a
second type parameter per node and a family that grows with every pair a
program casts between. Absorbing it into the leaf's read costs one plan
field and grows nothing.

**`f32`.** RFC-0037's "Not built" says the language has no `f32`, and it
still has none; `as` converts between types that exist. A `1.0 as f32` is
refused like any other unknown type name.

**A literal suffix.** `3 as u64` is how a `u64` constant is written, and
`3u64` is not added here. A suffix is a lexer and literal-typing change
whose interaction with RFC-0037's "the use decides the width" rule is its
own decision; the fold turns `3 as u64` into the constant anyway, so
nothing is paid for the spelling.

## Consequences

- `acvus-ast`: `Token::As`, `Expr::Cast`, and the `CastExpr` level of the
  grammar between `MulExpr` and `UnaryExpr`.
- `acvus-mir`: `NumTy`, `InstKind::Cast`, the checker's `CastSite` list
  settled after the solve, `MirErrorKind::{CastToUnknownType,
  CastOfNonNumber}`, the `Cast` arm of the IR validator, and
  `optimize::fold`'s `cast_result`.
- `acvus-interpreter`: `ops::cast`, `Num`'s nine `of_*` constructors,
  `AsNum`, `chain::{LeafRead, Reads}` and the `reads` field of
  `chain::Plan`, the `PLAIN` parameter of `tree{1,2,3}` that `chain_eval`
  settles, `Growing::leaf` in the recognizer, and `check_chain` asserting
  each leaf's register at the type its `LeafRead` names rather than at the
  chain's own.
- `acvus-ext`: `conversion.rs` loses `sig::to_float` and seven `to_int`
  instances.
- The value a fold produces and the value the machine produces are one
  claim, and it is executed: `acvus-interpreter-test/tests/fold_agreement.rs`
  runs every edge twice, once with the cast's source constant and once with
  it arriving through the page, and requires the same register word. A pair
  added to `Num`'s constructors without its counterpart in `fold`'s
  `cast_result` breaks that test.
- `to_int` is a shared signature with one instance. That is not a reason to
  demote it to a plain function: RFC-0019's mechanism is what lets a host
  add instances for its own types, and a `Bool` conversion is the one the
  standard library owns.
