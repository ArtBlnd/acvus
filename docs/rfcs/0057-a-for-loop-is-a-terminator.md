# RFC-0057: a `for` loop is a terminator

Status: Accepted — 2026-09-20
Extends: RFC-0047 (a slice is the one thing the machine indexes; a slice is
two registers), RFC-0052 (regions: `Loop<C>`, `Diamond`, `Select`),
RFC-0018 (references and loans), RFC-0045 (an assign releases what the
slot held), RFC-0048 (ownership is the machine's), RFC-0046 (tasks;
`spawn_split`)

## Problem

The language has two loops, `while` and `while let`. A loop over a
container is written as an index loop:

```
let i = 0u64;
while i < len(&v) { let x = &v[i]; …; i = i + 1; }
```

Every pass then rediscovers what the source meant: the loop recognizer
finds the back edge, the interval domain proves `i < len` to drop the
bounds check (RFC-0047 rule 7), LSR finds the induction variable, the
loans decide how long `&v` is borrowed, and `spawn_split` cannot tell an
iteration from the next without a dependence analysis the tree does not
have. The machine runs the loop as `Loop<C>` with a condition chain
(`Lt`), a `Yield`, and a `Mov` of `i` per iteration; `int while` is
2.8 ns per iteration of which the loop control is most.

The Brainfuck bench pays 337 flag `Mov`s per log line for the absence of
`break`/`continue` (`benches/logs.rs`).

## Decision

The rule this RFC is the second instance of, after RFC-0051's `Switch`: **a
terminator keeps the control shape the source wrote**; a general `Jump`
remains only for an exit that shape does not have (`break`, `continue`).
Information a terminator drops is information a later pass rediscovers
(`prepare`'s `Diamond` recognizer rebuilds what `JumpIf` forgot; the
removed `merge_of` field recorded a join no pass decided on). The next
instance is `if`: a `Diamond { cond, then, else, join }` terminator that
makes the machine's recognizer and the `Select` decision (RFC-0052)
questions asked of one terminator — a separate RFC, after this one.

1. **Syntax.** Four heads and no other:

   ```
   for x in &v { … }        // v: Vec<T> or Array<T, N>;  x: &T
   for x in &mut v { … }    // v: Vec<T> or Array<T, N>;  x: &mut T
   for x in a { … }         // a: Array<T, N>, consumed;   x: T
   for i in lo..hi { … }    // lo, hi: one integer width;  i: that width
   ```

   `lo..hi` is a `for` head, not a value: there is no `Range` type. A
   reversed or stepped range is a function over the loop, later, not a
   syntax. An iterator (`Iter`) is not a `for` source: an iterator is an
   extern's value and is consumed by `while let Some(x) = next(&mut it)`.

2. **MIR: the loop is one terminator.** The header block ends in

   ```
   Terminator::For {
       source: ForSource,   // Slice(s) | SliceMut(s) | Array(a) | Range { at, hi }
       body: Label,         // params: (elem, index, carried…)
       exit: Label,         // params: (carried…)
   }
   ```

   The terminator is the condition. No instruction writes `index < len`,
   `i = i + 1`, or the element read; the terminator yields `elem` and
   `index` as the body block's first parameters and advances. The body's
   last block jumps to the header with the carried values (the latch);
   `continue` is that jump; `break v` is a jump to `exit` with the carried
   values (and `v`, where the loop is an expression). A `Slice` source is
   the `AsSlice` of the container taken once, before the header; the
   borrow lives from there to `exit`, and the loans refuse a shape write
   to the container inside the loop as they refuse any write under a
   live shared borrow. `SliceMut` is the exclusive form: `elem: &mut T`
   and no other name of the container is live in the body. `Array`
   consumes: each iteration takes element `index` out of the array; an
   exit through `break` drops the elements not yet taken on the exit
   edge; the array is empty after the loop.

3. **What the terminator makes free.**
   - Bounds: the element access is in range by construction — the
     machine's `For` reads `elem` as `IndexUnchecked`; RFC-0047 rule 7's
     interval domain is not consulted.
   - Loans: the borrow's extent is the terminator's; no per-instruction
     region computation.
   - Loop recognition: the header is the block whose terminator is `For`;
     the induction variable is the terminator's `index`; no back-edge
     search, no LSR to find it.
   - `spawn_split`: an iteration is independent of the next when the
     body carries no value through the latch (`carried` is empty) and
     every element write goes through `elem` of a `SliceMut` source — a
     question asked of the terminator and the body once. The split rule
     itself is a later section of this RFC.
   - The machine: one region op `For<Src, …>` holding the pair (or the
     range's bound) and the index register: `cmp`, `jae exit`, element
     load, body, increment, jump — no `Lt` chain, no `Yield`, no `Mov`.

4. **`break` and `continue`.** Admitted inside `for` and `while`, the
   innermost loop only (no labels). `continue` lowers to the latch jump,
   `break` to the exit jump; the lowering's scope stack emits the drops
   of every scope between the statement and the loop body on that edge.
   A branch whose arm leaves the loop does not rejoin, so it is not a
   `Diamond`; it is an `Escape`, an operation of the body whose arm ends
   in the verdict the region above reads (RFC-0052 §3). The loop stays one
   region, and a loop with no such branch is the region it was, with no
   compare added.

5. **Aliasing.** `for x in &mut v` holds the container exclusively for
   the loop; the terminator carries that borrow, so `v` named in the body
   is a compile error from the loans, and two `for` loops over `&mut v`
   cannot nest.

## What it costs

- One terminator arm in every pass and reader of `Terminator`: the SSA
  builder (header parameters), `dce`, `forward`, `code_motion`, `loans`,
  `validate`, the printer, `prepare` (a new region op), later kovac. The
  enum is closed; the compiler enumerates the sites.
- Two loop forms in the language. `while` remains for a loop whose
  condition is not a traversal.
- A `Range` head admits two integer widths only when they are one type;
  `0..n` with `n: u64` gives `i: u64`. `lo > hi` runs zero times.

## Rejected

- **Lowering `for` to `while` with an index.** Every pass would
  rediscover the traversal the source stated, and the machine would run
  the condition chain the terminator removes.
- **A `Range` value type.** A range that exists only as a loop head has
  no other reader; a value would need a representation and a crossing.
- **`for` over an iterator.** An iterator's `next` is an extern call per
  element and cannot be a terminator's condition; `while let` is its
  loop.
- **Labeled `break`.** The innermost loop is the only target; a label is
  a second scope mechanism the block design does not have.
- **Tail duplication for `break`** (copying the loop's tail into each
  arm so every branch rejoins): 2^k code for k exits.

## Consequences

The language and MIR half is built: the four heads, `Terminator::For`, the
passes' arms, `break` and `continue`. What a traversal's MIR holds, over
`for x in &v { acc = acc + *x }`, is one `AsSlice` above the header, a
header holding nothing but its terminator, and a body of the addition
alone:

```
r8 = as_slice &r7
jump L0(0, r1)
L0(r10: i64):
  for slice(r8) -> L1 else L2(r10)
L1(r12: &i64, r13: u64):
  r14 = take (*r12)
  r15 = r10 + r14
  jump L0(r15)
L2(r19: i64):
```

No `Lt`, no `i + 1`, no `Index`: the element and the counter are the body
block's leading parameters, which the terminator fills. The carried values
are the header's parameters, the latch is `jump L0(carried…)`, `continue` is
that jump and `break` is `jump L2(carried…)`.

The parameter layout is the one the machine's `For` op reads: the counter is
a parameter of the body and not a value the body computes, so no pass has to
find it, and a `Range`'s element is its counter, so a range loop carries one
register where a container's loop carries two.

The machine half is built for a loop whose body rejoins. `prepare::
recognize_for` matches the shape the lowering emits -- the entry jump, a
header holding nothing but its terminator, the body block the terminator's
one edge reaches, the latch back to the header -- and `prepare::for_op`
collapses it into one region operation, `ops::control::For<S>`, where `S` is
the head: `Slice` for both `&v` and `&mut v`, `Array<LARGE, WORD>` for an
array by value, `Range<T>` at the width the two bounds share. The counter is
a local of `For::run`, not a register any operation advances, and the bound
is read once above the loop, so the region runs one chain per iteration
where `Loop<C>` runs two -- its head chain is the comparison the terminator
replaced. The element read is `IndexUnchecked`: the terminator is the bound.

What the `Array` head leaves behind is the loop's one cross-crate
obligation. Each iteration moves element `index` out of the array and leaves
`Undef` in its slot; the array itself is released by the `Drop`
`optimize::drop_insertion` puts on the exit block, which is also what
releases the elements a `break` never reached. The machine therefore does
not release the array, and must not.

A loop a `break` leaves, or one a `continue` returns to the head of, is one
region: the branch is an `Escape` in the body, the arm ends in `Break` or
`Continue`, and `For<S, Escapes>` reads the word the body handed back. What
`recognize_for` still refuses is the one shape whose `break` does not name
the loop's exit: where `optimize::drop_insertion` gives the exit edge a
block of its own, `lower` lays that block between the terminator and the
body and the `break` jumps past it to the continuation both reach, so the
verdict "leave, then run the region's successor" would run the exit block's
drops that the arm has already run. A traversal of a slice or an owning
array that a `break` leaves is that shape, and it runs as joints. The
header is then the terminator it is in the IR: `ForAt<S>` reads
the counter, compares it to the bound, and either lays the body block's
leading parameters and continues to the body or continues to the exit, with
the two edges' parallel moves in blocks of their own as a `JumpIf`'s are.
The counter lives in a frame register because that operation returns:
`ForStart<S>` on the loop's one entering edge lays its first value and
`ForStep<S>` advances it on every other edge into the header, which is the
latch and every `continue`. What it costs against the region is a load of
the counter and a read of the bound per iteration, and one `ret` and one
dispatch per iteration; what it buys is `break` and `continue`.

`return e` is the third exit edge, beside `break` and `continue`: it leaves
the enclosing body from any depth, so a loop holding one takes the joints
path for the reason a `break` puts a loop there -- the return's drop block
stands between the terminator and the body, which is not one straight run.
The lowering is the `?`'s without the test: the value, `emit_return`, and
then the unreachable label `break`'s jump also leaves behind. Nothing in
this decision changes for it, `recognize_for` included.

The counter's register is the body block's counter parameter, and liveness
has to be told about it: the terminator writes that parameter and the
terminator is its only reader, so `inst_info` reports no use and
`prepare::assign_slots` would hand its register to a value live across the
loop. `Edges::for_counter` is where the use is put back.

An `Array` head over this path releases nothing the region does not. The
lowering emits the array's `Drop` on the loop's exit block *and* on every
`break` edge -- `acvus mir` over `for x in a { if x == 2 { break; } … }`
prints `drop r0` in both -- so the machine leaves the array's register and
the frame's claim on it standing, exactly as Decision 3 requires. An array
of owners cannot `break` at all: the checker refuses it.

The parallel split section is still unwritten: `spawn_split` over a `for`
waits on it, not on the machine.

What is measured already, at the machine's listing, is what `break` changes
about region recognition: a loop whose exit is a flag is a `Loop<Slot>`
region, and the same loop with `break` is joints, because an arm that leaves
the loop does not rejoin. A scan whose flag lives in a slot rather than in a
block argument carries no `Mov` either way, so the flag `Mov`s of the log
bench are not this scan's: `acvus-interpreter-test/tests/loop_exit_moves.rs`
holds both counts.

## Order of work

`Terminator::For` in `ir.rs`/`cfg.rs` and the lowering of the four heads →
`validate` and the SSA builder → the passes' arms (`loans`, `dce`,
`forward`, `code_motion`, `dse`) → `prepare`'s `For` region op and the
machine → `break`/`continue` with the scope-drop rule → the parallel
split section, written after the loop exists.
