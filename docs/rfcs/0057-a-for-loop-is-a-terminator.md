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
   A branch whose arm leaves the loop does not rejoin, so the machine's
   `Diamond`/`Select` recognizer does not admit it and that loop runs its
   branches as joints; the loops that contain no such branch are regions
   as before.

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

Expected, to be replaced by measurement: `int while` written as `for i in
0..n` loses the `Lt` chain, the `Yield` and the `Mov` per iteration;
attention's inner loop written as `for x in &keys[t]` loses its bounds
check and its increment; the log bench's flag `Mov`s go with `break`.

## Order of work

`Terminator::For` in `ir.rs`/`cfg.rs` and the lowering of the four heads →
`validate` and the SSA builder → the passes' arms (`loans`, `dce`,
`forward`, `code_motion`, `dse`) → `prepare`'s `For` region op and the
machine → `break`/`continue` with the scope-drop rule → the parallel
split section, written after the loop exists.
