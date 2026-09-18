# RFC-0052: an operation is a struct, and the machine calls it once

Status: Accepted — owner and coordinator, 2026-09-19 (phase 1 merged 17718c76; the frame a6f1d50d; a synchronous call 550de866)
Supersedes: RFC-0044's machine representation (`Op`, `OpFn`, `Payload`,
`Flow`); RFC-0044's stages (prepare once, recognizers, chain, diamond,
loop, fused run, by-value ABI) stand as what the recognizers produce
Extends: RFC-0048 (ownership is the machine's — kept, re-implemented),
RFC-0047 (`AsSlice`/`Index`), RFC-0046 (a call's task)

## Problem

The machine was measured to the instruction on 2026-09-18 (`where-a-
dispatch-spends-its-cycles`): `int while` runs 128 instructions and 24
branches per iteration for three operations whose work is five
instructions. That measurement was taken in worktree `anatomy` at
`9726df06` — master plus RFC-0048 round 1 — where the case costs 4.44 ns.
**On master `50dd5c71` it costs 5.1 ns**: round 1 was worth −12 % on
exactly this case, and master does not carry it. Every band below is stated
against the 4.44 baseline; `int while`'s scales by 5.1 / 4.44 = 1.149 and
no other case's does, because no other differs from its measured master
base by more than 2 % (`scratchpad/expected-52.md`). Per operation, 8.06 cycles: the dispatch mechanism 61 %,
the register file as memory 33 %, ownership bookkeeping 2 %, the
arithmetic 4 %. The loop is throughput-bound at the front end — twelve
taken branches per iteration cut fetch into ~10-instruction blocks —
and neither mispredicts nor cache misses.

The mechanism is the shape RFC-0044 stage 1 chose and five stages
built on: a fixed 32-byte `Op { f: OpFn, a, b, c, d, p }` in an array
indexed by `pc`; a `Payload` enum in a second array, reached by
`payloads[p]` (a bounds check) and matched on its variant (a
discriminant branch) by every operation that needs more than four
words — every extern call, the loop, the diamond, the fused run, the
paths; behind the variant a `Box`, and behind the box the data. Then
the handler asks again what `prepare` already decided: `let
ExternHandler::Sync(SyncHandler::Arity1(f)) = … else panic` (two enum
levels), `fused_call`'s `match` on arity per call, `Arc<dyn Fn>`'s
vtable for the extern body, `define_dynamic`'s `kind` test. Then it
returns a 24-byte `Flow` enum through memory (`sret`), and the run loop
matches it through a jump table (`jmp *`). For one extern call: five
dependent memory steps to reach the data, three to five decisions the
function pointer was supposed to have made, and a memory round trip
to say "next". A function pointer means one dispatch; this machine
makes one dispatch and then four more.

The owner (2026-09-18 23:50): the core is **branchless**. Every branch
an operation takes beyond its own `call`/`ret` is a fact `prepare` knew
and threw away.

## Decision

1. **An operation is a struct that implements one trait.**

   ```rust
   pub trait Op: Send + Sync {
       fn run(&self, m: &mut Machine);
   }
   ```

   Its fields are the facts `prepare` decided, moved in and boxed once:
   register slots as `u16`, constants as words, a call's target as a
   plain `fn` pointer, a loop's blocks as blocks. There is no `Op`
   record, no `OpFn`, no `Payload`, no payload array, no `payload!`.
   The stream the machine walks is `Box<[Box<dyn Op>]>` — 16 bytes per
   operation: the data pointer and the vtable. Dispatch is the vtable
   call; the handler reads its own typed fields. A decision `prepare`
   made is a type or a field; a handler body contains no `match`, no
   `let … else`, no `kind` test on a fact its type carries.

2. **A block is straight-line, and only its terminator chooses.**

   ```rust
   pub struct Block { ops: Box<[Box<dyn Op>]>, end: Box<dyn Terminator> }
   pub trait Terminator: Send + Sync { fn next(&self, m: &mut Machine) -> BlockId; }
   ```

   Operations return nothing. The block runs its operations with
   `for op in ops { op.run(m) }`. The terminator
   — `Jump`, `JumpIf { cond: u16, then, else }`, `Return`, `Suspend`
   (RFC-0046: stores the `Pending` in `Machine`, returns the sentinel) —
   returns the next `BlockId`. The machine's loop is

   ```rust
   loop { b = blocks[b].run(m); if b >= BlockId::SENTINEL { break } }
   ```

   One compare per **block**. Per operation the branches are the
   vtable `call` and its `ret`, and nothing else. `Flow` is gone.

3. **A region is an operation, and it runs its parts straight.** A
   recognized `while` or `if` chooses nothing the recognizer did not
   already know, so it is one of its block's `ops`. The recognizers
   stand (RFC-0044 stages 3–6: chain, diamond, loop, fused run) and
   produce

   ```rust
   pub struct Loop { head: Box<[Box<dyn Op>]>, cond: Off, body: Box<[Box<dyn Op>]> }
   pub struct Diamond { cond: Off, on_true: Box<[Box<dyn Op>]>, on_false: Box<[Box<dyn Op>]> }
   ```

   `Loop::run` is `loop { head ops; if m.word(cond) == 0 { return }
   body ops }` and `Diamond::run` is one word read, one arm picked, one
   arm run. The head's `JumpIf` and the body's back `Jump` are not
   terminators a region dispatches — they are those two lines. A nested
   region is one more operation of the list it sits in, and the
   operations after it carry on where it left off, so no id is chosen
   inside a region and no `BlockId` leaves one.

   **A region's parts hold no terminator, and that is structural.**
   `prepare::straight_run` walks a candidate range and stops at the
   first instruction that is not straight-line; `recognize_loop` and
   `recognize_diamond` admit a shape only where that walk reached the
   shape's own `JumpIf`/`Jump`. A `return`, a `Switch`, a call into a
   body, an `Eval` — each is a stop, so a `while` or an `if` holding one
   is not recognized and prepares as the blocks it was. There is no flag
   a region tests for a `return` inside it, and no `RETURNS` parameter:
   the case that would need one cannot reach the type
   (`tests/block_splitting.rs::a_while_that_returns_is_not_a_region`,
   a `?` inside a `while`).

   **A terminator owns blocks and ids, never a move.** A jump's parallel
   move is `Mov<const LARGE: bool> { dst: Off, src: Off }` — an operation
   of the block the edge leaves from, in the order `prepare` sequenced
   (a cycle through the scratch register, as before). What each move
   carries is its type, so the list is no longer split into a word run
   and a `Large` run and a crossing between the two has no name. Where
   the edge belongs decides where the operation goes: a `Jump`'s moves
   end its own block and the terminator becomes `Goto`; a region's edge
   out ends the region's last block, or takes one block after it where
   that last unit is itself a terminator; a `JumpIf`'s two edges take a
   block each, and only where they carry a move — an edge that carries
   none names its target directly. `SlotMove` and `Moves` are gone, and
   with them the two run-time walkers the loop paid twice an iteration.

   Where the region's own edges go: the move into the body heads the
   body list, the back edge ends it, the move entering the loop
   precedes the region in the enclosing list and the move leaving it
   follows the region there; a diamond's join moves end each arm. A tail
   after an `if` is simply the next operation of the enclosing list,
   whether or not it reads an arm; `optimize::code_motion` decides
   which side of the `Diamond` it sits on
   (`acvus-interpreter-test/tests/block_splitting.rs` is both shapes).
   `EXIT`, `run_region` and `Arm` are gone.

   `Fused { calls: SmallVec<[Call; 2]>, tail }` stays an operation: it
   chooses nothing. `Chain3<T, S1, S2, S3> { shape: Shape, leaves: [u16;
   4], .. }` likewise — the `Shape` axis is a field, not a type
   parameter (1404 → 351 instances per entry; the research report
   `how-others-bound-a-fusion-table`). Large parts of a struct are
   boxed (`Box<[Block]>`, `Box<[Step]>`); the 16-byte stream and the
   fields a `run` reads first are inline.

   A body's own op counts follow from this: a nested `while` is an
   operation of the outer body's list, and the `Leave` that used to
   follow it is gone.

4. **No operation that may suspend is fused.** A call whose task is
   above `Sync` (RFC-0046) stays an operation of its own block with a
   `Suspend` terminator after it; a `Loop`/`Diamond`/`Fused` body holds
   only operations that cannot suspend, so a fused operation never
   re-enters. The static fact is the callee's task; the recognizer
   reads it.

5. **A `Value` is read and written through `&mut Machine`.** The
   register file is RFC-0048's design as the owner drew it: **a frame is
   one cell**, `#[repr(align(64))] Cell { slots: [MaybeUninit<Value>;
   15], marked: u64 }`, 256 bytes, four cache lines starting one; a
   `Store` is an array of cells and a body needing more than fifteen
   registers takes a `Heap` frame of its own. The mark word is the
   cell's, so `own_mask` is one `or` and `take_mask` one `and` on a
   field at a constant displacement: there is no chain-wide bitmap, no
   word-and-shift to compute, no straddle half.

   **An operation holds a byte displacement, not an index.** `Off` is
   `slot * 16`, multiplied once by `prepare`; `Slot` stays the
   language-level index and stays inside `prepare`, and the two are
   different types. `Add::<i64>::run` is eight instructions and no
   branch — three field loads, the frame's base, two operand loads
   folded into the `add`, one store, `ret` — where the index form paid
   three `shl $4` on top. A read is
   a load; a define is a store, plus one bit when the type is a
   `Large` (the operation's type says so: `CallExtern1<const LARGE:
   bool>`); a take of a `Large` clears its bit; a batched take clears one
   mask; the frame's exit releases the marked slots. Slot access is
   unchecked in release (`prepare` proved every slot below `frame_len`;
   `debug_assert!` in debug). A slot's kind is static — it is one SSA
   value's type — so `prepare` writes the kind byte of every word-typed
   slot once, when the frame is made, and a word-typed operation stores
   the **word only** (`Add::run`: two loads, one store); the one kind a
   run can change is an option's (`None` is a depth word, RFC-0039),
   and only an option-typed slot is written whole. A word-typed slot is
   **read** the same way: `Return<WORD>` and `Mov<LARGE, WORD>` take the
   same `WORD` parameter `CallExtern1` carries, and where it is set the
   operation reads the word and leaves the frame's mark word alone,
   because a slot the frame opened with a kind holds no `Large`.
   `Value: Copy`,
   `Owned<R>` in every Rust store, `Release` — RFC-0048 §1–§9 unchanged;
   the register file is rewritten to those nine rules and to no more.

6. **An extern is a `fn` pointer.** `CallExtern1 { dst: u16, arg: u16,
   f: fn(&Rt, Value) -> Value, order: u16 }` calls `f` directly — no
   `Arc<dyn Fn>`, no `SyncHandler` enum. The macro's glue is a `fn`
   item already; `Instances` hands out `fn` pointers. Where a handler
   today is a closure with captured state, that state becomes a field
   of the operation (a fact moved in) or the case is a refusal at
   registration.

   The handler's ABI is `fn(&Rt, Value) -> Value`: it is handed the
   runtime, never the machine. It is handed a frame instead — see §6.

   `CallExtern1::<false>::run` is now argument load, the frame's claim
   dropped with one `and`, `call *f`, result store, `ret` — **zero
   compares before the call**, which is that operation's stated done
   condition.

7. **A frame is made where the machine already is.** A closure call
   (`fn_value_call_sync`) does not construct a 384-byte `Registers` and
   sweep it per call (55 % of `map cap | sum`): the callee's slots are
   the caller's frame region above `frame_len`, marked by one word, and
   released by one sweep of that word — the frame is a window, not an
   allocation. **The window is the cells above the caller's**, so a call
   makes one capacity compare (`cells_for(callee.frame_len) <=
   regs.above_cells`) and steps past its own cells; a callee that does
   not fit, or a chain deeper than the store's cells, roots a `Store` of
   its own. One mask store to enter, one sweep of that word to leave.

## §6. The caller owns the frame, and a closure knows its entry

`Runtime` carries `type Frame` and `fn frame(&self) -> Self::Frame`, and
`call_now` takes `frame: &mut Self::Frame`. The borrow is the whole rule:
a synchronous closure call cannot own, make or free the frame it runs on,
because it is only lent one. `acvus-interpreter` answers `Store`; a
runtime with no frame answers `()`.

Every site in `acvus-ext` that calls a closure holds one. The four lazy
stages (`Map`, `Filter`, `TakeWhile`, `SkipWhile`) take `rt` in their
constructor and keep a `Rt::Frame` field for as long as the stage lives;
the six eager consumers (`reduce`, `fold`, `any`, `all`, `position`,
`extreme_by_key`, each in a synchronous and a suspending form) make one
before the drain loop. Nothing makes one per element.

### The frame is one `Vec`

`Store` is `{ cells: Vec<Cell>, bound: usize }`. A `Cell` is sixteen
registers — four cache lines, starting one — and nothing else: a frame
wider than a cell has to be one run of `Value`s, and a mark word
interleaved between cells would break the displacement an `Off` already
is. A frame of `n` registers therefore takes the `cells_for(n)` cells
that `n + 1` slots reach, and its mark word is the slot just past its
registers. An unbound frame is an empty `Vec` — twenty-four bytes, no
allocation — so a stage whose closure is an `Expr` pays nothing for a
frame it never enters.

`Store::bind(body)` is the only way to reach a frame. It answers whether
the frame already carries that body's slot kinds and entry constants,
sizes the `Vec` to the body's cells plus the window when the answer is
no, and hands back a `Regs` whose mark word it has already written with
the body's `param_marks`. There is no `Regs` whose mark word is
unwritten, which is why `own_mask` no longer exists as a separate step,
and there is no path that borrows a frame narrower than the body it
runs. `borrow` had a `match` on `slots <= INLINE_SLOTS`; `bind` has one
compare, on identity.

Skipping `open_frame` is sound because a word register's kind byte
survives every `set_word` (§5) and an entry constant's register is
scratch that `prepare` gives no writer. What still runs per call is what
differs per call: the captures, the parameters, and the sweep on the way
out.

The window keeps a constant three cells above the bound frame. The `Vec`
cannot grow while a chain runs — every frame below the growth point
borrows from it — so a chain deeper than that roots a `Store` of its own
at the call, exactly as a chain past the old `INLINE_CELLS` did.

### A closure's entry is chosen when the closure is made

`Code`'s payloads sit behind `Arc`: `Body(Arc<Body>)`, `Expr(Arc<Expr>)`.
`Body` and `Expr` each implement `Callable`, and a `FnValue` holds
`entry: Arc<dyn Callable>` in place of the `Arc<Code>` it used to hold.
`MakeClosure` carries that entry, projected once when the operation was
prepared, and a closure value copies the `Arc`. `fn_value_call_sync` is
`f.entry.call_on(f, args, frame)` and `Machine::call_fn_sync` is
`f.entry.call_in(f, args, self)`: the call reads a pointer and jumps. The
per-call `match f.code.as_ref()` is gone, and with it `Code::site` — the
site a panic names comes from the callable that panicked.

`Resume` replaces the old `Entry` on the asynchronous path, built by the
same trait method rather than by a second `match`.

## A synchronous call is an operation

Rule 2 says a block is straight-line and only its terminator chooses, and
rule 4 keeps a suspending operation out of a region. A call into another
body used to be a terminator on both counts — but on a run-time test:
`CallDirect` looked the callee's prepared `Code` up, asked
`may_suspend()`, and either suspended or ran the callee to its value.
Every user-function call therefore ended a block, paid a terminator
dispatch and a `Machine.at` round trip, and could not sit in a region.

The fact is static. The callee's task is in the type the checker settled
(RFC-0046), and `prepare` reads it off the callee register:
`call_task(callee_ty)`, `Sync` where the type carries no effect.

- A call whose task is `Sync` is an **operation**. `CallDirect<LARGE,
  WORD>` and `CallIndirect<LARGE, WORD, THROUGH>` run the callee to its
  value inside `run`, through the window above the caller's frame
  (`Machine::call_sync`, `call_fn_sync` — rule 7), and store the result
  with `Regs::store::<LARGE, WORD>`, the same three forms `CallExtern1`
  has. The callee is still reached by index at run time: `CallDirect`
  looks its module up through the table, because a caller's `prepare`
  may run before the callee's body exists. What it does not ask is
  whether that body can suspend.
- A call whose task is above `Sync` is `CallDirectAsync<LARGE>` or
  `CallIndirectAsync<LARGE, THROUGH>` — a terminator that hands the
  driver a future and leaves the block at `SUSPEND`, with no arm that
  runs to a value.
- `is_straight_line` admits the first and refuses the second, so rule 4
  now lets a region hold a non-suspending call.

`Body::may_suspend` stays, with one reader left: the assert in
`run_frame` that a body entered synchronously is one that cannot
suspend. That assert is what would catch a disagreement between the
effect the checker read and the body `prepare` produced; it is the guard
the removed run-time test used to be, moved to where the frame is
entered and paid once per call rather than consulted per decision.

The measurement is `accum`'s `call while` — `let step = |x| -> x + 1;
while i < n { i = step(i) }` — which went from five blocks with the call
its own terminator to two blocks, the `while` one `Loop` region whose
body is the single operation `CallIndirect<false, true, true>`: **17.9 →
10.4 ns per iteration, −41.9 %**.

## What it costs

- One vtable-slot load per operation (`(*data).vtable.run`) that a
  function pointer in the stream would not pay — against the five
  loads and three decisions it replaces. The chunked block loop issues
  the fat-pointer loads a cache line ahead, and the slot load has no
  dependency on the previous operation's result; the expectation is
  that it is hidden, and the brief's disassembly and cycle counts test
  that expectation, not a fallback.
- The boxed structs are heap-scattered; `prepare` may lay them in one
  arena later — measured first.
- A rewrite of `acvus-interpreter`'s `code.rs`, `machine.rs`, `prepare.rs`
  (the emitters), every `ops/*.rs` (each handler becomes a struct + one
  `impl Op`), `regs.rs`, and the tools that read operations (`oplist`,
  `asm_probe`, the loop-shape tests). RFC-0048's three WIP rounds in the
  interpreter worktree are **not** carried: their tests (the three
  drop-counter suites, the fixture runtimes following the model) are;
  their code is superseded by rule 5.
- The `Chain` instance count stays three-way (`T`, slots) — the table
  question (cold instances) is RFC-0044's recorded item, not this one.

## Rejected

- **`OpFn` + payload index** (RFC-0044 stage 1, today): measured above.
- **`OpFn` + `Box<dyn Data>`** (one function pointer, a typed data
  box): one load fewer than the vtable, but the pairing of `f` and
  its data is a convention the type system cannot see — a wrong pair
  is a cast; `dyn Op` makes the pairing the type.
- **Variable-length inline records** (`f` then fields, `pc` a byte
  offset): one load fewer than the vtable and the best locality — but
  every jump target is a byte offset `prepare` computes, every record
  is laid out by hand, a wrong offset is memory corruption, and the
  facts an operation holds are read through casts instead of fields.
  The owner (2026-09-19 00:10): dyn dispatch, used well, and linearity
  over the last load. Decided, not deferred.
- **`Flow` as a word** (the 2026-09-18 22:30 proposal): removes the
  `sret` and the jump table but keeps a compare-and-`cmov` per
  operation and a sentinel check; the block/terminator split removes the
  return altogether and leaves one compare per block.
- **`chunks(4)` in `Block::run`**, written to give LLVM four hoisted
  fat-pointer loads: **measured and removed**. It cost 12 instructions
  and one not-taken branch per block — 24 and 2 per `int while`
  iteration, 18 % of that iteration's instructions — and the hoist did
  not happen: the inner loop loaded one fat pointer at a time. The
  recognizers leave 1–3-operation lists, so the chunk pays its counters
  on every list and never reaches four. A manual 4× unroll of the
  dispatch is rejected for the reason it always was: a straight-line
  group of four with no jump is a superinstruction, which is the
  recognizers' job, not the loop's.
- **Trimming the RFC-0048 WIP stack** (rounds 1–3, +516 core lines,
  `Registers` 384 B, `marked: &mut [u64]`, `define_dynamic` at 47 of 66
  sites, `call_extern_*` 4 → 30 instances — the review of 2026-09-18
  23:50): its shape works and is not the RFC's; rebasing it under a new
  stream costs more than writing rule 5 once.

## Consequences

- **The count, corrected by the machine.** `int while`, one iteration,
  three operations (`Lt<i64>`, `Add<i64>`, `Add<i64>`) inside one
  `Loop`. The model said ~30 instructions, then ~44; the machine says
  **54**, and the disassembly says where the ten went. Measured on the
  `accum` bench binary (`perf stat`, n = 1e7 vs 1e8, difference over
  9e7 iterations, `taskset -c 15`): **60.0 instructions / 12.0 branches
  / 9.0 taken / 3.0 indirect / 16.8 cycles** per iteration, of which the
  harness's own Rust reference loop (`acc += black_box(i)`) is about six
  instructions and one taken branch.

  | part | instructions | branches |
  |---|---:|---:|
  | `Loop::run`, head cursor + body cursor + back edge (`mov`, `mov`, `jmp`) | 3 | 1 taken |
  | the op-list walk per operation (`mov` data, `mov` vtable, `mov` machine, `call *`, `add`, `cmp`, `jne`) × 3 | 21 | 3 taken + 3 indirect, 2 not taken |
  | the loop's condition (`mov` cell, `cmpq`, `je`) | 3 | 1 not taken |
  | the three `ret`s | — | 3 taken |
  | `Lt::<i64>::run` | 10 | 0 |
  | `Add::<i64>::run` × 2 | 16 | 0 |
  | **iteration, measured** | **54** | **11 (8 taken, 3 indirect)** |

  **Dispatch is seven instructions per operation, not four.** Four are
  the load of the data pointer, the load of the vtable slot, the machine
  argument and the `call`; the other three are the `add`/`cmp`/`jne` of
  walking a slice of fat pointers, and no shape of the loop removes them
  while the stream is a slice. The eight taken branches the model
  predicted are exactly the eight measured. **Zero terminator dispatches
  and zero `Mov` operations run in this iteration**: the region reads
  `cond` itself, and the register assignment made both loop edges
  identities.

  Against run 8's 130 instructions / 27 branches / 18 taken / 5 indirect
  / 27.5 cycles, and master's 128 / 24 / 12 / 24.2: **the 64
  instructions that ran between operations are down to 27** — the two
  block entries, the two `chunks(4)` counter sets and the two terminator
  dispatches are all gone, and what is left is the op-list walk itself.

  Wall time, three pinned reps at n = 1e6: `int while` **3.3 ns**
  against master's 5.1 (−35 %). The RFC band for this case is
  **2.53–2.99**; the measurement is above it and the disassembly matches
  the count above, so **the band is the model's error, not the
  machine's**: at seven instructions of dispatch per operation and three
  operations, 3.1–3.7 ns is what this shape costs, and 3.3 is inside it.

- **A synchronous call costs one operation.** `accum`'s `call while`
  (`let step = |x| -> x + 1; while i < n { i = step(i) }`), against the
  `a6f1d50d` base, five alternating pinned reps per case: **18.6 → 10.9
  ns per iteration, −41.4 %**. The body went from five blocks — the
  head's compare, the call's own block, the back edge — to two, the
  `while` a single `Loop` region whose body is the one operation
  `CallIndirect<false, true, true>`. What the iteration no longer pays:
  one terminator dispatch, two block entries and the `Machine.at` round
  trip between them, and the run-time `may_suspend()` load and branch.
  Every other case in the table is inside ±3 % of the base when measured
  per case; the whole-binary sweep on a loaded box is not, and the
  per-case alternating form is what the numbers above are taken with.

- **Measured, against the master base (`scratchpad/expected-52.md`),
  three alternating pinned reps, median of three.** Twenty of
  twenty-one cases are at or below master; one is above it:

  | case | master | band | measured | Δ vs master |
  |---|---:|---|---:|---:|
  | `int while` | 5.1 | 2.53–2.99 | **3.3** | −35 % |
  | `float while` | 7.85 | 3.5–4.5 | **4.8** | −39 % |
  | `range \| sum` | 1.7 | — | **1.7** | 0 % |
  | `map id \| sum` | 3.6 | 1.8–2.6 | **3.2** | −11 % |
  | `map add \| sum` | 5.95 | 3.0–4.0 | **5.6** | −6 % |
  | `map cap \| sum` | 13.7 | 5–6.5 | **18.9** | **+38 %** |
  | `extern while` | 7.8 | 3.5–4.5 | **4.9** | −37 % |
  | `branch while` | 10.2 | 5–7 | **7.0** | −31 % |
  | `option while` | 11.5 | 5–6.5 | **7.6** | −34 % |
  | `while let vec` | 13.3 | 7–9 | **10.1** | −24 % |
  | `while let map` | 15.3 | — | **11.8** | −23 % |
  | `collatz while` | 12.9 | 6–8 | **9.0** | −30 % |
  | `grade while` | 16.7 | 8–11 | **13.2** | −21 % |
  | attention 64×64 | 133.4 µs | 80–95 µs | **102.5 µs** | −23 % |
  | mandelbrot | 16.3 | 9–11 | **13.6** | −17 % |
  | shapes `field read` | 16.2 | 12–14 | **9.3** | −43 % |
  | shapes `field write` | 14.9 | — | **8.6** | −42 % |
  | shapes `construct` | 42.2 | 34–38 | **38.0** | −10 % |
  | shapes `enum match` | 33.9 | — | **27.4** | −19 % |
  | shapes `option match` | 11.5 | — | **7.6** | −34 % |
  | shapes `vec of objects` | 13.4 | — | **9.2** | −31 % |

  Three bands are met (`branch while` at 7.0, `construct` at 38.0,
  `field read` below its band at 9.3); the rest are missed on the high
  side by the same cause the count above names — dispatch is seven
  instructions per operation where every band was built on four. **The
  bands are the model's, and the model is corrected here, not the
  measurement explained away.** The four diamond-per-iteration cases
  that run 8 had above master (`branch while` +11 %, `collatz` +11 %,
  `grade` +18 %, mandelbrot +8 %) are all below master now: a `Diamond`
  is one word read, a branchless arm select and one straight arm.

- **§6 measured, three alternating pinned reps against a `17718c76`
  binary built on the same box in the same hour** (`taskset -c 15`,
  n = 1e6, median of three; the run-9 column above is reproduced by that
  binary to within 5 % on every case, so it is the same scale):

  | case | 17718c76 | §6 | Δ |
  |---|---:|---:|---:|
  | `int while` | 3.2 | **2.9** | −9.4 % |
  | `float while` | 4.9 | **4.8** | −2.0 % |
  | `range \| sum` | 1.7 | **1.7** | 0 % |
  | `map id \| sum` | 3.2 | **3.4** | **+6.2 %** |
  | `map add \| sum` | 5.5 | **5.6** | +1.8 % |
  | `map cap \| sum` | 18.2 | **9.0** | **−50.5 %** |
  | `extern while` | 4.7 | **4.7** | 0 % |
  | `branch while` | 6.8 | **6.8** | 0 % |
  | `option while` | 7.3 | **7.4** | +1.4 % |
  | `while let vec` | 10.1 | **10.2** | +1.0 % |
  | `while let map` | 12.2 | **12.2** | 0 % |
  | `collatz while` | 9.0 | **9.0** | 0 % |
  | `grade while` | 13.2 | **13.3** | +0.8 % |
  | attention 64×64 | 102.9 µs | **101.5 µs** | −1.4 % |
  | mandelbrot | 14.1 | **13.6** | −3.5 % |
  | shapes `field read` | 9.2 | **9.3** | +1.1 % |
  | shapes `field write` | 8.8 | **8.6** | −2.3 % |
  | shapes `construct` | 36.5 | **36.9** | +1.1 % |
  | shapes `enum match` | 27.3 | **28.0** | +2.6 % |
  | shapes `option match` | 7.3 | **7.8** | **+6.8 %** |
  | shapes `vec of objects` | 9.2 | **9.0** | −2.2 % |

  Two cases are above the ±3 % rule. `map id | sum` is explained below
  and is reproducible (3.2/3.2/3.2 against 3.4/3.4/3.4). `shapes option
  match` calls no closure and this change reaches nothing it runs; its
  reps are 7.4/7.3/7.2 against 7.8/7.4/7.9, so the medians differ by
  more than the spread of either — **it is recorded as unexplained, not
  as noise.**

- **`map cap | sum` was the one case above master (+38 %); §6 halves
  it, and it is still the one case that misses its band.** At run 9,
  `perf record` put **63.8 % of cycles in `AcvusRuntime::call_now`**,
  its two hottest lines the `open_frame` slot-kind walk and the frame
  set-up store, with `posix_memalign`/`cfree` beside them. With the
  frame owned by `map`'s stage: **18.1 → 9.1 ns** (−50 %). With the
  frame a `Vec` and the entry chosen at construction: **9.4 ns**, which
  is 9.1 within the spread of three alternating pinned reps against a
  `17718c76` binary built and measured on the same box in the same hour.

  The three lines the caller-owns-the-frame round left behind are
  answered, and two of them are gone. `perf annotate` on
  `Body::call_on` — the function the closure's entry now jumps to:

  ```
   8.24% : 2c6010: push %rbp                    ; the call's own prologue
   8.11% : 2c6011: push %r15
   7.32% : 2c6070: shl  $0x4,%edi               ; the frame's byte displacement
   7.31% : 2c6419: mov  %r11w,0x3c(%rsp)        ; Regs, two bytes at a time
  ```

  No allocator symbol appears on the path at all; the `Code`
  discriminant is not in the function, because the entry decided it; the
  `mov %rcx,0x68(%rsp)` that was 15 % is gone, because `Regs` fell from
  48 bytes to **24** and `Machine` from 120 to **96**. What is left is
  six `push`es and a 152-byte stack frame — the brief asked for fewer
  than four `push`es and did not get them.

  **The band was 5.5–8 ns and 9.4 misses it, and the frame is no longer
  where the time is.** Of the case's cycles: `Body::call_on` 28.4 %
  (bind, captures, parameters, `Machine::new`, sweep), the closure
  body's three dispatched operations 36.4 % (`TakeThrough` 14.9 %,
  `Return` 11.0 %, `Add<i64>` 10.5 %), the stage chain 9.6 %,
  `expect_type::<i64>` 3.7 %. The whole prologue prices at 0.44 ns, so
  no further shrinking of `Machine` reaches 7. What reaches it is fewer
  operations per element, which is what a call as an operation and a
  chain kernel are for.

- **A `Vec` frame gives back what an inline one cost a frameless
  closure.** `map id` and `map add` call `Expr` bodies, which run with
  no registers at all (RFC-0044 stage 4). The inline four-cell `Store`
  charged them 1 KiB inside the `Map` stage box for a frame they never
  enter: **3.2 → 3.4** (+6.2 %). An unbound frame is now an empty `Vec`
  and the `Expr` entry never binds one: **3.3** and **5.6**, both back
  inside ±3 % of master.

- **`shapes option match` is not unexplained; it was layout.** The
  caller-owns-the-frame round recorded +6.8 % on a case that calls no
  closure. It reads 7.6 against master's 7.4 here, inside the rule. The
  same shape appeared on `shapes field read` (+13.0 %) and was measured
  rather than argued: `perf stat` over the case alone counts
  **1,782,583,899** instructions against **1,782,596,988** — a
  difference of 0.0007 % — and run alone the two binaries measure 9.4
  and 9.6. The +13 % only appears when the case runs after the other
  five in one process. A count that does not move is not a code change.

- The operation instance count on the `accum` bench binary fell from
  **2696 to 1829** (`nm -C | grep -c 'acvus_interpreter::ops::'`), the
  chain family to 689 structs plus 337 `eval` functions. The two above
  run 8's 1827 are `CallExtern1`'s second const parameter: a result
  whose register the frame opened with a kind is stored as its word
  alone (rule 5), so the arity-1 extern call has a `<false, true>`
  instance beside `<false, false>` and `<true, false>`. The
  disassembly is the point — `<false, true>` ends in one
  `mov %rdx,0x8(%r14,%rax,1)`, where `<false, false>` writes the kind
  byte first.
- **The read side, measured on master `92fcc405`** (`accum`, `taskset -c
  4`, 15 alternating pinned reps). `Return<true>::next` is eight
  instructions where `Return`'s single form was fourteen: the six of the
  mark-word read-modify-write (`movzwl 0x22(%rsi)`, `shr`, `mov`, `rol`,
  `shl`, `and`) are gone. On `map cap | sum`, which returns an `i64` per
  element, that is **1,849.1 M → 1,807.1 M instructions (−2.3 %)** — and
  **452.96 M → 452.56 M cycles**, a wash: the dispatch loop is not
  instruction-bound there. Wall clock over all cases stays within the
  machine's ±3 % run-to-run spread. The rule is kept for what it makes
  unrepresentable, not for a number.
- The chain, diamond, loop and fused run keep their measured shapes as
  structs; `AsSlice`/`Index` peepholes (RFC-0047) and the `switch`
  operation (RFC-0051) are structs added to the same trait.
- kovac's decode: a 16-byte stream walked straight, straight-line
  blocks with a terminator, regions that are operations — the SIMD-window
  decode reflection of 2026-09-18 has its target shape.

## Order of work

One to-be, one agent, a fresh worktree from master: the trait, the
block/terminator machine, the register file to rule 5, `prepare`
emitting structs, every operation as a struct (the recognizers'
outputs included), extern `fn` pointers, the frame window, the tools.
Done conditions are disassembly and the whole bench table: `Add::run`
two loads, one store, zero branches; `CallExtern1::run` zero decisions
before the call; the block loop one compare per block; every bench
within its stated band or better, none slower; the four crates and the
drop-counter suites green.
