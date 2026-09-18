# RFC-0052: an operation is a struct, and the machine calls it once

Status: Draft — owner and coordinator, 2026-09-18/19
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
   and only an option-typed slot is written whole. `Value: Copy`,
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
   runtime, never the machine. **A closure an extern calls back
   therefore has no caller frame to take a window from and roots a
   `Store` of its own** (`machine::fn_value_call_sync`). That is a
   consequence of the ABI, not of rule 7; widen the ABI to carry the
   machine and that function has no callers left.

   `CallExtern1::<false>::run` is now argument load, the frame's claim
   dropped with one `and`, `call *f`, result store, `ret` — **zero
   compares before the call**, which is that operation's stated done
   condition.

7. **A frame is made where the machine already is.** A closure call
   (`fn_value_call_sync`) does not construct a 384-byte `Registers` and
   sweep it per call (55 % of `map cap | sum`): the callee's slots are
   the caller's frame region above `frame_len`, marked by one word, and
   released by one sweep of that word — the frame is a window, not an
   allocation. **The window is the next cell**, so a call makes one
   capacity compare (`callee.frame_len <= regs.above_cap`, where
   `above_cap` is a cell's worth while a cell is left and zero
   otherwise) and steps one cell up; a callee that does not fit, or a
   chain deeper than the store's cells, roots a `Store` of its own.
   One mask store to enter, one sweep of the cell's word to leave.

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

- **`map cap | sum` is the one case above master (+38 %), and its cost
  is not in this RFC's machine.** `perf record` on the bench binary:
  **63.8 % of cycles in `AcvusRuntime::call_now`**, whose two hottest
  lines are the frame it opens per element —

  ```
  14.75% : 31aefb: mov %rcx,0x68(%rsp)
  13.74% : 31afa0: movq $0x0,0x8(%rsi,%r13,1)   ; open_frame's slot_kinds walk
  11.71% : 31b1fa: cmp %rcx,%r14
  ```

  — with `posix_memalign` and `cfree` at 2.2 % and 2.7 % beside them.
  `map`'s stage is an extern, and an extern handler's ABI is
  `fn(&Rt, Value) -> Value` (rule 6): it is handed the runtime, never
  the calling `Machine`, so the closure it calls back cannot take rule
  7's window and roots a `Store` of its own per element
  (`machine::fn_value_call_sync` states the obligation). Rule 7 and rule
  6 meet here and rule 6 wins; widening that ABI is the next intent, not
  a fourth change in this round.

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
