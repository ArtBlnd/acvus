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
instructions. Per operation, 8.06 cycles: the dispatch mechanism 61 %,
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
   `for chunk in ops.chunks(4) { for op in chunk { op.run(m) } }` — the
   chunk is what lets LLVM lift the four fat-pointer loads ahead of the
   calls (a cache line of stream); no manual unrolling. The terminator
   — `Jump`, `JumpIf { cond: u16, then, else }`, `Return`, `Suspend`
   (RFC-0046: stores the `Pending` in `Machine`, returns the sentinel) —
   returns the next `BlockId`. The machine's loop is

   ```rust
   loop { b = blocks[b].run(m); if b >= BlockId::SENTINEL { break } }
   ```

   One compare per **block**. Per operation the branches are the
   vtable `call` and its `ret`, and nothing else. `Flow` is gone.

3. **A superinstruction is an operation that owns blocks.** The
   recognizers stand (RFC-0044 stages 3–6: chain, diamond, loop, fused
   run) and produce structs: `Loop { head: Block, body: Block, enter,
   into_body, back, exit: SmallVec<[SlotMove; 2]> }` runs its blocks
   inside `run`; the body's terminator returns a block-local id or
   `EXIT`. `Diamond { on_true: Block, on_false: Block }`. `Fused { calls:
   SmallVec<[Call; 2]>, tail }`. `Chain3<T, S1, S2, S3> { shape: Shape,
   leaves: [u16; 4], .. }` — the `Shape` axis is a field, not a type
   parameter (1404 → 351 instances per entry; the research report
   `how-others-bound-a-fusion-table`). Large parts of a struct are
   boxed (`Box<Block>`, `Box<[Step]>`); the 16-byte stream and the
   fields a `run` reads first are inline.

4. **No operation that may suspend is fused.** A call whose task is
   above `Sync` (RFC-0046) stays an operation of its own block with a
   `Suspend` terminator after it; a `Loop`/`Diamond`/`Fused` body holds
   only operations that cannot suspend, so a fused operation never
   re-enters. The static fact is the callee's task; the recognizer
   reads it.

5. **A `Value` is read and written through `&mut Machine`.** The
   register file is RFC-0048's design as the owner drew it: `Regs {
   slots: [Value; 15], marked: u64 }`, `#[repr(align(64))]`, 256 bytes;
   a body needing more takes a `Vec<Value>` and a `Vec<u64>`. A read is
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

7. **A frame is made where the machine already is.** A closure call
   (`fn_value_call_sync`) does not construct a 384-byte `Registers` and
   sweep it per call (55 % of `map cap | sum`): the callee's slots are
   the caller's frame region above `frame_len`, marked by one word, and
   released by one sweep of that word — the frame is a window, not an
   allocation. The exact form is the implementation's, measured; the
   rule is one mask store to enter and one sweep to leave.

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
- **Manual 4× unrolling of dispatch**: a straight-line group of four
  with no jump is a superinstruction — the recognizers' job, not the
  loop's; `chunks(4)` gives LLVM the loads without the constraint.
- **Trimming the RFC-0048 WIP stack** (rounds 1–3, +516 core lines,
  `Registers` 384 B, `marked: &mut [u64]`, `define_dynamic` at 47 of 66
  sites, `call_extern_*` 4 → 30 instances — the review of 2026-09-18
  23:50): its shape works and is not the RFC's; rebasing it under a new
  stream costs more than writing rule 5 once.

## Consequences

- **The expectation, counted** (the brief's done condition, no
  compromise — this is the most performance-critical spot in the
  system, owner 00:25). `int while`, one iteration, three operations
  (`Lt<i64>`, `Add<i64>`, `Add<i64>`), today 128 instructions / 24
  branches / 12 taken / 24.2 cycles:

  | part | instructions | branches |
  |---|---:|---:|
  | dispatch per operation: fat-pointer load (hoisted by the chunk), vtable-slot load, `call`, `ret` | 4 × 3 = 12 | 6 taken |
  | `Lt::run`: two loads, compare/set, one store | 4 | 0 |
  | `Add::run` × 2: two loads, add, one store | 8 | 0 |
  | `Loop::run`: condition load and branch, back edge, chunk counter | ~6 | 2 |
  | **iteration** | **~30** | **8 (7 taken)** |

  Front-end-bound, so cycles follow taken branches: **24.2 → 12–14
  cycles, 4.44 → 2.2–2.6 ns per iteration** (Rust: 0.185). Every other
  case is banded the same way from its operation list in the brief:
  attention 64×64 (chain and `Index` dominate; dispatch halves) 136 →
  **80–95 µs**; `map cap | sum` (the frame window removes the 55 %, then
  dispatch) 13.8 → **5–6.5 ns**; mandelbrot 16.3 → **9–11 ns**. A
  disassembly line the table does not have is a defect.
- The chain, diamond, loop and fused run keep their measured shapes as
  structs; `AsSlice`/`Index` peepholes (RFC-0047) and the `switch`
  operation (RFC-0051) are structs added to the same trait.
- kovac's decode: a 16-byte stream, chunked, straight-line blocks with
  a terminator — the SIMD-window decode reflection of 2026-09-18 has
  its target shape.

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
