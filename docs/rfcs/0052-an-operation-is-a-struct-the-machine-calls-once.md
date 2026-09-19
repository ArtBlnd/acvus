# RFC-0052: an operation is a struct, and the machine calls it once

Status: Accepted — 2026-09-19 (phase 1 merged 17718c76; the frame a6f1d50d; a synchronous call 550de866)
Supersedes: RFC-0044's machine representation (`Op`, `OpFn`, `Payload`,
`Flow`); RFC-0044's stages (prepare once, recognizers, chain, diamond,
loop, fused run, by-value ABI) stand as what the recognizers produce
Extends: RFC-0048 (ownership is the machine's — kept, re-implemented),
RFC-0047 (`AsSlice`/`Index`), RFC-0046 (a call's task)

## Problem

The machine was measured to the instruction on 2026-09-18: `int while` ran
128 instructions and 24 branches per iteration for three operations whose
work is five instructions. Per operation, 8.06 cycles: the dispatch
mechanism 61 %, the register file as memory 33 %, ownership bookkeeping 2 %,
the arithmetic 4 %. The loop was throughput-bound at the front end — twelve
taken branches per iteration cut fetch into ~10-instruction blocks — and
neither mispredicted nor missed cache.

The mechanism was the shape RFC-0044 stage 1 chose and five stages built on:
a fixed 32-byte `Op { f: OpFn, a, b, c, d, p }` in an array indexed by `pc`;
a `Payload` enum in a second array, reached by an index (a bounds check) and
matched on its variant (a discriminant branch) by every operation that needs
more than four words — every extern call, the loop, the diamond, the fused
run, the paths; behind the variant a `Box`, and behind the box the data.
Then the handler asked again what `prepare` had already decided — two enum
levels to reach a sync handler, a `match` on arity per fused call, a `dyn
Fn` vtable for the extern body, a kind test at the destination. Then it
returned a 24-byte `Flow` enum through memory (`sret`), and the run loop
matched it through a jump table. For one extern call: five dependent memory
steps to reach the data, three to five decisions the function pointer was
supposed to have made, and a memory round trip to say "next". A function
pointer means one dispatch; that machine made one dispatch and then four
more.

The core is **branchless**: every branch an operation takes beyond its own
`call`/`ret` is a fact `prepare` knew and threw away.

## Decision

1. **An operation is a struct that implements one trait, and it holds its
   successor.**

   ```rust
   pub trait Op: Send + Sync {
       fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit;
   }
   ```

   Its fields are the facts `prepare` decided, moved in and boxed once:
   register displacements, constants as words, a call's target as a plain
   `fn` pointer, a region's parts as chains. There is no `Op` record, no
   `OpFn`, no `Payload`, no payload array. A decision `prepare` made is a
   type or a field; a `run` body contains no `match`, no `let … else` and no
   kind test on a fact its own type carries.

   A straight-line run is a **chain**: each operation holds the next as a
   field and ends by calling it, and that call is a tail call, so the whole
   run is a line of `jmp *`. An operation with no successor ends the chain,
   and what it hands back is the chain's `Exit` — a word. At a joint that
   word is the `BlockId` the machine enters next; inside a region it is the
   word the part computed last. There is no separate `Terminator` trait and
   no `Block`.

2. **The stream holds the joints.** `Body::heads: Box<[Box<dyn Op>]>` is
   indexed by `BlockId`: one chain head per **joint**, a joint being a place
   where control genuinely chooses — `JumpIf`'s two targets, `Return`,
   `Suspend`, a real CFG join. The machine's loop is entered only at joints:

   ```rust
   loop { at = heads[at].run(self, 0); if at >= SENTINEL { return at } }
   ```

   What the chain removes is the slice walk. Walking a slice of fat pointers
   cost **seven instructions and two taken branches per operation**: the
   data pointer, the vtable slot, the machine argument, the `call`, then the
   `add`/`cmp`/`jne` of stepping the slice, then the `ret`. A chain is `mov`
   next, `mov` vtable, `jmp *` — **three instructions, one taken branch** —
   and its last node returns once per joint.

3. **A region is an operation, and its parts are chains.** A recognized
   `while` or `if` chooses nothing the recognizer did not already know, so
   it is an operation of the chain it sits in rather than a joint. The
   recognizers stand (RFC-0044: chain, diamond, loop, fused run) and produce

   ```rust
   pub struct Loop<C: Place> { head: Box<dyn Op>, cond: C::At, body: Box<dyn Op>, next: Box<dyn Op> }
   pub struct Diamond<C: Place> { cond: C::At, on_true: Box<dyn Op>, on_false: Box<dyn Op>, next: Box<dyn Op> }
   ```

   Each part is the head of its own chain, ended by `Yield` — the node with
   no successor that hands the region the word the part computed last.
   `Loop::run` is `loop { let c = head.run(m, r0); if C::read(regs, cond, c)
   == 0 { break } body.run(m, r0) }` and then its own tail call to `next`;
   `Diamond::run` is one word read, one arm picked, that arm's word handed
   to the successor, then its tail call. The head's `JumpIf` and the body's
   back `Jump` are not terminators a region dispatches — they are those two
   lines. A nested region is one more operation of the chain it sits in, so
   no id is chosen inside a region and no `BlockId` leaves one.

   **A region part hands its word forward.** A `while`'s condition is
   produced by the last operation of the head chain and read by the region
   above it; because the part *returns* its word, the condition rides out in
   the register the `ret` already uses, and `Loop` is specialized on where
   it reads it: `C = R0` where the head's last operation produced the
   condition, `C = Slot` where it did not — as in mandelbrot's `i < max && x
   * x + y * y < 4.0`, whose head ends in a `Diamond`. `prepare` chooses
   between the two, so neither `run` holds a test.

   **A region's parts hold no terminator, and that is structural.**
   `prepare::straight_run` walks a candidate range and stops at the first
   instruction that is not straight-line; the loop and diamond recognizers
   admit a shape only where that walk reached the shape's own
   `JumpIf`/`Jump`. A `return`, a `Switch`, a suspending call — each is a
   stop, so a `while` or an `if` holding one is not recognized and prepares
   as the blocks it was. There is no flag a region tests for a `return`
   inside it: the case that would need one cannot reach the type
   (`tests/block_splitting.rs::a_while_that_returns_is_not_a_region`).

   **A terminator owns blocks and ids, never a move.** A jump's parallel
   move is `Mov<const LARGE: bool, const WORD: bool> { dst: Off, src: Off }`
   — an operation of the block the edge leaves from, in the order `prepare`
   sequenced, a cycle going through the scratch register. What each move
   carries is its type, so no list is split into a word run and a `Large`
   run. Where the edge belongs decides where the operation goes: a `Jump`'s
   moves end its own block and the terminator becomes `Goto`; a region's
   edge out ends the region's last block; a `JumpIf`'s two edges take a
   block each, and only where they carry a move — an edge that carries none
   names its target directly. The moves into and out of a region sit in the
   enclosing list around it, and a diamond's join moves end each arm. A tail
   after an `if` is the next operation of the enclosing list, whether or not
   it reads an arm; `optimize::code_motion` decides which side of the
   `Diamond` it sits on.

   `Fused` stays an operation, and so does a chain: `Chain1/2/3<T, D, …>`
   carry the operand places as type parameters and the shape, the operators
   and the leaves in a `Plan` field, which is what took the chain family
   from 1404 instances per entry to 351. Large parts of a struct are boxed;
   the fields a `run` reads first are inline.

4. **No operation that may suspend is fused.** A call whose task is above
   `Sync` (RFC-0046) stays an operation of its own block with a suspending
   terminator after it; a `Loop`/`Diamond`/`Fused` body holds only
   operations that cannot suspend, so a fused operation never re-enters. The
   static fact is the callee's task, and the recognizer reads it.

5. **A `Value` is read and written through `&mut Machine`.** The register
   file is RFC-0048's design: **a frame is one cell**,
   `#[repr(C, align(64))] Cell { slots: [MaybeUninit<Value>; 16] }`, 256
   bytes, four cache lines starting one, with the frame's mark word in the
   slot past its registers, so `own_mask` is one `or` and `take_mask` one
   `and` on a field at a constant displacement: there is no chain-wide
   bitmap, no word-and-shift to compute, no straddle half. A body wider than
   a cell takes the cells its registers and its mark word reach, and one
   mark word covers a frame, which is where `prepare` stops
   (`MAX_FRAME_SLOTS`, 64).

   **An operation holds a byte displacement, not an index.** `Off` is
   `slot * 16`, multiplied once by `prepare`; `Slot` stays the
   language-level index and stays inside `prepare`, and the two are
   different types. A read is a load; a define is a store, plus one bit when
   the type is a `Large` (the operation's type says so: `CallExtern1<const
   LARGE: bool, const WORD: bool>`); a take of a `Large` clears its bit; a
   batched take clears one mask; the frame's exit releases the marked slots.

   **The mark word is an operand while the frame runs** (amended
   2026-09-20). `Op::run(&self, m, r0, marks) ->
   Exit` carries the frame's marks by value beside `r0`, and `Exit` is
   the pair `(at, marks)` — 16 bytes, two registers — so a chain's
   `take_mask` is `marks & !takes` and a `Large` define is `marks | bit`:
   register arithmetic, no load, no store. The word touches memory at
   three places only: the frame's entry (`param_marks` in), a suspension
   (the `SUSPEND` exit writes `marks` to the frame's slot and the resumed
   `run` reads it back), and the frame's exit (`RETURN` hands `marks` to
   `sweep`). A frameless chain (`Code::Expr`) has no marks and passes
   zero. Why: the memory RMW on the hot path was a load the CPU ordered
   against every in-flight store to the frame; on Zen it blocked on
   partial-address matches at ≈8.7 cycles each — `while let vec` paid 0.9
   of them per iteration at `d88f27be` and 1.8 after RFC-0059 moved every
   address (RFC-0059 Consequences). With the operand there is no load to
   block.

   Slot access is unchecked in release, on `prepare`'s proof that every slot
   is below `frame_len`. A slot's kind is static — it is one SSA value's
   type — so `prepare` writes the kind byte of every word-typed slot once,
   when the frame is made, and a word-typed operation stores the **word
   only**; the one kind a run can change is an option's (`None` is a depth
   word, RFC-0039), and only an option-typed slot is written whole. A
   word-typed slot is **read** the same way: `Return<WORD>` and `Mov<LARGE,
   WORD>` take the same `WORD` parameter `CallExtern1` carries, and where it
   is set the operation reads the word and leaves the mark word alone,
   because a slot the frame opened with a kind holds no `Large`. `Value:
   Copy`, `Owned<R>` in every Rust store, `Release` — RFC-0048 unchanged;
   the register file is rewritten to those rules and to no more.

   **A word with one use rides in the argument register.** `Op::run` takes
   `r0: u64` and returns one: a word-typed SSA value with exactly one use,
   in the immediately following operation of the **same chain**, is never
   written to the frame. The producer leaves it in the argument register the
   tail call already passes and the consumer reads it from there. This is
   wasm3's `r0`, on `dyn Op`.

   Where a word is, is a **type**, so no `run` tests for it:

   ```rust
   pub trait Place { type At; fn read(regs, at: Self::At, r0: u64) -> u64;
                     fn write(regs, at: Self::At, bits: u64) -> u64; }
   pub struct Slot;  // At = Off
   pub struct R0;    // At = ()
   ```

   `R0::At` is `()`, so an operation whose operand rides **holds no `Off`
   for it** — the displacement exists only where there is one, and nothing
   can read one that does not. `prepare` decides the place (`code::Where`)
   and the specialization follows: `Add<T, L, R, D>`, `NotBool<S, D>`,
   `Chain1/2/3<T, D, …>`, `JumpIf<C>`, `Diamond<C>`.

   `Add::<i64, Slot, Slot, Slot>::run` is twelve instructions: three
   `movzwl` field loads, the frame's base, two operand loads folded into the
   `add`, one store, the successor's data and vtable, `jmp *`.
   `Add::<i64, Slot, Slot, R0>::run` is **ten** — the `movzwl` of `dst` and
   the store are gone. `Add::<i64, R0, Slot, R0>::run` is **eight**.

   Two operands cannot both ride: a value that rides has exactly one use, so
   it is one operand of one operation. That combination is a defect in the
   ride analysis and panics at preparation. A word does not ride across a
   joint: a `Loop` reads its condition after its head chain has returned, so
   the condition stays in the frame, while a `Diamond` is an operation of
   the chain and its condition rides.

6. **An extern is a `fn` pointer.** `CallExtern1 { dst: Off, a: Off, takes:
   u64, f: Sync1, next }` calls `f` directly — no `Arc<dyn Fn>`, no handler
   enum at the call. The macro's glue is a `fn` item already, and `prepare`
   reads the declaration's arity form once (RFC-0044's by-value ABI) to pick
   the operation's type. The handler's ABI is `fn(&Rt, Value) -> Value`: it
   is handed the runtime, never the machine. `CallExtern1::<false, false>
   ::run` is an argument load, the frame's claim dropped with one `and`,
   `call *f`, a result store, `ret` — **zero compares before the call**,
   which is that operation's stated done condition.

7. **A frame is made where the machine already is.** A closure call does not
   construct a register file and sweep it per call: the callee's slots are
   the caller's frame region above `frame_len`, marked by one word, and
   released by one sweep of that word — the frame is a window, not an
   allocation. The window is the cells above the caller's, so a call makes
   one capacity compare and steps past its own cells; a callee that does not
   fit, or a chain deeper than the store's cells, roots a `Store` of its
   own.

   **A call's arguments are the callee's first registers.** `prepare` gives a
   body's parameters the frame's first registers, and the window begins at
   the cell above the caller's, so the caller writes each argument straight
   into the register the callee will read it from — one `LayArg` operation
   apiece, ahead of the call, exactly as `ArgWindow`'s `Mov`s do for an
   extern. `enter` copies no parameter, the frame's claim on them is the one
   `param_marks` store `Regs::of` already made, and there is no `Vec` and no
   `&mut [Value]` between the two frames: `CallDirect` and `CallIndirect`
   hold an `arity`, not an argument array (`Fused` never staged one: its
   operands are read from their registers at the call). A body that is one chain
   (`Code::Expr`) reads the run as `&[Value]` where it lies and `chain_value`
   runs on it, with no frame bound. Every frame keeps one cell above itself
   for that run, so `fits_above` asks for the callee's cells and that one;
   the rooted fallback copies the run into the frame it makes. The window
   also remembers, per calling frame, the body last bound above it, so a
   second call to the same body skips `open_frame` — `Store::bind`'s answer
   for a rooted frame, one compare on the machine's side.

   A call whose future outlives the frame — `CallDirectAsync`,
   `CallIndirectAsync`, the spawns, `CallHeavy` — still owns its arguments as
   a `Vec`, because the driver reads them after this frame is gone.

## The caller owns the frame, and a closure knows its entry

`Runtime` carries `type Frame` and `fn frame(&self) -> Self::Frame`, and
`call_now` takes `frame: &mut Self::Frame`. The borrow is the whole rule: a
synchronous closure call cannot own, make or free the frame it runs on,
because it is only lent one. `acvus-interpreter` answers `Store`; a runtime
with no frame answers `()`. Every site in `acvus-ext` that calls a closure
holds one: the four lazy stages (`Map`, `Filter`, `TakeWhile`, `SkipWhile`)
take `rt` in their constructor and keep a `Rt::Frame` for as long as the
stage lives, and the eager consumers make one before the drain loop. Nothing
makes one per element.

`Store` is `{ cells: Vec<Cell>, bound: usize }`. A `Cell` is sixteen
registers — four cache lines, starting one — and nothing else: a frame wider
than a cell has to be one run of `Value`s, and a mark word interleaved
between cells would break the displacement an `Off` already is. An unbound
frame is an empty `Vec` — no allocation — so a stage whose closure is an
`Expr` pays nothing for a frame it never enters.

`Store::bind(body)` is the only way to reach a frame. It answers whether the
frame already carries that body's slot kinds and entry constants, sizes the
`Vec` when the answer is no, and hands back a `Regs` whose mark word it has
already written with the body's `param_marks`. There is no `Regs` whose mark
word is unwritten, and no path that borrows a frame narrower than the body
it runs. Skipping the frame-opening walk is sound because a word register's
kind byte survives every word store (rule 5) and an entry constant's
register is scratch `prepare` gives no writer; what still runs per call is
what differs per call — the captures, the parameters, and the sweep on the
way out. The window keeps a constant three cells above the bound frame, and
the `Vec` cannot grow while a chain runs, because every frame below the
growth point borrows from it.

A closure's entry is chosen when the closure is made. `Code`'s payloads sit
behind `Arc`, `Body` and `Expr` each implement `Callable`, and a `FnValue`
holds `entry: Arc<dyn Callable>`. `MakeClosure` carries that entry,
projected once when the operation was prepared, so a closure call reads a
pointer and jumps; there is no per-call `match` on the code's shape, and the
site a panic names comes from the callable that panicked.

## A synchronous call is an operation

A call into another body used to be a terminator on a run-time test: it
looked the callee's prepared `Code` up, asked `may_suspend()`, and either
suspended or ran the callee to its value. Every user-function call therefore
ended a block, paid a terminator dispatch and a round trip through the
machine's block cursor, and could not sit in a region.

The fact is static. The callee's task is in the type the checker settled
(RFC-0046), and `prepare` reads it off the callee: `Sync` where the type
carries no effect.

- A call whose task is `Sync` is an **operation**. `CallDirect<LARGE, WORD>`
  and `CallIndirect<LARGE, WORD, THROUGH>` run the callee to its value
  inside `run`, through the window above the caller's frame, and store the
  result with `Regs::store::<LARGE, WORD>`. The callee is still reached by
  index at run time, because a caller's `prepare` may run before the
  callee's body exists; what it does not ask is whether that body can
  suspend.
- A call whose task is above `Sync` is `CallDirectAsync<LARGE>` or
  `CallIndirectAsync<LARGE, THROUGH>` — a terminator that hands the driver a
  future and leaves the block, with no arm that runs to a value.
- `is_straight_line` admits the first and refuses the second, so rule 4 lets
  a region hold a non-suspending call.

`Body::may_suspend` stays, with one reader: the assert, where a frame is
entered, that a body entered synchronously is one that cannot suspend. That
is what would catch a disagreement between the effect the checker read and
the body `prepare` produced, paid once per call rather than consulted per
decision.

## What it costs

- One vtable-slot load per operation that a function pointer in the stream
  would not pay — against the five loads and three decisions it replaces.
  The slot load has no dependency on the previous operation's result.
- **The tail call is a guarantee only a probe can hold.** Rust has no
  `become` on stable, so "every `run` ends in `jmp *`" is a property of the
  emitted code, not of the type.
  `acvus-interpreter-test/benches/asm_probe.rs` disassembles the release
  binary and asserts it — a bench and not a test, because a debug build has
  no tail call in it and a `cargo test` copy would pass on an artifact
  nobody runs. It carries a **closed list of eight families** that may end
  in `call` + `ret`, each with the stack address that is why: `Fused` (the
  staged-argument buffer), `CallIndirect` and `CallDirect` (the argument
  array), `SetStep` and `SetPath` (`&mut object`), `MakeObject` (the field
  buffer), `SpawnModule` and `SpawnExternAsync` (the argument window).
  LLVM's sibling-call rule refuses a tail call out of any function an
  alloca's address escapes, and no way of writing those `run`s changes it.
  Twenty-two instances, each a boundary operation costing 50–250
  instructions where one `call`/`ret` pair is a fraction. The list is closed
  in both directions: an operation outside it that ends so is red, and a
  listed family whose every instance tail-jumps is reported as an entry no
  longer needed. **A ninth family cannot join silently.**
- **The instance axis is what `r0` costs.** Specializing on `Place`
  multiplied the two families that carry it: on the `accum` bench binary,
  arith **135 → 716** and chain **337 → 674**, the whole machine **708 →
  1628** `Op::run` symbols and `.text` 5.67 → 6.69 MB (+18 %). Six place
  combinations are reachable for a binary operation — either operand may
  ride, never both; the result independently — and `prepare` dispatches on
  them at run time, so all six are emitted whether a given program reaches
  them or not. The estimate was ×2–3 for binary word ops; the measurement is
  ×5.3 for arith and ×2.3 overall, and no case regressed beyond the ±3 %
  rule.
- A rewrite of `acvus-interpreter`'s `code.rs`, `machine.rs`, `prepare.rs`,
  every `ops/*.rs`, `regs.rs`, and the tools that read operations (`oplist`,
  `asm_probe`, the loop-shape tests).

## Rejected

- **A `call`/`ret` per operation, through a slice of fat pointers** — the
  block form this RFC first decided. Measured: seven instructions and two
  taken branches per operation, and no shape of the loop removes the last
  three while the stream is a slice. The chain form pays three and one.
- **`chunks(4)` in the block loop**, written to give LLVM four hoisted
  fat-pointer loads: measured and removed. It cost 12 instructions and one
  not-taken branch per block — 24 and 2 per `int while` iteration, 18 % of
  that iteration's instructions — and the hoist did not happen: the inner
  loop loaded one fat pointer at a time. The recognizers leave 1–3-operation
  lists, so the chunk pays its counters on every list and never reaches
  four. A manual 4× unroll of the dispatch is rejected for the reason it
  always was: a straight-line group of four with no jump is a
  superinstruction, which is the recognizers' job, not the loop's.
- **A chain kernel reached through a pointer** rather than inlined into the
  operation: measured at 0.89 ns per node, which is a third of an `int
  while` iteration. `Chain1/2/3` keep their kernels inline.
- **`OpFn` + payload index** (RFC-0044 stage 1): measured in the Problem.
- **`OpFn` + `Box<dyn Data>`** (one function pointer, a typed data box): one
  load fewer than the vtable, but the pairing of `f` and its data is a
  convention the type system cannot see — a wrong pair is a cast, where
  `dyn Op` makes the pairing the type.
- **Variable-length inline records** (`f` then fields, the cursor a byte
  offset): one load fewer than the vtable and the best locality — but every
  jump target is a byte offset `prepare` computes, every record is laid out
  by hand, a wrong offset is memory corruption, and the facts an operation
  holds are read through casts instead of fields. Dyn dispatch used well,
  and linearity, are worth more than the last load.
- **`Flow` as a word**: removes the `sret` and the jump table but keeps a
  compare and a `cmov` per operation and a sentinel check; ending the chain
  instead removes the return altogether and leaves one compare per joint.
- **Rebasing RFC-0048's three work-in-progress rounds** (+516 core lines, a
  384-byte register file, a marked-slice bitmap, a dynamic define at 47 of
  66 sites): the shape works and is not this RFC's; rebasing it under a new
  stream costs more than writing rule 5 once. Its tests are carried; its
  code is not.

## Consequences

- **Measured against the master base, three alternating pinned reps, median
  of three, ns per iteration at n = 1e6** (bench profile, `opt-level = 3`,
  fat LTO, `taskset -c 15`). Twenty of twenty-one cases are at or below
  master; one is above it:

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

  Three bands are met and the rest are missed on the high side by one cause:
  dispatch is seven instructions per operation where every band was built on
  four. **The bands are the model's, and the model is corrected here rather
  than the measurement explained away.** The four cases that ran a diamond
  per iteration and were above master under the block form are all below it
  now: a `Diamond` is one word read, a branchless arm select and one
  straight arm.

- **`r0` removes instructions where a chain root or an arithmetic result
  feeds the next operation of the same chain, and nowhere else.** Every
  `while` body of the bench set rides its condition — eight of eight, one
  ride per loop, counted on the prepared listings — and mandelbrot goes from
  five rides to seven. What the ride removes is exact: in
  `Lt::<i64, Slot, Slot, R0>::run` the destination's `Off` load and the
  store are gone (13 → 11 instructions), and in `Loop<R0>::run` the frame
  base load and the compare against a cell become `test %rax, %rax`.
  `perf stat`, n = 1e7, instructions per iteration:

  | case | before | after | per iteration |
  |---|---:|---:|---:|
  | `int while` | 58.62 | 55.62 | **−3.0** |
  | `float while` | 97.64 | 94.64 | −3.0 |
  | `extern while` | 92.63 | 89.63 | −3.0 |
  | `call while` | 274.66 | 271.66 | −3.0 |
  | `branch while` | 128.66 | 124.66 | **−4.0** |
  | `option while` | 144.17 | 140.17 | −4.0 |
  | `collatz while` | 173.20 | 169.20 | −4.0 |
  | `grade while` | 205.10 | 200.44 | **−4.7** |
  | `while let vec` / `map`, `range \| sum`, `map *` | — | — | 0 |

  Branch counts are unchanged to within 0.0002 %, as expected: the ride
  removes a store and a load, not a jump. On mandelbrot, whole-binary
  `perf stat`: instructions −4.8 %, branches +0.03 %, cycles −6.0 %.

  **The wall clock does not follow the instruction count.** At 55
  instructions and 16 cycles per iteration the loop retires 3.5 instructions
  per cycle, so it is bound by the dependence chain through the indirect
  calls: removing three instructions from a chain that is not the critical
  path buys nothing on its own. Two cases are slower than their base
  (mandelbrot +3 %, `grade while` +2 %) while executing *fewer* instructions
  (−0.4 %, −2.3 %) and more cycles (+1.4 %, +2.2 %), with branch misses that
  do not account for it. The cause is placement. It is named and not tuned.

- **Three cases `r0` does not reach, each for a reason in the shape.**
  A loop's condition cannot ride, because `Loop::run` tests it after the
  head chain has returned and a joint lies between them — closed by §3,
  which makes the part hand its word forward. A call's arguments cannot
  ride: `CallDirect`, `CallIndirect` and `Fused` stage them in a window of
  consecutive registers, so there is no single place for a `Place` to name.
  A diamond's condition rides only where nothing was scheduled between it
  and its test — `grade while` gets one ride, `collatz while` none, because
  `optimize::code_motion` placed `i = i + 1` between its chain and its
  diamond. A chain's leaves never ride, by construction: a one-use word
  feeding a leaf is absorbed into the chain by the recognizer rather than
  reaching it as an operand, so only the root has a `Place`.

- **A synchronous call costs one operation.** `accum`'s `call while` —
  `let step = |x| -> x + 1; while i < n { i = step(i) }` — against the
  `a6f1d50d` base, five alternating pinned reps: **18.6 → 10.9 ns per
  iteration, −41.4 %**. The body went from five blocks to two, the `while`
  one `Loop` region whose body is the single operation
  `CallIndirect<false, true, true>`. What the iteration no longer pays: one
  terminator dispatch, two block entries and the block-cursor round trip
  between them, and the run-time `may_suspend()` load and branch.

- **The caller-owned frame halves the one case that was above master.** At
  the chain form, `perf record` put **63.8 % of `map cap | sum`'s cycles in
  `AcvusRuntime::call_now`**, its two hottest lines the frame-opening
  slot-kind walk and the frame set-up store, with `posix_memalign`/`cfree`
  beside them. With the frame owned by `map`'s stage: **18.1 → 9.1 ns**
  (−50 %). With the frame a `Vec` and the entry chosen at construction:
  **9.4 ns**. No allocator symbol appears on the path, the `Code`
  discriminant is not in the function, and `Regs` fell from 48 bytes to 24
  and `Machine` from 120 to 96. **The band was 5.5–8 ns and 9.4 misses it,
  and the frame is no longer where the time is**: of the case's cycles, the
  call's own entry is 28.4 %, the closure body's three dispatched operations
  36.4 %, the stage chain 9.6 %. The prologue prices at 0.44 ns, so no
  further shrinking of `Machine` reaches 7; what reaches it is fewer
  operations per element.

- **A call's arguments cost one operation each and no allocation.** Against
  master `641cd5cc`, three alternating pinned reps, median, `taskset -c 15`:
  `call while` **10.3 → 6.9 ns** (−33 %), `bf call` **22.1 → 20.3 ns/step**
  (−8.1 %), `logs` within the box's noise on every case (re-measured
  against a base rebuilt at `641cd5cc`), every other case of
  `accum` and `programs` inside ±3 % except `map cap | sum` at +3.7 %. `perf stat`, n = 1e7: `call
  while` runs **269.6 → 169.6 instructions** and **56.5 → 38.2 cycles** per
  iteration, and `CallIndirect::<false, true, true>::run` is **128 → 38
  instructions** ending in `jmp *`. `perf record` on it: `malloc` and `cfree`
  were 37.7 % of the case's cycles and are absent; what is left is
  `chain_value` 57.2 %, the call operation 17.5 %, `Expr::call_in` 9.9 %,
  `LayArg` 1.8 % — the two indirect calls and the operand space, not the
  arguments. The rooted fallback is never taken by the bench set: a
  `panic!` in it ran `accum`, `programs`, `logs`, `attention`, `mandelbrot`,
  `shapes` and `slice_ceiling` through without firing.

  `map cap | sum` is the one case above the band, at +2.0 instructions and
  +2.7 % cycles per element. Its closure is reached from an extern stage
  through `Callable::call_on`, which is handed `&mut [Value]` and fills the
  parameters itself: that path does not take rule 1, so the case gets none of
  the win, and the two instructions were not localized.

- **A `Vec` frame gives back what an inline one cost a frameless closure.**
  `map id` and `map add` call `Expr` bodies, which run with no registers at
  all (RFC-0044). An inline four-cell `Store` charged them 1 KiB inside the
  `Map` stage box for a frame they never enter: 3.2 → 3.4 (+6.2 %). An
  unbound frame is an empty `Vec` and the `Expr` entry never binds one: 3.3
  and 5.6, both inside ±3 % of master.

- **A count that does not move is not a code change.** `shapes field read`
  read +13 % in one round on a case the change could not reach; `perf stat`
  over the case alone counts 1,782,583,899 instructions against
  1,782,596,988 — a difference of 0.0007 % — and run alone the two binaries
  measure 9.4 and 9.6. The difference appears only when the case runs after
  the other five in one process.

- **The operation instance count** on the `accum` bench binary is 1829
  (`nm -C | grep -c 'acvus_interpreter::ops::'`), the chain family 689
  structs plus 337 `eval` functions. `CallExtern1`'s second const parameter
  is what a result whose register the frame opened with a kind costs: the
  arity-1 extern call has a `<false, true>` instance beside `<false, false>`
  and `<true, false>`, and the disassembly is the point — `<false, true>`
  ends in one `mov %rdx,0x8(%r14,%rax,1)` where `<false, false>` writes the
  kind byte first.
- **The read side of the same rule.** `Return<true>::next` is eight
  instructions where the single form was fourteen: the six of the mark-word
  read-modify-write are gone. On `map cap | sum`, which returns an `i64` per
  element, that is 1,849.1 M → 1,807.1 M instructions (−2.3 %) and 452.96 M
  → 452.56 M cycles, a wash — the dispatch loop is not instruction-bound
  there. The rule is kept for what it makes unrepresentable, not for a
  number.
- The chain, diamond, loop and fused run keep their measured shapes as
  structs; `AsSlice`/`Index` (RFC-0047) and the `switch` operation
  (RFC-0051) are structs added to the same trait.
- kovac's decode has its target shape: a 16-byte stream walked straight,
  straight-line runs with one exit, regions that are operations.

## What is left

- **The arena.** `prepare` lays a body's operations in one bump arena in
  chain order, so a successor is adjacent and the fat pointer's data load is
  a sequential line. Not built; measured one variable apart.
- **The eight families that hold a stack address.** Staging the argument run
  in the frame window was expected to close the list and did not: it removed
  the argument array, and each family turned out to hold a *different* stack
  local across its callee. Nineteen instances remain, down from twenty-two.
  `CallIndirect` tail-jumps in its three `THROUGH` instances and holds the
  `FnValue` it materialized out of the callee register in the other three;
  `CallDirect` holds the `Arc<Prepared>` the module table hands back, which
  it must also drop after the call; `Fused`'s is the held `Value`, whose
  address the tail `Deref` takes. Closing the list means removing those
  three locals, not the argument run.
- **The `Switch` operation** (RFC-0051), the one `todo!` in the machine.
- **A `Diamond` arm's value does not ride out of the region.** The arms
  already write the join's register directly, so no phi `Mov` stands between
  them and the successor; what remains is the store and the load, which a
  `Diamond` whose arms yield to `R0` would remove. It is also what would let
  mandelbrot's inner `while`, whose head is a short-circuit `&&` ending in a
  `Diamond`, ride its condition instead of taking `Loop<Slot>`.
