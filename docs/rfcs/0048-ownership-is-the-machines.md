# RFC-0048: ownership is the machine's — a value copies, a register is written once

Status: Accepted — owner and coordinator, 2026-09-18 (rule 7 landed first, `803d4f1a`)
Extends: RFC-0018 (references), RFC-0039 (one crossing), RFC-0041 (drop
insertion), RFC-0044 (a body is prepared once), RFC-0045 (`let` binds,
`=` assigns)

## Problem

`Value` implements `Drop`. That one fact costs the machine in four
places, all measured on 2026-09-18:

- **Every handler that holds a `Value` gets cleanup code.** `Drop::drop`
  takes `&mut self`, so a droppable value must have an address wherever
  it might be dropped — every scope end and every unwind path. In the
  bench binary 189 of 463 extern glue functions carry an unwind landing
  pad, and `ops::control::ret` keeps the returned `Value` in a stack
  frame (`sub $0x18,%rsp`) with a pad that writes it back to its register
  on unwind. A value that could live in two registers lives in memory.
- **A register is written after its definition.** `Value::use_from`
  decides at run time, by `kind`, whether a use is a move or a copy, and
  a move writes `Value::EMPTY` back into the slot; `Machine::take`
  asserts the slot is not already `Empty`; the arity-N glue
  `mem::take`s every argument regardless of type. Each is a store the
  frame's `Drop` glue needs so as not to drop twice — and the property
  they protect (a move-only SSA value has exactly one consuming use) is
  what `move_check` proved at compile time.
- **The frame drops by scanning.** `Registers`' `Drop` glue reads all 16
  slots' kinds at every return (4 % of `map cap | sum`).
- **The store's width is the compiler's to choose.** Because slots are
  written twice, the two writes can differ in width from the read: a
  `{ NonZeroU64, u64 }` head made every whole-`Value` copy a 16-byte
  load over two 8-byte stores, which does not forward (+17 %). Fixing
  the width was fixing a store that should not exist.

`prepare` knows, at every operand, whether it is consumed or copied
(`val_types`, `is_move_only`), and `drop_insertion` places every release
as an instruction. The machine already owns every fact `Drop` re-derives.

## Decision

**A `Value` is `Copy` and has no `Drop`.** Releasing a `Large` is an
explicit act of the machine — a drop instruction or the frame's exit —
and of a Rust holder that took ownership. Everything else copies.

1. **Two types, one bit pattern, one trait.** In the extern crate:
   `trait Release: Copy { fn release(self); }` and `Runtime { type Word:
   Release; .. }` — a runtime's word is `Copy`, and releasing it is the
   one thing the boundary knows how to do with it. `Owned<R>` is
   `#[repr(transparent)] struct Owned<R: Runtime>(ManuallyDrop<R::Word>)`
   whose `Drop` calls `release`: **the** owning type, defined once in
   the extern crate for every runtime, held by every Rust holder — a
   container's elements (`Vec<Owned<R>>`), an `Erased`, a closure
   carrier's value, an `Iter` stage's captured function. There is no
   `Runtime::Value` any more: a type variable `T` is `Owned<R>` at run
   time (RFC-0039's "the runtime's value" is the owning one), and the
   ABI — handler signatures, `Ref`, `Elements`, the glue's window — is
   `R::Word`. `Owned::from_word` is the identity, `into_word` is
   `ManuallyDrop::take`. The interpreter's `Value` implements `Release`
   (a `Large` drops through its header, a word does nothing) and that
   is all it says (owner, 17:05 and 17:20).
2. **The header keeps one thing the machine cannot know: the drop.** A
   `Large` erased from an extension type carries `drop_slot::<T>` in
   its header, because a Rust holder — an `Iter` stage owning a closure,
   a `Deque` owning its elements, an `Erased` in an `Object` field —
   releases values whose language type erased their Rust type, and no
   instruction stands at that release. The rest of today's vtable
   (registry, `type_id`, `composite`, `name`) is the next RFC's cut
   (owner, 15:05: after this one).
3. **`Registers { slots: [Value; 15], marked: u64 }`, `#[repr(align(64))]`**
   — 248 bytes padded to 256, four cache lines. Bit `i` of `marked` is
   "slot `i` owns a `Large`". `Heap { slots: Vec<Value>, marked: Vec<u64>
   }` above 15, one word per 64 slots. The slots are `MaybeUninit` (or
   `ManuallyDrop`): nothing runs per slot at frame end but the sweep.
4. **A register is written exactly once per definition.** `define` of a
   move-only type (`prepare` knows: `define::<LARGE>`) writes the slot
   and sets its bit; `define` of a word writes the slot. `assign`
   (RFC-0045) releases the old value if the bit is set, writes, and
   sets or clears. Nothing writes `EMPTY` after a take; `Kind::Empty`
   and `Kind::Undef` as run-time states go — `Undef` was an SSA
   definition and is emitted as one.
5. **A take is static and batched.** `prepare` emits which operand
   slots each operation consumes; `take::<N>` reads `N` slots into a
   tuple and clears their bits with one constant mask the operation
   carries; a copied word touches neither slot nor mask. `use_from`'s
   `match` on `kind` is deleted. The double-take check is `debug_assert!
   (marked & mask == mask)`. Tuple helpers are `#[inline(always)]`: a
   `(Value, Value)` is 32 bytes and crosses a call boundary through
   memory.
6. **Release is an instruction or the sweep.** `drop_value` (RFC-0041's
   op) takes the slot and calls `release`, clearing the bit. The frame's
   exit iterates the set bits of `marked` and releases those slots.
   `Machine.exit` and every other machine-held `Value` outside a
   register is released where the machine lets go of it.
7. **A Rust holder holds `Owned<R>`.** Every type in `acvus-extern` and
   `acvus-ext` that stores a runtime value it owns holds `Owned<R>`, and
   Rust's `Drop` releases it — nothing to call, nothing to forget.
   `Ref`, `RefMut`, `Elements` are words: they borrow and hold `R::Word`.
   The enumeration is the compiler's: a `Word` cannot be stored where an
   `Owned` is expected without `from_word`, and an `Owned` cannot be
   passed where the ABI wants a `Word` without `into_word` — both at the
   glue, never in a body.
8. **A panic is an exit, not a release.** Runtime errors are panics
   (RFC-0044 stage 2c); no cleanup runs and none is owed. A host that
   catches a script's panic and lives sweeps the frame's mark word at
   the catch — one place.

## What it costs

- Two names for one bit pattern, and a conversion at every store
  boundary — free at run time, a line in the source. The debug sweep
  asserts nothing is left marked at frame exit; a Rust holder cannot
  leak by omission, since `Owned` drops itself.
- `mem::drop(v)` on a `Copy` value is a no-op that reads like a release;
  the method is named `release` so the two do not read alike (owner,
  15:20).
- One bit test per `Large` `define`/`assign`, one mask clear per batched
  take, one bit iteration per frame exit — in place of a 16-slot kind
  scan, a store per move, a run-time `match` per use, and cleanup code
  in 189 glue functions.

## Rejected

- **Changing `Value`'s layout to make the second store cheap** — a
  `{ u8, [u8; 7] }` head (memory class, `sret`, +122 %) and a
  `{ NonZeroU64, u64 }` head (store-forwarding stall, +17 %): both
  measured 2026-09-18 under RFC-0047; both were fixing a store that
  should not exist.
- **`&[Released]` for the arity-N window** (a `ManuallyDrop<Value>` slice
  the glue `ptr::read`s once per position): correct for this
  interpreter, and a shape in which the glue knows how this interpreter
  tracks ownership. Extern glue assumes no interpreter; a `Copy` bound
  is the whole of what it may know.
- **`Value: Drop` kept, with `ManuallyDrop` only inside the register
  file**: removes the frame scan, keeps the 189 landing pads and the
  address requirement in every handler.
- **`Value: Copy` with manual `release` in every holder's `Drop`** (an
  explicit `Release` trait per type, or the composites' drop only via a
  `TypeId` list): `Value: Copy` does not break the build, so the ~30
  holders cannot be enumerated by the compiler and each can forget —
  measured 16:45 when the first attempt stopped on `Arr<T, N>: TyVar`'s
  blanket impl, which admits no per-type `release`. `Owned` makes the
  omission unwritable.
- **The drop fn in the operation instead of the header** (fully static
  release): right for every value a register holds and for nested
  containers whose element type the language names; wrong for a value a
  Rust holder owns under an erased type (an `Iter`'s closure), for which
  no instruction stands at the release. Possible as an operation with
  more design (owner, 15:15); the header word is the simple form and
  is kept.
- **Lazy drop** (`define::<LARGE>` releases a still-marked slot, so a
  loop body carries no drop op) and **drop fusion** (`drop_mask(m)` for
  consecutive drops): later stages, measured one variable each after
  this lands (owner, 13:58).

## Consequences

- Measured after it lands: `ops::control::ret` frame and pads; the
  landing-pad count in extern glue (189 → the number); `drop_glue::
  <Registers>` gone from every profile; `map cap | sum` (`ret` 6.3 % +
  `drop_glue` 4.2 % of its base profile); attention and mandelbrot.
- `use_from`, `Value::EMPTY`-after-take, `Kind::Empty`'s panic, the
  arity-N `mem::take` are gone; the glue reads `__args[i]` by copy.
- The next RFC cuts the vtable to the header's one drop word: the
  registry (a mutex and a `TypeId` hash on every `Large` erase, 86 % of
  an `AsSlice`-in-loop shape), `type_id`, `composite` (into `Kind`),
  `name`, and the release-mode `expect_type`.
- kovac inherits a value that is a `Copy` word pair and a register file
  whose ownership is a bitmask — lane ownership (RFC-0044 reflections).

## Order of work

Rule 7 landed first (`803d4f1a`: attention −34 %). One to-be for rules
1–6 and 8 in the interpreter worktree: `Release`/`Owned<R>` in the extern crate, `Runtime::Word` in place of
`Runtime::Value` through acvus-extern and acvus-ext, the
mark-word `Registers`, `define::<LARGE>`/`assign`, batched `take::<N>`,
`drop_value` and the sweep, the glue's copy. Measured first: the
landing-pad count (255 today), `ret`'s disassembly, `map cap`.
