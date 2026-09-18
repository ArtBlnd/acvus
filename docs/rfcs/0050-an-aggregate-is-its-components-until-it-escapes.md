# RFC-0050: an aggregate is its components until it escapes

Status: Accepted — owner and coordinator, 2026-09-19 ("이렇게 가자"; the owner reviews the code after it is built)
Extends: RFC-0053 (an aggregate that does not escape never exists — the
storage-slot form of this rule), RFC-0052 (§5 the register file, §6 the
frame, §7 the window), RFC-0048 (`Value: Copy`, `Release`, the mark
word), RFC-0051 (`match`, `Switch`), RFC-0039 (an option is its payload),
RFC-0046 (tasks), RFC-0018/0024 (storage, paths, references)
Depends on: RFC-0051's `Switch` operation (the enum half rests on it);
the slice's two-register form is RFC-0047's amendment, not this RFC.
Written from: the enumeration `every-site-an-aggregate-crosses` (94
sites, 2026-09-19) and the owner's design of the same day.

## Problem

An object or an enum value is built on the heap the moment it is
constructed and read through a hash lookup every time it is used.
RFC-0053 removed both for an aggregate that never leaves its body's
storage; what remains is every aggregate that is borrowed, passed,
returned, matched after a merge, or read out of a container:

- `enum match` 13.6 ns after RFC-0053 — a `MakeVariant` box and a tag
  chain remain.
- `vec of objects` 9.6 ns — every field read is `assert!(is_object)` +
  `FxHashMap::get` (`ops/storage.rs:132-138`); RFC-0053 does not reach
  an element of a container.
- `&Obj` cannot cross into a Rust handler at all — `Obj<V>` has no
  `impl` block and `#[derive(TyArg)]` emits no `deref`; the call panics
  (`acvus-extern/src/obj.rs:51`, `NO_STORAGE`).
- A frame holds 64 `Value` registers and nothing wider; a value that
  does not fit one register is not spilled — it is heap-allocated, at
  construction, forever.

The owner's reading of the last point (2026-09-19): the machine has
general registers and memory and no register class between them. A CPU
has three: general registers, wide (SIMD) registers, memory. A value
that does not fit a general register goes to a wide one; only a value
that fits neither is spilled. Our frame should have the same three, so
that the heap is what a spill is — the exception — and not where every
object lives.

## Decision

1. **An aggregate is its components.** In a body, an object is its
   fields, a tuple its positions, an enum or a `Result` its `(tag,
   payload)`, each an SSA value in its own register. Field read is the
   register; field write is a new SSA version with the existing phi
   builder (`SsaVar::Part`, RFC-0053); `match` is a `Switch` on the tag
   register; `?` is a compare on the tag and the payload as it is.
   **No `Make` operation exists for a value that stays in its body.**
   This is RFC-0053's rule widened from storage slots to SSA values:
   the escape predicate (`analysis/escape.rs`) is asked of every
   aggregate SSA value, and a `Return` and a borrowed argument are no
   longer escapes (rules 4 and 5).

2. **The frame has a wide register class.** Beside the `Value`
   registers, every frame has one wide `Cell` (256 bytes, 64-aligned)
   divided into **three 64-byte and two 32-byte slots** (offsets 0, 64,
   128 and 192, 224). A wide slot holds an aggregate in its **field
   layout**: `[Value; n]` in shape order (a variant: `[tag, payload…]`),
   so a 64-byte slot holds up to four fields and a 32-byte slot two.
   `prepare` chooses the class by the shape's size, which the type
   fixes, and assigns slots by live range as it assigns registers; a
   32-byte aggregate may take a 64-byte slot, never the reverse. An
   aggregate goes to a wide slot when its address is needed (rule 3)
   and stays in registers otherwise. The wide slot is owned by the
   frame: its `Large` fields are marked in the slot's own mark bits
   and released by the frame's sweep like any register.

3. **A reference to an aggregate is a projection.** `&obj` / `&mut obj`
   is one word, `Kind::LargeRef`, pointing at the aggregate's field
   layout — in a wide slot or on the heap (rule 4) — and a reader does
   not know which: field `i` is `base + i * 16` in both. `Regs::run_of`
   (a contiguous run lent as `&[Value]`) is the machine's form of it
   today. A projection never outlives its frame: the loans that forbid
   a borrow from escaping its body are the guarantee, and a `Vec`
   buffer's stability under a move is stated as an invariant where a
   projection is held across a suspension.

4. **The heap is the spill.** An aggregate is realized on the heap —
   one `Large` holding the same `[Value; n]` — only where its value
   must outlive the frame: stored as a container element, as another
   realized aggregate's field, as a variant payload that itself is
   realized, into `Commit`, captured by value into a closure, passed to
   an extern that keeps it (`Cross` erase), or handed to `Spawn`. That
   realization is **the only `Make`**, and `prepare` emits it from the
   escape predicate's verdict; the checker does not see it — there is
   no realized/unrealized distinction in the type system (`Repr`'s `#`
   keeps its meaning: a specialized, native representation). A body
   that needs more wide slots than the frame has, or an aggregate of
   more than four fields that must be addressed, spills to the heap the
   same way — listed by `prepare`, never a panic.

5. **Return is multi-value, read by the caller.** A body returning an
   aggregate leaves its components in its own registers; the caller
   reads them at static window offsets (`prepare` knows the callee's
   `frame_len` and the component registers) into its destinations. The
   copy is n reads; rule 7's window starts above the caller's frame, so
   the registers cannot coincide, and the RFC says so rather than
   promising zero.

6. **At a Rust boundary the glue converts, and nothing is heap-
   allocated.** An object by `&`/`&mut`: the handler receives the
   projection as `&[Value]` + shape (`&mut` writes through it — the wide
   slot is the object). An object by value into a Rust container: rule
   4's realization, at the glue. An enum: the glue builds a **real Rust
   enum** — `#[acvus::enum]` on the Rust type declares variant ↔ tag —
   by value from `(tag, payload)`; `&E` a stack temporary; `&mut E` a
   temporary plus write-back of `(tag, payload)`; a returned Rust enum
   is split back. An extern that **returns** an object writes it into
   the caller's wide slot through `Context` — an associated type of the
   runtime beside `Frame` (RFC-0052 §6), **owned by the frame**: it is
   the frame's wide `Cell`, made with the frame and dropped with it,
   never held by `Rt` (which is shared, `&`). A handler receives it as
   `&mut Rt::Context` beside `&Rt` for the call's duration, so a
   returned object is not a heap object either. A new frame is a new
   `Context`; a callee body has its own; the async path's future owns
   its frame and so its `Context`. Two frames never see one `Context`,
   which is what makes a second `&mut` unwritable — the isolation a
   multi-threaded runtime needs is the type's, not a rule's (`Context:
   Send`, not `Sync`).

7. **A container's element is realized, and a container is never a
   component set.** `Vec`, arrays, deques: their elements are heap
   aggregates (rule 4), read by projection (`&v[i]` is a `LargeRef`
   into the element). Scalar replacement of a container itself is
   **refused** — an array or a `Vec` is not an object with positional
   fields, and treating it as one is a wrong implementation, not an
   optimization (owner, 2026-09-19).

Rule 5 of RFC-0052 stands: a register holds one kind class. A payload
register whose variants disagree in class (`A(i64) | B(String)`) is
whole-typed and `Release` decides by kind; there is no conditional
`Drop`.

## What it costs

- The allocator gains a second register class and a size-class choice
  per aggregate; the frame's wide `Cell` is 256 bytes per frame,
  present whether used or not (a body with no aggregate still pays the
  space, not the time).
- Two forms of every aggregate read: register (component) and
  projection (`base + i*16`). `prepare` picks per site; a `run` never
  tests which.
- `Machine::exit` and `Return` become multi-value; every host that
  reads a returned aggregate reads components.
- The extern crate gains a projection crossing (`Obj<V>`'s first `impl`
  block), an enum glue (`#[acvus::enum]`, new — `#[derive(TyArg)]` is
  its nearest form and refuses generics), a `&mut` write-back mechanism
  (new: today `Cross::deref_mut` on an aggregate panics), and `Context`.
- The escape predicate changes its verdicts on `Return` and borrowed
  arguments; `exhaustive` shares it (RFC-0053), so `match` verdicts are
  re-run on the test set and reported.
- RFC-0051's `Switch` operation must exist first.
- The `TypeId` obligation across crates (`Object = Obj<Owned<Rt>>`)
  moves with the layout: `value.rs`, `layout.rs`, `space.rs`, the macro,
  together.
- Tag numbering: one numbering per program from the settled union type;
  a Rust enum crosses by variant name at the glue.

## Rejected

- **A realized/unrealized mark in the type system** (`#Obj` or another
  sigil): `#` already means `Repr::Specialized` — the native, unboxed
  form, the opposite of a heap object — and RFC-0041 states Object/Enum
  carry none; and the fact is the machine's (where a value lives), which
  `prepare` derives from the SSA, references and escapes it already
  has. The owner: "안 넣어. SSA와 ref와 MakeVariant로 다 표현된다. prepare가
  해야 한다."
- **A universal `(tag, payload)` SSA with conditional drops**: rule 5
  forbids a conditional `Drop`; the whole-typed payload with `Release`
  by kind is the answer, and it is what RFC-0053 already does.
- **Realizing at construction with a shape table and `Box<[Value]>`**
  (the earlier RFC-0050 sketch): one allocation and one query per
  access remain; RFC-0053 rejected it with that count.
- **A hash-map layout** for realized objects: `ops/storage.rs:132-138`
  is the cost per field; `layout.rs` already allocates a `String` per
  field name only to order it.
- **Adjacent general registers as the projection's home** (the design
  as first drawn): a contiguity constraint per aggregate on a 64-slot
  file that a component-scattered body already fills; the wide class
  gives the projection a fixed home the allocator does not have to
  carve.
- **Scalar replacement of arrays and `Vec`**: refused as a category
  (rule 7).
- **The slice inside this RFC**: a slice is one MIR value given two
  registers by the machine — RFC-0047's amendment; it needs neither a
  wide slot nor the escape predicate.

## Consequences

- `enum match`: `MakeVariant` gone, `Switch` on a tag register, payload
  phi — counted 6–9 ns (from 13.6). `vec of objects`: field reads by
  projection into the element, no hash — counted ~4–6 ns (from 9.6).
  `option match` unchanged (flat). The Brainfuck bench's `program[pc]`
  read becomes two loads through a projection; its row A becomes a
  `Switch`.
- `&Obj` handlers become expressible; a Rust handler receives a real
  Rust enum.
- A frame overflow of either class is a `prepare` refusal naming the
  body, never a run-time panic (today `code.rs:47`).
- The measured table is filled when the implementation lands.

## Order of work

No code before every site's form is written. The enumeration (94 rows)
is the checklist; this RFC's rules are the answers. Order: RFC-0051's
`Switch` operation → the wide register class and `LargeRef` in the
machine (with the projection read and the frame-owned sweep) → the
escape predicate widened to SSA values with `Return` and borrowed
arguments removed, and `prepare` emitting realization only at the
escape sites → multi-value return → the extern glue (`&[Value]` + shape,
`#[acvus::enum]`, `&mut` write-back, `Context`) → the benches.
