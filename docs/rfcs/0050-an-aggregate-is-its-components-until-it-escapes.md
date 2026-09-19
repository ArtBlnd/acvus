# RFC-0050: an aggregate is its components until it escapes

Status: Accepted — 2026-09-19; rules 2, 3, 5, 6 amended and 8, 9 added 2026-09-20
Extends: RFC-0053 (an aggregate that does not escape never exists — the
storage-slot form of this rule), RFC-0052 (§5 the register file, §6 the
frame, §7 the window), RFC-0048 (`Value: Copy`, `Release`, the mark
word), RFC-0051 (`match`, `Switch`), RFC-0039 (an option is its payload),
RFC-0046 (tasks), RFC-0018/0024 (storage, paths, references)
Depends on: RFC-0051's `Switch` operation (the enum half rests on it);
the slice's two-register form is RFC-0047's amendment, not this RFC.
Written from: the enumeration `every-site-an-aggregate-crosses` (94
sites, 2026-09-19).

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

The last point is a missing register class: the machine has general
registers and memory and nothing between them. A CPU has three: general
registers, wide (SIMD) registers, memory. A value
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

2. **A frame is registers, and an aggregate that needs an address is a
   run of them** (amended 2026-09-20; re-amended the same day after the
   first build showed the earlier wording made a second register class).
   There is one kind of register. A body's frame is `frame_len` slots
   (`Cell`s of sixteen `Value`s, `Off` addressing), bounded at **320**
   (the 64 a scalar body may use plus 256 for runs); `prepare` colors
   scalar values one register at a time and an aggregate that needs an
   address (rule 3) as a **run** of adjacent registers holding its
   flattened field layout (rule 8), by live range as it colors everything
   else. A body with no run has the frame it has today. The scratch
   register `order_moves` may take is reserved before emission, so
   `frame_len` is final when the first `Off` is written and a run begins
   where the scalar registers end — not at a constant. When the runs
   would take the frame past 320, aggregates are spilled to the heap
   (rule 4) **in ascending loop depth**, the shallowest first, ties broken
   by the longest live range first: what a loop touches stays in the
   frame. **Marks are per register, as today**: the mark word covers
   registers 0–63, and a frame past 64 registers has ⌈`frame_len` / 64⌉
   mark words in its mark slots; an operation's `take_mask`/define mask
   names the word its registers fall in, decided at `prepare`. A run's
   `Large` fields are released by the sweep exactly as any register's:
   no run bit, no second kind of mark, no operation that only aggregates
   run. The machine reads nothing new: a field of a run is an ordinary
   register at `base + off(i)`, and `Regs::run_of` is the projection.
   The window (`WINDOW_CELLS`) is a fact of the caller's frame; a callee
   whose frame does not fit roots a `Store` as today, and the count of
   such calls in the bench set is the number that decides whether the
   window grows.

3. **A reference to an aggregate is a projection, and an addressed
   aggregate has one home.** `&obj` / `&mut obj` is one word,
   `Kind::LargeRef`, pointing at the aggregate's flattened layout — in a
   wide run or on the heap (rule 4) — and a reader does not know which:
   field `i` is `base + off(i)` in both. An aggregate SSA variable whose
   address is taken anywhere in its body **lives in its wide run for its
   whole live range**: every read of a component after that point is a
   load from the run, every write a store to it, so a write through a
   projection is never stale against a register copy. An aggregate never
   addressed stays in registers (rule 1). `prepare` decides the home
   once per variable; a `run` never tests which. A projection never
   outlives its frame: the loans that forbid a borrow from escaping its
   body are the guarantee, and a `Vec` buffer's stability under a move
   is stated as an invariant where a projection is held across a
   suspension.

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

5. **Return is multi-value, at the window's run** (amended 2026-09-20).
   The run of registers at the window boundary — where a call lays its
   arguments (`callindirect-without-a-heap`, RFC-0052 §7) — is also
   where a callee leaves an aggregate result: the arguments are consumed
   by then, the callee writes the components at `window_base + i`, and
   the caller reads them into its destinations from offsets it knows
   statically **without knowing the callee** — which is what a
   `CallIndirect` through a closure value requires, since two bodies of
   one type assign their own registers differently. The copy is n
   reads; the window starts above the caller's frame, so the registers
   cannot coincide, and the RFC says so rather than promising zero.

6. **At a Rust boundary the glue converts, and nothing is heap-
   allocated** (amended 2026-09-20: no `Context` type; projection types
   replace the stack temporary). **An aggregate crosses as a projection
   type the derive generates**: `#[derive(TyArg)]` on `struct S { a: A,
   b: B }` emits `SRef<'a> { a: A::Ref<'a>, b: B::Ref<'a> }` and
   `SMut<'a>`, and a handler that declares `&S` / `&mut S` receives them.
   `Cross` has `type Ref<'a>` and `type Mut<'a>`: `&'a T` / `&'a mut T`
   for a plain type (built by `Runtime::inline_mut` / `value_as_mut` on
   the field's `Value`), a view for a flat option (rule 9 — a Rust
   `Option<T>` has another layout), the derived projection for a nested
   aggregate (its sub-run, rule 8). Field offsets come from the settled
   shape at `prepare` and reach the glue as a table in the op; the glue
   builds the projection by `split_at_mut` over the run — **no name
   lookup at call time**; a partial projection borrows only the fields
   it names. An enum derives `ERef<'a>` / `EMut<'a>` — real Rust enums
   whose payloads are borrowed — from the tag and the payload's run;
   `EMut::set(E)` rewrites `[tag, payload…]` in place by `into_run`,
   the layout being the widest variant's. `Rt::Object<'a>` /
   `Rt::Enum<'a>` name the runtime's representation these are built
   from. By value: `from_run` / `into_run` (rule 6's ABI paragraph). An
   object by value into a Rust container: rule 4's realization, at the
   glue. The stack-temporary `&E` and the `&mut E` write-back are
   withdrawn (amended 2026-09-20). An extern that **returns** an object receives its
   destination as `Out<'_, Rt>` — a `&mut [Value]` over the caller's
   destination run, lent for the call's duration — and writes the
   components; it is not a heap object either. The frame owns the run;
   `Out` is a borrow of it, so two callers cannot hold one and no
   runtime-level `Context` type is needed to say so. **The frame a
   handler receives to call a closure is the window above the caller's
   `frame_len`, and the handle to it is one word**: `Runtime::Frame<'a>`
   is `&'a mut FrameState`, a state the frame below owns — a field of its
   `Machine`, or the `Store`'s — so the operation passes an address that
   already exists and builds nothing on its own stack. The window is
   disjoint by type from the caller's registers, its wide region and
   `Out` — so a handler holding `Out` may call back into the machine, and
   the callee's own region lives in its window. The async path's future
   owns its window the same way.

   **The ABI is the runtime's contract, not the macro's** (amended
   2026-09-20): it is an internal contract, so it is resolved in internal
   functions. Every fact about how a value lies in registers — how many
   slots a crossing takes, how
   a slice pair becomes `Elements`, how components are written to `Out`,
   how a result is read at the window's run — is a `Runtime` method or
   a `Cross` constant (`const WIDTH: usize`, `from_run`/`into_run`), and
   the macro emits only calls to them. Today the macro decides the ABI
   form by counting parameters (`SyncAbi::Arity1` at `extern-macro/src/
   lib.rs:555`) and detects a slice return by reading the type's last
   path segment (`returns_slice`, `lib.rs:801` — an alias defeats it);
   both go: the form is picked by the library from the sum of the
   parameters' `WIDTH`s, and a type says how it crosses. Multi-value
   return (rule 5) is the same contract from the machine's side:
   `Return` writes `WIDTH` values at the run, the caller reads `WIDTH`.

7. **A container's element is realized, and a container is never a
   component set.** `Vec`, arrays, deques: their elements are heap
   aggregates (rule 4), read by projection (`&v[i]` is a `LargeRef`
   into the element). Scalar replacement of a container itself is
   **refused** — an array or a `Vec` is not an object with positional
   fields, and treating it as one is a wrong implementation, not an
   optimization.

8. **An aggregate's layout is flat** (added 2026-09-20). The layout of
   an aggregate is the concatenation of its fields' layouts: a word or
   `Large` field is one `Value`, a nested aggregate field is its own
   layout inline, an enum is `[tag, payload layout]`, and `off(i)` is
   the prefix sum — fixed by the settled type (the union type, RFC-0041:
   one field order and one tag numbering per program), known to
   `prepare` and to the glue alike. `&line.a` is `base + off(a)`, a
   projection into the middle of a run. A heap realization (rule 4)
   holds the same flat `[Value; n]` behind its header, so a reader is
   the same either way. In registers (rule 1) the same flattening is
   what SROA does today one level down (`sroa.rs`: `[PathSeg::Field]`),
   widened to a path of any depth. A field the settled union type has
   and a construction lacks is `Undef` at its offset — whether the
   checker admits a read of it is the type system's question, listed
   under Order of work. For a declared struct's type it is answered: the
   checker refuses an object that lacks a field the struct declares
   (RFC-0042 R1), so no value of a declared type has an `Undef` field.
   For an object literal's type the question stays here.

9. **`Option<Aggregate>` is flat over the run**: the run's first
   `Value` is `Kind::None` for `None` and the payload's first component
   otherwise, the rest `Undef` — RFC-0039's rule (an option is its
   payload) at the width of the payload.

Rule 5 of RFC-0052 stands: a register holds one kind class. A payload
register whose variants disagree in class (`A(i64) | B(String)`) is
whole-typed and `Release` decides by kind; there is no conditional
`Drop`.

## What it costs

- The allocator gains run allocation over a per-body region with a
  spill order (loop depth, then live range); a frame's size is no
  longer one number per program but one per body, up to 4 KB more, and
  a frame with more than 64 slots carries more than one mark word — the
  sweep and `prepare`'s 64-register stop both change.
- Two forms of every aggregate read: register (component) and
  projection (`base + off(i)`). `prepare` picks once per variable (rule
  3); a `run` never tests which.
- `Machine::exit` and `Return` become multi-value at the window's run;
  every host that reads a returned aggregate reads components there.
- Every handler ABI form (`SyncAbi::Arity0..3/Window`, `StateAbi`, the
  slice and heavy forms) changes twice: the frame parameter narrows to
  the `Window`, and an aggregate-returning form gains `Out`.
- The extern crate gains a projection crossing (`Obj<V>`'s first `impl`
  block), an enum glue (`#[acvus::enum]`, new — `#[derive(TyArg)]` is
  its nearest form and refuses generics), a `&mut` write-back mechanism
  (new: today `Cross::deref_mut` on an aggregate panics), and `Out`.
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
  has. So no mark is added: the SSA, the references and `MakeVariant`
  already express it, and deriving it is `prepare`'s job.
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
- **A fixed 256-byte wide `Cell` cut into three 64-byte and two 32-byte
  slots** (the design as accepted on 2026-09-19): four fields per
  aggregate before a heap spill, and a nested aggregate (rule 8) fills
  it faster; the cost of the size itself — cache lines per call, the
  frame `Vec` growing — is why the region is sized per body and bounded
  at 4 KB rather than fixed larger (amended 2026-09-20).
- **A `Context` associated type on the runtime** (the design as
  accepted): the frame's whole wide `Cell` handed to a handler as
  `&mut`. The handler does not know which slot is its destination, so an
  offset travels beside it and the pair is `&mut [Value]` — `Out`; and a
  handler holding `&mut Context` while passing `&mut Store` to `call_now`
  (`runtime.rs:147`) is two `&mut` into one frame, which Rust refuses —
  the `Window` narrowing is what makes both borrows hold at once.
- **Return in the callee's own component registers**: unreadable by a
  caller through a closure value, whose body is not known to `prepare`.
- **A per-site choice of register or projection for one aggregate**:
  a write through a projection then leaves a register copy stale; the
  home is one per variable (rule 3).
- **Nested aggregates realized as the outer's `Large` fields**: the
  literal reading of rule 4 before rule 8; a `Line` of two `Point`s
  would box both on every `&line`.

## Consequences

Nothing here is built yet; the first three are expectations, each with the
count it rests on, and the measured table replaces them when the
implementation lands.

- **Expected** `enum match` at 6–9 ns, from 13.6: `MakeVariant` gone, a
  `Switch` on a tag register, the payload a phi. **Expected** `vec of
  objects` at 4–6 ns, from 9.6: field reads by projection into the element,
  no hash. `option match` holds no aggregate and is expected flat. The
  Brainfuck bench's `program[pc]` read becomes two loads through a
  projection, and its dispatch row becomes a `Switch`.
- Rule 6's window half is built, and the handle to it is one word. Against
  master `f3160466`, min of three alternating pinned reps: `extern while`
  3.7 → 3.5 ns, `option while` 5.4 → 5.3, `branch while` 5.0 → 4.9,
  `while let vec` 8.9 → 8.0, `map add | sum` 5.9 → 5.5, `map cap | sum`
  8.4 → 7.8, `attention` -0.9 to -1.5 %, `bf table` -0.6 %; `while let map`
  10.1 → 10.5 is the one regression, +5 instructions and +2.2 cycles per
  element on the `CallExtern1` that calls `next`, where the same operation
  without a closure is -3. The staging `[Value; n]` and the `fill` that
  copied parameters out of it are gone, which is 9 instructions per element
  off the closure path, and `CallExtern1::run` is 33 instructions against
  master's 36 with no stack frame at all. A window passed by value instead
  of a handle was measured first and refused: three words is past the two
  the SysV ABI hands an argument in, so every extern call stored it and
  passed that address, which cost 16 instructions, 3 cycles and the tail
  call of `CallExtern0..3`, `CallWindow` and `AsSlice`, and regressed
  `extern while` +16 %, `option while` +13 %, `while let map` +23 %.
  What the one word still costs is one register the handler never reads
  where it calls no closure; choosing the operation by whether the handler
  calls a closure at all, which `Glue` already knows, is what removes that.
- `&Obj` handlers become expressible; a Rust handler receives a real
  Rust enum.
- A frame overflow of either class is a `prepare` refusal naming the
  body, never a run-time panic.

## Order of work

No code before every site's form is written. The enumeration (94 rows)
is the checklist, re-read against rules 2–9 as amended on 2026-09-20;
this RFC's rules are the answers. Two questions are settled before the
first brief: whether the checker admits a read of a union-type field a
construction lacks (rule 8's `Undef`), and the `Window` narrowing of
every handler's frame parameter, which `callindirect-without-a-heap`
(running) sets the argument-run half of. Order: RFC-0051's `Switch`
operation (done, `347268d8`) → the wide region and `LargeRef` in the
machine (run allocation, the spill order, the mark words, the
projection read, the frame-owned sweep) → the escape predicate widened
to SSA values with `Return` and borrowed arguments removed, SROA over a
path of any depth (rule 8), and `prepare` emitting realization only at
the escape sites with rule 3's one home → multi-value return at the
window's run → the extern glue (`Window`, `Out`, `&[Value]` + shape,
`#[acvus::enum]`, `&mut` write-back) → the benches.
