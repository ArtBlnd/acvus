# RFC-0047: a slice is the one thing the machine indexes

Status: Draft (owner and coordinator; to be settled with RFC-0046's runtime half in the next session)
Date: 2026-09-18
Extends: RFC-0018 (references), RFC-0028 (container signatures), RFC-0044
(a body is prepared once), RFC-0007 (motion)

## Problem

Element access is an extern call. `@keys.get(t).get(i)` is two calls of
`get`, each materializing a `Ref<Vec<T>>` and an `i64`, checking the
bound, building a `&T`, erasing it as a `Ref`, returning, and a `*`
reading through it. After the chain, the diamond and the fused run,
attention's inner iteration is two such runs and one arithmetic chain;
fusion took the dispatches out and moved the clock 6 % — a removed
dispatch there is worth 0.44 ns. What remains of `get` is `get`.

Two facts the compiler cannot use while the check is inside the extern:
`i < d` with `d = len(query)` proves every `query.get(i)` in bounds, and
`@keys.get(t)` does not change across the `i` loop. Rust's LLVM elides
the first and hoists the second; ours can do neither, because a call
that may panic is not hoisted (RFC-0007) and a check it cannot see is not
eliminated.

The machine knows one indexed shape today: `Array<T, N>`, through
`InstKind::ArrayIndex`/`ArrayGet` over the runtime's own `Array`. Every
other container is a black box, by intent: the interpreter does not know
`Vec`'s layout, and must not.

## Decision

The machine indexes exactly one thing, a **slice** — Rust's `&[T]` /
`&mut [T]`: a pointer, a length, and an element width the static type
gives. Nothing else is indexed. A container that can be indexed says so
by implementing **`core::as_slice`** (and `core::as_slice_mut`); the
compiler introduces the slice with an **`AsSlice`** instruction and
indexes it with **`Index`**, which generalizes `ArrayIndex`/`ArrayGet`.

1. **Types.** `&[T]` and `&mut [T]` are reference types (RFC-0018): they
   borrow the container they were taken from, live as long as that
   borrow, and are never stored beyond it. The `Array<T, N>` value is the
   one container the machine holds itself; its `as_slice` is the
   identity on its storage.
2. **`core::as_slice(&C) -> &[T]`, `core::as_slice_mut(&mut C) -> &mut
   [T]`** are signatures (RFC-0028) a container implements in its extern
   crate — `Vec<T>` and `Array<T, N>` do, `Deque<T>` does not (two halves;
   it may offer `as_slices` later), `Map` does not. The layout stays with
   the implementer.
3. **`AsSlice { dst, container }`** is an intrinsic instruction: the
   lowering emits it where an index expression's container is not
   already a slice, and it prepares as a call of the container's
   `as_slice` instance — the one extern call, invariant in the
   container. Its result is a slice `Value`.
4. **`Index { dst, slice, index }`** replaces `ArrayIndex` (constant
   index) and `ArrayGet` (variable index): it reads element `index` of a
   slice. For an element type that is a word (the inline kinds), the
   result is the element **by value** (Copy — the owner's decision
   2026-09-18); otherwise a `&T` into the slice. `IndexSet { slice,
   index, value }` writes through `&mut [T]`. Both check `index <
   len` and panic with Rust's text (`index out of bounds: the len is {len}
   but the index is {index}`); `IndexUnchecked`/`IndexSetUnchecked` are
   the same instructions without the check, which **only the compiler
   emits, only where it has proved the bound** (below).
5. **Representation.** A slice `Value` is `{ head, word }`: the low byte
   of `head` is `Kind::Slice`, the high 56 bits are the length, `word`
   is the pointer. `Value` stays two `u64` scalars (a `ScalarPair`, 2a);
   the kind byte's seven padding bytes become the length. `Index` reads
   the length from `head >> 8`, the pointer from `word`, the width from
   the element kind — three fields, no call, no layout.

   The width is static because RFC-0039's `#` marker is in the type:
   `Vec<#T>` (uniform) gives `&[#T]` = `&[Value]`, width 16, the element a
   `Value` to define as is; `Vec<T>` specialized to an inline kind gives
   `&[T]` with the kind's width (`f64` → 8) and the element becomes
   `Value::inline(kind, bits)`; `Vec<T>` specialized to a Rust struct
   gives `&[T]` with the width the `Cross` instance states as a constant,
   and the element is a `&T`. `prepare` reads which of the three from
   `val_types` and picks the `Index` instance; no run-time branch on
   representation.
6. **Bounds-check elimination.** A range analysis over the MIR's loop
   structure (the natural loops `LoopDepth` already computes): an index
   that is a loop's induction variable stepping by 1 from a constant `a
   ≥ 0` under the condition `i < n`, where `n` is `len` of the same
   slice's container and the container is not written in the loop
   (`Loans::storage_effect`), is in bounds; the `Index` becomes
   `IndexUnchecked`. The proof is a MIR pass with the loan condition the
   hoist uses; a checked `Index` remains where nothing proves it.
7. **Motion.** `AsSlice` is a shared borrow of the container; it hoists
   under the borrow hoist's rule (`657545e3`) — out of every loop that
   does not write the container. An `IndexUnchecked` whose slice and
   index are loop-invariant hoists as a pure instruction; a checked
   `Index` does not (it may panic, RFC-0007). Writes: `IndexSet` and
   `as_slice_mut` are exclusive takes of the container, and RFC-0018's
   exclusion holds: a live `&[T]` refuses them.
8. **Syntax.** `a[i]` and `a[i][j]` are index expressions, lowered as
   `AsSlice` + `Index` per level (an inner `Index` yielding `&Vec<T>`
   feeds the next `AsSlice`). `get`/`get_mut` externs remain for a
   transition and then go: one way to index.

## The cut, and what replaces each thing

Necessity by absence: each removal breaks the build, and the compile
errors enumerate the dependents; nothing is patched around.

| removed | replaced by |
|---|---|
| `InstKind::ArrayIndex { array, index: usize }` (constant index into `Array<T, N>`) | `AsSlice` (identity on `Array`) + `Index` with a `Const` index; the constant fold of a constant index into an `Array` of known `N` is the range pass's trivial case → `IndexUnchecked` |
| `InstKind::ArrayGet { array, index: ValueId }` | `AsSlice` + `Index` |
| the interpreter's array element ops and `Composite::Array`'s index paths (`prepare.rs`, `ops/composite.rs`) | the slice `Index`/`IndexSet` instances over `Array`'s storage as a slice |
| `Vec::get`, `Vec::get_mut`, `Array::get`, `Array::get_mut` (acvus-ext) | `AsSlice`/`as_slice_mut` + `Index`/`IndexSet`; the externs are deleted once the lowering emits the instructions (a transition where both exist is a defect, not a stage) |
| `first`, `last` on `Vec`/`Array` | stay as externs for now (they return `Option<&T>`, which is a slice's `first()`); a later RFC may make them `Index` with a bound test — recorded, not decided |
| `len`, `is_empty` on `Vec`/`Array` | stay as externs; a slice knows its length (`head >> 8`), so `len(&v)` could become `AsSlice` + `Len` — one more instruction for one saved call; **measure before adding** (the range pass needs `len` as a value either way) |
| the `*get(..)` read-through (`Take { Through }` after a call) in every indexing site | gone for word elements: `Index` yields the value; for others the `&T` is the same `Ref` as before |
| `Deque::get` and any container without `as_slice` | unchanged: a call, with its check inside, never hoisted — the cost of not being a slice, by the container's own choice |

Every consumer the compiler names when `ArrayIndex`/`ArrayGet` go —
`const_dedup`, `ssa_pass`, `move_check`, `type_check`, `drop_insertion`,
`code_motion`, the printer, `lower.rs:1437`, and `prepare` — is rewritten
to `Index`; none keeps a dead arm.

## What it costs

- One more kind (`Slice`) and the `head` word's split into kind and
  length: every `Kind` reader masks the low byte. Measure it on the
  probes (`asm_probe.rs`): the mask is one `movzbl`, which most readers
  already do.
- Two instructions (`Index`, `IndexSet`) with checked and unchecked
  forms, and `AsSlice`; `ArrayIndex`/`ArrayGet` are deleted, their
  consumers become `Index` on `Array`'s identity slice.
- A range analysis pass, and the loan condition it borrows from the
  hoist.
- Containers that cannot give a slice (`Deque`, `Map`) keep `get` as an
  extern with the call cost; `Deque` may offer `as_slices` later.
- The interpreter learns one Rust ABI fact — a slice is `(ptr, len)` and
  element `i` is at `ptr + i * width` — and no container's layout.

## Rejected

- **`core::index_k` externs with `Copy` return** (the first shape): no
  layout knowledge, one call per level, but the bound check stays inside
  the call — the compiler can neither eliminate it nor hoist the call,
  and the fusion stage measured that the remaining cost is the call's
  own work, not the dispatch.
- **`Index` reaching into `Vec`'s layout**: refused by the owner — the
  interpreter does not know containers.
- **A fat-pointer `Value` (24 B) or a two-register slice**: the padding
  bytes hold the length; nothing widens.
- **Unchecked indexing as a language-level `unsafe`**: only the
  compiler's proof emits the unchecked form; the user cannot ask for it.

## Consequences

- attention's inner iteration: `AsSlice(query)` hoisted to the entry,
  `AsSlice(keys[t])` hoisted above the `i` loop, two `IndexUnchecked` by
  value and one chain per element — the Rust scalar shape. Measured
  after it lands.
- `for x in v` and `while let` over a slice can be a `Loop` over
  `IndexUnchecked` later (an iterator stage over a slice is `(ptr, len,
  i)`), which is the remaining half of the iteration idiom's cost.
- kovac inherits a static, layout-free indexing instruction with a
  static bound proof.
