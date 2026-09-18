# RFC-0047: a slice is the one thing the machine indexes

Status: Accepted — owner and coordinator, 2026-09-18 (Draft the same
morning; the decisions below were settled at 11:50)
Extends: RFC-0018 (references), RFC-0028 (container signatures), RFC-0039
(one crossing), RFC-0043 (a bare name is settled by evidence), RFC-0044
(a body is prepared once), RFC-0007 (motion)

## Problem

Element access is an extern call. `*get(get(&keys, t), i)` is two calls
of `get`, each materializing a `Ref<Vec<T>>` and an `i64`, checking the
bound, building a `&T`, erasing it as a `Ref`, returning, and a `*`
reading through it. After the arithmetic chain, the diamond and the
fused run (RFC-0044), attention's inner iteration is two such runs and
one chain; fusing the dispatches out moved the clock 6 % — a removed
dispatch there is worth 0.44 ns. What remains of `get` is `get`'s own
work.

Two facts the compiler cannot use while the check is inside the extern:
`i < d` with `d = len(&query)` proves every `get(&query, i)` in bounds,
and `get(&keys, t)` does not change across the `i` loop. Rust's LLVM
elides the first and hoists the second; ours can do neither, because a
call that may panic is not hoisted (RFC-0007) and a check it cannot see
is not eliminated.

The machine indexes one shape today: a matched array at a constant
position (`InstKind::ArrayIndex`, pattern destructuring). Every
container is otherwise a black box, by intent: the interpreter does not
know `Vec`'s layout, and must not.

## Decision

The machine indexes exactly one thing, a **slice** — Rust's `&[T]` /
`&mut [T]`: a pointer and a length. A container that can be indexed says
so by implementing **`core::as_slice`** (and `core::as_slice_mut`), a
container signature (RFC-0028); the compiler takes the slice with an
**`AsSlice`** instruction and indexes it with **`Index`** /
**`IndexSet`**. No instruction knows a container: `AsSlice` runs the
container's own `as_slice` instance.

1. **One element width.** Every container the machine can slice stores
   `Vec<Value>`: the runtime's own `Array` is `Arr<Value, ()>`, and a
   `Vec<T>` reaches the store as `Vec<Value>` (RFC-0039). A converted
   container (`Vec<f64>` in a Rust signature) has no storage of its own
   and cannot be sliced. So a slice is always `&[Value]`, the width is
   always 16, and the interpreter's one ABI fact is its own `Value`.
2. **Types.** `&[T]` and `&mut [T]` are reference types (RFC-0018): a
   slice borrows the container it was taken from, holds that loan, and
   is never stored beyond it. `&mut [T]` is an exclusive take; a live
   `&[T]` refuses `as_slice_mut` and `IndexSet` on its container.
3. **Signatures and `AsSlice`.** `core::as_slice<T>(c: &C) -> &[T]` and
   `core::as_slice_mut<T>(c: &mut C) -> &mut [T]` are bare names settled
   by the container's evidence (RFC-0043). `Vec<T>` and `Array<T, N>`
   implement them in acvus-ext; `Deque<T>` (two halves) and `Map` do not.
   The lowering emits **`AsSlice { dst, container, mutability, instance
   }`**, not a `FunctionCall`: the instruction kind itself states what
   the hoist must know — a pure, **infallible** borrow projection of its
   container — so `code_motion` classifies it `SharedBorrow { storage }`
   structurally, as it does `Ref`, and lifts it out of every loop that
   does not write the container. A `FunctionCall` is never hoisted (it
   may panic, RFC-0007), and the owner ruled (12:20) that the distinction
   stays in the instruction rather than in a declaration marker. In
   `prepare` an `AsSlice` is the one extern call of the resolved
   instance, fused like any other (RFC-0044).
4. **`Index { dst, slice, index, mode }`** reads element `index` of a
   slice. The index is **`u64`** and nothing else (owner, 11:50): the
   one check is `index < len`. `mode` is decided by the checker,
   statically, from the element type:
   - `Copy` — the element type is a word (`is_move_only` false): `dst` is
     the element `Value` itself.
   - `Ref` — otherwise: `dst` is `Value::reference(&slice[index])`, a
     `Ref` into the slice's storage carrying the slice's loan (the
     element is a `Value` in a `Vec<Value>`, so this is the same `Ref`
     `get` returned).
   `IndexSet { slice, index, value }` writes through `&mut [T]` with
   `assign` semantics (the old element drops). Both panic with Rust's
   text: `index out of bounds: the len is {len} but the index is {index}`.
   `IndexUnchecked` / `IndexSetUnchecked` are the same without the
   check; **only the compiler emits them, only where it has proved the
   bound** (§7). The user cannot ask for them.
5. **Syntax and place semantics.** `a[i]` is an index expression and a
   place, as in Rust. In value position its element type must be a
   word — otherwise the refusal is Rust's: `cannot move out of index of
   `Vec<T>``, and the way is `clone(&a[i])` (RFC-0028: a value of the
   element type is a clone). `&a[i]` and `&mut a[i]`, the inner level of
   `a[i][j]`, and a method receiver auto-borrow → `Index` in `Ref` mode;
   the outer level's `as_slice` takes that `Ref<Vec<T>>` as its `&C`.
   `a[i] = v` is `as_slice_mut` + `IndexSet`. A statement beginning with
   `[` is an array literal, as in Rust; a postfix `[` binds to the
   expression before it.
6. **Representation.** A slice `Value` is `{ head, word }` with `head:
   NonZeroU64` — the low byte is the `Kind` (no kind is zero), the high
   56 bits the length (zero for every non-slice value) — and `word` the
   pointer. `Value` stays two scalars: rustc keeps it a `ScalarPair`,
   returned in two registers from every handler, and `Option<Value>`
   stays 16 bytes through the zero niche. `Index` reads the length from
   `head >> 8`, the pointer from `word`, the element at `ptr + index *
   16` — three fields, no call, no layout. Length is 56 bits (owner,
   11:50).

   The first form, `head = { kind: u8, len: [u8; 7] }`, was measured
   (T1, 12:40) and refuted: a byte array is not a scalar, so the
   aggregate fell to memory class and every handler returned its
   `Value` through `sret` — attention +122 %, `while let` +125 %. The
   padding bytes are free in size, not in ABI; the head must be one
   scalar.
7. **Bounds-check elimination is an interval domain and nothing more**
   (owner, 11:40: induction variables and recurrences are far future).
   Each `Int` value carries `[lo, hi]` whose endpoints are constants or
   one other SSA value (symbolic). Transfer: constants and `± constant`
   are interval arithmetic; φ is join with widening; everything else is
   ⊤. Refinement: on the true edge of `i < n`, `i.hi = n − 1`
   (symbolic). An `Index(s, i)` becomes `IndexUnchecked` when `i.hi < n`,
   `s = as_slice(&c)`, `n = len(&c)`, and `c` is not written between the
   `len` and the `Index` (`Loans::storage_effect`, the hoist's own
   condition). `u64` has no lower bound to prove. `a[i + 1]` and an index
   derived elsewhere stay checked; that is the correct answer, not a gap.
8. **Motion.** `AsSlice` hoists as a shared borrow of its container; a checked `Index`
   does not (it may panic, RFC-0007); an `IndexUnchecked` whose slice and
   index are loop-invariant hoists as a pure instruction.

## The cut, and what replaces each thing

Necessity by absence: each removal breaks the build, the compile errors
enumerate the dependents, nothing is patched around.

| removed | replaced by |
|---|---|
| `InstKind::ArrayGet` (variable index into a matched array) | deleted first (`2f3e923b`): it had nine consumers and no producer |
| `InstKind::ArrayIndex` | **stays**: it moves an element out of an owned scrutinee at a constant position — pattern destructuring, not indexing a borrowed container. It exists without `get`, so it is not taint |
| `Vec::get`, `Vec::get_mut`, `Array::get`, `Array::get_mut` (acvus-ext) | `as_slice`/`as_slice_mut` + `Index`/`IndexSet`; deleted when the lowering emits the instructions (a transition where both exist is a defect, not a stage) |
| `first`, `last` on `Vec`/`Array` | stay as externs (`Option<&T>`); a later RFC may make them `Index` with a bound test — recorded, not decided |
| `len`, `is_empty` on `Vec`/`Array` | stay as externs; a slice knows its length (`head >> 8`), so `len(&v)` could become `as_slice` + `Len` — **measure before adding** (the interval pass needs `len(&c)` as a value either way) |
| the `*get(..)` read-through (`Take { Through }` after a call) at every indexing site | gone for word elements: `Index` in `Copy` mode yields the value; in `Ref` mode the `Ref` is the one `get` returned |
| `Deque::get` and any container without `as_slice` | unchanged: a call with its check inside, never hoisted — the cost of not being a slice, by the container's own choice; `a[i]` on it is refused at the checker (`cannot index into a value of type `Deque<T>``) |

## What it costs

- `Kind::Slice`, and `kind: Kind` becoming the low byte of a `NonZeroU64`
  head: every `Kind` reader masks the low byte (`movzbl`, which most
  already do), and `Kind` starts at 1. Measured on `asm_probe` and the
  benches before anything else is built on it.
- `AsSlice`, and two instructions with checked and unchecked forms; the `Copy`/`Ref`
  modes are instances chosen in `prepare` from `val_types`, no run-time
  branch.
- An interval pass, and the loan condition it borrows from the hoist.
- `Deque` and `Map` keep `get` with the call cost.
- The interpreter learns one Rust ABI fact — `&[Value]` is `(ptr, len)`
  and element `i` is at `ptr + i * 16` — and no container's layout.

## Rejected

- **`core::index_k` externs with `Copy` return** (the first shape): no
  layout knowledge, one call per level, but the bound check stays inside
  the call — the compiler can neither eliminate it nor hoist the call,
  and the fusion stage measured that the remaining cost is the call's
  own work, not the dispatch.
- **`Index` reaching into `Vec`'s layout**: the interpreter does not know
  containers (owner).
- **Three element widths (uniform 16 / inline kind / `Cross` struct)**,
  the Draft's §5: on the tree every sliceable store is `Vec<Value>` and a
  converted container has no storage; the other two widths had no case.
- **`as_slice` as a plain `FunctionCall`, hoisted by a `total`
  declaration marker** (the coordinator's 11:50 proposal): the hoist
  moves only `Ref` today, never a call, so the call form needed a new
  marker (`#[extern_fn(effect = pure, total)]`) and a hoist rule over
  every pure total call. A general mechanism for one fact the instruction
  kind already carries; the owner kept the distinction in the
  instruction (12:20).
- **Folding `ArrayIndex` into `Index` as a `Move` mode**: `ArrayIndex`
  acts on an owned `Array`, not a slice; two representations in one
  instruction is a run-time branch or a second instruction under one
  name.
- **Index type `i64`**: a negative check for nothing; `u64` is Rust's
  `usize` and the interval pass has one bound to prove.
- **Induction-variable / recurrence analysis for bounds**: far future;
  the interval domain with one symbolic endpoint covers `while i < len
  { a[i] … i = i + 1 }`, which is the shape in every bench.
- **A fat-pointer `Value` (24 B) or a two-register slice**: the head
  word holds the length; nothing widens.
- **The length as a third operand of `Index`** (proposed when the
  byte-array head was refuted): keeps `Value` untouched, but every
  crossing of a slice — a call, a capture, a store — must carry the
  length beside it in the compiler's hands, where the value carries it
  for free in its head word.
- **A `Head { kind: u8, len: [u8; 7] }` struct**: measured, memory
  class; see §6.
- **Unchecked indexing as a language-level `unsafe`**: only the
  compiler's proof emits the unchecked form.

## Consequences

- attention's inner iteration: `AsSlice(query)` hoisted to the entry,
  `AsSlice(keys[t])` above the `i` loop, two `Index` (`Copy`, unchecked
  after §7) and one chain per element — the Rust scalar shape. Measured
  after each half lands, bands from dispatch counts.
- `for x in &v` and `while let` over a slice can be a `Loop` over
  `IndexUnchecked` later (an iterator stage over a slice is `(ptr, len,
  i)`, `Task::Sync`) — the remaining half of the iteration idiom's cost,
  after RFC-0046's runtime half (`b2b04b57`).
- kovac inherits a static, layout-free indexing instruction with a
  static bound proof.

## Order of work

T0 `ArrayGet` deleted (done). T1 interpreter + extern: `head` split,
`Kind::Slice`, the slice's `Cross`, `AsSlice` prepared as the instance's
call and hoisted as a borrow, `Index`/`IndexSet` handlers with the
probe-only unchecked forms, `as_slice`/`as_slice_mut` on `Vec`/`Array`,
and the checked-vs-unchecked ceiling measured. T2 compiler: types,
signatures, `a[i]` grammar and place lowering, loans, `get`/`get_mut`
removed. T3 the interval pass — only if T1's ceiling pays. T1 and T2 take
disjoint crates and run in parallel; T3 after both.
