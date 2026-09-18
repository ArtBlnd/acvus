# RFC-0047: a slice is the one thing the machine indexes

Status: Accepted — owner and coordinator, 2026-09-18
Extends: RFC-0018 (references), RFC-0028 (container signatures), RFC-0039
(one crossing), RFC-0043 (a bare name is settled by evidence), RFC-0044
(a body is prepared once), RFC-0007 (motion)

## Problem

Element access was an extern call. `*get(get(&keys, t), i)` is two calls of
`get`, each materializing a `Ref<Vec<T>>` and an `i64`, checking the bound,
building a `&T`, erasing it as a `Ref`, returning, and a `*` reading through
it. After the arithmetic chain, the diamond and the fused run (RFC-0044),
attention's inner iteration is two such runs and one chain; fusing the
dispatches out moved the clock 6 %, a removed dispatch there being worth
0.44 ns. What remains is `get`'s own work.

Two facts the compiler cannot use while the check is inside the extern:
`i < d` with `d = len(&query)` proves every `get(&query, i)` in bounds, and
`get(&keys, t)` does not change across the `i` loop. A call that may panic
is not hoisted (RFC-0007), and a check the compiler cannot see is not
eliminated.

The machine indexed one shape: a matched array at a constant position
(`InstKind::ArrayIndex`). Every container is otherwise a black box, by
intent — the interpreter does not know `Vec`'s layout, and must not.

## Decision

The machine indexes exactly one thing, a **slice** — Rust's `&[T]` /
`&mut [T]`: a pointer and a length. A container that can be indexed says so
by implementing **`core::as_slice`** (and `core::as_slice_mut`), a container
signature (RFC-0028); the compiler takes the slice with an **`AsSlice`**
instruction and indexes it with **`Index`** / **`IndexSet`**. No instruction
knows a container: `AsSlice` runs the container's own `as_slice` instance.

1. **One element width.** Every container the machine can slice stores
   `Vec<Value>`: the runtime's own `Array` is `Arr<Value, ()>`, and a
   `Vec<T>` reaches the store as `Vec<Value>` (RFC-0039). A converted
   container (`Vec<f64>` in a Rust signature) has no storage of its own and
   cannot be sliced. So a slice is always `&[Value]`, the width is always
   16, and the interpreter's one ABI fact is its own `Value`.
2. **Types.** `&[T]` and `&mut [T]` are reference types (RFC-0018): a slice
   borrows the container it was taken from, holds that loan, and is never
   stored beyond it. `&mut [T]` is an exclusive take; a live `&[T]` refuses
   `as_slice_mut` and `IndexSet` on its container.
3. **Signatures and `AsSlice`.** `core::as_slice<T>(c: &C) -> &[T]` and
   `core::as_slice_mut<T>(c: &mut C) -> &mut [T]` are bare names settled by
   the container's evidence (RFC-0043). `Vec<T>` and `Array<T, N>` implement
   them; `Deque<T>` (two halves) and `Map` do not. The lowering emits
   **`AsSlice { dst, container, mutability, instance }`**, not a
   `FunctionCall`: the instruction kind itself states what the hoist must
   know — a pure, **infallible** borrow projection of its container — so
   `code_motion` classifies it as a shared borrow structurally, as it does
   `Ref`, and lifts it out of every loop that does not write the container.
   The owner ruled (12:20) that the distinction stays in the instruction
   rather than in a declaration marker. In `prepare` an `AsSlice` is the one
   extern call of the resolved instance, fused like any other (RFC-0044).
4. **`Index { dst, slice, index, mode }`** reads element `index` of a slice.
   The index is **`u64`** and nothing else (owner, 11:50): the one check is
   `index < len`. `mode` is decided by the checker, statically, from the
   element type:
   - `Copy` — the element type is a word: `dst` is the element `Value`.
   - `Ref` — otherwise: `dst` is `Value::reference(&slice[index])`, a `Ref`
     into the slice's storage carrying the slice's loan, which is the same
     `Ref` `get` returned.

   `IndexSet { slice, index, value }` writes through `&mut [T]` with
   `assign` semantics (the old element drops). Both panic with Rust's text:
   `index out of bounds: the len is {len} but the index is {index}`.
   `IndexUnchecked` / `IndexSetUnchecked` are the same without the check;
   **only the compiler emits them, only where it has proved the bound**
   (§7). The user cannot ask for them.
5. **Syntax and place semantics.** `a[i]` is an index expression and a
   place, as in Rust. In value position its element type must be a word —
   otherwise the refusal is Rust's, ``cannot move out of index of
   `Vec<T>` ``, and the way is `clone(&a[i])` (RFC-0028). `&a[i]`,
   `&mut a[i]`, the inner level of `a[i][j]` and a method receiver's
   auto-borrow are `Index` in `Ref` mode; the outer level's `as_slice` takes
   that `Ref<Vec<T>>` as its `&C`. `a[i] = v` is `as_slice_mut` +
   `IndexSet`. A statement beginning with `[` is an array literal; a postfix
   `[` binds to the expression before it.
6. **Representation.** A slice is an extension type like any other:
   `Slice { ptr: *const Value, len: usize }`, crossing as a `Large`
   (RFC-0039), so `Value` does not change — its layout, its `ScalarPair` ABI
   and its `Option` niche are what they were. `Index` reads `word` to the
   `Slice`, compares `index < len`, and reads the element at
   `ptr + index * 16`: one dependent load more than a register word, no
   call, no layout. The one allocation is `AsSlice`'s, and `AsSlice` is
   hoisted, so it is paid once per loop entry (owner, 13:05).
7. **Bounds-check elimination is an interval domain and nothing more**
   (owner, 11:40). Each `Int` value carries `[lo, hi]` whose endpoints are
   constants or one other SSA value. Transfer: constants and `± constant`
   are interval arithmetic; φ is join with widening; everything else is ⊤.
   Refinement: on the true edge of `i < n`, `i.hi = n − 1`. An
   `Index(s, i)` becomes `IndexUnchecked` when `i.hi < n`,
   `s = as_slice(&c)`, `n = len(&c)`, and `c` is not written between the
   `len` and the `Index` (`Loans::storage_effect`, the hoist's own
   condition). `u64` has no lower bound to prove. `a[i + 1]` and an index
   derived elsewhere stay checked.
8. **Motion.** `AsSlice` hoists as a shared borrow of its container; a
   checked `Index` does not, because it may panic (RFC-0007); an
   `IndexUnchecked` whose slice and index are loop-invariant hoists as a
   pure instruction.

## The cut

Necessity by absence: each removal breaks the build, the compile errors
enumerate the dependents, nothing is patched around.

| removed | replaced by |
|---|---|
| `InstKind::ArrayGet` (variable index into a matched array) | deleted first (`2f3e923b`): nine consumers, no producer |
| `InstKind::ArrayIndex` | **stays**: it moves an element out of an owned scrutinee at a constant position — pattern destructuring, not indexing a borrowed container |
| `Vec::get`, `Vec::get_mut`, `Array::get`, `Array::get_mut` | `as_slice`/`as_slice_mut` + `Index`/`IndexSet`; deleted when the lowering emits the instructions, with no transition in which both exist |
| `first`, `last` on `Vec`/`Array` | stay as externs (`Option<&T>`); making them `Index` with a bound test is recorded, not decided |
| `len`, `is_empty` | stay as externs; a `Slice` knows its length, so `len(&v)` could become `as_slice` + `Len` — measure before adding, since the interval pass needs `len(&c)` as a value either way |
| the `*get(..)` read-through at every indexing site | gone for word elements: `Index` in `Copy` mode yields the value; in `Ref` mode the `Ref` is the one `get` returned |
| `Deque::get`, and any container without `as_slice` | unchanged: a call with its check inside, never hoisted — the cost of not being a slice, by the container's own choice. ``a[i]`` on it is refused at the checker |

## What it costs

- One `Large` allocation per `AsSlice` execution, and one dependent load per
  `Index` to reach `ptr` and `len`.
- `AsSlice`, and two instructions with checked and unchecked forms; the
  `Copy`/`Ref` modes are instances chosen in `prepare` from `val_types`,
  with no run-time branch.
- An interval pass, and the loan condition it borrows from the hoist.
- `Deque` and `Map` keep `get` and its call cost.
- The interpreter learns one Rust ABI fact — `&[Value]` is `(ptr, len)` and
  element `i` is at `ptr + i * 16` — and no container's layout.
- **A length is a `u64`.** §4 fixes the index at `u64` and §7 proves
  `i.hi < n` with `n = len(&c)`; with `len` returning `i64` the two can
  never meet and `while i < len(&v) { v[i] }` is a unification refusal. So
  `len` on `Vec`, `Array`, `Deque` and `String` returns `u64`, and
  `core::to_float` became a shared signature with an instance per integer
  type. `to_int` has no `u64` instance, so a length and an `i64` element
  cannot be added. This is a language change beyond the RFC as accepted, and
  it is the owner's to judge.
- **A slice never reaches user code.** `TypeEnv` holds two maps:
  `functions`, which `resolve_fn` reads and which is the only place a
  script's name can resolve, and `machine`, which holds the `as_slice` /
  `as_slice_mut` declarations and which `resolve_fn` does not read.
  `as_slice(&v)` is `undefined function` because the name is in no table the
  lookup searches, so the only `&[T]` in a body is an `AsSlice`'s `dst`,
  read by an `Index` beside it.

## Rejected

- **`core::index_k` externs with `Copy` return** (the first shape): no
  layout knowledge and one call per level, but the bound check stays inside
  the call — the compiler can neither eliminate it nor hoist the call, and
  the fusion stage measured that the remaining cost is the call's own work,
  not the dispatch.
- **`Index` reaching into `Vec`'s layout**: the interpreter does not know
  containers (owner).
- **Three element widths** (uniform 16 / inline kind / `Cross` struct): on
  the tree every sliceable store is `Vec<Value>` and a converted container
  has no storage, so the other two widths have no case.
- **`as_slice` as a plain `FunctionCall`, hoisted by a `total` declaration
  marker**: the hoist moves no call, so the call form needed a new marker
  and a hoist rule over every pure total call — a general mechanism for one
  fact the instruction kind already carries (owner, 12:20).
- **Folding `ArrayIndex` into `Index` as a `Move` mode**: `ArrayIndex` acts
  on an owned `Array`, not a slice; two representations in one instruction
  is a run-time branch, or a second instruction under one name.
- **Index type `i64`**: a negative check for nothing; `u64` is Rust's
  `usize` and the interval pass has one bound to prove.
- **Induction-variable or recurrence analysis for bounds**: the interval
  domain with one symbolic endpoint covers `while i < len { a[i] … i = i + 1 }`,
  which is the shape in every bench.
- **A fat-pointer `Value` (24 B)**: nothing widens.
- **A two-register slice** (`ptr` and `len` in two machine registers): it
  saves the load and keeps `Value` untouched, but a slice then cannot cross
  an extern boundary in either direction, so `as_slice` could not be an
  extern and `Slice` could not be a parameter.
- **The length inside `Value`'s head word.** Three forms, all measured. A
  `{ kind: u8, len: [u8; 7] }` head falls to memory class and returns
  through `sret` from every handler (attention +122 %); a
  `{ NonZeroU64, u64 }` head keeps the ABI but makes every whole-`Value`
  copy a 16-byte load over two 8-byte stores, which does not forward
  (`map cap | sum` +17 %). Retried on the RFC-0052 machine (2026-09-19, four
  runs) as `Value { head: u64, word: u64 }` with the kind in the head's low
  byte: the slice side works (attention −12 %, `IndexCopy`'s three dependent
  loads become one, the ABI class kept) and its price is the head's niche —
  `Option<Value>` grows to 24 bytes and returns through `sret` on the stage
  protocol, and the flat-option protocol that would replace it costs a
  `some`/`unwrap_some` per element (the `map` family +5–20 %). A slice's
  load is what hoisting removes; a `Value` layout is what every crossing
  pays.
- **The length as a third operand of `Index`**: every crossing of a slice —
  a call, a capture, a store — would carry the length beside it in the
  compiler's hands; the box carries it in one value.
- **Unchecked indexing as a language-level `unsafe`**: only the compiler's
  proof emits the unchecked form.

## Consequences

**The instruction is worth −7.77 ns per element pair, and the hoist
−41.5 ns.** The ceiling probe (`benches/slice_ceiling.rs`) runs
`acc += q[i] * k[i]` over two `Vec<f64>` of a million elements, four ways on
one binary, medians of six alternating reps; the dispatch counts are printed
by the probe, not assumed:

| shape | dispatches | ns / element pair |
|---|---|---|
| `get` | 5 | 16.78 |
| `as_slice` in the loop | 9 | 50.51 |
| `as_slice` hoisted | 5 | 9.02 |
| hoisted, unchecked | 5 | 8.63 |

`perf` puts 58 % of the `get` shape in `fused_call`, `fused::<1, true>` and
`vec::get`'s own closure — ≈5 ns per call, none of it dispatch: it is the
two `deref`s, the `Value::reference` build, the deref tail and the downcast
in `Ref::map`. Of the un-hoisted shape, 86 % is in erasing the `Slice` as a
`Large`, which is why §3 puts the hoist's condition in the instruction. The
check itself is worth −0.38 ns, inside a band of 0 to 0.5, so the interval
pass buys 4 % of this loop — it pays, and it is the smallest of the three
numbers by an order of magnitude.

**The compiler half regressed attention, and the cause is the out pass's
loop order.** 64×64 execute: 174.6 µs → 199.4 µs (**+14 %**); 256×128:
1352.0 → 1506.2 (**+11 %**). Medians of three alternating reps, two
`--profile bench` binaries; mandelbrot 17.1 → 17.0 ns/iteration and every
`accum` case within 3 %. `attention_loop_shape`'s four loops name it:
`head 1 body 4 back 0`, `head 1 body 12 back 0`, `head 1 body 9 back 0`,
`head 1 body 7 back 1` against the `get` shape's 4, 8, 5, 7. The scores pass
reads `@keys[t]` once per row into a binding, so its slices sit where §8
predicts; the out pass indexes `@values[t]` with `t` as the **inner**
variable, so the row's `AsSlice` is defined by the loop that would have to
hoist it, and nothing rises above its own operand. At the ceiling above, an
un-hoisted `AsSlice` costs (50.505 − 9.016) / 2 = 20.7 ns, and the out pass
runs n·d = 4096 of them at 64×64: about +85 µs against about −29 µs the
scores pass saves, which is the sign and the order of the measured +25 µs.

- attention's inner iteration is `AsSlice(query)` hoisted to the entry,
  `AsSlice(keys[t])` above the `i` loop, two `Index` and one chain per
  element — the Rust scalar shape — wherever the container is not the loop's
  own.
- **The unchecked form is the machine's, and the MIR does not yet hold it.**
  The index operation carries `CHECKED` as a type parameter, and the
  unchecked instance is reachable only by the ceiling probe; §7's interval
  pass, which is what would let the compiler emit it, is not built.
- `for x in &v` and `while let` over a slice can be a `Loop` over an
  unchecked index later: an iterator stage over a slice is `(ptr, len, i)`
  at `Task::Sync`. Not built.
- kovac inherits a static, layout-free indexing instruction with a static
  bound proof.
