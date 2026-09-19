# RFC-0059: the macro emits only calls — the runtime owns the ABI

Status: Proposed — 2026-09-19; rule 4 amended 2026-09-20
Extends: RFC-0039 (one crossing), RFC-0044 stage 2c (the by-value cut at
three), RFC-0046 (a call's task), RFC-0047 amended (a slice is two
registers), RFC-0050 rule 6 (a crossing's width, `from_run`/`into_run`),
RFC-0052 §6–§7 (the operation holds the callee, the arguments are the
callee's first registers)

## Problem

`acvus-extern-macro` decided facts that are the runtime's, by counting
tokens. Against the tree at `e225a350` the macro decided: the
register-passing cut (`Arity{n}` for n ≤ 3 else `Window`, from
`params.len()`); that a return was a slice, by the type's last path segment;
that the specialization casts were `SyncAbi::Arity1` handlers; that an object
is an `FxHashMap<Astr, Value>` keyed by `rt.symbol(name)`; that an enum is a
`Variant { tag, payload }` matched by a symbol-compare chain ending in
`panic!`; that `&Option<T>` is refused, by the type's *name*; the
`repr(transparent)` pointer cast of an extension type; that state is an
`Arc<dyn Any + Send + Sync>`, downcast with an `expect` on **every call**;
and fourteen handler forms as `fn` pointers, one per statefulness × arity.

Two costs followed. A layout change — RFC-0050's flat object — had to be
made in a proc macro rather than in the library. And every stateful call paid
an `Any` downcast and an `expect` for a value whose type was known when the
registry was built.

## Decision

Seven rules. The proc macro's whole output for a handler becomes one call to
a library constructor, and every ABI fact it used to spell out is a constant
or an associated type of the crossing.

1. **A crossing is the run of values it occupies.** `Cross<Rt>` carries
   `type Form: Form`, `unsafe fn from_run(rt, run) -> Self` and
   `fn into_run(self, rt, out)`. `Form` is the width *as a type* — `One` or
   `Pair` — and `Form::WIDTH` is the only place a width is a number:
   `Cross::WIDTH` and `Ret::WIDTH` are defaults that read it, and no impl
   writes either. `Slice<T, Rt>` and `SliceMut<T, Rt>` are `Form = Pair` and
   cross through two `Runtime` methods — `slice_into_run(words, out)` and
   `slice_from_run(run) -> Words` — because only a runtime knows what one of
   its values is made of. A declaration's slot count is the sum of its
   parameters' widths, computed in the library from the types; no token is
   read.

2. **Two crossings, not one, and no blanket.** `Cross<Rt>` is the run.
   `OneValue<Rt>: Cross<Rt>` is the crossing that is *one of the runtime's
   values*: `erase`, `materialize`, `deref`, `deref_mut`, `STORED_AS_VALUE`.
   Every bound that needs a value says `OneValue` — a parameter by borrow,
   an object's field, a container's element, a closure carrier's argument
   and result — and `Slice`/`SliceMut` implement `Cross` alone, so a slice
   in any of those positions is a compile error with the sentence
   `OneValue`'s `#[diagnostic::on_unimplemented]` carries. A parameter **by
   value** is built by its crossing at that crossing's width (`Cross`,
   `from_run`), which is what admits `Slice<T, Rt>` as a parameter
   (RFC-0047 rule 6); a separate `Arg` impl for `ByValue<Slice>` beside the
   `OneValue` one is refused by coherence (a downstream crate may implement
   `OneValue` for `Slice`), so the one bound is `Cross`. There is no
   blanket `impl<T: OneValue> Cross for T`: coherence cannot admit one beside
   `impl Cross for Slice`, so each one-value type states both impls, the
   `Cross` half through one macro (`cross_one_value!`) or, inside the proc
   macro, the same two forwards. The bodies of those forwards live once, in
   `one_from_run` and `one_into_run`. `CrossSpecialized<Rt>` is not split: a
   `Monomorphize` member is one of the runtime's values at every impl, so its
   run is its value and its width is 1 where the two `Arg`/`Ret` impls read
   it.

3. **Argument modes are one trait.** `Arg<'a, Rt>` carries `type Out`,
   `const WIDTH` and `unsafe fn take(rt, run) -> Out`, implemented for three
   marker types the macro names from the Rust parameter's mode: `ByValue<T,
   C>` (the crossing), `ByRef<T, C>` (`deref`), `ByRefMut<T, C>`
   (`deref_mut`), where `C` is `Uniform` or `Specialized`. The by-value mode
   requires `Cross<Rt>` (rule 2), the borrow modes `Borrowable<Rt>`, which carries
   `#[diagnostic::on_unimplemented]`: the refusal of `&Option<T>` and of a
   Rust `&[T]` is a trait error, not a name check.

4. **A handler is the operation's type parameter.** Amended 2026-09-20; the
   first build of this rule put the handler behind a `dyn` in the operation
   and is recorded under Rejected. Two traits carry a declaration now. The
   registry's is object-safe: `HandlerFactory<Rt>` answers `width() ->
   Width { args, ret }`, `clone_box()`, `into_op(self: Box<Self>, shape:
   Rt::CallShape) -> Rt::Op` and `into_fused(self: Box<Self>, shape:
   Rt::FusedShape) -> Rt::FusedCall`; `ExternHandler`'s three task variants
   hold factories and `prepare` clones one out of the module table per call
   site. The operation's is not object-safe: `Handler<Rt>` carries
   `const WIDTH: Width` and the calls — `call(rt, frame, run, out)`, the
   register forms `call0`..`call3`, `call_run` over a window, and
   `call_slice(rt, a) -> Elements<Rt>` for the one shape RFC-0047 §3 admits
   for a slice result. A call site's operation holds the handler **by value
   under its own type parameter** — `CallExtern1<H, LARGE, WORD>` — so
   `H::call1` is a static call and its body is what the operation runs. One
   `dyn` is left on the path and it is taken at preparation, in `into_op`.

   Which forms a handler's operations exist at is a fact of its type, not a
   test at preparation: `Runtime` carries one entry per form —
   `op_no_argument`, `op_one_argument`, `op_two_arguments`,
   `op_three_arguments`, `op_wide`, `op_slice`, and `fused_no_argument`
   through `fused_two_arguments` — and the arity of the glue names its entry
   where the glue is written. The one form a type cannot name outright is
   arity 1, whose result decides between the register and the slice form, and
   `AtArity1`, implemented for `One` and `Pair`, is that choice as a type.
   A handler therefore instantiates its own form, the window form, and the
   two thread-crossing forms, and no others.

   A call that crosses a thread keeps the `dyn`: `CallHeavy` and
   `SpawnExternSync` hold an `Arc<dyn SentCall>` and `CallExternAsync` and
   `SpawnExternAsync` an `Arc<dyn SentAsync>`, because such a call is sent to
   a pool rather than run in the caller's frame, and the send — not the call
   — is what its cost is. A fused run keeps the `dyn` over its nodes rather
   than over its handlers: the calls of a run reach different declarations,
   so `Call` is a `Box<dyn Invoke>` whose node holds one handler as its type
   parameter (RFC-0044 stage 6).

   The library implements `Handler` and `HandlerFactory` for closures
   generically, one impl pair per arity through a `macro_rules!` over tuples
   of `Arg` markers; the `Width` sum is written once, in that macro's body,
   and every arity but one binds `R: Ret<Rt, Form = One>`, so a slice
   returned at any other arity is a compile error. `ExternHandler::heavy` and
   `awaited` take the handlers whose result is one value, because a call the
   caller waits for outlives the frame its arguments were lent from.
   `#[state]` is a capture: the macro emits `move |rt, a, b| f(rt, &state.0,
   a, b)` over an `Arc<(T0, …)>` — typed, never `Any`, never downcast.

   A host that has no registers to lay a call in implements the twelve
   entries with one macro, `direct_call_forms!`, and gets `DirectOp`: the
   handler behind a closure that takes the argument run as it comes.

5. **Object and enum glue are library functions.** `#[derive(TyArg)]` calls
   `acvus_extern::object::{Building, Opened}` for a struct and
   `acvus_extern::variant::{erase, opened}` for an enum. The layout —
   `Obj<Owned<R>>`'s map, `Variant`'s boxed payload — lives in those two
   files alone, so RFC-0050's flat layout changes them and not the macro. The
   symbol-compare chain becomes one position lookup and the `panic!` is the
   library's, stated once. The `repr(transparent)` cast becomes
   `acvus_extern::transparent::{erase, materialize, deref, deref_mut}`,
   guarded by `unsafe trait Transparent<P>` which the derive implements only
   for a `#[repr(transparent)]` struct, and whose `SAME_SIZE` constant every
   one of those functions evaluates. The specialization casts are `Glue`s
   like every other handler: a `glue1` whose argument marker is
   `ByValue<T, Specialized>` and whose result marker is `Val<T, Uniform>`, or
   the reverse; the body is the identity and the two markers are the whole
   conversion.

6. **Async stays boxed.** One `Pin<Box<dyn Future>>` per call, from
   `AsyncCall::call`, which owns the runtime and a copy of the argument run
   so the future outlives the frame. Storing that future where it lies needs
   `size_of` of the handler's own future type, and an `async` block's type
   cannot be named in an associated type without
   `impl_trait_in_assoc_type` — unstable, rust-lang/rust#63063 — on the
   toolchain this repository pins (1.97.1 stable). Until that feature lands
   the box is the shape, and `Pending` holds a `BoxFuture<'static, Value>`
   as it did.

7. **The interpreter reads the object.** `prepare`'s extern arm asks
   `HandlerFactory::width()` for the form and the slot count, builds the
   `CallShape` that form names, and hands it to `into_op`; the operation the
   factory returns calls `call1`/`call2`/`call3`/`call_run` statically. `Slice` is no longer an `ExternHandler` variant: a
   slice-returning handler is a `Glue` whose result is `Pair`, and `AsSlice`
   takes the `Elements` back in the register pair `call_slice` returns it in
   and stores the two words.

## What it costs

- **One operation instance per handler and form.** The `accum` bench binary
  grows from 17 506 680 to 20 521 808 bytes, 17.2 %, and the `asm_probe`
  binary instantiates 2 814 `Op::run` symbols against base's 1 970. What the
  instances buy is the call: `CallExtern1<Glue<id_of>, false, false>::run` is
  15 instructions and holds no `call` at all, against the boxed handler's 36
  instructions and its `call *0x38(%r8)` through the vtable.
- **An operation now holds its handler's stack locals.** A handler is called
  from exactly one operation after monomorphization, so LLVM inlines its body
  there whatever the inline hints say, and a body that takes the address of a
  local across a callee takes the operation's tail call with it. Eleven of the
  2 814 instances are in that state — `iterator::next`, `max_by_key`,
  `min_by_key` and their neighbours, in `CallExtern1`, `CallExtern2` and
  `CallWindow` — where at `e4b67e70` the address stayed inside
  `Glue::call` and every extern operation tail-jumped.
- Nine `macro_rules!` tuple impls (arities 0–8) for `Handler` and
  `AsyncHandler`, and nine constructors: a closure is inferred higher-ranked
  only where the bound is in scope at its own site, so `Glue` has no `new`.
- One `Cross` impl per one-value type beside its `OneValue` impl, because the
  blanket is not available. Fifteen impls, each two forwarding lines from one
  macro, and the bodies live once.
- `Width` read once at preparation instead of an enum destructured there,
  and one box allocated per call site at preparation instead of an `Arc`
  refcount bump.
- `call_slice`'s default body holds the two-value run as its own local, so
  `AsSlice` holds none: `asm_probe`'s list of families that hold a stack
  address is the eight it was at `d88f27be`.

## Rejected

- **`Box<dyn Handler>` in the operation**, this RFC's own first build. The
  operation loaded the box's data pointer and the vtable pointer and issued
  one indirect call, and the handler's Rust body was a function the operation
  could not inline. The type parameter removes both: `extern while` is
  2.9 ns per iteration against 3.6, `branch while` 3.1 against 4.2, and
  `option while` 4.3 against 5.3.
- **Every form instantiated for every handler.** The first build of the
  amendment let `prepare` hand any shape to any handler, so each handler
  emitted all twelve operations and the nine its width does not name were
  dead code that LLVM compiled to `slice_index_fail`. It cost 222 dead
  `Op::run` symbols, 3.9 MB of the `accum` binary, and 43 operations that
  held no tail call because their only path was a panic. The forms as types
  (rule 4) make the dead instance unwritable.
- **`fn` pointers with `Arc<dyn Any>` state**, the shape being replaced. A
  `fn` pointer cannot close over a typed state, so the state had to be erased
  and re-checked on every call; and fourteen forms existed only because the
  cross product of statefulness and arity had to be enumerated by hand. The
  typed capture removes seven of the fourteen outright, and `logs sync ext`'s
  handler body lost a `type_id` call through a vtable and a 16-byte `TypeId`
  compare with it.
- **`Arc<dyn Handler>` in the operation**, the first build's shape. An
  `Arc`'s unsized payload sits behind a header of no static offset, so the
  call read the vtable's alignment field and computed
  `(data + ((align - 1) & ~15)) + 16` before jumping: two loads and four
  instructions per call, measured in `CallExtern1::run`'s disassembly. The
  box removes them.
- **Token detection** — `returns_slice`, `names_option`,
  `by_value_variant` — because each answers a question the type system
  already answers, and each answers it wrongly for an alias or a type
  parameter. `returns_slice`'s own doc said so.
- **A panicking one-value crossing for a slice** (`NO_ONE_VALUE`, the first
  build's `Slice::erase` and `Slice::materialize`). A slice has no one-value
  crossing, and the honest form of "has none" is a missing impl, not a body
  that panics: rule 2's split makes the same mistake a compile error carrying
  the same sentence. The first build could not have it because it kept one
  trait and a blanket impl; with no blanket, coherence has nothing to refuse.
- **`AsSlice` writing the pair into a local across the `dyn`.** The local's
  address escaped into `Handler::call`, so the operation could not tail-call
  its successor and the family joined `asm_probe`'s exception list.
  `call_slice` returns `Elements` by value — two registers, RFC-0047's own
  ABI — and the list is eight families again.
- **The future in the run** (RFC-0050): it needs rule 2 there first, and this
  run's cut is already the whole ABI.

## Consequences

- The inline future waits for `impl_trait_in_assoc_type`. Rule 4's amendment
  was drafted with `CallExternAsync<H, LARGE, INLINE>` choosing between a
  future stored in the operation and a boxed one by `size_of::<H::Fut>()` at
  monomorphization. `H::Fut` cannot be written on a stable toolchain: the
  future of an `async` block has no nameable type, and an associated type
  cannot be `impl Future` (rust-lang/rust#63063). Without the name there is
  no size, so there is no table of the 24 `async fn` externs' future sizes
  and no `FUTURE_INLINE`. The async path keeps rule 6's box and keeps its
  `dyn`: monomorphizing a call whose cost is an allocation and a suspension
  buys nothing and costs instances.
- `asm_probe`'s list of families that hold a stack address is the eight it
  was, and the instances it counts are 19, as at `e4b67e70`; the operations
  that tail-call their successor are 2 769 against 1 925, one per handler and
  form. Beside them stand eleven operations in `CallExtern1`, `CallExtern2`
  and `CallWindow` whose inlined handler holds a stack address, which no list
  entry covers. Whether those three families join the list, or a handler that
  materializes a large argument keeps a `dyn`, is not settled here.
- An operation's name in a listing now carries its handler's type, closure
  span included: `oplist` and the tests that read op names match a family by
  prefix where they matched it by equality.
- A slice is a result and never a parameter: `ByValue<Slice<T, Rt>>` does not
  implement `Arg`, and a declaration that takes one is refused with
  `OneValue`'s message. A slice *parameter* needs a `Form = Pair` `Arg` impl
  and a coercion in the checker — the next run, not settled here.
- RFC-0050's flat layout changes `acvus-extern/src/object.rs` and
  `variant.rs` only. RFC-0050's wide argument will add a `Form` beside `One`
  and `Pair`, and the window form (`Handler::call` on `&mut [Value]`) is
  already the shape it needs; nothing but the window form reaches it today.
- `Borrowable<Rt>` admits exactly the set today's `names_option` admitted: a
  `#[derive(TyArg)]` struct is admitted, because `decl.rs`'s `eq_point`
  declares `&Point` as an instance of the shared signature
  `eq<T>(a: &T, b: &T)` and a test asserts that signature's bound is
  `OneOf([I64, Point])`. Whether a converted struct may be borrowed at all is
  not settled here; today it compiles and would
  panic if called, and that is unchanged.
- The frame's mark word is read-modify-written once per operation that takes
  registers, and on Zen 5 that RMW is where every blocked store-to-load
  forward in the extern-call operation lands, in this build and at
  `d88f27be` alike. Making the number of blocked forwards a property of the
  layout rather than of the addresses — a mark word of its own line, or a
  mask written rather than and-ed — is a cut in `Regs`, not in the ABI, and
  it is named here because this build's byte growth in the operation is what
  exposed it.
- `Callable::call_on` and `Runtime::call_now` still take `&mut [Value]`.
  Moving them onto the window would change every `Fn0`/`Fn1`/`Fn2`/`Fn3`
  carrier, which is a second cut; it is not started and nothing stands in for
  it.

## Measured

Base `d88f27be`, three alternating pinned reps (`taskset -c 4`), bench
profile on both sides, medians in µs at n = 1e6 unless the row says
otherwise:

| case | base | new | Δ |
|---|---|---|---|
| `accum while let vec` | 7416.1 | 9042.6 | **+21.9 %** |
| `accum map add \| sum` | 5527.8 | 5933.0 | **+7.3 %** |
| `logs sync ext` (1e4) | 521.2 | 567.8 | +8.9 % |
| `logs sync ext` (1e5) | 6390.3 | 6108.0 | −4.4 % |
| `logs heavy pure` (1e5) | 282999.1 | 293810.0 | +3.8 % |
| `logs heavy opq` (1e4) | 27659.2 | 28851.7 | +4.3 % |
| `logs inline` / `closure` (1e4, 1e5) | — | — | ±0.2 % |
| `accum extern while` | 3649.3 | 3722.5 | +2.0 % |
| `accum branch while` | 4951.1 | 5095.6 | +2.9 % |
| `accum option while` | 5509.9 | 5515.4 | +0.1 % |
| `accum call while` | 6741.5 | 6956.2 | +3.2 % |
| `accum while let map` | 10504.7 | 10163.0 | −3.3 % |
| `accum grade while` | 10717.0 | 10492.0 | −2.1 % |
| `accum map id \| sum` | 3194.1 | 3151.8 | −1.3 % |
| `accum map cap \| sum` | 8469.6 | 8381.3 | −1.0 % |
| `accum int while` / `float while` / `range \| sum` / `collatz while` | — | — | ±0.8 % |
| `shapes field read` / `write` / `construct` | 2924.1 | 2751.1 | −6.0 % |
| `shapes enum match` / `option match` / `vec of objects` | — | — | ±1.8 % |
| `mandelbrot 200x100x200` | 8345.9 | 8453.7 | +1.3 % |
| `programs bf table` / `scan` / `call` (1e6, 5e6) | — | — | ±1.6 % |
| `attention` (deque, vec × 2x2, 64x64, 256x128) | — | — | ±3.3 % |
| `slice_ceiling` (three shapes) | — | — | ±0.5 % |

`CallExtern1<false, false>::run`, the operation `while let vec` runs, is 36
instructions against base's 33. Between the argument read and the call:

```
mov    (%r14),%rdi        # the box's data pointer — the handler
mov    0x8(%r14),%r8      # the vtable pointer
mov    0x38(%rbx),%rsi    # m.rt
mov    %eax,%edx          # the argument's tag, moved off %rsi
call   *0x38(%r8)         # the vtable's call1 slot
```

where base issued `mov 0x38(%r14),%rdi` and `call *(%r15)`. The three extra
instructions are the vtable load and two register moves: `&self` occupies
`%rdi`, so the runtime and the argument's two halves shift one register each.
The handler side is cheaper: `Glue<next>::call1` is 19 instructions on the
taken path against the `fn` shim's 20, and `logs sync ext`'s `Glue::call1` is
83 where base's `fn` was 99 — base's `type_id` call through a vtable and its
16-byte `TypeId` compare are gone, and the inlined `glob` loop is
instruction-for-instruction the same in both.

**The object's three instructions cost under 1.4 % of cycles.** The cases
that make one extern call per iteration pay the instructions and not the
time: `extern while` +4.23 % instructions for +1.35 % cycles, `branch while`
+3.08 % for +0.17 %, `option while` +3.44 % for +0.65 %, `while let map`
+0.81 % for −2.51 % (`perf stat`, `taskset -c 15`, n = 1e6, both sides built
with `-C llvm-args=-align-all-functions=6`). A `fn` pointer resolved at
preparation and stored beside the box was built and measured: it brings the
operation back to 34 instructions with a single-level call target and changes
`while let vec` by nothing — 56.90 M cycles against the box's 56.83 M. The
`dyn` is not what the two outliers measure.

**`while let vec` is the frame's mark word, not the call.** Its +9.2 cycles
per iteration survive forced 64-byte function alignment (+19.3 %) and forced
block alignment on top of it (+18.9 %), and survive shifting the heap by
16…4080 bytes. What moves with it is `ls_bad_status2.stli_other` — a load a
store cannot forward to — which goes from 1.164 M to 2.224 M per 1e6
iterations, one more blocked forward per iteration, at Zen 5's ~8.7 cycles
each. Sampled on that event, 100 % of the blocked forwards inside the
operation land on one instruction, on **both** sides: the `and` that
`Regs::take_mask` does read-modify-write on the mark word. Base pays 0.9 of
them per iteration and this build pays 1.8; the operation grew from 40 to 48
bytes, so the register file, the op nodes and the mark word sit at different
addresses, and the mark word's RMW now aliases a hot store twice as often.
The fix is the mark word's placement in `Regs`, which is a cut of its own and
is not started here.

**`map add | sum` is where the linker put the code.** Every function on its
per-element path is instruction-for-instruction identical between base and
this build, addresses aside: `machine::chain_value` 182 instructions,
`Expr::call_on` 30, `Map::next` 56, `expect_type::<i64>` 122,
`Generate<range>::next` 16, `chain::eval1` identical outright. The measured
instruction delta is +0.01 % and the cycle delta +5.5 %, and it survives
function alignment, block alignment and the heap shift. Its two siblings run
the same code over the same stages and do not move: `map id | sum` −0.9 %,
`map cap | sum` −0.2 %. `logs sync ext` is the same finding with its sign
reversed between two sizes, and an earlier build of this tree one
prepare-time assertion apart — a `Width` equality that no call executes —
measured it at +12.5 % and +25.6 % instead of +8.9 % and −4.4 %.

## Measured — rule 4 amended

Base `e4b67e70`, three alternating pinned reps (`taskset -c 4`), bench
profile on both sides, medians. `execute/us` unless the row says otherwise.

| case | base | new | Δ |
|---|---|---|---|
| `accum extern while` (1e6) | 3551.2 | 2942.6 | **−17.1 %** |
| `accum branch while` (1e6) | 4173.6 | 3051.4 | **−26.9 %** |
| `accum option while` (1e6) | 5325.2 | 4333.7 | **−18.6 %** |
| `accum while let vec` (1e6) | 8166.0 | 7657.5 | −6.2 % |
| `accum while let map` (1e6) | 10514.0 | 10785.9 | +2.6 % |
| `accum map add \| sum` (1e6) | 5725.6 | 5588.1 | −2.4 % |
| `accum map cap \| sum` (1e6) | 7924.9 | 8106.6 | +2.3 % |
| `accum grade while` (1e6) | 10726.0 | 10537.1 | −1.8 % |
| `accum int while` / `float while` / `range \| sum` / `map id \| sum` / `call while` / `collatz while` | — | — | ±0.8 % |
| `shapes option match` (1e5) | 530.2 | 437.2 | **−17.5 %** |
| `shapes enum match three` (1e6) | 20577.8 | 19988.2 | −2.9 % |
| `shapes field read` / `field write` / `construct` (1e6) | — | — | ±0.5 % |
| `slice_ceiling as_slice in loop` | 9964.3 | 8195.5 | **−17.8 %** |
| `slice_ceiling as_slice hoisted` / `unchecked` | — | — | ±0.6 % |
| `attention vec 256x128` (in-language) | 334.9 | 303.8 | −9.3 % |
| `attention vec 64x64` (in-language) | 48.3 | 44.0 | −8.9 % |
| `attention deque 256x128` (in-language) | 338.2 | 310.1 | −8.3 % |
| `logs heavy pure` (1e4) | 30063.0 | 28854.8 | −4.0 % |
| `logs sync ext` (1e5) | 7577.9 | 7480.0 | −1.3 % |
| `logs sync ext` (1e4) | 671.1 | 690.5 | +2.9 % |
| `logs inline` / `closure` | — | — | ±1.2 % |
| `mandelbrot` (both sizes) | — | — | ±0.5 % |
| `programs bf table` / `scan` / `call` | — | — | ±1.5 % |
| `spawn` (24 rows, `acvus/us`) | — | — | ±2.2 % |

`CallExtern1<Glue<accum::id_of>, false, false>::run`, the operation
`extern while` runs, is 15 instructions with the handler inlined:

```
movzwl 0x1a(%rdi),%eax        # self.a
mov    0x10(%rsi),%rcx        # the frame's cells
mov    0x8(%rcx,%rax,1),%rax  # the argument — and `id_of`'s whole body
mov    0x10(%rdi),%r8         # self.takes
movzwl 0x22(%rsi),%r9d
shl    $0x4,%r9d
not    %r8
and    %r8,(%rcx,%r9,1)       # the mark word
movzwl 0x18(%rdi),%r8d        # self.dst
mov    %rax,0x8(%rcx,%r8,1)
mov    (%rdi),%rax            # self.next
mov    0x8(%rdi),%rcx
mov    0x20(%rcx),%rcx
mov    %rax,%rdi
jmp    *%rcx
```

The identity body left no instruction of its own: the argument's load is the
result's store. `spawn`'s rows do not move because an async or heavy call's
cost is its allocation and its thread hop, and rule 4's amendment leaves
both where they were.
