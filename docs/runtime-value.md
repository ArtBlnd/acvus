# The Runtime boundary

A host runs the language by signing one thin contract. The contract carries
no value taxonomy: a value is opaque to it, extraction and construction are a
single `transmute`-based pair, and everything a host does to run a call is five
methods and one future. The interpreter is one host; kovac and a GPU
backend are others, each signing the same trait with its own representation.

## The contract

```rust
trait Runtime: Send + Sync + 'static {
    type Value: Send + Sync + 'static;
    type Error: From<ExternError> + Send + Sync + 'static;
    type CallFuture<'a>: Future<Output = Result<Self::Value, Self::Error>> + Send + 'a
    where Self: 'a;

    unsafe fn materialize<T: Send + Sync + 'static>(&self, v: Self::Value) -> T;
    unsafe fn erase<T: Send + Sync + 'static>(&self, t: T) -> Self::Value;
    unsafe fn deref<'a, T: Send + Sync + 'static>(&self, r: &'a Self::Value) -> &'a T;
    unsafe fn deref_mut<'a, T: Send + Sync + 'static>(&self, r: &'a Self::Value) -> &'a mut T;
    unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

    fn call_0<'a>(&'a self, f: &'a Self::Value, _: CallToken) -> Self::CallFuture<'a>;
    fn call_1<'a>(&'a self, f: &'a Self::Value, a: Self::Value, _: CallToken)
        -> Self::CallFuture<'a>;
    fn call_n<'a>(&'a self, f: &'a Self::Value, args: Vec<Self::Value>, _: CallToken)
        -> Self::CallFuture<'a>;
}
```

And the side an extern sees, in `acvus_extern::func`:

```rust
trait ClosureFn<Rt: Runtime> {
    type Args;          // Fn0: ()  Fn1: (Value,)  Fn2: (Value, Value)  Fn3: …
    fn call<'a>(&'a self, rt: &'a Rt, args: Self::Args) -> Rt::CallFuture<'a>;
}
```

A reference is a value (RFC-0018): an extern whose closure parameter is
`Ref<T>` passes `rt.reference(&lent)`, and an extern parameter `&T` /
`&mut T` is read through `deref` / `deref_mut` from the reference the
caller passed.

- **`materialize` / `erase` are the whole extraction/construction story.** A host
  reads its own `Value` (`match` on its tag), and once the branch is fixed the
  payload and `T` share a layout, so it `transmute`s: `Small` carries bits (its
  size asserted `<= 8`), `Large` carries a pointer. When `T` is the host's own
  `Value` — a type variable at runtime — both are the identity: the value
  passes through untouched (RFC-0010). A round trip is by Rust type, so an
  extension type's effect, length, and identity parameters are `()` at
  runtime — the macro's stand-in — and a value built by hand carries `()`
  there too. They are `unsafe` because
  that layout match is the caller's contract, and they return `T` — not
  `Result<T>` — because a type mismatch here is a compiler bug, not a runtime
  failure, so it panics.
- **A value has no taxonomy at the boundary.** No `into_string` / `into_array` /
  `into_object`, no `Str` / `Array` / `Object` associated types, no `FromValue`
  / `IntoValue` / `small_bits`. Only a few user-defined types ever call
  `materialize`; most values flow as `Self::Value` untouched.
- **A closure is just a `Value`, and an extern names its arity.** There is no
  `type Closure`, no `closure()` / `into_closure()`. An extern's signature says
  `Fn1<T, U, E, Rt>` (or `Fn0`/`Fn2`/`Fn3`), which is what the type solver
  reads the closure's type from, and the body calls it as
  `f.call(rt, (&a,))` — the argument tuple's positions are the closure's
  `_0`, `_1`, `_2`. `ClosureFn::call` is the only road to the host's
  `call_0/1/n`: they take a `CallToken` only `func.rs` can mint, so a handler
  never runs a raw value. `Fn0` and `Fn1` have their own host entry (the
  `map` hot path); `Fn2`/`Fn3` go through `call_n` with a stack array of
  references the host reads before its future is first polled. The future is
  a GAT over the borrow of `&self` and the lent arguments, so a host need
  not clone anything to make it `'static`.
- **No equality on the contract.** Equality is a property of a type, not of
  the erased value: a body that compares materializes both sides as the type
  they are and uses that type's `==` — two regexes compare as `Regex`, and
  `contains` is a `Monomorphize` instance per comparable member with no
  fallback. The language's own `==` reads its operands by their MIR type.
- **`Send + Sync + 'static` is the contract, not a workaround.** `Runtime`,
  `Value`, and `Error` carry it, `TyVar` carries it, and `materialize`/`erase`
  require it of `T` — the last is what makes a host's `unsafe impl Send + Sync`
  for its erased value true. Values cross `spawn_blocking`, sit in closure
  captures, and are pulled by spawned generators; a runtime is held by async
  handlers. An attempt to place these bounds only "where needed" produced a
  macro that copied each extern's runtime predicates onto its declaration and
  the same requirement written twice per fn; it was removed.
- **A `Monomorphize` extern falls back to the erased value.** Its instances are
  one per member type, then one for `Value`, selected last; the fallback is
  omitted when the parameter carries a trait bound the erased value cannot
  satisfy.

## How the interpreter meets it

Three seams, and no more:

1. **Extern dispatch.** The interpreter collects `Vec<Value>` arguments and hands
   them, with `&Runtime`, to a handler. The handler — generated mechanically by
   the macro — calls `rt.materialize::<Tᵢ>(argᵢ)` per argument and
   `rt.erase::<Ret>(result)`. The old wrapper traits (`FromValues`, `IntoValue`)
   dissolve into these calls.
2. **Closure.** `Fn1` (and `Fn0`/`Fn2`/`Fn3`) holds a `Value`, not a closure
   handle; `Fn1::call` is `rt.call_1(&value, arg, token)`. This is what
   dissolves `type Closure` and its two conversions. The iterator pipeline
   keeps its closures as `Fn1<Value, Value, (), Rt>` — the same value under
   erased element types (`Fn1::erased`) — so one op shape serves every `T`.
3. **Runtime instance.** `materialize`/`erase` are `&self`, so a host
   instance must exist, and `Runtime: 'static` rules out a borrowing view. For
   the interpreter it is `AcvusRuntime(InterpreterContext)` — the shared
   context, all `Arc`s, cloned at the dispatch site (`ctx.shared.runtime()`)
   and handed to the handler by reference (sync) or by value (async). `call_*`
   borrow the `FnValue` out of the closure value for the future's lifetime,
   copy each lent argument into the callee's frame (every closure parameter
   is callee-owned until the type checker marks borrowed ones), and run
   `fn_value_call`. The interpreter's body keeps handling `Value` directly.

## Open

- **`Pointee` for unsized `T`.** `materialize`/`erase` are `T: Sized` today, and
  `Large` is a `Box`. When `core::ptr::Pointee` stabilizes, splitting a value
  into (thin pointer, metadata) lifts the `Sized` bound — an `str`, a `[Value]`,
  a `dyn` erases and rematerializes with no shape change to the contract. Held
  until then.
- **The interpreter's `Value`.** The first instance is
  `Small(u64) | Large(NonNull<Header>)`, self-contained, non-`Clone`, 16
  bytes. A `Large` allocation begins with its vtable — `type_id` (an assert
  only; kovac drops it), `drop`, `clone`, `debug`, and which composite it is —
  so the value reaches its own vtable without any runtime handle, and
  `impl Drop for Value` releases the payload wherever Rust drops it: a frame
  register, a `Vec<Value>` an extern lets go, an unwinding stack. The
  interpreter's composites are process-wide static vtables; an extension
  type's vtable is registered through the `VtableRegistry` on first `erase` and
  leaked for the life of the process. A type rides in `Small` when it fits the word and owns nothing
  (`size_of <= 8 && !needs_drop`); everything else is `Large`, allocated by
  `Box::into_raw` and taken back by `Box::from_raw`, so allocation and release
  are one pair. `Large` holds the composite in its current Rust shape
  (`Vec<Value>`, `FxHashMap<Astr, Value>`, `VariantValue`, `FnValue`), read by
  the MIR type the interpreter already carries per value (`val_types`). An
  extension type's vtable knows only how to drop: it can be released and
  materialized, but sharing one panics: the IR never asks for a copy, and
  the interpreter's own copies on read go with the reference redesign. The host asserts
  `unsafe impl Send + Sync for Value`: every payload entered through
  `erase<T: Send + Sync>`, and vtables are shared statics. Other hosts may
  use a richer tag; the contract permits it, never requires it.
- **The journal and `Clone`.** The context page (`journal.rs`) sets, gets, and
  snapshots by cloning. That belongs with context dump/restore, which is not
  settled, so it is not redesigned here: the table carries an explicit `clone`
  vtable the page calls by name, as a bridge. When dump/restore lands, a
  snapshot is a dump and a branch is a head pointer, and the vtable goes.
- **No copy primitive on the contract.** UB can only enter through an extern,
  so an extern is given no way to make a second owner of a value: `call_*`
  lend their arguments (`&Value`), a handler's `Value` is a name for storage
  the host owns, and whether the callee takes a copy or an alias is the
  host's decision from the closure's type. The IR has no copy instruction
  either: a real copy is an extern that takes the value and returns a
  second one — `map(|v| clone(v))`, as `v.clone()` in Rust — and that
  extern's Rust `Clone` is where the copy is made. A closure that consumed
  a lent argument would not type-check, so `filter`/`find` are never called
  with one. What still copies today is the interpreter's own register read
  (`use_val` shares every non-move-only value) and the lent argument
  entering a callee frame: both stand in for the alias the reference
  redesign will give them, and an extension type's vtable stays drop-only
  until then.
- **Persistence.** The `ExternValue` type and the `PersistEntry` table keyed on
  it are removed — they existed for the boundary's removed `into_extern`. Per-type
  dump/restore re-attaches to the vtable beside `drop` and `clone`.

- **Container elements: real monomorphization, `Material` opt-in.**
  `List<String>` stays `List<String>`; a generic extern is a `Monomorphize`
  instance per `T`, and each instance `materialize`s/`erase`s the container
  whole, as the same `T`. Nothing unifies layouts, so nothing clashes, and
  Rust already copies per generic — the code growth is the price of speed.
  `Material<T>` — an 8-byte typed handle into `Rt::MaterialPool`, an
  allocator with fixed slots — is opt-in for pointer-shaped types (a linked
  list), where the inline handle removes the double indirection and nodes
  sit contiguous in the pool. Held until the pool exists.

## The cut

The current `Runtime` is thirty-odd methods accreted as patches — the small-value
`into_*`, the large `into_*`, `Str`/`Array` associated types, `object`/`into_object`,
`FromValue`/`IntoValue`/`FromValues`/`IntoValues`, `small_bits`, `type Closure`.
The path to the contract above:

1. Introduce `materialize`/`erase` as the interpreter's own (`transmute`-based),
   and the thin `AcvusRuntime` view holding the interner.
2. Rewrite the macro to emit `materialize`/`erase` per argument and return.
3. Collapse `Fn0/1/2` onto `call_0/1/n` over `Value`; delete `type Closure`.
4. Delete the whole extraction/construction taxonomy the contract no longer
   names, letting the compile errors enumerate every dependent.
