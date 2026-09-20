# The Runtime boundary

A host runs the language by signing one trait (RFC-0022). The contract
carries no value taxonomy: a value is opaque to it, extraction and
construction are one `transmute`-based pair, a reference is a value the host
makes, and a closure is run by three calls. `acvus-interpreter` is one host;
kovac is another, signing the same trait with its own representation.

## The contract

`acvus_extern::Runtime`:

```rust
trait Runtime: Sized + Send + Sync + 'static {
    type Value: Cross<Self> + FromValue<Self> + Release + Copy + Default;
    type Frame: Send + Sync;
    type CallFuture<'a>: Future<Output = Self::Value> + Send + 'a where Self: 'a;

    fn frame(&self) -> Self::Frame;

    fn type_of(&self, v: &Self::Value) -> Option<TypeId>;
    fn type_name_of(&self, v: &Self::Value) -> Option<&'static str>;

    unsafe fn materialize<T>(&self, v: Self::Value) -> T;
    unsafe fn erase<T>(&self, t: T) -> Self::Value;
    unsafe fn value_as_ref<'a, T>(&'a self, v: &'a Self::Value) -> &'a T;
    unsafe fn value_as_mut<'a, T>(&'a self, v: &'a mut Self::Value) -> &'a mut T;
    unsafe fn inline_ref<T: Inline>(v: &Self::Value) -> &T;
    unsafe fn inline_mut<T: Inline>(v: &mut Self::Value) -> &mut T;
    unsafe fn deref<'a, T>(&self, r: &'a Self::Value) -> &'a T;
    unsafe fn deref_mut<'a, T>(&self, r: &'a Self::Value) -> &'a mut T;
    unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

    fn none(&self) -> Self::Value;
    fn some(&self, payload: Self::Value) -> Self::Value;
    fn is_none(&self, v: &Self::Value) -> bool;
    fn unwrap_some(&self, v: Self::Value) -> Self::Value;
    fn symbol(&self, name: &str) -> Astr;

    fn call_is_sync(&self, f: &Self::Value) -> bool;
    fn call_now(&self, f: &Self::Value, args: &mut [Self::Value],
                frame: &mut Self::Frame, _: CallToken) -> Self::Value;
    fn call_0<'a>(&'a self, f: &'a Self::Value, _: CallToken) -> Self::CallFuture<'a>;
    fn call_1<'a>(&'a self, f: &'a Self::Value, a: Self::Value, _: CallToken)
        -> Self::CallFuture<'a>;
    fn call_n<'a>(&'a self, f: &'a Self::Value, args: &mut [Self::Value], _: CallToken)
        -> Self::CallFuture<'a>;
}
```

The side an extern sees, in `acvus_extern::func`:

```rust
trait ClosureFn<Rt: Runtime> {
    type Args;          // Fn0: ()  Fn1: (Value,)  Fn2: (Value, Value)  Fn3: …
    type Ret;
    fn call_now(&self, rt: &Rt, frame: &mut Rt::Frame, args: Self::Args) -> Self::Ret;
    fn call<'a>(&'a self, rt: &'a Rt, frame: &'a mut Rt::Frame, args: Self::Args)
        -> impl Future<Output = Self::Ret> + Send + 'a;
}
```

- **`materialize` / `erase` are the whole crossing, by Rust type.** A host
  reads its own `Value`, and once the branch is fixed the payload and `T`
  share a layout. When `T` is the host's own `Value` — a type variable at
  runtime — both are the identity. An extension type's effect, length and
  identity parameters are `()` at runtime, so a value built by hand carries
  `()` there too. They are `unsafe` because the layout match is the caller's
  contract, and they return `T` rather than `Result<T>` because a mismatch
  is a compiler bug and panics.
- **A container of a type variable crosses whole.** `Vec<String>` is the
  same `Vec<String>` on both sides: a `Monomorphize` extern is one instance
  per member type, plus the erased value as the instance of last resort
  where the parameter carries no other bound. Nothing unifies layouts.
- **A reference is a value** (RFC-0018). `reference(&v)` makes the word that
  names `v`'s storage; `deref`/`deref_mut` read through it. An extern whose
  closure parameter is `Ref<T>` passes `rt.reference(&lent)` and keeps the
  reference no longer than the call; an extern parameter `&T` / `&mut T` is
  read by the glue through `deref` / `deref_mut` from the reference the
  caller passed.
- **A closure is a `Value`, and an extern names its arity.** `Fn1<T, U, E,
  Rt>` (or `Fn0`/`Fn2`/`Fn3`) in a signature is what the type solver reads
  the closure's type from; the body calls `f.call(rt, frame, (a,))`, each
  argument moving into the callee's parameter. `ClosureFn` is the only road
  to the host's `call_0/1/n` and `call_now`: they take a `CallToken` only
  `func.rs` can mint. The future is a GAT over the borrow of `&self` and
  `f`, so a host clones nothing to make it `'static`.
- **A closure call is synchronous when the type said so.** `call_is_sync`
  is asked once, when `Fn0`/`Fn1`/`Fn2`/`Fn3` is built, and `call_now` is
  the call that follows from a true answer (RFC-0046). The interpreter
  answers it from the callee's prepared entry.
- **No copy on the contract.** A `Value` is `Copy` and has no `Drop`
  (RFC-0048): releasing a `Large` is an explicit act — a drop operation,
  the frame's sweep, or the `Drop` of an `Owned<R>` a Rust holder keeps.
  Sharing in the language is the extern `clone(&x)` (RFC-0019).
- **No equality on the contract.** A word compares as a word and a `String`
  by `StringEq` (RFC-0020); an extension type compares through the shared
  signature `core::eq` (RFC-0019), and one with no instance cannot be
  compared.
- **`Send + Sync + 'static` is the contract.** `Runtime`, `Value` and
  `Var<K>` carry it, and `materialize`/`erase` require it of `T`. Values
  cross `spawn_blocking`, sit in closure captures, and are pulled by
  spawned generators; a runtime is held by async handlers.

## Registration

A registry contributes a manifest (types, shared signatures, function
declarations) and a handler table (RFC-0021). `Externs::combine` joins every
registry once, always starting with `acvus_extern::core` (the `clone` and
`eq` signatures): it rejects a name declared twice, collects each
signature's instances into one function, and yields the compiler's
`functions` and `types` and the runtime's `handlers`. A handler that calls
a signature takes the instance as a parameter (RFC-0067, pending). Every name lives under its registry's namespace (`std::len`,
`llm::chat`); a bare name two namespaces declare is the candidate set of
RFC-0043.

## How the interpreter meets it

1. **Extern dispatch.** `ExternHandler` is `Sync`, `Heavy` or `Async`
   (RFC-0046), each holding a plain `fn` pointer or a stateful pair. A
   synchronous declaration compiles to one `SyncAbi` shape:
   `Arity0`..`Arity3` is one Rust parameter per argument, read out of the
   register the operation names; `Window` — four or more arguments, every
   asynchronous handler, and every spawn — is lent a contiguous run of the
   caller's registers; `Slice` returns its run in two registers with no
   boxing (RFC-0047 §6). `prepare` reads the shape once and gives the call
   operation the bare `fn` pointer, so no decision stands between the
   dispatch and the handler (RFC-0052 §6).
2. **Closure.** `Fn1::call` is `rt.call_1(&value, arg, token)`. The iterator
   pipeline keeps its closures as `Fn1<Value, Value, (), Rt>` — the same
   value under erased element types (`Fn1::erased`) — so one op shape serves
   every `T`. A callee parameter of type `&T` receives the reference word as
   it is; a parameter of type `T` receives the moved value.
3. **Runtime instance.** `AcvusRuntime(InterpreterContext)` is the shared
   context, all `Arc`s, cloned at the dispatch site and handed to the
   handler by reference (sync) or by value (async).

## The interpreter's `Value`

`#[repr(C)] struct Value { kind: Kind, word: u64 }`: one byte that says what
the word is and one word that is the bits, an address, or nothing. 16 bytes,
`Copy`, no `Drop`. `Kind` is `#[repr(u8)]` over `Undef`, `Ref`, `Large`,
`None` and one variant per `Inline` type (`I8`..`U64`, `F64`, `Char`,
`Bool`, `Unit`), so the byte a value carries is both its discriminant and
the Rust
type it was erased from; `Kind::of::<T>()` const-folds, and the spare values
above the last variant are the niche that keeps `Option<Value>` at 16 bytes.
One scalar in the first word and one in the second is a `ScalarPair` in
rustc's x86-64 ABI, so a `Value` — and an `Option<Value>` — is passed and
returned in `rax`/`rdx` and stored with two instructions;
`acvus-interpreter-test`'s `asm_probe` example is the contract that says so,
and `value.rs` const-asserts both sizes.

`Undef` is the SSA initial value of a loop-defined variable, and `None` the
language's `None` — with a word that counts the `Some`s around it, so
`Some(Some(None))` is `{None, 2}` and `Some(v)` for any `v` that is not one
of these is `v` itself (RFC-0039). An option costs no allocation and no tag
of its own, and neither direction consults a type: `Value::some` reads the
payload's kind, `is_none` reads the word, `some_payload` undoes `some`. What
that leaves the preparation to decide is which of three a `PathSeg::Payload`
reads, since an `Option<Result<A, B>>` value *is* the `Result` value:
`prepare` resolves every step against the type it stands on and drops an
option's step where the payload type is not itself an option (`code::Step`).

A type rides in the word when it fits and owns nothing (`size_of <= 8 &&
!needs_drop`, const-asserted at the `Inline` impls); a reference is the
address of the target register under `Kind::Ref`; everything else is
`Kind::Large`, a `Box<Slot<T>>` whose header holds one `&'static Vtable` —
`type_id` (an assert only), `name`, `drop`, `debug`, and which composite it
is. A vtable is a constant of the type it describes (RFC-0048 §7), so
erasing a `Large` reads no registry and takes no lock. Releasing one is
`Release::release`, called by a drop operation, by the frame's sweep, or by
the `Drop` of the `Owned<R>` a Rust holder keeps. `target`, `peek`, `bits`
and the `as_*` readers check the kind under `debug_assert!` only: the MIR
type checker gives each of those slots a type that admits one kind, and the
boundary a program can actually reach — `acvus_extern::expect_type` — still
panics in release on a value no Rust type was erased into. The host asserts
`unsafe impl Send + Sync for Value`: every payload entered through
`erase<T: Send + Sync>`, and vtables are constants.

## The frame

A frame is a run of **cells** in one `Vec` (`regs.rs`): its registers, then
the one `Value`-wide slot holding the word that marks which of them own a
`Large`, then the window a call out of it takes its callee's frame from. A
cell is `CELL_SLOTS = 16` slots — 256 bytes, four cache lines, starting one
— and holds no mark word of its own, because a frame wider than a cell has
to be one run of `Value`s and an interleaved word would break the
displacement an `Off` already is. A frame of `n` registers takes the
`cells_for(n)` cells that `n + 1` slots reach, and `MAX_FRAME_SLOTS = 64` is
where `prepare` stops, one mark word covering a frame (RFC-0048 §3,
RFC-0052 §5, §6).

A call's frame is the cells above the caller's, taken after one capacity
compare against the `WINDOW_CELLS = 3` the `Vec` keeps above the bound
frame; a callee that does not fit, or a chain deeper than that, roots a
`Store` of its own. An operation names a register by its **byte
displacement** inside the frame (`code::Off`, `slot * 16`, multiplied once
by `prepare`), never by its index: `Slot` is the language-level index and it
does not leave `prepare`.

A synchronous call is lent its frame; it never makes one. `Runtime` carries
`type Frame` and `fn frame(&self) -> Self::Frame`, and `call_now` takes
`&mut Self::Frame` (RFC-0052 §6): a stage in `acvus-ext` makes one `Store`
when it is built and lends it per element, and a consumer makes one before
its drain loop. An unbound `Store` is an empty `Vec` and reaches no
allocator. `Store::bind(body)` is the only way to a frame: it sizes the
`Vec` once per binding, writes the body's `param_marks`, and answers whether
the frame already carries that body's slot kinds and entry constants.

A closure value holds its entry rather than its code: `FnValue.entry` is an
`Arc<dyn Callable>` projected out of the `Code` when the closure was made,
so a synchronous closure call reads a pointer and jumps instead of asking
whether the code is a `Body` or an `Expr`.

A register whose type is a word — an integer, a float, a `Bool`, a `Unit` —
is opened once with its kind when the frame is made (`Body::slot_kinds`),
and every write to it afterwards stores the word alone. `Regs::store::
<LARGE, WORD>` is the one store that says which of the two a result takes;
the pair `LARGE && WORD` is a `const` assert, because a register the frame
gave a kind holds no `Large`. `prepare::Dest` reads `word_kind` for `WORD` —
the same predicate `slot_kinds` opened the register by.

Every storage is a register: a variable slot, a temporary, a parameter. A
`Take` moves out of one, an `Assign` moves in, a `Ref` makes the word naming
one; through a reference (`RefTarget::Through`) a `Take` copies a word and
an `Assign` moves a value in. A context is a variable of the body that names
it (RFC-0025): `Fetch` moves its whole value out of the run's page into a
register (a page with no value for it is a panic) and `Commit` moves one
back; a body fetches at entry, commits at return, and commits and fetches
around a call whose summary touches the context.

## Open

- **`Pointee` for unsized `T`.** `materialize`/`erase` are `T: Sized` today,
  and `Large` is a `Box`. When `core::ptr::Pointee` stabilizes, splitting a
  value into (thin pointer, metadata) lifts the `Sized` bound — an `str`, a
  `[Value]`, a `dyn` erases and rematerializes with no shape change to the
  contract.
- **Persistence.** Per-type dump/restore re-attaches to the vtable beside
  `drop` and `debug`; the journal's version store is where it lands.
- **`Material<T>`.** An 8-byte typed handle into a host pool, opt-in for
  pointer-shaped types (a linked list), where the inline handle removes the
  double indirection. Held until the pool exists.
