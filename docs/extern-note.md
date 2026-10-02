# Extern boundary: the note

The single basis of the extern rework. Every step derives from it, and structure derives from it, never from instances; where a step does not, the work stops and the question goes to its owner. It is fixed until its owner changes it. The compiler's meaning is `docs/rfc_compiler`.

## Aim

- Every ordinary Rust function signature is expressible at the extern boundary, strictly more than before. Excluded from signatures are type-level tricks, such as associated types, GATs, HKTs, HRTB tricks and upcasting, and lazy monomorphization.
- Generality is the intersection of Rust and acvus. What acvus defines and Rust expresses by what Rust guarantees is a general case, met by one rule over the whole concept: `Option`, `Result`, tuples of every arity Rust's means reach, objects and enums as Rust structs and enums projected as pin-project projects them, primitives, `String`, `str`, slices, references, functions and passing modes. A type only Rust has, such as `Vec`, is no part of acvus: an extern-defined type is a Rust type its author defines and marks `#[derive(Opaque)]`, so that what is opaque is written where the type is, and a type the author did not define enters wrapped in one, as `#[derive(Opaque)] struct Stack<T>(Vec<T>)`. What Rust does not guarantee is not guaranteed.
- A signature that cannot be expressed is a defect of the structure, and the structure is revisited.

## Invariants and constraints

1. A round trip holds at every position.
2. Lifetimes are exact.
3. A value that crossed is kept nowhere its declaration does not show.
4. The boundary has no dynamic check and no dynamic conversion.

These hold by the primitives below, with nothing added on top. The standard is a safe API, and `unsafe` is welcome where it is stated honestly and kept local: an explicit `unsafe` can be safer than a safe-looking API. A function that consumes a `Value` rests on premises the `Value` does not carry, that it was made as a type, that it is live, and that its holder owns it or holds it exclusively; every such function is an `unsafe fn` stating them, and a function that only makes a `Value` is safe. Boundary and author code have no branch on a type's form or kind, no forced unwrap, no cast, and no assert or type check outside the runtime. Only the runtime contract, `Uniform`'s boxing included, may require `'static`. Inputs are values or views; outputs are owned values or Rust references. A shared or exclusive reference to a structure is a Rust reference to the structure as the runtime holds it, its parts uniform, and an exclusive one reaches the variant's place, so a store through it writes the variant and the payload together (ownership §5); `Option<&T>` reaches the payload alone. A tuple is the exception: Rust guarantees no tuple layout and a runtime holds a tuple of any arity as a run of values, so a reference to a tuple is the tuple of its parts' references. An object or an enum is not lent by reference at all (`Structural`). Where attaching something to a structure or to its parts means different things, the difference is expressed without ambiguity. The registry is the extern side's; the compiler only receives declarations, and every declaration form is writable and written correctly.

## Primitives

| Primitive | Meaning | Guarded by |
|---|---|---|
| `Value` | The runtime's value: pointer-like and `Copy`, so it carries neither its type nor its ownership. Any code may hold one; only an `unsafe fn` consumes one. | `unsafe` at every consumer |
| `Uniform<Rt, X>` | A value boxed by the runtime, its type kept by the marker `X`. Made by `Uniform::new(rt, value)`, or by `Crossing::take`, which is `unsafe`, so a `Uniform` holds a value made as `X` and its views and its drop are safe. Projected to `&` or `&mut` only through the runtime, never taken out; a `Copy` value may be got. Moved, popped and dropped whole. Requires `X: 'static`, and is invariant in `X` by `PhantomData<fn(X) -> X>` (types §4). Its layout is `R::Value`'s for every `X`: it carries no niche of `X`, which would make `Option<Uniform<Rt, X>>` differ from one `X` to another. Used where a concrete type is needed: `Stack<Uniform<Rt, String>>` is an unmarked `Stack<String>` seen uniformly. | marker, privacy, `'static` |
| `TypeVar`, `EffectVar`, `IdentityVar` | The one marker bound of a generic parameter, one per kind. Carried through other types, so Rust keeps each variable one type. | Rust's type checker |
| `InstanceOf` | An impl's witness apart from its value, given only by reference and so never kept: kept, it could meet another type of the same signature. | borrow checker |
| `Instance` | A value and its impl as one, of one generic function. Its mode, owned, shared or exclusive, is a marker the macro reads from the signature; the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, its flows tracked. | type, flows |
| `Crossing` | A type's crossing by value: `give` hands the runtime a value, and `take`, an `unsafe fn`, takes one back whole on the premise that it was given as the type and that the caller owns it. Every type of the intersection implements it, so an entry or a derive opens each argument or field by its own kind, never by its name. | Rust's type checker |
| `#[derive(Structural)]` | A Rust struct with named fields crosses as an acvus object, and a Rust enum as an acvus enum, by value alone: the derive gives each field by its own crossing and takes each back. Neither is lent by reference, the second trade-off taken, since such uses are rare; `Option` and `Result` keep their references. Fields and variants take their places in their names' text order (types §2). A variant's payload is nothing for a unit variant, its field for one unnamed field, the tuple of its fields for several, and the object of its fields for named ones. | Rust's type checker |
| Flows | Rust lifetime parameters. rustc checks the body; the compiler tracks liveness in the program, so references may sit inside values. | rustc, compiler |
| Generic function, impl | A generic function enters the compiler as one; an impl enters with the function and the choice it implements. It pairs with `Instance` and `InstanceOf`. | compiler |
| `#`, demotion | A whole concrete type such as `Stack<String>` is a specialized opaque type, apart from the generic `Stack<T>` over uniform values. A value that passes a non-`#` function even once is unmarked from where it is made, as the compiler decides (types §8). | compiler |
| Optimization declarations | Effect, declared facts, ends, casts and conversions, conveyed as declared and trusted. | author |
| Runtime | Lowers the IR and runs it, implementing the compiler's primitives as functions so nothing else needs `unsafe`. Generic over backends and zero cost from outside. Its contract states only the minimal conditions for a safe runtime, with associated types; its safe kinds are implementable without `unsafe`. It may check internally: misuse panics in a safe runtime and is UB in an optimized one. The compiler's correctness is its premise. | Rust |

## Type variables

A type variable is never concrete and never wrapped in `Uniform`; it lives in containers. It carries no `'static`: the macro sees every lifetime bound and refuses one. So nothing keeps it out of sight, and nothing wraps it. A value holding a variable or a loan crosses as its own type, where its declaration shows them.

It is monomorphized as `Uniform<Rt, Never>`, never as `Value`, where `Never` is one fixed uninhabited marker, apart from Rust's `!`, which stands for acvus's `!`. So one value is read under several views: generic code sees `Stack<Uniform<Rt, Never>>`, a concrete view `Stack<Uniform<Rt, String>>`, and a value goes from one to the other and back. The views differ only in `Uniform`'s marker, and three constraints keep them one layout:

- Associated types. `TypeVar` holds for every `Uniform<Rt, X>`, and has no associated type and no supertrait. So an impl that generic code reaches through its one bound covers every view alike, and coherence refuses a second impl for one view; an impl naming `Never` itself is one that generic code cannot reach.
- Storage. A type variable carries no `'static`, so generic code keeps no value of one anywhere. A `Uniform` kept anyway carries its release in its drop.
- Niche and layout. `Uniform<Rt, X>` is `R::Value` behind `repr(transparent)` for every `X`, its marker a zero-sized `PhantomData<fn(X) -> X>`. The views' fields differ only in a marker that adds no niche and no field.

The claim is this narrow one. That `X<T>` and `X<U>` share a layout whenever `T` and `U` do is not claimed: Rust does not guarantee it, and rustc's `-Zrandomize-layout` breaks it, laying out `S<u64>` and `S<i64>` apart. That instantiations differing only in a marker inside `PhantomData` share a layout is rustc's behaviour rather than the language's, and it held under `-Zrandomize-layout` with every seed tried (nightly 1.99.0, 2026-07-20), whose seed a marker inside `PhantomData` does not reach. It is the second exception under Forbidden. Where the extern crate rereads a structure in place, an `Option`'s or a `Result`'s payloads, a const check confirms the sizes and alignments agree; an opaque value's views have no such check, since the type it was made at is erased.

The argument assumes Rust admits no specialization, no default impls and no negative impls, any of which would let an impl single out one view. The owner's assessment is that none will arrive in a form that reaches this: negative impls are all but ruled out, and specialization and default impls remain unfinished and, under the next-generation trait solver, held back by lifetime coercion. Should one arrive, this section is revisited.

## Runtime and registry

Safety is decided per kind. Every kind's consuming functions are `unsafe fn`, since the contract's premises are ones an optimized runtime does not check. A safe kind is implementable with no unsafe operation: a safe runtime writes `unsafe` only in those signatures, checks the premises in their bodies, and panics on a misuse such as aliasing or a released value. An unsafe kind reads storage as a Rust type, in an implementation the extern crate provides once and Miri checks, and which a runtime may replace with its own; misuse within it is UB, and a value's type there is the compiler's premise.

A runtime performs an unsafe operation in two places alone, each asserting the compiler's premise about values it holds: where it calls an extern fn, `ExternFn::call`, with the call's arguments, and where it builds an enum value by the constructor for its tag. In the extern crate, storage is read as a Rust type in `repr` alone; elsewhere an `unsafe fn` states its premises and forwards them. This is a trade-off taken: `unsafe` appears outside one module, but each appearance states a premise where it is used, and none is a case treated apart. The derives' output implements `take` and the views as `unsafe fn` under an `allow(unsafe_code)`, so an author's crate may `deny(unsafe_code)` and still derive; one that `forbid`s it cannot, since `forbid` refuses the `allow` (E0453, observed with rustc 1.97.1). A view is lent in continuation-passing style: the runtime calls a closure with it, so its lifetime is the closure's, and the contract carries no lifetime-generic associated type for it. Each kind lends its own views so, `Option` as `Option<&_>`, and each argument is opened by its own kind; the entry nests the body inside these closures and gives its result inside them. Its zero cost rests on the closures being inlined, which is to be verified, not assumed.

A reference is returned as a Rust reference, `&'a X` or `&'a mut X`. It crosses as the runtime's reference value, which holds the reference's words, and is read back in the extern crate's `repr`. This is the first trade-off taken: a reference points into Rust memory and is read without the runtime, so a runtime's places are Rust memory; in return the references are Rust's own, with their lifetimes, copies and reborrows checked by Rust. A value that holds a loan is plain Rust, such as `Cursor<'a, T>` holding `&'a [T]`.

The unsafe kinds' implementations sit on `Stores`, which every runtime implements with no unsafe operation, its consuming functions `unsafe fn` as every kind's are: it holds the extern crate's `Boxed` behind a `Value`, lends it back shared or exclusive, and gives it back whole, and it holds a reference's `Words`. `Boxed` owns one value of any type, `Send + Sync`, behind one pointer to a header and the value, the header holding an atomic count and the value's drop; it needs no `'static`, since it erases the type. `Words` holds a reference's two words. A `Boxed` or `Words` is read back as a Rust type in the extern crate's `repr` alone, on the premise the `unsafe fn` around the read states, which is the compiler's.

`Runtime::share` gives a second owner of a value; `release` ends one. A boxed value shares through its header's count; a value held inline copies. Sharing is for the boundaries that need it, such as a spawn, not for every value: a value never shared is released after one atomic load of its count.

A value that holds a loan is kept in a `Boxed` with its type, its lifetimes included, erased. An entry opens it again at one named lifetime of its own, `'call`, that every loan of the call takes: never `'static`, and never a lifetime the code around it can stretch, so the body, checked against its own signature's lifetimes, only narrows what it holds, and the loans' life is the flows' premise.

A declaration mirrors the Rust types as written. `Uniform<R, X>` stands for `X`'s acvus type, held uniformly; a whole concrete extern-defined type written natively is marked `#`, and one that holds a `Uniform` or a variable anywhere is not. So an impl or an overload written with `Uniform` is the one generic code reaches, and one written natively is its `#` counterpart, which the compiler picks between (types §8). A body that awaits declares `awaits` and an effect that suspends; the compiler is left as it is.

`Registry: Runtime` is a runtime that registers extern fns. An extern fn reaches it as a type implementing `ExternFn`, which states its own declaration and whose fixed parameter shape is an associated type: its `N` arguments are `Self::Args`, a `[Value; N]`, which `call` takes; the registry gives the declarations to the compiler. Because the runtime receives each extern fn as a type rather than as an erased function, it can build specialized operations around one: a natural closed loop with its `next` inlined differs greatly from one that calls through an erased function. The macros are what implement `ExternFn`.

## Kinds

The runtime contract gives one associated type per kind, each with a trait of its own functions; no function serves every kind. A kind groups the constructors of the compiler's `TyKind` that the boundary treats alike: how a value is made and given back, how a reference to it opens, and whether it is `Copy`. Each kind is safe or unsafe.

| Kind | `TyKind` | Safety |
|---|---|---|
| Scalar | `Int`, `Float`, `Char`, `Bool`, `Unit` | safe |
| String | `String` | safe |
| Tuple | `Tuple` | safe |
| Object | `Object` | safe |
| Option | `Option` | safe |
| Result | `Result` | safe |
| Enum | `Enum` | safe |
| Fn | `Fn`, `Closures` | safe |
| Opaque | `UserDefined` | unsafe |
| Ref | `Ref`, with `Str` and `Slice` as what it reaches | unsafe |

Opaque is unsafe because one value is seen as several Rust types, generic and uniform, so it is held untyped and read by layout. Ref is unsafe because a reference is read back from its words; a reference to text or a run, `&str` or `&[T]`, is two words, the fat-pointer exception, so `Str` and `Slice` need no kind of their own. How a `Value` is laid out is the runtime's choice. An opaque value is `Send + Sync`, so that a spawn may carry it.

`Never` has no value, `Order` and `Handle` are no program's, and `Array` is no declaration's. `Option` and `Result` are held in Rust's own shapes, so a reference to one is a Rust reference to it, and its payload opens natively, as `Option<&_>` and `Result<&_, &_>`. Function values cross both ways: acvus gives its closures to an extern, and an extern gives Rust functions to acvus. Either is a Rust lambda, run on an acvus frame or as a Rust function.

Object and Enum have `make` and `take` alone, since neither is lent. An enum value is a tag and one payload, `()` for a variant that carries none. The runtime picks a variant from a table of the derive's constructors by its tag, checking in a safe runtime, so that no code outside the runtime branches on a tag it cannot exhaust. A constructor is an `unsafe fn` taking the payload, since it takes it back as the variant's fields; calling it is the runtime's assertion that the payload is the one made for its tag.

## Lifetimes in the macro

Branded lifetimes, as GhostCell's, serve well but sit uneasily with associated types, and Rust's trait solver is incomplete there: it can fail to prove what holds, and it can ICE. Where it stops the work, the cause is told apart first, with a minimal reproduction: a wrong structure Rust rightly refuses, the solver's incompleteness, or an ICE. A workaround is a layer added; it is brought back for agreement, never put in alone, and it applies to the whole concept, never to the instance that tripped the solver.

Set aside: lending a struct's projection in continuation-passing style. Through a closure, `impl for<'a> FnOnce(Self::Ref<'a>)` with `Self: 'a` on the GAT, rustc takes the bound to imply `'static` and refuses a generic or loan-holding struct; this is the solver's incompleteness, and an implied bound through an extra `PhantomData<&'a Self>` argument does not lift it. A continuation trait whose method quantifies its own lifetime under `where S: 'a` passes every such case, nesting included. It was not taken, since struct and enum cross by value alone.

Open:

- What a Rust closure given to acvus may capture.
- A function value's body. `FnKind::make` takes a Rust closure of raw `Value`s, called safely, which consumes its arguments on a premise its type does not state. Rust has no `unsafe` closure, so an honest body needs a type of its own with an `unsafe` call, a primitive to agree on; it is settled with the question above.
- Async. Work such as a closure is spawned on a thread pool and so crosses `'static`; where it dies there, what it holds is dropped while acvus counts it alive. `Runtime::share` is the means a spawn keeps its values alive by; how a spawn takes it is open.

## Forbidden

- Instantiation: a named type or form treated apart, a marker, trait, wrapper or conversion for one case, a flag parting two meanings held in one place. One makes a path Rust does not guard; nothing built on it can then be guarded, and no further work repairs that.
- Reliance on what Rust does not guarantee. Two exceptions are admitted, each with its check. A fat pointer's layout as two `usize`s, with no assumption about which of them holds the pointer and which the metadata, checked by Miri. And instantiations of one type that differ only in a marker inside `PhantomData` sharing a layout (Type variables), checked by Miri and by the tests under `-Zrandomize-layout` with several seeds.
- Lifetime detach: a lifetime made through `unsafe`, such as an unsafe GAT projection `'x` over `'t`, that is not bound to its source. In generic code it is unbounded, and every borrow check around it passes vacuously.
- A step the primitives do not derive: stop and ask.

## Verification

Each change to the extern crates passes, on the safe runtime:

- `cargo test -p acvus-extern -p acvus-extern-macro -p acvus-extern-test`
- `cargo +nightly miri test -p acvus-extern-test`
- `RUSTFLAGS="-Zrandomize-layout -Zlayout-seed=N" cargo +nightly test -p acvus-extern-test` for several `N`, in a target directory of its own.
