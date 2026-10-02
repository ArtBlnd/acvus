# Extern boundary: the note

The single basis of the extern rework. Every step derives from it, and structure derives from it, never from instances; where a step does not, the work stops and the question goes to its owner. It is fixed until its owner changes it. The compiler's meaning is `docs/rfc_compiler`.

## Aim

- Every ordinary Rust function signature is expressible at the extern boundary, strictly more than before. Excluded from signatures are type-level tricks, such as associated types, GATs, HKTs, HRTB tricks and upcasting, and lazy monomorphization.
- Generality is the intersection of Rust and acvus. What acvus defines and Rust expresses by what Rust guarantees is a general case, met by one rule over the whole concept: `Option`, `Result`, tuples of every arity Rust's means reach, objects and enums as Rust structs and enums projected as pin-project projects them, primitives, `String`, `str`, slices, references, functions and passing modes. A type only Rust has, such as `Vec`, is an extern-defined type to acvus and reaches it only through the general mechanisms. What Rust does not guarantee is not guaranteed.
- A signature that cannot be expressed is a defect of the structure, and the structure is revisited.

## Invariants and constraints

1. A round trip holds at every position.
2. Lifetimes are exact.
3. A value that crossed is kept nowhere its declaration does not show.
4. The boundary has no dynamic check and no dynamic conversion.

These hold by the primitives below, with nothing added on top. The standard is a safe API, and `unsafe` is welcome where it is stated honestly and isolated: an explicit `unsafe` can be safer than a safe-looking API. Boundary and author code have no branch on a type's form or kind, no forced unwrap, no cast, and no assert or type check outside the runtime. Only the runtime contract, `Uniform`'s boxing included, may require `'static`. Inputs are values or views; outputs are owned values or Rust references. A reference to a structure is written as the structure of references, `Option<&T>` or `(&A, &B)`, never `&Option<T>`, as the IR has it (ownership §5), so a borrow means what it means there. Where attaching something to a structure or to its parts means different things, the difference is expressed without ambiguity. The registry is the extern side's; the compiler only receives declarations, and every declaration form is writable and written correctly.

## Primitives

| Primitive | Meaning | Guarded by |
|---|---|---|
| `Value` | The runtime's value: pointer-like, `Copy`, never exposed. | privacy |
| `Uniform<Rt, X>` | A value boxed by the runtime, its type kept by the marker `X`. Made by `Uniform::new(rt, value)`, projected to `&` or `&mut` only through the runtime, never taken out; a `Copy` value may be got. Moved, popped and dropped whole. Requires `X: 'static`, and is invariant in `X` by `PhantomData<fn(X) -> X>` (types §4). Used where a concrete type is needed: `Vec<Uniform<Rt, String>>` is an unmarked `Vec<String>` seen uniformly. | marker, privacy, `'static` |
| `TypeVar`, `EffectVar`, `IdentityVar` | The one marker bound of a generic parameter, one per kind. Carried through other types, so Rust keeps each variable one type. | Rust's type checker |
| `InstanceOf` | An impl's witness apart from its value, given only by reference and so never kept: kept, it could meet another type of the same signature. | borrow checker |
| `Instance` | A value and its impl as one, of one generic function. Its mode, owned, shared or exclusive, is a marker the macro reads from the signature; the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, its flows tracked. | type, flows |
| `Structural` | A transparent structured type's projections, as pin-project's: `Ref<'a>`, the structure of its parts' shared references, and `Mut<'a>`, its exclusive reach. Derived for a Rust struct or enum; tuples, `Option` and `Result` have their own. A signature writes the generated type by name, never as an associated type. | Rust's type checker |
| Flows | Rust lifetime parameters. rustc checks the body; the compiler tracks liveness in the program, so references may sit inside values. | rustc, compiler |
| Generic function, impl | A generic function enters the compiler as one; an impl enters with the function and the choice it implements. It pairs with `Instance` and `InstanceOf`. | compiler |
| `#`, demotion | A whole concrete type such as `Vec<String>` is a specialized opaque type, apart from the generic `Vec<T>` over uniform values. A value that passes a non-`#` function even once is unmarked from where it is made, as the compiler decides (types §8). | compiler |
| Optimization declarations | Effect, declared facts, ends, casts and conversions, conveyed as declared and trusted. | author |
| Runtime | Lowers the IR and runs it, implementing the compiler's primitives as functions so nothing else needs `unsafe`. Generic over backends and zero cost from outside. Its contract states only the minimal conditions for a safe runtime, with associated types; its safe kinds are implementable without `unsafe`. It may check internally: misuse panics in a safe runtime and is UB in an optimized one. The compiler's correctness is its premise. | Rust |

## Type variables

A type variable is never concrete and never wrapped in `Uniform`; it lives in containers. It carries no `'static`: the macro sees every lifetime bound and refuses one. So nothing keeps it out of sight, and nothing wraps it. A value holding a variable or a loan crosses as its own type, where its declaration shows them.

It is monomorphized as `Uniform<Rt, Never>`, never as `Value`. `X<T>` transmutes to `X<U>` whenever `T` does to `U`, except where `X` consumes `T` through an associated type; `Never` is generated under a fresh name at every compilation, so no code names it to give it an impl of its own. A const check confirms the sizes agree, and the round trip makes the type in between irrelevant.

## Runtime and registry

Safety is decided per kind. A safe kind is implementable without `unsafe`, and misuse within it, such as aliasing or a released value, panics. An unsafe kind's `unsafe` stays inside its implementation, which the extern crate provides once, isolated and checked by Miri, and which a runtime may replace with its own; misuse within it is UB, and a value's type there is the compiler's premise. A view is lent in continuation-passing style: the runtime calls a closure with it, so its lifetime is the closure's, and the contract carries no lifetime-generic associated type for it. Each kind lends its own views so, `Option` as `Option<&_>`, and each argument is opened by its own kind; the entry nests the body inside these closures and gives its result inside them. Its zero cost rests on the closures being inlined, which is to be verified, not assumed.

A reference is returned as a Rust reference, `&'a X` or `&'a mut X`. It crosses as the runtime's reference value, which holds the reference's words, and is read back at one isolated `unsafe` of the extern crate. This is the first trade-off taken: a reference points into Rust memory and is read without the runtime, so a runtime's places are Rust memory; in return the references are Rust's own, with their lifetimes, copies and reborrows checked by Rust. A value that holds a loan is plain Rust, such as `Cursor<'a, T>` holding `&'a [T]`.

`Registry: Runtime` is a runtime that registers extern fns. An extern fn reaches it as a type implementing `ExternFn`, which states its own declaration and whose fixed parameter shape is an associated type: its `N` arguments are `Self::Args`, a `[Value; N]`, which `call` takes; the registry gives the declarations to the compiler. Because the runtime receives each extern fn as a type rather than as an erased function, it can build specialized operations around one: a natural closed loop with its `next` inlined differs greatly from one that calls through an erased function. The macros are what implement `ExternFn`.

## Kinds

The runtime contract gives one associated type per kind, each with a trait of its own functions; no function serves every kind. A kind groups the constructors of the compiler's `TyKind` that the boundary treats alike: how a value is made and given back, how a reference to it opens, and whether it is `Copy`. Each kind is safe or unsafe.

| Kind | `TyKind` | Safety |
|---|---|---|
| Scalar | `Int`, `Float`, `Char`, `Bool`, `Unit` | safe |
| String | `String` | safe |
| View | `Str`, `Slice` | unsafe |
| Tuple | `Tuple` | safe |
| Object | `Object` | safe |
| Option | `Option` | safe |
| Result | `Result` | safe |
| Enum | `Enum` | safe |
| Fn | `Fn`, `Closures` | safe |
| Opaque | `UserDefined` | unsafe |
| Ref | `Ref` | unsafe |

Opaque is unsafe because one value is seen as several Rust types, generic and uniform, under fresh `Never`s, so it is held untyped and read by layout. Ref is unsafe because a reference is read back from its words. View is unsafe so that a runtime may hold a text or a run as it likes, a short text inline for one; a view may also cross as two `Value`s.

`Never` has no value, `Order` and `Handle` are no program's, and `Array` is no declaration's. `Option` and `Result` open by reference natively, as `Option<&_>` and `Result<&_, &_>`. Function values cross both ways: acvus gives its closures to an extern, and an extern gives Rust functions to acvus. Either is a Rust lambda, run on an acvus frame or as a Rust function.

## Lifetimes in the macro

Branded lifetimes, as GhostCell's, serve well but sit uneasily with associated types, and Rust's trait solver is incomplete there: it can fail to prove what holds, and it can ICE. Where it stops the work, the cause is told apart first, with a minimal reproduction: a wrong structure Rust rightly refuses, the solver's incompleteness, or an ICE. A workaround is a layer added; it is brought back for agreement, never put in alone, and it applies to the whole concept, never to the instance that tripped the solver.

Open:

- A value that holds a loan is not `'static`, so the runtime can keep it only with its lifetime erased, and it is opened again bound to its call's one lifetime, never unbounded. How, at one isolated `unsafe`, is designed first.
- What a Rust closure given to acvus may capture.
- Async. Work such as a closure is spawned on a thread pool and so crosses `'static`; where it dies there, what it holds is dropped while acvus counts it alive. An `Arc` per value makes `map` and `collect` too slow. Candidates are one `Arc` at the spawn rather than one per value, or a shared-ownership primitive the runtime provides.

## Forbidden

- Instantiation: a named type or form treated apart, a marker, trait, wrapper or conversion for one case, a flag parting two meanings held in one place. One makes a path Rust does not guard; nothing built on it can then be guarded, and no further work repairs that.
- Reliance on what Rust does not guarantee. The one admitted exception is a fat pointer's layout as two `usize`s, with no assumption about which of them holds the pointer and which the metadata. It is admitted because Miri checks it at every compilation in CI.
- Lifetime detach: a lifetime made through `unsafe`, such as an unsafe GAT projection `'x` over `'t`, that is not bound to its source. In generic code it is unbounded, and every borrow check around it passes vacuously.
- A step the primitives do not derive: stop and ask.
