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

These hold by the primitives below, with nothing added on top. The standard is a safe API, and `unsafe` is welcome where it is stated honestly and isolated: an explicit `unsafe` can be safer than a safe-looking API. Boundary and author code have no branch on a type's form or kind, no forced unwrap, no cast, and no assert or type check outside the runtime. Only the runtime contract, `Uniform`'s boxing included, may require `'static`. Inputs are values or views; outputs are owned values or runtime references, so a borrow means what it means in the compiler's IR. Where attaching something to a structure or to its parts means different things, the difference is expressed without ambiguity. The registry is the extern side's; the compiler only receives declarations, and every declaration form is writable and written correctly.

## Primitives

| Primitive | Meaning | Guarded by |
|---|---|---|
| `Value` | The runtime's value: pointer-like, `Copy`, never exposed. | privacy |
| `Uniform<Rt, X>` | A value boxed by the runtime, its type kept by the marker `X`. Made by `Uniform::new(rt, value)`, projected to `&` or `&mut` only through the runtime, never taken out; a `Copy` value may be got. Moved, popped and dropped whole. Requires `X: 'static`, and is invariant in `X` by `PhantomData<fn(X) -> X>` (types §4). Used where a concrete type is needed: `Vec<Uniform<Rt, String>>` is an unmarked `Vec<String>` seen uniformly. | marker, privacy, `'static` |
| `Ref`, `Mut` | A subset of `Uniform`: a `Uniform` is a value, so it may be taken as a reference or a mutable one. Where a `Uniform` came in by reference, they stand for `&T` and `&mut T` in return position, carrying that reference's lifetime. | lifetimes |
| `TypeVar`, `EffectVar`, `IdentityVar` | The one marker bound of a generic parameter, one per kind. Carried through other types, so Rust keeps each variable one type. | Rust's type checker |
| `InstanceOf` | An impl's witness apart from its value, given only by reference and so never kept: kept, it could meet another type of the same signature. | borrow checker |
| `Instance` | A value and its impl as one, of one generic function. Its mode, owned, `Ref` or `Mut`, is a marker the macro reads from the signature; the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, its flows tracked. | type, flows |
| Flows | Rust lifetime parameters. rustc checks the body; the compiler tracks liveness in the program, so references may sit inside values. | rustc, compiler |
| Generic function, impl | A generic function enters the compiler as one; an impl enters with the function and the choice it implements. It pairs with `Instance` and `InstanceOf`. | compiler |
| `#`, demotion | A whole concrete type such as `Vec<String>` is a specialized opaque type, apart from the generic `Vec<T>` over uniform values. A value that passes a non-`#` function even once is unmarked from where it is made, as the compiler decides (types §8). | compiler |
| Optimization declarations | Effect, declared facts, ends, casts and conversions, conveyed as declared and trusted. | author |
| Runtime | Lowers the IR and runs it, implementing the compiler's primitives as functions so nothing else needs `unsafe`. Generic over backends and zero cost from outside. Its contract states only the minimal conditions for a safe runtime, with associated types, and is implementable without `unsafe`. It may check internally: misuse panics in a safe runtime and is UB in an optimized one. The compiler's correctness is its premise. | Rust |

## Type variables

A type variable is never concrete and never wrapped in `Uniform`; it lives in containers. It carries no `'static`: the macro sees every lifetime bound and refuses one. So nothing keeps it out of sight, and nothing wraps it. A value holding a variable or a loan crosses as its own type, where its declaration shows them.

It is monomorphized as `Uniform<Rt, Never>`, never as `Value`. `X<T>` transmutes to `X<U>` whenever `T` does to `U`, except where `X` consumes `T` through an associated type; `Never` is generated under a fresh name at every compilation, so no code names it to give it an impl of its own. A const check confirms the sizes agree, and the round trip makes the type in between irrelevant.

## Forbidden

- Instantiation: a named type or form treated apart, a marker, trait, wrapper or conversion for one case, a flag parting two meanings held in one place. One makes a path Rust does not guard; nothing built on it can then be guarded, and no further work repairs that.
- Reliance on what Rust does not guarantee. The one admitted exception is a fat pointer's layout as two `usize`s, with no assumption about which of them holds the pointer and which the metadata. It is admitted because Miri checks it at every compilation in CI.
- Lifetime detach: a lifetime made through `unsafe`, such as an unsafe GAT projection `'x` over `'t`, that is not bound to its source. In generic code it is unbounded, and every borrow check around it passes vacuously.
- A step the primitives do not derive: stop and ask.
