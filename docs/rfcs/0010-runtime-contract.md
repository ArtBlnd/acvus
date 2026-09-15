# RFC-0010: The runtime contract

Status: Superseded by RFC-0022
Date: 2026-09-10
Supersedes: none

## Ruling

An ExternFn declaration names no runtime. A runtime is anything that
implements the `Runtime` trait, and every declared ExternFn, extension
type, and registry is generic over it.

    trait Runtime {
        type Value;     // the erased representation
        type Closure;   // a callable script value
        type Error;     // absorbs ExternError
        // one constructor and one opener per value shape, plus call
    }

A generic ExternFn never opens a type variable. `TyVar` is a marker; a
body that must cross the boundary says so with `FromValue<Rt>` or
`IntoValue<Rt>` on the variable, and only where it crosses. At runtime a
type variable is `Runtime::Value`, so those crossings are the identity.

Crossings happen in exactly three places:

- the ExternFn boundary, done by the generated glue;
- a callback, done inside `Fn0`, `Fn1`, `Fn2`;
- the edges of a lazy pipeline, done inside the pipeline's typed API when a
  typed source enters or a typed consumer pulls.

Concrete positions cross through the runtime's own shape methods, so a
runtime with an unboxed representation pays nothing for them.

`TypesOnly` is the runtime with no values, for registering declarations
where nothing will run.

## Rationale

A Rust function is compiled before any script exists, so a polymorphic
body cannot see a per-script unboxed representation; it sees a uniform
one. That is a fact about monomorphization, not a design choice, and the
ruling puts the uniform representation where it belongs: chosen by the
runtime, invisible to the body. What the body can be spared is opening
that representation itself, which is the only thing "dynamic realization"
ever was.

A second runtime with its own value representation already exists for the
compiler's side of this system. Binding declarations to one interpreter's
enum would make it unable to host the same extensions.

## Not built

- No runtime-generic overload sets. A generic body over a set of concrete
  unboxed types needs the type checker to choose among same-named
  candidates, which it does not do; that is a later ruling.
- No runtime-chosen extension payload. An extension value is a static name
  over an erased payload the extension chose; a runtime only carries it.

## Consequences

- `FromValue` and `IntoValue` are parameterized by the runtime; the
  implementations for scalars, options, arrays, tuples, and structural
  objects are written once against the shape methods.
- `Runtime::Value` converts to and from itself, so every generic body's
  crossing bounds hold at runtime without being restated by callers.
- A container of a type variable crosses as the runtime's own container:
  a sequence of the runtime's values converts to itself whole, and an
  element is converted only where the body names a concrete type for it,
  which only a `Monomorphize` member can. Nothing else materializes a
  type variable.
- A runtime's error type converts from `ExternError`, the only error a
  handler body can raise on its own.
- A registry is registered for one runtime and yields that runtime's
  handlers next to the compiler's functions.

## Open questions

- Whether a runtime should be allowed to refuse a shape it does not have
  at registration time rather than at the first call.
- Whether the shape methods should also lend a view of a value without
  taking it. Every shape method takes ownership, so a body that names a
  nested container, `List<List<T>>`, takes each inner container apart and
  puts it back once, even when it only reads. A borrowed view would let
  it read in place, and a mutable one change in place, the way RFC-0015
  lends a place to a call.
