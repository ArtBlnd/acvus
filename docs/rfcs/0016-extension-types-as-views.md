# RFC-0016: An extension type is a view over the runtime's value

Status: Superseded by RFC-0022
Date: 2026-09-10
Supersedes: none

## Ruling

An extension type with type parameters is a transparent view over the
runtime's value: a newtype with the same layout as `Runtime::Value`,
carrying its parameters only as phantoms. A value crosses the extern
boundary as itself at every depth; a sequence of views is the same
memory as a sequence of values. A body reads and changes a view through
the runtime's shape methods, which lend the value rather than take it,
and converts an element to a Rust type only where it names one.

Only a Rust-owned payload, a regular expression or a pipeline, lives
inside the value as an opaque object, and its view is the same newtype
over the value that holds it.

## Rationale

RFC-0010 puts the uniform representation in the runtime and says a body
never opens a type variable. Two things still cost time in proportion
to the data: a container whose Rust representation is parametric, such
as `List<T>` over `Vec<T>`, rebuilt its elements on each crossing; and a
nested container, `List<List<T>>`, took each inner container apart and
put it back, because every shape method takes ownership. The first is
closed by converting a sequence whole; the second cannot be closed while
the Rust type and the runtime type have different layouts. Making the
view the Rust type removes both at the definition: the crossing is a
reinterpretation, and the only work left is the element access a body
asks for.

The type checker has already established the shape of every value that
reaches a body, so a view need not check a tag on creation; a runtime
that lacks a shape refuses it at registration.

## Not built

- Nothing yet. This records a direction.

## Consequences

- The `Runtime` contract gains shape methods that lend a value, shared
  and mutable, next to the ones that take it.
- `List<T>`, `Arr<T, N>`, and any parametric extension type become views;
  an extension author loses `Vec<T>` fields and gains typed accessors.
- `FromValue` and `IntoValue` for a view are the identity; the whole-
  sequence conversion of RFC-0010 is no longer needed.

## Open questions

- Whether the same view design serves a runtime whose value is unboxed.
- How a body takes a Rust-owned payload out of a view for mutation, and
  whether that path stays `Arc`-based.
