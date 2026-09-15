# RFC-0021: A registry is a manifest and a handler table, combined once

Status: Accepted
Date: 2026-09-15
Supersedes: none

## Ruling

A registry contributes two things: a manifest — the declarations a
compiler needs, free of any runtime — and a handler table — one handler
per declared function, for one runtime. The manifest holds type
declarations, shared signatures (RFC-0019), and function declarations;
a function declaration carries its name, its polymorphic type, the bound
of each type variable, whether it is a cast, the signature it is an
instance of if any, and the signature each type variable requires if any.

Every registry declares one namespace, and every name it declares lives
under it: `core::eq`, `llm::chat`. There is one way to declare a function:
`#[extern_fn]` on a Rust function whose first parameter is the runtime.
A function that needs state beyond its arguments — a client, a cache —
takes it as a parameter marked `#[state]`; the registry is built with
that state, and the handler holds it.

All registries are combined once, before anything is checked or run.
Combining rejects a name declared twice and a second instance of a
signature for one type, collects every instance of a signature into one
function under the signature's name, lowers every requirement `T: sig` to
the bound `OneOf(the types with an instance)`, derives cast rules, and
yields the compiler's input — the functions and the type registry — and
the runtime's input — the handler table. Nothing is registered one
registry at a time.

## Rationale

The registry that grew with the extern system joined a declaration to
its handler in one value, so a registry was generic over a runtime even
where only its declarations were wanted; it registered one registry at a
time into a type registry by side effect, so no place held all registries
together, and the harness assembled the compiler's and the runtime's
inputs by hand; and it accepted declarations two ways — an attribute that
reads a Rust signature and a closure whose types a trait reads — so every
consumer met both. Shared signatures need the one place where all
registries meet, and one declaration form keeps that place simple.

State is why the closure form existed. A parameter marked as state gives
the same capability inside the one form: the body reads the state as an
argument, the caller of the registry supplies it, and the declaration's
acvus type does not mention it.

## Not built

- No registration after combining. A program's set of externs is fixed
  before its first check.
- No unnamespaced registry. A name without a namespace is a script's
  own.
- No declaration by closure, no handler trait over closure types, no
  `with_effect`: the effect is declared on the attribute.

## Consequences

- `Registry<R> { manifest: Manifest, handlers: Handlers<R> }`;
  `Externs<R>` is the combined result with `functions`, `types`, and
  `handlers`; `Externs::combine(Vec<Registry<R>>, &Interner)` is the one
  constructor and fails on a duplicate name or instance.
- `extern_registry! { ns: "core", types: [..], signatures: [..], fns: [..] }`
  builds a `Registry<R>`; a stateful function is listed as
  `chat(client)` with its state value.
- `extern_signature! { eq<T>(a: &T, b: &T) -> bool }` declares a shared
  signature and a marker type of the same name; `#[extern_fn(instance_of
  = eq)]` declares an instance; `T: HasInstance<eq>` on a type parameter
  declares a requirement.
- The interpreter's context and the test harnesses take an `Externs`.
- A script's bare name resolves to its own function if it declares one,
  else to the one extern of that name under any namespace; a name two
  namespaces declare is an error at the call. A qualified call form in
  the grammar is not part of this ruling.
