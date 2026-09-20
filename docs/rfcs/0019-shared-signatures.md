# RFC-0019: A shared signature and its instances

Status: Accepted
Date: 2026-09-15
Supersedes: none
Extended by: RFC-0027

## Ruling

A shared signature is a name with one polymorphic function type and no
body: `core::clone<T>(&T) -> T`, `core::eq<T>(&T, &T) -> Bool`. Any
registry declares any signature it wants, under its own namespace; the
language fixes none. An instance is an ExternFn declared for one
concrete `T` whose type is the signature's type at that `T`; whoever
declares an extension type may declare its instances, in any registry. A
type has at most one instance of a signature.

A script calls the signature by name; the call resolves like a
`Monomorphize` function (RFC-0011): the instance whose type matches the
call's resolved type runs. When the registries are combined, every
instance of a signature is collected into that one function, so the
script sees one name whatever registry each instance came from.

An ExternFn that needs a signature of a type variable takes the instance
as a parameter the call site fills (RFC-0067). A type variable carries no
requirement of its own: the bound `OneOf(every type with an instance)` is
computed for a signature's own type, not for a caller's variable.

A shared signature declares no body, a type declares no impl block, and
no value carries a table: the signature is a helper name with a fixed
shape, and the set of types that fill it is closed at registration.

## Rationale

Copying, comparing, printing, and hashing are the operations a language
wants on many types with one name, and the ruling that only a primitive
copies (RFC-0018) made the first of them a call: `clone(&x)`. A
`Monomorphize` function already gives one name many instances, but its
member list is written where the function is declared, and `Regex`'s
`clone` is written where `Regex` is. This ruling opens the member list to
other registries and closes it again when they are combined; the solver then
sees the `OneOf` bound it already carries.

Traits with implementation blocks and dynamic dispatch would make a value
carry what it can do. Here a type's abilities are a fact of the registry,
known before a script is checked, and a call is resolved by type; the
runtime never asks a value.

## Not built

- No generic functions in the language, and no `dyn`. A signature is
  called on a concrete type; a script cannot write a function over "any
  `T: eq`".
- No instances for structural types declared in a script. An object type
  gets an instance only if a registry declares one for that exact type
  (the standard registry decides whether `eq` on objects is structural).
- No default instance and no fallback: a type without an instance of
  `eq` cannot be compared, and that is a type error at the call.
- No inheritance between signatures. `ord` does not imply `eq`; a type
  declares both.

## Consequences

- A registry item kind for a signature declaration: the name, its
  polymorphic type, and its namespace, declarable by any registry. The
  first two, and the only ones introduced with this ruling, are
  `core::clone` and `core::eq`, with instances for the primitives and
  `String` in the standard registry.
- An ExternFn attribute naming the signature it instantiates; registration
  checks the Rust signature against the declaration at the instance's
  type and rejects a second instance for the same type.
- Combining registries collects instances per signature into one
  `Monomorphize`-shaped function.
- Combining registries lowers a signature's own first bound to `OneOf`
  over the collected instance types before type checking begins. A
  declaration carries no per-variable requirement: there is no Rust bound
  the macro reads into one, and no error for a requirement naming an
  unknown signature.

## Open questions

- Whether `eq` on structural object types is provided field-wise by the
  standard registry. This ruling leaves it to that registry.
