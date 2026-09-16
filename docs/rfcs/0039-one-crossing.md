# RFC-0039: One crossing at the boundary

Status: Accepted
Date: 2026-09-16
Extends: RFC-0022, RFC-0032, RFC-0036, RFC-0038

## Ruling

Every type an ExternFn takes or returns implements one trait, `Cross`,
and the glue calls it and nothing else: `erase` hands the runtime a
value, `materialize` takes one back, `deref` and `deref_mut` read a value
through a reference the runtime holds. There are no tiers chosen where a
macro expands; the type says how it crosses.

- A scalar is stored as itself.
- An extension type — `#[derive(ExternType)]` — is stored as its payload,
  the first field, and is `#[repr(transparent)]` over it so that a
  reference to the payload is a reference to the type. Its phantom
  parameters never reach the store: every `Iter<T, E, I, Rt>` is one
  `Pipeline<Rt>` there, and the checker's types live only in the name.
- A derived struct or enum is rebuilt field by field (RFC-0032,
  RFC-0036). It has no storage of its own type, so reading it through a
  reference is refused.
- A container — `Option`, `Result`, `Vec`, `Array` — crosses each
  element by the element's own `Cross`, whatever that is: a struct inside
  a `Vec` inside a struct is rebuilt at every level. A container is read
  through a reference only when its element is the runtime's value; a
  converted container has no storage of its element type.
- A carrier — `Ref`, `RefMut`, `Fn0`..`Fn3` — is the runtime value it
  holds.
- The runtime's own value crosses as itself: `Runtime::Value: Cross<Self>`.
  A type variable of an ExternFn is that value at run time, so a
  `Vec<T>` parameter reaches the store as `Vec<Value>` in one erase.

## Rationale

RFC-0022's three tiers were chosen by autoref specialization at the
expansion site, and autoref resolves at the generic definition, not at
the instantiation: inside `erase_field<T>` the compiler never saw
`T: Cross`, so a derived struct nested in another crossed as an opaque
Rust value, and `Result<Regex, E>` — whose `Cross` impl demanded
`T: Cross` of a type that had none — fell to the as-is tier as a whole.
Both surfaced on 2026-09-16, one from each side of the same seam. A
single trait that every boundary type implements makes the element's
crossing a bound the compiler checks, and puts the choice where the type
is declared.

Storing an extension type as its payload is what RFC-0022 meant by
`Repr`, applied uniformly: the phantoms are the checker's, the payload is
the store's, and `repr(transparent)` is the one fact that lets a
reference cross between them, so the derive requires it.

A closure call is typed at its declaration: `Fn1<A, R>::call` takes an
`A` and returns an `R`, both crossing inside `call`. A type variable of an
ExternFn is `Rt::Value` at run time, and `Rt::Value: Cross<Rt>`, so a
Rust function bounds its type variable `T: TyVar + Cross<Rt>` and
converts with `T::materialize` / `erase` where a value enters or leaves
Rust; an iterator's items stay runtime values inside the pipeline
(`next_value`), are lent to a predicate through `Ref::lend`, and become a
`T` only through `FromValue` (RFC-0041). Nothing in `iterator.rs` or
`iter.rs` is unsafe.

## Not built

- A checker rule refusing `&T` parameters of converted types in extern
  signatures; today the refusal is at run time.

## Consequences

- `acvus-extern`: `Cross` with `deref`/`deref_mut`; `cross_as_stored!`;
  `repr.rs` (`Crossing`, `AsCross`, `AsIs`, `HasRepr`) and `Carried` are
  gone; `Runtime: Sized` with `Value: Cross<Self>`; `SpaceHooks::of`
  reads a journaled value through its `Cross`.
- `acvus-extern-macro`: the glue calls `<T as Cross<__R>>::…`; `is_carrier`
  is gone; `#[derive(ExternType)]` emits the payload crossing and
  requires `#[repr(transparent)]`.
- `acvus-ext`: `Regex`, `DateTime`, `Decimal`, `Iter` are
  `#[repr(transparent)]`; `Deque` uses `cross_as_stored!`.
- `acvus-interpreter`: `Value: Cross<AcvusRuntime>`.
- `acvus-extern`: `ClosureFn::call` has `Args` and `Ret`; `Ref::lend`.
- `acvus-ext`: `Iter::next_value`; the item-pulling functions bound
  `T: TyVar + Cross<Rt>`; the pipeline's filter holds a typed predicate.
