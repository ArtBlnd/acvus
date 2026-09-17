# RFC-0041: `#τ` is the representation of a slot, and an extension holds values through `Erased`

Status: Accepted
Date: 2026-09-18
Extends: RFC-0022, RFC-0023, RFC-0039, RFC-0040

## Ruling

A value of type `τ` has a representation: uniform, the runtime's
`Value`, which every polymorphic position holds; or specialized, the
Rust type `τ` itself. `#τ` names the specialized representation and is
a fact about a **slot** — a type argument of a user-defined type, or the
target of a reference — never about a bare type: there is no `##τ` and
no `#` at a position that is not a slot. A user-defined type declares
per parameter whether its slot can specialize (`specializable`); a
slot that cannot is uniform. `Fn`, `Object`, `Enum` and handles have no
representation the language defines and carry no `#`.

Only a `Monomorphize` member makes `#`. `#[extern_fn] fn reverse<T:
Monomorphize<(f64,)>>(Vec<T>) -> Vec<T>` has the concrete instance
`reverse@#f64 : Vec<#f64> -> Vec<#f64>` and, when `T` has no other
bound, the generic instance `Vec<ρT> -> Vec<ρT>`, where `ρ` is the one
representation variable of the signature. A plain concrete signature
(`-> Vec<String>`) stays uniform, as before this RFC. The compiler
chooses the instance by type (RFC-0040); a member's glue crosses the
family whole (`CrossSpecialized`: one box, O(1)); every family a member
names declares its two casts `F<#m> -> F<m>` (erase) and `F<m> -> F<#m>`
(materialize), one generic fn each with concrete instances, merged
across registries by type.

A signature's `ρ` is bound only by a decision — the instance choice, or
`solve`'s default `Uniform` — never by a value flow. A flow whose only
disagreement with its target is such an open `ρ` against a fixed
representation is a conversion decision at that site (RFC-0042): a call
argument, a store into a typed place, a return, a pattern's source, an
`else` branch. It is answered by identity when the decision agrees and
by the family's cast when it does not. A conversion consumes the value
it converts. At a `&place` argument that is the place's value: taken,
converted, stored back, and from then until the call the place holds
the referent type the parameter names; every later lend of that place
inside the call — a later argument, or an argument of a nested call —
lends the held type, and after the call the place is restored to its
binding's type, for `&` and `&mut` alike. A later lend of a held place
is a conversion decision like the first, from the held reference to the
parameter, whose only answer is identity, so a second representation
demanded of a held place is a type mismatch; a take while a reference
is live is a borrow error. A reference that is not a borrow of a place
is an error naming both types.

An extension reads and edits uniform values in place through
`Erased<R, T>`: `repr(transparent)` over `R::Value`, made only by
`Erased::new(rt, T)`, read by `as_ref(&self, rt)` / `as_mut`, and for
an `Inline` type (one that fits the value word) by `Deref`, `get`,
`PartialEq` with no runtime in hand. `Vec<Erased<R, T>>` is the
runtime's `Vec<Value>` and crosses whole. A value leaves an extension
type by identity (`T` is the runtime's value) or by a checked downcast
(`FromValue`: the value's `TypeId`, carried by a `Large` payload's
vtable and by a `Small` value's tag, must equal `T`'s); nothing else
reinterprets a `Value`. A reference into storage is read only through a
layout the language promises: the same type, `repr(transparent)`, or a
slice under `TransparentOver`. `Cross::materialize` is `unsafe fn` with
the contract "erased from `Self`"; every caller states its proof, and
the extension crates contain no `unsafe`.

`Iter<T>` is a lazy `Value -> Value` pipeline of sealed stages: each
stage's only constructor is its ExternFn, which pairs source and
closure at one `T` before the stage is boxed; `T` is the declared type
and is never read from a `Value` except through `FromValue`.

## Rationale

Two representations of one type need one rule for where they meet.
Putting `#` on the slot keeps it structural (RFC-0022's `Vec<Value>` is
`Vec<T>` with a uniform slot) and lets the solver treat it as one more
component of a type; making it only by `Monomorphize` keeps the default
program exactly today's and confines the second box to fns that ask for
native layout (`&[f64]`, `Vec<T>` by value into a Rust API). Deciding
`ρ` by instance choice rather than by flow is what lets a `#` value
reach a generic-only fn through one erase instead of a mismatch, and
what lets a uniform value reach a member through one materialize.
`Erased` removes the reason an extension ever needed to name a Rust
`T` for a uniform value; `FromValue` and the `Small` tag make the one
remaining reinterpretation checked; sealing `Iter`'s stages closes the
one place a wrong source could have been paired with a closure.

## Not built

- A cast for a family nested under a non-family (`Option<Vec<ρ>>`);
  conversions at a `ContextBind` nested in a compound pattern, at `?`'s
  leaving type, at binary-operator operands; `Erased` for a derived
  `ExternType` (bound is `Stored`); an arena for element boxes; `Iter`
  without the stage box.
- `#` slots in a Space (`layout.rs` refuses them).

## Rejected

- `ρ` on a plain concrete signature and on `let` bindings, so that a
  value no `#` consumer touches is uniform from birth without a cast
  node. The checker is conservative: a value keeps the representation
  it is born with, and `#` arises only by conversion at a site whose
  instance demands it. The cost is a copy at that site; moving a
  representation beyond it is the optimizer's, later.

## Consequences

- `acvus-mir`: `TypeArg { repr, ty }` on `UserDefined` arguments and
  `Ref` targets; `Repr::{Uniform, Specialized, Var}`; `ReprOwner::
  {Signature, Local}`; `MismatchReason::ReprOpen`; `Conversion::
  {Identity, Cast, ThroughRef}`; `CastKind::ThroughRef`;
  `ConversionSite` kinds `Store | Return | Pattern`; `Unsettled::
  ConversionNeedsPlace`; display `#` / `#?`.
- `acvus-extern`: `TyArg::slot` / `SlotRepr`, `Spec<T>`,
  `CrossSpecialized`, family casts from the macro, `Erased`, `Inline`,
  `FromValue`, `TransparentOver`, `Runtime::{value_as_ref, value_as_mut,
  inline_ref, inline_mut, type_of, type_name_of}`, `Arr`
  `repr(transparent)`, `Cross::materialize` unsafe.
- `acvus-interpreter`: `Value::Small(Tag, u64)`, `Value::Ref`; a `Copy`
  type outside the `Inline` set is a `Large` box.
- `acvus-ext`: `Iter` as sealed stages in `iter.rs`; `split_str` returns
  `Vec<Erased<Rt, String>>`; `contains` on `Iter<Erased<Rt, T>>`; no
  `unsafe`.
