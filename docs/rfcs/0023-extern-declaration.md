# RFC-0023: Declaring an ExternFn

Status: Accepted
Date: 2026-09-15
Supersedes: RFC-0009

## Ruling

An ExternFn is declared once, as a Rust function under `#[extern_fn]`;
its acvus type and its handler both come from that signature.

    #[extern_fn(effect = pure)]
    fn len<R>(_: &R, s: &String) -> i64 where R: Runtime

- The first parameter is the runtime, `&R` with `R: Runtime`; it is not
  part of the acvus type.
- A parameter marked `#[state]` is not part of the acvus type either: its
  value is supplied when the registry is built and the handler holds it
  (RFC-0021).
- Every other parameter is an acvus parameter. `T` names the acvus type
  `T` through `TyArg`; `&T` names `&T` and `&mut T` names `&mut T`
  (RFC-0018). The return type names the acvus return type; `Result<T, E>`
  means `T`, with `E: Into<R::Error>`.
- The acvus name is the Rust identifier unless `name = "..."` says
  otherwise; the namespace is the registry's.
- `effect = pure | idempotent | opaque | <effect variable>`; undeclared is
  Opaque. `commutative` marks a commuting effect (RFC-0013).
- `instance_of = sig` declares an instance of a shared signature
  (RFC-0019).

Generic parameters are the declaration's variables, one bound each: `T:
TyVar`, `E: EffectVar`, `N: LenVar`, `I: IdentityVar`, `R: Runtime`. A type
variable may add `Monomorphize<(T0, ..)>` (RFC-0011) or `HasInstance<sig>`
(RFC-0019). A body never opens a type variable; it crosses through the
runtime (RFC-0022).

Positions acvus has and Rust does not are spelled by host types:
`Arr<T, N>` for an array of variable length, `Fn0<R, E, Rt>` ..
`Fn3<A, B, C, R, E, Rt>` for a closure parameter, `Ref<T>` / `RefMut<T>`
inside a closure's parameter list, and `Pure` / `Opaque` / `()` where an
effect, length, or identity argument is fixed.

## Rationale

RFC-0009 named the interner as the first parameter and made `TyVar` a
conversion pair; the runtime contract (RFC-0022) made the first parameter
the runtime and the crossing a pair of unsafe functions on it, and
references (RFC-0018) made `&T` a type of its own. The declaration form
follows those rulings and adds nothing beyond them: state (RFC-0021) and
instances (RFC-0019) are attributes on the same form, not second forms.

## Not built

- No closure-declared ExternFn: state is a parameter, not a capture.
- No parameter mode beside the type.

## Consequences

- The macro reads `#[state]` off a parameter, wraps the value in an `Arc`
  held by the handler, and passes `&State` to the body.
- The macro reads `HasInstance<sig>` into the declaration's requirement
  list and `instance_of` into its instance link.
