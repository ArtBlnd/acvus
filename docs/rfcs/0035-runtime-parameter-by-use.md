# RFC-0035: An extern fn takes its runtime only when it uses it

Status: Accepted
Date: 2026-09-16
Extends: RFC-0023

## Ruling

The runtime parameter of an `#[extern_fn]` is optional. When the first
parameter is `&R` and `R` is the function's parameter bounded by
`Runtime`, it is the runtime and is not part of the acvus type, as
RFC-0023 said; when the first parameter is anything else, the function
takes no runtime and every parameter is an acvus parameter. A function
that does not name `R` anywhere has no `R` at all.

    #[extern_fn(effect = pure)]
    fn trim(s: String) -> String

    #[extern_fn(effect = pure)]
    fn as_iter_vec<T, E, I, Rt>(items: Ref<Vec<T>, Rt>) -> Iter<Ref<T, Rt>, E, I, Rt>

    #[extern_fn(effect = pure)]
    fn get_vec<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>, index: i64) -> Result<Ref<T, Rt>, ExternError>

## Rationale

Of the extern fns in this repository, 123 took `_: &R` and read nothing
through it; 102 of those named `R` nowhere else and carried the
generic and its bound for the parameter alone. A parameter a function
must declare and may not use is the glue's convenience written into every
signature. The glue knows whether it passed a runtime; the signature is
the one place that knowledge was not.

## Not built

- No runtime taken in another position: it is first or absent.
- The `Journaled` methods keep their runtime parameters; they are a trait,
  not extern fns.

## Consequences

- `acvus-extern-macro`: `parse_params` reports whether the first
  parameter was the runtime; the generated call passes `__rt` only then.
- Every extern fn under `acvus-ext`, `acvus-ext-llm`, `acvus-ext-net`,
  and the tests that ignored its runtime no longer takes one.
