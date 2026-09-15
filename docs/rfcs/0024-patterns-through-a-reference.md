# RFC-0024: A pattern matched against a reference binds references

Status: Proposed
Date: 2026-09-15
Supersedes: none

## Ruling

A match source of type `&T` is matched by the same patterns as a `T`. The
`&` is written once, on the source — `[a, b, ..] = &@items`, `Some(v) =
&@opt`, `{ name, } = &@user`, `"admin" = &@role` — and never inside the
pattern. Every name the pattern binds is a reference into the source:
`a: &Int`, `v: &T`, `name: &String`. A literal in the pattern is compared
through the operator rule (RFC-0020): `"admin" = &@role` is
`core::eq(&@role, &"admin")`. Nothing is moved out of the source, and the
source is a place lent for the match.

A match source of type `T` is matched as before: it is moved into the
match, and every binding owns its part.

## Rationale

Only a primitive copies and a context is never read out without being
assigned back (RFC-0018), so the match that every template writes —
`{{ "admin" = @role }}`, `{{ [a, ..] = @items }}` — must lend its source.
With the `&` on the source, the pattern language does not change: a
pattern already says which parts it names, and the source's type says
whether those names own or borrow. Writing `&` inside patterns would say
the same thing twice and open a way to write it inconsistently
(`&[a, b, ..]` binding owned parts of a borrowed list has no meaning
here).

A binding of type `&T` is used like any reference: `*a` for a primitive,
`&`-taking functions for the rest, `clone(a)` to own it.

## Not built

- No `&` in a pattern.
- No mixed mode: a source is lent whole or moved whole; no pattern borrows
  one part and moves another.
- No `&mut` match source. A place changed by parts is changed by `&mut`
  lends to functions, not by pattern.

## Consequences

- The type checker gives a pattern matched against `&T` the binding types
  it would give against `T`, each wrapped in `&`; a literal pattern against
  `&T` type-checks as an `eq` call.
- Lowering matches a `&T` source through the reference: the test
  instructions read through it, and each binding is a `Ref` into the
  source (a `Ref` whose target is the storage the source reference names,
  `RefTarget::Through`, with the part's path).
- A match arm body sees the bindings as references; a body that needs an
  owned part writes `clone(part)`.
