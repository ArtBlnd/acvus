# RFC-0024: A pattern matched against a reference binds references

Status: Accepted
Date: 2026-09-15
Supersedes: none

## Ruling

A pattern lives in one of two dimensions, and the dimension is the
source's, never the pattern's.

- **Value.** A pattern matched against a value reads each part it names
  as the part's type reads — a word copies, a `String` copies
  (RFC-0026), anything else moves out. A value with a part moved out is
  partly moved: its other parts may still be read, and a use of it whole
  is a use after move.
- **Reference.** A pattern matched against a `&T` is the same pattern
  matched against `T`, and every name it binds is a `&part`:
  `[a, b, ..] = &@items` binds `a: &Int`; `{ name, } = &@user` binds
  `name: &String`; `Some(v) = &@opt` binds `v: &T`. The source is lent
  for the match and nothing moves. A literal in the pattern is compared
  through the reference.

- **An open head.** The dimension is the source's, so a source whose type
  is still a variable where the pattern is written has no dimension yet.
  The pattern is checked against a referent of its own, and the two are
  joined when the head resolves: a reference reads through it, anything
  else reads the value, and every name the pattern binds takes its type
  from that same answer. A source nothing ever makes a reference is not
  one, so a head that stays open to the end reads the value — the least
  element of the two, not a default. In
  `let f = |r| -> { 1 = r { … }; … }`
  the literal is compared through the reference when the call lends `f`
  its argument, and against the value when it hands one over.

The `&` is written on the source and nowhere else: there is no `&[a, b]`
and no `{ &name, .. }`. A binding through a reference is used as any
reference: `*a` for a word, `{{ name }}` emits a `&String`, `clone(name)`
owns it.

## Rationale

The two dimensions are the ones RFC-0018 already gives every expression:
a place used as a value copies or moves by its type, and `&place` is a
reference. A pattern is several such reads at once, so it takes the
dimension of the one expression it is applied to. Letting a binding pick
its own dimension (`{ &name, .. }`) or a pattern lift its source
(`&[a, b]`) would put two dimensions in one expression and make the
source's dimension depend on what the pattern says.

Binding `&Int` rather than copying the word is the price of one
dimension; `*a` names the copy. Copying words through a reference and
borrowing the rest would be a third rule keyed on the type, and the
owner declined it.

## Not built

- No `&` in a pattern.
- No `&mut` match source. A place changed by parts is changed through
  `&mut` lent to functions, not by pattern.
- No context bind (`@x = ...`) through a reference: a context holds data,
  never a reference (RFC-0014).

## Consequences

- A tag-form body's statements are the script's statements (RFC-0045):
  `let x = …;` introduces a binding that ends with the body, and `x = …;`
  assigns the binding the enclosing block introduced, which stays assigned
  after the match. `let out = 0.0; Some(v) = Some(1.5) { out = v; }; out`
  is `1.5`, and the payload no longer has to leave through a context.

- `PathSeg::{Field, Index, Payload}` replaces the field-name path of
  `Ref`/`Take`/`Assign`, so a reference can name an array element, a
  tuple element, or a variant's payload.
- The type checker checks a pattern against `&T` as against `T` with
  every binding wrapped in `&`; a context bind there is a
  `ReferenceInData` error.
- A head the checker cannot read is a `Decision::Match`, which carries
  the pattern's referent and every name it binds. Settling it joins the
  referent with what the head names and gives each name its type —
  `&part` through a reference, `part` on a value — and a context bind
  under a head that settles on a reference is the same `ReferenceInData`
  error. A match closes on its least element before any lend closes, so
  a name closed to a value is what a lend of that name then lends, and no
  `&&T` is formed (RFC-0029).
- The lowering reads the frozen type of the source, which the settled
  mode agrees with by construction, so the mode is not recorded twice.
- Lowering, on a `&T` source, makes a `Ref` through the source for each
  part the pattern names and matches the sub-pattern against that `&part`;
  `TestLiteral`, `TestVariant`, and `TestObjectKey` read through a
  reference source.
