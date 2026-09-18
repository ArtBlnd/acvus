# RFC-0012: Identity as a parameter of a user-defined type

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

Identity is a kind of type parameter, like effect. A user-defined type
declares whether it has an identity parameter, at most one: a value is one
source. A type expression fills it with an identity argument; nothing else
in the type language carries an identity. There is no identity type, and
no structural type is tagged. Two values may each carry their own source,
as a closure's captures do; one value never carries two.

An identity argument names a source, and says nothing more: two parts of
one source share its identity, and the identity does not say which part is
which. Two identities unify only when they are the same, so a value from
one source never mixes with a value from another. A derived value keeps
the source it came from or is a new source; no operation takes a value
back to the sources it was made from. An identity variable that nothing ties to a source is a source of
its own when it freezes: an ExternFn that returns a type with an identity
variable found nowhere in its parameters returns a new source at every
call, and one that carries the variable from a parameter to its return
returns the same source it was given.

A user-defined value with an identity parameter is one source. Every
user-defined value moves, with or without an identity parameter: only
the machine's word primitives — the integer widths, `f64`, `Bool`,
`Unit` — copy (`move_check`; owner, 2026-09-20). The identity parameter
decides sameness, not whether the value moves.

Identity lives in the compiler. The runtime sees none of it: what the
compiler proved about sources is already spent by the time code runs. A
rejoin of parts is a rewrite the compiler applies at the operation that
split them, where it can see their order; a shared identity is a
precondition of that rewrite and never its justification.

## Rationale

Identity as a type of its own sits in any position, including a struct
field, and needs a runtime notion of provenance to mean anything there; and
with every user-defined type move-only, a value such as a regular expression
cannot be used twice. As a parameter kind, identity sits where effect already
sits, the move rule has a declared source to read, and the compiler solves
identity variables in the shape it solves type, effect, and length
variables.

## Not built

- No type with two identity parameters. Remembering two sources in one
  value would only serve taking the value apart into them again, and that
  is the operation that can put parts back in the wrong order.
- No erasure of identity at a join. Two values with different sources do
  not unify and no least upper bound forgets the difference; a script that
  wants both in one place joins them through a function that declares a
  new source, such as `chain`.
- No runtime provenance. Rejoin, when built, is code the compiler emits.
- No generalization of a local function's returned identities. A local
  function that returns a source returns the same source at every call.

## Consequences

- A user-defined type's declaration carries `identity_params`; its type
  carries `identity_args`; the polymorphic phases carry identity variables
  next to type, effect, and length variables.
- An extension type declares an identity parameter as a generic bounded
  by `IdentityVar`; its runtime conversion refuses a shared payload exactly
  when it has one.
- Serialized types carry identity arguments as their source numbers.
- A source number names one source for a whole compilation: every
  solver a compilation runs mints from one `Sources`, so a frozen type
  that passes from one solver to another (an earlier SCC's result, a
  cached signature) still names the source it was frozen with, and no
  later solver mints that number again.
- A declaration names no source. A host declaring a context or a
  parameter from a concrete type lifts it with `lift_declaration`, which
  turns every identity argument into a variable the compilation mints a
  source for; a source number carried in from outside would collide with
  the compilation's own.

## Open questions

- Whether a local function's fresh identities should be generalized so
  that each call is a new source.
- Whether a declared join with identity erasure is ever needed, or
  functions that declare a new source always suffice.
