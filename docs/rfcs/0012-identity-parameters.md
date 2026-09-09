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

A user-defined value with an identity parameter is one source and moves;
a user-defined value without one is a plain value and copies. This is the
only rule that decides whether a user-defined type moves.

Identity lives in the compiler. The runtime sees none of it: what the
compiler proved about sources is already spent by the time code runs. A
rejoin of parts is a rewrite the compiler applies at the operation that
split them, where it can see their order; a shared identity is a
precondition of that rewrite and never its justification.

## Rationale

The previous design carried identity as a type of its own that could sit
in any position, including a struct field, and paired it with a runtime
notion of provenance it never built. Nothing produced such a type outside
tests, and every user-defined type was move-only by fiat, which made a
regular expression unusable twice. Making identity a parameter kind puts
it where effect already lives, gives the move rule a declared source, and
leaves the compiler with exactly the three variable kinds it already
solves in the same shape.

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

## Open questions

- Whether a local function's fresh identities should be generalized so
  that each call is a new source.
- Whether a declared join with identity erasure is ever needed, or
  functions that declare a new source always suffice.
