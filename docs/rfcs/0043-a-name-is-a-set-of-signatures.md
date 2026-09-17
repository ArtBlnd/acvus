# RFC-0043: A bare name is a set of signatures

Status: Accepted
Date: 2026-09-17
Extends: RFC-0021, RFC-0027, RFC-0040, RFC-0042

## Ruling

A bare name that several namespaces declare resolves to the set of their
signatures. Candidates whose arity differs from the call are dropped at
check time; one left is the call, as a qualified name is. Several left
form one call: a fresh function type whose parameter variables carry
`TyVarBound::OneOf` of the union of the candidates' parameter shapes at
that position, a candidate's own `OneOf`-bounded variables expanded to
their shapes, a bare variable making the position `Any`; its return is
the candidates' common return pattern (`get`'s `&T`, `first`'s
`Option<&T>`) with fresh variables, or a bare variable where they have
none (`min`'s `T` and `Option<T>`), so a field read or a method call on
the result sees the reference before the signature settles. A receiver
whose type is still a variable is passed as it is; the parameter it meets
decides whether it was a reference. An argument
flowing into such a parameter meets the bound as any bounded variable's
does: two variables intersect their bounds; an argument some shape admits
joins the parameter; an argument no shape admits, but one declared
conversion rule (RFC-0023) takes to a shape of the bound, leaves the
parameter open and opens the conversion decision at that argument, as a
direct call would; and an argument neither admits empties the set at that
argument, which is reported as `NoMatchingFunction` there and opens no
decision. An argument whose type is still a variable is admitted as it
is; a conversion is admitted from a resolved head only.

The call opens `Decision::Signature` over the candidates, stepped as
`Decision::Instance` is (RFC-0040, RFC-0042): a candidate stays while the
call type would join its type on a copy of the terms, at every parameter
the call still has as a variable the variable's bound meets the bound the
candidate's parameter gives it, and at every argument admitted through a
conversion, one declared rule takes the argument to the candidate's
parameter or the argument joins it. One left settles: the candidate is
instantiated as a call of it — its own instance decision and bounds
opened, as at a qualified call — and joined with the call type, which
resolves the parameter the conversion decision waits on;
the answer names the function, its instance choice, and its bounded
variables, which the checker verifies when the body freezes. None left is
`NoMatchingFunction { name, call type }`. More than one left when the body
is solved is `AmbiguousFunction`, listing the candidates that remain.
Nothing is defaulted.

The lowering reads a decided call's callee through the settled answer,
then its instance, as it reads a resolved call's; the IR has no new shape.

A method call lends its receiver when every remaining candidate's first
parameter is a reference of one mutability, and passes it as a value
otherwise; a pipe passes its left side as a value.

A conversion decision whose one side is a `OneOf`-bounded variable answers
identity only where the other side could match a shape of the bound: the
join itself binds such a variable to any term and verifies the bound when
the variable freezes.

## Rationale

A lambda's parameter has no type when the call inside it is checked:
`|c| -> contains(c, 1)` says nothing about `c` until the lambda meets a
consumer, or another use of `c` does. A dispatch on the first argument's
syntax or on the checked type at the call would have to decide there, and
would decide wrong or refuse. A decision that shrinks as the terms
resolve is what the solver already has for instances (RFC-0040) and
conversions (RFC-0023); a signature is one more position with several
admissible answers, and settles by the same fixpoint.

The bound on the fresh parameter is what makes a lambda variable's type
the intersection of what its uses admit: each use meets the variable's
bound with its own, and the candidate set follows the bound. A use that
binds the variable to a term decides by shape, as `len(c)` does with its
reference parameter.

Two layers, not one: the generic instance of RFC-0040 is a fallback inside
one signature, taken when every concrete instance is excluded. Between
signatures there is no fallback — the candidates do not overlap by design,
they are different functions — so a signature decision has no least
element, and what stays wide stays undecided.

## Cost

A call with several same-arity candidates is checked at the union bound,
so a mismatch inside it is reported when the decision settles or fails,
not at the argument, except where the argument meets no shape of the
union at all. The report names the call type as far as it is known.

A receiver whose candidates disagree on lending is passed as a value; a
candidate wanting a reference then drops unless the receiver is already a
reference value.

Each step of a signature decision joins the call type against every
remaining candidate on a copy of the terms; the cost is that of an
instance decision per candidate, plus a rule lookup per converted
argument on the same copy. A chain of two conversions is not searched:
one declared rule, or none.

A settled signature is a function whose instance may still be open:
`iter::contains` settles on an iterator argument while its `Monomorphize`
member is chosen by the element type, so a call whose element stays open
has no callee (RFC-0040). A lambda parameter used only as a container —
`|c| -> contains(c, 1) && len(c) > 0` — stays ambiguous among the
container namespaces, and the message names the candidates in sorted
display order, not the order the registries were combined in.

## Rejected

- Unique bare names with suffixes (`contains_iter`, `min_of`): the names
  the standard library uses are the names a script should write, and a
  suffix moves the decision from the checker to the reader.
- A qualified path at every use (`std::min`, `container::last`): correct
  and unambiguous, and the reason RFC-0021's one-name rule was written;
  it makes the common call the verbose one and is kept as the escape
  hatch, not the rule.
- Dispatch on the first argument's syntax (`&x` selects the reference
  candidates): decides before a lambda parameter has a type.
- One combined signature with a single `OneOf` over the candidates'
  parameter types and no decision: it cannot name which function runs,
  and RFC-0040 made the instance a compile-time fact.

## Consequences

- `acvus-mir/src/ty.rs`: `FnLookup::Overloaded(Vec<(QualifiedRef,
  &Scheme)>)`; `Scheme::{params, ret, param_bound}`; `TyVarBound::union`.
- `acvus-mir/src/solver.rs`: `Decision::Signature` with its
  `ConvertedArgument`s, `SignatureCandidate`,
  `Answer::Signature(SettledSignature)`, `Unsettled::{NoSignature,
  AmbiguousSignature}`, `step_signature`, `Solver::{would_unify,
  fresh_var_with, admit_argument}` with `Admission`, and
  `identity_within_bounds` on a conversion decision.
- `acvus-mir/src/typeck.rs`: `check_overloaded_call`, `CalleeChoice`,
  `ArgumentMismatch`, `CheckedArgs` with its converted positions,
  `receiver_arg`; `report_unsettled` maps the two failures; `solve_body`
  verifies a settled signature's bounds.
- `acvus-mir/src/error.rs`: `MirErrorKind::NoMatchingFunction`.
- RFC-0021's one-name rule is amended: a name two namespaces declare is a
  set, decided by this RFC.
- A library function that exists per container type is a plain function
  in that type's namespace (`vec::len`, `string::len`, `string::contains`
  beside `iter::contains`), and the bare name is this RFC's set
  (RFC-0028). `core::` — `clone`, `eq`,
  `hash`, `to_string`, `to_int` — remains the shared-signature mechanism:
  one signature, an instance per type, which a generic function asks for
  by name.
