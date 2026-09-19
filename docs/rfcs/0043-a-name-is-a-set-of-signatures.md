# RFC-0043: A bare name is a set of signatures

Status: Accepted
Date: 2026-09-17
Extends: RFC-0021, RFC-0027, RFC-0040, RFC-0042

## Ruling

A bare name that several namespaces declare resolves to the set of their
signatures. Candidates whose arity differs from the call are dropped at
check time; one left is the call, as a qualified name is. Several left
form one call: a fresh function type whose parameter variables carry
`TyVarBound::OneOf` of the union of the parameter shapes the candidates
still in the set have at that position, a candidate's own `OneOf`-bounded
variables expanded to their shapes, a bare variable making the position
`Any`; its return is the candidates' common return pattern (`get`'s `&T`,
`first`'s `Option<&T>`) with fresh variables, or a bare variable where
they have none (`min`'s `T` and `Option<T>`), so a field read or a method
call on the result sees the reference before the signature settles. That
bound is the parameter's range and nothing more: no argument is admitted
by reading it. A receiver that is not a place and whose type is still a
variable is passed as it is; the parameter it meets decides whether it
was a reference.

How an argument is taken is a property of the pair (argument, candidate),
asked of that candidate's own parameter there. A candidate takes an
argument one of its parameter's shapes admits directly, joining the call's
parameter with it. An argument no shape admits, but one declared rule
(RFC-0023) casts to a shape — through a reference, one rule each way,
since the value is cast back when the call ends (RFC-0041) — it takes by
conversion. Any other it refuses, and a candidate that refuses an argument
leaves the set there, as the arity filter leaves one out at the call. An
argument still a variable is admitted as it is where the two bounds
intersect; a conversion is admitted from a resolved head only. A set an
argument empties is `NoMatchingFunction` there, and opens no decision.

**Admission is ordered by evidence** (amended 2026-09-20, with RFC-0062).
An argument is taken by one of four admissions, and they are ordered:
`Direct` — the argument's own type is one of the parameter's shapes;
`Converted` — one declared rule (RFC-0023) casts it there; `Viewed` — the
argument is a borrow of a storage and the parameter takes a view of that
storage (`&v` at `&[T]`, RFC-0047; `&s` at `&str`, RFC-0062), recorded as
a coercion at the argument and not unified with the parameter; `Refused`.
The rules that follow hold for `Converted` and `Viewed` alike, so that no
kind of admission behaves as an exception:

1. **Direct first.** At an argument, if any candidate takes it directly,
   every candidate that would take it only by conversion or by view
   leaves the set there. The argument's own type is the evidence; a
   weaker admission would change the value the callee sees.
2. **A weaker admission needs a resolved head.** A conversion or a view is
   admitted only from an argument whose head is resolved. An argument
   still a variable is admitted directly where the two bounds intersect
   and by nothing else: it narrows the set toward no converted or viewed
   candidate. The decision waits for the head (the deferral RFC-0047 uses
   for a container whose element type is still open); when the variable
   freezes to a head no remaining candidate takes directly, admission is
   asked again at that argument with the resolved head, and rule 1
   applies to what it finds.
3. **Equal strength is the existing question.** Two candidates that take
   an argument at the same admission are told apart by the rest of the
   call, as before; two that remain at the end are `AmbiguousFunction`.
   Two viewed candidates are no different.
4. **A receiver is an argument.** A method call's receiver is admitted by
   the same four admissions and the same three rules, in the mode each
   candidate sees it in (below); a receiver that is a variable is admitted
   directly where bounds intersect and by no view until it resolves.
   A local binding is a candidate like any other under these rules: where
   it takes an argument directly, a signature that would take it by
   conversion leaves the set (settled 2026-09-20; the earlier text that
   named `let count = |k| -> 7.0; count(q)` ambiguous is superseded).

5. **The view is the checker's, the parameter's type is the callee's.**
   Where a viewed candidate is settled on, the checker records the view
   at the argument (`CastKind::Slice`, `CastKind::Str`) and the call's
   parameter is the view's type; the argument's own type is untouched,
   so a later use of the same place sees what it was.

At an argument some candidate that stays takes only by conversion, the
call's parameter is left open and one conversion decision (RFC-0023) from
the argument to it is opened, as a direct call would. It has no answer of
its own: settling it to identity would bind the parameter and so decide
the signature by the argument's shape. It waits for the signature and
settles once that has — identity where the settled candidate took the
argument directly, the rule where it took it by conversion.

The call opens `Decision::Signature` over the candidates, stepped as
`Decision::Instance` is (RFC-0040, RFC-0042): a candidate stays while the
call type would join its type on a copy of the terms, at every parameter
the call still has as a variable the variable's bound meets the bound the
candidate's parameter gives it, and at every argument that candidate takes
by conversion, one declared rule takes the argument to its parameter or
the argument joins it. One left settles: the candidate is instantiated as
a call of it — its own instance decision and bounds opened, as at a
qualified call — and joined with the call type, which resolves the
parameter the conversion decision waits on; the answer names the function,
its instance choice, and its bounded variables, which the checker verifies
when the body freezes. None left is
`NoMatchingFunction { name, call type }`, the call type shown as written
with an open variable closed to `!`. More than one left when the body
is solved is `AmbiguousFunction`, listing the candidates that remain.
Nothing is defaulted.

A type any report shows is the type as written, every variable nothing
resolved closed to `!` — whatever bound that variable carries, since a
bound is the variable's range and not a type it reached. `<error>` in a
message names an `ErrorToken`, a subexpression that already failed to
check, and nothing else: a report never renders a type the checker
merely had not settled. What a resolution carries into lowering is the
stricter freeze, which refuses a variable a bound left open, because no
shape of a bound is the program's.

The lowering reads a decided call's callee through the settled answer,
then its instance, as it reads a resolved call's; the IR has no new shape.

A binding of a name is one more signature of it. The candidates at a call
of a bare name are the namespaces' signatures and the binding of that name
in scope where the call stands, whatever its type's head — a `let` of a
lambda, a `let` of a number, a lambda's own parameter before anything
fixes it. The binding is decided as the rest are, and its head is what the
decision decides: the arity filter counts the parameters of a head already
`Fn` and keeps a head still open, whose arity the call fixes; the binding
stays while the call type would unify with its type, which a head still
open does and a head resolved to a non-`Fn` (`let len = 3`) does not; one
left settles by joining the call type with it, which for a head still open
is what fixes the binding's type to the call's `Fn`. So in `let f = |len|
-> len(1)` no declared `len` takes `(i64)`, the parameter is alone, and
the body's call gives it a function type taking an `i64`; `f(3)`, fixing
it to `i64` instead, drops it — leaving no `len` that takes the call.
`AmbiguousFunction` lists the binding as ``the binding `len` ``. Settling
on it names no function, so the call freezes no direct callee and
the lowering takes the indirect path off the variable, as a call of a name
no namespace declares already does.

A binding's parameters are inference types, not shapes a scheme names, so
it gives the call's parameters `TyVarBound::Any` and contributes no return
pattern. `Any` is what the union of it with any `OneOf` is, so in a set
holding a binding every parameter is `Any` and the return is a bare
variable. A binding takes every argument directly, so rule 1 makes it the
argument's strongest admission: a declared signature that takes the same
argument only through a cast leaves the set there. `count(q)` over an owned
array is `iter::count`, through `into_iter_array`; `let count = |k| -> 7.0;
count(q)` is the binding, whose parameter takes the array as it is, and
`count(&q)` is the binding alone, no rule reaching an iterator from a
reference.

A method call's receiver is one more argument, admitted per candidate
before the mode the call takes is fixed, each candidate seeing it in its
own mode: a candidate whose first parameter is a reference sees the
receiver lent, of that parameter's mutability; a candidate whose first
parameter is a value or a variable — a binding's is one, or its head is
still open and names no parameter at all — sees the place by value. A
candidate that refuses the receiver in its mode leaves the set there, as
a refusal at any argument leaves it. Narrowing at the receiver is
admission and nothing more: the decision's own predicate is what the
decision applies, over the call's parameter once the arguments have met
it, and a receiver the checker sees as a variable that resolves to the
receiver's type carries no evidence a join could read.

The mode the call takes is the one the candidates that stay see, told
apart by the type each sees and not by the form of its parameter. Two
modes that yield one type are one mode — a receiver that is already a
reference has a reborrow and a word copy of the same `&T`, and there is
nothing to choose between — and that one mode is the lend, the type being
the reference already, so a candidate settled on later takes exactly the
type its own mode showed it. Candidates whose types differ have no mode
in common, and that is `AmbiguousFunction` at the call over them:
choosing between a lend and a move would be a default, and there is none
here. A pipe passes its left side as a value.

Telling the modes apart needs a receiver type that distinguishes them. A
receiver that is not a place has no lend to be admitted in, and a place
whose head is still a variable has no lend to read off it, so in neither
is any candidate narrowed and the set's agreement alone decides: lent
where every candidate's first parameter is a reference of one mutability,
by value otherwise.

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

An argument is matched against each candidate's shapes at it, so an
argument no candidate takes is reported at the call, while a disagreement
under a shape some candidate does take is reported when the decision
settles or fails. The report names the call type as far as it is known.

A receiver two candidates see as different types is reported, not
resolved: the call is written `array::len(&q)` or the binding is renamed.
The cost is a lend trial per distinct mutability among the candidates,
read off the receiver's type without the borrow's bookkeeping, which is
applied once for the mode the call takes. A set the receiver leaves with
one mode and several candidates is narrowed by the decision, as any call
of the bare name is, and reported there.

Checking an argument costs a shape match per candidate, and a rule lookup
on a copy of the terms per candidate no shape of which admits it. Each
step of a signature decision joins the call type against every remaining
candidate on a copy of the terms; the cost is that of an instance decision
per candidate, plus a rule lookup per argument that candidate converts on
the same copy. A chain of two conversions is not searched: one declared
rule, or none.

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
- `acvus-mir/src/solver.rs`: `Decision::Signature` over
  `SignatureOption`s — a `SignatureCandidate` with the
  `ConvertedArgument`s it takes by conversion —
  `Answer::Signature(SettledSignature)`, `Unsettled::{NoSignature,
  AmbiguousSignature}`, `step_signature`, `Solver::{would_unify,
  fresh_var_with, admits}` with `Admission`, the `converts` predicate, and
  `identity_within_bounds` and `awaits_signature` on a conversion
  decision.
- `acvus-mir/src/solver.rs`: `Open::AsWritten` and `Solver::written_ty`,
  the freeze a report uses; `Solver::close_ty` stays the resolution's.
- `acvus-mir/src/typeck.rs`: `TypeChecker::type_as_written`, which every
  report's type goes through, and `closed_or_reported`, which every type
  the resolution carries goes through; `check_overloaded_call` over `admit_args`,
  `call_param`, `admit_arg` and `no_matching_function`, `CalleeChoice`,
  `admit_receiver` over `CandidateReceiver`, `receiver_as`,
  `one_receiver_mode`, `receiver_in` and `receiver_arg`, with `lend_place`
  split out of `check_borrow` so the receiver's place is checked once and
  lent only in the mode taken;
  `report_unsettled` maps the two failures; `solve_body`
  verifies a settled signature's bounds.
- A binding joins the set through `typeck.rs`'s `signature_set` over
  `local_signature`, with `SignatureCandidate::{Named, Local}`,
  `SettledSignature::{Named, Local}`, `SignatureName` and
  `shown_candidates` for the message, `Solver::receiver_mode` answering
  `ReceiverMode`, and `check_local_call`, which records the callee
  type and freezes no `direct_calls` entry. `SignatureCandidate::arity`
  answers `Option<usize>`, `None` where the binding's head is still open,
  and `takes_arity` keeps such a candidate; `call_param` names the call's
  parameter after the first candidate that names it, and after its own
  index where none does.
- `lower.rs`: `lent_closure` lends the binding a call settled on, at a
  named call and at a method call alike, so `q.len()` settling on a
  binding lowers to `Callee::Indirect` of it.
- `acvus-mir/src/error.rs`: `MirErrorKind::NoMatchingFunction`.
- The admission order: `Admission::{Direct, Converted, Viewed, Refused}` and
  `Solver::{admits, admission_waits}`; `typeck.rs`'s `admit_arg` applies rule
  1 over the whole set at each argument, and records an argument whose
  admission waits for a head as `UnjoinedArgument` on the decision instead of
  joining it with the call's parameter. `Solver::admitted_again` asks
  admission afresh at every step and applies rule 1 to what it finds;
  `takes_signature` tests each unjoined argument against the candidate's own
  parameter on the trial terms; `join_unjoined` joins, at the settle, the
  unjoined arguments the settled candidate takes directly. `SliceArg`'s
  `DeferredView::OfSettledParam` is where the view a settled parameter asks
  for becomes the coercion.
- A `Converted` argument needs no deferral of its own: the conversion
  decision `convert_argument_at` opens already waits for the signature,
  through `Solver::awaits_signature`.
- A view at an argument is reachable in every registry set: `slice_coercion`
  resolves the declaration out of the environment's `machine_set`, and
  `core::as_str` is carried by `core_registry`, which `Externs::combine`
  prepends to whatever it is given.
- RFC-0021's one-name rule is amended: a name two namespaces declare is a
  set, decided by this RFC.
- A library function that exists per container type is a plain function
  in that type's namespace (`vec::len`, `string::len`, `string::contains`
  beside `iter::contains`), and the bare name is this RFC's set
  (RFC-0028). With a string literal a `&str` (RFC-0062), the admission
  order reaches `string::contains` in every form its name is written —
  `contains(&s, "x")`, `contains("abc", "b")`, `s.contains("x")` and
  `"abc".contains("b")` — because the argument that decides between it and
  `iter::contains` is a view at the first parameter either way. `core::` — `clone`, `eq`,
  `hash`, `to_string`, `to_int` — remains the shared-signature mechanism:
  one signature, an instance per type, which a generic function asks for
  by name.
