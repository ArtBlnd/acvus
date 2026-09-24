# The solver

`acvus-mir`'s type solver answers two different kinds of question, and the
split between them is the whole design.

**Equality now.** Two types that must be one type are made one type the
moment the checker says so, by a union-find over variables. This is the
join, R1.

**Decisions later.** A position with more than one admissible answer — which
signature a bare name is, which instance of an extern a call runs, which
conversion takes an argument to its parameter, what a reference to a place
names — is not answered where it is met. It is recorded as a `Decision`, it
narrows as the terms resolve, and it settles when one answer remains. This
is R2.

A body is therefore checked once (`fresh`, `unify`, `decide`), solved once
(`solve`), and frozen once. `solve` settles every decision it can, then
closes what is still open by its least element (R3), then settles again.
Anything that neither settled nor could take a least element is reported.

`Solver` owns the union-find (`Terms`), the open decisions, the
`TypeRegistry` that says which slots specialize and which conversions exist,
and the compilation's identity sources. `typeck.rs` drives it; `ty.rs`
holds the terms it works on.

## Variables and their bounds

A type is a `TyTerm` over a phase: `PolyTy` (a declaration's, with numbered
variables), `InferTy` (a body's, with solver variables), `Ty` (frozen, with
none). Five kinds of variable are solved, each with its own bound, each in
its own union-find arena inside `Terms`:

| Variable | Bound | What the bound says |
| --- | --- | --- |
| type | `TypeBound` (`Unresolved { bound: TyVarBound }`, `Resolved`, `Forward`) | `TyVarBound::Any`, `OneOf(shapes)`, or `Integer { signed, among }` |
| effect | `EffectBound` | `Range { lower, upper }` on the reissue chain (RFC-0013 rule 1), or `Bound` |
| length | `LenBound` | an array's length, `Unbound` until an element count fixes it |
| identity | `IdentityBound` | which source a value came from (RFC-0012); identities are invariant |
| representation | `ReprBound` | `Uniform`, or `Specialized` with the slot's whole `#` tree (RFC-0041, hash-types.md) |

`TyVarBound` is a lattice under `meet`. `Any` meets anything; two `OneOf`
sets meet at the shapes that unify pairwise, and an empty meet is a
refusal; `Integer` meets `OneOf` at the integer widths in both, and two
`Integer`s at the widths in both with `signed` the disjunction. A bound
travels with its variable and is checked once more where the variable
freezes (`TyVarBound::admits`).

## R1 — The join

`Terms::join(a, b, position, kind, registry)` is the one unification in the
solver. `a` is the value, `b` the type it flows into. The result is written
to the root variable of whichever side is a variable; where no join exists
the result is a `Mismatch` carrying a `MismatchReason`.

A variable nothing has bound takes the other side (`take_other`): another
variable by forwarding to its root, so the two stay one name for whatever
the root comes to hold; a term by binding, after the two bounds meet.

Structural types join by union, not by equality: an object met with an
object naming a field it did not have grows to carry that field. A declared
struct's field set is the exception — it is the type, and an object that
disagrees with it is refused (RFC-0042).

`Position::Value` carries `!`: `! ⊔ T = T`, so the variable that named `!`
comes to name `T` (`yield_bottom`, RFC-0038).

### The two `JoinKind`s

`JoinKind` decides one thing only: **whether this join may name a slot's
representation.**

- `Flow` — a value flowing into a position. A signature's representation
  slot that is still open is *undecided*, not disagreed, and `Flow` refuses
  to name it: the join returns `MismatchReason::ReprOpen`, which `typeck`
  turns into a conversion decision at the site (R4).
- `Decision` — a call settling on the signature it took. This join may name
  an open representation, because the decision it belongs to is exactly the
  thing that decides it. At the top of a decision's join the effect also
  runs both ways rather than one: the caller runs what the callee does
  (RFC-0046), where at a value position the value's effect is only at most
  what the position allows (RFC-0046 rule 7).
- `Pattern` — a pattern tested against its source. The source's structural
  type grows by what the pattern names, and a pattern naming fewer members
  than a ground source is within it (RFC-0024).

## R2 — Decisions

A `Decision` is a position with more than one admissible answer. The
checker opens one only where the answer is not already written down; where
the head is known it answers on the spot. Each kind, and what it waits on:

| `Decision` | Question | Waits on |
| --- | --- | --- |
| `Signature` | which of a bare name's declarations this call is (RFC-0043) | the arguments' heads |
| `Instance` | which instance of an extern the call runs (RFC-0019, RFC-0040) | the call's type, and the body's task |
| `Conversion` | which conversion takes `from` to `to` (RFC-0023) | both heads, and any open representation |
| `Lend` | what `&place` names, since no `&&T` exists (RFC-0029) | the head of `of` |
| `Capture` | how a lambda's body reads a captured name (RFC-0018) | the head of `of` |
| `Match` | how a pattern reads its scrutinee (RFC-0024) | the head of the scrutinee |

`settle` walks the open decisions until none makes progress. A decision
narrows when some candidates drop out, settles when one remains, and fails
when none does.

### Signature resolution (RFC-0043)

A bare name is a set of signatures: its own function if it has one, else
every namespace's declaration of it, plus any local binding of that name,
plus the machine coercions offered back (see below). `step_signature`
filters the options to those that still take the call, and settles when one
is left.

Four rules govern which argument may narrow the set:

1. **An argument whose type is known narrows.** `admits` says how a
   candidate takes it: `Direct`, `Converted` (a declared cast, RFC-0023),
   `Viewed` (a `&String` at a `&str` parameter, RFC-0062 rule 3), or
   `Refused`.
2. **An argument whose head the solve owns narrows nothing.** If the
   storage the argument lends has no head yet
   (`Solver::lends_an_unnamed_head`) and the candidate takes a run of that
   storage rather than the storage itself, then the argument's own type is
   evidence for neither `Direct` nor `Viewed`, and the head is the solve's
   to name, not this candidate's. Every site asks the one predicate:
   `admission_waits`, `join_unjoined`, `takes_signature`,
   `settle_slice_args`, `meet_settled_argument`.
3. **A settled candidate joins the arguments it held back.** `join_unjoined`
   joins each argument the decision left unjoined, unless rule 2 still
   holds at it, or the candidate takes it as a view (whose type stays what
   the caller wrote, RFC-0043 rule 5).
4. **Two candidates with no type in common are reported, and a fully
   qualified name is the answer.** `NoSignature` names the call as written.

## R3 — The least element

After `settle` makes no more progress, every decision still open is closed
by the least element of what it admits, in this order:

1. an integer width, if its bound has a default: `i64` where the bound
   admits `i64`, else the one width if exactly one remains, else the
   variable stays open and is reported — see the literal's width below;
   a text, where the bound is `String` or `str` and nothing else, as `==`
   against text leaves an open operand: `String`;
2. a representation: `Uniform`;
3. a pattern's mode, *before* a lend's — a binding closed to a value is what
   a later lend of that name lends, and closing the lend first would name a
   referent the pattern had not settled, which is how a `&&T` would be
   formed (RFC-0029);
4. a lend: a reference;
5. a capture: a word;
6. an instance: the one the body's task allows.

An open effect closes the same way: it is the least element of its
interval, the join of what the body requires (RFC-0013 rule 5).

`settle` runs again after each close. A decision that neither settled nor
could take a least element is an `Unsettled`, which `typeck` turns into a
refusal.

### An integer literal's width

An unsuffixed literal starts at a variable bounded `Integer { signed: false,
among: IntTy::ALL }` (`fresh_int_var`). Every use narrows that bound by the
`TyVarBound` meet: a second `Integer` leaves the widths in both, and a
`OneOf` — which is what a shared signature's instances come to (RFC-0019) —
leaves the integer widths the `OneOf` lists. A negation drops the unsigned
widths.

`integer_default` answers the width: `i64` where the narrowed set still
admits it, the only member where one remains, and otherwise **none**. None
is a refusal at the literal, naming the widths that are left
(`settle_int_literals`, `IntegerLiteralWidthUnsettled`): `8.is_power_of_two()`,
whose signature has instances at the unsigned widths only, is refused with
"an integer literal here may be u8, u16, u32 or u64: write a suffix".

Nothing downstream may treat that variable as settled. `freeze_ty` answers
`UnresolvedType` for it, and a caller that turns that answer into
`Ty::error()` is producing poison out of a program nothing refused — which
is how such a literal once reached a poison instruction with the machine
running.

## R4 — Conversions

A value that does not join its position may still reach it by a declared
cast (RFC-0023). `typeck::flow` joins first; a join whose one disagreement
is `ReprOpen` becomes a `Conversion` decision at the site, as does a value
met at a position it may need converting to (`convert_at`). The conversion
answers `Identity` exactly where its two sides join, and otherwise a
declared rule. No conversion answers a declared struct's field set: the
field set is the type (RFC-0042).

Through a reference, the value is cast back when the call ends (RFC-0041).

## Poison

`Ty::Error` is poison. It exists only where something was already refused,
and it carries that refusal forward so that nothing downstream is refused a
second time for a cause the reader has already been told about.

Three rules, and nothing else:

1. **Poison joins with everything, and binds.** A variable joined with
   poison is bound to poison. It does not stay open — an open variable
   refuses in its own words (`AmbiguousSignature`, `UnresolvedType`), which
   would be a second refusal for the first one's cause.
2. **Poison satisfies every bound.** `TyVarBound::admits` admits `Ty::Error`
   whatever the bound is, so freezing a poisoned variable is not a bound
   violation.
3. **A refusal about poison is not reported.** A call one of whose argument
   types is poison is not refused for having no matching signature
   (`no_matching_function`), and `typeck::reported` drops an error that is a
   consequence of an earlier one.

A component that is refused becomes poison and is carried into the
aggregate as poison (`as_data`); there is no second path that reports and
keeps the original type.

## `Never`

`Ty::Never` is the type of an expression that does not produce a value. At
a value position `! ⊔ T = T` (RFC-0038). A variable nothing constrained
closes to `!` (`close_ty`). A slot declared `!` states no type and so
accepts any value (RFC-0054) — which is why `types_match` answers `true`
for `(Never, _)` and there is no second arm for `(Never, Never)`.

## Machine coercions

Some declarations are not functions a script calls but coercions the
machine settles for instructions of its own: `a[i]` takes the container's
own `as_slice` (RFC-0047 rules 3 and 5), `for x in &v` the same, and a `&String`
argument reaches a `&str` parameter through `as_str` (RFC-0062 rule 3).

These live in `TypeEnv::machine`, which `TypeEnv::resolve_fn` does not
read, so a script's name resolution does not reach them.

**What makes a declaration one.** Its declaration says so. A registry marks
it `#[extern_view]`, `Externs::combine` reads the coercion off its type
(`Viewed::of_declaration` — a single reference parameter over storage handed
back as the run of that same storage at the same mutability: `&Vec<T>` to
`&[T]`, `&mut [T; N]` to `&mut [T]`, `&String` to `&str`), and the pair goes
into `TypeRegistry::machine_view`. A declaration whose shape is not a view
is refused there (`CombineError::ViewShape`), so the marker and the type
cannot disagree.

Neither the shape nor the name decides it. An extern
`fn view1(s: &String) -> StrView` has the shape and stays a function a
script calls; so does one spelled `as_str` that no registry marked. There is
no predicate over a name or a type that answers this question, and adding
one would put the answer back where a declaration cannot see it.
`Viewed::spelling` survives as the text a refusal prints, nothing more.

**How one is found.** `TypeEnv::machine` carries each declaration with the
`Viewed` its registry settled. `TypeEnv::machine_views` takes a `Viewed` — a
`View` (`Slice` or `Str`) at a `Mutability` — and returns every declaration
that is that coercion; `machine_coercions` returns all of them.
`check_index`, `check_for_slice`, `slice_coercion` and `signature_set` all
ask that way.

**What a script may write.** Every machine coercion is also offered back to
a script's own call as a candidate alongside `resolve_fn`'s
(`view_signatures`), so `v.as_slice()` and `s.as_str()` both resolve, as
they do in Rust. Being offered back is not the same as being resolvable:
`resolve_fn` still does not see them, so the offer is made at the call, by
the name the script wrote, and nowhere else.

## Names the language owns

Three kinds of name are written in `typeck.rs` as text, and they are not
special cases of any rule above: they are the language's own vocabulary,
which nothing in a registry could rename without changing the language.

- `core::clone` and `core::eq` — the shared signatures (RFC-0020) the
  compiler has instances of its own for. `compiler_instances` offers
  `Intrinsic::StringClone` at `core::clone`, and `check_operator_call`
  resolves `==` and `!=` to `core::eq` at every operand type.
- `Some`, `None`, `Ok`, `Err` — the variants of the two built-in enums
  (`resolve_builtin_variant`).
- `f64` and `char` — the primitive cast targets a script writes after `as`
  (`CastTy::of_name`).
- `break`, `continue`, `return`, `?`, and the operator spellings — the
  keywords a refusal prints back (`op_str`, `check_loop_jump`,
  `check_body_exit`).

These are output text and language vocabulary, not lookups into anything a
declaration could have said instead.

## Where the rules are applied

Every site in `solver.rs`, `typeck.rs`, `ty.rs` and `graph/infer.rs` that is
keyed by a name string, by a `Ty` shape outside the type's own match, or by
a local re-derivation of a fact another site owns.

Each row is an **instance of a rule** (rewritten as that rule's general
path), a **rule with no statement** (a section above now states it), or
**not a rule** (removed).

| Site | Was keyed by | Disposition |
| --- | --- | --- |
| `ty::is_machine_signature`, then `is_machine_coercion` | the names `as_slice`, `as_slice_mut`, `as_str`; then shape *and* those names | instance of Machine coercions: gone; `#[extern_view]` on the declaration, `TypeRegistry::machine_view` reads it back, and no predicate over a name or a type exists |
| `typeck::Viewed::declaration` | a three-entry name table, used to look a declaration up | gone; `Viewed::spelling` remains as the text a refusal prints |
| `typeck::check_index` | interning `"as_slice"` / `"as_slice_mut"` by mutability | instance: `machine_views(Viewed { Slice, mutability })` |
| `typeck::check_for_slice` | the same two names | instance: `machine_views(Viewed { Slice, mutability })` |
| `typeck::slice_view_signatures` | a guard on the names `as_slice`, `as_slice_mut`, with `as_str` held out | instance: `view_signatures` over `machine_coercions`, filtered by the name the script wrote; `s.as_str()` resolves as `v.as_slice()` does |
| `ty::TypeEnv::machine` | a map of schemes whose coercion was re-read from each one's shape | instance: `MachineCoercion` carries the `Viewed` its registry settled |
| `solver::admission_waits` | a local `Ref(_, Var)` test | instance of R2 rule 2: `lends_an_unnamed_head` |
| `typeck::meet_slice_parameter` | a local `referent is Var` test | instance of R2 rule 2: `lends_an_unnamed_head` |
| `solver::takes_signature` (`takes_unjoined`) | nothing — it did not ask, and eliminated candidates on an argument whose head the solve owns | instance of R2 rule 2: `admission_waits` |
| `solver::take_other` | returning `Ok` without binding when the other side was poison | instance of Poison rule 1: binds |
| `ty::TyVarBound::admits` | — | instance of Poison rule 2: admits `Ty::Error` whatever the bound |
| `typeck::no_matching_function` | — | instance of Poison rule 3: silent when an argument is poison |
| `typeck::reject_reference_in_data` | a second path for components "the solver never binds to poison" | gone; every component goes through `as_data` |
| `typeck::solve_body`'s literal loop | `let Ok(Ty::Int(k)) = freeze_ty(..) else { continue }` — a freeze failure discarded | instance of R3: `settle_int_literals` refuses, naming the widths |
| `graph::optimize::optimize` | a `_context_types` parameter no pass reads | not a rule; gone, with its four call sites |
| `graph::infer`'s three `context_extract_*` tests and four others | `let _ = infer(..)` | not a rule; each asserts what its name claims |
| `typeck::compiler_instances`, `check_operator_call` | `core::clone`, `core::eq` spelled as text | rule with no statement; now "Names the language owns" |
| `typeck::resolve_builtin_variant` | `"Some"`, `"None"`, `"Ok"`, `"Err"` | rule with no statement; now "Names the language owns" |
| `ty::CastTy::of_name` | `"f64"`, `"char"` | rule with no statement; now "Names the language owns" |
| `typeck::check_loop_jump` | `keyword == "break"`, re-derived from a `&'static str` the caller made out of `Stmt::Break` vs `Stmt::Continue` | rule with no statement; the distinction is a type the caller already had |
| `typeck::is_pair`, `solver::borrows_a_view` | `Ref(_, Str \| Slice(_))`, written twice; `acvus_interpreter::prepare::is_slice` is a third copy across crates | rule with no statement: a run is two adjacent registers and a reference to one is a pair |
| `validate::type_check::types_match` | `(Never, Never)` after `(Never, _)` | not a rule; gone, the first arm covers it |
| `validate::type_check::identities_match` | — | not a rule; gone, nothing called it |
| `solver::Identities` | a two-variant enum with one variant ever constructed | not a rule; gone, `instantiate_with` reads the declaration's parameters directly |
| `lower::param_slot`, `lower::extract_range_bounds` | — | not a rule; gone, nothing called them |

One row checked and dismissed: `ty::unify`'s `(Error, _) => false` reads
like the opposite of `solver::join`'s `(Error, _) => Ok(())`, but it is
`PatternUnifier::unify` over two declaration patterns, not the join over a
body's terms, and a declaration pattern is never poison.
