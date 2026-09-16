# RFC-0042: The solver separates equality from decision

Status: Accepted
Date: 2026-09-18
Extends: RFC-0017, RFC-0027, RFC-0037, RFC-0038, RFC-0040, RFC-0041

## Ruling

Four rules. **R1** Unification is the join of the type lattice, taken
where it is asked: `!` is the bottom at a value position only; inside a
constructor every argument is invariant; two `Object`s join to the
union of their fields and two `Enum`s to the union of their variants; a
side that must grow and has no variable (an extern's declared type)
cannot grow — for an `Object` that is the definite-assignment check's
report, for an `Enum` the checker's mismatch; a pattern may name fewer
members than its source. **R2** A decision is a position with more than
one admissible answer; it holds its answer set and only shrinks:
integer width (on the variable, as `TyVarBound::Integer`), effect
interval, instance, representation `ρ`, conversion. **R3** A decision
settles when one answer remains; open at the end it takes the least
element (`!`, the lower effect, `Uniform`, the generic instance,
identity) or `i64` for a literal; nothing left is an error naming the
position. `settle` is one fixpoint, idempotent, run once per body
before `freeze`. **R4** A conversion is a decision between two types at
a site, answered from the registry once both resolve; identity where
they agree; nothing else converts.

The body is checked in one pass that only creates variables, joins and
opens decisions (`check`); decisions query resolved terms inside
`settle` (`query`); `solve` closes what remains and only then renders
messages (`solve`). Check reads the solver only through
`shallow_resolve_ty`.

## Rationale

Every question the language asks after a join — width, instance,
effect, representation, conversion — used to be answered at the join
by its own device, with eight snapshot/rollback pairs where the answer
had to be guessed. One shape for a decision and one settlement remove
the devices, the polarity machinery and the mid-body settlement, and
make a message name the final type.

## Not built

`ρ` opened on a concrete signature; a global (non-greedy) choice of
representation; conversions at the sites listed in RFC-0041's "Not
built".

## Consequences

- `acvus-mir/src/solver.rs` rewritten: `Terms` union-find over type,
  effect, length, identity and representation; `join(a, b, Position,
  JoinKind, registry)`; `Decision::{Instance, Conversion}`, `Answer`,
  `Unsettled`; `settle`, `solve`, `freeze_ty`. `Polarity`, `lub`,
  `coerce`, `settle_choices`, `settle_pending_choices`, `as_generic`
  and every snapshot are gone. RFC-0040's `Solver::settle_pending_
  choices`/`settled_instance` are `Decision::Instance` and `answer(id)`.
- `acvus-mir/src/typeck.rs`: `flow`, `convert_at`, `solve_body`,
  `report_unsettled`, `resolve_conversions`; `Intrinsic` in `ir.rs`.
