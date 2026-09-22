# RFC-0042: The solver separates equality from decision

Status: Accepted
Date: 2026-09-18
Extends: RFC-0017, RFC-0027, RFC-0037, RFC-0038, RFC-0040, RFC-0041

## Ruling

Four rules. **R1** Unification is the join of the type lattice, taken
where it is asked: `!` is the bottom at a value position only; inside a
constructor every argument is invariant; two `Object`s join to the
union of their fields and two `Enum`s to the union of their variants.
A value the body constructs — an object literal, a structural variant —
has a type variable of its own, and a flow or a decision that joins two
such variables makes them one: a member one gains, the other has, and a
construction is laid at the union with the members it did not write
undefined (RFC-0050 rule 8). A type no variable of the body names — a
context's, an extern's parameter or result, a declared struct — has a
layout the body did not choose: a variable joined to one takes it as it
is, and a join that would have it gain a member is refused by that
member. A read's object and a pattern are lower bounds and are not
refused for lacking one; a pattern may name fewer members than its
source. A value may lack a field only while it moves between the body's
storages and registers: a read of a field takes that field, and every
other use takes the value whole, so an element, a payload, an argument,
a return and a commit are whole (the definite-assignment check). An object type carries which field set it has:
the fields a struct declares, under the struct's name; the fields an
object literal wrote; or at least the fields a read or a pattern named.
Two undeclared field sets join to their union as above. A declared field
set is the field set of every value of that type, so an object that is
one and lacks a field the struct declares, or carries a field it does
not declare, is refused by that field's name, and what only asks an
object for fields joins to the declared type. **R2** A decision is a position with more than
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

Width, instance, effect, representation and conversion are five questions
the language asks after a join, and each answered at the join by a device of
its own: eight snapshot/rollback pairs, a polarity, and a settlement in the
middle of a body, each having to guess an answer the terms had not yet
resolved. One shape for a decision and one settlement remove the devices,
the polarity machinery and the mid-body settlement, and let a message name
the final type instead of the type at the guess.

## Not built

`ρ` opened on a concrete signature; a global (non-greedy) choice of
representation; conversions at the sites listed in RFC-0041's "Not
built".

## Consequences

- `acvus-mir/src/solver.rs` holds `Terms`, the union-find over type,
  effect, length, identity and representation, with
  `join(a, b, Position, JoinKind, registry)`; `Decision`, `Answer`,
  `Unsettled`; `settle`, `solve`, `freeze_ty`. It holds no `Polarity`, no
  `lub`, no `coerce` and no snapshot outside a decision's own step.
  RFC-0040's instance choice is `Decision::Instance`, read by `answer(id)`.
- `acvus-mir/src/typeck.rs`: `flow`, `convert_at`, `solve_body`,
  `report_unsettled`, `resolve_conversions`; `Intrinsic` in `ir.rs`.
- `acvus-mir/src/ty.rs` holds `FieldSet`, `ObjectTy` and
  `ObjectTy::meet`, which the join calls for two objects. `#[derive(TyArg)]`
  emits `ObjectTy::declared` for a struct, so every extern parameter of a
  declared struct's type carries its field set; a struct variant's payload
  is an object a literal writes and carries `Written`.
