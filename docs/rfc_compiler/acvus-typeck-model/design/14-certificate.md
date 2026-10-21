# 14 The certificate and its checker

Rests on the spec's l.287 (acceptance), l.219 (meeting and boundary), l.221–231
(what the text records), l.279 (entering values), l.120, l.127, l.181 (the types a
class is given), l.144–154 (projections), l.245 (whether a class is a reference),
l.293, l.295, l.303, l.317–330 (what a decision gives) and l.188–215 (instances),
with the crate's I4. The code is `src/certificate.rs`.

## Meaning

An order of solving that accepts a program leaves a certificate: the value of every
class each binding group's solving made, with its name, and every decision. The
checker regenerates, from the program's items and the decisions alone, every
relation the text and the decisions give, and checks each on the certificate's
closed values. A relation that does not hold on an accepted program is a soundness
violation: the program is accepted with a typing that does not satisfy it.

The checker reads `Program` and the certificate, never a `State`'s log, and works on
closed types alone, with no equivalence class, union-find, dependency or value read.
So an error in recording, in splits, in the dependency graph, in fallback or in
copying C shows as a violation, whichever order made it.

## Domain

Every binding group, in every context its text is solved for:

- The entry, under its final values.
- A group other than the entry, under its values at the fixpoint, for the
  relations that name no class of its C.
- Each instance of a group's C (l.212, l.215), made at a use of one of its
  functions from another group, or at a call of a function value naming one of its
  lambdas (l.196), under that instance's assignment: a class of C has its copy's
  value in the state that made the instance, a class outside C its group's value,
  and a class of an enclosing group the value of the class of that name in the
  state that made the instance.

A relation is checked in each context that gives every class it names a value. A
lambda never instantiated leaves its relations on enclosing groups' classes
unchecked, since nothing records them (l.215). Read by the captured environment
(note 17), those relations are the environment's, checked in the enclosing group's
contexts whether or not the lambda is instantiated, and an instance's argument
meets the environment's parameter there.

Besides the relations, l.287's conditions: no refusal, every class outside C with a
value, and every node outside C reached.

## Contract

What each item and decision gives, and what holding means on closed values:

| Given by | Relation | Holds where |
|---|---|---|
| `Origin::Written(τ)`, `Origin::Host(τ)` | the class's kind | its value is `τ` with each part's value |
| `Origin::Lit(l)` | a LitVar | its value is a type of `l`'s family (l.127) |
| `Origin::Lambda(λ)` | a function value naming `λ` | its value is `Fn/n`, `n` being `λ`'s parameters |
| `Pass` into an unknown of the place's group (l.219) | `τ ≤ α` | `≤` reaches it, or a coercion does where an equation decides `α` (l.279) |
| `Pass` otherwise | `τ ⇝ U` | a coercion reaches it (l.279) |
| `Equation` | `a = b` | the two values are equal |
| a use of a function, own group (l.500) | each argument into its universal by a boundary, the result type as a lower bound | as above |
| a use of a function, another group | the same, into the instance's copies | as above, under the instance's assignment |
| `Use(k)` | `k` is a candidate (l.317); each argument into its parameter, a meeting where the signature writes a variable, a boundary where it writes a type; the result type as a lower bound of the result | as above, the variables taking the values of the classes the decision made |
| `Never` | `!` as a lower bound of the result, and an argument the use reads has the value `!` (l.330) | as above |
| `Lend { target, mutability }` | the result is a reference of `mutability` to the place, `mutability` being the lend's own; or, where the place is a reference, to its target, `mutability` being the lesser of the lend's and the place's (l.293) | the values have those shapes |
| `Applied` | the operand has the member or part, of the result's value (l.144–152) | its value has it |
| `AppliedToTarget(m)` | the operand is a reference, of `m` where decided, to a type that has it (l.295) | as above |
| `Read` | the position as a lower bound of the binding (l.303) | as above |
| `BindLend { target, mutability }` | a reference to the position, or its target, as a lower bound of the binding (l.303) | as above |
| `Calls(λ̄)` | each `λ` is named by a function value that reaches the callee by the regenerated lower bounds and equations (l.181); each argument meets its copy of `λ`'s parameter, and the copy of its result type is a lower bound of the result (l.196) | as above, under each instance |
| a use's record of l.245 | at each position every candidate would record a lower bound or an equation into, the result and the parts of an argument an invariant part equates (note 08, reading 13), that it is a reference where every candidate's type there is one, and no reference where none is; under `Use(k)`, as `k`'s type there | its value is, or is not, a reference, or is `!` |

The l.245 row is not implied by `Use(k)`'s: a cast changes not whether a type is a
reference (l.78), but a declared conversion may, and admits a lower bound into a
class an equation decides (l.279). That `!` holds either is the solver's reading
(`Classes::normalize`), shared here.

Whether a `Pass` meets is recomputed from the program: the class passed into is an
unknown of its own group by its text (l.211) and is equated by no regenerated
equation to a class the text writes, and its group is the place's, or it is
reached by name; a place reached through a reference is a boundary (l.219,
`design/09-group.md`, reading 8). The relations a split
records are parts of the relations above and need no check of their own (note 12,
reading 3).

## Interface

- `Certificate::of(&Final)` takes a `Sheet` of each state of a final state, the
  entry's and each group's solved before it: every class's name and root, each
  root's value and whether an equation decides it, every decision, and each
  instance the state made (`Made`), with its group and the map from the group's
  roots to the copies made. It is not part of `Outcome`, since the names of the
  classes solving makes may differ between orders (note 13, reading 1).
- `fn check(program: &Program, certificate: &Certificate) -> Vec<Violation>`, where
  `Violation` names the context, the item, node or class that gives the relation,
  and the relation with the values it fails on. It returns rather than panics, so
  that exploration collects every violation of every order before a test fails.

Two additions outside this module follow: `State::instances` keeps each instance
the state made, and `explore_with` and `sample_with` hand each final state, with
the groups solved before it, to the caller (note 13, Checks).

## Not covered

- The checker shares `coerce`, `along_le`, the LitVar families and `Ty` with the
  solver. An error there is shared, and the checker does not see it.
- Whether an equation decides a class is taken from the certificate. Deciding it
  needs the equations splits derive (l.277), which the checker does not regenerate;
  a solver that marks a class wrongly, and so admits a coercion where `≤` was due,
  is not seen.
- Which candidate a use is decided as, and what a lend or a projection reads, are
  not this check's: any decision whose relations hold passes here. Whether the
  decision is the one its node should make is note 15's.
- A refused program has nothing to check here.
- Which lambdas meet a function value's class is followed along the text's passes
  and equations alone; a function value passed through a use's result is not, so
  a `Calls` whose lambda reached the callee that way is reported though it holds.

## Expected results

Before the checker runs: every accepted outcome of the current examples passes.
A violation there is the first soundness hole. A certificate whose value or
decision is changed, as a solver that erred would leave it, fails at each relation
the change breaks.

Run, every accepted final state of every example passes (Example 14, refused,
has none), and `cases/certificate/a-wrong-value-or-decision-fails.case` finds `x: B` in place of `x: A` at the pass of `make()` into `X` and at
`probe(A)`'s boundary, `probe(B)` in place of `probe(A)` at its boundary, and
`probe(x): &A` in place of `()` at its result and at the record of l.245. A
first run reported Example 3's `push` refused at `u16` into `u8`: the checker
dropped whether an equation decides the parameter's variable, which admits the
conversion (l.279); it was the checker's error.
