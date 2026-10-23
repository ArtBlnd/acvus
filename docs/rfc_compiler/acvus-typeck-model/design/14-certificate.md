# 14 The certificate and its checker

Rests on the spec's l.290 (acceptance), l.219 (meeting and boundary), l.221–231
(what the text records), l.280 (entering values), l.120, l.127, l.181 (the types a
class is given), l.144–154 (projections), l.246 (whether a class is a reference),
l.296, l.298, l.306, l.320–333 (what a decision gives) and l.188–215 (instances),
with the crate's I4. The code is `src/certificate.rs`.

## Meaning

An order of solving that accepts a program leaves a certificate: the value of every
class each binding group's solving made, with its name, and every decision. A
function value in a value carries the lambdas its Values names (`State::values_of`),
as a function value carries its facts on its type (l.184), so that a value is
`Option⟨Fn/1{λ}⟩` where the solver's type is `Option⟨Fn/1⟩`. No declaration, text
or context writes a function value (l.184; note 02, reading 2), so every function
value in a value is a class's, and names its Values. The
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
lambda's relations on enclosing groups' classes are its environment's (l.186),
checked in the enclosing group's contexts whether or not the lambda is
instantiated, and an instance's argument meets the environment's parameter
there.

Besides the relations, l.290's conditions: no refusal, every class outside C with a
value, and every node outside C reached.

## Contract

What each item and decision gives, and what holding means on closed values:

| Given by | Relation | Holds where |
|---|---|---|
| `Origin::Written(τ)`, `Origin::Host(τ)` | the class's kind | its value is `τ` with each part's value |
| `Origin::Lit(l)` | a LitVar | its value is a type of `l`'s family (l.127) |
| `Origin::Lambda(λ)` | a function value naming `λ` | its value is `Fn/n`, `n` being `λ`'s parameters, whose Values name `λ` |
| `Pass` into an unknown of the place's group (l.219) | `τ ≤ α` | `≤` reaches it (l.280) |
| `Pass` otherwise | `τ ⇝ U` | a coercion reaches it (l.280) |
| `Equation` | `a = b` | the two values are equal |
| a use of a function, own group (l.508) | each argument into its universal by a boundary, the result type as a lower bound | as above |
| a use of a function, another group | the same, into the instance's copies | as above, under the instance's assignment |
| `Use(k)` | `k` is a candidate (l.320); each argument into its parameter, a meeting where the signature writes a variable, a boundary where it writes a type; the result type as a lower bound of the result | as above, the variables taking the values of the classes the decision made |
| `Never` | `!` as a lower bound of the result, and an argument the use reads has the value `!` (l.333) | as above |
| `Lend { target, mutability }` | the result is a reference of `mutability` to the place, `mutability` being the lend's own; or, where the place is a reference, to its target, `mutability` being the lesser of the lend's and the place's (l.296) | the values have those shapes |
| `Applied` | the operand has the member or part, of the result's value (l.144–152) | its value has it, or is `!`, which meets every kind (l.276) |
| `AppliedToTarget(m)` | the operand is a reference, of `m` where decided, to a type that has it (l.298) | as above |
| `Read` | the position as a lower bound of the binding (l.306) | as above |
| `BindLend { target, mutability }` | a reference to the position, or its target, as a lower bound of the binding (l.306) | as above |
| `Calls(λ̄)` | the callee's Values name exactly `λ̄`: every lambda that reaches it is called, and none it does not reach (l.181, l.196); each argument meets its copy of `λ`'s parameter, and the copy of its result type is a lower bound of the result (l.196) | as above, under each instance |
| a function value reaching a function type, at the whole value or at a part, through a cast, a conversion or a view (l.197, l.278) | each lambda its Values names has an instance made where the split met the two, at the classes they are read at: each parameter of the type passes into the lambda's parameter, and the lambda's result into the type's result, each a meeting or a boundary as the result's class is an unknown or not (l.219) | as above, under that instance |
| `e?`'s exit (l.337) | its value is `Option⟨K⟩` under `?(Option⟨T⟩)` and `Result⟨O, X⟩` under `?(Result⟨T, E⟩)`, `E`'s value wrapped into `X`: as it is where it is an `Error` or `!`, into the payload of an `Error` whose one member is its name, or into that of an `Error⟨..⟩`; the value passes into the exit's class, a meeting where that is an unknown | as above |
| a use's common part (l.246) | regenerated from the candidates (`CommonPart::of`): each unknown's value a reference, or not, where the common part records it so; each argument passing into the common parameter; the common result entering the use's result; under `Use(k)`, `k`'s type at each of the result's unknowns entering it; a use of one candidate, which has no node, as that candidate | each holds, a reference's record holding of `!` too |

Holding compares Values as part of the value. Two values are equal where their
types are and every function value in them names the same lambdas. Where `≤` or
a coercion reaches, each function value of the one reaching is carried along it as
the split carries it (`reach`, l.278): at the same position where no stage moves
it, a reborrow keeping the target; along a cast's, a conversion's or a view's
signature, from where its parameter writes a variable to where its result writes
it (`image`); at each part both have, by that part's own coercion, where the
structural stage changes the members. Where it reaches a function value, its
lambdas are among that one's; where it reaches a function type, the row of l.197
holds at the classes the two are read at (`Checker::flows`), which the checker
reads by each class's parts as its sheet keeps them. So a lambda a value carries is
among the Values of every class the value reaches, and a call of that class calls
it. Since the review of 2026-10-23, as 준량 asked, text, solver and checker carry
Values alike: the checker had followed none across a cast, a conversion or a view,
and met a function type at the top of a value alone, and panicked at
`run(Some(|x| x))` and `take_rc(boxed(|x| x))`.

The common part's row is checked on its own: it is what a use records before it
is decided, which `Use(k)`'s rows do not name. That `!` is either a reference or
not is the solver's reading (`Classes::normalize`), shared here.

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
  root's value, every decision, and each
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
- Which candidate a use is decided as, and what a lend or a projection reads, are
  not this check's: any decision whose relations hold passes here. Whether the
  decision is the one its node should make is note 15's.
- A refused program has nothing to check here.
- Whether a class's Values are the least its function values give: a value whose
  Values name a lambda that no function value brings passes where every relation
  holds, and only makes a call check one instance more.

## Open

Where Values meet what no rule decides yet, the checker panics, so that the case
fails there and the rule is decided with 준량:

- A function value reaching a function type where the two sides are read at two
  levels of a context: the instances of l.197 are found at the level that split
  them.
- A function value reaching a function type where the checker reads no class: the
  wrapping of `?`'s error, whose payload holds no function type the text writes.

## Expected results

Before the checker runs: every accepted outcome of the current examples passes.
A violation there is the first soundness hole. A certificate whose value or
decision is changed, as a solver that erred would leave it, fails at each relation
the change breaks.

Run, every accepted final state of every example passes (Examples 3 and 14,
refused, have none), and `cases/certificate/a-wrong-value-or-decision-fails.case` finds `x: B` in place of `x: A` at the pass of `make()` into `X` and at
`probe(A)`'s boundary, `probe(B)` in place of `probe(A)` at its boundary, and
`probe(x): &A` in place of `()` at its result and at the record of l.246. A lower
bound reaches by `≤` alone (l.280), so the checker reads no class as decided by an
equation. A value a case plants writes each function value's Values after it,
`Fn/1{λ}`, `Fn/1{}` naming none (note 19), as a certificate's value is written:
`cases/certificate/a-lambda-through-a-cast` and `a-lambda-through-a-view` give the
class λ reaches through the cast or the view `Fn/1{}`, and the checker fails there
and at the use's `T` of that class, as derived, so that Values are compared along a
cast and a view, not only carried.
