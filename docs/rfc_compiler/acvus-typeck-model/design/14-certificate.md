# 14 The certificate and its checker

The code is `src/certificate.rs`.

## Meaning

An order of solving that accepts a program leaves a certificate: the value of every
class each binding group's solving made, with its name, and every decision. A
function value in a value carries the lambdas its Values names (`State::values_of`),
as a function value carries its facts on its type, so that a value is
`Option⟨Fn/1{λ}⟩` where the solver's type is `Option⟨Fn/1⟩`. No declaration, text
or context writes a function value, so every function
value in a value is a class's, and names its Values. The
checker regenerates, from the program's items and the decisions alone, every
relation the text and the decisions give, and checks each on the certificate's
closed values. A relation that does not hold on an accepted program is a soundness
violation: the program is accepted with a typing that does not satisfy it.

The checker reads `Program` and the certificate, never a `State`'s log, and works on
closed types alone, with no equivalence class, union-find, dependency or value read.
So an error in recording, in decompositions, in the dependency graph, in fallback or in
copying C shows as a violation, whichever order made it.

## Domain

Every binding group, in every context its text is solved for:

- The entry, under its final values.
- A group other than the entry, under its values at the fixpoint, for the
  relations that name no class of its C.
- Each instance of a group's C, made at a use of one of its
  functions from another group, or at a call of a function value naming one of its
  lambdas, under that instance's assignment: a class of C has its copy's
  value in the state that made the instance, a class outside C its group's value,
  and a class of an enclosing group the value of the class of that name in the
  state that made the instance.

A relation is checked in each context that gives every class it names a value. A
lambda's relations on enclosing groups' classes are its environment's,
checked in the enclosing group's contexts whether or not the lambda is
instantiated, and an instance's argument enters the environment's parameter
there.

Besides the relations: no refusal, every class outside C with a
value, and every disjunctive constraint outside C reached.

## Contract

What each item and decision gives, and what holding means on closed values:

| Given by | Relation | Holds where |
|---|---|---|
| `Origin::Written(τ)`, `Origin::Host(τ)` | the class's structure | its value is `τ` with each part's value |
| `Origin::Lit(l)` | a LitVar | its value is a type of `l`'s family |
| `Origin::Lambda(λ)` | a function value naming `λ` | its value is `Fn/n`, `n` being `λ`'s parameters, whose Values name `λ` |
| `Pass` into an unknown of the place's group | `τ ≤ α` | `≤` reaches it |
| `Pass` otherwise | `τ ⇝ U` | a coercion reaches it |
| `Equation` | `a = b` | the two values are equal |
| a use of a function, own group | each argument into its universal by a coercion constraint, the result type as a lower bound | as above |
| a use of a function, another group | the same, into the instance's copies | as above, under the instance's assignment |
| `Use(k)` | `k` is a candidate; each argument into its parameter, a lower bound where the signature writes a variable, a coercion constraint where it writes a type; the result type as a lower bound of the result | as above, the variables taking the values of the classes the decision made |
| `Never` | `!` as a lower bound of the result, and an argument the use reads has the value `!` | as above |
| `Lend { target, mutability }` | the result is a reference of `mutability` to the place, `mutability` being the lend's own; or, where the place is a reference, to its target, `mutability` being the lesser of the lend's and the place's | the values have those shapes |
| `Applied` | the operand has the member or part, of the result's value | its value has it, or is `!`, which satisfies every structure constraint |
| `AppliedToTarget(m)` | the operand is a reference, of `m` where decided, to a type that has it | as above |
| `Read` | the position as a lower bound of the binding | as above |
| `BindLend { target, mutability }` | a reference to the position, or its target, as a lower bound of the binding | as above |
| `Calls(λ̄)` | the callee's Values name exactly `λ̄`: every lambda that reaches it is called, and none it does not reach; each argument enters its copy of `λ`'s parameter, and the copy of its result type is a lower bound of the result | as above, under each instance |
| a function value reaching a function type, at the whole value or at a part, through a cast, a conversion or a view | each lambda its Values names has an instance made where the decomposition found the two, at the classes they are read at: each parameter of the type passes into the lambda's parameter, and the lambda's result into the type's result, each a lower bound or a coercion constraint as the result's class is an unknown or not | as above, under that instance |
| `e?`'s exit | its value is `Option⟨K⟩` under `?(Option⟨T⟩)` and `Result⟨O, X⟩` under `?(Result⟨T, E⟩)`, `E`'s value wrapped into `X`: as it is where it is an `Error` or `!`, into the payload of an `Error` whose one member is its name, or into that of an `Error⟨..⟩`; the value passes into the exit's class, a lower bound where that is an unknown | as above |
| a use's generalization | regenerated from the candidates (`Generalization::of`): each unknown's value a reference, or not, where the generalization records it so; each argument passing into the generalization's parameter; the generalization's result entering the use's result; under `Use(k)`, `k`'s type at each of the result's unknowns entering it; a use of one candidate, which has no disjunctive constraint, as that candidate | each holds, a reference's record holding of `!` too |

Holding compares Values as part of the value. Two values are equal where their
types are and every function value in them names the same lambdas. Where `≤` or
a coercion reaches, each function value of the one reaching is carried along it as
the decomposition carries it (`reach`): at the same position where no stage moves
it, a reborrow keeping the target; along a cast's, a conversion's or a view's
signature, from where its parameter writes a variable to where its result writes
it (`image`), from each place it writes a variable, a variable it writes twice
taking one type and each place its own Values; at each part both have, by that
part's own coercion, where the structural stage changes the members. Where it reaches a function value, its
lambdas are among that one's. So a lambda a value carries is
among the Values of every class the value reaches, and a call of that class calls
it.

The generalization's row is checked on its own: it is what a use records before it
is decided, which `Use(k)`'s rows do not name. That `!` is either a reference or
not is the solver's reading (`Classes::normalize`), shared here.

Whether a `Pass` enters is recomputed from the program: the class passed into is an
unknown of its own group by its text and is equated by no regenerated
equation to a class the text writes, and its group is the place's, or it is
reached by name; a place reached through a reference is a coercion constraint. The relations a decomposition
records are parts of the relations above and need no check of their own.

## Interface

- `Certificate::of(&Final)` takes a `Sheet` of each state of a final state, the
  entry's and each group's solved before it: every class's name and root, each
  root's value, every decision, and each
  instance the state made (`Made`), with its group and the map from the group's
  roots to the copies made. It is not part of `Outcome`, since the names of the
  classes solving makes may differ between orders.
- `fn check(program: &Program, certificate: &Certificate) -> Vec<Violation>`, where
  `Violation` names the context, the item, disjunctive constraint or class that gives the relation,
  and the relation with the values it fails on. It returns rather than panics, so
  that exploration collects every violation of every order before a test fails.

Two additions outside this module follow: `State::instances` keeps each instance
the state made, and `explore_with` and `sample_with` hand each final state, with
the groups solved before it, to the caller.

## Not covered

- The checker shares `coerce`, `along_le`, the LitVar families and `Ty` with the
  solver. An error there is shared, and the checker does not see it.
- Which candidate a use is decided as, and what a lend or a projection reads, are
  not this check's: any decision whose relations hold passes here.
- A refused program has nothing to check here.
- Whether a class's Values are the least its function values give: a value whose
  Values name a lambda that no function value brings passes where every relation
  holds, and only makes a call check one instance more.

## Open

Where Values reach what no rule decides yet, the checker panics, so that the case
fails there and the rule is decided:

- A function value reaching a function type where the two sides are read at two
  levels of a context: the instances are found at the level that decomposed
  them.
- A function value reaching a function type where the checker reads no class: the
  wrapping of `?`'s error, whose payload holds no function type the text writes.

## Expected results

A certificate whose value or decision is changed, as a solver that erred would leave
it, fails at each relation the change breaks.

Run, `cases/certificate/a-wrong-value-or-decision-fails.case` finds `x: B` in place
of `x: A` at the pass of `make()` into `X` and at `probe(A)`'s coercion constraint, `probe(B)` in
place of `probe(A)` at its coercion constraint, and
`probe(x): &A` in place of `()` at its result. A lower
bound reaches by `≤` alone, so the checker reads no class as decided by an
equation. A value a case plants writes each function value's Values after it,
`Fn/1{λ}`, `Fn/1{}` naming none, as a certificate's value is written:
`cases/certificate/a-lambda-through-a-cast` and `a-lambda-through-a-view` give the
class λ reaches through the cast or the view `Fn/1{}`, and the checker fails there
and at the use's `T` of that class, as derived, so that Values are compared along a
cast and a view, not only carried. Since the solver checks the Values a
conversion carries, `a-lambda-through-a-conversion` is
accepted, and gives take_rc's T `Fn/1{}`, failing at the conversion and at the
pass of |y| y into T; `a-lambda-at-each-place-a-conversion-writes-one-variable`
is accepted with λ and μ at the two places of one variable, which the checker
had asked to be equal.
