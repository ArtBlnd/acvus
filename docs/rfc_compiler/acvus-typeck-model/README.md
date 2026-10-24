# acvus-typeck-model

An executable definition of how the transparency-constraint type system solves a
program.

The model serves two ends:

- It checks, over every input up to a given size, that the outcome is a function of
  the constraint graph alone, whatever the order of solving, and that solving ends.
- It is the oracle that the type checker in `acvus-mir` is compared against.

## Invariants

- The order of solving is an input: the same choices give the same outcome.
  No other order enters, so the model iterates only over sorted containers.
- The outcome is every equivalence class's value, the partition the unions
  made, every decision, every relation checked with the coercion function it chose,
  every place of refusal, and whether the program is accepted. Equivalence classes
  are compared by names that do not depend on the order of solving. Two outcomes are
  one where their typing is, all but the partition (`Outcome::same_typing`): which
  classes are one is no part of what does not depend on the order.
- The model depends on the standard library alone, so that its errors are
  independent of those of the code it checks.

## Design notes

Each note under `design/` states one primitive: its meaning, what it is responsible
for and what it is not, its interface, its contract, and the places left open for
change. A note is the contract until its code exists. Once the code carries the
contract in its types, signatures and docs, the note keeps only what the code cannot
carry.

The primitives, each assuming only those before it:

| Note | Primitive |
|---|---|
| 01 | Types: names, terms, constructors, members, parts, variance |
| 02 | Types in declarations: declared types, signatures, substitution and matching |
| 03 | Declarations and the cast tree: extern fns and impls by name, casts and views, conversions |
| 04 | Join and `≤` |
| 05 | Coercion and rank: LitVars, values a use reads, coercion, the steps of `≤`, rank |
| 06 | Equivalence classes: names, union-find, type variables, kinds |
| 07 | Relations and what records them |
| 08 | Uses, lends, projections and bindings |
| 09 | Binding groups, universals, C and instances |
| 10 | The dependency graph |
| 11 | Values, splits and fills |
| 12 | Steps, state, fixpoint and fallback |
| 13 | Outcome and exploration |
| 14 | The certificate and its checker |
| 15 | I\* |
| 16 | Inputs up to a size (draft) |
| 17 | The captured environment |
| 18 | Own relations first, invariant points last (draft) |
| 19 | Cases and their runner, and the phases' reports |

Writing a note may change the notes after it.
