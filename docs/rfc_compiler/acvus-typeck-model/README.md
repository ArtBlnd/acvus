# acvus-typeck-model

An executable definition of how the transparency-constraint type system solves a
program. The system is specified in `docs/transparency-constraint-type-system-kr.md`,
whose Korean text is authoritative; `docs/transparency-constraint-type-system.md` is
its English text, aligned line by line. Below, "the spec" is that pair, and a line
number is the same in both.

The model serves three ends:

- Each sentence of the spec it encodes is read in one way only.
- It checks, over every input up to a given size, that the outcome is a function of
  the constraint graph alone, whatever the order of solving
  (`docs/notes/typeck-pivot.md` §6), and that solving ends.
- It is the oracle that the type checker in `acvus-mir` is compared against.

## Invariants

- **I1.** Every rule cites the spec lines it encodes. A rule that no line states is
  not written. Where the spec leaves a reading open, the model names the reading and
  takes it as a parameter.
- **I2.** The order of solving is an input: the same choices give the same outcome.
  No other order enters, so the model iterates only over sorted containers.
- **I3.** The outcome is every equivalence class's value, the partition the unions
  made, every decision, every relation checked with the coercion function it chose,
  every place of refusal, and whether the program is accepted. Equivalence classes
  are compared by names that do not depend on the order of solving.
- **I4.** The model depends on the standard library alone, so that its errors are
  independent of those of the code it checks.

## Design notes

Each note under `design/` states one primitive: its meaning, what it is responsible
for and what it is not, its interface, its contract, the lines it rests on, the
places left open for change, and the questions the spec leaves open there. A note
is the contract until its code exists. Once the code carries the contract in its
types, signatures and docs, the note keeps only what the code cannot carry.

The primitives, each assuming only those before it:

| Note | Primitive | Spec |
|---|---|---|
| 01 | Types: names, terms, constructors, members, parts, variance | l.39–74 |
| 02 | Types in declarations: declared types, signatures, substitution and matching | l.54–55, l.64, l.320, l.347 |
| 03 | Declarations and the cast tree: extern fns and impls by name, casts and views, conversions | l.76–78, l.105, l.320 |
| 04 | Join and `≤` | l.80–91 |
| 05 | Coercion and rank: LitVars, values a use reads, coercion, the steps of `≤`, rank | l.93–109, l.127, l.162, l.261, l.280, l.333 |
| 06 | Equivalence classes: names, union-find, type variables, kinds | l.111–169 |
| 07 | Relations and what records them | l.221–231 |
| 08 | Uses, lends, projections and bindings | l.142–154, l.292–334 |
| 09 | Binding groups, universals, C and instances | l.188–219, l.240, l.286–290 |
| 10 | The dependency graph | l.234–265 |
| 11 | Values, splits and fills | l.267–282 |
| 12 | Steps, state, fixpoint and fallback | l.284–290 |
| 13 | Outcome and exploration | I2, I3 |
| 14 | The certificate and its checker | l.219–231, l.280, l.290 |
| 15 | I\* | l.238–263, `docs/notes/typeck-pivot.md` §1–2 |
| 16 | Inputs up to a size (draft) | l.219–231, l.292–310 |
| 17 | The captured environment | l.186, l.196, l.203–219, l.238, functions §2 |
| 18 | Own relations first, invariant points last (draft, superseded by l.278 and l.282) | l.221, l.234–290, l.333 |
| 19 | Cases and their runner, and the phases' reports | I2, I3, I4, notes 13–15 |

Writing a note may change the notes after it.
