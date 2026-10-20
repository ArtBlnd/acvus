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
| 02 | Types in declarations: declared types, signatures, substitution and matching | l.54–55, l.64, l.331, l.354 |
| 03 | Declarations and the cast tree: extern fns and impls by name, casts and views, conversions | l.76–78, l.112, l.331 |
| 04 | Join and `≤` | l.80–97 |
| 05 | Coercion and rank: LitVars, values a use reads, coercion, the steps of `≤`, rank | l.99–116, l.134, l.169, l.267, l.293, l.344 |
| 06 | Equivalence classes: names, union-find, type variables, kinds | l.118–176 |
| 07 | Relations and what records them | l.228–238 |
| 08 | Uses, lends, projections and bindings | l.149–161, l.303–345 |
| 09 | Binding groups, universals, C and instances | l.195–226, l.246, l.297–301 |
| 10 | The dependency graph | l.240–271 |
| 11 | Values and splits | l.273–293 |
| 12 | Steps, state, fixpoint and fallback | l.295–301 |
| 13 | Outcome and exploration | I2, I3 |
| 14 | The certificate and its checker | l.226–238, l.293, l.301 |
| 15 | I\* | l.244–269, `docs/notes/typeck-pivot.md` §1–2 |
| 16 | Inputs up to a size (draft) | l.226–238, l.303–321 |
| 17 | The captured environment | l.203, l.210–226, functions §2 |

Writing a note may change the notes after it.
