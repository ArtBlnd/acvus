# acvus-typeck-model

An executable definition of how the transparency-constraint type system solves a
program.

The model serves two ends:

- It checks, over every input up to a given size, that the outcome is a function of
  the constraint graph alone, whatever the order of solving, and that solving ends.
- It is the oracle that the type checker in `acvus-mir` is compared against.

## Solving

Solving is one loop, and has no other step:

```
                  the relations the text gives
                               │
                               ▼
              ┌─────────────────────────────────┐
     ┌───────▶│ unify                           │  equations and structures make
     │        │                                 │  classes one, and unification
     │        │                                 │  stops at each class whose value
     │        │                                 │  is the join of what enters it:
     │        │                                 │  a join position
     │        └────────────────┬────────────────┘
     │                         ▼
     │        ┌─────────────────────────────────┐
     │        │ instantiate                     │  a lambda a decided type names
     │        │                                 │  is unfolded: its C made anew
     │        │                                 │  where a function value is called
     │        │                                 │  or passes into a function type
     │        └────────────────┬────────────────┘
     │                         ▼
     │        ┌─────────────────────────────────┐
     │        │ join                            │  every relation reaching a join
     │        │  + record at each join position │  position recorded there, what
     │        │                                 │  enters it and what it enters; a
     │        │                                 │  join position with nothing more
     │        │                                 │  to come takes, at once, the type
     │        │                                 │  its record gives
     │        └────────────────┬────────────────┘
     │                         │
     │   something changed     │
     └─────────────────────────┤
                               │  nothing changed: the fixpoint
                               ▼
              ┌─────────────────────────────────┐
     ┌────────┤ fallback                        │  each class left with no value
     │        │                                 │  takes its default
     │        └────────────────┬────────────────┘
     │   a class fell back     │  none fell back
     ▼                         ▼
   unify                  the outcome
```

A type is decided once, from everything that bears on it, and the order of the
loop gives each decision all of that:

- **Unify** comes first. An equation chooses nothing: it only makes classes
  one, so it is taken as far as it goes. It stops at a join position, whose
  value depends on all that enters it.
- **Instantiate** comes next. A lambda is unfolded only where its type is
  known: the function value's Values and the call or the function type it
  meets. Without that type there is nothing to unfold, so it waits for what
  unify and join decide. Unfolding records new relations, which reach join
  positions before those decide.
- **Join** comes last. Each relation that reaches a join position is recorded
  there, what enters it and what it enters alike, and this record is all a join
  reads. A join position decides once nothing more is to come to it, at once,
  from its record.
- **Fallback** comes only when the three change nothing. A class still with no
  value takes its default, which may give a type that unfolds more or decides
  a join, so the loop runs again.

A change to solving changes where unification stops and what a class waits for,
or what a join position records and a join takes; it adds no step to the loop.

The loop in the code, for each binding group in `Program::group_order` and then
the entry (`explore::sample_with`, `cases::run::run`):

```
State::start                                 the text recorded
├─ record_origins, record_equations          unify    (State::record → hold → Classes::unifying)
├─ record_items                              unify; a use of a function: instantiate (Instance)
├─ decide_at_once                            a Deref
└─ settle
loop
├─ steps → eligible over graph               what has no unbroken dependency left (dep.rs)
│  ├─ Step::Solve(class)                     join     (solve → join)
│  └─ Step::Decide(disjunctive constraint)   join at it: its alternative chosen from the value read
│                                            (decide → value_read); instantiate (record_decision)
├─ step → settle
└─ no step left
   ├─ a binding group                        its fixpoint, kept in `solved` for its instances
   └─ the entry                              default  (fall_back → take_defaults → settle);
                                             nothing fell back: the outcome
settle, until nothing changes
├─ merge_cycles, link_views                  unify
├─ record_generalizations                    unify: what every alternative of an undecided
│                                            disjunctive constraint shares
└─ for each relation not decomposed
   ├─ decompose_at                           unify: the parts of two solved sides; a function value
   │                                         into a function type: instantiate (instantiate_lambda)
   └─ domain_up                              join: the domain to be joined, recorded at the class
                                             a lower bound holding no structure bounds
```

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
| 06 | Equivalence classes: names, union-find, type variables, structures |
| 07 | Relations and what records them |
| 08 | Uses, lends, projections and bindings |
| 09 | Binding groups, universals, C and instances |
| 10 | The dependency graph |
| 11 | Values, decompositions and fills |
| 12 | Steps, state, fixpoint and fallback |
| 13 | Outcome and exploration |
| 14 | The certificate and its checker |
| 15 | I\* |
| 16 | Inputs up to a size (draft) |
| 17 | The captured environment |
| 18 | Own relations first, invariant points last (draft) |
| 19 | Cases and their runner, and the phases' reports |

Writing a note may change the notes after it.
