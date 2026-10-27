# 12 The judgment, the loop and fallback

The code is `src/judgment.rs`, which decides, `src/solve.rs`, which composes the
loop, fallback in `src/fallback.rs`, and acceptance in `src/check.rs`. A step, a
settling and a fallback return the trace of what their phases reported,
`src/trace.rs`.

## Meaning

The judgment is the one place where solving decides. It reads the record and gives
an equivalence class its type by one rule, from what the record holds at the class:
its domain `D`, the constructors its value may take; its lower bounds `L`, the heads
of the decided values that enter it, by `≤` or by `≤ₚ`, and what its structure
constraints give; and its upper bounds `U`, the heads of the decided classes it
passes into, by `≤` or by a coercion constraint. A bound that is a class not yet
decided gives its own bounds in its place, so a class and the undecided classes it
is bound by are decided together.

| `L` | `U` | the type |
|---|---|---|
| not empty | any | `⊔L`, each lower bound raised along the casts and each part lower bound with no cast; refused `no_join` where they have none. `U` decides nothing here: each `T ⇝ u` is a check |
| empty | not empty | `⊓U`, the greatest type that reaches every upper bound: along the casts, the one of them below the others, and none where no one is below the others; refused `no_meet` where they have none |
| empty | empty | nothing: the domain is recorded, and fallback reads it |

`!` is no candidate of the rule, though it reaches every type: it is fallback's
alone. A LitVar gives its class a domain and no lower bound, so a class only LitVars
enter is one nothing enters. `D` is enforced where a structure is recorded
(`Classes::unifying`, `Conflict::Domain`), so the rule does not read it. The rule is
`State::decided`; `head_join` and `head_meet` in `src/value.rs` are its two
operations, dual to each other: two references of two mutabilities join at the shared
one and meet at the exclusive one, two nodes join at their nearest common ancestor and
meet at the one below the other, and an object's or an enum's members are united by
both.

The judgment decides a class once nothing undecided may still change the bounds that
decide it: no undecided disjunctive constraint may record into it or into a class it
takes lower bounds from (`Bounds::lower_pending`), and, where nothing enters it, none
may take what it passes into (`Bounds::upper_pending`). A disjunctive constraint is
decided by the same reading of the record, each class it reads as the rule gives it
now, without the relations the disjunctive constraint itself would record.

The loop takes judgments until none is left, the fixpoint, and fallback then gives
each class whose bounds decide nothing its domain's default, or `!`, and the loop runs
again, until nothing falls back. The program is accepted where no refusal was
recorded, every class has a value, every disjunctive constraint is decided, and every
value the text or a disjunctive constraint makes enter a type reaches it: a lower
bound by `≤`, and a coercion constraint by a coercion.

## Readings

1. **Fallback is no equation.** A class that falls back holds its default as a
   class holds a joined value, so its lower bounds reach it by `≤`.
2. **What is checked.** The relations the text and the disjunctive constraints
   record; a decomposition's relations are parts of those.

## Left open for change

- `Bounds::lower_pending` and `Bounds::upper_pending`: when the bounds that decide
  a class may still change, which is what a class waits for.
- `State::decomposes_now` in `src/decomposition.rs`: when a lower bound's, a part lower
  bound's or a coercion constraint's parts are related. What they
  are related by is `State::decompose_parts`, by the one rule of
  `State::part_relation`.
