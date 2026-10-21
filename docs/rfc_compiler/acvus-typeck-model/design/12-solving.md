# 12 Steps, state, fixpoint and fallback

Rests on the spec's l.286–292. The code is `src/solve.rs`, which composes the
steps, fallback in `src/fallback.rs`, and acceptance in `src/check.rs`. A step, a
settling and a fallback return the trace of what their phases reported,
`src/trace.rs` (note 19).

## Meaning

A step solves an eligible class, fills one nothing enters, or decides an eligible
node, and what follows at once is recorded: merged cycles of lower bounds and part
lower bounds, splits, and LitVars meeting solved classes. With no step left, the
fixpoint (l.288), fallback gives each unsolved class with no unbroken dependency
its family's default or `!`, and solving resumes, until nothing falls back. The
program is accepted where no refusal was recorded, every class has a value, every
node is reached, and every value the text or a node makes enter a type reaches it:
a lower bound by `≤`, and a boundary by a coercion (l.282).

## Readings

1. **One binding group.** So far the model's text is one binding group, taken as
   the run's instance (l.201), so fallback applies to all of it.
2. **Fallback is no equation.** A class that falls back holds its default as a
   class holds a joined value, so its lower bounds reach it by `≤`.
3. **What is checked.** The relations the text and the nodes record; a split's
   relations are parts of those.

## Left open for change

- `State::splits_now` in `src/split.rs`: when a lower bound's, a part lower
  bound's or a boundary's parts are related (l.280; `holes.md`, T2). What they
  are related by is `State::split_parts`, by the one rule of
  `State::part_relation` (`holes.md`, H3).
