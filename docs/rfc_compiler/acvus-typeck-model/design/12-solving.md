# 12 Steps, state, fixpoint and fallback

The code is `src/solve.rs`, which composes the steps, fallback in `src/fallback.rs`,
and acceptance in `src/check.rs`. A step, a settling and a fallback return the trace
of what their phases reported, `src/trace.rs`.

## Meaning

A step solves an eligible class, fills one nothing enters, or decides an eligible
disjunctive constraint, and what follows at once is recorded: merged cycles of lower bounds and part
lower bounds, decompositions, and LitVars entering solved classes. With no step left, the
fixpoint, fallback gives each unsolved class with no unbroken dependency
its family's default or `!`, and solving resumes, until nothing falls back. The
program is accepted where no refusal was recorded, every class has a value, every
disjunctive constraint is reached, and every value the text or a disjunctive constraint makes enter a type reaches it:
a lower bound by `≤`, and a coercion constraint by a coercion.

## Readings

1. **One binding group.** So far the model's text is one binding group, taken as
   the run's instance, so fallback applies to all of it.
2. **Fallback is no equation.** A class that falls back holds its default as a
   class holds a joined value, so its lower bounds reach it by `≤`.
3. **What is checked.** The relations the text and the disjunctive constraints record; a decomposition's
   relations are parts of those.

## Left open for change

- `State::decomposes_now` in `src/decomposition.rs`: when a lower bound's, a part lower
  bound's or a coercion constraint's parts are related. What they
  are related by is `State::decompose_parts`, by the one rule of
  `State::part_relation`.
