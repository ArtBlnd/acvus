# 12 Steps, state, fixpoint and fallback

Rests on the spec's l.295–301. The code is `src/solve.rs`.

## Meaning

A step solves an eligible class or decides an eligible node, and what follows at
once is recorded: merged cycles of lower bounds, splits, and LitVars meeting solved
classes. With no step left, fallback gives each unsolved class with no unbroken
dependency its family's default or `!`, and solving resumes, until nothing falls
back. The program is accepted where no refusal was recorded, every class has a
value, every node is reached, and every value the text or a node makes enter a type
reaches it: a lower bound by `≤`, or by a coercion where an equation decides the
value, and a boundary by a coercion (l.293).

## Readings

1. **One binding group.** So far the model's text is one binding group, taken as
   the run's instance (l.206), so fallback applies to all of it.
2. **Fallback is no equation.** A class that falls back holds its default as a
   class holds a joined value, so its lower bounds reach it by `≤`.
3. **What is checked.** The relations the text and the nodes record; a split's
   relations are parts of those.
