# 13 Outcome and exploration

Rests on the crate's I2 and I3. The code is `src/explore.rs` and `Outcome` in
`src/check.rs`.

## Meaning

The outcome is each text class's value, the partition the text classes fall into,
each node's decision, each checked relation with its coercion, the refusals, and
whether the program is accepted. Exploration takes every step each state allows,
explores a state reached twice once, falls back where no step is left, and collects
the distinct outcomes. Order independence holds of a program where exactly one
outcome is reached.

## Readings

1. **States compared by a fingerprint.** Two states compare by a text that names
   classes by their least member's name. Names of classes made while solving may
   differ between orders (note 06), so two states alike but for those names are
   both explored; this costs states, never outcomes.
2. **A sample of orders.** Where a program has too many states to explore, as
   Example 14 has, orders are sampled: the first step always, the last always,
   and sequences a fixed generator gives, each group solved in turn. One outcome
   over a sample is weaker evidence than over every order.
3. **Groups.** Each group other than the entry is explored to its fixpoints, and
   the next group is explored once for each combination of the fixpoints before
   it (note 09, reading 6).
4. **Checks compare as a set.** The checks of an outcome are ordered by their
   relations' names, so two orders that check the same relations reach one
   outcome.

## Checks

`explore_with` and `sample_with` hand the final states on; the cases' runner
(`design/19-cases.md`) runs the checks on them and states I2 where a case has one
outcome, which it expects unless it writes otherwise.

- **I2.** A program meets I2 where exploration is complete and reaches exactly one
  outcome. An exploration its budget cuts short shows nothing of I2, neither that
  it holds nor that it fails.
- **Ending.** The budget of states is the model's only bound. A program whose
  exploration exceeds it is reported as not ended within that many states, never
  as one whose solving does not end: that would need a measure each step
  decreases, which the model does not have.
- **Final states.** Exploration hands each final state, with the groups solved
  before it, to the checks of notes 14 and 15, and only then reduces it to an
  outcome.
