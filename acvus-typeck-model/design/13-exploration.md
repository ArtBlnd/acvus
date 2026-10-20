# 13 Outcome and exploration

Rests on the crate's I2 and I3. The code is `src/explore.rs` and `Outcome` in
`src/solve.rs`.

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
