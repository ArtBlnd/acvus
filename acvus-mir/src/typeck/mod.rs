//! Type checking of one compilation unit by the type system
//! (docs/transparency-constraint-type-system.md): `generate` reads the unit
//! into constraints, `solve` runs the Lifecycle over them, `post` takes the
//! decisions made after inference, and `out` gives what lowering reads.

mod algo;
mod constraint;
mod generate;
mod out;
mod post;
mod rel;
mod solve;
mod term;
