//! An executable definition of how the transparency-constraint type system
//! solves a program. The system is specified in
//! `docs/transparency-constraint-type-system-kr.md`, and a line number in this
//! crate is that text's, which its English text keeps.

pub mod coerce;
pub mod decl;
#[cfg(test)]
mod fixture;
pub mod join;
pub mod lit;
pub mod read;
pub mod sig;
pub mod ty;
