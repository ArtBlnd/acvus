//! An executable definition of how the transparency-constraint type system
//! solves a program. The system is specified in
//! `docs/transparency-constraint-type-system-kr.md`, and a line number in this
//! crate is that text's, which its English text keeps.

pub mod certificate;
pub mod class;
pub mod coerce;
pub mod decl;
pub mod dep;
#[cfg(test)]
mod examples;
pub mod explore;
#[cfg(test)]
mod fixture;
pub mod instance;
pub mod invariant;
pub mod istar;
pub mod join;
pub mod lit;
pub mod node;
pub mod program;
pub mod read;
pub mod relation;
pub mod sig;
pub mod solve;
pub mod ty;
pub mod value;
