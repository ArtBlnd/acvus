//! An executable definition of how the transparency-constraint type system
//! solves a program.

#[cfg(test)]
mod cases;
pub mod certificate;
pub mod check;
pub mod class;
pub mod coerce;
pub mod cut;
pub mod decl;
pub mod decision;
pub mod dep;
pub mod explore;
pub mod fallback;
#[cfg(test)]
mod fixture;
pub mod instance;
pub mod istar;
pub mod join;
pub mod lit;
pub mod node;
pub mod program;
pub mod range;
pub mod read;
pub mod record;
pub mod relation;
pub mod sig;
pub mod solve;
pub mod split;
pub mod trace;
pub mod ty;
pub mod unify;
pub mod value;
