//! The constant domain: bottom, one known value, or top; height 3, so its
//! widening is its join, stated. Abstract operations compute as the language
//! does (operators §1): an operation that would trap on the known operands
//! is top, since the trap is the run's, not a value.

use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Bottom,
    Known(acvus_ast::Literal),
    Top,
}

pub struct ConstantDomain;
