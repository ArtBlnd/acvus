//! Value domains (passes-structure §1). A domain knows values and the
//! language's value operations, and nothing of CFG, storage or calls.

mod constant;
mod interval;

pub use constant::{Constant, ConstantDomain};
pub use interval::{Bound, Interval, IntervalDomain};

use acvus_ast::Literal;

use crate::ir::{BinOp, UnaryOp};
use crate::ty::CastTy;

/// The lattice every engine runs over. `join` returns whether `self` grew,
/// so an engine knows when a fixpoint is reached. Laws, checked by every
/// domain's tests: join is associative, commutative and idempotent;
/// `bottom` is its unit; `le` agrees with join.
pub trait JoinSemiLattice: Clone + PartialEq {
    fn bottom() -> Self;
    fn join(&mut self, other: &Self) -> bool;
    fn le(&self, other: &Self) -> bool;
}

/// Every lattice an engine runs names its widening, applied at loop heads, so
/// every fixpoint is bounded (compilation §2). A lattice of finite height
/// states that its widening is its join; nothing defaults it.
pub trait Widen: JoinSemiLattice {
    fn widen(&mut self, next: &Self) -> bool;
}

/// A domain of values: the lattice, and the abstract value operations. Each
/// operation over-approximates the concrete one; `top` is what a value from
/// outside the body is.
pub trait ValueDomain {
    type Value: Widen;

    fn top(&self) -> Self::Value;
    fn constant(&self, literal: &Literal) -> Self::Value;
    fn binary(&self, op: BinOp, left: &Self::Value, right: &Self::Value) -> Self::Value;
    fn unary(&self, op: UnaryOp, operand: &Self::Value) -> Self::Value;
    fn cast(&self, from: CastTy, to: CastTy, operand: &Self::Value) -> Self::Value;
}
