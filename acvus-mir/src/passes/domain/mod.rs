//! Value domains (passes-structure §1). A domain knows values and the
//! language's value operations, and nothing of CFG, storage or calls.

mod affine;
mod constant;
mod interval;

pub use affine::{Affine, AffineDomain, Combination, Limit, Symbol, bounds_on};
pub use constant::{Constant, ConstantDomain};
pub use interval::{Bound, Interval, IntervalDomain};

use std::cmp::Ordering;

use acvus_ast::Literal;

use crate::ir::{BinOp, Overflow, UnaryOp};
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
/// outside the body is. An operation names the type of its operands, `ty`,
/// since width and signedness are part of what it computes; an operation
/// whose operands have no `CastTy` is not a domain's.
pub trait ValueDomain {
    type Value: Widen;

    fn top(&self) -> Self::Value;
    fn constant(&self, literal: &Literal, ty: CastTy) -> Self::Value;
    fn binary(&self, op: BinOp, ty: CastTy, left: &Self::Value, right: &Self::Value) -> Self::Value;
    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &Self::Value) -> Self::Value;
    fn cast(&self, from: CastTy, to: CastTy, operand: &Self::Value) -> Self::Value;
}

/// A value the language's value operations act on, as a literal spells it
/// at its type.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scalar {
    Int(i128),
    F64(f64),
    Char(char),
    Bool(bool),
}

impl Scalar {
    /// The value `literal` spells at `ty`. An integer literal is its value,
    /// never a bit pattern read at the width (D60).
    ///
    /// # Panics
    /// `literal` is not a value of `ty`, or an integer does not fit its
    /// width: whoever made the constant made a defect.
    /// A shift's amount: the integer it is, at its own width, which is not
    /// the shifted value's (D79). Its own constant checked that it fits.
    ///
    /// # Panics
    /// `literal` is not an integer: a shift's amount is one.
    fn amount(literal: &Literal) -> Scalar {
        match literal.desugared() {
            Literal::Int(n) => Scalar::Int(n),
            other => panic!("a shift's amount is an integer, not {other:?}"),
        }
    }

    fn of(literal: &Literal, ty: CastTy) -> Scalar {
        if let Literal::IntOf(suffixed) = literal {
            assert!(
                ty == CastTy::Int(suffixed.width.into()),
                "the constant {literal:?} names a width other than its type {ty:?}"
            );
        }
        match (literal.desugared(), ty) {
            (Literal::Int(n), CastTy::Int(width)) => {
                assert!(width.holds(n), "the constant {n} does not fit its type {}", width.name());
                Scalar::Int(n)
            }
            (Literal::Float(x), CastTy::F64) => Scalar::F64(x),
            (Literal::Char(c), CastTy::Char) => Scalar::Char(c),
            (Literal::Bool(b), CastTy::Bool) => Scalar::Bool(b),
            (literal, ty) => panic!("the constant {literal:?} is not a value of its type {ty:?}"),
        }
    }

    fn literal(self) -> Literal {
        match self {
            Scalar::Int(n) => Literal::Int(n),
            Scalar::F64(x) => Literal::Float(x),
            Scalar::Char(c) => Literal::Char(c),
            Scalar::Bool(b) => Literal::Bool(b),
        }
    }
}

/// An operation the language does not define at its operands' type: the
/// IR that holds it is ill-typed.
fn not_an_operation(op: impl std::fmt::Debug, ty: CastTy) -> ! {
    panic!("{op:?} is not an operation of the language at {}", ty.name())
}

/// A binary operation, by what it computes: a comparison, which every
/// scalar type has and which yields a `bool`, or arithmetic at the
/// operands' type.
enum Operation {
    Compare(Comparison),
    Compute(Arithmetic),
}

#[derive(Debug, Clone, Copy)]
enum Comparison {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

#[derive(Debug, Clone, Copy)]
enum Arithmetic {
    Add(Overflow),
    Sub(Overflow),
    Mul(Overflow),
    Div,
    Mod,
    Shl(Overflow),
    Shr(Overflow),
    BitAnd,
    BitOr,
    Xor,
    And,
    Or,
    Min,
    Max,
}

impl Operation {
    fn of(op: BinOp) -> Operation {
        match op {
            BinOp::Eq => Operation::Compare(Comparison::Eq),
            BinOp::Neq => Operation::Compare(Comparison::Neq),
            BinOp::Lt => Operation::Compare(Comparison::Lt),
            BinOp::Gt => Operation::Compare(Comparison::Gt),
            BinOp::Lte => Operation::Compare(Comparison::Lte),
            BinOp::Gte => Operation::Compare(Comparison::Gte),
            BinOp::Add(overflow) => Operation::Compute(Arithmetic::Add(overflow)),
            BinOp::Sub(overflow) => Operation::Compute(Arithmetic::Sub(overflow)),
            BinOp::Mul(overflow) => Operation::Compute(Arithmetic::Mul(overflow)),
            BinOp::Div => Operation::Compute(Arithmetic::Div),
            BinOp::Mod => Operation::Compute(Arithmetic::Mod),
            BinOp::Shl(overflow) => Operation::Compute(Arithmetic::Shl(overflow)),
            BinOp::Shr(overflow) => Operation::Compute(Arithmetic::Shr(overflow)),
            BinOp::BitAnd => Operation::Compute(Arithmetic::BitAnd),
            BinOp::BitOr => Operation::Compute(Arithmetic::BitOr),
            BinOp::Xor => Operation::Compute(Arithmetic::Xor),
            BinOp::And => Operation::Compute(Arithmetic::And),
            BinOp::Or => Operation::Compute(Arithmetic::Or),
            BinOp::Min => Operation::Compute(Arithmetic::Min),
            BinOp::Max => Operation::Compute(Arithmetic::Max),
        }
    }
}

impl Comparison {
    /// Whether the comparison holds of two operands so ordered; `None` is
    /// an unordered pair, of which only `!=` holds.
    fn holds(self, ordering: Option<Ordering>) -> bool {
        let Some(ordering) = ordering else {
            return matches!(self, Comparison::Neq);
        };
        match self {
            Comparison::Eq => ordering == Ordering::Equal,
            Comparison::Neq => ordering != Ordering::Equal,
            Comparison::Lt => ordering == Ordering::Less,
            Comparison::Gt => ordering == Ordering::Greater,
            Comparison::Lte => ordering != Ordering::Greater,
            Comparison::Gte => ordering != Ordering::Less,
        }
    }
}
