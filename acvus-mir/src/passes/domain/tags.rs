//! The tag domain (`check::matchck`, patterns §3): which variants a value of
//! an enum, `Option` or `Result` may hold. It is read only for values of
//! those types; the language's value operations do not act on tags, so
//! every one of them gives `Any`.

use std::collections::BTreeSet;

use acvus_ast::Literal;
use acvus_utils::Astr;

use super::{JoinSemiLattice, ValueDomain, Widen};
use crate::ir::{BinOp, UnaryOp};
use crate::ty::CastTy;

/// `None ≤ Of(s) ≤ Of(s ∪ t) ≤ Any`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tags {
    /// No value reaches it yet: the bottom.
    None,
    /// One of these variants.
    Of(BTreeSet<Astr>),
    /// Every variant the value's type has: the top. It is read against the
    /// value's type, so a value the analysis knows nothing of is still held
    /// to its type's variants.
    Any,
}

impl JoinSemiLattice for Tags {
    fn bottom() -> Self {
        Tags::None
    }

    fn join(&mut self, other: &Self) -> bool {
        let joined = match (&*self, other) {
            (_, Tags::None) => return false,
            (Tags::Any, _) => return false,
            (Tags::None, other) => other.clone(),
            (_, Tags::Any) => Tags::Any,
            (Tags::Of(mine), Tags::Of(theirs)) => Tags::Of(mine.union(theirs).copied().collect()),
        };
        let grew = joined != *self;
        *self = joined;
        grew
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Tags::None, _) | (_, Tags::Any) => true,
            (Tags::Any, _) | (Tags::Of(_), Tags::None) => false,
            (Tags::Of(mine), Tags::Of(theirs)) => mine.is_subset(theirs),
        }
    }
}

/// The body's tag names are finite, so the height is: the widening is the
/// join.
impl Widen for Tags {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// The domain: its value operations are no tag's, so each gives `Any`.
pub struct TagsDomain;

impl ValueDomain for TagsDomain {
    type Value = Tags;

    fn top(&self) -> Tags {
        Tags::Any
    }

    fn constant(&self, _literal: &Literal, _ty: CastTy) -> Tags {
        Tags::Any
    }

    fn binary(&self, _op: BinOp, _ty: CastTy, _left: &Tags, _right: &Tags) -> Tags {
        Tags::Any
    }

    fn unary(&self, _op: UnaryOp, _ty: CastTy, _operand: &Tags) -> Tags {
        Tags::Any
    }

    fn cast(&self, _from: CastTy, _to: CastTy, _operand: &Tags) -> Tags {
        Tags::Any
    }
}
