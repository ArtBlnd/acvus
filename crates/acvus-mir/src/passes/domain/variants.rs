//! The variant domain (`check::matchck`, patterns §3): which variants a
//! value of an enum, `Option` or `Result` may hold. It is read only for
//! values of those types; the language's value operations do not act on
//! variants, so every one of them gives `Any`.

use std::collections::BTreeSet;

use acvus_ast::Literal;

use super::{JoinSemiLattice, ValueDomain, Widen};
use crate::ir::{BinOp, UnaryOp};
use crate::ty::CastTy;
use crate::ty::structural::VariantKey;

/// `None ≤ Of(s) ≤ Of(s ∪ t) ≤ Any`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Variants {
    /// No value reaches it yet: the bottom.
    None,
    /// One of these variants.
    Of(BTreeSet<VariantKey>),
    /// Every variant the value's type has: the top. It is read against the
    /// value's type, so a value the analysis knows nothing of is still held
    /// to its type's variants.
    Any,
}

impl JoinSemiLattice for Variants {
    fn bottom() -> Self {
        Variants::None
    }

    fn join(&mut self, other: &Self) -> bool {
        let joined = match (&*self, other) {
            (_, Variants::None) => return false,
            (Variants::Any, _) => return false,
            (Variants::None, other) => other.clone(),
            (_, Variants::Any) => Variants::Any,
            (Variants::Of(mine), Variants::Of(theirs)) => Variants::Of(mine.union(theirs).copied().collect()),
        };
        let grew = joined != *self;
        *self = joined;
        grew
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Variants::None, _) | (_, Variants::Any) => true,
            (Variants::Any, _) | (Variants::Of(_), Variants::None) => false,
            (Variants::Of(mine), Variants::Of(theirs)) => mine.is_subset(theirs),
        }
    }
}

/// A body's variants are finite, so the height is: the widening is the
/// join.
impl Widen for Variants {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// The domain: its value operations act on no variant, so each gives `Any`.
pub struct VariantsDomain;

impl ValueDomain for VariantsDomain {
    type Value = Variants;

    fn top(&self) -> Variants {
        Variants::Any
    }

    fn constant(&self, _literal: &Literal, _ty: CastTy) -> Variants {
        Variants::Any
    }

    fn binary(&self, _op: BinOp, _ty: CastTy, _left: &Variants, _right: &Variants) -> Variants {
        Variants::Any
    }

    fn unary(&self, _op: UnaryOp, _ty: CastTy, _operand: &Variants) -> Variants {
        Variants::Any
    }

    fn cast(&self, _from: CastTy, _to: CastTy, _operand: &Variants) -> Variants {
        Variants::Any
    }
}
