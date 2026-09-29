//! `MatchFacts`: every value's `MatchValue` over one body, conditional on
//! the arms its facts leave (the sparse engine, Wegman and Zadeck), for
//! `check::matchck` (patterns §3). Cost: the sparse engine's; the tags' height
//! is the body's tag names, the constant's 3, the interval's bounded by its
//! widening.
//!
//! An instruction that is not a value operation (`transfer`):
//! - `MakeVariant { tag, .. }`: tags `Of({tag})`; constant and interval top.
//! - `TestVariant { src, tag }`: a `bool` whose constant is, from `src`'s
//!   tags: `true` where they are `Of({tag})`; `false` where they are `Of(s)`
//!   and `tag ∉ s`; bottom where they are `None`; top otherwise, `Any`
//!   included. A `src` that is a reference has tags `Any`.
//! - Every other: top.
//!
//! Which successors run (`executable`):
//! - `Diamond`, `JumpIf`: by the condition's constant, as `ConstantValues`.
//! - `Switch` on `Tag` keys: the set is the tag value's tags, `Any` read as
//!   every variant of its type, a reference's target's where the tag value
//!   is a reference. An arm runs where its key is in the set; `default`
//!   where the set holds a tag no key names.
//! - `Switch` on `Bool` keys: by the constant; `default` where a `bool` value
//!   the constant allows has no key.
//! - `Switch` on `Int` or `Char` keys: by the interval; `default` where the
//!   interval holds a value no key names.
//! - Every other terminator: all its successors.

use acvus_ast::Literal;

use crate::ir::cfg::{BodyId, Module};
use crate::ir::{BinOp, UnaryOp};
use crate::passes::analysis::{Analysis, AnalysisManager, SparseFacts};
use crate::passes::domain::{Constant, ConstantDomain, Interval, IntervalDomain, JoinSemiLattice, Tags, TagsDomain, ValueDomain, Widen};
use crate::ty::CastTy;

/// This analysis's own product: a value's tags, its constant and its
/// interval, run together in one engine so that a fact of any of them may
/// rule an arm out. Each component is its own domain's; the product only
/// pairs them, reducing nothing between them.

#[derive(Debug, Clone, PartialEq)]
pub struct MatchValue {
    pub tags: Tags,
    pub constant: Constant,
    pub interval: Interval,
}

/// Each component joins, compares and widens by its own domain.
impl JoinSemiLattice for MatchValue {
    fn bottom() -> Self {
        todo!("each component's bottom")
    }

    fn join(&mut self, other: &Self) -> bool {
        todo!("each component's join; whether any grew")
    }

    fn le(&self, other: &Self) -> bool {
        todo!("every component's le")
    }
}

impl Widen for MatchValue {
    fn widen(&mut self, next: &Self) -> bool {
        todo!("each component's widen; whether any grew")
    }
}

/// Each value operation is each component domain's.
pub struct MatchDomain {
    pub tags: TagsDomain,
    pub constant: ConstantDomain,
    pub interval: IntervalDomain,
}

impl ValueDomain for MatchDomain {
    type Value = MatchValue;

    fn top(&self) -> MatchValue {
        todo!("each component's top")
    }

    fn constant(&self, literal: &Literal, ty: CastTy) -> MatchValue {
        todo!("each component's constant")
    }

    fn binary(&self, op: BinOp, ty: CastTy, left: &MatchValue, right: &MatchValue) -> MatchValue {
        todo!("each component's binary over the components")
    }

    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &MatchValue) -> MatchValue {
        todo!("each component's unary")
    }

    fn cast(&self, from: CastTy, to: CastTy, operand: &MatchValue) -> MatchValue {
        todo!("each component's cast")
    }
}

pub struct MatchFacts;

impl Analysis for MatchFacts {
    type Facts = SparseFacts<MatchDomain>;

    const READS: &'static [crate::ir::cfg::Aspect] = &[
        crate::ir::cfg::Aspect::Instructions,
        crate::ir::cfg::Aspect::CfgShape,
        crate::ir::cfg::Aspect::Types,
    ];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        todo!("solve_sparse over the body with a SparseAnalysis whose transfer and executable are the module doc's")
    }
}
