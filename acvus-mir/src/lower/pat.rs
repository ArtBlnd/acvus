//! Patterns (patterns §1). A `match` whose arms all test one enum's tags or one kind of literal is one `Switch` dispatch, with a default where it has a catch-all, and its join held by the `Switch`; any other `match` tests its arms one at a time, in order, within the same region and to the same join. Exhaustiveness is `check`'s.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_match<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, scrutinee: &Expr<S>, arms: &[MatchExprArm<S>]) -> ValueId {
    todo!("lower::pat::lower_match")
}

/// Tests `pattern` against `value` and binds its names; the result is the
/// `bool` of the test, and the binds stand where it holds.
pub(super) fn lower_test<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, value: ValueId) -> ValueId {
    todo!("lower::pat::lower_test")
}
