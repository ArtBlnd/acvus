//! Contexts (context §3): a context that is not volatile lives in a slot, loaded when the body is entered, committed and loaded again around a call whose effect touches it, and committed at return; a volatile one is loaded at each read and committed at each write. Removing unread loads and unwritten commits is the required step's.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_context_read(cx: &mut Lcx<'_, '_>, site: AstId, context: acvus_utils::QualifiedRef) -> ValueId {
    todo!("lower::context::lower_context_read")
}

pub(super) fn bracket_call(cx: &mut Lcx<'_, '_>, callee_touches: &[acvus_utils::QualifiedRef], call: impl FnOnce(&mut Lcx<'_, '_>) -> ValueId) -> ValueId {
    todo!("lower::context::bracket_call")
}
