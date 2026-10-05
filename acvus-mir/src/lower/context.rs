//! Contexts (context §3): a context is a place outside the body, which a
//! body names by `RefTarget::Context` and reads, takes, lends and assigns
//! as any place. A body states the contexts it names, each with the
//! volatility its declaration states; where the page is
//! seen beyond the program, at a call that may touch a context and at a
//! return, is `passes::analysis::Loans`' to record, not lower's. A
//! volatile context's place operations are effects, on the order chain.
//!
//! The contexts a body names, and the run's inputs (programs §4), are
//! typeck's (`Typeck::contexts`, `Typeck::inputs`): each input is a
//! parameter of the entry body, in the order of the unit's parameters, held
//! in a slot, which a lambda that reads it captures.

use acvus_ast::Span;
use acvus_utils::QualifiedRef;

use super::ctxt::{Declared, Home, Lcx, Lowered};
use super::place::{Operand, Place as LPlace, assign, read};
use crate::ir::{RefTarget, ValueId};
use crate::ty::Ty;
use crate::typeck::RunInput;

/// The entry body's parameters: the run's inputs (`Typeck::inputs`).
pub(super) fn input_params(inputs: &[RunInput]) -> Vec<Declared> {
    inputs.iter().map(|RunInput { name, ty, span }| Declared { name: *name, ty: *ty, span: *span }).collect()
}

/// The entry body's inputs, `params` as `input_params` declares them: each
/// held in a slot of its own. A lambda reaches an input by capturing it.
pub(super) fn enter_inputs(cx: &mut Lcx<'_, '_>, inputs: &[RunInput], params: &[ValueId]) {
    for (RunInput { name, ty, span }, param) in inputs.iter().zip(params) {
        let slot = cx.slot(*ty, *span);
        assign(cx, slot_place(slot, *ty), *param, *span);
        cx.inputs.insert(*name, Home::Slot(slot));
    }
}

fn slot_place(slot: ValueId, ty: Ty) -> LPlace {
    LPlace { target: RefTarget::Var(slot), path: Vec::new(), ty }
}

/// The whole of `context`, a `ty`, as a place.
pub(super) fn context_place(context: QualifiedRef, ty: Ty) -> LPlace {
    LPlace { target: RefTarget::Context(context), path: Vec::new(), ty }
}

/// `@x` read: a take of the context, which copies it where its type does.
pub(super) fn lower_context_read(cx: &mut Lcx<'_, '_>, context: QualifiedRef, ty: Ty, span: Span) -> Lowered<ValueId> {
    read(cx, Operand::Place(context_place(context, ty)), span)
}

/// A store into `context` of a value a pattern bound (`@x` in a pattern).
pub(super) fn store(cx: &mut Lcx<'_, '_>, context: QualifiedRef, value: ValueId, span: Span) {
    let ty = cx.value_ty(value);
    assign(cx, context_place(context, ty), value, span);
}
