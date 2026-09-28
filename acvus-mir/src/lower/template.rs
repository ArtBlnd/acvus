//! A template's body: one `String` slot it returns; text is appended as a
//! literal and `{{ x }}` through `core::display(&x, &mut out)` (programs §1,
//! operators §2); `%` lines lower as a script's statements.

use acvus_ast::{AstId, Expr, Literal, Slot, Span};

use super::call::{call_reached, convert};
use super::ctxt::Lcx;
use super::place::{Operand, lend, operand};
use crate::ir::cfg::Op;
use crate::ir::RefTarget;
use crate::ty::{Mutability, TyKind};

/// An `Append`: a text line, which the parser writes as a string literal,
/// is appended as that text; any other value is formatted onto the output
/// by the `core::display` impl typeck reached at `site`.
pub(super) fn lower_append<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, expr: &Expr<S>, span: Span) {
    let out = cx.out.expect("an `Append` is a template's, and a template's body holds its output");
    let string = cx.value_ty(out);
    let out_ty = cx.ref_ty(Mutability::Mut, string);
    let target = cx.builder.push(Op::Ref { target: RefTarget::Var(out), path: Vec::new(), mutability: Mutability::Mut }, out_ty, span);
    if let Expr::Literal { value: Literal::String(text), span, .. } = expr {
        let str_ty = cx.builder.intern(TyKind::Str);
        let text_ty = cx.ref_ty(Mutability::Shared, str_ty);
        let part = cx.builder.push(Op::ConstStr { text: text.clone() }, text_ty, *span);
        cx.builder.push_unit(Op::StringAppend { target, part }, *span);
        return;
    }
    let shown = operand(cx, expr, Mutability::Shared);
    let shown = lend(cx, shown, Mutability::Shared, expr.span());
    let shown = convert(cx, expr.id(), shown, expr.span());
    let target = convert(cx, site, target, span);
    call_reached(cx, site, vec![shown, target], span);
}

/// The template's output, an empty `String` to begin with.
pub(super) fn open_output(cx: &mut Lcx<'_, '_>, span: Span) {
    let string = cx.builder.intern(TyKind::String);
    let out = cx.builder.slot(string, span);
    let empty = cx.constant(Literal::String(String::new()), string, span);
    cx.builder.push_unit(Op::Assign { target: RefTarget::Var(out), path: Vec::new(), value: empty, restores: false }, span);
    cx.out = Some(out);
}

/// The template's output, taken to be returned.
pub(super) fn take_output(cx: &mut Lcx<'_, '_>, span: Span) -> crate::ir::ValueId {
    let out = cx.out.expect("a template's body holds its output");
    let ty = cx.value_ty(out);
    let place = super::place::Place { target: RefTarget::Var(out), path: Vec::new(), ty };
    super::place::read(cx, Operand::Place(place), span)
}
