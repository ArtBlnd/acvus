//! A template's body: one `String` slot it returns; its text is appended
//! as a literal and `{{ x }}` through `core::display(&x, &mut out)` (programs §1,
//! operators §2); `%` lines lower as a script's statements.

use acvus_ast::{AstId, Expr, Literal, Slot, Span};

use super::call::{call_reached, convert};
use super::ctxt::{Lcx, Lowered};
use super::place::{Operand, lend, operand};
use crate::ir::cfg::Op;
use crate::ir::{RefTarget, ValueId};
use crate::ty::{Mutability, TyKind};

/// The output as a `&mut String`, for one append: a lend of its slot.
fn output(cx: &mut Lcx<'_, '_>, span: Span) -> Lowered<ValueId> {
    let out = cx.out.expect("a template's text and `{{ }}` append to its output, which its body holds");
    let string = cx.value_ty(out);
    let ty = cx.ref_ty(Mutability::Mut, string);
    let op = Op::Ref { target: RefTarget::Var(out), path: Vec::new(), mutability: Mutability::Mut };
    cx.emit(op, ty, span)
}

/// A template's text, appended as it is written (programs §1).
pub(super) fn lower_text(cx: &mut Lcx<'_, '_>, text: &str, span: Span) -> Lowered<()> {
    let target = output(cx, span)?;
    let str_ty = cx.intern(TyKind::Str);
    let text_ty = cx.ref_ty(Mutability::Shared, str_ty);
    let part = cx.made(Op::ConstStr { text: text.to_string() }, text_ty, span);
    cx.emit_unit(Op::StringAppend { target, part }, span);
    Ok(())
}

/// `{{ x }}`: the value formatted onto the output by the `core::display`
/// impl typeck reached at `site` (operators §2, §7).
pub(super) fn lower_append<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, expr: &Expr<S>, span: Span) -> Lowered<()> {
    let shown = operand(cx, expr, Mutability::Shared)?;
    let shown = lend(cx, shown, Mutability::Shared, expr.span())?;
    let shown = convert(cx, expr.id(), shown, expr.span())?;
    let target = output(cx, span)?;
    let target = convert(cx, site, target, span)?;
    call_reached(cx, site, vec![shown, target], span)?;
    Ok(())
}

/// The template's output, an empty `String` to begin with.
pub(super) fn open_output(cx: &mut Lcx<'_, '_>, span: Span) {
    let string = cx.intern(TyKind::String);
    let out = cx.slot(string, span);
    let empty = cx.constant(Literal::String(String::new()), string, span);
    cx.emit_unit(Op::Assign { target: RefTarget::Var(out), path: Vec::new(), value: empty, restores: false }, span);
    cx.out = Some(out);
}

/// The template's output, taken to be returned.
pub(super) fn take_output(cx: &mut Lcx<'_, '_>, span: Span) -> Lowered<ValueId> {
    let out = cx.out.expect("a template's body holds its output");
    let ty = cx.value_ty(out);
    let place = super::place::Place { target: RefTarget::Var(out), path: Vec::new(), ty };
    super::place::read(cx, Operand::Place(place), span)
}
