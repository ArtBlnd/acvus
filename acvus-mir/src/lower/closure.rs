//! A lambda is a `MakeClosure` of a body of its own, lowered over a nested
//! Builder: ordered where its effect is not pure. A capture by move passes
//! the value; by reference, a `&T` value, or the reborrow of a reference
//! (functions §2, types §4). Inside, a name captured by reference lives in
//! the storage its reference names, and every other captured name and
//! every parameter in a slot of the body's own. A lambda written inside an
//! `anyorder` belongs to it: its body's effects each consume its own entry
//! token (effects §3).

use acvus_ast::{AstId, Binder, Expr, Slot, Span};

use super::call::convert;
use super::context::{Namer, enter_body};
use super::control::return_value;
use super::ctxt::{Home, Lcx, ordered};
use super::expr::lower_expr;
use super::place::{Operand, Place, assign, binding_place, input_place, input_ty, lend, read};
use crate::ir::cfg::{Op, Terminator};
use crate::ir::{RefTarget, ValueId, known_effect};
use crate::ty::{Ty, TyKind};
use crate::typeck::{Capture, CaptureOf, Captured};

/// One captured name as the enclosing body passes it.
struct Passed {
    captured: Captured,
    value: ValueId,
    ty: Ty,
    /// The captured binding's own type.
    binding_ty: Ty,
    span: Span,
}

pub(super) fn lower_lambda<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, params: &[Binder], body: &Expr<S>, span: Span) -> ValueId {
    // G: a lambda of a generalized binding used at more than one instance
    // has code per instance, which lower does not make yet.
    // TODO(G, lower): specialize the body to each of `Typeck::instances`
    // over the one environment (functions §2).
    if cx.typeck.instances.values().any(|instance| instance.types.contains_key(&body.id())) {
        panic!("the lambda at {site:?} is generalized and used at several instances; specializing its code is the lowerer's next round (functions §2)");
    }
    let fn_ty = cx.ty(site);
    let TyKind::Fn { params: param_tys, ret, effect, .. } = cx.kind(fn_ty).clone() else {
        panic!("a lambda's type is a function type")
    };
    let ordered = ordered(known_effect(&effect));
    let closure = cx
        .typeck
        .closures
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records every lambda's closure, and not {site:?}"));
    let passed: Vec<Passed> = closure.captures.iter().map(|captured| pass(cx, *captured, span)).collect();
    let named = {
        let mut namer = Namer::new(cx.typeck);
        namer.expr(body);
        namer.finish()
    };
    let decls = cx.decls;
    let typeck = cx.typeck;
    let anyorder = cx.anyorder;
    let nested = cx.builder.build_nested(ordered, ret);
    let mut inner = Lcx::new(decls, typeck, nested, anyorder);
    if anyorder > 0 {
        inner.builder.open_anyorder();
    }
    for passed in &passed {
        let capture = inner.builder.capture(passed.captured.name, passed.ty, passed.span);
        let by_reference = matches!(passed.captured.capture, Capture::Ref(_));
        let home = match by_reference && !matches!(inner.kind(passed.binding_ty), TyKind::Ref(..)) {
            true => Home::Through(capture),
            false => {
                let slot = inner.builder.slot(passed.ty, passed.span);
                assign(&mut inner, Place { target: RefTarget::Var(slot), path: Vec::new(), ty: passed.ty }, capture, passed.span);
                Home::Slot(slot)
            }
        };
        match passed.captured.binder {
            CaptureOf::Binding(binder) => inner.bind(binder, home, passed.span),
            CaptureOf::Input(name) => {
                inner.inputs.insert(name, home);
            }
        }
    }
    for (param, ty) in params.iter().zip(param_tys.iter()) {
        let value = inner.builder.param(param.name, *ty, param.span);
        let slot = inner.builder.slot(*ty, param.span);
        assign(&mut inner, Place { target: RefTarget::Var(slot), path: Vec::new(), ty: *ty }, value, param.span);
        inner.bind(param.id, Home::Slot(slot), param.span);
    }
    enter_body(&mut inner, &named);
    let value = lower_expr(&mut inner, body);
    let value = convert(&mut inner, body.id(), value, body.span());
    return_value(&mut inner, value, span);
    if anyorder > 0 {
        inner.builder.close_anyorder();
    }
    inner.builder.terminate(Terminator::Diverge, span);
    let lowered = inner.builder.seal().unwrap_or_else(|open| panic!("a lowered lambda's body is sealed whole: {open:?}"));
    let captures = passed.iter().map(|passed| passed.value).collect();
    cx.builder.push(Op::MakeClosure { body: lowered, captures }, fn_ty, span)
}

/// What the enclosing body passes for `captured`: the value by move, a
/// reference to the binding by reference.
fn pass(cx: &mut Lcx<'_, '_>, captured: Captured, span: Span) -> Passed {
    let (place, binder_span) = match captured.binder {
        CaptureOf::Binding(binder) => {
            let place = binding_place(cx, binder, cx.ty(binder));
            let binder_span = *cx
                .binder_spans
                .get(&binder)
                .unwrap_or_else(|| panic!("the binder {binder:?} a lambda captures is bound before the lambda"));
            (place, binder_span)
        }
        CaptureOf::Input(name) => (input_place(cx, name, input_ty(cx, name)), span),
    };
    let binding_ty = place.ty;
    let value = match captured.capture {
        Capture::Move => read(cx, Operand::Place(place), span),
        Capture::Ref(mutability) => lend(cx, Operand::Place(place), mutability, span),
    };
    Passed { captured, value, ty: cx.value_ty(value), binding_ty, span: binder_span }
}
