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
use crate::typeck::{Capture, CaptureOf, Captured, InstanceDecided, InstanceId};

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
    if let Some(binder) = generalized_binder(cx, site) {
        return lower_generalized_lambda(cx, site, binder, params, body, span);
    }
    let fn_ty = cx.ty(site);
    let passed = passed(cx, site, span);
    let lowered = lower_body(cx, Lambda { params, body, span, decided: cx.decided, fn_ty }, &passed);
    let captures = passed.iter().map(|passed| passed.value).collect();
    cx.builder.push(Op::MakeClosure { bodies: vec![lowered], captures }, fn_ty, span)
}

/// Each name the lambda at `site` captures, as the enclosing body passes
/// it: once, where the lambda is written (functions §2).
fn passed(cx: &mut Lcx<'_, '_>, site: AstId, span: Span) -> Vec<Passed> {
    let closure = cx
        .typeck
        .closures
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records every lambda's closure, and not {site:?}"));
    closure.captures.iter().map(|captured| pass(cx, *captured, span)).collect()
}

/// A lambda's body to lower at one function type, over what `decided`
/// says of its sites.
struct Lambda<'e, 'd, S: Slot> {
    params: &'e [Binder],
    body: &'e Expr<S>,
    span: Span,
    decided: super::decided::Decided<'d>,
    fn_ty: Ty,
}

/// The lambda's body, lowered as a body of its own at its function type:
/// its captures its first parameters' homes, and its parameters the rest.
fn lower_body<S: Slot>(cx: &mut Lcx<'_, '_>, lambda: Lambda<'_, '_, S>, passed: &[Passed]) -> crate::ir::cfg::BodyId {
    let Lambda { params, body, span, decided, fn_ty } = lambda;
    let TyKind::Fn { params: param_tys, ret, effect, .. } = cx.kind(fn_ty).clone() else {
        panic!("a lambda's type is a function type")
    };
    let ordered = ordered(known_effect(&effect));
    let named = {
        let mut namer = Namer::new(cx.typeck);
        namer.expr(body);
        namer.finish()
    };
    let decls = cx.decls;
    let typeck = cx.typeck;
    let anyorder = cx.anyorder;
    let (layouts, bodies) = (cx.layouts.clone(), cx.bodies.clone());
    let nested = cx.builder.build_nested(ordered, ret);
    let mut inner = Lcx::new(decls, typeck, nested, anyorder);
    inner.decided = decided;
    inner.layouts = layouts;
    inner.bodies = bodies;
    if anyorder > 0 {
        inner.builder.open_anyorder();
    }
    for passed in passed {
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
    inner.builder.seal().unwrap_or_else(|open| panic!("a lowered lambda's body is sealed whole: {open:?}"))
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

/// The binder of the generalized binding the lambda at `site` is written as
/// a value of, where it is one (`Typeck::generalized`).
fn generalized_binder(cx: &Lcx<'_, '_>, site: AstId) -> Option<AstId> {
    cx.typeck.generalized.iter().find(|(_, lambdas)| lambdas.contains(&site)).map(|(binder, _)| *binder)
}

/// Whether `binder` is a generalized binding (functions §2).
pub(super) fn is_generalized(cx: &Lcx<'_, '_>, binder: AstId) -> bool {
    cx.typeck.generalized.contains_key(&binder)
}

/// One signature a generalized binding has at this body: its function
/// type, and an instance of it, whose view its code is lowered at.
#[derive(Clone, Copy)]
struct Signature {
    ty: Ty,
    instance: InstanceId,
}

/// Every signature `binder` has at this body, in one order the binding
/// fixes: its instances made within this body's instance, or all of them
/// in the run, each signature at its first instance.
fn signatures(cx: &Lcx<'_, '_>, binder: AstId) -> Vec<Signature> {
    let mut instances: Vec<(InstanceId, &InstanceDecided)> = cx
        .typeck
        .instances
        .iter()
        .filter(|(_, instance)| instance.binder == Some(binder) && within_view(cx, instance))
        .map(|(id, instance)| (*id, instance))
        .collect();
    instances.sort_by_key(|(id, _)| id.0);
    let mut out: Vec<Signature> = Vec::new();
    for (id, instance) in instances {
        let ty = *instance.types.get(&binder).unwrap_or_else(|| panic!("typeck records each instance's signature at its binder, and not {id:?}'s"));
        if !out.iter().any(|signature| signature.ty == ty) {
            out.push(Signature { ty, instance: id });
        }
    }
    out
}

/// Whether `instance` is one this body's code is lowered for: made within
/// the instance this body is at, at any depth, or any at the run's level.
fn within_view(cx: &Lcx<'_, '_>, instance: &InstanceDecided) -> bool {
    let Some(view) = cx.decided.instance else {
        return true;
    };
    let mut at = instance.within;
    while let Some(within) = at {
        if within == view {
            return true;
        }
        at = cx.typeck.instances.get(&within).and_then(|instance| instance.within);
    }
    false
}

/// The type a value of `binder` has at this body (functions §2): the one
/// signature's function type where it has one, and the closure of its
/// signatures otherwise.
pub(super) fn layout(cx: &mut Lcx<'_, '_>, binder: AstId) -> Ty {
    let signatures: Vec<Ty> = signatures(cx, binder).iter().map(|signature| signature.ty).collect();
    match signatures[..] {
        [one] => one,
        _ => cx.builder.intern(TyKind::Closures(signatures.into())),
    }
}

/// A lambda of the generalized binding `binder` (functions §2): its
/// environment captured once, here, as any lambda's, and its body lowered
/// once at each distinct signature the binding's instances have, each over
/// the view of an instance of that signature (`Decided::at`). The closure
/// is of a `Closures` type of those signatures, in one order the binding
/// fixes, the same for every lambda written as its value; of one
/// signature, it is that function type.
fn lower_generalized_lambda<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    binder: AstId,
    params: &[Binder],
    body: &Expr<S>,
    span: Span,
) -> ValueId {
    let signatures = signatures(cx, binder);
    let passed = passed(cx, site, span);
    let mut bodies = Vec::with_capacity(signatures.len());
    for signature in &signatures {
        let cached = cx.bodies.borrow().get(&(site, signature.ty)).copied();
        let made = match cached {
            Some(made) => made,
            None => {
                let decided = cx.decided.at(signature.instance);
                let made = lower_body(cx, Lambda { params, body, span, decided, fn_ty: signature.ty }, &passed);
                cx.bodies.borrow_mut().insert((site, signature.ty), made);
                made
            }
        };
        bodies.push(made);
    }
    let ty = layout(cx, binder);
    let captures = passed.iter().map(|passed| passed.value).collect();
    cx.builder.push(Op::MakeClosure { bodies, captures }, ty, span)
}

/// A use of the generalized binding at `use_site`, whose value is
/// `closure`: `Specialize` to its instance's signature, or, where the use
/// is a value entering another generalized binding, to that binding's
/// signatures (functions §2). A value already at those signatures is
/// itself.
pub(super) fn specialize(cx: &mut Lcx<'_, '_>, closure: ValueId, use_site: AstId, span: Span) -> ValueId {
    let view = cx.decided.instance;
    let at_site: Vec<(InstanceId, &InstanceDecided)> = cx
        .typeck
        .instances
        .iter()
        .filter(|(_, instance)| instance.use_site == Some(use_site))
        .map(|(id, instance)| (*id, instance))
        .collect();
    let signature_of = |instance: &InstanceDecided| {
        let binder = instance.binder.expect("typeck records each instance's binder");
        *instance.types.get(&binder).unwrap_or_else(|| panic!("typeck records each instance's signature at its binder"))
    };
    let wanted: Vec<Ty> = match at_site.iter().find(|(_, instance)| instance.within == view) {
        // A use at this body: its instance's one signature.
        Some((_, instance)) => vec![signature_of(instance)],
        // A value entering another generalized binding: made anew within
        // each of that binding's instances, so at each of its signatures.
        None => {
            let entered = at_site
                .iter()
                .filter_map(|(_, instance)| instance.within)
                .find_map(|within| cx.typeck.instances.get(&within).and_then(|outer| outer.binder))
                .unwrap_or_else(|| panic!("typeck records an instance of every use of a generalized binding, and none at {use_site:?}"));
            signatures(cx, entered)
                .iter()
                .map(|signature| {
                    let (_, instance) = at_site
                        .iter()
                        .find(|(_, instance)| instance.within == Some(signature.instance))
                        .unwrap_or_else(|| panic!("each instance of the binding a value enters makes that value's instance within it"));
                    signature_of(instance)
                })
                .collect()
        }
    };
    let ty = match wanted[..] {
        [one] => one,
        _ => cx.builder.intern(TyKind::Closures(wanted.clone().into())),
    };
    if cx.value_ty(closure) == ty {
        return closure;
    }
    cx.builder.push(Op::Specialize { closure, signatures: wanted }, ty, span)
}
