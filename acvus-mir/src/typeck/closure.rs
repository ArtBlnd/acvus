//! Lambdas (functions §2-4): captures by move or by reference, once or
//! many, and no recursion. A lambda's flows are claimed at the type level,
//! by union over positions (ownership §5); validate's borrow check verifies
//! the claim.
//!
//! A lambda without `move` captures each name by reference (functions §2),
//! and the language's lambdas carry no `move`, so every capture is by
//! reference. A body that holds its captures only by reference moves none
//! of them out, so every lambda can be called many times (functions §3).

use acvus_ast::{AstId, Binder, Expr, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use super::expr::check_expr;
use super::output::{Capture, Closure};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::infer::Term;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Holds, Mutability, TyKind};

/// The lambda's parameters start as fresh variables, constrained by their
/// uses and by the slot the lambda enters (D27).
pub(super) fn check_lambda<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, params: &[Binder], body: &Expr<S>, _span: Span) -> Term {
    cx.out.closures.insert(
        site,
        Closure {
            captures: Box::new([]),
            calls: Calls::Many,
        },
    );
    let result = cx.fresh();
    cx.scope.push_lambda(site, result);
    let mut param_terms = Vec::with_capacity(params.len());
    for param in params {
        let term = cx.fresh();
        cx.record(param.id, term);
        cx.scope.define(
            param.name,
            Binding {
                binder: param.id,
                term,
            },
        );
        param_terms.push(term);
    }
    let value = check_expr(cx, body, Demand::Read);
    cx.enter(body.id(), value, result, body.span());
    // A body that calls nothing has no effect to order (effects §1).
    let effect = cx
        .scope
        .pop_lambda()
        .unwrap_or_else(|| EffectTerm::Known(Effect::pure()));
    let closure = &cx.out.closures[&site];
    let holds = holds(closure);
    let calls = closure.calls;
    let flows = FlowTerm::Var(cx.infer.fresh_flow());
    cx.held(TyKind::Fn {
        params: param_terms.into(),
        ret: result,
        effect,
        flows,
        calls,
        holds,
    })
}

/// The strongest loan the captures hold (ownership §5).
fn holds(closure: &Closure) -> Holds {
    closure
        .captures
        .iter()
        .map(|captured| match captured.capture {
            Capture::Ref(Mutability::Shared) => Holds::Shared,
            Capture::Ref(Mutability::Mut) => Holds::Mut,
            Capture::Move => panic!("no capture is by move: a lambda carries no `move` (functions §2)"),
        })
        .fold(Holds::Nothing, Holds::join)
}

/// A lambda stored into the binding it captures would call itself: only a
/// `fn` may recurse (functions §4).
pub(super) fn refuse_recursion(cx: &mut Ctxt<'_>, value: Term, name: Astr, span: Span) {
    let recursive = cx.out.closures.iter().any(|(lambda, closure)| {
        cx.out.types.get(lambda) == Some(&value) && closure.captures.iter().any(|captured| captured.name == name)
    });
    if recursive {
        refusal::push(cx, RefusalKind::LambdaRecursion, span);
    }
}
