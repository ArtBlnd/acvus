//! Lambdas (functions §2-4): captures by move or by reference, once or
//! many, and no recursion. A lambda's flows are claimed at the type level,
//! by union over positions (ownership §5); validate's borrow check verifies
//! the claim.
//!
//! `move |…| -> e` captures each name by move, and a lambda without `move`
//! by reference (functions §2). A `move` lambda's body that uses a
//! captured value whole where its type does not copy moves it out, and the
//! lambda is called once; every other lambda many times (functions §3).
//! A body holding its captures by reference moves none of them out: a move
//! out of a borrow is the move check's to refuse.

use acvus_ast::{AstId, Binder, Expr, RefKind, Slot, Span};
use acvus_utils::Astr;
use rustc_hash::FxHashSet;

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::expr::check_expr;
use super::output::{Capture, Closure};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::infer::Term;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Holds, LifetimesOf, Mutability, ObjectTy, TyKind};

/// The lambda's parameters start as fresh variables, constrained by their
/// uses and by the slot the lambda enters (D27).
pub(super) fn check_lambda<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    captures_by_move: bool,
    params: &[Binder],
    body: &Expr<S>,
) -> Term {
    cx.out.closures.insert(
        site,
        Closure {
            captures: Box::new([]),
            calls: Calls::Many,
        },
    );
    let result = cx.fresh();
    cx.scope.push_lambda(site, captures_by_move, result);
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

/// The strongest loan the captures by reference hold (ownership §5). A
/// capture by move holds what its value's type holds, which
/// `Decision::MovedCapture` joins in once that type is solved.
fn holds(closure: &Closure) -> Holds {
    closure
        .captures
        .iter()
        .map(|captured| match captured.capture {
            Capture::Ref(Mutability::Shared) => Holds::Shared,
            Capture::Ref(Mutability::Mut) => Holds::Mut,
            Capture::Move => Holds::Nothing,
        })
        .fold(Holds::Nothing, Holds::join)
}

/// A value of `expr` read whole, where `expr` is a place rooted at a name
/// some `move` lambda around it captured: that lambda's body may move it
/// out (functions §3), decided once its type is solved.
pub(super) fn note_whole_use<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>, term: Term) {
    let Some(root) = place_root(expr) else {
        return;
    };
    let Some(found) = cx.scope.lookup(root) else {
        return;
    };
    for lambda in found.crossed {
        if cx.scope.captures_by_move(lambda) {
            let span = expr.span();
            cx.decisions.push(Decision::MovesOut { lambda, span, term });
        }
    }
}

/// The binding a place expression is rooted at.
fn place_root<S: Slot>(expr: &Expr<S>) -> Option<Astr> {
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } if name.namespace.is_none() => Some(name.name),
        Expr::FieldAccess { object, .. } | Expr::Index { object, .. } => place_root(object),
        Expr::Paren { inner, .. } => place_root(inner),
        _ => None,
    }
}

/// The strongest loan a value of a solved term holds at any position
/// (ownership §5): a reference's own, a function value's captures', and
/// an extern-defined type's declared lifetime, which is exclusive (D56).
///
/// # Panics
/// `term` holds an unbound variable: the decision reads it only once none
/// is left.
pub(super) fn strongest_loan(cx: &Ctxt<'_>, term: Term) -> Holds {
    let mut strongest = Holds::Nothing;
    let mut stack = vec![term];
    let mut seen = FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        let kind = cx.infer.types().kind(at);
        let here = match kind {
            TyKind::Ref(Mutability::Shared, _) => Holds::Shared,
            TyKind::Ref(Mutability::Mut, _) => Holds::Mut,
            TyKind::Fn { holds, .. } => *holds,
            TyKind::UserDefined { id, .. } if cx.decls.lifetimes_of(*id) > 0 => Holds::Mut,
            TyKind::Var(var) => panic!("strongest_loan: {var:?} is unbound on a solved term"),
            _ => Holds::Nothing,
        };
        strongest = strongest.join(here);
        if !matches!(kind, TyKind::Fn { .. }) {
            kind.for_each_child(|child| stack.push(child));
        }
    }
    strongest
}

/// Whether a value of a solved term copies (ownership §1).
///
/// # Panics
/// `term` holds an unbound variable: the decision reads it only once none
/// is left.
pub(super) fn copies(cx: &Ctxt<'_>, term: Term) -> bool {
    let mut stack = vec![term];
    let mut seen = FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        match cx.infer.types().kind(at) {
            TyKind::Int(_) | TyKind::Float | TyKind::Bool | TyKind::Char | TyKind::Unit => {}
            TyKind::Ref(Mutability::Shared, _) => {}
            kind @ (TyKind::Tuple(_)
            | TyKind::Array(..)
            | TyKind::Option(_)
            | TyKind::Result(..)
            | TyKind::Object(ObjectTy::Structural(..))) => kind.for_each_child(|child| stack.push(child)),
            TyKind::Var(var) => panic!("copies: {var:?} is unbound on a solved term"),
            _ => return false,
        }
    }
    true
}

/// Joins into `lambda`'s function type what solving learned of it: that
/// it is called once, or that it holds a loan (types §2: calls and holds
/// join).
///
/// NOTE: a slot the lambda entered before this took its own copy of the
/// type (D14), and does not learn it.
pub(super) fn relax(cx: &mut Ctxt<'_>, lambda: AstId, calls: Calls, holds: Holds, span: Span) {
    let term = *cx
        .out
        .types
        .get(&lambda)
        .expect("a lambda's type is recorded when it is checked");
    let TyKind::Fn {
        params,
        ret,
        effect,
        flows,
        calls: was_calls,
        holds: was_holds,
    } = cx.infer.types().kind(cx.infer.shallow(term)).clone()
    else {
        assert!(
            !cx.out.refusals.is_empty(),
            "a lambda's term is a function type unless a refused relation left it otherwise"
        );
        return;
    };
    let joined = cx.node(TyKind::Fn {
        params,
        ret,
        effect,
        flows,
        calls: was_calls.join(calls),
        holds: was_holds.join(holds),
    });
    cx.join(term, joined, span);
    if calls == Calls::Once {
        cx.out
            .closures
            .get_mut(&lambda)
            .expect("a lambda records its closure when it is checked")
            .calls = Calls::Once;
    }
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
