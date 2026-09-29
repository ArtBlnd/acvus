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
//!
//! How each name is captured is set once the types are solved, as Rust's
//! upvar analysis does (D85): the walk records each use of a captured name,
//! and `settle_captures` reads them after the fixpoint, when every receiver
//! adjustment that lends a name is known. Inside the lambda, a captured
//! name has its binding's own type.

use acvus_ast::{AstId, Binder, Expr, RefKind, Slot, Span};
use acvus_utils::Astr;
use rustc_hash::FxHashSet;

use super::ctxt::Ctxt;
use super::expr::check_expr;
use super::output::{Capture, Captured, Closure, Lambda, WholeUse};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::infer::Term;
use crate::ty::{
    Calls, CallsTerm, Effect, EffectTerm, FlowTerm, Holds, HoldsTerm, HoldsVarId, LifetimesOf, Mutability, ObjectTy, TyKind,
};

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
    let holds = cx.infer.fresh_holds();
    let calls = cx.infer.fresh_calls();
    cx.out.lambdas.push(Lambda {
        site,
        by_move: captures_by_move,
        holds,
        calls,
    });
    let flows = FlowTerm::Var(cx.infer.fresh_flow());
    cx.held(TyKind::Fn {
        params: param_terms.into(),
        ret: result,
        effect,
        flows,
        calls: CallsTerm::Var(calls),
        holds: HoldsTerm::Var(holds),
    })
}

/// Each lambda's captures, from every use of a name it captured, once the
/// types are solved (functions §2-3, ownership §5): by move in a lambda
/// written `move`, and otherwise by `&mut` where some use writes the name
/// or lends it `&mut`, and by `&` else. What the captures hold, and whether
/// the lambda moves a captured value out and is called once, join the
/// lambda's own `holds` and `calls` variables, which every copy of its
/// type shares.
///
/// A `move` lambda holds what the values it moved in hold, and one of those
/// may be a function value whose own lambda settles later, as when a
/// binding a lambda captured is assigned a later lambda. So `holds` is
/// settled to its least fixpoint: each round joins in what the others
/// hold so far, until a round changes none.
///
/// Cost: O(lambdas · uses) for the captures and calls, and per round a walk
/// of each moved value's type; a round that changes nothing ends it, and
/// each variable rises at most twice.
pub(super) fn settle_captures(cx: &mut Ctxt<'_>) {
    let lambdas = cx.out.lambdas.clone();
    let mut holding: Vec<Holding> = Vec::with_capacity(lambdas.len());
    for lambda in &lambdas {
        let mut captures: Vec<Captured> = Vec::new();
        let mut moved_terms: Vec<Term> = Vec::new();
        for used in cx.out.capture_uses.iter().filter(|used| used.crossed.contains(&lambda.site)) {
            match captures.iter_mut().find(|captured| captured.binder == used.of) {
                Some(Captured { capture: Capture::Ref(mutability), .. }) => {
                    if used.mutability == Mutability::Mut {
                        *mutability = Mutability::Mut;
                    }
                }
                Some(Captured { capture: Capture::Move, .. }) => {}
                None => {
                    let capture = match lambda.by_move {
                        true => Capture::Move,
                        false => Capture::Ref(used.mutability),
                    };
                    if lambda.by_move {
                        moved_terms.push(used.term);
                    }
                    captures.push(Captured { name: used.name, binder: used.of, capture });
                }
            }
        }
        let wholes: Vec<Term> = cx
            .out
            .whole_uses
            .iter()
            .filter(|whole| whole.lambdas.contains(&lambda.site))
            .map(|whole| whole.term)
            .collect();
        if !unbound_vars_in(cx, moved_terms.iter().chain(&wholes).copied()) {
            let calls = match wholes.iter().any(|term| !copies(cx, *term)) {
                true => Calls::Once,
                false => Calls::Many,
            };
            cx.infer.join_calls(lambda.calls, calls);
            let by_reference = captures
                .iter()
                .map(|captured| match captured.capture {
                    Capture::Ref(Mutability::Shared) => Holds::Shared,
                    Capture::Ref(Mutability::Mut) => Holds::Mut,
                    Capture::Move => Holds::Nothing,
                })
                .fold(Holds::Nothing, Holds::join);
            holding.push(Holding {
                holds: lambda.holds,
                by_reference,
                moved_terms,
            });
            cx.out
                .closures
                .get_mut(&lambda.site)
                .expect("a lambda records its closure when it is checked")
                .calls = calls;
        }
        cx.out
            .closures
            .get_mut(&lambda.site)
            .expect("a lambda records its closure when it is checked")
            .captures = captures.into();
    }
    loop {
        let mut changed = false;
        let mut read_unsettled = false;
        for lambda in &holding {
            let mut holds = lambda.by_reference;
            for term in &lambda.moved_terms {
                let loan = strongest_loan(cx, *term);
                read_unsettled |= loan.unsettled;
                holds = holds.join(loan.known);
            }
            let before = cx.infer.holds(HoldsTerm::Var(lambda.holds));
            cx.infer.join_holds(lambda.holds, holds);
            changed |= cx.infer.holds(HoldsTerm::Var(lambda.holds)) != before;
        }
        if !changed {
            assert!(
                !read_unsettled || !cx.out.refusals.is_empty(),
                "a moved value's function type holds a variable no lambda settles and no default bound"
            );
            return;
        }
    }
}

/// A lambda whose captures are known, and what they hold but for the loans
/// of the values it moved in.
struct Holding {
    holds: HoldsVarId,
    by_reference: Holds,
    moved_terms: Vec<Term>,
}

/// Whether any of `terms` holds a type variable still open, which only a
/// refusal leaves after solving.
fn unbound_vars_in(cx: &Ctxt<'_>, terms: impl Iterator<Item = Term>) -> bool {
    let mut stack: Vec<Term> = terms.collect();
    let mut seen = FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        let kind = cx.infer.types().kind(at);
        if matches!(kind, TyKind::Var(_)) {
            return true;
        }
        kind.for_each_child(|child| stack.push(child));
    }
    false
}

/// A value of `expr` read whole, where `expr` is a place rooted at a name
/// or an input some `move` lambda around it captured: that lambda's body
/// may move it out (functions §3), decided once its type is solved.
pub(super) fn note_whole_use<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>, term: Term) {
    let crossed = match place_root(expr) {
        Some(Root::Binding(name)) => match cx.scope.lookup(name) {
            Some(found) => found.crossed,
            None => return,
        },
        Some(Root::Input) => cx.scope.lambdas(),
        None => return,
    };
    let lambdas: Box<[AstId]> = crossed.into_iter().filter(|lambda| cx.scope.captures_by_move(*lambda)).collect();
    if !lambdas.is_empty() {
        cx.out.whole_uses.push(WholeUse { lambdas, term });
    }
}

/// What a place expression is rooted at.
enum Root {
    Binding(Astr),
    Input,
}

fn place_root<S: Slot>(expr: &Expr<S>) -> Option<Root> {
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } if name.namespace.is_none() => Some(Root::Binding(name.name)),
        Expr::Ident {
            ref_kind: RefKind::ExternParam,
            ..
        } => Some(Root::Input),
        Expr::FieldAccess { object, .. } | Expr::Index { object, .. } => place_root(object),
        Expr::Paren { inner, .. } => place_root(inner),
        _ => None,
    }
}

/// The strongest loan a value of a solved term holds at any position
/// (ownership §5): a reference's own, a function value's captures', and
/// an extern-defined type's declared lifetime, which is exclusive (D56).
/// A function value whose `holds` is not settled yet adds nothing and is
/// reported as `unsettled`.
///
/// # Panics
/// `term` holds an unbound type variable: the decision reads it only once
/// none is left.
pub(super) fn strongest_loan(cx: &Ctxt<'_>, term: Term) -> Loan {
    let mut loan = Loan {
        known: Holds::Nothing,
        unsettled: false,
    };
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
            TyKind::Fn { holds, .. } => match cx.infer.holds(*holds) {
                Some(holds) => holds,
                None => {
                    loan.unsettled = true;
                    Holds::Nothing
                }
            },
            TyKind::UserDefined { id, .. } if cx.decls.lifetimes_of(*id) > 0 => Holds::Mut,
            TyKind::Var(var) => panic!("strongest_loan: {var:?} is unbound on a solved term"),
            _ => Holds::Nothing,
        };
        loan.known = loan.known.join(here);
        if !matches!(kind, TyKind::Fn { .. }) {
            kind.for_each_child(|child| stack.push(child));
        }
    }
    loan
}

/// What `strongest_loan` found: the strongest loan settled so far, and
/// whether some function value's `holds` was not settled yet.
pub(super) struct Loan {
    pub known: Holds,
    pub unsettled: bool,
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

/// A lambda stored into the binding it captures would call itself: only a
/// `fn` may recurse (functions §4).
pub(super) fn refuse_recursion(cx: &mut Ctxt<'_>, value: Term, name: Astr, span: Span) {
    let recursive = cx.out.lambdas.iter().any(|lambda| {
        cx.out.types.get(&lambda.site) == Some(&value)
            && cx.out.capture_uses.iter().any(|used| used.name == name && used.crossed.contains(&lambda.site))
    });
    if recursive {
        refusal::push(cx, RefusalKind::LambdaRecursion, span);
    }
}
