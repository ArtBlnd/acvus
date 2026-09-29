//! A call: of a name, a `Reach` decision over the declarations it may reach
//! (types §6, extern-fn §5); of a value, a function form at the call's
//! arity. Arguments `enter` parameters (types §7). Every other form that
//! reaches a declaration comes here.

use acvus_ast::{AstId, Expr, RefKind, Slot, Span};
use acvus_utils::QualifiedRef;

use super::ctxt::Ctxt;
use super::decision::{self, Callee, Decision};
use super::effect::note_call_effect;
use super::expr::check_expr;
use super::output::{Chosen, Impl, Name};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::variant::check_variant;
use crate::infer::Term;
use crate::ty::{CallsTerm, EffectTerm, FlowTerm, FnValue, FnValues, HoldsTerm, TyKind};

/// `f(args)`.
pub(super) fn check_call<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, func: &Expr<S>, args: &[Expr<S>], span: Span) -> Term {
    let args: Vec<&Expr<S>> = args.iter().collect();
    call(cx, site, func, &args, span)
}

/// `left | func`: `func(left)`.
pub(super) fn check_pipe<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, func: &Expr<S>, span: Span) -> Term {
    call(cx, site, func, &[left], span)
}

/// A call of `func` with `args`. A bare name bound in scope is a value; a
/// bare name not in scope reaches among its declarations; a qualified name
/// is `qualified_call`'s.
fn call<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, func: &Expr<S>, args: &[&Expr<S>], span: Span) -> Term {
    let Expr::Ident {
        id,
        name,
        ref_kind: RefKind::Value,
        span: name_span,
    } = func
    else {
        return value_call(cx, func, args);
    };
    if !name.namespace.is_root() {
        return qualified_call(cx, site, *id, *name, *name_span, args, span);
    }
    if cx.scope.lookup(name.name).is_some() {
        return value_call(cx, func, args);
    }
    if cx.decls.overloads(name.name).is_empty() {
        check_args(cx, args);
        refusal::push(cx, RefusalKind::UnknownName { name: name.name }, *name_span);
        return cx.fresh();
    }
    let args = check_args(cx, args);
    reach_call(cx, site, Callee::Name(name.name), args, span)
}

/// A call of a qualified name: its declaration where one is declared at the
/// whole name, and otherwise `Enum::Tag(payload)`, the variant the parser
/// leaves as a call (the parser's `build_call`). A variant is written with
/// two segments only, so a deeper name with no declaration is unknown.
fn qualified_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    name_site: AstId,
    name: QualifiedRef,
    name_span: Span,
    args: &[&Expr<S>],
    span: Span,
) -> Term {
    if cx.decls.scheme(name).is_some() {
        cx.out.names.insert(name_site, Name::Declaration(name));
        let args = check_args(cx, args);
        return reach_call(cx, site, Callee::Declaration(name), args, span);
    }
    let [enum_name] = cx.interner.path(name.namespace)[..] else {
        check_args(cx, args);
        refusal::push(cx, RefusalKind::UnknownName { name: name.name }, name_span);
        return cx.fresh();
    };
    match args {
        [] => check_variant::<S>(cx, site, Some(enum_name), name.name, None, span),
        [payload] => check_variant(cx, site, Some(enum_name), name.name, Some(payload), span),
        _ => {
            check_args(cx, args);
            refusal::push(cx, RefusalKind::VariantArity { tag: name.name }, span);
            cx.fresh()
        }
    }
}

/// Each argument's term, at the site it is written.
fn check_args<S: Slot>(cx: &mut Ctxt<'_>, args: &[&Expr<S>]) -> Vec<(AstId, Term)> {
    args.iter().map(|arg| (arg.id(), check_expr(cx, arg, Demand::Read))).collect()
}

/// A call of a value: the value is a function of the call's arity
/// (types §6), its parameters the slots the arguments enter (types §7). The
/// form joins the value, so a value that is called once or holds loans
/// keeps that (types §2).
fn value_call<S: Slot>(cx: &mut Ctxt<'_>, func: &Expr<S>, args: &[&Expr<S>]) -> Term {
    let callee = check_expr(cx, func, Demand::Read);
    let values = check_args(cx, args);
    let params: Vec<Term> = values.iter().map(|_| cx.fresh()).collect();
    let ret = cx.fresh();
    let effect = EffectTerm::Var(cx.infer.fresh_effect());
    let flows = FlowTerm::Var(cx.infer.fresh_flow());
    let calls = CallsTerm::Var(cx.infer.fresh_calls());
    let holds = HoldsTerm::Var(cx.infer.fresh_holds());
    let function_values = cx.infer.fresh_values();
    let form = cx.held(TyKind::Fn {
        params: params.clone().into(),
        ret,
        effect: effect.clone(),
        flows,
        calls,
        holds,
        values: function_values,
    });
    cx.join(callee, form, crate::ty::Reach::Named, func.span());
    for ((arg, (site, value)), param) in args.iter().zip(values).zip(params) {
        cx.enter(site, value, param, crate::ty::Reach::Named, arg.span());
    }
    note_call_effect(cx, &effect, func.span());
    ret
}

/// The one path to a declaration: a `Reach` at `site` over `callee` with
/// argument terms `args`; the result's term. The call's effect joins its
/// body's now, and the reached declaration's joins it when the reach fires
/// (effects §1).
pub(super) fn reach_call(cx: &mut Ctxt<'_>, site: AstId, callee: Callee, args: Vec<(AstId, Term)>, span: Span) -> Term {
    let result = cx.fresh();
    let effect = EffectTerm::Var(cx.infer.fresh_effect());
    note_call_effect(cx, &effect, span);
    cx.decisions.push(Decision::Reach {
        site,
        span,
        callee,
        args: args.into(),
        result,
        effect,
    });
    result
}

/// A reach pushed by a decision as it fires, after the walk: the call's
/// effect joins `body`, the effect of the body the deciding form was
/// written in (effects §1).
pub(super) fn reach_in(
    cx: &mut Ctxt<'_>,
    body: &EffectTerm<crate::ty::Infer>,
    site: AstId,
    callee: Callee,
    args: Vec<(AstId, Term)>,
    span: Span,
) -> Term {
    let result = cx.fresh();
    let effect = EffectTerm::Var(cx.infer.fresh_effect());
    super::effect::flows_into(cx, &effect, body, span);
    cx.decisions.push(Decision::Reach {
        site,
        span,
        callee,
        args: args.into(),
        result,
        effect,
    });
    result
}

/// A declaration used as a value: its signature at this use. A generic
/// function is a value only through its generic impl, since no argument
/// decides among its concrete ones (types §6).
pub(super) fn declaration_value(cx: &mut Ctxt<'_>, site: AstId, declaration: QualifiedRef, span: Span) -> Term {
    let scheme = cx
        .decls
        .scheme(declaration)
        .expect("a name reaches only a declared function");
    let r#impl = match &scheme.impls {
        None => None,
        Some(impls) if impls.concrete.is_empty() && impls.generic.is_some() => Some(Impl::Generic),
        Some(_) => {
            let kind = RefusalKind::SeveralDeclarations {
                name: declaration.name,
                left: Box::new([declaration]),
            };
            refusal::push(cx, kind, span);
            return cx.fresh();
        }
    };
    let instantiated = cx.infer.instantiate(cx.decls, scheme);
    let value = as_value(cx, instantiated.ty, declaration);
    cx.out.names.insert(site, Name::Declaration(declaration));
    cx.out.reached.insert(
        site,
        Chosen {
            declaration,
            r#impl,
            required: Box::new([]),
        },
    );
    decision::settle_bounded(cx, instantiated.ty, span);
    let body = super::effect::body_effect(cx);
    for required in instantiated.requirements {
        decision::push_required(cx, site, span, required, &body);
    }
    value
}

/// A declaration's function type `ty` as the one function value it is:
/// the declaration, by its name (types §2).
fn as_value(cx: &mut Ctxt<'_>, ty: Term, declaration: QualifiedRef) -> Term {
    let TyKind::Fn { params, ret, effect, flows, calls, holds, .. } = cx.infer.types().kind(cx.infer.shallow(ty)).clone() else {
        panic!("an instantiated signature is a function type (NotAFunction)");
    };
    let values = cx.infer.fresh_values_of(FnValues::Of([FnValue::Declaration(declaration)].into()));
    cx.node(TyKind::Fn { params, ret, effect, flows, calls, holds, values })
}
