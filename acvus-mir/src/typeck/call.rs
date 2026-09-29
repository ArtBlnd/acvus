//! A call: of a name, a `Reach` decision over the declarations it may reach
//! (types §6, extern-fn §5); of a unit, by its whole name and its
//! arguments' names, its interface instantiated there (programs); of a
//! value, a function form at the call's arity. Arguments `enter`
//! parameters (types §7). Every other form that reaches a declaration comes
//! here.

use acvus_ast::{Args, AstId, Binder, Expr, RefKind, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::decision::{self, Callee, Decision};
use super::effect::note_call_effect;
use super::expr::check_expr;
use super::output::{Chosen, Impl, Name, UnitCall};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::variant::check_variant;
use crate::infer::Term;
use crate::ty::{CallsTerm, EffectTerm, FlowTerm, FnValue, FnValues, HoldsTerm, TyKind};

/// `f(args)`: by position, of a declaration or a value; by name, of a
/// unit.
pub(super) fn check_call<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, func: &Expr<S>, args: &Args<S>, span: Span) -> Term {
    match args {
        Args::Positional(args) => {
            let args: Vec<&Expr<S>> = args.iter().collect();
            call(cx, site, func, &args, span)
        }
        Args::Named(args) => named_call(cx, site, func, args, span),
    }
}

/// A call by name, which reaches a unit only, by its whole name: a bare
/// name never does (programs).
fn named_call<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, func: &Expr<S>, args: &[(Binder, Expr<S>)], span: Span) -> Term {
    let unit = match func {
        Expr::Ident { id, name, ref_kind: RefKind::Value, .. } if !name.namespace.is_root() && cx.interfaces.get(*name).is_some() => Some((*id, *name)),
        _ => None,
    };
    let values = args.iter().map(|(_, arg)| (arg.id(), check_expr(cx, arg, Demand::Read))).collect();
    match unit {
        Some((name_site, unit)) => unit_call(cx, site, name_site, unit, args, values, span),
        None => {
            refusal::push(cx, RefusalKind::NamedArgumentsToNonUnit, span);
            cx.fresh()
        }
    }
}

/// A call of `unit`, its interface instantiated at the call as an extern
/// fn's scheme is (programs §1, §2): each argument enters the parameter it
/// names, the call's effect is the unit's, and each requirement is made
/// here (extern-fn §6).
fn unit_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    name_site: AstId,
    unit: QualifiedRef,
    args: &[(Binder, Expr<S>)],
    values: Vec<(AstId, Term)>,
    span: Span,
) -> Term {
    let interfaces = cx.interfaces;
    let scheme = &interfaces.get(unit).expect("a unit call names a unit with an interface").scheme;
    let Some(arguments) = argument_order(cx, unit, &scheme.param_names, args, span) else {
        return cx.fresh();
    };
    cx.out.names.insert(name_site, Name::Unit(unit));
    let instantiated = cx.infer.instantiate_in(interfaces.types(), scheme);
    let TyKind::Fn { params, ret, effect, .. } = cx.infer.types().kind(cx.infer.shallow(instantiated.ty)).clone() else {
        panic!("a unit's interface is a function type: the unit is one function (programs)");
    };
    assert_eq!(params.len(), arguments.len(), "a unit's interface names each of its parameters once");
    for (param, at) in params.iter().zip(&arguments) {
        let (arg_site, value) = values[*at];
        cx.enter(arg_site, value, *param, crate::ty::Reach::Named, args[*at].1.span());
    }
    note_call_effect(cx, &effect, span);
    cx.out.callees.insert(site, instantiated.ty);
    cx.out.units.insert(site, UnitCall { unit, arguments: arguments.into() });
    decision::settle_bounded(cx, instantiated.ty, span);
    let body = super::effect::body_effect(cx);
    for required in instantiated.requirements {
        decision::push_required(cx, site, span, required, &body);
    }
    ret
}

/// Which argument each of `params` takes, by the names written: every
/// parameter named exactly once and every name a parameter's. Each name
/// that is not is refused where it is written, and each parameter none
/// names at the call.
fn argument_order<S: Slot>(cx: &mut Ctxt<'_>, unit: QualifiedRef, params: &[Astr], args: &[(Binder, Expr<S>)], span: Span) -> Option<Vec<usize>> {
    let mut given: Vec<Option<usize>> = vec![None; params.len()];
    let mut fits = true;
    for (at, (label, _)) in args.iter().enumerate() {
        let name = label.name;
        match params.iter().position(|param| *param == name) {
            None => {
                refusal::push(cx, RefusalKind::UnknownArgument { unit, name }, label.span);
                fits = false;
            }
            Some(param) if given[param].is_some() => {
                refusal::push(cx, RefusalKind::DuplicateArgument { unit, name }, label.span);
                fits = false;
            }
            Some(param) => given[param] = Some(at),
        }
    }
    for (name, taken) in params.iter().zip(&given) {
        if taken.is_none() {
            refusal::push(cx, RefusalKind::MissingArgument { unit, name: *name }, span);
            fits = false;
        }
    }
    match fits {
        true => given.into_iter().collect(),
        false => None,
    }
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

/// A call of a qualified name by position: its declaration where one is
/// declared at the whole name; a unit there is called by name, so an
/// argument passed to it by position is refused; and otherwise
/// `Enum::Tag(payload)`, the variant the parser leaves as a call (the
/// parser's `build_call`). A variant is written with two segments only, so
/// a deeper name with no declaration is unknown.
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
    if cx.interfaces.get(name).is_some() {
        // `ns::f()` passes no argument by position, and names none.
        if args.is_empty() {
            return unit_call::<S>(cx, site, name_site, name, &[], Vec::new(), span);
        }
        check_args(cx, args);
        refusal::push(cx, RefusalKind::UnitCalledByPosition { unit: name }, span);
        return cx.fresh();
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
