//! A call: of a name, a `Reach` decision over the declarations it may reach
//! (types §6, extern-fn §5); of a unit, by its whole name and its
//! arguments' names, its interface instantiated there (programs); of a
//! value, a function form at the call's arity. Arguments `enter`
//! parameters (types §7). Every other form that reaches a declaration comes
//! here.

use acvus_ast::{Args, AstId, Binder, Expr, RefKind, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::decision::{Callee, Decision};
use super::effect::note_call_effect;
use super::expr::check_expr;
use super::output::{Name, UnitCall};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::variant::check_variant;
use crate::infer::Term;
use crate::ty::{CallsTerm, EffectTerm, FlowTerm, HoldsTerm, TyKind};

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

/// A call of `unit`: a use of its scheme, brought into this check once
/// (`generalize::import`) and instantiated at the call as any scheme's use
/// is, each requirement made here (functions §2, programs §1-§2); then
/// called as a function value is, each argument entering the parameter it
/// names.
fn unit_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    name_site: AstId,
    unit: QualifiedRef,
    args: &[(Binder, Expr<S>)],
    values: Vec<(AstId, Term)>,
    span: Span,
) -> Term {
    let params = cx.interfaces.get(unit).expect("a unit call names a unit with an interface").params.clone();
    let Some(arguments) = argument_order(cx, unit, &params, args, span) else {
        return cx.fresh();
    };
    cx.out.names.insert(name_site, Name::Unit(unit));
    let scheme = super::generalize::import(cx, unit);
    let callee = super::generalize::instantiate(cx, scheme, name_site, span);
    let (form, params, ret, effect) = function_form(cx, params.len());
    cx.join(callee, form, crate::ty::Reach::Named, span);
    for (param, at) in params.iter().zip(&arguments) {
        let (arg_site, value) = values[*at];
        cx.enter(arg_site, value, *param, crate::ty::Reach::Named, args[*at].1.span());
    }
    note_call_effect(cx, &effect, span);
    cx.out.callees.insert(site, form);
    cx.out.units.insert(site, UnitCall { unit, arguments: arguments.into() });
    ret
}

/// A function type of `arity` fresh parameters, a fresh result, and fresh
/// variables for the rest, which a callee is joined with where a value is
/// called: its parameters, result and effect.
fn function_form(cx: &mut Ctxt<'_>, arity: usize) -> (Term, Vec<Term>, Term, EffectTerm<crate::ty::Infer>) {
    let params: Vec<Term> = (0..arity).map(|_| cx.fresh()).collect();
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
    (form, params, ret, effect)
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
    let (form, params, ret, effect) = function_form(cx, values.len());
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

/// A declaration named where no call of it stands: refused as
/// `ExternAsValue` (extern-fn §1); the term left is fresh, so the program
/// around it is still checked.
pub(super) fn declaration_value(cx: &mut Ctxt<'_>, declaration: QualifiedRef, span: Span) -> Term {
    refusal::push(cx, RefusalKind::ExternAsValue { declaration }, span);
    cx.fresh()
}
