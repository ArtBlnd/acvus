//! Source (extern-fn §3): how the stream a call returns yields its
//! elements. A parameter is numbered as a call's arguments are.

use acvus_utils::QualifiedRef;

use rustc_hash::FxHashMap;

use super::relation::{Comparison, Term, refuse_ill_formed_term};
use super::{Stated, fits, fits_part, step_state};
use crate::decl::{Refused, Scheme};
use crate::ty::{Poly, TyId, TyKind, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// The stream may be split at any position, each part yielding its
    /// elements.
    Splittable,
    /// The stream yields its elements only one after another.
    Stepwise,
    /// The number of elements the stream yields stands in `comparison` to
    /// `bound`.
    Count { comparison: Comparison, bound: Term },
    /// Yields each element of `from` through the closure argument `through`.
    Map { from: Stream, through: usize },
    /// Yields the elements of `from` that the closure argument `predicate`
    /// accepts.
    Filter { from: Stream, predicate: usize },
    /// Yields the elements of `from` until the first the closure argument
    /// `predicate` does not accept.
    TakeWhile { from: Stream, predicate: usize },
    /// Yields, one after another, the streams `from` yields.
    Flatten { from: Stream },
    /// The call consumes `from`: a state starts at `init`, each element in
    /// order updates it by `update`, and the result is `finish` of the
    /// final state, or the state itself. Where `update` is a declaration
    /// stating `Action::Combine`, the consumption may be split; where the
    /// state reaches that combine's absorbing element, the remaining
    /// elements are not consumed. A closure argument states no laws, so it
    /// is consumed in order.
    Consume { from: Stream, init: Step, update: Step, finish: Option<QualifiedRef> },
}

/// The stream a form reads: a parameter's, or another form's, so the forms
/// compose (`flat_map` is `Flatten` over `Map`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stream {
    Param(usize),
    Of(Box<Source>),
}

/// A step a consumption takes: a declared function, or a closure argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Declared(QualifiedRef),
    Param(usize),
}

/// Refuses each source form that is not well formed at `stated`'s
/// signature.
pub(super) fn refuse_ill_formed(
    stated: &Stated<'_>,
    sources: &[Source],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    let states = |form: fn(&Source) -> bool| sources.iter().any(form);
    if states(|form| matches!(form, Source::Splittable)) && states(|form| matches!(form, Source::Stepwise)) {
        refused.push(Refused::SplittableAndStepwise { declared });
    }
    for source in sources {
        refuse_ill_formed_form(stated, source, functions, types, refused);
    }
}

fn refuse_ill_formed_form(
    stated: &Stated<'_>,
    source: &Source,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    match source {
        Source::Splittable | Source::Stepwise => {}
        Source::Count { bound, .. } => refuse_ill_formed_term(stated, types, bound, refused),
        Source::Map { from, through } => {
            refuse_ill_formed_stream(stated, from, functions, types, refused);
            refuse_unfit_closure(stated, types, *through, false, refused);
        }
        Source::Filter { from, predicate } | Source::TakeWhile { from, predicate } => {
            refuse_ill_formed_stream(stated, from, functions, types, refused);
            refuse_unfit_closure(stated, types, *predicate, true, refused);
        }
        Source::Flatten { from } => refuse_ill_formed_stream(stated, from, functions, types, refused),
        Source::Consume { from, init, update, finish } => {
            refuse_ill_formed_stream(stated, from, functions, types, refused);
            refuse_ill_formed_consume(stated, *init, *update, *finish, functions, types, refused);
        }
    }
}

fn refuse_ill_formed_stream(
    stated: &Stated<'_>,
    stream: &Stream,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    match stream {
        Stream::Param(param) => {
            stated.param(*param, refused);
        }
        Stream::Of(inner) => match **inner {
            Source::Map { .. } | Source::Filter { .. } | Source::TakeWhile { .. } | Source::Flatten { .. } => {
                refuse_ill_formed_form(stated, inner, functions, types, refused);
            }
            Source::Splittable | Source::Stepwise | Source::Count { .. } | Source::Consume { .. } => {
                refused.push(Refused::StreamOfNoStreamMadeFromAnother { declared: stated.declared });
            }
        },
    }
}

/// A consumption's steps agree on one state type `S`: `init` is a declared
/// `fn() -> S` or a parameter of type `S`; `update` is a declared function
/// or a closure argument `fn(S, E) -> S` or `fn(&mut S, E)` over the
/// stream's element `E`; and `finish` is a declared `fn(S) -> R`, `R` the
/// call's result, or, where none is stated, `S` is the result. `S` is read
/// from the signature where it says it, by `init`, by the result without a
/// `finish`, or by a closure `update`, and from a declared `init`
/// otherwise. What the signature does not say of `E` is not compared.
fn refuse_ill_formed_consume(
    stated: &Stated<'_>,
    init: Step,
    update: Step,
    finish: Option<QualifiedRef>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    let declared_ty = |function: QualifiedRef| functions.get(&function).map(|scheme| scheme.ty);
    let init_param = match init {
        Step::Param(param) => match stated.param(param, refused) {
            Some(ty) => Some(ty),
            None => return,
        },
        Step::Declared(_) => None,
    };
    let update_closure = match update {
        Step::Param(param) => match stated.param(param, refused) {
            Some(ty) => Some(closure_state(types, ty)),
            None => return,
        },
        Step::Declared(_) => None,
    };
    let declared_init = match init {
        Step::Declared(function) => declared_ty(function).and_then(|ty| nullary_result(types, ty)),
        Step::Param(_) => None,
    };
    let state = init_param
        .or(finish.is_none().then_some(stated.ret))
        .or(update_closure.flatten())
        .or(declared_init);
    let Some(state) = state else {
        refused.push(Refused::ConsumeInitDoesNotFit { declared, init });
        return;
    };

    let init_fits = match init {
        Step::Param(_) => init_param == Some(state),
        Step::Declared(function) => declared_ty(function).is_some_and(|ty| fits(types, ty, &[], state)),
    };
    if !init_fits {
        refused.push(Refused::ConsumeInitDoesNotFit { declared, init });
    }
    let update_fits = match update {
        Step::Param(_) => update_closure.flatten() == Some(state),
        Step::Declared(function) => declared_ty(function).is_some_and(|ty| declared_step_fits(types, ty, state)),
    };
    if !update_fits {
        refused.push(Refused::ConsumeUpdateDoesNotFit { declared, update });
    }
    let finish_fits = match finish {
        None => state == stated.ret,
        Some(function) => declared_ty(function).is_some_and(|ty| fits(types, ty, &[state], stated.ret)),
    };
    if !finish_fits {
        refused.push(Refused::ConsumeFinishDoesNotFit { declared, finish });
    }
}

/// `S` of a closure argument `fn(S, E) -> S` or `fn(&mut S, E)`.
fn closure_state(types: &Types<Poly>, ty: TyId<Poly>) -> Option<TyId<Poly>> {
    match types.kind(ty) {
        TyKind::Fn { params, ret, .. } => step_state(types, params, *ret).map(|(state, _)| state),
        _ => None,
    }
}

/// `R` of a declared `fn() -> R`.
fn nullary_result(types: &Types<Poly>, ty: TyId<Poly>) -> Option<TyId<Poly>> {
    match types.kind(ty) {
        TyKind::Fn { params, ret, .. } if params.is_empty() => Some(*ret),
        _ => None,
    }
}

/// Whether the declared function type `ty` is a step over `state` at some
/// choice of its own variables, its operand left as it declares it.
fn declared_step_fits(types: &mut Types<Poly>, ty: TyId<Poly>, state: TyId<Poly>) -> bool {
    let own_state = match types.kind(ty) {
        TyKind::Fn { params, ret, .. } => step_state(types, params, *ret).map(|(own_state, _)| own_state),
        _ => None,
    };
    own_state.is_some_and(|own_state| fits_part(types, ty, own_state, state))
}

/// A closure argument takes one element; a predicate answers a `bool`.
fn refuse_unfit_closure(stated: &Stated<'_>, types: &Types<Poly>, param: usize, answers_bool: bool, refused: &mut Vec<Refused>) {
    let Some(ty) = stated.param(param, refused) else {
        return;
    };
    let fits = match types.kind(ty) {
        TyKind::Fn { params, ret, .. } => params.len() == 1 && (!answers_bool || matches!(types.kind(*ret), TyKind::Bool)),
        _ => false,
    };
    if !fits {
        refused.push(Refused::ClosureDoesNotFit { declared: stated.declared, param });
    }
}
