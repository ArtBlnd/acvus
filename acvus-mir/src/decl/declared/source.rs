//! Source (extern-fn §3): how the stream a call returns yields its
//! elements. A parameter is numbered as a call's arguments are.

use acvus_utils::QualifiedRef;

use super::relation::{Comparison, Term, refuse_ill_formed_term};
use super::Stated;
use crate::decl::Refused;
use crate::ty::{Poly, TyKind, Types};

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
pub(super) fn refuse_ill_formed(stated: &Stated<'_>, types: &Types<Poly>, sources: &[Source], refused: &mut Vec<Refused>) {
    let declared = stated.declared;
    let states = |form: fn(&Source) -> bool| sources.iter().any(form);
    if states(|form| matches!(form, Source::Splittable)) && states(|form| matches!(form, Source::Stepwise)) {
        refused.push(Refused::SplittableAndStepwise { declared });
    }
    for source in sources {
        match source {
            Source::Splittable | Source::Stepwise => {}
            Source::Count { bound, .. } => refuse_ill_formed_term(stated, types, bound, refused),
            Source::Map { from, through } => {
                stated.param(*from, refused);
                refuse_unfit_closure(stated, types, *through, false, refused);
            }
            Source::Filter { from, predicate } => {
                stated.param(*from, refused);
                refuse_unfit_closure(stated, types, *predicate, true, refused);
            }
            Source::Flatten { from } => {
                stated.param(*from, refused);
            }
        }
    }
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
