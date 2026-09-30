//! Contexts (context): a fixed, invariant type; a value stored into one, and
//! a run's result, have types with no positions (ownership §5).

use acvus_ast::{AstId, Span};
use acvus_utils::QualifiedRef;

use super::ctxt::Ctxt;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::place::Demand;
use crate::infer::Term;
use crate::ty::{Contexts, Effect, EffectTerm, LifetimesOf, Mutability, Reissue, SplitCost, TyKind};

/// How a body touches a context: its load and its commit are the effects
/// (context §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    /// A read of a value for `demand`: one lent mutably or written in part
    /// is read and may be written.
    pub fn of(demand: Demand) -> Access {
        match demand {
            Demand::Read | Demand::Lend(Mutability::Shared) => Access::Read,
            Demand::Lend(Mutability::Mut) | Demand::Write => Access::ReadWrite,
        }
    }
}

/// `@x` read for `demand`: the declared type, invariant (context §2).
pub(super) fn check_context_ref(cx: &mut Ctxt<'_>, site: AstId, context: QualifiedRef, demand: Demand, span: Span) -> Term {
    match context_term(cx, site, context, span) {
        Some(term) => {
            cx.out.names.insert(site, Name::Context(context));
            note_access(cx, context, Access::of(demand), span);
            term
        }
        None => cx.fresh(),
    }
}

/// The body touching `context` has it in its effect, joined as a call's
/// effect is (effects §1): among the effect's reads where it is loaded,
/// and its writes where it is committed (context §3). What a load or a
/// commit does beyond the program is unknown to the language, as a
/// syscall's is (context §3), so it is opaque, which is always sound
/// (extern-fn §2).
pub(super) fn note_access(cx: &mut Ctxt<'_>, context: QualifiedRef, access: Access, span: Span) {
    let touched = Contexts::from([context]);
    let (reads, writes) = match access {
        Access::Read => (touched, Contexts::new()),
        Access::Write => (Contexts::new(), touched),
        Access::ReadWrite => (touched.clone(), touched),
    };
    let effect = Effect::new(Reissue::Opaque, false, reads, writes, SplitCost::Unavailable);
    super::effect::note_call_effect(cx, &EffectTerm::Known(effect), span);
}

/// A context's declared type at this body, every node `Opaque`, so it never
/// grows and equals only itself (context §2, types §1).
pub(super) fn context_term(cx: &mut Ctxt<'_>, _site: AstId, context: QualifiedRef, span: Span) -> Option<Term> {
    match cx.decls.context(context) {
        Some(ty) => Some(cx.context_type(context, ty)),
        None => {
            refusal::push(cx, RefusalKind::UnknownContext { context }, span);
            None
        }
    }
}

/// A value stored into a context outlives the run, so the context's type
/// has no positions (context §3, ownership §5).
pub(super) fn refuse_held_loans(cx: &mut Ctxt<'_>, context: Term, span: Span) {
    if holds_positions(cx, context) {
        refusal::push(cx, RefusalKind::HoldsLoans, span);
    }
}

/// The body's result: the tail enters it at `site`, its own, as a
/// `return` or a `?` enters it at theirs (types §3, §7). A unit's result
/// may hold what its parameters lend, as a lambda's may (ownership §5);
/// only a run's result holds no loan, which is judged where a host runs
/// the unit as an entry (`graph`). The result slot is returned.
pub(super) fn check_body_result(cx: &mut Ctxt<'_>, site: AstId, tail: Term, span: Span) -> Term {
    let result = super::control::body_result(cx);
    cx.enter(site, tail, result, crate::ty::Reach::Named, span);
    result
}

/// Whether a value of a closed term holds a loan at some position
/// (ownership §5): a reference, a function value, an extern-defined type's
/// declared lifetime, or any of these in a part.
///
/// # Panics
/// `term` holds an unbound variable: it is read only on a declared type,
/// which has none, or after solving, which binds every one.
pub(super) fn holds_positions(cx: &Ctxt<'_>, term: Term) -> bool {
    let mut stack = vec![term];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        match cx.infer.types().kind(at) {
            TyKind::Ref(..) | TyKind::Fn { .. } | TyKind::Handle(_) => return true,
            TyKind::UserDefined { id, .. } if cx.decls.lifetimes_of(*id) > 0 => return true,
            TyKind::Var(var) => panic!("holds_positions: {var:?} is unbound on a declared or solved type"),
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    false
}
