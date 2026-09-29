//! G, let-bound function values (functions §2). A binding a `let` makes is
//! generalized where every function value entering it has its body in this
//! body: a lambda, a name of such a binding, or a join of them. Its scheme is
//! settled from every value that entered it, the `let`'s own and each
//! assignment's, and two values joined into it require what either
//! requires. Each use of the name instantiates the scheme, and each of its
//! requirements is made there, as a declaration's are reached
//! (extern-fn §6). A value whose body is elsewhere, such as a call's
//! result, is not generalized: its instance was fixed where it was made.
//!
//! A lambda is checked as a module's `fn` is (programs §3): its body's
//! decisions are its own (`Decisions` records each one's owner), decided as
//! far as they can be before its binding is generalized, and a decision
//! still open over a generalized variable is a requirement, carrying the
//! site it arose at.
//!
//! What the enclosing scope or the captured environment holds is not
//! generalized: it belongs to where it was written (types §1), and a
//! decision over it stays the enclosing body's.
//!
//! The walk ends before any binding is generalized, so a binding's scheme
//! sees every value that entered it; `generalize_all` runs between the walk
//! and `decision::solve`, innermost binding first.

use acvus_ast::{AstId, Span};

use super::ctxt::Ctxt;
use super::decision::Decision;
use crate::infer::Term;
use crate::ty::TyVarId;

/// A binding whose values are generalized, by its binder's site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SchemeId(pub u32);

/// One use of a generalized binding: which, where, and the instance of an
/// enclosing binding it was made inside, where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceId(pub u32);

/// A generalized binding while checking.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LetScheme {
    pub binder: AstId,
    /// The binding's term: its function type, over the generalized
    /// variables once `generalize_all` ran.
    pub term: Term,
    /// The lambdas whose values entered the binding: their bodies'
    /// decisions are what its requirements come from.
    pub lambdas: Vec<AstId>,
    /// `None` until `generalize_all` settles it.
    pub settled: Option<Settled>,
}

/// A scheme settled: what it generalizes, and what it requires.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Settled {
    /// Every variable of every kind made while checking the values that
    /// entered the binding and reached from no term the enclosing scope or
    /// a capture holds.
    pub generalized: Generalized,
    /// Each decision still open over a generalized variable, made at each
    /// use (functions §2).
    pub requirements: Box<[Decision]>,
}

/// The generalized variables, by kind.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct Generalized {
    pub types: Box<[TyVarId]>,
    // Effects, lengths, identities, marks, flows, holds and calls follow
    // the same rule; each kind is listed here as it is generalized.
}

/// One instance of a scheme at a use.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Instance {
    pub scheme: SchemeId,
    pub use_site: AstId,
    pub span: Span,
    /// The instance of the binding this use was checked inside, where the
    /// use is in a generalized binding's own value.
    pub within: Option<InstanceId>,
    /// Each generalized variable's fresh term at this instance, by position
    /// in `Settled::generalized`.
    pub terms: Vec<Term>,
}

/// Whether a `let` whose value is `value` makes a generalized binding: every
/// function value that can enter it has its body in this body (functions §2).
pub(super) fn generalizes(cx: &Ctxt<'_>, value: Term) -> bool {
    todo!("the value's function values: lambdas written here, generalized bindings, or joins of them")
}

/// A generalized binding at `binder`, of term `term`, registered with no
/// value yet: each value entering it adds its lambdas (`generalizes`
/// finds them), and `generalize_all` settles it.
pub(super) fn register(cx: &mut Ctxt<'_>, binder: AstId, term: Term) -> SchemeId {
    todo!("a `LetScheme` pushed to `Output::schemes`, its lambdas those of the `let`'s value")
}

/// Settles every scheme, innermost first: each binding's lambdas' decisions
/// are decided as far as they can be without a default reaching a
/// generalizable variable (types §5 applies at the use, not here), and the
/// scheme takes what is left.
pub(super) fn generalize_all(cx: &mut Ctxt<'_>) {
    todo!("a local fixpoint over each scheme's lambdas' decisions, then its generalized variables and requirements")
}

/// A use of the generalized binding `scheme` at `use_site`: a term of the
/// binding's type at fresh variables, and each requirement made there as a
/// `Decision::In` of the new instance. It waits as a decision until the
/// scheme is settled.
pub(super) fn instantiate(cx: &mut Ctxt<'_>, scheme: SchemeId, use_site: AstId, span: Span) -> Term {
    todo!("the scheme's term copied at fresh variables, the requirements pushed at the instance")
}
