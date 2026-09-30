use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashSet;

use super::TyVarBound;
use super::declared::{Access, Action, NamedFact, Relation, Source};
use super::labels::LabelledSignature;
use crate::ty::{EffectTerm, Poly, TyId, TyKind, Types};

/// An extern fn's declaration (extern-fn): its signature, a function type
/// whose effect carries its split (effects §6), the bound of each type
/// parameter by position, its impls, what it requires, and its declared
/// facts.
///
/// A call's declared facts are stated in one place, on what the call
/// reaches: the declaration itself where it has no impls, and the impl
/// that answers where it has them. A fact set per signature, read whole,
/// needs no rule for how two sets of one kind would combine, and the
/// ledger states none. So a declaration with impls states its facts on
/// them, and `verify` refuses facts on its own `facts`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scheme {
    pub ty: TyId<Poly>,
    /// One name per parameter of `ty`, in order (extern-fn §1). A name
    /// exists only in its declaration: a function type's parameters are
    /// types alone. `verify` refuses any other count.
    pub param_names: Box<[Astr]>,
    /// Exactly one bound per type variable of `ty`, by position; `verify`
    /// refuses any other count.
    pub bounds: Box<[TyVarBound]>,
    /// `Some` for a generic function's impls (extern-fn §6).
    pub impls: Option<Impls>,
    pub requires: Box<[Requirement]>,
    /// The facts a call of a declaration without impls reaches.
    pub facts: DeclaredFacts,
    /// The effect variables of `ty` whose work a call runs before its
    /// result is taken (extern-fn §2). The call's effect is the join of the
    /// function type's own effect in `ty` and these: instantiation gives
    /// the function type a fresh effect variable into which the own effect
    /// and each of these flow. An effect variable no parameter of `ty`
    /// gives is pure. `verify` refuses an effect variable a parameter gives
    /// that is neither here nor in the result.
    pub runs: Box<[u32]>,
}

/// The impls of a generic function: the concrete ones in order, then
/// the generic one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Impls {
    pub concrete: Box<[ImplSig]>,
    pub generic: Option<GenericImplSig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplSig {
    pub ty: TyId<Poly>,
    /// Exactly one bound per type variable of `ty` the impl leaves open,
    /// by position, as `Scheme::bounds`.
    pub bounds: Box<[TyVarBound]>,
    /// One name per parameter of `ty`, in order, as `Scheme::param_names`.
    pub param_names: Box<[Astr]>,
    pub requires: Box<[Requirement]>,
    pub facts: DeclaredFacts,
    /// As `Scheme::runs`.
    pub runs: Box<[u32]>,
}

/// The generic impl: its type is the generic function's own.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenericImplSig {
    pub facts: DeclaredFacts,
}

/// What a declaration states beyond its signature, trusted absolutely
/// (extern-fn §3-4): the declared facts in their four kinds, and its
/// lifetimes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeclaredFacts {
    pub access: Box<[Access]>,
    pub action: Box<[Action]>,
    pub source: Box<[Source]>,
    pub relations: Box<[Relation]>,
    /// The lifetimes the Rust signature names, which `verify` turns into
    /// the declared function type's flows. `None`: the flows are every
    /// input to every output (extern-fn §4).
    pub lifetimes: Option<LabelledSignature>,
}

impl DeclaredFacts {
    /// Whether no fact of any kind is stated.
    pub(super) fn states_nothing(&self) -> bool {
        *self == DeclaredFacts::default()
    }
}

/// An impl of `generic_function` whose type joins `pattern`, written at the
/// declaration's own variables, stating `fact` where one is named
/// (extern-fn §6). Its impls are the generic function's, read from the
/// declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    pub generic_function: QualifiedRef,
    pub pattern: TyId<Poly>,
    pub fact: Option<NamedFact>,
}

/// Every effect variable `tys` mention: a function type's effect and an
/// extern-defined type's, at any depth.
pub(crate) fn effect_variables(types: &Types<Poly>, tys: impl IntoIterator<Item = TyId<Poly>>) -> FxHashSet<u32> {
    let mut variables = FxHashSet::default();
    let mut seen = FxHashSet::default();
    let mut stack: Vec<TyId<Poly>> = tys.into_iter().collect();
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        let kind = types.kind(at);
        let effect = match kind {
            TyKind::Fn { effect, .. } => Some(effect),
            TyKind::UserDefined { effect, .. } => effect.as_ref(),
            _ => None,
        };
        if let Some(EffectTerm::Var(var)) = effect {
            variables.insert(*var);
        }
        kind.for_each_child(|child| stack.push(child));
    }
    variables
}
