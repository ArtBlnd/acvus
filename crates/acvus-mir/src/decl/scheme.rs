use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashSet;

use super::TyVarBound;
use super::declared::{Access, Action, NamedFact, Relation, Source};
use super::labels::LabelledSignature;
use crate::ty::{EffectTerm, IdentityTerm, Poly, TyId, TyKind, Types};

/// An extern fn's declaration (extern-fn): its signature, a function type
/// whose effect carries its split (effects §6), the bound of each type
/// parameter by position, what it requires, and its declared facts. An
/// impl is one too, and states the generic function it is an impl of; a
/// call reaches it only through that generic function (extern-fn §6).
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
    /// The generic function this declaration is an impl of, where it is
    /// one (extern-fn §6).
    pub impl_of: Option<ImplOf>,
    pub requires: Box<[Requirement]>,
    pub facts: DeclaredFacts,
    /// The effect variables of `ty` whose work a call runs before its
    /// result is taken (extern-fn §2). The call's effect is the join of the
    /// function type's own effect in `ty` and these: instantiation gives
    /// the function type a fresh effect variable into which the own effect
    /// and each of these flow. An effect variable no parameter of `ty`
    /// gives is pure. `verify` refuses an effect variable a parameter gives
    /// that is neither here nor in the result.
    pub runs: Box<[u32]>,
    /// Whether the declaration has a body that may suspend (extern-fn §2).
    pub awaits: bool,
    /// The whole concrete types a registry named and made this declaration
    /// concrete at once more (types §8, rfc_extern registry): the uniform
    /// declaration and its `#` counterparts are this one declaration. At a
    /// use, every position of `ty` whose type is one of these, once known,
    /// shares one mark: marked from the start where the use's type there
    /// is one of them, unmarked where it is not, and joined as any mark
    /// after. Which of the pair a call is follows from that mark.
    pub specialized: Box<[TyId<Poly>]>,
}

/// The generic function an impl is an impl of, and which of its impls it
/// is: a concrete one, or the generic one, which reach takes only where no
/// concrete one answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplOf {
    pub generic_function: QualifiedRef,
    pub kind: ImplKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplKind {
    Concrete,
    Generic,
}

/// A generic function (extern-fn §6): a name with a signature and no
/// body. It states nothing a body states; its impls do.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericFunction {
    pub ty: TyId<Poly>,
    /// As `Scheme::param_names`.
    pub param_names: Box<[Astr]>,
    /// As `Scheme::bounds`.
    pub bounds: Box<[TyVarBound]>,
    /// As `Scheme::specialized`.
    pub specialized: Box<[TyId<Poly>]>,
}

/// Which of a `#` pair a call of a declaration with `specialized` types
/// is (types §8): the uniform declaration, or its `#` counterpart, as its
/// shared mark settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairMember {
    Uniform,
    Specialized,
}

/// A generic function's impls, each a declaration of its own: the concrete
/// ones in declaration order, then the generic one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Impls {
    pub concrete: Box<[QualifiedRef]>,
    pub generic: Option<QualifiedRef>,
}

/// What a declaration states beyond its signature, trusted absolutely
/// (extern-fn §3-4): the declared facts in their four kinds, its
/// lifetimes, and how a call ends.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeclaredFacts {
    pub access: Box<[Access]>,
    pub action: Box<[Action]>,
    pub source: Box<[Source]>,
    pub relations: Box<[Relation]>,
    /// The lifetimes the Rust signature names, which each call of the
    /// declaration reads (`call_flows`). `None`: the flows are every input
    /// to every output (extern-fn §4).
    pub lifetimes: Option<LabelledSignature>,
    pub ends: Ends,
}

/// How the call's own code ends, for every call of a declaration
/// (extern-fn §4). Work the call runs, a closure argument or work a value
/// it is given holds, ends and traps as that work does, so a call returns
/// where its own code returns and that work does, and is total where both
/// are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ends {
    /// Nothing is stated: the call's own code may return, trap, or never
    /// end.
    #[default]
    Unstated,
    /// The call's own code returns or traps.
    Returns,
    /// The call's own code returns a value and never traps.
    Total,
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

/// Every identity variable `tys` mention: an extern-defined type's, at any
/// depth.
pub(crate) fn identity_variables(types: &Types<Poly>, tys: impl IntoIterator<Item = TyId<Poly>>) -> FxHashSet<u32> {
    let mut variables = FxHashSet::default();
    let mut seen = FxHashSet::default();
    let mut stack: Vec<TyId<Poly>> = tys.into_iter().collect();
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        let kind = types.kind(at);
        if let TyKind::UserDefined { identity: Some(IdentityTerm::Var(var)), .. } = kind {
            variables.insert(*var);
        }
        kind.for_each_child(|child| stack.push(child));
    }
    variables
}
