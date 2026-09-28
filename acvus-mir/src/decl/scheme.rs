use acvus_utils::{Astr, QualifiedRef};

use super::TyVarBound;
use super::labels::LabelledSignature;
use super::facts::{Laws, NamedLaw, Postcondition, Reaches, Returns};
use crate::means::Means;
use crate::ty::{Poly, TyId};

/// An extern fn's declaration (extern-fn): its signature, a function type
/// whose effect carries its split (effects §6), the bound of each type
/// parameter by position, its instances, and what it requires.
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
    /// `Some` for a shared signature's instances (extern-fn §6).
    pub instances: Option<Instances>,
    pub requires: Box<[Requirement]>,
}

/// The instances of a shared signature: the concrete ones in order, then
/// the generic one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Instances {
    pub concrete: Box<[InstanceSig]>,
    pub generic: Option<GenericSig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstanceSig {
    pub ty: TyId<Poly>,
    /// One name per parameter of `ty`, in order, as `Scheme::param_names`.
    pub param_names: Box<[Astr]>,
    pub requires: Box<[Requirement]>,
    pub facts: DeclaredFacts,
}

/// The generic instance: its type is the signature's own.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenericSig {
    pub facts: DeclaredFacts,
}

/// What a declaration states beyond its signature, trusted absolutely
/// (extern-fn §3-4).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeclaredFacts {
    pub laws: Laws,
    pub ensures: Box<[Postcondition]>,
    pub reaches: Reaches,
    pub returns: Returns,
    pub means: Option<Means>,
    /// The lifetimes the Rust signature names, which `verify` turns into
    /// the declared function type's flows. `None`: the flows are every
    /// input to every output (extern-fn §4).
    pub lifetimes: Option<LabelledSignature>,
}

/// An instance of `signature` whose type joins `pattern`, written at the
/// declaration's own variables, stating `law` where one is named
/// (extern-fn §6). Its instances are the signature's, read from the
/// declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    pub signature: QualifiedRef,
    pub pattern: TyId<Poly>,
    pub law: Option<NamedLaw>,
}
