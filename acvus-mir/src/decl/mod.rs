//! The declarations the compiler takes in (rfc_compiler: extern fn, context,
//! programs), verified once. A `Declarations` exists only as
//! `Declarations::verify` built it, so every reader holds a verified set:
//! the casts form a tree (types §3), every cast and conversion joins two
//! heads, every requirement is smaller than its declaration (extern-fn §6),
//! and no declaration carries an array type (extern-fn §1).
//!
//! Every declared type lives in the one `Types<Poly>` table the set owns.

mod bound;
mod casts;
mod conversions;
mod core;
pub mod declared;
mod impl_fit;
mod labels;
mod lift;
mod loop_markers;
mod scheme;
mod segments;
mod user_type;
mod verify;

use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use crate::ty::{LifetimesOf, Poly, TyId, Types};

pub use bound::TyVarBound;
pub use casts::{Base, CastRule, CastTree, Head, VerifiedExternCast};
pub use conversions::{ConversionRule, VerifiedConversion};
pub use core::{Core, Tags};
pub use declared::NamedFact;
pub use labels::{EndLabels, LabelledSignature, Lifetime, Outlives, SegmentLabel};
pub use lift::{Lift, LiftShape, Taking};
pub use loop_markers::{KeyingMarkers, LoopMarkers};
pub use segments::{Segment, segments};
pub use scheme::{DeclaredFacts, GenericImplSig, ImplSig, Impls, Requirement, Scheme};
pub(crate) use scheme::effect_variables;
pub use user_type::UserTypeDecl;
pub use verify::{Refused, Written};

/// What the embedder and the extern registries hand the compiler, before
/// verification.
pub struct RawDeclarations {
    pub types: Types<Poly>,
    /// Every declared function but the lang items, whose generic functions
    /// the compiler builds (operators §2).
    pub functions: Vec<(QualifiedRef, Scheme)>,
    /// The impls each lang item is given, by its `core` name.
    pub lang_impls: Vec<(QualifiedRef, Impls)>,
    pub user_types: Vec<UserTypeDecl>,
    pub casts: Vec<CastRule>,
    pub conversions: Vec<ConversionRule>,
    /// Each context with its declaration (context).
    pub contexts: Vec<(QualifiedRef, ContextDecl)>,
    /// Each function a run may start at (programs).
    pub entries: Vec<EntryDecl>,
    pub loop_markers: LoopMarkers,
    /// `core`'s names, from `Core::of`.
    pub core: Core,
}

/// A function a run may start at, with the types the host declares for its
/// inputs, by name, and its result, where it declares one (host, programs).
#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub function: QualifiedRef,
    pub inputs: Vec<(Astr, TyId<Poly>)>,
    pub result: Option<TyId<Poly>>,
}

/// A context's declaration: its fixed type (context §2), and whether it is
/// volatile, held in no slot, each read one load and each write one commit
/// (context §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextDecl {
    pub ty: TyId<Poly>,
    pub volatile: bool,
}

/// A verified set of declarations.
pub struct Declarations {
    types: Types<Poly>,
    functions: FxHashMap<QualifiedRef, Scheme>,
    /// Every declaration a bare name may reach, for reach to decide among
    /// (types §6, extern-fn §5).
    overloads: FxHashMap<Astr, Box<[QualifiedRef]>>,
    user_types: FxHashMap<QualifiedRef, UserTypeDecl>,
    casts: CastTree,
    /// Each head's declared conversions, in declaration order.
    conversions: FxHashMap<Head, Box<[VerifiedConversion]>>,
    /// Each generic function's impl at a transparent structured type, where
    /// it has one (types §1).
    lifts: FxHashMap<QualifiedRef, Lift>,
    contexts: FxHashMap<QualifiedRef, ContextDecl>,
    entries: Vec<EntryDecl>,
    loop_markers: LoopMarkers,
    core: Core,
}

impl Declarations {
    /// The only way to a `Declarations`.
    pub fn verify(raw: RawDeclarations) -> Result<Declarations, Vec<Refused>> {
        verify::verify(raw)
    }

    pub fn types(&self) -> &Types<Poly> {
        &self.types
    }

    pub fn scheme(&self, function: QualifiedRef) -> Option<&Scheme> {
        self.functions.get(&function)
    }

    /// The declarations a bare name may reach; empty when none.
    pub fn overloads(&self, name: Astr) -> &[QualifiedRef] {
        self.overloads.get(&name).map_or(&[], |all| all)
    }

    pub fn user_type(&self, id: QualifiedRef) -> Option<&UserTypeDecl> {
        self.user_types.get(&id)
    }

    pub fn casts(&self) -> &CastTree {
        &self.casts
    }

    /// The declared conversions from a value of `head`; empty when none.
    pub fn conversions_from(&self, head: Head) -> &[VerifiedConversion] {
        self.conversions.get(&head).map_or(&[], |all| all)
    }

    /// How `function` has an impl at a transparent structured type, where
    /// it has one (types §1).
    pub fn lift(&self, function: QualifiedRef) -> Option<&Lift> {
        self.lifts.get(&function)
    }

    pub fn context(&self, context: QualifiedRef) -> Option<TyId<Poly>> {
        self.contexts.get(&context).map(|declared| declared.ty)
    }

    /// Whether `context` is declared volatile; `None` where it is not
    /// declared.
    pub fn context_is_volatile(&self, context: QualifiedRef) -> Option<bool> {
        self.contexts.get(&context).map(|declared| declared.volatile)
    }

    /// Every function a run may start at, as the host declared them.
    pub fn entries(&self) -> &[EntryDecl] {
        &self.entries
    }

    pub fn loop_markers(&self) -> &LoopMarkers {
        &self.loop_markers
    }

    pub fn core(&self) -> &Core {
        &self.core
    }

    /// The facts a call reaches (`Scheme`): a declaration without impls,
    /// called at no impl, states them itself; otherwise the impl the call
    /// chose states them, a concrete impl by its index, the generic impl
    /// after them.
    ///
    /// # Panics
    /// `function` is not declared; it has impls and the call names none, or
    /// it has none and the call names one; or it has no impl `r#impl`: a
    /// call names a declared function and, where it has impls, one of them.
    pub(crate) fn call_facts(&self, function: QualifiedRef, r#impl: Option<usize>) -> &DeclaredFacts {
        let scheme = self.scheme(function).unwrap_or_else(|| panic!("the call names {function:?}, which is not declared"));
        let (impls, r#impl) = match (&scheme.impls, r#impl) {
            (None, None) => return &scheme.facts,
            (Some(impls), Some(r#impl)) => (impls, r#impl),
            (None, Some(r#impl)) => panic!("a call of {function:?}, which has no impls, names impl {}", r#impl),
            (Some(_), None) => panic!("a call of {function:?}, which has impls, names none"),
        };
        match impls.concrete.get(r#impl) {
            Some(concrete) => &concrete.facts,
            None if r#impl == impls.concrete.len() => {
                &impls.generic.as_ref().unwrap_or_else(|| panic!("{function:?} has no impl {}", r#impl)).facts
            }
            None => panic!("{function:?} has no impl {}", r#impl),
        }
    }

    /// Every declared function with its scheme, in no promised order.
    pub fn functions(&self) -> impl Iterator<Item = (QualifiedRef, &Scheme)> {
        self.functions.iter().map(|(id, scheme)| (*id, scheme))
    }
}

impl LifetimesOf for Declarations {
    /// # Panics
    /// `id` names no declared type: a type term names only declared types.
    fn lifetimes_of(&self, id: QualifiedRef) -> usize {
        self.user_types
            .get(&id)
            .map(|decl| decl.lifetimes)
            .expect("a user-defined type names a declared type")
    }
}
