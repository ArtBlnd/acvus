//! The declarations the compiler takes in (rfc_compiler: extern fn, context,
//! programs §4), verified once. A `Declarations` exists only as
//! `Declarations::verify` built it, so every reader holds a verified set:
//! the casts form a tree (types §3), every requirement is smaller than its
//! declaration (extern-fn §6), and no declaration carries an array type
//! (extern-fn §1).
//!
//! Every declared type lives in the one `Types<Poly>` table the set owns.

mod bound;
mod casts;
mod core;
mod instance_fit;
mod labels;
mod loop_markers;
mod scheme;
mod user_type;
mod verify;

use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use crate::ty::{LifetimesOf, Poly, TyId, Types};

pub use bound::TyVarBound;
pub use casts::{Base, CastRule, CastTree, Head, VerifiedExternCast};
pub use core::{Core, Tags};
pub use labels::{EndLabels, LabelledSignature, Lifetime, Outlives, SegmentLabel};
pub use loop_markers::{KeyingMarkers, LoopMarkers};
pub use scheme::{DeclaredFacts, GenericSig, InstanceSig, Instances, Requirement, Scheme};
pub use user_type::UserTypeDecl;
pub use verify::{Refused, Written};

/// What the embedder and the extern registries hand the compiler, before
/// verification.
pub struct RawDeclarations {
    pub types: Types<Poly>,
    pub functions: Vec<(QualifiedRef, Scheme)>,
    pub user_types: Vec<UserTypeDecl>,
    pub casts: Vec<CastRule>,
    /// Each context with its fixed type (context).
    pub contexts: Vec<(QualifiedRef, TyId<Poly>)>,
    /// Each input `$x` with its type (programs §4).
    pub inputs: Vec<(Astr, TyId<Poly>)>,
    pub loop_markers: LoopMarkers,
    /// `core`'s names, from `Core::of`.
    pub core: Core,
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
    contexts: FxHashMap<QualifiedRef, TyId<Poly>>,
    inputs: FxHashMap<Astr, TyId<Poly>>,
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

    pub fn context(&self, context: QualifiedRef) -> Option<TyId<Poly>> {
        self.contexts.get(&context).copied()
    }

    pub fn input(&self, name: Astr) -> Option<TyId<Poly>> {
        self.inputs.get(&name).copied()
    }

    pub fn loop_markers(&self) -> &LoopMarkers {
        &self.loop_markers
    }

    pub fn core(&self) -> &Core {
        &self.core
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
