use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use crate::ty::{IntTy, Mutability, Poly, TyId};

/// A declared cast (extern-fn §7): `from` converts to `to` through `via`,
/// the two patterns sharing their placeholders.
#[derive(Debug, Clone, PartialEq)]
pub struct CastRule {
    pub from: TyId<Poly>,
    pub to: TyId<Poly>,
    pub via: QualifiedRef,
}

/// A cast `verify` admitted into the tree. Only `verify` builds one.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedExternCast(pub(super) CastRule);

impl VerifiedExternCast {
    pub fn rule(&self) -> &CastRule {
        &self.0
    }
}

/// The head a cast step is taken from and to: the constructor of a type,
/// with a named type by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Head {
    Int(IntTy),
    Float,
    Char,
    String,
    Bool,
    Unit,
    Str,
    Array,
    Slice,
    Tuple(usize),
    Object,
    Enum(QualifiedRef),
    Fn,
    Named(QualifiedRef),
    Ref(Mutability),
}

/// The declared casts and the views (operators §5) as one tree: every head
/// has at most one next step up, no step returns to a head below it, and an
/// identity-carrying head steps first to its own form without identity
/// (types §3). Every order the tree gives is total.
#[derive(Debug, Default)]
pub struct CastTree {
    pub(super) up: FxHashMap<Head, VerifiedExternCast>,
}

impl CastTree {
    /// The one step up from `head`, if any.
    pub fn step_up(&self, head: Head) -> Option<&VerifiedExternCast> {
        self.up.get(&head)
    }
}
