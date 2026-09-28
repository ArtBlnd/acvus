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

/// The head a cast step is taken from and to: a type's constructor, and for
/// a reference, its mutability and its target's constructor, so a view
/// (`&C` to `&[T]`, operators §5) is a step between two heads (types §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Head {
    Base(Base),
    Ref(Mutability, Base),
}

/// A constructor. `Ref` here is a reference's target that is itself a
/// reference: no view or declared cast steps from one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base {
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
    Option,
    Result,
    Object,
    Enum(QualifiedRef),
    Fn,
    Named(QualifiedRef),
    Ref,
}

/// The declared casts and the views (operators §5) as one tree: every head
/// has at most one next step up, and no step returns to a head below it
/// (types §3). Every order the tree gives is total. Two values of two
/// identities meet at their nearest common ancestor that carries no
/// identity: an identity-carrying ancestor is passed, not joined at.
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
