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
pub struct VerifiedExternCast {
    pub(super) rule: CastRule,
    /// For a view (operators §5), the concrete impl of `rule.via` the
    /// step is; `None` for a declared cast, whose `via` has no impls.
    pub(super) view_impl: Option<usize>,
}

impl VerifiedExternCast {
    pub fn rule(&self) -> &CastRule {
        &self.rule
    }

    pub(crate) fn view_impl(&self) -> Option<usize> {
        self.view_impl
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

/// A constructor. A reference's target is never a reference (types §4), so
/// no reference is a `Base`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base {
    /// The bottom type (types §4): a closed head for decisions on a diverging
    /// term. No tree step leaves or reaches it.
    Never,
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
    /// An extern-defined type, marked or not: its mark is a member, not
    /// its constructor (types §6, §8).
    Named(QualifiedRef),
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
    /// The one step up from `head`, if any. A mark is never stepped: a
    /// cast from `F<T>` taken by a marked value raises its mark (types §8).
    pub fn step_up(&self, head: Head) -> Option<&VerifiedExternCast> {
        self.up.get(&head)
    }
}
