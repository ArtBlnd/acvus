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

impl Head {
    /// Whether its constructor, or a reference's target's, is a marked
    /// extern-defined type (types §8).
    pub fn is_marked(self) -> bool {
        matches!(self, Head::Base(Base::Named { marked: true, .. }) | Head::Ref(_, Base::Named { marked: true, .. }))
    }

    /// The head `Step::Unmark` reaches from this one: the same with its
    /// extern-defined type unmarked.
    ///
    /// # Panics
    /// The head is not marked: only a marked head steps up by `Unmark`.
    pub fn unmarked(self) -> Head {
        match self {
            Head::Base(Base::Named { id, marked: true }) => Head::Base(Base::Named { id, marked: false }),
            Head::Ref(mutability, Base::Named { id, marked: true }) => Head::Ref(mutability, Base::Named { id, marked: false }),
            unmarked => panic!("only a marked head steps up by Unmark, not {unmarked:?}"),
        }
    }
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
    /// An extern-defined type, marked `#` or not: two heads, the marked one
    /// a step below the unmarked (types §8).
    Named { id: QualifiedRef, marked: bool },
}

/// One step up the tree (types §3, §8): a declared cast or a view, or the
/// step from a marked extern-defined type to the same type unmarked, where
/// a backend takes the unmarked type's representation.
#[derive(Debug, Clone, Copy)]
pub enum Step<'t> {
    Cast(&'t VerifiedExternCast),
    Unmark,
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
    /// The one step up from `head`, if any: a marked head's is `Unmark`,
    /// which no declared cast shares (types §8).
    pub fn step_up(&self, head: Head) -> Option<Step<'_>> {
        match head.is_marked() {
            true => Some(Step::Unmark),
            false => self.up.get(&head).map(Step::Cast),
        }
    }
}
