//! What a type is (rfc_compiler: types). A type is interned: a `TyId` names
//! one shape in a `Types` table, and two ids of one table are the same type
//! exactly when they are equal, so a type written in k steps costs O(k)
//! (compilation). Nothing here solves or reads a declaration.

mod arg;
mod effect;
mod facts;
mod flow;
mod kind;
mod num;
pub mod print;
pub mod structural;

use std::convert::Infallible;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};

use indexmap::IndexSet;
use rustc_hash::FxBuildHasher;

pub use arg::{EffectArg, IdentityId, IdentityTerm, LenTerm, TypeArg};
pub use effect::{Contexts, Cost, Effect, EffectTerm, Reissue, SplitCost};
pub use flow::{Alignment, Declared, Flow, FlowEnd, FlowTerm, Flows, Labelled, Source};
pub use facts::{LifetimesOf, PositionKind, Resolve, copies, copies_alone, copies_in, for_each_position, position_kinds, positions};
pub use kind::{Calls, CallsTerm, Holds, HoldsTerm, Mutability, ObjectTy, TyKind};
pub use num::{CastTy, IntTy, WordTy};

/// Which variables a type of a phase may hold.
pub trait Phase: 'static + Copy + fmt::Debug + Eq + Hash {
    type TyVar: fmt::Debug + Copy + Eq + Hash;
    type EffectVar: fmt::Debug + Copy + Eq + Hash;
    type LenVar: fmt::Debug + Copy + Eq + Hash;
    type IdentityVar: fmt::Debug + Copy + Eq + Hash;
    /// The open `#` of an argument whose mark is not decided yet (types §8).
    type ReprVar: fmt::Debug + Copy + Eq + Hash;
    type FlowVar: fmt::Debug + Copy + Eq + Hash;
    /// What a function type's `holds` is in this phase: a fixed `Holds`,
    /// or, while solving, a term that copies of one type share.
    type HoldsTerm: fmt::Debug + Copy + Eq + Hash;
    /// What a function type's `calls` is in this phase, as `HoldsTerm`.
    type CallsTerm: fmt::Debug + Copy + Eq + Hash;
    /// Which function values a function type may be (types §2): while
    /// solving, a variable every copy of the type shares, grown where the
    /// type is transparent; nothing once solved, since no later reader
    /// asks it.
    type ValuesTerm: fmt::Debug + Copy + Eq + Hash;
    /// Who wrote a transparent structured type's node (types §1): fixed
    /// where the node is born, and part of its identity. Only solving asks
    /// it.
    type Label: fmt::Debug + Copy + Eq + Hash;
}

/// Every variable is `Infallible`: a concrete type holds none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concrete;

impl Phase for Concrete {
    type TyVar = Infallible;
    type EffectVar = Infallible;
    type LenVar = Infallible;
    type IdentityVar = Infallible;
    type ReprVar = Infallible;
    type FlowVar = Infallible;
    type HoldsTerm = Holds;
    type CallsTerm = Calls;
    type ValuesTerm = ();
    type Label = ();
}

/// A declaration's signature: its variables are positional placeholders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Poly;

impl Phase for Poly {
    type TyVar = u32;
    type EffectVar = u32;
    type LenVar = u32;
    type IdentityVar = u32;
    type ReprVar = u32;
    type FlowVar = Infallible;
    type HoldsTerm = Holds;
    type CallsTerm = Calls;
    type ValuesTerm = ();
    type Label = ();
}

/// A type under solving: its variables are the solver's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infer;

impl Phase for Infer {
    type TyVar = TyVarId;
    type EffectVar = EffectVarId;
    type LenVar = LenVarId;
    type IdentityVar = IdentityVarId;
    type ReprVar = ReprVarId;
    type FlowVar = FlowVarId;
    type HoldsTerm = HoldsTerm;
    type CallsTerm = CallsTerm;
    type ValuesTerm = ValuesVarId;
    type Label = crate::ty::Label;
}

/// How a relation reached the node it may grow (types §1): from a binding
/// by name, through fields and parts named from it, or through a
/// reference, which reaches some place of its target's type and not which
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reach {
    Named,
    Through,
}

/// Whether a node of `label`, reached by `reach`, may grow (types §1): the
/// one test of transparency every relation that grows a type asks. A node
/// the body wrote, reached by name.
pub fn is_transparent(label: Label, reach: Reach) -> bool {
    label == Label::Here && reach == Reach::Named
}

/// C, the transparency rule (types §1): who wrote a transparent structured
/// type's node, the body being checked or anything else. A value entering a
/// slot grows the slot's type where the body wrote it, and is coerced to it
/// otherwise (types §1, §3). Every one of the six kinds carries it: the
/// tuple, the array, `Option` and `Result` never grow, but a value entering
/// one written elsewhere is coerced part by part, and one written here is
/// not (types §3, §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// The body being checked wrote it.
    Here,
    /// A declaration, the host, or another body's scheme wrote it.
    Elsewhere,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LenVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReprVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HoldsVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallsVarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValuesVarId(pub u32);

/// One function value a function type may be (types §2): a lambda, by its
/// site, or a declaration used as a value, by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FnValue {
    Lambda(acvus_ast::AstId),
    Declaration(acvus_utils::QualifiedRef),
}

/// What a `ValuesVarId` holds: the function values the type may be, or
/// `Any` for a function type a declaration or the host wrote, whose values
/// are not the program's to name; a value entering it is coerced and its
/// own are forgotten (types §1, §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FnValues {
    Any,
    Of(std::collections::BTreeSet<FnValue>),
}

impl FnValues {
    /// The function values either side may be (types §2): `Any` where
    /// either is.
    pub fn join(&self, other: &FnValues) -> FnValues {
        match (self, other) {
            (FnValues::Any, _) | (_, FnValues::Any) => FnValues::Any,
            (FnValues::Of(a), FnValues::Of(b)) => FnValues::Of(a.union(b).copied().collect()),
        }
    }
}

static NEXT_TABLE: AtomicU32 = AtomicU32::new(1);

/// One shape in one `Types` table. It carries its table so that an id is
/// never read in another table, as an `Astr` carries its interner.
pub struct TyId<V: Phase> {
    table: NonZero<u32>,
    index: u32,
    phase: PhantomData<fn() -> V>,
}

impl<V: Phase> Clone for TyId<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Phase> Copy for TyId<V> {}

impl<V: Phase> PartialEq for TyId<V> {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(
            self.table, other.table,
            "comparing types of two tables ({} != {})",
            self.table, other.table
        );
        self.index == other.index
    }
}

impl<V: Phase> Eq for TyId<V> {}

impl<V: Phase> Hash for TyId<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.table.hash(state);
        self.index.hash(state);
    }
}

impl<V: Phase> TyId<V> {
    /// Where the type is interned in its table: a key for a canonical
    /// order of types, so that one set of them is written one way. It
    /// means nothing else.
    pub fn canonical_key(&self) -> u32 {
        self.index
    }
}

impl<V: Phase> fmt::Debug for TyId<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ty({}:{})", self.table, self.index)
    }
}

/// A concrete type: what the checker writes back and every later pass reads.
pub type Ty = TyId<Concrete>;

/// The hash-consing table of one phase's types. Its owner interns through
/// `&mut`; every reader reads through `&`.
pub struct Types<V: Phase> {
    table: NonZero<u32>,
    kinds: IndexSet<TyKind<V>, FxBuildHasher>,
}

impl<V: Phase> Types<V> {
    /// # Panics
    /// Every `u32` table number is taken.
    pub fn new() -> Self {
        let table = NEXT_TABLE.fetch_add(1, Ordering::Relaxed);
        Self {
            table: NonZero::new(table).expect("Types::new: every u32 table number is taken"),
            kinds: IndexSet::default(),
        }
    }

    /// The one id of `kind` in this table.
    ///
    /// # Panics
    /// The table holds `u32::MAX` shapes, or `kind` names an id of another
    /// table.
    pub fn intern(&mut self, kind: TyKind<V>) -> TyId<V> {
        kind.for_each_child(|child| self.assert_own(child));
        let (index, _) = self.kinds.insert_full(kind);
        TyId {
            table: self.table,
            index: u32::try_from(index).expect("Types::intern: the table holds u32::MAX shapes"),
            phase: PhantomData,
        }
    }

    /// # Panics
    /// `ty` is an id of another table.
    pub fn kind(&self, ty: TyId<V>) -> &TyKind<V> {
        self.assert_own(ty);
        &self.kinds[ty.index as usize]
    }

    /// `ty` of the table `from`, interned in this one: the same type, its
    /// parts moved with it (graph: a type crosses from one module's table
    /// to another's).
    pub fn transplant(&mut self, from: &Types<V>, ty: TyId<V>) -> TyId<V> {
        let kind = from.kind(ty).with_children(&mut |child| self.transplant(from, child));
        self.intern(kind)
    }

    /// Every id of this table, in the order its kinds were interned, so
    /// that each kind's parts come before it.
    pub fn ids(&self) -> impl Iterator<Item = TyId<V>> + '_ {
        (0..self.kinds.len()).map(|index| TyId {
            table: self.table,
            index: u32::try_from(index).expect("Types::ids: the table holds u32::MAX shapes"),
            phase: PhantomData,
        })
    }

    fn assert_own(&self, ty: TyId<V>) {
        assert_eq!(
            ty.table, self.table,
            "a type of table {} read in table {}",
            ty.table, self.table
        );
    }
}

impl<V: Phase> Default for Types<V> {
    fn default() -> Self {
        Self::new()
    }
}
