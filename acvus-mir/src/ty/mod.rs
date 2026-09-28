//! What a type is (rfc_compiler: types). A type is interned: a `TyId` names
//! one shape in a `Types` table, and two ids of one table are the same type
//! exactly when they are equal, so a type written in k steps costs O(k)
//! (compilation). Nothing here solves or reads a declaration.

mod arg;
mod effect;
mod facts;
mod kind;
mod num;

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
pub use facts::{LifetimesOf, copies, positions};
pub use kind::{Calls, Holds, Mutability, ObjectTy, ParamTerm, TyKind};
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
    /// Whether a structural object or enum may grow (types §1): fixed where
    /// the node is born, and part of its identity. Only solving asks it.
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
    type Label = crate::ty::Label;
}

/// C, the transparency rule (types §1): a node the program wrote may grow;
/// a node a declaration or the host gave never does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    Transparent,
    Opaque,
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
