//! Records: what recorded them, boundaries, changes and uses.

use std::collections::BTreeSet;

use super::{ClassId, ItemId, Name};

/// What recorded something: a place in the text, and the rule that recorded
/// it (O22).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Item {
    pub at: Name,
    pub rule: Rule,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Rule {
    /// O1
    Meeting,
    /// O2
    Boundary,
    /// The Lower bounds table: a literal, a lend, a constructor application,
    /// a lambda, `return`, `?`.
    Written,
    /// A projection, which names a part (`*r`, `p.f`, `t.0`, `a[i]`, a
    /// payload pattern), or a call.
    Projection,
    /// O9
    Change,
    /// O13
    Signature,
    /// O14
    Decision,
    /// O16
    Result,
    /// O26
    Instance,
}

/// A change the language gives (Changes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Coercion {
    /// `&mut T → &T`
    Reborrow,
    /// `! → U`
    Never,
    /// An object to one with only some of its fields.
    DropFields,
    /// An enum or an `Error` to one with more variants.
    AddVariants,
    /// `&[T; n] → &[T]`, `&mut [T; n] → &mut [T]`
    Unsize,
    /// A function value type to one with more closures.
    AddClosures,
    /// `λ̄ → Fn(P̄ → R)` (O30)
    ToFnType,
}

/// O11: the coercions taken together in one stage, then one chain of
/// conversions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Change {
    pub coercions: BTreeSet<Coercion>,
    /// Names in `Decls::conversions`, in the order they apply.
    pub conversions: Vec<Name>,
}

/// O2: the argument's class is a lower bound of the boundary's class, whose
/// upper bound is the parameter's type or the context type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Boundary {
    pub arg: ClassId,
    pub class: ClassId,
    pub upper: ClassId,
    /// Made once the two class nodes are decided (O18).
    pub change: Option<Change>,
}

/// A candidate is an index into `Decls::fns[callee]`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum UseState {
    /// Candidates only decrease (O18).
    Undecided { left: BTreeSet<usize> },
    Decided(usize),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Use {
    pub at: ItemId,
    pub callee: Name,
    pub args: Vec<ClassId>,
    pub result: ClassId,
    pub state: UseState,
}
