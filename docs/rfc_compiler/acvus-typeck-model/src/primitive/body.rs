//! Bodies, closures and instances.

use std::collections::{BTreeMap, BTreeSet};

use super::{BodyId, ClassId, GroupId, ItemId};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BodyKind {
    FreeFn,
    Method,
    Lambda { enclosing: BodyId },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Body {
    pub kind: BodyKind,
    pub group: GroupId,
    pub params: Vec<ClassId>,
    pub result: ClassId,
    /// The classes of its own text: those an instance makes anew (O26).
    pub own: BTreeSet<ClassId>,
}

/// A lambda, together with the classes its captures are where its expression
/// lies.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Closure {
    pub lambda: BodyId,
    pub captures: BTreeSet<ClassId>,
}

/// A body recorded anew at a use (O26).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Instance {
    pub body: BodyId,
    pub at: ItemId,
    /// Each class of the body, to its class in this instance.
    pub renaming: BTreeMap<ClassId, ClassId>,
}
