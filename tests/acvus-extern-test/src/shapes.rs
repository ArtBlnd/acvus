//! Shapes whose views are written by hand as `shape!` writes them, standing
//! in for it until it is ported: each implements `Projected` through the
//! hidden path the macro uses, and its seal is signed in `boundary`. A shape's
//! parts are in the order of their names' text (compiler: types §2), and a
//! variant's tag is its place in that order.

use acvus_extern::Uniform;
use acvus_extern::guard::{Guard, GuardMut};
use acvus_extern::runtime::Runtime;
use acvus_extern::shape::{Aggregate, LentParts, LentPartsMut, OwnedParts, Projected};

use crate::boundary::Tally;

/// The tuple `(String, i64)`.
pub struct Labelled;

pub struct LabelledRef<'c, R: Runtime> {
    pub label: Guard<'c, R::Shared<'c, String>>,
    pub count: Guard<'c, R::Shared<'c, i64>>,
}

pub struct LabelledMut<'c, R: Runtime> {
    pub label: GuardMut<'c, R::Exclusive<'c, String>>,
    pub count: GuardMut<'c, R::Exclusive<'c, i64>>,
}

pub struct LabelledOwned<R: Runtime> {
    pub label: Uniform<R, String>,
    pub count: Uniform<R, i64>,
}

impl<R: Runtime> Projected<R> for Labelled {
    type Ref<'a> = LabelledRef<'a, R>;
    type Mut<'a> = LabelledMut<'a, R>;
    type Owned = LabelledOwned<R>;

    const AGGREGATE: Aggregate = Aggregate::Tuple;

    fn project<'c>(parts: LentParts<'c, R>) -> LabelledRef<'c, R> {
        LabelledRef { label: parts.leaf(0), count: parts.leaf(1) }
    }

    fn project_mut<'c>(mut parts: LentPartsMut<'c, R>) -> LabelledMut<'c, R> {
        let label = parts.leaf(0);
        let count = parts.leaf(1);
        LabelledMut { label, count }
    }

    fn owned(mut parts: OwnedParts<R>) -> LabelledOwned<R> {
        let label = parts.next();
        let count = parts.next();
        LabelledOwned { label, count }
    }
}

/// The tuple `((String, i64), bool)`: a shape nested in a shape.
pub struct Nested;

pub struct NestedRef<'c, R: Runtime> {
    pub inner: LabelledRef<'c, R>,
    pub flag: Guard<'c, R::Shared<'c, bool>>,
}

pub struct NestedMut<'c, R: Runtime> {
    pub inner: LabelledMut<'c, R>,
    pub flag: GuardMut<'c, R::Exclusive<'c, bool>>,
}

pub struct NestedOwned<R: Runtime> {
    pub inner: Uniform<R, Labelled>,
    pub flag: Uniform<R, bool>,
}

impl<R: Runtime> Projected<R> for Nested {
    type Ref<'a> = NestedRef<'a, R>;
    type Mut<'a> = NestedMut<'a, R>;
    type Owned = NestedOwned<R>;

    const AGGREGATE: Aggregate = Aggregate::Tuple;

    fn project<'c>(parts: LentParts<'c, R>) -> NestedRef<'c, R> {
        NestedRef { inner: parts.nested::<Labelled>(0), flag: parts.leaf(1) }
    }

    fn project_mut<'c>(mut parts: LentPartsMut<'c, R>) -> NestedMut<'c, R> {
        let inner = parts.nested::<Labelled>(0);
        let flag = parts.leaf(1);
        NestedMut { inner, flag }
    }

    fn owned(mut parts: OwnedParts<R>) -> NestedOwned<R> {
        let inner = parts.next();
        let flag = parts.next();
        NestedOwned { inner, flag }
    }
}

/// The object `{ count: i64, name: String }`.
pub struct Record;

pub struct RecordRef<'c, R: Runtime> {
    pub count: Guard<'c, R::Shared<'c, i64>>,
    pub name: Guard<'c, R::Shared<'c, String>>,
}

pub struct RecordMut<'c, R: Runtime> {
    pub count: GuardMut<'c, R::Exclusive<'c, i64>>,
    pub name: GuardMut<'c, R::Exclusive<'c, String>>,
}

pub struct RecordOwned<R: Runtime> {
    pub count: Uniform<R, i64>,
    pub name: Uniform<R, String>,
}

impl<R: Runtime> Projected<R> for Record {
    type Ref<'a> = RecordRef<'a, R>;
    type Mut<'a> = RecordMut<'a, R>;
    type Owned = RecordOwned<R>;

    const AGGREGATE: Aggregate = Aggregate::Object;

    fn project<'c>(parts: LentParts<'c, R>) -> RecordRef<'c, R> {
        RecordRef { count: parts.leaf(0), name: parts.leaf(1) }
    }

    fn project_mut<'c>(mut parts: LentPartsMut<'c, R>) -> RecordMut<'c, R> {
        let count = parts.leaf(0);
        let name = parts.leaf(1);
        RecordMut { count, name }
    }

    fn owned(mut parts: OwnedParts<R>) -> RecordOwned<R> {
        let count = parts.next();
        let name = parts.next();
        RecordOwned { count, name }
    }
}

/// The enum `Maybe { None, Some(Tally) }`: `None` is tag 0 and `Some` tag 1.
pub struct Maybe;

pub enum MaybeRef<'c, R: Runtime> {
    None,
    Some(Guard<'c, R::Shared<'c, Tally>>),
}

pub enum MaybeMut<'c, R: Runtime> {
    None,
    Some(GuardMut<'c, R::Exclusive<'c, Tally>>),
}

pub enum MaybeOwned<R: Runtime> {
    None,
    Some(Uniform<R, Tally>),
}

const NONE: u32 = 0;
const SOME: u32 = 1;

impl<R: Runtime> Projected<R> for Maybe {
    type Ref<'a> = MaybeRef<'a, R>;
    type Mut<'a> = MaybeMut<'a, R>;
    type Owned = MaybeOwned<R>;

    const AGGREGATE: Aggregate = Aggregate::Variant;

    fn project<'c>(parts: LentParts<'c, R>) -> MaybeRef<'c, R> {
        match parts.tag() {
            NONE => MaybeRef::None,
            SOME => MaybeRef::Some(parts.leaf(0)),
            tag => unreachable!("Maybe has no variant {tag}"),
        }
    }

    fn project_mut<'c>(mut parts: LentPartsMut<'c, R>) -> MaybeMut<'c, R> {
        match parts.tag() {
            NONE => MaybeMut::None,
            SOME => MaybeMut::Some(parts.leaf(0)),
            tag => unreachable!("Maybe has no variant {tag}"),
        }
    }

    fn owned(mut parts: OwnedParts<R>) -> MaybeOwned<R> {
        match parts.tag() {
            NONE => MaybeOwned::None,
            SOME => MaybeOwned::Some(parts.next()),
            tag => unreachable!("Maybe has no variant {tag}"),
        }
    }
}
