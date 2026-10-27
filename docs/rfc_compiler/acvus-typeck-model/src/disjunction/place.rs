//! Places: the structure constraint a projection gives its operand, and the
//! reference a lend gives.

use crate::class::{ClassId, StructureConstraint};
use crate::program::Projection;
use crate::record::{Owner, Recording};
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Mutability, Term};

use super::{DisjunctionAt, LendTarget};

impl Projection {
    /// The key of the part the projection names.
    pub fn key(&self) -> crate::ty::PartKey {
        match self {
            Projection::Deref => crate::ty::PartKey::Target,
            Projection::Field(field) => crate::ty::PartKey::Field(field.clone()),
            Projection::Component(i) => crate::ty::PartKey::Component(*i),
            Projection::Elem => crate::ty::PartKey::Element,
            Projection::Payload { variant, .. } => crate::ty::PartKey::Payload(variant.clone()),
        }
    }

    /// The structure the projection gives the class it applies to, naming `part`.
    pub(super) fn structure_constraint(&self, part: ClassId) -> StructureConstraint {
        match self {
            Projection::Deref => StructureConstraint::Target(part),
            Projection::Field(field) => StructureConstraint::Field(field.clone(), part),
            Projection::Component(i) => StructureConstraint::Component(*i, part),
            Projection::Elem => StructureConstraint::Element(part),
            Projection::Payload { enum_name, variant } => StructureConstraint::Payload {
                enum_name: enum_name.clone(),
                variant: variant.clone(),
                payload: part,
            },
        }
    }
}

impl State {
    /// The reference a lend gives, of `mutability`, to `place` or to its
    /// target, the relations it records kept in `recorded`.
    pub(super) fn lend(
        &mut self,
        mutability: Mutability,
        place: ClassId,
        target: LendTarget,
        owner: &Owner,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let target = match target {
            LendTarget::Place => place,
            LendTarget::Through => {
                let target = self.make(owner.made(DisjunctionAt::Target));
                let relation =
                    Relation::StructureConstraint(place, StructureConstraint::Target(target));
                recorded.push(self.record(relation, recorder.clone()));
                target
            }
        };
        let lent = Term::Ref(mutability, target);
        self.written(owner.made(DisjunctionAt::Lent), lent, recorder, recorded)
    }
}
