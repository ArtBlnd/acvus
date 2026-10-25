//! Places: what a lend's place, a projection's operand and a
//! binding's path read as, and the reference a lend gives.

use std::collections::BTreeSet;

use crate::class::{ClassId, ClassName, StructureConstraint};
use crate::domain::Domain;
use crate::program::Projection;
use crate::read::ReadValue;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{HasTerm, Mutability, Name, Term};

/// What a class reads as in a value read: its value there, and
/// so none where it reads as no reference or as nothing, and a reference, of
/// its mutability where the value read decides one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AsReference {
    pub class: ClassId,
    pub value: ReadValue,
    pub reads_as: Option<Option<Mutability>>,
}

use super::{Decision, DisjunctionAt, LendTarget};

/// The lesser of two mutabilities, where either may be undecided.
pub(super) fn lesser(a: Option<Mutability>, b: Option<Mutability>) -> Option<Mutability> {
    match (a, b) {
        (Some(Mutability::Shared), _) | (_, Some(Mutability::Shared)) => Some(Mutability::Shared),
        (Some(Mutability::Exclusive), Some(Mutability::Exclusive)) => Some(Mutability::Exclusive),
        _ => None,
    }
}

impl Projection {
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
    /// What the class reads as in the value read leaving out `excluded`: none
    /// where it reads as no reference or as nothing, and a reference, of its
    /// mutability where the value read decides one. Where the
    /// value read decides no constructor, a generalization's record that a class
    /// it reaches by lower bounds is a reference still reads as a reference.
    pub(super) fn reads_as_reference(&self, id: ClassId, excluded: &BTreeSet<Name>) -> AsReference {
        let read = self.read_state(excluded);
        let value = read.read_value(id);
        let reads_as = match value.decided() {
            Some(Term::Ref(mutability, _)) => Some(Some(*mutability)),
            Some(_) => None,
            None => read.recorded_as_reference(id).then_some(None),
        };
        AsReference {
            class: id,
            value,
            reads_as,
        }
    }

    /// Whether a disjunctive constraint's generalization or a lend gave the class, or one it
    /// reaches by lower bounds, a domain of references.
    pub(super) fn recorded_as_reference(&self, id: ClassId) -> bool {
        let mut seen = BTreeSet::from([self.root(id)]);
        let mut work = vec![self.root(id)];
        while let Some(at) = work.pop() {
            if self
                .classes
                .held(at)
                .domain
                .as_ref()
                .is_some_and(Domain::is_references)
            {
                return true;
            }
            for lower in self.lower_bounds(at) {
                if seen.insert(self.root(lower)) {
                    work.push(self.root(lower));
                }
            }
        }
        false
    }

    /// Whether the projections on a binding's path pass a reference, after its
    /// last Deref, which matches a reference itself, and
    /// the least of their mutabilities where decided. A projection not yet
    /// decided has passed none.
    pub(super) fn passed(&self, path: &[Name]) -> Option<Option<Mutability>> {
        let after_deref = path
            .iter()
            .rposition(|step| !self.is_inner_projection(step))
            .map_or(0, |deref| deref + 1);
        let passed: Vec<Option<Mutability>> = path[after_deref..]
            .iter()
            .filter_map(|step| match self.disjunctions[step].decision {
                Some(Decision::AppliedToTarget(mutability)) => Some(mutability),
                _ => None,
            })
            .collect();
        if passed.is_empty() {
            return None;
        }
        Some(passed.into_iter().fold(Some(Mutability::Exclusive), lesser))
    }

    /// The reference a lend gives, of `mutability`, to `place` or to its
    /// target, the relations it records kept in `recorded`.
    pub(super) fn lend<F>(
        &mut self,
        mutability: Mutability,
        place: ClassId,
        target: LendTarget,
        at: &F,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId
    where
        F: Fn(DisjunctionAt) -> ClassName,
    {
        let target = match target {
            LendTarget::Place => place,
            LendTarget::Through(_) => {
                let target = self.arena.intern(at(DisjunctionAt::Target));
                self.classes.grow(&self.arena);
                let relation =
                    Relation::StructureConstraint(place, StructureConstraint::Target(target));
                recorded.push(self.record(relation, recorder.clone()));
                target
            }
        };
        let lent = Term::Ref(mutability, target);
        self.written(at(DisjunctionAt::Lent), lent, recorder, recorded)
    }
}
