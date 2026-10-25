//! What a use, a lend, a projection or a binding records into, on which the
//! classes it records into depend, and what it reads.

use std::collections::BTreeSet;

use crate::class::{ClassId, ClassName, StructureConstraint};
use crate::program::Projection;
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::{Decision, Disjunction, Read, Reads};

impl State {
    /// The classes the disjunctive constraint records into, by what each decision still open to
    /// it records, directly or by the decompositions of what it records: a use, what
    /// a candidate left would record into (`State::would_record_by`), and its
    /// result where it is decided `!`; a lend, its result, and its place, which
    /// a lend through a reference gives a target made anew, holding nothing; a
    /// projection, its operand and its result, and the operand's target, which
    /// it gives the member it names where it applies to the target; a binding,
    /// its binding, and its position, which a binding that lends gives a target
    /// made anew; a call of a function value, its result and the
    /// parameters in the environment of each lambda its callee's Values name,
    /// which its arguments enter. A class depends on the disjunctive constraint, until the disjunctive constraint
    /// has recorded, where it is one of these alone.
    pub(crate) fn recorded_into_by(&self, name: &Name) -> BTreeSet<ClassId> {
        let disjunction = &self.disjunctions[name];
        let classes: BTreeSet<ClassId> = match &disjunction.kind {
            Disjunction::Use {
                callee,
                args,
                result,
                ..
            } => match &disjunction.decision {
                Some(Decision::Never) => result.iter().copied().collect(),
                Some(Decision::Use(id)) => {
                    let candidate = self.candidate_index(callee, args.len(), id);
                    self.would_record_by(name, candidate).into_keys().collect()
                }
                Some(other) => unreachable!("a use is decided as a candidate or never: {other:?}"),
                None => self
                    .would_record_per_candidate(name)
                    .into_iter()
                    .flat_map(|found| found.into_keys())
                    .collect(),
            },
            Disjunction::Lend { place, result, .. } => [*place, *result].into_iter().collect(),
            Disjunction::Project {
                operand, result, ..
            } => [*operand, *result]
                .into_iter()
                .chain(self.targets(*operand))
                .collect(),
            Disjunction::Bind {
                position, binding, ..
            } => [*position, *binding].into_iter().collect(),
            Disjunction::CallValue { callee, result, .. } => {
                let mut classes = BTreeSet::from([*result]);
                for lambda in self.values_of(self.called(*callee)) {
                    let mut held = self.environment_of(&lambda);
                    held.pop();
                    classes.extend(held.into_iter().flatten());
                }
                classes
            }
        };
        classes.into_iter().map(|class| self.root(class)).collect()
    }

    /// The parameter types of the lambda, and then its result type, each the
    /// class here where its environment holds it, none where its binding group does:
    /// for a lambda an instance
    /// made anew, that instance's copies.
    pub(crate) fn environment_of(&self, lambda: &Name) -> Vec<Option<ClassId>> {
        self.environment_in(lambda).unwrap_or_else(|| {
            panic!(
                "{lambda}'s value reaches a group its environment is not in, \
                 which is not yet encoded"
            )
        })
    }

    /// As `environment_of`, none where its environment is not here.
    pub(crate) fn environment_in(&self, lambda: &Name) -> Option<Vec<Option<ClassId>>> {
        if let Some(copy) = self.lambdas.get(lambda) {
            return Some(copy.environment.clone());
        }
        let function = self.lambda_function(lambda);
        function
            .params
            .iter()
            .chain([&function.result])
            .map(|name| {
                if self.environment.contains(name) {
                    self.arena.get(&ClassName::Text(name.clone())).map(Some)
                } else {
                    Some(None)
                }
            })
            .collect()
    }

    /// The lambda of the program a lambda here is, or copies.
    pub(crate) fn lambda_function(&self, lambda: &Name) -> &crate::program::Function {
        let of = self.lambdas.get(lambda).map_or(lambda, |copy| &copy.of);
        self.functions
            .iter()
            .find(|function| function.name == *of)
            .expect("a lambda is a function")
    }

    /// The classes the equivalence class of `id` is known to be a reference
    /// to, by its structure or a structure constraint.
    pub(super) fn targets(&self, id: ClassId) -> BTreeSet<ClassId> {
        let held = self.classes.held(id);
        let by_structure = match &held.structure {
            Some((Term::Ref(_, target), _)) => Some(self.root(*target)),
            _ => None,
        };
        let by_structure_constraint =
            held.structure_constraints
                .iter()
                .filter_map(|structure_constraint| match structure_constraint {
                    StructureConstraint::Target(target) => Some(self.root(*target)),
                    _ => None,
                });
        by_structure
            .into_iter()
            .chain(by_structure_constraint)
            .collect()
    }

    /// The classes at the positions inside `class`'s structure where `ty` writes a
    /// term, down to where it writes a variable.
    pub(super) fn within_depth(&self, class: ClassId, ty: &SigTy) -> BTreeSet<ClassId> {
        let mut found = BTreeSet::new();
        let SigTy::Term(term) = ty else {
            return found;
        };
        let Some((structure, _)) = &self.classes.held(class).structure else {
            return found;
        };
        for (key, written) in term.parts() {
            if let (SigTy::Term(_), Some(&part)) = (written, structure.part(&key)) {
                found.insert(self.root(part));
                found.extend(self.within_depth(part, written));
            }
        }
        found
    }

    /// The classes at the positions a disjunctive constraint reads: for a use, the arguments
    /// where its candidates differ and those within the depth its candidates
    /// reach there;
    /// for a lend, its place; for a
    /// projection other than a Deref, its operand; for a binding, its position.
    /// A Deref reads nothing.
    pub(crate) fn read_positions(&self, kind: &Disjunction) -> Vec<ClassId> {
        match kind {
            Disjunction::Use { callee, args, .. } => {
                let candidates: Vec<&Signature> = self
                    .candidates(callee, args.len())
                    .into_iter()
                    .map(|decl| &decl.sig)
                    .collect();
                if candidates.len() < 2 {
                    return Vec::new();
                }
                let mut read = BTreeSet::new();
                for i in Self::differing(&candidates, args.len()) {
                    let class = Self::differing_class(args, i);
                    read.insert(self.root(class));
                    for sig in &candidates {
                        read.extend(self.within_depth(class, &sig.params()[i]));
                    }
                }
                read.into_iter().collect()
            }
            Disjunction::Lend { place, .. } => vec![*place],
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => Vec::new(),
            Disjunction::Project { operand, .. } => vec![*operand],
            Disjunction::Bind { position, .. } => vec![*position],
            Disjunction::CallValue { callee, .. } => {
                let called = self.called(*callee);
                [*callee, called]
                    .into_iter()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            }
        }
    }

    /// The class whose function value a call of `callee` calls: the target of
    /// the reference `callee` holds, as a projection is taken of a reference's
    /// target, and `callee` itself otherwise.
    pub(crate) fn called(&self, callee: ClassId) -> ClassId {
        let root = self.root(callee);
        match &self.classes.held(root).structure {
            Some((Term::Ref(_, target), _)) => self.root(*target),
            _ => root,
        }
    }

    /// The lambdas the Values of the class's function value names: those of
    /// the function values that reach it, by equations or lower bounds.
    pub(crate) fn values_of(&self, id: ClassId) -> BTreeSet<Name> {
        let mut seen = BTreeSet::from([self.root(id)]);
        let mut work = vec![self.root(id)];
        let mut values = BTreeSet::new();
        while let Some(at) = work.pop() {
            values.extend(self.classes.held(at).values.iter().cloned());
            for lower in self.lower_bounds(at) {
                if seen.insert(self.root(lower)) {
                    work.push(self.root(lower));
                }
            }
        }
        values
    }

    /// Whether the disjunctive constraint is a projection other than a Deref, whose relations a
    /// read of whether a class is a reference leaves out.
    pub(crate) fn is_inner_projection(&self, name: &Name) -> bool {
        matches!(
            &self.disjunctions[name].kind,
            Disjunction::Project { projection, .. } if *projection != Projection::Deref
        )
    }

    /// Whether the disjunctive constraint's generalization recorded whether the class is a
    /// reference.
    pub(super) fn recorded_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        self.disjunctions[name]
            .references
            .iter()
            .any(|&recorded| self.root(recorded) == root)
    }

    /// What a disjunctive constraint reads, and on which it depends.
    pub(crate) fn reads(&self, name: &Name) -> Reads {
        let kind = &self.disjunctions[name].kind;
        Reads {
            read: kind.read(),
            classes: self.read_positions(kind),
        }
    }

    /// Whether the relations the disjunctive constraint `recorder` records may change what a
    /// disjunctive constraint reading `read` at `position` reads at `class`, a class at the
    /// recorder's position or reached from one. For a
    /// value, every such disjunctive constraint; for whether a class is a
    /// reference, a disjunctive constraint that may decide it, other than a projection of a
    /// field, a component, an index or a variant whose operand is the class
    /// read, whose structure constraint on it says nothing of whether it is one.
    pub(crate) fn records_into(
        &self,
        recorder: &Name,
        position: ClassId,
        class: ClassId,
        read: Read,
    ) -> bool {
        match read {
            Read::Value => true,
            Read::Reference => {
                !self.projects_from(recorder, position)
                    && self.may_decide_reference(recorder, class)
            }
        }
    }

    /// Whether the disjunctive constraint is a projection of a field, a component, an index or
    /// a variant whose operand is the class of `class`.
    pub(crate) fn projects_from(&self, name: &Name, class: ClassId) -> bool {
        matches!(
            &self.disjunctions[name].kind,
            Disjunction::Project { projection, operand, .. }
                if *projection != Projection::Deref && self.root(*operand) == self.root(class)
        )
    }

    /// Whether what the disjunctive constraint records may yet decide whether the class is a
    /// reference, which is decided by its lower bounds and equations:
    /// for a use, where some candidate would record one into it and its
    /// generalization has not recorded whether it is one; for a
    /// lend, a binding and a call of a function value, where the class is the
    /// result it records into: a lend's generalization records that its result
    /// is a reference, but not its mutability. A lend records into its place, and a binding
    /// into its position, only that it is a reference to a class, and only
    /// where it already reads as one.
    pub(super) fn may_decide_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        match &self.disjunctions[name].kind {
            Disjunction::Use { .. } => {
                !self.recorded_reference(name, class)
                    && self
                        .would_record_per_candidate(name)
                        .iter()
                        .any(|found| found.contains_key(&root))
            }
            Disjunction::Lend { result, .. } | Disjunction::CallValue { result, .. } => {
                self.root(*result) == root
            }
            Disjunction::Bind { binding, .. } => self.root(*binding) == root,
            Disjunction::Project { .. } => true,
        }
    }
}
