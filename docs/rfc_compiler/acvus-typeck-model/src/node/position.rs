//! The positions a use, a lend, a projection or a binding lies at, and what
//! it reads (l.238 (b), l.241–244, l.261).

use std::collections::BTreeSet;

use crate::class::{ClassId, ClassName, Fact};
use crate::program::Projection;
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::common::CommonPart;
use super::{Common, NodeKind, NodeState, Read, Reads};

/// What a use has yet to record once decided (l.238 (b), l.246).
pub(crate) struct Left {
    /// The arguments it holds that its common part does not hold whole, by
    /// index with their classes.
    pub(crate) args: Vec<(usize, ClassId)>,
    /// The classes its result enters, each with the type each candidate
    /// writes there.
    pub(crate) results: Vec<(ClassId, Vec<SigTy>)>,
}

impl State {
    /// What a use has yet to record once decided (l.238 (b), l.246): the
    /// arguments it holds, which its common part does not hold whole, by
    /// index with their classes, and the classes its result enters, each with
    /// the type each candidate writes there: the result, where it holds it,
    /// or, once the common part is recorded, each of its variables the result
    /// writes.
    pub(crate) fn left_to_record(&self, node: &NodeState) -> Left {
        let NodeKind::Use {
            callee,
            args,
            result,
            ..
        } = &node.kind
        else {
            return Left {
                args: Vec::new(),
                results: Vec::new(),
            };
        };
        let sigs: Vec<&Signature> = self
            .candidates(callee, args.len())
            .into_iter()
            .map(|decl| &decl.sig)
            .collect();
        let held: Vec<(usize, ClassId)> = args
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| Some((i, (*arg)?)))
            .collect();
        match &node.common {
            Some(Common::Use { part, classes }) => {
                let results = CommonPart::vars_in(&part.result)
                    .into_iter()
                    .map(|k| (classes[k], part.vars[k].clone()))
                    .collect();
                Left {
                    args: held,
                    results,
                }
            }
            _ => {
                let results = result
                    .iter()
                    .map(|&result| {
                        (
                            result,
                            sigs.iter().map(|sig| sig.result().clone()).collect(),
                        )
                    })
                    .collect();
                Left {
                    args: held,
                    results,
                }
            }
        }
    }

    /// The classes at a node's positions, and those within the depth its
    /// candidates reach, where it has relations yet to record (l.238 (b)):
    /// for a use, the arguments and the classes its result enters that its
    /// common part does not hold (l.246), and for `e?`, its exit's value too
    /// (l.337); for a call of a function value, the parameters and the result
    /// type in the environment of each lambda its callee's Values name
    /// (`design/17-environment.md`, row 3).
    pub(crate) fn positions(&self, node: &NodeState) -> BTreeSet<ClassId> {
        match &node.kind {
            NodeKind::Use {
                callee, args, exit, ..
            } => {
                let Left {
                    args: left,
                    results,
                } = self.left_to_record(node);
                let mut positions: BTreeSet<ClassId> = exit.iter().copied().collect();
                let candidates = self.candidates(callee, args.len());
                for (i, arg) in left {
                    positions.insert(arg);
                    for candidate in &candidates {
                        let param = &candidate.sig.params()[i];
                        positions.extend(self.within_depth(arg, param));
                        let (params, results) = self.environment_within(arg, param);
                        positions.extend(params);
                        positions.extend(results);
                    }
                }
                for (class, tys) in results {
                    positions.insert(class);
                    for ty in &tys {
                        positions.extend(self.within_depth(class, ty));
                    }
                }
                positions
            }
            NodeKind::Lend { place, result, .. } => [*place, *result].into(),
            NodeKind::Project {
                operand, result, ..
            } => {
                let mut positions: BTreeSet<ClassId> = [*operand, *result].into();
                positions.extend(self.targets(*operand));
                positions
            }
            NodeKind::Bind {
                position, binding, ..
            } => [*position, *binding].into(),
            NodeKind::CallValue {
                callee,
                args,
                result,
            } => {
                let mut positions: BTreeSet<ClassId> = [callee]
                    .into_iter()
                    .chain(args)
                    .chain([result])
                    .copied()
                    .collect();
                for lambda in self.values_of(*callee) {
                    positions.extend(self.environment_of(&lambda).into_iter().flatten());
                }
                positions
            }
        }
    }

    /// The parameter types of the lambda, and then its result type, each the
    /// class here where its environment holds it (`design/17-environment.md`,
    /// row 3), none where its binding group does: for a lambda an instance
    /// made anew, that instance's copies (l.215).
    pub(crate) fn environment_of(&self, lambda: &Name) -> Vec<Option<ClassId>> {
        self.environment_in(lambda).unwrap_or_else(|| {
            panic!(
                "{lambda}'s value reaches a group its environment is not in, \
                 which is not yet encoded (design/17-environment.md, Open)"
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

    /// The lambda of the program a lambda here is, or copies (l.215).
    pub(crate) fn lambda_function(&self, lambda: &Name) -> &crate::program::Function {
        let of = self.lambdas.get(lambda).map_or(lambda, |copy| &copy.of);
        self.functions
            .iter()
            .find(|function| function.name == *of)
            .expect("a lambda is a function")
    }

    /// The classes the equivalence class of `id` is known to be a reference
    /// to, by its kind or a kind known in part.
    pub(super) fn targets(&self, id: ClassId) -> BTreeSet<ClassId> {
        let held = self.classes.held(id);
        let by_kind = match &held.kind {
            Some((Term::Ref(_, target), _)) => Some(self.root(*target)),
            _ => None,
        };
        let by_fact = held.facts.iter().filter_map(|fact| match fact {
            Fact::Ref(target) => Some(self.root(*target)),
            _ => None,
        });
        by_kind.into_iter().chain(by_fact).collect()
    }

    /// The classes at a node's positions into which all the node records is a
    /// boundary out of them (l.238 (b)): for a use, an argument each candidate
    /// writes a type for, and each class within the depth a candidate writes
    /// terms there, a split relating them by boundaries toward the types it
    /// writes (l.278), unless the class lies elsewhere at the use too, at its
    /// result, its exit, within the depth of its result, or at an argument a
    /// candidate writes a variable for. No other node records only so.
    pub(crate) fn passed_out_only(&self, node: &NodeState) -> BTreeSet<ClassId> {
        let NodeKind::Use {
            callee, args, exit, ..
        } = &node.kind
        else {
            return BTreeSet::new();
        };
        let Left {
            args: left,
            results,
        } = self.left_to_record(node);
        let candidates = self.candidates(callee, args.len());
        let mut out = BTreeSet::new();
        let mut into: BTreeSet<ClassId> = exit.iter().map(|&class| self.root(class)).collect();
        for (class, tys) in &results {
            into.insert(self.root(*class));
            for ty in tys {
                into.extend(self.within_depth(*class, ty));
            }
        }
        for candidate in &candidates {
            let sig = &candidate.sig;
            for &(i, arg) in &left {
                let (arg, param) = (&arg, &sig.params()[i]);
                match param {
                    SigTy::Term(_) => {
                        out.insert(self.root(*arg));
                        out.extend(self.within_depth(*arg, param));
                        let (params, results) = self.environment_within(*arg, param);
                        into.extend(params);
                        out.extend(results);
                    }
                    SigTy::Var(_) => {
                        into.insert(self.root(*arg));
                    }
                }
            }
        }
        out.difference(&into).copied().collect()
    }

    /// Where `ty` writes a function type at a class holding a function value,
    /// at any depth it writes terms, the parameter and the result types in the
    /// environment of each lambda its Values names, as the parts at the function
    /// type's parameters and result (l.238 (b)): the parameters, into which the
    /// boundary's split records, and the results, which pass out of it.
    pub(super) fn environment_within(
        &self,
        class: ClassId,
        ty: &SigTy,
    ) -> (BTreeSet<ClassId>, BTreeSet<ClassId>) {
        let (mut params, mut results) = (BTreeSet::new(), BTreeSet::new());
        let SigTy::Term(term) = ty else {
            return (params, results);
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return (params, results);
        };
        if let (Term::Function { .. }, Term::FunctionValue { .. }) = (&**term, kind) {
            for lambda in self.values_of(class) {
                let mut held = self.environment_of(&lambda);
                let result = held.pop().expect("the result type follows the parameters");
                params.extend(held.into_iter().flatten().map(|c| self.root(c)));
                results.extend(result.map(|c| self.root(c)));
            }
            return (params, results);
        }
        for (key, written) in term.parts() {
            if let Some(&part) = kind.part(&key) {
                let (p, r) = self.environment_within(part, written);
                params.extend(p);
                results.extend(r);
            }
        }
        (params, results)
    }

    /// The classes at the positions inside `class`'s kind where `ty` writes a
    /// term, down to where it writes a variable.
    pub(super) fn within_depth(&self, class: ClassId, ty: &SigTy) -> BTreeSet<ClassId> {
        let mut found = BTreeSet::new();
        let SigTy::Term(term) = ty else {
            return found;
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return found;
        };
        for (key, written) in term.parts() {
            if let (SigTy::Term(_), Some(&part)) = (written, kind.part(&key)) {
                found.insert(self.root(part));
                found.extend(self.within_depth(part, written));
            }
        }
        found
    }

    /// The classes at the positions a node reads: for a use, those where its
    /// candidates differ and those within the depth its candidates reach there
    /// (`design/08-node.md`, reading 11); for a lend, its place; for a
    /// projection other than a Deref, its operand; for a binding, its position.
    /// A Deref reads nothing (reading 6).
    pub(crate) fn read_positions(&self, kind: &NodeKind) -> Vec<ClassId> {
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
                ..
            } => {
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
                    let class = Self::differing_class(args, *result, i);
                    read.insert(self.root(class));
                    for sig in &candidates {
                        read.extend(self.within_depth(class, Self::at_position(sig, i)));
                    }
                }
                read.into_iter().collect()
            }
            NodeKind::Lend { place, .. } => vec![*place],
            NodeKind::Project {
                projection: Projection::Deref,
                ..
            } => Vec::new(),
            NodeKind::Project { operand, .. } => vec![*operand],
            NodeKind::Bind { position, .. } => vec![*position],
            NodeKind::CallValue { callee, .. } => vec![*callee],
        }
    }

    /// The lambdas the Values of the class's function value names: those of
    /// the function values that meet in it, by equations or lower bounds
    /// (l.173, l.181).
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

    /// Whether the node is a projection other than a Deref, whose relations a
    /// read of whether a class is a reference leaves out (l.242, l.261, l.298).
    pub(crate) fn is_inner_projection(&self, name: &Name) -> bool {
        matches!(
            &self.nodes[name].kind,
            NodeKind::Project { projection, .. } if *projection != Projection::Deref
        )
    }

    /// Whether the node's common part recorded whether the class is a
    /// reference (l.246).
    pub(super) fn recorded_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        self.nodes[name]
            .references
            .iter()
            .any(|&recorded| self.root(recorded) == root)
    }

    /// What a node reads, and on which it depends (l.241–242, l.261).
    pub(crate) fn reads(&self, name: &Name) -> Reads {
        let kind = &self.nodes[name].kind;
        Reads {
            read: kind.read(),
            classes: self.read_positions(kind),
        }
    }

    /// Whether the relations the node `recorder` records may change what a
    /// node reading `read` at `position` reads at `class`, a class at the
    /// recorder's position or reached from one (l.238 (b), l.244). For a
    /// value, every such node (`holes.md`, H7); for whether a class is a
    /// reference, a node that may decide it, other than a projection of a
    /// field, a component, an index or a variant whose operand is the class
    /// read (l.242), whose kind on it says nothing of whether it is one.
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

    /// Whether the node is a projection of a field, a component, an index or
    /// a variant whose operand is the class of `class` (l.242, l.261).
    pub(crate) fn projects_from(&self, name: &Name, class: ClassId) -> bool {
        matches!(
            &self.nodes[name].kind,
            NodeKind::Project { projection, operand, .. }
                if *projection != Projection::Deref && self.root(*operand) == self.root(class)
        )
    }

    /// Whether what the node records may yet decide whether the class is a
    /// reference, which is decided by its lower bounds and equations (l.298):
    /// for a use, where some candidate would record one into it and its
    /// common part has not recorded whether it is one (l.242, l.246); for a
    /// lend, a binding and a call of a function value, where the class is the
    /// result it records into: a lend's common part records that its result
    /// is a reference, but not its mutability. A lend records into its place, and a binding
    /// into its position, only that it is a reference to a class, and only
    /// where it already reads as one (l.296; `design/08-node.md`, reading 12).
    pub(super) fn may_decide_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        match &self.nodes[name].kind {
            NodeKind::Use { .. } => {
                !self.recorded_reference(name, class)
                    && self
                        .would_record_per_candidate(name)
                        .iter()
                        .any(|found| found.contains_key(&root))
            }
            NodeKind::Lend { result, .. } | NodeKind::CallValue { result, .. } => {
                self.root(*result) == root
            }
            NodeKind::Bind { binding, .. } => self.root(*binding) == root,
            NodeKind::Project { .. } => true,
        }
    }
}
