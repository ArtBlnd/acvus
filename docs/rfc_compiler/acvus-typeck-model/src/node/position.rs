//! The positions a use, a lend, a projection or a binding lies at, and what
//! it reads (l.240 (b), l.243–246, l.263).

use std::collections::BTreeSet;

use crate::class::{ClassId, ClassName, Fact};
use crate::program::{BindingPath, Projection};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::{NodeKind, Read, Reads};

impl State {
    /// The classes at a node's positions, and those within the depth its
    /// candidates reach (l.240 (b)): for `e?`, its exit's value too (l.339);
    /// for a call of a function value, the parameters and the result type in
    /// the environment of each lambda its callee's Values name
    /// (`design/17-environment.md`, row 3).
    pub(crate) fn positions(&self, kind: &NodeKind) -> BTreeSet<ClassId> {
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
                exit,
            } => {
                let mut positions: BTreeSet<ClassId> =
                    args.iter().chain([result]).chain(exit).copied().collect();
                for candidate in self.candidates(callee, args.len()) {
                    let sig = &candidate.sig;
                    for (arg, param) in args.iter().zip(sig.params()) {
                        positions.extend(self.within_depth(*arg, param));
                    }
                    positions.extend(self.within_depth(*result, sig.result()));
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
    /// row 3), none where its binding group does.
    pub(crate) fn environment_of(&self, lambda: &Name) -> Vec<Option<ClassId>> {
        let function = self
            .functions
            .iter()
            .find(|function| function.name == *lambda)
            .expect("a lambda is a function");
        function
            .params
            .iter()
            .chain([&function.result])
            .map(|name| {
                self.environment.contains(name).then(|| {
                    self.arena
                        .get(&ClassName::Text(name.clone()))
                        .unwrap_or_else(|| {
                            panic!(
                                "{lambda}'s value reaches a group its environment is not in, \
                             which is not yet encoded (design/17-environment.md, Open)"
                            )
                        })
                })
            })
            .collect()
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
    /// boundary out of them (l.240 (b)): for a use, an argument each candidate
    /// writes a type for, and each class within the depth a candidate writes
    /// terms there, a split relating them by boundaries toward the types it
    /// writes (l.280), unless the class lies elsewhere at the use too, at its
    /// result, its exit, within the depth of its result, or at an argument a
    /// candidate writes a variable for. No other node records only so.
    pub(crate) fn passed_out_only(&self, kind: &NodeKind) -> BTreeSet<ClassId> {
        let NodeKind::Use {
            callee,
            args,
            result,
            exit,
        } = kind
        else {
            return BTreeSet::new();
        };
        let candidates = self.candidates(callee, args.len());
        let mut out = BTreeSet::new();
        let mut into: BTreeSet<ClassId> = [result]
            .into_iter()
            .chain(exit)
            .map(|&class| self.root(class))
            .collect();
        for candidate in &candidates {
            let sig = &candidate.sig;
            into.extend(self.within_depth(*result, sig.result()));
            for (arg, param) in args.iter().zip(sig.params()) {
                match param {
                    SigTy::Term(_) => {
                        out.insert(self.root(*arg));
                        out.extend(self.within_depth(*arg, param));
                    }
                    SigTy::Var(_) => {
                        into.insert(self.root(*arg));
                    }
                }
            }
        }
        out.difference(&into).copied().collect()
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
                    let class = args.get(i).copied().unwrap_or(*result);
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
    /// (l.175, l.183).
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
    /// read of whether a class is a reference leaves out (l.244, l.263, l.300).
    pub(crate) fn is_inner_projection(&self, name: &Name) -> bool {
        matches!(
            &self.nodes[name].kind,
            NodeKind::Project { projection, .. } if *projection != Projection::Deref
        )
    }

    /// Whether the node is a use that has recorded whether the class is a
    /// reference (l.248).
    pub(super) fn recorded_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        self.nodes[name]
            .references
            .iter()
            .any(|&recorded| self.root(recorded) == root)
    }

    /// What a node reads, and on which it depends (l.243–244, l.263).
    pub(crate) fn reads(&self, name: &Name) -> Reads {
        let kind = &self.nodes[name].kind;
        let path = match (kind, self.readings.binding_path) {
            (NodeKind::Bind { path, .. }, BindingPath::AfterPath) => path.clone(),
            _ => Vec::new(),
        };
        Reads {
            read: kind.read(),
            classes: self.read_positions(kind),
            path,
        }
    }

    /// Whether the relations the node `recorder` records may change what a
    /// node reading `read` reads at `class`, a class at the recorder's
    /// position or reached from one (l.240 (b), l.246). For a value, every
    /// such node (`holes.md`, H7); for whether a class is a reference, a node
    /// that may decide it, other than a projection of a field, a component, an
    /// index or a variant (l.244).
    pub(crate) fn records_into(&self, recorder: &Name, class: ClassId, read: Read) -> bool {
        match read {
            Read::Value => true,
            Read::Reference => {
                !self.is_inner_projection(recorder) && self.may_decide_reference(recorder, class)
            }
        }
    }

    /// Whether what the node records may yet decide whether the class is a
    /// reference, which is decided by its lower bounds and equations (l.300):
    /// for a use, where some candidate would record one into it and the use
    /// has not recorded whether it is one (l.244, l.248); for a lend, a binding
    /// and a call of a function value, where the class is the result it
    /// records into. A lend records into its place, and a binding into its
    /// position, only that it is a reference to a class, and only where it
    /// already reads as one (l.298; `design/08-node.md`, reading 12).
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
