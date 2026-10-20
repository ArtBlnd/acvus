//! Uses, lends, projections and bindings: the positions each lies at, what
//! each reads, and what each records once decided (l.303–345).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Fact, KindSource};
use crate::coerce::coerce_read;
use crate::decl::{Decl, DeclId};
use crate::program::{BindingPath, Deferral, Projection, Recording};
use crate::read::ReadValue;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{HasTerm, Mutability, Name, PartKey, Term, Variance};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NodeKind {
    /// A use that calls the declarations `callee` names (l.331).
    Use {
        callee: Name,
        args: Vec<ClassId>,
        result: ClassId,
    },
    /// `&ₘ p` (l.307).
    Lend {
        mutability: Mutability,
        place: ClassId,
        result: ClassId,
    },
    /// A projection of `operand`, naming `result` (l.151).
    Project {
        projection: Projection,
        operand: ClassId,
        result: ClassId,
    },
    /// A binding at `position`, reached by the projections `path` (l.317).
    Bind {
        path: Vec<Name>,
        position: ClassId,
        binding: ClassId,
    },
    /// A use that calls the function value of `callee` (l.203).
    CallValue {
        callee: ClassId,
        args: Vec<ClassId>,
        result: ClassId,
    },
}

/// What a node reads at the classes at its positions (l.267).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Read {
    /// Their values: a use, and a call of a function value.
    Value,
    /// Whether they are references: a lend, a projection, a binding.
    Reference,
}

impl NodeKind {
    pub(crate) fn read(&self) -> Read {
        match self {
            NodeKind::Lend { .. } | NodeKind::Project { .. } | NodeKind::Bind { .. } => {
                Read::Reference
            }
            NodeKind::Use { .. } | NodeKind::CallValue { .. } => Read::Value,
        }
    }
}

/// What a node reads (l.267), on which it depends (l.247–248).
pub(crate) struct Reads {
    pub(crate) read: Read,
    /// The classes at the positions it reads.
    pub(crate) classes: Vec<ClassId>,
    /// The projections on a binding's path, where the reading takes it so
    /// (`holes.md`, O1).
    pub(crate) path: Vec<Name>,
}

/// What a use, a lend, a projection or a binding became (l.267).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decision {
    /// The use is decided as this candidate (l.344).
    Use(DeclId),
    /// The use leaves no candidate, or several, and is refused (l.344).
    Unreached(Vec<DeclId>),
    /// The use's argument reads as `!`, so it is never carried out and is
    /// decided with the result `!` (l.344).
    Never,
    /// The lend gives a reference of `mutability` to `target` (l.307).
    Lend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The projection applies to its operand (l.151).
    Applied,
    /// The projection's operand is a reference, so it applies to that
    /// reference's target (auto-deref, l.309); the mutability is the
    /// reference's where the value read decides it.
    AppliedToTarget(Option<Mutability>),
    /// The binding's path passes no reference, so it reads its position
    /// (l.317).
    Read,
    /// The binding's path passes a reference, so it lends its position, giving
    /// a reference of `mutability` to `target` (l.317).
    BindLend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// What the node reads decides nothing of the mutability its reference
    /// needs, or the function value it calls names no lambda, so it gives no
    /// result (l.267, l.301).
    NoResult,
    /// The call reads the function value as one whose Values names these
    /// lambdas, and makes an instance of each (l.203).
    Calls(BTreeSet<Name>),
}

/// What a lend's place reads as (l.307).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LendTarget {
    /// Not a reference, so the target is the place.
    Place,
    /// A reference, so the target is that reference's; the mutability is the
    /// reference's where the value read decides it.
    Through(Option<Mutability>),
}

impl Decision {
    /// Whether the node is reached by what it reads, and records.
    pub fn is_reached(&self) -> bool {
        !matches!(self, Decision::Unreached(_) | Decision::NoResult)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NodeState {
    pub kind: NodeKind,
    pub decision: Option<Decision>,
    /// The positions of a use's signature that all its candidates share, as
    /// argument indexes with the result last, recorded before the use is
    /// decided (l.335).
    pub shared: BTreeSet<usize>,
    /// The classes of which a use recorded whether they are references before
    /// it is decided (l.252).
    pub references: BTreeSet<ClassId>,
}

/// Whether the type a signature writes is a reference, none where it is a
/// variable.
fn is_reference(ty: &SigTy) -> Option<bool> {
    match ty {
        SigTy::Var(_) => None,
        SigTy::Term(term) => Some(matches!(**term, Term::Ref(..))),
    }
}

/// The lesser of two mutabilities, where either may be undecided (l.307).
fn lesser(a: Option<Mutability>, b: Option<Mutability>) -> Option<Mutability> {
    match (a, b) {
        (Some(Mutability::Shared), _) | (_, Some(Mutability::Shared)) => Some(Mutability::Shared),
        (Some(Mutability::Exclusive), Some(Mutability::Exclusive)) => Some(Mutability::Exclusive),
        _ => None,
    }
}

/// Which class a node made anew is (l.331).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum NodeAt {
    /// A variable of the signature of the candidate the use is decided as.
    Var(crate::sig::SigVar),
    /// The type of the parameter at this place.
    Param(usize),
    Result,
    /// The reference a lend gives.
    Lent,
    /// The target of the reference a lend's place, or a projection's operand,
    /// is.
    Target,
}

impl Projection {
    /// The kind the projection gives the class it applies to, naming `part`
    /// (l.153–159).
    fn fact(&self, part: ClassId) -> Fact {
        match self {
            Projection::Deref => Fact::Ref(part),
            Projection::Field(field) => Fact::Field(field.clone(), part),
            Projection::Component(i) => Fact::Component(*i, part),
            Projection::Elem => Fact::Indexes(part),
            Projection::Payload { enum_name, variant } => Fact::Variant {
                enum_name: enum_name.clone(),
                variant: variant.clone(),
                payload: part,
            },
        }
    }
}

impl State {
    /// The candidates of a use: the declarations its name reaches whose
    /// parameters are as many as its arguments (l.331).
    pub(crate) fn candidates(&self, callee: &Name, arity: usize) -> Vec<&Decl> {
        self.decls
            .candidates(callee)
            .iter()
            .filter(|decl| decl.sig.params().len() == arity)
            .collect()
    }

    /// The classes at a node's positions, and those within the depth its
    /// candidates reach (l.244 (b)).
    pub(crate) fn positions(&self, kind: &NodeKind) -> BTreeSet<ClassId> {
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
            } => {
                let mut positions: BTreeSet<ClassId> =
                    args.iter().chain([result]).copied().collect();
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
            } => [callee]
                .into_iter()
                .chain(args)
                .chain([result])
                .copied()
                .collect(),
        }
    }

    /// The classes the equivalence class of `id` is known to be a reference
    /// to, by its kind or a kind known in part.
    fn targets(&self, id: ClassId) -> BTreeSet<ClassId> {
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

    /// The classes at the positions inside `class`'s kind where `ty` writes a
    /// term, down to where it writes a variable.
    fn within_depth(&self, class: ClassId, ty: &SigTy) -> BTreeSet<ClassId> {
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

    /// A position of a candidate's signature, by argument index with the
    /// result last.
    fn at_position(sig: &Signature, i: usize) -> &SigTy {
        sig.params().get(i).unwrap_or(sig.result())
    }

    /// The positions where a use's candidates differ, which it reads (l.267).
    fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        (0..=args)
            .filter(|&i| {
                let first = Self::at_position(candidates[0], i);
                candidates
                    .iter()
                    .any(|sig| Self::at_position(sig, i) != first)
            })
            .collect()
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
    /// (l.180, l.188).
    fn values_of(&self, id: ClassId) -> BTreeSet<Name> {
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
    /// read of whether a class is a reference leaves out (l.248, l.267, l.309).
    pub(crate) fn is_inner_projection(&self, name: &Name) -> bool {
        matches!(
            &self.nodes[name].kind,
            NodeKind::Project { projection, .. } if *projection != Projection::Deref
        )
    }

    /// Whether the node is a use that has recorded whether the class is a
    /// reference (l.252).
    fn recorded_reference(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        self.nodes[name]
            .references
            .iter()
            .any(|&recorded| self.root(recorded) == root)
    }

    /// What a node reads, and on which it depends (l.247–248, l.267).
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
    /// position or reached from one (l.244 (b), l.250). For a value, every
    /// such node, or, as `Recording::Relation` reads l.250, one that would
    /// record into the class (`holes.md`, H7); for whether a class is a
    /// reference, a node that may decide it, other than a projection of a
    /// field, a component, an index or a variant (l.248).
    pub(crate) fn records_into(&self, recorder: &Name, class: ClassId, read: Read) -> bool {
        match (read, self.readings.recording) {
            (Read::Value, Recording::Position) => true,
            (Read::Value, Recording::Relation) => self.may_record_into(recorder, class),
            (Read::Reference, _) => {
                !self.is_inner_projection(recorder) && self.may_decide_reference(recorder, class)
            }
        }
    }

    /// Whether what the node records may yet decide whether the class is a
    /// reference, which is decided by its lower bounds and equations (l.309):
    /// for a use, where some candidate would record one into it and the use
    /// has not recorded whether it is one (l.248, l.252); for a lend, a binding
    /// and a call of a function value, where the class is the result it
    /// records into. A lend records into its place, and a binding into its
    /// position, only that it is a reference to a class, and only where it
    /// already reads as one (l.307; `design/08-node.md`, reading 12).
    fn may_decide_reference(&self, name: &Name, class: ClassId) -> bool {
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

    /// Whether the node would record a lower bound, an equation or a kind into
    /// the class, or into a type the class is an invariant part of, through
    /// which an equation reaches it (`Recording`; `design/08-node.md`, reading
    /// 18): a use where some candidate would (reading 13); a lend into its
    /// result and its place, the target of the reference it equates with its
    /// result (l.307); a binding into its own type and its position, which it
    /// may lend; and a projection or a call of a function value into every
    /// class at its positions.
    fn may_record_into(&self, name: &Name, class: ClassId) -> bool {
        let root = self.root(class);
        match &self.nodes[name].kind {
            NodeKind::Use { .. } => self
                .would_record_per_candidate(name)
                .iter()
                .any(|found| found.contains_key(&root)),
            NodeKind::Lend { place, result, .. } => {
                self.root(*place) == root || self.root(*result) == root
            }
            NodeKind::Bind {
                position, binding, ..
            } => self.root(*position) == root || self.root(*binding) == root,
            NodeKind::Project { .. } | NodeKind::CallValue { .. } => true,
        }
    }

    /// The nodes whose relations the node's read leaves out: itself and those
    /// that transitively depend on it, and for a lend, a projection or a
    /// binding, which reads whether a class is a reference, every projection
    /// other than a Deref (l.267).
    fn excluded(&self, name: &Name) -> BTreeSet<Name> {
        let mut excluded = self.dependents(name);
        excluded.extend(self.broken.get(name).into_iter().flatten().cloned());
        if self.nodes[name].kind.read() == Read::Reference {
            excluded.extend(
                self.nodes
                    .keys()
                    .filter(|node| self.is_inner_projection(node))
                    .cloned(),
            );
        }
        excluded
    }

    /// What the class reads as in the value read leaving out `excluded`: none
    /// where it reads as no reference or as nothing, and a reference, of its
    /// mutability where the value read decides one (l.267, l.307). Where the
    /// value read decides no constructor, a use's record that a class it
    /// reaches by lower bounds is a reference still reads as a reference
    /// (l.252).
    fn reads_as_reference(
        &self,
        id: ClassId,
        excluded: &BTreeSet<Name>,
    ) -> Option<Option<Mutability>> {
        let read = self.read_state(excluded);
        match read.read_value(id).decided() {
            Some(Term::Ref(mutability, _)) => Some(Some(*mutability)),
            Some(_) => None,
            None => read.recorded_as_reference(id).then_some(None),
        }
    }

    /// Whether a use recorded that the class, or one it reaches by lower
    /// bounds, is a reference (l.252).
    fn recorded_as_reference(&self, id: ClassId) -> bool {
        let mut seen = BTreeSet::from([self.root(id)]);
        let mut work = vec![self.root(id)];
        while let Some(at) = work.pop() {
            if self.classes.held(at).reference == Some(true) {
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
    /// last Deref, which matches a reference itself (l.317, patterns §1), and
    /// the least of their mutabilities where decided. A projection not yet
    /// decided has passed none.
    fn passed(&self, path: &[Name]) -> Option<Option<Mutability>> {
        let after_deref = path
            .iter()
            .rposition(|step| !self.is_inner_projection(step))
            .map_or(0, |deref| deref + 1);
        let passed: Vec<Option<Mutability>> = path[after_deref..]
            .iter()
            .filter_map(|step| match self.nodes[step].decision {
                Some(Decision::AppliedToTarget(mutability)) => Some(mutability),
                _ => None,
            })
            .collect();
        if passed.is_empty() {
            return None;
        }
        Some(passed.into_iter().fold(Some(Mutability::Exclusive), lesser))
    }

    /// Decides as it is recorded a node that reads nothing: a Deref, and a use
    /// with one candidate or none (`design/08-node.md`, readings 1 and 6).
    pub(crate) fn decide_at_once(&mut self) {
        let at_once: Vec<(Name, Decision)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter_map(|(name, node)| match &node.kind {
                NodeKind::Project {
                    projection: Projection::Deref,
                    ..
                } => Some((name.clone(), Decision::Applied)),
                NodeKind::Use { callee, args, .. } => {
                    match self.candidates(callee, args.len()).as_slice() {
                        [] => Some((name.clone(), Decision::Unreached(Vec::new()))),
                        [one] => Some((name.clone(), Decision::Use(one.id.clone()))),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect();
        for (name, decision) in at_once {
            self.set_decision(&name, decision);
            self.record_decision(&name);
        }
    }

    fn set_decision(&mut self, name: &Name, decision: Decision) {
        self.nodes
            .get_mut(name)
            .expect("the node is the state's")
            .decision = Some(decision);
    }

    /// Decides the node by the value it reads, and records what it records
    /// as the reading of l.269 takes it.
    pub(crate) fn decide(&mut self, name: &Name) {
        let decision = self.decision_of(name);
        let waiting = self.broken_on(name);
        self.set_decision(name, decision);
        self.broken.remove(name);
        self.defer(name, waiting);
    }

    /// Records what the node just decided records, or defers it, as the
    /// reading takes l.269 (`Deferral`; `holes.md`, O2), `waiting` being the
    /// undecided nodes that broke their dependency on it; then records what
    /// each deferred node records once the nodes it waits for are decided.
    fn defer(&mut self, name: &Name, waiting: BTreeSet<Name>) {
        match self.readings.deferral {
            Deferral::Recording if !waiting.is_empty() => {
                self.deferred.insert(name.clone(), waiting);
            }
            Deferral::Recording => self.record_decision(name),
            Deferral::Reading => {
                for node in waiting {
                    self.broken.entry(node).or_default().insert(name.clone());
                }
                self.record_decision(name);
            }
        }
        loop {
            let ready: Vec<Name> = self
                .deferred
                .iter()
                .filter(|(_, waiting)| {
                    waiting
                        .iter()
                        .all(|node| self.nodes[node].decision.is_some())
                })
                .map(|(name, _)| name.clone())
                .collect();
            if ready.is_empty() {
                return;
            }
            for name in ready {
                self.deferred.remove(&name);
                self.record_decision(&name);
            }
        }
    }

    fn decision_of(&self, name: &Name) -> Decision {
        let kind = &self.nodes[name].kind;
        let excluded = self.excluded(name);
        let lend = |mutability: Option<Mutability>, place: ClassId| {
            let (target, through) = match self.reads_as_reference(place, &excluded) {
                Some(through) => (LendTarget::Through(through), through),
                None => (LendTarget::Place, Some(Mutability::Exclusive)),
            };
            lesser(mutability, through).map(|mutability| (target, mutability))
        };
        match kind {
            NodeKind::Use {
                callee,
                args,
                result,
            } => self.narrow(callee, args, *result, &excluded),
            NodeKind::Lend {
                mutability, place, ..
            } => match lend(Some(*mutability), *place) {
                Some((target, mutability)) => Decision::Lend { target, mutability },
                None => Decision::NoResult,
            },
            NodeKind::Project {
                projection: Projection::Deref,
                ..
            } => Decision::Applied,
            NodeKind::Project { operand, .. } => {
                match self.reads_as_reference(*operand, &excluded) {
                    Some(mutability) => Decision::AppliedToTarget(mutability),
                    None => Decision::Applied,
                }
            }
            NodeKind::Bind { path, position, .. } => match self.passed(path) {
                None => Decision::Read,
                Some(passed) => match lend(passed, *position) {
                    Some((target, mutability)) => Decision::BindLend { target, mutability },
                    None => Decision::NoResult,
                },
            },
            NodeKind::CallValue { callee, .. } => {
                let values = self.read_state(&excluded).values_of(*callee);
                if values.is_empty() {
                    Decision::NoResult
                } else {
                    Decision::Calls(values)
                }
            }
        }
    }

    /// Records what the decided node records.
    fn record_decision(&mut self, name: &Name) {
        let node = self.nodes[name].clone();
        let Some(decision) = node.decision else {
            return;
        };
        let recorder = Recorder::Node(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        match (&node.kind, &decision) {
            (NodeKind::Use { args, result, .. }, Decision::Use(id)) => {
                let sig = self.decls.decl(id).sig.clone();
                let mut vars = BTreeMap::new();
                let var_name = |var| at(NodeAt::Var(var));
                for (i, (arg, param)) in args.iter().zip(sig.params()).enumerate() {
                    if !node.shared.contains(&i) {
                        self.record_argument(
                            *arg,
                            param,
                            &mut vars,
                            at(NodeAt::Param(i)),
                            &var_name,
                            &recorder,
                        );
                    }
                }
                if !node.shared.contains(&args.len()) {
                    self.record_result(
                        *result,
                        sig.result(),
                        &mut vars,
                        at(NodeAt::Result),
                        &var_name,
                        &recorder,
                    );
                }
            }
            (NodeKind::Use { result, .. }, Decision::Never) => {
                let never = self.written(at(NodeAt::Result), Term::Never, &recorder);
                self.record(
                    Relation::LowerBound {
                        lower: never,
                        upper: *result,
                    },
                    recorder,
                );
            }
            (NodeKind::Lend { place, result, .. }, Decision::Lend { target, mutability }) => {
                let lent = self.lend(*mutability, *place, *target, &at, &recorder);
                self.record(Relation::Equation(*result, lent), recorder);
            }
            (
                NodeKind::Project {
                    projection,
                    operand,
                    result,
                },
                Decision::Applied,
            ) => {
                self.record(Relation::Fact(*operand, projection.fact(*result)), recorder);
            }
            (
                NodeKind::Project {
                    projection,
                    operand,
                    result,
                },
                Decision::AppliedToTarget(_),
            ) => {
                let target = self.arena.intern(at(NodeAt::Target));
                self.classes.grow(&self.arena);
                self.record(
                    Relation::Fact(*operand, Fact::Ref(target)),
                    recorder.clone(),
                );
                self.record(Relation::Fact(target, projection.fact(*result)), recorder);
            }
            (
                NodeKind::Bind {
                    position, binding, ..
                },
                Decision::Read,
            ) => {
                self.record(
                    Relation::LowerBound {
                        lower: *position,
                        upper: *binding,
                    },
                    recorder,
                );
            }
            (NodeKind::CallValue { args, result, .. }, Decision::Calls(lambdas)) => {
                for lambda in lambdas {
                    let function = self
                        .functions
                        .iter()
                        .find(|function| function.name == *lambda)
                        .expect("a lambda is a function")
                        .clone();
                    let from = Rc::clone(
                        self.solved
                            .get(&function.group)
                            .expect("a lambda's group is solved first (l.220)"),
                    );
                    let at = Name::new(&format!("{name}#{lambda}"));
                    let mut copy = crate::instance::Instance::new(&at, recorder.clone(), &from);
                    let params: Vec<ClassId> = function
                        .params
                        .iter()
                        .map(|param| copy.class(self, from.text(param)))
                        .collect();
                    let returned = copy.class(self, from.text(&function.result));
                    copy.record_rest(self);
                    self.record_call(args, &params, returned, *result, &recorder);
                }
            }
            (
                NodeKind::Bind {
                    position, binding, ..
                },
                Decision::BindLend { target, mutability },
            ) => {
                let lent = self.lend(*mutability, *position, *target, &at, &recorder);
                self.record(
                    Relation::LowerBound {
                        lower: lent,
                        upper: *binding,
                    },
                    recorder,
                );
            }
            _ => {}
        }
    }

    /// An argument passing into its parameter: a meeting where the parameter
    /// is an unknown, a boundary where a declaration writes it (reading 3).
    fn record_argument<F>(
        &mut self,
        arg: ClassId,
        param: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let param = self.instantiate(param, vars, name, var_name, recorder);
        let relation = if self.is_unknown(param) {
            Relation::LowerBound {
                lower: arg,
                upper: param,
            }
        } else {
            Relation::Boundary {
                from: arg,
                to: param,
            }
        };
        self.record(relation, recorder.clone());
    }

    /// The signature's result type entering the use's result (note 07,
    /// reading 2).
    fn record_result<F>(
        &mut self,
        result: ClassId,
        ty: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let returned = self.instantiate(ty, vars, name, var_name, recorder);
        self.record(
            Relation::LowerBound {
                lower: returned,
                upper: result,
            },
            recorder.clone(),
        );
    }

    /// Records, for each undecided use with several candidates, what all its
    /// candidates share: each position their signatures write alike with no
    /// variable (l.335), and whether each class they would record a lower
    /// bound or an equation into is a reference, where they agree (l.252).
    pub(crate) fn record_at_once(&mut self) {
        let uses: Vec<Name> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter(|(_, node)| matches!(node.kind, NodeKind::Use { .. }))
            .map(|(name, _)| name.clone())
            .collect();
        for name in uses {
            self.record_shared(&name);
            self.record_references(&name);
        }
    }

    fn record_shared(&mut self, name: &Name) {
        let node = self.nodes[name].clone();
        let NodeKind::Use {
            callee,
            args,
            result,
        } = &node.kind
        else {
            return;
        };
        let sigs: Vec<Signature> = self
            .candidates(callee, args.len())
            .into_iter()
            .map(|decl| decl.sig.clone())
            .collect();
        let Some(first) = sigs.first() else {
            return;
        };
        let differing = Self::differing(&sigs.iter().collect::<Vec<_>>(), args.len());
        let recorder = Recorder::Node(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        let var_name = |var| at(NodeAt::Var(var));
        for i in 0..=args.len() {
            let ty = Self::at_position(first, i).clone();
            if differing.contains(&i) || node.shared.contains(&i) || !ty.is_closed() {
                continue;
            }
            let mut vars = BTreeMap::new();
            match args.get(i) {
                Some(arg) => self.record_argument(
                    *arg,
                    &ty,
                    &mut vars,
                    at(NodeAt::Param(i)),
                    &var_name,
                    &recorder,
                ),
                None => self.record_result(
                    *result,
                    &ty,
                    &mut vars,
                    at(NodeAt::Result),
                    &var_name,
                    &recorder,
                ),
            }
            self.nodes
                .get_mut(name)
                .expect("the node is the state's")
                .shared
                .insert(i);
        }
    }

    fn record_references(&mut self, name: &Name) {
        let per_candidate = self.would_record_per_candidate(name);
        let Some((first, rest)) = per_candidate.split_first() else {
            return;
        };
        let agreed: Vec<(ClassId, bool)> = first
            .iter()
            .filter_map(|(class, reference)| Some((*class, (*reference)?)))
            .filter(|(class, reference)| {
                rest.iter()
                    .all(|other| other.get(class) == Some(&Some(*reference)))
            })
            .filter(|(class, _)| !self.recorded_reference(name, *class))
            .collect();
        for (class, reference) in agreed {
            self.nodes
                .get_mut(name)
                .expect("the node is the state's")
                .references
                .insert(class);
            self.record(
                Relation::Reference(class, reference),
                Recorder::Node(name.clone()),
            );
        }
    }

    /// For each candidate of a use, the classes it would record a lower bound
    /// or an equation into, each with whether that type is a reference, none
    /// where it is an unknown: its result, and the classes inside its
    /// arguments' kinds that its parameters reach.
    fn would_record_per_candidate(&self, name: &Name) -> Vec<BTreeMap<ClassId, Option<bool>>> {
        let NodeKind::Use {
            callee,
            args,
            result,
        } = &self.nodes[name].kind
        else {
            return Vec::new();
        };
        self.candidates(callee, args.len())
            .into_iter()
            .map(|decl| {
                let mut found = BTreeMap::new();
                for (arg, param) in args.iter().zip(decl.sig.params()) {
                    self.would_record(*arg, param, false, &mut found);
                }
                found.insert(self.root(*result), is_reference(decl.sig.result()));
                found
            })
            .collect()
    }

    /// The classes inside an argument's kind that a candidate's parameter
    /// would record a lower bound or an equation into, by the table of l.286:
    /// the parts an invariant part equates, and every part beneath them. A
    /// part the parameter writes as a variable is a lower bound of that
    /// unknown, recording into it only, unless an invariant part equates it.
    fn would_record(
        &self,
        class: ClassId,
        ty: &SigTy,
        equated: bool,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        if equated {
            found.insert(self.root(class), is_reference(ty));
        }
        let SigTy::Term(term) = ty else {
            return;
        };
        let Some((kind, _)) = &self.classes.held(class).kind else {
            return;
        };
        if kind.constructor() != term.constructor() {
            return;
        }
        let invariant = kind.constructor().variance() == Variance::Invariant;
        for (key, written) in term.parts() {
            if let Some(&part) = kind.part(&key) {
                self.would_record(part, written, equated || invariant, found);
            }
        }
    }

    /// The reference a lend gives, of `mutability`, to `place` or to its
    /// target (l.307).
    fn lend<F>(
        &mut self,
        mutability: Mutability,
        place: ClassId,
        target: LendTarget,
        at: &F,
        recorder: &Recorder,
    ) -> ClassId
    where
        F: Fn(NodeAt) -> ClassName,
    {
        let target = match target {
            LendTarget::Place => place,
            LendTarget::Through(_) => {
                let target = self.arena.intern(at(NodeAt::Target));
                self.classes.grow(&self.arena);
                self.record(Relation::Fact(place, Fact::Ref(target)), recorder.clone());
                target
            }
        };
        self.written(at(NodeAt::Lent), Term::Ref(mutability, target), recorder)
    }

    /// Narrows the candidates by the value the use reads at each position
    /// where they differ, then by the result (l.337–344).
    fn narrow(
        &self,
        callee: &Name,
        args: &[ClassId],
        result: ClassId,
        excluded: &BTreeSet<Name>,
    ) -> Decision {
        let candidates = self.candidates(callee, args.len());
        let ids: Vec<DeclId> = candidates.iter().map(|decl| decl.id.clone()).collect();
        let sigs: Vec<&Signature> = candidates.iter().map(|decl| &decl.sig).collect();
        let differing = Self::differing(&sigs, args.len());
        let read = self.read_state(excluded);
        let values: Vec<(usize, ReadValue)> = differing
            .iter()
            .map(|&i| (i, read.read_value(args.get(i).copied().unwrap_or(result))))
            .collect();
        let never = values
            .iter()
            .any(|(i, value)| *i < args.len() && matches!(value.decided(), Some(Term::Never)));
        if never {
            return Decision::Never;
        }
        let narrowed = |values: &[(usize, ReadValue)]| {
            let mut remaining: Vec<usize> = (0..sigs.len()).collect();
            for (i, value) in values {
                let ranks: Vec<(usize, Option<crate::coerce::Rank>)> = remaining
                    .iter()
                    .map(|&c| {
                        let written = ReadValue::of_sig(Self::at_position(sigs[c], *i));
                        let rank = if *i < args.len() {
                            coerce_read(&self.decls, value, &written)
                        } else {
                            coerce_read(&self.decls, &written, value)
                        };
                        (c, rank.ok().and_then(|coercion| coercion.rank()))
                    })
                    .collect();
                let lowest = ranks.iter().filter_map(|(_, rank)| rank.clone()).min();
                remaining = ranks
                    .into_iter()
                    .filter(|(_, rank)| rank.is_some() && *rank == lowest)
                    .map(|(c, _)| c)
                    .collect();
            }
            remaining
        };
        let mut remaining = narrowed(&values);
        if remaining.len() > 1 && values.iter().any(|(_, value)| value.has_lit()) {
            let defaulted: Vec<(usize, ReadValue)> = values
                .iter()
                .map(|(i, value)| (*i, value.with_defaults()))
                .collect();
            remaining = narrowed(&defaulted);
        }
        match remaining.as_slice() {
            [one] => Decision::Use(ids[*one].clone()),
            _ => Decision::Unreached(remaining.into_iter().map(|c| ids[c].clone()).collect()),
        }
    }

    /// A class made anew that holds `term` as a declaration writes it.
    pub(crate) fn written(
        &mut self,
        name: ClassName,
        term: Term<ClassId>,
        recorder: &Recorder,
    ) -> ClassId {
        let class = self.arena.intern(name);
        self.classes.grow(&self.arena);
        self.decided.insert(class);
        self.record(
            Relation::Kind(class, term, KindSource::Written),
            recorder.clone(),
        );
        class
    }

    /// The classes of a type a declaration writes: the type named `name`, its
    /// parts named after it, and its variables those of `vars`, each made
    /// where first met and named by `var_name`. A variable is an unknown, and
    /// a written term is decided (l.218).
    pub(crate) fn instantiate<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) -> ClassId
    where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        match ty {
            SigTy::Var(var) => {
                if let Some(&class) = vars.get(var) {
                    return class;
                }
                let class = self.arena.intern(var_name(*var));
                self.classes.grow(&self.arena);
                vars.insert(*var, class);
                class
            }
            SigTy::Term(term) => {
                let shared = Rc::new(name.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    let part_name = ClassName::Part(shared.clone(), key);
                    self.instantiate(&part, vars, part_name, var_name, recorder)
                });
                self.written(name, term, recorder)
            }
        }
    }
}
