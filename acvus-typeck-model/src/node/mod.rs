//! Uses, lends, projections and bindings: the positions each lies at, what
//! each reads, and what each records once decided (l.303–345). Here, what each
//! is, and how it is decided and records once decided (l.267, l.269); the
//! positions and what each reads are `position`, places `place`, overload
//! resolution `overload`, and what an undecided use's candidates share
//! `shared`.

mod overload;
mod place;
mod position;
mod shared;

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Fact};
use crate::decl::DeclId;
use crate::program::{Deferral, Projection};
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Mutability, Name, Term};

use place::lesser;

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

impl State {
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
        self.decision_under(name, &self.excluded(name))
    }

    /// What the node becomes by the value it reads leaving out the relations
    /// of the nodes `excluded`, and, for a lend, a projection or a binding,
    /// which reads whether a class is a reference, of every projection other
    /// than a Deref (l.267).
    pub(crate) fn decision_under(&self, name: &Name, excluded: &BTreeSet<Name>) -> Decision {
        let kind = &self.nodes[name].kind;
        let mut excluded = excluded.clone();
        if kind.read() == Read::Reference {
            excluded.extend(
                self.nodes
                    .keys()
                    .filter(|node| self.is_inner_projection(node))
                    .cloned(),
            );
        }
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
                    let mut met = Vec::new();
                    let held = self.environment_of(lambda);
                    for (name, held) in function.params.iter().chain([&function.result]).zip(held) {
                        let class = match held {
                            Some(class) => class,
                            None => copy.class(self, from.text(name)),
                        };
                        met.push(class);
                    }
                    let returned = met.pop().expect("the result type follows the parameters");
                    let params = met;
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
}
