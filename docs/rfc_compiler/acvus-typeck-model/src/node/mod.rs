//! Uses, lends, projections and bindings: the positions each lies at, what
//! each reads, and what each records once decided (l.292–334). Here, what each
//! is, and how it is decided and records once decided (l.261, l.263); the
//! positions and what each reads are `position`, places `place`, overload
//! resolution `overload`, and the common part of all an undecided node may
//! become `common`.

pub(crate) mod common;
mod overload;
mod place;
mod position;

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Fact, KindSource};
use crate::decl::DeclId;
use crate::program::{Position, Projection};
use crate::read::ReadValue;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::{State, left_out_by};
use crate::ty::{Mutability, Name, Term};

pub use overload::{Narrowed, Narrowing};
pub use place::AsReference;
use place::lesser;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NodeKind {
    /// A use that calls the declarations `callee` names (l.320); for `e?`,
    /// with the class its exit's value passes into (l.337). It holds the
    /// positions of the use's constraints its group records where the choice
    /// of a candidate matters, none at the others: those another group
    /// records, and, of several candidates, those where every candidate
    /// writes one closed type, recorded apart (l.186, l.246).
    Use {
        callee: Name,
        args: Vec<Option<ClassId>>,
        result: Option<ClassId>,
        exit: Option<ClassId>,
    },
    /// `&ₘ p` (l.296).
    Lend {
        mutability: Mutability,
        place: ClassId,
        result: ClassId,
    },
    /// A projection of `operand`, naming `result` (l.144).
    Project {
        projection: Projection,
        operand: ClassId,
        result: ClassId,
    },
    /// A binding at `position`, reached by the projections `path` (l.306).
    Bind {
        path: Vec<Name>,
        position: ClassId,
        binding: ClassId,
    },
    /// A use that calls the function value of `callee` (l.196).
    CallValue {
        callee: ClassId,
        args: Vec<ClassId>,
        result: ClassId,
    },
}

/// What a node reads at the classes at its positions (l.261).
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

/// What a node reads (l.261), on which it depends (l.241–242).
pub(crate) struct Reads {
    pub(crate) read: Read,
    /// The classes at the positions it reads.
    pub(crate) classes: Vec<ClassId>,
}

/// What a use, a lend, a projection or a binding became (l.261).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decision {
    /// The use is decided as this candidate (l.333).
    Use(DeclId),
    /// The use's argument reads as `!`, so it is never carried out and is
    /// decided with the result `!` (l.333).
    Never,
    /// The lend gives a reference of `mutability` to `target` (l.296).
    Lend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The projection applies to its operand (l.144).
    Applied,
    /// The projection's operand is a reference, so it applies to that
    /// reference's target (auto-deref, l.298); the mutability is the
    /// reference's where the value read decides it.
    AppliedToTarget(Option<Mutability>),
    /// The binding's path passes no reference, so it reads its position
    /// (l.306).
    Read,
    /// The binding's path passes a reference, so it lends its position, giving
    /// a reference of `mutability` to `target` (l.306).
    BindLend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The call reads the function value as one whose Values names these
    /// lambdas, and makes an instance of each (l.196).
    Calls(BTreeSet<Name>),
}

/// What a lend's place reads as (l.296).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LendTarget {
    /// Not a reference, so the target is the place.
    Place,
    /// A reference, so the target is that reference's; the mutability is the
    /// reference's where the value read decides it.
    Through(Option<Mutability>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NodeState {
    pub kind: NodeKind,
    pub decision: Option<Decision>,
    /// The common part the node recorded before it was decided (l.246).
    pub common: Option<Common>,
    /// The classes of which the common part recorded whether they are
    /// references (l.246).
    pub references: BTreeSet<ClassId>,
}

/// What a node recorded before it was decided: the common part of all it may
/// become (l.246).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Common {
    /// A use's: the common part of its candidates' signatures, and the class
    /// made for each of its variables.
    Use {
        part: Rc<common::CommonPart>,
        classes: Vec<ClassId>,
    },
    /// A lend's: its result is a reference.
    Lend,
}

/// Which class a node made anew is (l.320).
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
    /// The value of `e?`'s exit (l.337).
    Exit,
    /// A variable of a use's common part (l.246).
    Common(usize),
    /// The type of the parameter at this place in a use's common part.
    CommonParam(usize),
    /// The result type in a use's common part.
    CommonResult,
    /// The type the candidate a use is decided as writes at a variable of its
    /// common part.
    Chosen(usize),
}

/// A node decided (l.261), with the relations recorded once it is: its own,
/// and those of the nodes deferred on it that it lets record (l.263). A node
/// decided by the value it read has what it read; one decided as it is
/// recorded reads nothing.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decided {
    pub node: Name,
    pub decision: Decision,
    pub recorded: Recording,
    pub read: Option<DecidedBy>,
}

/// How a node was decided by the value it read (l.261, l.263).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DecidedBy {
    /// The node and the nodes that transitively depend on it (l.261).
    pub dependents: BTreeSet<Name>,
    /// The undecided nodes that broke their dependency on it (l.263).
    pub broken_on: BTreeSet<Name>,
    pub value: ValueRead,
}

/// The value a node read, and what it became by it (l.261).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValueRead {
    /// The nodes whose relations it leaves out: those it was given and, for
    /// whether a class is a reference, every projection other than a Deref.
    pub left_out: BTreeSet<Name>,
    /// Their relations, by their places in the log.
    pub relations: Vec<usize>,
    pub reading: Reading,
    /// What the node becomes by it, none where it decides nothing (l.261,
    /// l.333).
    pub decision: Option<Decision>,
}

/// What a node read at its positions (l.261).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Reading {
    /// A Deref, and a binding whose path passes no reference, read nothing.
    Nothing,
    /// A use: the value read where its candidates differ, and its narrowing
    /// (l.326–333).
    Use(Narrowing),
    /// A lend's place or a projection's operand, and what it reads as
    /// (l.296, l.298).
    Place(AsReference),
    /// A binding whose path passes a reference, the least mutability of
    /// those passed where decided, and what its position reads as (l.306).
    Binding {
        passed: Option<Mutability>,
        position: AsReference,
    },
    /// A call of a function value: the value read at its callee, and the
    /// lambdas its Values names there (l.196).
    Values {
        callee: ClassId,
        value: ReadValue,
        lambdas: BTreeSet<Name>,
    },
}

impl State {
    /// Decides as it is recorded a node that reads nothing: a Deref
    /// (`design/08-node.md`, reading 6). A use of one candidate has no node
    /// (`State::record_one`), and a use of none is never decided (l.333).
    pub(crate) fn decide_at_once(&mut self) -> Vec<Decided> {
        let at_once: Vec<(Name, Decision)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .filter_map(|(name, node)| match &node.kind {
                NodeKind::Project {
                    projection: Projection::Deref,
                    ..
                } => Some((name.clone(), Decision::Applied)),
                _ => None,
            })
            .collect();
        let mut decided = Vec::new();
        for (name, decision) in at_once {
            self.set_decision(&name, decision.clone());
            let recorded = self.record_decision(&name);
            decided.push(Decided {
                node: name,
                decision,
                recorded,
                read: None,
            });
        }
        decided
    }

    pub(crate) fn set_decision(&mut self, name: &Name, decision: Decision) {
        self.nodes
            .get_mut(name)
            .expect("the node is the state's")
            .decision = Some(decision);
    }

    /// The value the node reads now, as deciding it would read it: without
    /// itself and those that transitively depend on it (l.261).
    pub(crate) fn read_now(&self, name: &Name) -> ValueRead {
        self.value_read(name, &self.dependents(name))
    }

    /// Decides the node by the value it reads, which decides it (`steps`),
    /// and records what it records, or defers it (l.263).
    pub(crate) fn decide(&mut self, name: &Name) -> Decided {
        let dependents = self.dependents(name);
        let value = self.value_read(name, &dependents);
        let decision = value
            .decision
            .clone()
            .expect("a node is decided where the value it reads decides it");
        let broken_on = self.broken_on(name);
        self.set_decision(name, decision.clone());
        let recorded = self.defer(name, broken_on.clone());
        Decided {
            node: name.clone(),
            decision,
            recorded,
            read: Some(DecidedBy {
                dependents,
                broken_on,
                value,
            }),
        }
    }

    /// Records what the node just decided records, or defers it until the
    /// undecided nodes that broke their dependency on it, `waiting`, are
    /// decided (l.263); then records what each deferred node records once the
    /// nodes it waits for are decided. Returns the relations recorded.
    pub(crate) fn defer(&mut self, name: &Name, waiting: BTreeSet<Name>) -> Recording {
        let mut recorded = Recording::default();
        if waiting.is_empty() {
            recorded.append(self.record_decision(name));
        } else {
            self.deferred.insert(name.clone(), waiting);
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
                return recorded;
            }
            for name in ready {
                self.deferred.remove(&name);
                recorded.append(self.record_decision(&name));
            }
        }
    }

    /// What the node becomes by the value it reads leaving out the relations
    /// of the nodes `excluded`, as `value_read` reads it; none where that
    /// value decides nothing.
    pub(crate) fn decision_under(
        &self,
        name: &Name,
        excluded: &BTreeSet<Name>,
    ) -> Option<Decision> {
        self.value_read(name, excluded).decision
    }

    /// The value the node reads leaving out the relations of the nodes
    /// `excluded`, and, for a lend, a projection or a binding, which reads
    /// whether a class is a reference, of every projection other than a Deref
    /// whose operand is a class it reads (l.261), and what the node becomes by
    /// it, none where it decides nothing (l.261, l.333).
    pub(crate) fn value_read(&self, name: &Name, excluded: &BTreeSet<Name>) -> ValueRead {
        let kind = &self.nodes[name].kind;
        let mut excluded = excluded.clone();
        if kind.read() == Read::Reference {
            let read = self.read_positions(kind);
            excluded.extend(
                self.nodes
                    .keys()
                    .filter(|node| read.iter().any(|&class| self.projects_from(node, class)))
                    .cloned(),
            );
        }
        let lend = |mutability: Option<Mutability>, place: ClassId| {
            let read = self.reads_as_reference(place, &excluded);
            let (target, through) = match read.reads_as {
                Some(through) => (LendTarget::Through(through), through),
                None => (LendTarget::Place, Some(Mutability::Exclusive)),
            };
            let lent = lesser(mutability, through).map(|mutability| (target, mutability));
            (read, lent)
        };
        let (reading, decision) = match kind {
            NodeKind::Use {
                callee,
                args,
                result,
                ..
            } => {
                let narrowing = self.narrow(name, callee, args, *result, &excluded);
                let decision = narrowing.decision.clone();
                (Reading::Use(narrowing), decision)
            }
            NodeKind::Lend {
                mutability, place, ..
            } => {
                let (read, lent) = lend(Some(*mutability), *place);
                let decision =
                    lent.map(|(target, mutability)| Decision::Lend { target, mutability });
                (Reading::Place(read), decision)
            }
            NodeKind::Project {
                projection: Projection::Deref,
                ..
            } => (Reading::Nothing, Some(Decision::Applied)),
            NodeKind::Project { operand, .. } => {
                let read = self.reads_as_reference(*operand, &excluded);
                let decision = match read.reads_as {
                    Some(mutability) => Decision::AppliedToTarget(mutability),
                    None => Decision::Applied,
                };
                (Reading::Place(read), Some(decision))
            }
            NodeKind::Bind { path, position, .. } => match self.passed(path) {
                None => (Reading::Nothing, Some(Decision::Read)),
                Some(passed) => {
                    let (read, lent) = lend(passed, *position);
                    let decision =
                        lent.map(|(target, mutability)| Decision::BindLend { target, mutability });
                    let reading = Reading::Binding {
                        passed,
                        position: read,
                    };
                    (reading, decision)
                }
            },
            NodeKind::CallValue { callee, .. } => {
                let read = self.read_state(&excluded);
                let lambdas = read.values_of(*callee);
                let decision = (!lambdas.is_empty()).then(|| Decision::Calls(lambdas.clone()));
                let reading = Reading::Values {
                    callee: *callee,
                    value: read.read_value(*callee),
                    lambdas,
                };
                (reading, decision)
            }
        };
        let relations = self
            .log
            .iter()
            .enumerate()
            .filter(|(_, recorded)| left_out_by(&recorded.recorder, &excluded))
            .map(|(at, _)| at)
            .collect();
        ValueRead {
            left_out: excluded,
            relations,
            reading,
            decision,
        }
    }

    /// An instance of the lambda's C made at `at` by `by` (l.215), each value
    /// of `args` passing into the copy of the lambda's parameter in its generic
    /// part, or into its parameter in the environment, and the copy of its
    /// result type in its generic part, or the environment's, passing into
    /// `result` (l.196, l.197).
    pub(crate) fn instantiate_lambda(
        &mut self,
        at: &Name,
        by: &Recorder,
        lambda: &Name,
        args: &[ClassId],
        result: ClassId,
    ) -> Recording {
        let function = self.lambda_function(lambda).clone();
        let from = Rc::clone(
            self.solved
                .get(&function.group)
                .expect("a lambda's group is solved first (l.213)"),
        );
        let mut copy = crate::instance::Instance::new(at, by.clone(), &from);
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
        let mut recorded = copy.record_rest(self);
        let pairs: Vec<(ClassId, ClassId)> = args
            .iter()
            .copied()
            .zip(met)
            .chain([(returned, result)])
            .collect();
        recorded.append(self.record_call(&pairs, by));
        recorded
    }

    /// `e?`'s exit, decided as the candidate of `sig` whose variables are the
    /// classes of `vars` (l.337): as `return None`, a value `Option⟨K⟩`, or as
    /// `return Err(x)`, a value `Result⟨O, X⟩`, `K` and `O` made anew and the
    /// candidate's `E` wrapped into `X`. The value passes into `exit`, the
    /// class the text gives it, a meeting where that is an unknown (l.219).
    fn record_exit(
        &mut self,
        sig: &crate::sig::Signature,
        vars: &BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        exit: ClassId,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        let named = Rc::new(name.clone());
        let part = |key: crate::ty::PartKey| ClassName::Part(named.clone(), key);
        let some = crate::ty::PartKey::Payload(Name::new(crate::ty::SOME));
        let ok = crate::ty::PartKey::Payload(Name::new(crate::ty::OK));
        let err = crate::ty::PartKey::Payload(Name::new(crate::ty::ERR));
        let term = match sig.params() {
            [crate::sig::SigTy::Term(param)] if matches!(**param, Term::Option(_)) => {
                Term::Option(self.fresh(part(some)))
            }
            [crate::sig::SigTy::Term(param)] if matches!(**param, Term::Result { .. }) => {
                let error = *vars
                    .get(&crate::sig::SigVar::Type(1))
                    .expect("the argument made the candidate's `E`");
                let wrapped = self.fresh(part(err));
                let relation = Relation::Wrap {
                    from: error,
                    to: wrapped,
                };
                recorded.push(self.record(relation, recorder.clone()));
                Term::Result {
                    ok: self.fresh(part(ok)),
                    err: wrapped,
                }
            }
            _ => unreachable!("`?`'s candidates are the language's two (l.337)"),
        };
        let value = self.written(name, term, recorder, &mut recorded);
        let relation = if self.is_unknown(exit) {
            Relation::LowerBound {
                lower: value,
                upper: exit,
            }
        } else {
            Relation::Boundary {
                from: value,
                to: exit,
            }
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// Records the use `name` of one candidate at the positions `held`, as it
    /// is recorded: its common part is that candidate, its type parameters
    /// made anew (l.246, l.320), so nothing is left to decide, and it has no
    /// node. Each argument passes into its parameter's type and the result
    /// type into the result, an identity its result carries and no parameter
    /// does made anew (l.347), recorded by the common part.
    pub(crate) fn record_one(
        &mut self,
        name: &Name,
        callee: &Name,
        arity: usize,
        held: &[(Position, ClassId)],
    ) -> Recording {
        let mut recorded = Recording::default();
        let sig = match self.candidates(callee, arity).as_slice() {
            [candidate] => candidate.sig.clone(),
            _ => panic!("{name}: a use of one candidate"),
        };
        let recorder = Recorder::Common(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        let mut vars = BTreeMap::new();
        let var_name = |var| at(NodeAt::Var(var));
        for &(position, class) in held {
            recorded.append(match position {
                Position::Arg(k) => self.record_argument(
                    class,
                    &sig.params()[k],
                    &mut vars,
                    at(NodeAt::Param(k)),
                    &var_name,
                    &recorder,
                ),
                Position::Result => self.record_result(
                    class,
                    sig.result(),
                    &mut vars,
                    at(NodeAt::Result),
                    &var_name,
                    &recorder,
                ),
                Position::Exit => unreachable!("`?`'s candidates are the language's two (l.337)"),
            });
        }
        if sig.existential_identity()
            && let Some(&identity) = vars.get(&crate::sig::SigVar::Identity)
        {
            let made = Term::Identity(Name::new(&format!("new {name}")));
            let relation = Relation::Kind(identity, made, KindSource::Written);
            recorded.push(self.record(relation, recorder));
        }
        recorded
    }

    /// Records what the decided node records.
    pub(crate) fn record_decision(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let node = self.nodes[name].clone();
        let Some(decision) = node.decision else {
            return recorded;
        };
        let recorder = Recorder::Node(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: NodeAt| ClassName::Node(owner.clone(), at);
        match (&node.kind, &decision) {
            (
                NodeKind::Use {
                    callee,
                    args,
                    result,
                    exit,
                },
                Decision::Use(id),
            ) => {
                // What the common part holds is not recorded again (l.246):
                // an argument the common part's parameter holds whole, and
                // the result but at the common part's variables, into each
                // of which the candidate's type there enters. An argument it
                // does not hold whole passes whole into the candidate's
                // parameter, a coercion being of a whole value (l.109).
                let sig = self.decls.decl(id).sig.clone();
                let common = match &node.common {
                    Some(Common::Use { part, classes }) => Some((part.clone(), classes.clone())),
                    _ => None,
                };
                // The node holds no position where the common part writes a
                // closed type, nor one another group records (l.186).
                let mut vars = BTreeMap::new();
                let var_name = |var| at(NodeAt::Var(var));
                for (i, (arg, param)) in args.iter().zip(sig.params()).enumerate() {
                    let Some(arg) = arg else {
                        continue;
                    };
                    recorded.append(self.record_argument(
                        *arg,
                        param,
                        &mut vars,
                        at(NodeAt::Param(i)),
                        &var_name,
                        &recorder,
                    ));
                }
                match &common {
                    None => {
                        if let Some(result) = result {
                            recorded.append(self.record_result(
                                *result,
                                sig.result(),
                                &mut vars,
                                at(NodeAt::Result),
                                &var_name,
                                &recorder,
                            ));
                        }
                    }
                    Some((part, classes)) => {
                        let candidate = self.candidate_index(callee, args.len(), id);
                        for k in common::CommonPart::vars_in(&part.result) {
                            recorded.append(self.record_result(
                                classes[k],
                                &part.vars[k][candidate],
                                &mut vars,
                                at(NodeAt::Chosen(k)),
                                &var_name,
                                &recorder,
                            ));
                        }
                    }
                }
                if sig.existential_identity()
                    && let Some(&identity) = vars.get(&crate::sig::SigVar::Identity)
                {
                    let made = Term::Identity(Name::new(&format!("new {name}")));
                    let relation = Relation::Kind(identity, made, KindSource::Written);
                    recorded.push(self.record(relation, recorder.clone()));
                }
                if let Some(exit) = exit {
                    recorded.append(self.record_exit(
                        &sig,
                        &vars,
                        at(NodeAt::Exit),
                        *exit,
                        &recorder,
                    ));
                }
            }
            // Where the result is a closed type of the common part, the node
            // does not hold it, and records nothing there.
            (
                NodeKind::Use {
                    result: Some(result),
                    ..
                },
                Decision::Never,
            ) => {
                let never = self.written(at(NodeAt::Result), Term::Never, &recorder, &mut recorded);
                let relation = Relation::LowerBound {
                    lower: never,
                    upper: *result,
                };
                recorded.push(self.record(relation, recorder));
            }
            (NodeKind::Lend { place, result, .. }, Decision::Lend { target, mutability }) => {
                let lent = self.lend(*mutability, *place, *target, &at, &recorder, &mut recorded);
                recorded.push(self.record(Relation::Equation(*result, lent), recorder));
            }
            (
                NodeKind::Project {
                    projection,
                    operand,
                    result,
                },
                Decision::Applied,
            ) => {
                let relation = Relation::Fact(*operand, projection.fact(*result));
                recorded.push(self.record(relation, recorder));
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
                let relation = Relation::Fact(*operand, Fact::Ref(target));
                recorded.push(self.record(relation, recorder.clone()));
                let relation = Relation::Fact(target, projection.fact(*result));
                recorded.push(self.record(relation, recorder));
            }
            (
                NodeKind::Bind {
                    position, binding, ..
                },
                Decision::Read,
            ) => {
                let relation = Relation::LowerBound {
                    lower: *position,
                    upper: *binding,
                };
                recorded.push(self.record(relation, recorder));
            }
            (NodeKind::CallValue { args, result, .. }, Decision::Calls(lambdas)) => {
                for lambda in lambdas {
                    let at = Name::new(&format!("{name}#{lambda}"));
                    recorded.append(self.instantiate_lambda(&at, &recorder, lambda, args, *result));
                }
            }
            (
                NodeKind::Bind {
                    position, binding, ..
                },
                Decision::BindLend { target, mutability },
            ) => {
                let lent = self.lend(
                    *mutability,
                    *position,
                    *target,
                    &at,
                    &recorder,
                    &mut recorded,
                );
                let relation = Relation::LowerBound {
                    lower: lent,
                    upper: *binding,
                };
                recorded.push(self.record(relation, recorder));
            }
            _ => {}
        }
        recorded
    }
}
