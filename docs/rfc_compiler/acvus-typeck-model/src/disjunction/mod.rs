//! Uses, lends, projections and bindings: the positions each lies at, what
//! each reads, and what each records once decided. Here, what each
//! is, and how it is decided and records once decided; the
//! positions and what each reads are `position`, places `place`, overload
//! resolution `overload`, and the generalization of all an undecided disjunctive constraint may
//! become `generalization`.

pub(crate) mod generalization;
mod overload;
mod place;
mod position;

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, StructureConstraint, StructureSource};
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
pub enum Disjunction {
    /// A use that calls the declarations `callee` names; for `e?`,
    /// with the class its exit's value passes into. It holds the
    /// positions of the use's constraints its group records where the choice
    /// of a candidate matters, none at the others: those another group
    /// records, and, of several candidates, those where every candidate
    /// writes one closed type, recorded apart.
    Use {
        callee: Name,
        args: Vec<Option<ClassId>>,
        result: Option<ClassId>,
        exit: Option<ClassId>,
    },
    /// `&ₘ p`.
    Lend {
        mutability: Mutability,
        place: ClassId,
        result: ClassId,
    },
    /// A projection of `operand`, naming `result`.
    Project {
        projection: Projection,
        operand: ClassId,
        result: ClassId,
    },
    /// A binding at `position`, reached by the projections `path`.
    Bind {
        path: Vec<Name>,
        position: ClassId,
        binding: ClassId,
    },
    /// A use that calls the function value of `callee`.
    CallValue {
        callee: ClassId,
        args: Vec<ClassId>,
        result: ClassId,
    },
}

/// What a disjunctive constraint reads at the classes at its positions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Read {
    /// Their values: a use, and a call of a function value.
    Value,
    /// Whether they are references: a lend, a projection, a binding.
    Reference,
}

impl Disjunction {
    pub(crate) fn read(&self) -> Read {
        match self {
            Disjunction::Lend { .. } | Disjunction::Project { .. } | Disjunction::Bind { .. } => {
                Read::Reference
            }
            Disjunction::Use { .. } | Disjunction::CallValue { .. } => Read::Value,
        }
    }
}

/// What a disjunctive constraint reads, on which it depends.
pub(crate) struct Reads {
    pub(crate) read: Read,
    /// The classes at the positions it reads.
    pub(crate) classes: Vec<ClassId>,
}

/// What a use, a lend, a projection or a binding became.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decision {
    /// The use is decided as this candidate.
    Use(DeclId),
    /// The use's argument reads as `!`, so it is never carried out and is
    /// decided with the result `!`.
    Never,
    /// The lend gives a reference of `mutability` to `target`.
    Lend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The projection applies to its operand.
    Applied,
    /// The projection's operand is a reference, so it applies to that
    /// reference's target (auto-deref); the mutability is the
    /// reference's where the value read decides it.
    AppliedToTarget(Option<Mutability>),
    /// The binding's path passes no reference, so it reads its position.
    Read,
    /// The binding's path passes a reference, so it lends its position, giving
    /// a reference of `mutability` to `target`.
    BindLend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The call reads the function value as one whose Values names these
    /// lambdas, and makes an instance of each.
    Calls(BTreeSet<Name>),
}

/// What a lend's place reads as.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LendTarget {
    /// Not a reference, so the target is the place.
    Place,
    /// A reference, so the target is that reference's; the mutability is the
    /// reference's where the value read decides it.
    Through(Option<Mutability>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DisjunctionState {
    pub kind: Disjunction,
    pub decision: Option<Decision>,
    /// The generalization the disjunctive constraint recorded before it was decided.
    pub generalized: Option<Generalized>,
    /// The classes of which the generalization recorded whether they are
    /// references.
    pub references: BTreeSet<ClassId>,
}

/// What a disjunctive constraint recorded before it was decided: the generalization of all it may
/// become.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Generalized {
    /// A use's: the generalization of its candidates' signatures, and the class
    /// made for each variable its result writes.
    Use {
        part: Rc<generalization::Generalization>,
        classes: BTreeMap<usize, ClassId>,
    },
    /// A lend's: its result is a reference.
    Lend,
}

/// Which class a disjunctive constraint made anew is.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DisjunctionAt {
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
    /// The value of `e?`'s exit.
    Exit,
    /// A variable of a use's generalization.
    Generalized(usize),
    /// The type of the parameter at this place in a use's generalization.
    GeneralizedParam(usize),
    /// The result type in a use's generalization.
    GeneralizedResult,
    /// The type the candidate a use is decided as writes at a variable of its
    /// generalization.
    Chosen(usize),
}

/// A disjunctive constraint decided, with the relations recorded once it is: its own,
/// and those of the disjunctive constraints deferred on it that it lets record. A disjunctive constraint
/// decided by the value it read has what it read; one decided as it is
/// recorded reads nothing.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decided {
    pub disjunction: Name,
    pub decision: Decision,
    pub recorded: Recording,
    pub read: Option<DecidedBy>,
}

/// How a disjunctive constraint was decided by the value it read.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DecidedBy {
    /// The disjunctive constraint and the disjunctive constraints that transitively depend on it.
    pub dependents: BTreeSet<Name>,
    /// The undecided disjunctive constraints that broke their dependency on it.
    pub broken_on: BTreeSet<Name>,
    pub value: ValueRead,
}

/// The value a disjunctive constraint read, and what it became by it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValueRead {
    /// The disjunctive constraints whose relations it leaves out: those it was given and, for
    /// whether a class is a reference, every projection other than a Deref.
    pub left_out: BTreeSet<Name>,
    /// Their relations, by their places in the log.
    pub relations: Vec<usize>,
    pub reading: Reading,
    /// What the disjunctive constraint becomes by it, none where it decides nothing.
    pub decision: Option<Decision>,
}

/// What a disjunctive constraint read at its positions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Reading {
    /// A Deref, and a binding whose path passes no reference, read nothing.
    Nothing,
    /// A use: the value read where its candidates differ, and its narrowing.
    Use(Narrowing),
    /// A lend's place or a projection's operand, and what it reads as.
    Place(AsReference),
    /// A binding whose path passes a reference, the least mutability of
    /// those passed where decided, and what its position reads as.
    Binding {
        passed: Option<Mutability>,
        position: AsReference,
    },
    /// A call of a function value: the value read at its callee, and the
    /// lambdas its Values names there.
    Values {
        callee: ClassId,
        value: ReadValue,
        lambdas: BTreeSet<Name>,
    },
}

impl State {
    /// Decides as it is recorded a disjunctive constraint that reads nothing: a Deref.
    /// A use of one candidate has no disjunctive constraint
    /// (`State::record_one`), and a use of none is never decided.
    pub(crate) fn decide_at_once(&mut self) -> Vec<Decided> {
        let at_once: Vec<(Name, Decision)> = self
            .disjunctions
            .iter()
            .filter(|(_, disjunction)| disjunction.decision.is_none())
            .filter_map(|(name, disjunction)| match &disjunction.kind {
                Disjunction::Project {
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
                disjunction: name,
                decision,
                recorded,
                read: None,
            });
        }
        decided
    }

    pub(crate) fn set_decision(&mut self, name: &Name, decision: Decision) {
        self.disjunctions
            .get_mut(name)
            .expect("the disjunctive constraint is the state's")
            .decision = Some(decision);
    }

    /// The value the disjunctive constraint reads now, as deciding it would read it: without
    /// itself and those that transitively depend on it.
    pub(crate) fn read_now(&self, name: &Name) -> ValueRead {
        self.value_read(name, &self.dependents(name))
    }

    /// Decides the disjunctive constraint by the value it reads, which decides it (`steps`),
    /// and records what it records, or defers it.
    pub(crate) fn decide(&mut self, name: &Name) -> Decided {
        let dependents = self.dependents(name);
        let value = self.value_read(name, &dependents);
        let decision = value
            .decision
            .clone()
            .expect("a disjunctive constraint is decided where the value it reads decides it");
        let broken_on = self.broken_on(name);
        self.set_decision(name, decision.clone());
        let waiting = self.reading_what_it_records(name, &broken_on);
        let recorded = self.defer(name, waiting);
        Decided {
            disjunction: name.clone(),
            decision,
            recorded,
            read: Some(DecidedBy {
                dependents,
                broken_on,
                value,
            }),
        }
    }

    /// Those of `broken_on` whose read reaches a class the disjunctive constraint `name`, just
    /// decided, records into: the relations of a disjunctive constraint a broken
    /// dependency pointed to wait for the disjunctive constraint that broke it only where that
    /// disjunctive constraint reads them. A use records into what its candidate would record
    /// into by the decompositions, or its result where it is decided `!`;
    /// another disjunctive constraint, into what its own relations record into. A read of
    /// whether a class is a reference reaches only what may decide it.
    fn reading_what_it_records(&self, name: &Name, broken_on: &BTreeSet<Name>) -> BTreeSet<Name> {
        if broken_on.is_empty() {
            return BTreeSet::new();
        }
        let mut probe = self.clone();
        probe.record_decision(name);
        let disjunction = &self.disjunctions[name];
        let into: Vec<ClassId> = match (&disjunction.kind, &disjunction.decision) {
            (Disjunction::Use { callee, args, .. }, Some(Decision::Use(id))) => {
                let candidate = self.candidate_index(callee, args.len(), id);
                self.would_record_by(name, candidate).into_keys().collect()
            }
            (Disjunction::Use { result, .. }, Some(Decision::Never)) => {
                result.iter().copied().collect()
            }
            _ => probe
                .log
                .iter()
                .filter(|recorded| recorded.recorder.disjunctions().contains(name))
                .flat_map(|recorded| recorded_into(&recorded.relation))
                .collect(),
        };
        let into: BTreeSet<ClassId> = into.into_iter().map(|class| probe.root(class)).collect();
        let mut cannot_decide: BTreeSet<ClassId> = disjunction
            .references
            .iter()
            .map(|&class| probe.root(class))
            .collect();
        match &disjunction.kind {
            Disjunction::Lend { place, .. } => {
                cannot_decide.insert(probe.root(*place));
            }
            Disjunction::Bind { position, .. } => {
                cannot_decide.insert(probe.root(*position));
            }
            Disjunction::Use { .. }
            | Disjunction::Project { .. }
            | Disjunction::CallValue { .. } => {}
        }
        let decides_no_reference = matches!(
            &disjunction.kind,
            Disjunction::Project { projection, .. } if *projection != Projection::Deref
        );
        broken_on
            .iter()
            .filter(|breaker| {
                let Reads { read, classes } = probe.reads(breaker);
                let reached = probe.reached_by_lower_bounds(classes);
                match read {
                    Read::Value => !into.is_disjoint(&reached),
                    Read::Reference => {
                        !decides_no_reference
                            && into.iter().any(|class| {
                                reached.contains(class) && !cannot_decide.contains(class)
                            })
                    }
                }
            })
            .cloned()
            .collect()
    }

    /// The roots of `classes` and of every class they reach along the lower
    /// bounds and part lower bounds recorded, a decomposition's among them.
    fn reached_by_lower_bounds(&self, classes: Vec<ClassId>) -> BTreeSet<ClassId> {
        let mut reached: BTreeSet<ClassId> =
            classes.into_iter().map(|class| self.root(class)).collect();
        let mut work: Vec<ClassId> = reached.iter().copied().collect();
        while let Some(at) = work.pop() {
            for recorded in &self.log {
                if let Relation::LowerBound { lower, upper }
                | Relation::PartLowerBound { lower, upper } = recorded.relation
                    && self.root(upper) == at
                    && reached.insert(self.root(lower))
                {
                    work.push(self.root(lower));
                }
            }
        }
        reached
    }

    /// Records what the disjunctive constraint just decided records, or defers it until the
    /// undecided disjunctive constraints that broke their dependency on it and read what it
    /// records, `waiting`, are decided; then records what each
    /// deferred disjunctive constraint records once the disjunctive constraints it waits for are decided. Returns
    /// the relations recorded.
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
                        .all(|disjunction| self.disjunctions[disjunction].decision.is_some())
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

    /// What the disjunctive constraint becomes by the value it reads leaving out the relations
    /// of the disjunctive constraints `excluded`, as `value_read` reads it; none where that
    /// value decides nothing.
    pub(crate) fn decision_under(
        &self,
        name: &Name,
        excluded: &BTreeSet<Name>,
    ) -> Option<Decision> {
        self.value_read(name, excluded).decision
    }

    /// The value the disjunctive constraint reads leaving out the relations of the disjunctive constraints
    /// `excluded`, and, for a lend, a projection or a binding, which reads
    /// whether a class is a reference, of every projection other than a Deref
    /// whose operand is a class it reads, and what the disjunctive constraint becomes by
    /// it, none where it decides nothing.
    pub(crate) fn value_read(
        &self,
        name: &Name,
        excluded: &BTreeSet<Name>,
    ) -> ValueRead {
        let kind = &self.disjunctions[name].kind;
        let mut excluded = excluded.clone();
        if kind.read() == Read::Reference {
            let read = self.read_positions(kind);
            excluded.extend(
                self.disjunctions
                    .keys()
                    .filter(|disjunction| {
                        read.iter()
                            .any(|&class| self.projects_from(disjunction, class))
                    })
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
            Disjunction::Use { callee, args, .. } => {
                let narrowing = self.narrow(callee, args, &excluded);
                let decision = narrowing.decision.clone();
                (Reading::Use(narrowing), decision)
            }
            Disjunction::Lend {
                mutability, place, ..
            } => {
                let (read, lent) = lend(Some(*mutability), *place);
                let decision =
                    lent.map(|(target, mutability)| Decision::Lend { target, mutability });
                (Reading::Place(read), decision)
            }
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => (Reading::Nothing, Some(Decision::Applied)),
            Disjunction::Project { operand, .. } => {
                let read = self.reads_as_reference(*operand, &excluded);
                let decision = match read.reads_as {
                    Some(mutability) => Decision::AppliedToTarget(mutability),
                    None => Decision::Applied,
                };
                (Reading::Place(read), Some(decision))
            }
            Disjunction::Bind { path, position, .. } => match self.passed(path) {
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
            Disjunction::CallValue { callee, .. } => {
                let read = self.read_state(&excluded);
                let lambdas = read.values_of(read.called(*callee));
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

    /// An instance of the lambda's C made at `at` by `by`, each value
    /// of `args` passing into the copy of the lambda's parameter in its generic
    /// part, or into its parameter in the environment, and the copy of its
    /// result type in its generic part, or the environment's, passing into
    /// `result`.
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
                .expect("a lambda's group is solved first"),
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
    /// classes of `vars`: as `return None`, a value `Option⟨K⟩`, or as
    /// `return Err(x)`, a value `Result⟨O, X⟩`, `K` and `O` made anew and the
    /// candidate's `E` wrapped into `X`. The value passes into `exit`, the
    /// class the text gives it, a lower bound where that is an unknown.
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
            _ => unreachable!("`?`'s candidates are the language's two"),
        };
        let value = self.written(name, term, recorder, &mut recorded);
        let relation = if self.is_unknown(exit) {
            Relation::LowerBound {
                lower: value,
                upper: exit,
            }
        } else {
            Relation::CoercionConstraint {
                from: value,
                to: exit,
            }
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// Records the use `name` of one candidate at the positions `held`, as it
    /// is recorded: its generalization is that candidate, its type parameters
    /// made anew, so nothing is left to decide, and it has no
    /// disjunctive constraint. Each argument passes into its parameter's type and the result
    /// type into the result, an identity its result carries and no parameter
    /// does made anew, recorded by the generalization.
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
        let recorder = Recorder::Generalization(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: DisjunctionAt| ClassName::Disjunction(owner.clone(), at);
        let mut vars = BTreeMap::new();
        let var_name = |var| at(DisjunctionAt::Var(var));
        for &(position, class) in held {
            recorded.append(match position {
                Position::Arg(k) => self.record_argument(
                    class,
                    &sig.params()[k],
                    &mut vars,
                    at(DisjunctionAt::Param(k)),
                    &var_name,
                    &recorder,
                ),
                Position::Result => self.record_result(
                    class,
                    sig.result(),
                    &mut vars,
                    at(DisjunctionAt::Result),
                    &var_name,
                    &recorder,
                ),
                Position::Exit => unreachable!("`?`'s candidates are the language's two"),
            });
        }
        if sig.existential_identity()
            && let Some(&identity) = vars.get(&crate::sig::SigVar::Identity)
        {
            let made = Term::Identity(Name::new(&format!("new {name}")));
            let relation = Relation::Structure(identity, made, StructureSource::Written);
            recorded.push(self.record(relation, recorder));
        }
        recorded
    }

    /// Records what the decided disjunctive constraint records.
    pub(crate) fn record_decision(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let disjunction = self.disjunctions[name].clone();
        let Some(decision) = disjunction.decision else {
            return recorded;
        };
        let recorder = Recorder::Disjunction(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: DisjunctionAt| ClassName::Disjunction(owner.clone(), at);
        match (&disjunction.kind, &decision) {
            (
                Disjunction::Use {
                    callee,
                    args,
                    result,
                    exit,
                },
                Decision::Use(id),
            ) => {
                // Each argument passes whole into the candidate's parameter,
                // a coercion being of a whole value. What the generalization
                // holds of the result is not recorded again: the candidate's
                // type enters each of the generalization's variables there.
                let sig = self.decls.decl(id).sig.clone();
                let generalized = match &disjunction.generalized {
                    Some(Generalized::Use { part, classes }) => {
                        Some((part.clone(), classes.clone()))
                    }
                    _ => None,
                };
                // The disjunctive constraint holds no position where the generalization writes a
                // closed type, nor one another group records.
                let mut vars = BTreeMap::new();
                let var_name = |var| at(DisjunctionAt::Var(var));
                for (i, (arg, param)) in args.iter().zip(sig.params()).enumerate() {
                    let Some(arg) = arg else {
                        continue;
                    };
                    recorded.append(self.record_argument(
                        *arg,
                        param,
                        &mut vars,
                        at(DisjunctionAt::Param(i)),
                        &var_name,
                        &recorder,
                    ));
                }
                match &generalized {
                    None => {
                        if let Some(result) = result {
                            recorded.append(self.record_result(
                                *result,
                                sig.result(),
                                &mut vars,
                                at(DisjunctionAt::Result),
                                &var_name,
                                &recorder,
                            ));
                        }
                    }
                    Some((part, classes)) => {
                        let candidate = self.candidate_index(callee, args.len(), id);
                        for k in generalization::Generalization::vars_in(&part.result) {
                            recorded.append(self.record_result(
                                classes[&k],
                                &part.vars[k][candidate],
                                &mut vars,
                                at(DisjunctionAt::Chosen(k)),
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
                    let relation = Relation::Structure(identity, made, StructureSource::Written);
                    recorded.push(self.record(relation, recorder.clone()));
                }
                if let Some(exit) = exit {
                    recorded.append(self.record_exit(
                        &sig,
                        &vars,
                        at(DisjunctionAt::Exit),
                        *exit,
                        &recorder,
                    ));
                }
            }
            // Where the result is a closed type of the generalization, the disjunctive constraint
            // does not hold it, and records nothing there.
            (
                Disjunction::Use {
                    result: Some(result),
                    ..
                },
                Decision::Never,
            ) => {
                let never = self.written(
                    at(DisjunctionAt::Result),
                    Term::Never,
                    &recorder,
                    &mut recorded,
                );
                let relation = Relation::LowerBound {
                    lower: never,
                    upper: *result,
                };
                recorded.push(self.record(relation, recorder));
            }
            (Disjunction::Lend { place, result, .. }, Decision::Lend { target, mutability }) => {
                let lent = self.lend(*mutability, *place, *target, &at, &recorder, &mut recorded);
                recorded.push(self.record(Relation::Equation(*result, lent), recorder));
            }
            (
                Disjunction::Project {
                    projection,
                    operand,
                    result,
                },
                Decision::Applied,
            ) => {
                let relation = Relation::StructureConstraint(
                    *operand,
                    projection.structure_constraint(*result),
                );
                recorded.push(self.record(relation, recorder));
            }
            (
                Disjunction::Project {
                    projection,
                    operand,
                    result,
                },
                Decision::AppliedToTarget(_),
            ) => {
                let target = self.arena.intern(at(DisjunctionAt::Target));
                self.classes.grow(&self.arena);
                let relation =
                    Relation::StructureConstraint(*operand, StructureConstraint::Target(target));
                recorded.push(self.record(relation, recorder.clone()));
                let relation =
                    Relation::StructureConstraint(target, projection.structure_constraint(*result));
                recorded.push(self.record(relation, recorder));
            }
            (
                Disjunction::Bind {
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
            (Disjunction::CallValue { args, result, .. }, Decision::Calls(lambdas)) => {
                for lambda in lambdas {
                    let at = Name::new(&format!("{name}#{lambda}"));
                    recorded.append(self.instantiate_lambda(&at, &recorder, lambda, args, *result));
                }
            }
            (
                Disjunction::Bind {
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

/// The classes a relation records into: the upper side of a lower
/// bound or a part lower bound, the type passed into at a coercion constraint or a
/// wrapping, both sides of an equation, and the class a structure, a structure constraint, a
/// domain or a lambda is given to.
fn recorded_into(relation: &Relation) -> Vec<ClassId> {
    match relation {
        Relation::LowerBound { upper, .. } | Relation::PartLowerBound { upper, .. } => vec![*upper],
        Relation::CoercionConstraint { to, .. } | Relation::Wrap { to, .. } => vec![*to],
        Relation::Equation(a, b) => vec![*a, *b],
        Relation::StructureConstraint(class, _)
        | Relation::Structure(class, ..)
        | Relation::Domain(class, _)
        | Relation::Values(class, _) => vec![*class],
    }
}
