//! Disjunctive constraints: uses, lends, projections, bindings and calls of a
//! function value, what each is, where its classes occur in it, and what
//! each records once decided; places are `place`, a use's candidates
//! `overload`, and the generalization of a use's candidates, by which the
//! positions where every candidate writes one closed type are recorded
//! apart, `generalization`.

pub(crate) mod generalization;
mod overload;
mod place;

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, StructureConstraint, StructureSource};
use crate::decl::DeclId;
use crate::program::{Position, Projection};
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation, Role};
use crate::solve::State;
use crate::ty::{Mutability, Name, Term};

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

impl Disjunction {
    /// The classes occurring in the disjunctive constraint as it is
    /// recorded, each with its role: a use's, a call's and a lend's
    /// positions, a lend's place, a projection's operand and result, and a
    /// binding's position and binding.
    pub fn occurrences(&self) -> Vec<(ClassId, Role)> {
        match self {
            Disjunction::Use {
                args, result, exit, ..
            } => args
                .iter()
                .enumerate()
                .filter_map(|(k, arg)| Some((((*arg)?), Role::Position(Position::Arg(k)))))
                .chain(result.iter().map(|&result| (result, Role::Position(Position::Result))))
                .chain(exit.iter().map(|&exit| (exit, Role::Position(Position::Exit))))
                .collect(),
            Disjunction::Lend { place, result, .. } => {
                vec![(*place, Role::Place), (*result, Role::Position(Position::Result))]
            }
            Disjunction::Project {
                operand,
                result,
                projection,
            } => vec![
                (*operand, Role::Operand),
                (*result, Role::Part(projection.key())),
            ],
            Disjunction::Bind {
                position, binding, ..
            } => vec![(*position, Role::Bound), (*binding, Role::Binding)],
            Disjunction::CallValue {
                callee,
                args,
                result,
            } => std::iter::once((*callee, Role::Callee))
                .chain(
                    args.iter()
                        .enumerate()
                        .map(|(k, &arg)| (arg, Role::Position(Position::Arg(k)))),
                )
                .chain([(*result, Role::Position(Position::Result))])
                .collect(),
        }
    }

    /// Whether the disjunctive constraint holds no position to decide here:
    /// a use whose positions where its candidates differ another group
    /// records, or every one of which the candidates write one closed type.
    pub fn is_empty(&self) -> bool {
        matches!(
            self,
            Disjunction::Use { args, result, exit, .. }
                if args.iter().all(Option::is_none) && result.is_none() && exit.is_none()
        )
    }

    /// The classes the disjunctive constraint names.
    pub fn classes(&self) -> Vec<ClassId> {
        self.occurrences()
            .into_iter()
            .map(|(class, _)| class)
            .collect()
    }

    /// The disjunctive constraint with each class it names put through `f`,
    /// and each step of a binding's path through `step`.
    pub fn map<F, G>(self, mut f: F, step: G) -> Disjunction
    where
        F: FnMut(ClassId) -> ClassId,
        G: Fn(&Name) -> Name,
    {
        match self {
            Disjunction::Use {
                callee,
                args,
                result,
                exit,
            } => Disjunction::Use {
                callee,
                args: args.into_iter().map(|arg| arg.map(&mut f)).collect(),
                result: result.map(&mut f),
                exit: exit.map(&mut f),
            },
            Disjunction::Lend {
                mutability,
                place,
                result,
            } => Disjunction::Lend {
                mutability,
                place: f(place),
                result: f(result),
            },
            Disjunction::Project {
                projection,
                operand,
                result,
            } => Disjunction::Project {
                projection,
                operand: f(operand),
                result: f(result),
            },
            Disjunction::Bind {
                path,
                position,
                binding,
            } => Disjunction::Bind {
                path: path.iter().map(step).collect(),
                position: f(position),
                binding: f(binding),
            },
            Disjunction::CallValue {
                callee,
                args,
                result,
            } => Disjunction::CallValue {
                callee: f(callee),
                args: args.into_iter().map(&mut f).collect(),
                result: f(result),
            },
        }
    }
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
    /// reference's target (auto-deref).
    AppliedToTarget,
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

/// What a lend's reference points to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LendTarget {
    /// The place, which is not a reference.
    Place,
    /// The target of the reference the place is.
    Through,
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
    /// A variable of a use's generalization, which its result writes.
    Generalized(usize),
    /// The type of the parameter at this place in a use's generalization,
    /// where every candidate writes one closed type there.
    GeneralizedParam(usize),
    /// The result type in a use's generalization, where every candidate
    /// writes one closed type there.
    GeneralizedResult,
    /// The type the candidate a use is decided as writes at a variable of its
    /// generalization.
    Chosen(usize),
}

impl State {
    pub(crate) fn set_decision(&mut self, name: &Name, decision: Decision) {
        let at = self.disjunction_at(name);
        self.classes.decide(at, Some(decision));
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
        making: Making,
        exit: ClassId,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        let named = Rc::new(making.label.clone());
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
        let value = self.written(making, term, recorder, &mut recorded);
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

    /// Records the use `name` of one candidate: a disjunctive constraint of
    /// one alternative, decided as it is recorded, so that what it records
    /// and the classes it makes anew are its, as any decided use's.
    pub(crate) fn record_one(
        &mut self,
        name: &Name,
        callee: &Name,
        kind: Disjunction,
        recorder: Recorder,
    ) -> Recording {
        let Disjunction::Use { args, .. } = &kind else {
            unreachable!("a use of one candidate is a use");
        };
        let id = match self.candidates(callee, args.len()).as_slice() {
            [candidate] => candidate.id.clone(),
            _ => panic!("{name}: a use of one candidate"),
        };
        let mut recorded = Recording::default();
        recorded.push(self.add_disjunction(name, kind, recorder));
        self.set_decision(name, Decision::Use(id));
        recorded.append(self.record_decision(name));
        recorded
    }

    /// Records what the decided disjunctive constraint records.
    pub(crate) fn record_decision(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let (kind, decision) = self.disjunction(name);
        let (kind, decision) = (kind.clone(), decision.cloned());
        let Some(decision) = decision else {
            return recorded;
        };
        let recorder = Recorder::Disjunction(name.clone());
        let owner = self.owner(name);
        let at = |at: DisjunctionAt| owner.made(at);
        match (&kind, &decision) {
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
                // a coercion being of a whole value, and the candidate's
                // result type enters the result. The disjunctive constraint
                // holds no position where every candidate writes one closed
                // type, nor one another group records.
                let sig = self.decls.decl(id).sig.clone();
                let mut vars = BTreeMap::new();
                let var_name = |var| owner.made(DisjunctionAt::Var(var));
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
                // What the generalization holds of the result is not recorded
                // again: the candidate's type enters each of the
                // generalization's variables there.
                let candidates: Vec<crate::sig::Signature> = self
                    .candidates(callee, args.len())
                    .into_iter()
                    .map(|decl| decl.sig.clone())
                    .collect();
                let generalization = (candidates.len() >= 2)
                    .then(|| generalization::Generalization::of(&candidates.iter().collect::<Vec<_>>()));
                match (&generalization, result) {
                    (Some(part), Some(_)) => {
                        let candidate = candidates
                            .iter()
                            .position(|candidate| *candidate == sig)
                            .expect("the decided candidate is among the candidates");
                        for k in generalization::Generalization::vars_in(&part.result) {
                            let class = self
                                .classes
                                .in_role(owner.item, &Role::Made(DisjunctionAt::Generalized(k)))
                                .expect("the generalization made a class for each variable its result writes");
                            recorded.append(self.record_result(
                                class,
                                &part.vars[k][candidate],
                                &mut vars,
                                at(DisjunctionAt::Chosen(k)),
                                &var_name,
                                &recorder,
                            ));
                        }
                    }
                    (None, Some(result)) => {
                        recorded.append(self.record_result(
                            *result,
                            sig.result(),
                            &mut vars,
                            at(DisjunctionAt::Result),
                            &var_name,
                            &recorder,
                        ));
                    }
                    (_, None) => {}
                }
                if sig.existential_identity()
                    && let Some(&identity) = vars.get(&crate::sig::SigVar::Identity)
                {
                    let made = Term::Identity(Name::new(&format!("new {name}")));
                    let relation = Relation::Structure(identity, made, StructureSource::Solved);
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
            // Where every candidate writes one closed result type, the
            // disjunctive constraint does not hold the result, and records
            // nothing there.
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
                let lent = self.lend(*mutability, *place, *target, &owner, &recorder, &mut recorded);
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
                Decision::AppliedToTarget,
            ) => {
                let target = self.make(owner.made(DisjunctionAt::Target));
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
                    &owner,
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
