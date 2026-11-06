//! Disjunctive constraints: lends, projections, bindings and calls of a
//! function value, what each is, where its classes occur in it, and what
//! each records once decided; the structure constraint a projection gives
//! and the reference a lend gives are `place`. A disjunctive constraint
//! stands where the AST gives a hint whose meaning depends on the
//! constructor of a class it reads: `&r` lends `r`'s target where `r` is a
//! reference, since `&&T` does not exist, and `r` itself otherwise; a
//! projection applies to its operand or, the operand being a reference, to
//! its target; a binding reached through a reference lends its position, as
//! a lend does; a call calls the lambdas its callee's value names. A use of
//! declarations is none: its candidates are `overload`, and the
//! generalization of its candidates, recorded at every position as the use
//! is recorded, `generalization`, with what every alternative of a lend
//! shares.

pub(crate) mod generalization;
mod overload;
mod place;

use std::collections::BTreeSet;

use crate::class::{ClassId, StructureConstraint};
use crate::program::{Position, Projection};
use crate::record::Recording;
use crate::relation::{Recorder, Relation, Role};
use crate::solve::State;
use crate::ty::{Mutability, Name};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Disjunction {
    /// `&ₘ p`: a reference of `mutability` to `place`, or to its target where
    /// `place` is a reference, which `result` holds.
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
    /// recorded, each with its role: a lend's place and result, a call's
    /// callee and positions, a projection's operand and result, and a
    /// binding's position and binding.
    pub fn occurrences(&self) -> Vec<(ClassId, Role)> {
        match self {
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

/// What a lend, a projection, a binding or a call of a function value
/// became.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decision {
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
    /// The binding's path passes a reference, so it lends its position, as
    /// a lend does: a reference of `mutability` to `target`.
    BindLend {
        target: LendTarget,
        mutability: Mutability,
    },
    /// The call reads the function value as one whose Values names these
    /// lambdas, and makes an instance of each.
    Calls(BTreeSet<Name>),
}

/// What a lend's reference points to (D1: `&&T` does not exist, so `&r`
/// with `r: &T` is `&T`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LendTarget {
    /// The place, which is no reference.
    Place,
    /// The target of the reference the place is.
    Through,
}

/// Which class a disjunctive constraint, or a use of declarations, made
/// anew is.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DisjunctionAt {
    /// A variable of the signature of the candidate a use records.
    Var(crate::sig::SigVar),
    /// The type of the parameter at this place.
    Param(usize),
    Result,
    /// The reference a lend, or a binding that lends, gives.
    Lent,
    /// The target of the reference a lend's place, or a projection's
    /// operand, is.
    Target,
    /// The value of `e?`'s exit.
    Exit,
    /// A variable of a use's generalization.
    Generalized(usize),
    /// The type of the parameter at this place in a use's generalization.
    GeneralizedParam(usize),
    /// The result type in a use's generalization.
    GeneralizedResult,
    /// The type the one candidate a use's choice admits writes at a
    /// variable of its generalization.
    Chosen(usize),
}

impl State {
    pub(crate) fn set_decision(&mut self, name: &Name, decision: Decision) {
        let at = self.disjunction_at(name);
        self.classes.decide(at, Some(decision));
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
        match (&kind, &decision) {
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
