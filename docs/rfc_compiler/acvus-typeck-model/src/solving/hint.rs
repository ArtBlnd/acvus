//! A place where the AST gives a hint: the term's meaning depends on the
//! constructor of a class, which the hint reads and never decides.

use std::collections::BTreeSet;

use crate::decl::Decls;
use crate::program::Projection;
use crate::ty::{Constructor, Mutability, Name, Term, lesser};

use super::read::head_now;
use super::record::{ClassId, Hint, HintDecision, HintKind, Record, Relation, Relations, Requirement};

fn constructor_now(record: &Record, relations: &Relations, class: ClassId) -> Option<Constructor> {
    head_now(record, relations, class, &mut BTreeSet::new()).map(|head| head.constructor)
}

// A class whose only lower bound is the result of a use not yet decided has
// one of the candidates' result constructors. A hint asks only whether it is a
// reference, which the set answers where every candidate answers alike.
fn is_reference_now(record: &Record, relations: &Relations, decls: &Decls, class: ClassId) -> Option<Option<Mutability>> {
    if let Some(constructor) = constructor_now(record, relations, class) {
        return Some(match constructor {
            Constructor::Ref(m) => Some(m),
            _ => None,
        });
    }
    let root = record.root(class);
    let mut answers: BTreeSet<Option<Mutability>> = BTreeSet::new();
    let mut lowers = record.lower_bounds(relations, root).into_iter();
    let (var, _) = lowers.next()?;
    if lowers.next().is_some() {
        return None;
    }
    let var = record.root(var);
    for choice in &record.choices {
        let Some(k) = choice.vars.iter().position(|&v| record.root(v) == var) else {
            continue;
        };
        if choice.remaining.len() < 2 {
            return None;
        }
        let sigs: Vec<&crate::sig::Signature> = choice
            .candidates
            .iter()
            .map(|id| &decls.decl(id).sig)
            .collect();
        let generalization = crate::disjunction::generalization::Generalization::of(&sigs);
        for &i in &choice.remaining {
            match &generalization.vars[k][i] {
                crate::sig::SigTy::Term(term) => {
                    answers.insert(match &**term {
                        Term::Ref(m, _) => Some(*m),
                        _ => None,
                    });
                }
                crate::sig::SigTy::Var(_) => return None,
            }
        }
    }
    match answers.iter().collect::<Vec<_>>().as_slice() {
        [one] => Some(**one),
        _ => None,
    }
}

fn lend(record: &mut Record, relations: &Relations, decls: &Decls, name: &Name, mutability: Mutability, place: ClassId, result: ClassId) -> Option<HintDecision> {
    match is_reference_now(record, relations, decls, place)? {
        Some(m) => {
            let mutability = lesser(Some(mutability), Some(m))?;
            let target = record.classes.fresh(name);
            record.relations.push(Relation::Requires {
                class: place,
                requirement: Requirement {
                    projection: Projection::Deref,
                    part: target,
                },
            });
            record.relations.push(Relation::Written {
                class: result,
                structure: Term::Ref(mutability, target),
            });
            Some(HintDecision::Through { mutability })
        }
        _ => {
            record.relations.push(Relation::Written {
                class: result,
                structure: Term::Ref(mutability, place),
            });
            Some(HintDecision::Place)
        }
    }
}

fn project(record: &mut Record, relations: &Relations, decls: &Decls, name: &Name, projection: &Projection, operand: ClassId, result: ClassId) -> Option<HintDecision> {
    if *projection == Projection::Deref {
        record.relations.push(Relation::Requires {
            class: operand,
            requirement: Requirement {
                projection: projection.clone(),
                part: result,
            },
        });
        return Some(HintDecision::Applied);
    }
    match is_reference_now(record, relations, decls, operand)? {
        Some(_) => {
            let target = record.classes.fresh(name);
            record.relations.push(Relation::Requires {
                class: operand,
                requirement: Requirement {
                    projection: Projection::Deref,
                    part: target,
                },
            });
            record.relations.push(Relation::Requires {
                class: target,
                requirement: Requirement {
                    projection: projection.clone(),
                    part: result,
                },
            });
            Some(HintDecision::AppliedToTarget)
        }
        _ => {
            record.relations.push(Relation::Requires {
                class: operand,
                requirement: Requirement {
                    projection: projection.clone(),
                    part: result,
                },
            });
            Some(HintDecision::Applied)
        }
    }
}

fn bind(record: &mut Record, relations: &Relations, decls: &Decls, name: &Name, path: &[Name], position: ClassId, binding: ClassId) -> Option<HintDecision> {
    let after_deref = path
        .iter()
        .rposition(|step| {
            record.hints.iter().any(|hint| {
                hint.name == *step
                    && matches!(hint.kind, HintKind::Project { projection: Projection::Deref, .. })
            })
        })
        .map_or(0, |i| i + 1);
    let mut passed: Option<Mutability> = None;
    for step in &path[after_deref..] {
        let hint = record.hints.iter().find(|hint| hint.name == *step)?;
        let HintKind::Project { operand, .. } = &hint.kind else {
            return None;
        };
        match hint.decision.as_ref()? {
            HintDecision::AppliedToTarget => {
                let m = is_reference_now(record, relations, decls, *operand)??;
                passed = Some(match passed {
                    None => m,
                    Some(so_far) => lesser(Some(so_far), Some(m))?,
                });
            }
            _ => {}
        }
    }
    match passed {
        None => {
            record.relations.push(Relation::Enters {
                lower: position,
                upper: binding,
                part: false,
            });
            Some(HintDecision::Read)
        }
        Some(mutability) => {
            let lent = record.classes.fresh(name);
            let decision = lend(record, relations, decls, name, mutability, position, lent)?;
            record.relations.push(Relation::Enters {
                lower: lent,
                upper: binding,
                part: false,
            });
            Some(match decision {
                HintDecision::Through { mutability } => HintDecision::Through { mutability },
                _ => HintDecision::Through { mutability },
            })
        }
    }
}

pub fn decide(record: &mut Record, decls: &Decls) -> bool {
    let mut changed = false;
    let relations = Relations::of(record);
    let relations = &relations;
    for at in 0..record.hints.len() {
        if record.hints[at].decision.is_some() {
            continue;
        }
        let Hint { name, kind, .. } = record.hints[at].clone();
        let decision = match &kind {
            HintKind::Lend {
                mutability,
                place,
                result,
            } => lend(record, relations, decls, &name, *mutability, *place, *result),
            HintKind::Project {
                projection,
                operand,
                result,
            } => project(record, relations, decls, &name, projection, *operand, *result),
            HintKind::Bind {
                path,
                position,
                binding,
            } => bind(record, relations, decls, &name, path, *position, *binding),
        };
        if decision.is_some() {
            record.hints[at].decision = decision;
            changed = true;
        }
    }
    changed
}
