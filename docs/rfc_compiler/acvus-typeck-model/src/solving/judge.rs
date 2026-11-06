//! What the record gives a class now: rules 1 to 4 of the judgment.

use std::collections::BTreeSet;

use crate::lit::LitVar;
use crate::ty::{Constructor, Member, Term, lesser};

use super::record::{ClassId, Record, Refusal, Relations};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Head {
    pub constructor: Constructor,
    pub members: BTreeSet<Member>,
}

impl Head {
    pub fn of(term: &Term<ClassId>) -> Head {
        Head {
            constructor: term.constructor(),
            members: term.members(),
        }
    }

    pub fn bare(constructor: Constructor) -> Head {
        Head {
            constructor,
            members: BTreeSet::new(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Judged {
    Head { head: Head, by_lower: bool },
    Wait { occurrence_only: bool },
    Refuse(Refusal),
    Nothing,
}

pub fn literal_admits(lit: &LitVar, constructor: &Constructor) -> bool {
    match constructor {
        Constructor::Scalar(scalar) => lit.admits(*scalar),
        _ => false,
    }
}

pub fn join_heads(a: &Head, b: &Head) -> Option<Head> {
    match (&a.constructor, &b.constructor) {
        (Constructor::Never, _) => Some(b.clone()),
        (_, Constructor::Never) => Some(a.clone()),
        (Constructor::Ref(m), Constructor::Ref(n)) => {
            let mutability = lesser(Some(*m), Some(*n))?;
            Some(Head::bare(Constructor::Ref(mutability)))
        }
        (c, d) if c != d => None,
        (Constructor::Object | Constructor::Enum(_), _) => Some(Head {
            constructor: a.constructor.clone(),
            members: a.members.union(&b.members).cloned().collect(),
        }),
        _ if a.members == b.members => Some(a.clone()),
        _ => None,
    }
}

pub fn reaches_own_part(record: &Record, relations: &Relations, root: ClassId) -> bool {
    let mut seen: BTreeSet<(ClassId, bool)> = BTreeSet::new();
    let mut work: Vec<(ClassId, bool)> = vec![(root, false)];
    while let Some((at, through_part)) = work.pop() {
        let mut next: Vec<(ClassId, bool)> = record
            .lower_bounds(relations, at)
            .into_iter()
            .map(|(lower, _)| (lower, through_part))
            .collect();
        if let Some(structure) = record.structure(at) {
            next.extend(
                structure
                    .parts()
                    .into_iter()
                    .map(|(_, part)| (record.root(*part), true)),
            );
        }
        for step in next {
            if step.0 == root && step.1 {
                return true;
            }
            if seen.insert(step) {
                work.push(step);
            }
        }
    }
    false
}

pub fn judge(record: &Record, relations: &Relations, root: ClassId) -> Judged {
    let root = record.root(root);
    if reaches_own_part(record, relations, root) {
        return Judged::Refuse(Refusal::Infinite);
    }
    if record.may_enter(relations, root) {
        return Judged::Wait {
            occurrence_only: false,
        };
    }
    let mut heads: Vec<Head> = Vec::new();
    let mut literals: Vec<&LitVar> = record.literals(root);
    for (lower, _) in record.lower_bounds(relations, root) {
        if record.is_refused(lower) {
            continue;
        }
        match record.structure(lower) {
            Some(structure) => heads.push(Head::of(structure)),
            None if relations.same_component(root, lower) && !record.may_enter(relations, lower) => {
                literals.extend(record.literals(lower))
            }
            None => return Judged::Wait { occurrence_only: false },
        }
    }
    let admits_literals = |head: &Head| -> bool {
        head.constructor == Constructor::Never
            || literals
                .iter()
                .all(|lit| literal_admits(lit, &head.constructor))
    };
    if let Some(first) = heads.first() {
        let joined = heads[1..]
            .iter()
            .try_fold(first.clone(), |joined, next| join_heads(&joined, next));
        let Some(joined) = joined else {
            return Judged::Refuse(Refusal::NoJoin);
        };
        if !admits_literals(&joined) {
            return Judged::Refuse(Refusal::Conflict);
        }
        return Judged::Head {
            head: joined,
            by_lower: true,
        };
    }
    if record.may_take(relations, root) {
        return Judged::Wait {
            occurrence_only: true,
        };
    }
    let mut contexts: Vec<Head> = Vec::new();
    for occurrence in record.occurrences(relations, root) {
        if record.is_refused(occurrence) {
            continue;
        }
        match record.structure(occurrence) {
            Some(structure) => contexts.push(Head::of(structure)),
            None if relations.same_component(root, occurrence) && !record.may_enter(relations, occurrence) => {}
            None => return Judged::Wait { occurrence_only: true },
        }
    }
    let Some(first) = contexts.first() else {
        return Judged::Nothing;
    };
    if !contexts.iter().all(|head| head == first) {
        return Judged::Refuse(Refusal::Contexts);
    }
    if !admits_literals(first) {
        return Judged::Refuse(Refusal::Conflict);
    }
    Judged::Head {
        head: first.clone(),
        by_lower: false,
    }
}
