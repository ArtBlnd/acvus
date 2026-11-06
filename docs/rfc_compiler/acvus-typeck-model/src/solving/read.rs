//! Reading a class as the value it would be, without recording anything.

use std::collections::BTreeSet;
use std::rc::Rc;

use crate::domain::Domain;
use crate::read::ReadValue;
use crate::ty::{Constructor, Name, Term};

use super::judge::{Head, join_heads};
use super::record::{ClassId, Record, Relations};

pub fn head_now(record: &Record, relations: &Relations, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Head> {
    head_now_for(record, relations, id, None, visiting)
}

// Reading for a use's narrowing leaves out what that use would itself record:
// its result variables give nothing, and a class whose lower bounds give no
// head is read as the one head of its occurrences, rule 3, since what the
// use records cannot be an input of choosing it.
pub fn head_now_for(
    record: &Record,
    relations: &Relations,
    id: ClassId,
    excluding: Option<&Name>,
    visiting: &mut BTreeSet<ClassId>,
) -> Option<Head> {
    let root = record.root(id);
    if record.is_refused(root) {
        return None;
    }
    if let Some(structure) = record.structure(root) {
        return Some(Head::of(structure));
    }
    if let Some(name) = excluding.filter(|_| !visiting.is_empty()) {
        let excluded = record.choices.iter().any(|choice| {
            choice.name == *name
                && choice
                    .result_vars(record)
                    .iter()
                    .any(|&var| record.root(var) == root)
        });
        if excluded {
            return None;
        }
    }
    if !visiting.insert(root) {
        return None;
    }
    let mut joined: Option<Head> = None;
    for (lower, _) in record.lower_bounds(relations, root) {
        let Some(head) = head_now_for(record, relations, lower, excluding, visiting) else {
            continue;
        };
        joined = Some(match joined {
            None => head,
            Some(so_far) => match join_heads(&so_far, &head) {
                Some(joined) => joined,
                None => {
                    visiting.remove(&root);
                    return None;
                }
            },
        });
    }
    let joined =
        joined.filter(|head| head.constructor != Constructor::Never || record.lower_bounds(relations, root).len() == 1);
    if joined.is_some() || excluding.is_none() {
        visiting.remove(&root);
        return joined;
    }
    let mut context: Option<Head> = None;
    for occurrence in record.occurrences(relations, root) {
        let Some(head) = head_now_for(record, relations, occurrence, excluding, visiting) else {
            continue;
        };
        match &context {
            None => context = Some(head),
            Some(so_far) if *so_far == head => {}
            Some(_) => {
                visiting.remove(&root);
                return None;
            }
        }
    }
    visiting.remove(&root);
    context
}

fn literal_domain(record: &Record, relations: &Relations, root: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Domain> {
    if !visiting.insert(root) {
        return None;
    }
    let mut domain: Option<Domain> = None;
    let own = record.literals(root).into_iter().map(Domain::of_literal);
    let below = record
        .lower_bounds(relations, root)
        .into_iter()
        .filter_map(|(lower, _)| literal_domain(record, relations, lower, visiting));
    for next in own.chain(below) {
        domain = Some(match domain {
            None => next,
            Some(so_far) => so_far
                .meet(&next)
                .unwrap_or_else(|| Domain::Only(BTreeSet::new())),
        });
    }
    visiting.remove(&root);
    domain
}

pub fn read_value(record: &Record, relations: &Relations, id: ClassId) -> ReadValue {
    read_value_for(record, relations, id, None)
}

pub fn read_value_for(record: &Record, relations: &Relations, id: ClassId, excluding: Option<&Name>) -> ReadValue {
    fn read_in(
        record: &Record,
        relations: &Relations,
        id: ClassId,
        excluding: Option<&Name>,
        visiting: &mut BTreeSet<ClassId>,
    ) -> ReadValue {
        let root = record.root(id);
        if record.is_refused(root) || !visiting.insert(root) {
            return ReadValue::Undecided;
        }
        let value = match record.structure(root) {
            Some(structure) => {
                let term = structure
                    .clone()
                    .map(|_, part| read_in(record, relations, part, excluding, visiting));
                ReadValue::Term(Rc::new(term))
            }
            None => match head_now_for(record, relations, root, excluding, &mut BTreeSet::new()) {
                Some(head)
                    if matches!(head.constructor, Constructor::Struct(_) | Constructor::Extern(_)) =>
                {
                    ReadValue::Domain(Domain::Only([head.constructor.clone()].into()))
                }
                Some(head) => ReadValue::Term(Rc::new(bare_term(&head))),
                None => literal_domain(record, relations, root, &mut BTreeSet::new())
                    .map_or(ReadValue::Undecided, ReadValue::Domain),
            },
        };
        visiting.remove(&root);
        value
    }
    read_in(record, relations, id, excluding, &mut BTreeSet::new())
}

fn bare_term(head: &Head) -> Term<ReadValue> {
    let part = || ReadValue::Undecided;
    let named = |members: &BTreeSet<crate::ty::Member>| -> Vec<crate::ty::Name> {
        members
            .iter()
            .map(|member| match member {
                crate::ty::Member::Field(name) | crate::ty::Member::Variant(name) => name.clone(),
            })
            .collect()
    };
    match &head.constructor {
        Constructor::Ref(m) => Term::Ref(*m, part()),
        Constructor::Tuple(n) => Term::Tuple((0..*n).map(|_| part()).collect()),
        Constructor::Array(n) => Term::Array(*n, part()),
        Constructor::Slice => Term::Slice(part()),
        Constructor::Option => Term::Option(part()),
        Constructor::Result => Term::Result {
            ok: part(),
            err: part(),
        },
        Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
        Constructor::Object => {
            Term::Object(named(&head.members).into_iter().map(|f| (f, part())).collect())
        }
        Constructor::Enum(name) => Term::Enum(
            name.clone(),
            named(&head.members).into_iter().map(|v| (v, part())).collect(),
        ),
        Constructor::ErrorOf => Term::ErrorOf(part()),
        Constructor::Struct(_) | Constructor::Extern(_) => {
            unreachable!("a struct or an extern type is read as its domain")
        }
        Constructor::Scalar(scalar) => Term::Scalar(*scalar),
        Constructor::String => Term::String,
        Constructor::Str => Term::Str,
        Constructor::Unit => Term::Unit,
        Constructor::Never => Term::Never,
        Constructor::Identity(name) => Term::Identity(name.clone()),
    }
}
