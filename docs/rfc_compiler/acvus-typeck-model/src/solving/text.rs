//! The record of a program's text: the entry group, without uses, hints, lambdas or other groups yet.

use crate::program::{ENTRY, Item, Origin, Place, Program};
use crate::ty::Name;

use super::choice;
use super::record::{ClassId, ClassName, Hint, HintKind, Record, Relation};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NotCovered(pub String);

pub fn record(program: &Program, decls: &crate::decl::Decls) -> Result<Record, NotCovered> {
    if !program.groups.is_empty() || !program.functions.is_empty() {
        return Err(NotCovered("a binding group other than the entry".into()));
    }
    if !program.universals.is_empty() {
        return Err(NotCovered("a universal class".into()));
    }
    let mut record = Record::default();
    let text = |record: &mut Record, name: &Name| -> ClassId {
        record.classes.intern(ClassName::Text(name.clone()))
    };
    for class in &program.classes {
        if class.group.as_str() != ENTRY {
            return Err(NotCovered("a class of another group".into()));
        }
        let id = text(&mut record, &class.name);
        match &class.origin {
            Origin::Var => {}
            Origin::Lit(lit) => record.relations.push(Relation::Literal {
                class: id,
                lit: lit.clone(),
            }),
            Origin::Written(term) | Origin::Host(term) | Origin::Outward(term) => {
                let structure = term.clone().map(|_, part| text(&mut record, &part));
                record.relations.push(Relation::Written {
                    class: id,
                    structure,
                });
            }
            Origin::Lambda(_) => return Err(NotCovered("a lambda".into())),
        }
    }
    for (_, item) in &program.items {
        match item {
            Item::Equation(a, b) => {
                let (a, b) = (text(&mut record, a), text(&mut record, b));
                record.classes.equate(a, b);
            }
            Item::Pass { value, into, place } => {
                let (value, into) = (text(&mut record, value), text(&mut record, into));
                let written = record.structure(into).is_some();
                let relation = match (place, written) {
                    (Place::ThroughRef, _) | (_, true) => Relation::Into { value, written: into },
                    (Place::Slot | Place::ByName, false) => Relation::Enters {
                        lower: value,
                        upper: into,
                        part: false,
                    },
                };
                record.relations.push(relation);
            }
            Item::Use {
                name,
                callee,
                args,
                result,
                exit,
            } => {
                if exit.is_some() {
                    return Err(NotCovered("`e?`".into()));
                }
                if program.function(callee).is_some() {
                    return Err(NotCovered("a call of a function of the program".into()));
                }
                let args: Vec<ClassId> = args.iter().map(|arg| text(&mut record, arg)).collect();
                let result = text(&mut record, result);
                choice::record_use(&mut record, decls, name, callee, args, result);
            }
            Item::Lend {
                name,
                mutability,
                place,
                result,
            } => {
                let (place, result) = (text(&mut record, place), text(&mut record, result));
                record.hints.push(Hint {
                    name: name.clone(),
                    kind: HintKind::Lend {
                        mutability: *mutability,
                        place,
                        result,
                    },
                    decision: None,
                });
            }
            Item::Project {
                name,
                projection,
                operand,
                result,
            } => {
                let (operand, result) = (text(&mut record, operand), text(&mut record, result));
                record.hints.push(Hint {
                    name: name.clone(),
                    kind: HintKind::Project {
                        projection: projection.clone(),
                        operand,
                        result,
                    },
                    decision: None,
                });
            }
            Item::Bind {
                name,
                path,
                position,
                binding,
            } => {
                let (position, binding) = (text(&mut record, position), text(&mut record, binding));
                record.hints.push(Hint {
                    name: name.clone(),
                    kind: HintKind::Bind {
                        path: path.clone(),
                        position,
                        binding,
                    },
                    decision: None,
                });
            }
            Item::CallValue { .. } => return Err(NotCovered("a call of a function value".into())),
        }
    }
    Ok(record)
}
