//! `e as _`: the type `e` comes to by itself enters what is outside.
//!
//! It asks no head of `e` and holds nothing apart, so every class `e` makes is
//! decided by what is recorded on it alone, fallback included. What `e` only names
//! is decided as the rules outside decide it, and this waits for it.

use std::collections::BTreeSet;

use crate::primitive::ClassId;
use crate::primitive::constraint::Bound;

use super::super::law::Looked;
use super::{Told, View};

/// What `lower as _` comes to: it waits until the type of `lower` has a head at
/// every depth, and then `lower` enters `upper`.
///
/// No class of that type is undecided by then, and a decided class is compared
/// with whatever it is connected to and never changed by it. So the type enters
/// as a declared one does, and nothing outside reaches into it.
pub(super) fn look(view: &View<'_>, lower: ClassId, upper: ClassId) -> Told {
    let on = without_a_type(view, lower);
    if !on.is_empty() {
        return Told::done(Looked::Waits {
            on,
            apart: BTreeSet::new(),
            pairs: Vec::new(),
        });
    }
    Told::done(Looked::Holds {
        pairs: Vec::new(),
        adjustments: Vec::new(),
        constraints: vec![Bound::Enter { lower, upper }.into()],
    })
}

/// The classes within the type of `class` that have no type yet: each with no
/// head, and each that is a part of itself, which never comes to have one.
fn without_a_type(view: &View<'_>, class: ClassId) -> BTreeSet<ClassId> {
    let classes = view.classes;
    let mut found = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut stack = vec![class];
    while let Some(class) = stack.pop() {
        let root = classes.find(class).id();
        if !seen.insert(root) {
            continue;
        }
        match view.head(root) {
            Some(_) if classes.is_own_part(root) => {
                found.insert(root);
            }
            Some(head) => stack.extend(head.places().into_iter().map(|(_, part)| *part)),
            None => {
                found.insert(root);
            }
        }
    }
    found
}
