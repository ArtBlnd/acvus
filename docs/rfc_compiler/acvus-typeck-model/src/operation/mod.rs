//! Solving: three operations over the store, and the law they ask.
//!
//! | Operation | What it does | The one place it is done |
//! |---|---|---|
//! | connecting | makes two classes one type | [`connect::connect`] |
//! | looking at a constraint | judges a constraint between two classes, or says what it waits for | [`judge::look`] |
//! | deciding | gives an undecided class its head | [`decide::decide`] |
//!
//! [`settle::settle`] takes them in turn until nothing is left to do. None of them
//! knows a rule of the language: each asks a [`law::Law`], which says what a constraint
//! comes to, what head a class may take, and where a named part lies. The
//! language's own rules are [`language::Language`], and any other law that keeps
//! the contracts of [`law`] runs as well.

pub mod connect;
pub mod decide;
pub mod judge;
pub mod language;
pub mod law;
pub mod settle;

#[cfg(test)]
mod calls;
#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;

use crate::primitive::adjustment::Adjustments;
use std::collections::BTreeSet;

use crate::primitive::class::{ClassKind, Pending, State};
use crate::primitive::refusal::{Refusal, Refused};
use crate::store::classes::Classes;

/// What solving comes to besides the classes themselves.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Outcome<K>
where
    K: ClassKind,
{
    /// The refusals, each with what settled the classes it is of. They are
    /// gathered when settling ends ([`Outcome::gather`]).
    pub refusals: Vec<Refusal>,
    /// What has been refused and not gathered yet.
    pub refused: Vec<Refused<K>>,
    pub adjustments: Adjustments,
    /// The calls on classes decided, which no law judges yet.
    pub calls: Vec<Pending<K>>,
}

impl<K> Outcome<K>
where
    K: ClassKind,
{
    /// Records what is refused, unless the same stands already. What stays as it
    /// is gives the same one each time it is looked at.
    pub fn refuse(&mut self, refused: Refused<K>) {
        if !self.refused.contains(&refused) {
            self.refused.push(refused);
        }
    }

    /// Makes a refusal of each thing refused, with what settled the classes it is
    /// of: what decided the head of each, and every connection that made it.
    ///
    /// This is the only place that is read, and it is read once nothing more is
    /// to be connected or decided. So a refusal names everything that met,
    /// whichever of it came first, and no rule sees what a refusal names.
    pub fn gather(&mut self, classes: &Classes<K>) {
        for refused in std::mem::take(&mut self.refused) {
            let mut items = refused.items;
            let mut connections = BTreeSet::new();
            for class in refused.of {
                if let State::Decided(decided) = &classes.get(classes.find(class)).state {
                    items.extend(&decided.by);
                }
                connections.extend(classes.connections(class).into_iter().map(|(_, _, at)| at));
            }
            let refusal = Refusal {
                cause: refused.cause,
                items,
                connections,
            };
            if !self.refusals.contains(&refusal) {
                self.refusals.push(refusal);
            }
        }
    }
}
