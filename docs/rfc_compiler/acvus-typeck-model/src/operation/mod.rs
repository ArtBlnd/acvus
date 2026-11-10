//! Solving: three operations over the store, and the law they ask.
//!
//! | Operation | What it does | The one place it is done |
//! |---|---|---|
//! | connecting | makes two classes one type | [`connect::connect`] |
//! | looking at a bound | judges a bound between two classes, or says what it waits for | [`judge::look`] |
//! | deciding | gives an undecided class its head | [`decide::decide`] |
//!
//! [`settle::settle`] takes them in turn until nothing is left to do. None of them
//! knows a rule of the language: each asks a [`law::Law`], which says what a bound
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
mod testing;
#[cfg(test)]
mod tests;

use crate::primitive::adjustment::Adjustments;
use crate::primitive::class::{ClassKind, Pending};
use crate::primitive::refusal::Refusal;

/// What solving comes to besides the classes themselves.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Outcome<K>
where
    K: ClassKind,
{
    pub refusals: Vec<Refusal>,
    pub adjustments: Adjustments,
    /// The calls on classes decided, which no law judges yet.
    pub calls: Vec<Pending<K>>,
}

impl<K> Outcome<K>
where
    K: ClassKind,
{
    /// Records `refusal`, unless the same refusal stands already. What stays as
    /// it is gives the same one each time it is looked at.
    pub fn refuse(&mut self, refusal: Refusal) {
        if !self.refusals.contains(&refusal) {
            self.refusals.push(refusal);
        }
    }
}
