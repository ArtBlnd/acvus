//! The decisions that read the upper side of a relation: the class a value
//! passes into, which is the `α` of a lower bound or a part lower bound, the
//! `U` of a boundary, and the parameter and result types a use's arguments
//! and result pass into or out of.
//!
//! A class's value is the join of its sources, and a decision reads its
//! inputs: a use its arguments, a lend its place, a projection its operand, a
//! binding its path and position, a call of a function value its callee, and
//! a class its sources leave undecided its range. Nothing else reads the upper
//! side, so information from it reaches a typing only through a row of this
//! table, and a row states all it does there:
//!
//! - the alternatives it adds to a decision;
//! - the alternatives it removes: those under which what follows by the
//!   sources alone fails a check whatever the other decisions become;
//! - the alternative it prefers once nothing is left to remove;
//! - what it makes wait for the nodes that record the upper side it reads.
//!
//! Without a row, a use whose candidates differ only in their results is not
//! decided, and a class nothing enters falls back.

use crate::class::ClassId;
use crate::join::{le, le_part};
use crate::range::Range;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Term, Ty};

/// A row of the table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Row {
    /// A class that holds no term and a range admitting no `!`, a literal's
    /// given or carried up, takes the type of where it passes:
    /// - alternatives: the constructors its range admits;
    /// - removal: each the value of a place it passes into does not take,
    ///   by `≤` at a lower bound, `≤ₚ` at a part lower bound and a coercion
    ///   at a boundary;
    /// - preference: the one type the places it passes into have, where its
    ///   range admits it, and its family's default otherwise;
    /// - waiting: none, a node reading the range being decided only where one
    ///   of its candidates is left, which a removal keeps or refuses.
    Literal,
}

impl Row {
    /// Every row of the table, in order.
    pub const ALL: [Row; 1] = [Row::Literal];

    pub fn name(self) -> &'static str {
        match self {
            Row::Literal => "literal",
        }
    }
}

/// How a class passes into a place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Passing {
    Lower,
    PartLower,
    Boundary,
}

impl State {
    /// The places the class passes into: the `α` of each lower bound and
    /// part lower bound whose `τ` it is, and the `U` of each boundary whose
    /// `T` it is. Only a row reads them.
    fn places(&self, root: ClassId) -> Vec<(Passing, ClassId)> {
        let root = self.root(root);
        self.log
            .iter()
            .filter_map(|recorded| match recorded.relation {
                Relation::LowerBound { lower, upper }
                    if self.root(lower) == root && self.root(upper) != root =>
                {
                    Some((Passing::Lower, upper))
                }
                Relation::PartLowerBound { lower, upper }
                    if self.root(lower) == root && self.root(upper) != root =>
                {
                    Some((Passing::PartLower, upper))
                }
                Relation::Boundary { from, to } if self.root(from) == root && self.root(to) != root => {
                    Some((Passing::Boundary, to))
                }
                _ => None,
            })
            .collect()
    }

    /// The classes the literal row applies to: those holding no term, with a
    /// value still to come, and a range admitting no `!`.
    fn literal_rows(&self) -> Vec<(ClassId, Range)> {
        self.classes
            .roots()
            .filter(|&root| !self.has_no_value(root))
            .filter_map(|root| {
                let held = self.classes.held(root);
                let range = held.range.as_ref()?;
                (held.kind.is_none() && !range.admits(&Constructor::Never))
                    .then(|| (root, range.clone()))
            })
            .collect()
    }

    /// The literal row's removal: each class it applies to narrowed to the
    /// constructors that the value of every place it passes into takes.
    /// Returns, for each class narrowed, the relation recorded.
    pub(crate) fn narrow_literals(&mut self) -> Vec<(ClassId, Recording)> {
        let mut narrowed = Vec::new();
        for (root, range) in self.literal_rows() {
            let Range::Only(admitted) = &range else {
                continue;
            };
            let places: Vec<(Passing, Ty)> = self
                .places(root)
                .into_iter()
                .filter_map(|(passing, place)| Some((passing, self.value(place)?)))
                .collect();
            let kept: std::collections::BTreeSet<Constructor> = admitted
                .iter()
                .filter(|constructor| {
                    places
                        .iter()
                        .all(|(passing, value)| self.takes(constructor, *passing, value))
                })
                .cloned()
                .collect();
            if kept == *admitted {
                continue;
            }
            let mut recorded = Recording::default();
            let relation = Relation::Range(root, Range::Only(kept));
            recorded.push(self.record(relation, Recorder::Row(Row::Literal)));
            narrowed.push((root, recorded));
        }
        narrowed
    }

    /// Whether a value of the scalar `constructor` reaches `value` by the
    /// passing.
    fn takes(&self, constructor: &Constructor, passing: Passing, value: &Ty) -> bool {
        let Constructor::Scalar(scalar) = constructor else {
            return false;
        };
        let Ok(given) = Ty::new(Term::Scalar(*scalar)) else {
            return false;
        };
        match passing {
            Passing::Lower => le(&self.decls, &given, value),
            Passing::PartLower => le_part(&given, value),
            Passing::Boundary => crate::coerce::coerce(&self.decls, &given, value).is_ok(),
        }
    }

    /// The literal row's preference for a class it applies to: the one type
    /// the places it passes into have, where its range admits it; none where
    /// they have none or two.
    pub(crate) fn literal_preference(&self, root: ClassId) -> Option<Term<ClassId>> {
        let range = self.classes.held(root).range.clone()?;
        if range.admits(&Constructor::Never) {
            return None;
        }
        let values: std::collections::BTreeSet<Ty> = self
            .places(root)
            .into_iter()
            .filter_map(|(_, place)| self.value(place))
            .collect();
        let values: Vec<Ty> = values.into_iter().collect();
        let [one] = values.as_slice() else {
            return None;
        };
        let Term::Scalar(scalar) = one.term() else {
            return None;
        };
        range
            .admits(&Constructor::Scalar(*scalar))
            .then_some(Term::Scalar(*scalar))
    }
}
