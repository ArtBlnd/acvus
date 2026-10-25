//! The decisions that read the upper side of a relation: the class a value
//! passes into, which is the `α` of a lower bound or a part lower bound, the
//! `U` of a boundary, and the parameter and result types a use's arguments
//! and result pass into or out of.
//!
//! A class's value is the join of its sources, and a decision reads its
//! inputs: a use its arguments, a lend its place, a projection its operand, a
//! binding its path and position, a call of a function value its callee, and
//! a class its sources leave undecided its domain. Nothing else reads the upper
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
use crate::domain::Domain;
use crate::join::{le, le_part};
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Term, Ty};

/// A row of the table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Row {
    /// A class that holds no term and a domain admitting no `!`, a literal's
    /// given or carried up, takes the type of where it passes:
    /// - alternatives: the constructors its domain admits;
    /// - removal: each the value of an upper bound does not take,
    ///   by `≤` at a lower bound, `≤ₚ` at a part lower bound and a coercion
    ///   at a boundary;
    /// - preference: the one type its upper bounds have, where its
    ///   domain admits it, and its family's default otherwise;
    /// - waiting: none, a node reading the domain being decided only where one
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

/// How a class passes into an upper bound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Passing {
    Lower,
    PartLower,
    Boundary,
}

impl State {
    /// The upper bounds of the class: the `α` of each lower bound and
    /// part lower bound whose `τ` it is, and the `U` of each boundary whose
    /// `T` it is. Only a row reads them.
    fn upper_bounds(&self, root: ClassId) -> Vec<(Passing, ClassId)> {
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
    /// value still to come, and a domain admitting no `!`.
    fn literal_rows(&self) -> Vec<(ClassId, Domain)> {
        self.classes
            .roots()
            .filter(|&root| !self.has_no_value(root))
            .filter_map(|root| {
                let held = self.classes.held(root);
                let domain = held.domain.as_ref()?;
                (held.structure.is_none() && !domain.admits(&Constructor::Never))
                    .then(|| (root, domain.clone()))
            })
            .collect()
    }

    /// The literal row's removal: each class it applies to narrowed to the
    /// constructors that the value of every upper bound takes.
    /// Returns, for each class narrowed, the relation recorded.
    pub(crate) fn narrow_literals(&mut self) -> Vec<(ClassId, Recording)> {
        let mut narrowed = Vec::new();
        for (root, domain) in self.literal_rows() {
            let Domain::Only(admitted) = &domain else {
                continue;
            };
            let upper_bounds: Vec<(Passing, Ty)> = self
                .upper_bounds(root)
                .into_iter()
                .filter_map(|(passing, upper_bound)| Some((passing, self.value(upper_bound)?)))
                .collect();
            let kept: std::collections::BTreeSet<Constructor> = admitted
                .iter()
                .filter(|constructor| {
                    upper_bounds
                        .iter()
                        .all(|(passing, value)| self.takes(constructor, *passing, value))
                })
                .cloned()
                .collect();
            if kept == *admitted {
                continue;
            }
            let mut recorded = Recording::default();
            let relation = Relation::Domain(root, Domain::Only(kept));
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
    /// its upper bounds have, where its domain admits it; none where
    /// they have none or two.
    pub(crate) fn literal_preference(&self, root: ClassId) -> Option<Term<ClassId>> {
        let domain = self.classes.held(root).domain.clone()?;
        if domain.admits(&Constructor::Never) {
            return None;
        }
        let values: std::collections::BTreeSet<Ty> = self
            .upper_bounds(root)
            .into_iter()
            .filter_map(|(_, upper_bound)| self.value(upper_bound))
            .collect();
        let values: Vec<Ty> = values.into_iter().collect();
        let [one] = values.as_slice() else {
            return None;
        };
        let Term::Scalar(scalar) = one.term() else {
            return None;
        };
        domain
            .admits(&Constructor::Scalar(*scalar))
            .then_some(Term::Scalar(*scalar))
    }
}
