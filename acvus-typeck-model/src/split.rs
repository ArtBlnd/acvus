//! Splits (l.284–291): once an equivalence class takes a constructor, the
//! relations between the parts of its lower bounds and boundaries are recorded
//! in the parts' equivalence classes, by the table of l.286, through the
//! signatures of the declared casts where the two sides' constructors differ;
//! and `?N ⇝ U` unifies `?N` with `U`.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName, LinkAt};
use crate::decl::Node;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Mutability, Term, Variance};

/// How a relation's two sides are related part by part (l.286).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Lower,
    Boundary,
}

impl State {
    /// Splits the relation at `i` in the log where its two sides are solved,
    /// and marks it split where there is nothing left to split: two sides of
    /// one equivalence class, or a relation that is no lower bound and no
    /// boundary. Returns whether it is marked.
    pub(crate) fn split_at(&mut self, i: usize) -> bool {
        match self.log[i].relation.clone() {
            Relation::LowerBound { lower, upper } => {
                if self.root(lower) == self.root(upper) {
                    self.split.insert(i);
                } else if self.splits_now(lower, upper) {
                    self.split.insert(i);
                    self.split_parts(lower, upper, Side::Lower, upper);
                } else {
                    return false;
                }
            }
            Relation::Boundary { from, to } => {
                if self.root(from) == self.root(to) {
                    self.split.insert(i);
                } else if self.splits_now(from, to) {
                    self.split.insert(i);
                    self.split_parts(from, to, Side::Boundary, from);
                } else {
                    return false;
                }
            }
            Relation::Equation(..)
            | Relation::Fact(..)
            | Relation::Kind(..)
            | Relation::Lit(..)
            | Relation::Reference(..)
            | Relation::Values(..) => {
                self.split.insert(i);
            }
        }
        true
    }

    /// Whether the parts of a lower bound's or a boundary's two sides are
    /// related now (l.284): once both sides have taken a constructor
    /// (`holes.md`, T2).
    fn splits_now(&self, a: ClassId, b: ClassId) -> bool {
        self.is_solved(a) && self.is_solved(b)
    }

    /// `?N ⇝ U` unifies `?N` with `U` (l.291): where the equivalence class
    /// holds a LitVar and no kind, it is unified with each type it passes into
    /// by a boundary.
    pub(crate) fn unify_lit_boundaries(&mut self, root: ClassId) {
        if self.is_lit(root) {
            for to in self.boundaries_from(root) {
                self.record(Relation::Equation(root, to), Recorder::Split(root));
            }
        }
    }

    /// Records the relations between the parts of a lower bound's or a
    /// boundary's two sides (l.286), linking the parts through the declared
    /// casts where their constructors differ (l.291).
    pub(crate) fn split_parts(&mut self, from: ClassId, to: ClassId, side: Side, by: ClassId) {
        let recorder = Recorder::Split(by);
        let (Some((from_kind, _)), Some((to_kind, _))) = (
            self.classes.held(from).kind.clone(),
            self.classes.held(to).kind.clone(),
        ) else {
            return;
        };
        let (from_c, to_c) = (from_kind.constructor(), to_kind.constructor());
        if from_c == to_c {
            let variance = from_c.variance();
            for (key, from_part) in from_kind.parts() {
                let Some(&to_part) = to_kind.part(&key) else {
                    continue;
                };
                let relation = match (variance, side) {
                    (Variance::Invariant, _) => Relation::Equation(*from_part, to_part),
                    (Variance::Covariant, Side::Lower) => Relation::LowerBound {
                        lower: *from_part,
                        upper: to_part,
                    },
                    (Variance::Covariant, Side::Boundary) if self.is_unknown(to_part) => {
                        Relation::LowerBound {
                            lower: *from_part,
                            upper: to_part,
                        }
                    }
                    (Variance::Covariant, Side::Boundary) => Relation::Boundary {
                        from: *from_part,
                        to: to_part,
                    },
                };
                self.record(relation, recorder.clone());
            }
            return;
        }
        if let (
            Term::Ref(Mutability::Exclusive, from_target),
            Term::Ref(Mutability::Shared, to_target),
        ) = (&from_kind, &to_kind)
        {
            self.record(Relation::Equation(*from_target, *to_target), recorder);
            return;
        }
        let path = self
            .decls
            .tree()
            .path(&Node::Owned(from_c.clone()), &Node::Owned(to_c.clone()));
        let Some(path) = path else {
            return;
        };
        let (from_name, to_name) = (Rc::new(self.class_name(from)), Rc::new(self.class_name(to)));
        let mut current = from;
        for (step, cast) in path.iter().enumerate() {
            let sig = self.decls.decl(cast).sig.clone();
            let link = |at: LinkAt| ClassName::Link {
                from: from_name.clone(),
                to: to_name.clone(),
                step,
                at,
            };
            let mut vars = BTreeMap::new();
            let var_name = |var| link(LinkAt::Var(var));
            let param = self.instantiate(
                &sig.params()[0],
                &mut vars,
                link(LinkAt::Param),
                &var_name,
                &recorder,
            );
            let result = self.instantiate(
                sig.result(),
                &mut vars,
                link(LinkAt::Result),
                &var_name,
                &recorder,
            );
            let (Some((current_kind, _)), Some((param_kind, _))) = (
                self.classes.held(current).kind.clone(),
                self.classes.held(param).kind.clone(),
            ) else {
                return;
            };
            for ((_, current_part), (_, param_part)) in
                current_kind.parts().into_iter().zip(param_kind.parts())
            {
                self.record(
                    Relation::Equation(*current_part, *param_part),
                    recorder.clone(),
                );
            }
            current = result;
        }
        self.split_parts(current, to, side, by);
    }
}
