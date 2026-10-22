//! Splits (l.280): once an equivalence class takes a constructor, a relation
//! is recorded at each part of the two sides of each of its lower bounds, part
//! lower bounds and boundaries, as at a place the value's part passes into: a
//! meeting where that place is an unknown (l.221), recording a part lower bound
//! between the parts of a structural type and an equation between other parts,
//! and a boundary where it is not. Where the two sides' constructors differ,
//! the parts are linked by the same rule through the declared casts'
//! parameters and results, or between the targets where `&mut T` passes into
//! `&U`; and `?N ⇝ U` unifies `?N` with `U`.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName, KindSource, LinkAt};
use crate::decl::Node;
use crate::join::{ERROR, is_error};
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Mutability, Name, PartKey, Term};

/// How a relation's value passes into its other side (l.280).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Passing {
    /// By a lower bound or a boundary: the value passes whole, so that two
    /// constructors are linked through the declared casts or a reborrow.
    Whole,
    /// By a part lower bound: a part passes, which takes no cast (l.74, l.89).
    Part,
}

/// What became of a relation of the log at a split (l.280).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Split {
    /// A lower bound, a part lower bound or a boundary whose two sides are one
    /// equivalence class: marked split, nothing left to relate.
    OneClass,
    /// A lower bound, a part lower bound or a boundary whose two sides are
    /// solved: marked split, with the relations recorded between their parts.
    Parts(Recording),
    /// A lower bound, a part lower bound or a boundary whose two sides are not
    /// both solved: left as it is.
    Unsolved,
    /// No lower bound, no part lower bound and no boundary: marked split,
    /// having no parts to relate.
    NoParts,
}

impl State {
    /// Splits the relation at `i` in the log where its two sides are solved,
    /// and marks it split where there is nothing left to split: two sides of
    /// one equivalence class, or a relation that is no lower bound, no part
    /// lower bound and no boundary.
    pub(crate) fn split_at(&mut self, i: usize) -> Split {
        match self.log[i].relation {
            Relation::LowerBound { lower, upper } => {
                self.split_sides(i, lower, upper, Passing::Whole, upper)
            }
            Relation::PartLowerBound { lower, upper } => {
                self.split_sides(i, lower, upper, Passing::Part, upper)
            }
            Relation::Boundary { from, to } => self.split_sides(i, from, to, Passing::Whole, from),
            Relation::Wrap { from, to } => self.split_wrap(i, from, to),
            Relation::Equation(..)
            | Relation::Fact(..)
            | Relation::Kind(..)
            | Relation::Lit(..)
            | Relation::Reference(..)
            | Relation::Values(..) => {
                self.split.insert(i);
                Split::NoParts
            }
        }
    }

    /// Splits the relation at `i`, whose value passes from `from` into `to`,
    /// recorded as `by` taking a constructor.
    fn split_sides(
        &mut self,
        i: usize,
        from: ClassId,
        to: ClassId,
        passing: Passing,
        by: ClassId,
    ) -> Split {
        if self.root(from) == self.root(to) {
            self.split.insert(i);
            Split::OneClass
        } else if self.splits_now(from, to) {
            self.split.insert(i);
            Split::Parts(self.split_parts(from, to, passing, by))
        } else {
            Split::Unsolved
        }
    }

    /// Whether the parts of a relation's two sides are related now (l.280):
    /// once both sides have taken a constructor (`holes.md`, T2).
    pub(crate) fn splits_now(&self, a: ClassId, b: ClassId) -> bool {
        self.is_solved(a) && self.is_solved(b)
    }

    /// Splits `e?`'s wrapping at `i` of the error value of the type `from`
    /// into `to`, once `from` takes a constructor (l.280, l.339): an `Error`
    /// or `!` is not wrapped and passes into `to` as it is; a type of a name
    /// `N` passes into the payload of `to`, an `Error` whose one member is
    /// `N`; and any other type into that of `to`, an `Error⟨..⟩`. Each payload
    /// is made anew, an unknown the value meets (l.221, l.318).
    fn split_wrap(&mut self, i: usize, from: ClassId, to: ClassId) -> Split {
        let root = self.root(from);
        if !self.is_solved(root) {
            return Split::Unsolved;
        }
        self.split.insert(i);
        let recorder = Recorder::Split(root);
        let mut recorded = Recording::default();
        let (term, _) = self
            .classes
            .held(root)
            .kind
            .clone()
            .expect("a solved class holds a kind");
        let wrapped = Rc::new(self.class_name(to));
        let into = if is_error(&term) || matches!(term, Term::Never) {
            to
        } else {
            let named = match &term {
                Term::Enum(name, _) | Term::Struct(name, _) | Term::Extern { name, .. } => {
                    Some(name.clone())
                }
                _ => None,
            };
            let key = match &named {
                Some(name) => PartKey::Payload(name.clone()),
                None => PartKey::Wrapped,
            };
            let payload = self.fresh(ClassName::Part(wrapped, key));
            let kind = match named {
                Some(name) => Term::Enum(Name::new(ERROR), [(name, payload)].into()),
                None => Term::ErrorOf(payload),
            };
            let relation = Relation::Kind(to, kind, KindSource::Written);
            recorded.push(self.record(relation, recorder.clone()));
            payload
        };
        let relation = Relation::LowerBound {
            lower: from,
            upper: into,
        };
        recorded.push(self.record(relation, recorder));
        Split::Parts(recorded)
    }

    /// `?N ⇝ U` unifies `?N` with `U` (l.280): where the equivalence class
    /// holds a LitVar and no kind, and only LitVars enter it, no solved lower
    /// bound, part lower bound or kind known in part giving it a constructor
    /// (l.284), it is unified with each type it passes into by a boundary.
    /// Returns the equations recorded.
    pub(crate) fn unify_lit_boundaries(&mut self, root: ClassId) -> Recording {
        let mut recorded = Recording::default();
        let sources = self.sources(root);
        let lits_alone =
            sources.lowers.is_empty() && sources.part_lowers.is_empty() && sources.facts.is_empty();
        if self.is_lit(root) && lits_alone {
            for to in self.boundaries_from(root) {
                let relation = Relation::Equation(root, to);
                recorded.push(self.record(relation, Recorder::Split(root)));
            }
        }
        recorded
    }

    /// The relation recorded where the part `from` passes into the part `to`
    /// of a type of constructor `of` (l.280): where `to` is an unknown the
    /// place is a meeting (l.221), recording a part lower bound where `of` is
    /// structural and an equation otherwise, and elsewhere it is a boundary.
    fn part_relation(&self, of: &Constructor, from: ClassId, to: ClassId) -> Relation {
        if !self.is_unknown(to) {
            Relation::Boundary { from, to }
        } else if of.is_structural() {
            Relation::PartLowerBound {
                lower: from,
                upper: to,
            }
        } else {
            Relation::Equation(from, to)
        }
    }

    /// Records the relations between the parts of a relation's two sides
    /// (l.280). Where their constructors differ and the value passes whole,
    /// the parts are linked through the declared casts' parameters and
    /// results, or between the targets of a reborrow; a part lower bound's
    /// sides of two constructors relate nothing (l.89). Where a function value
    /// passes whole into a function type, an instance of each lambda its
    /// Values names links them, as though the type called it (l.199). Returns
    /// the relations recorded.
    fn split_parts(
        &mut self,
        from: ClassId,
        to: ClassId,
        passing: Passing,
        by: ClassId,
    ) -> Recording {
        let mut recorded = Recording::default();
        let recorder = Recorder::Split(by);
        let (Some((from_kind, _)), Some((to_kind, _))) = (
            self.classes.held(from).kind.clone(),
            self.classes.held(to).kind.clone(),
        ) else {
            return recorded;
        };
        let (from_c, to_c) = (from_kind.constructor(), to_kind.constructor());
        if let (Term::FunctionValue { .. }, Term::Function { params, ret }) = (&from_kind, &to_kind)
        {
            if passing == Passing::Part {
                return recorded;
            }
            let recorder = Recorder::FunctionType { from, to };
            for lambda in self.values_of(from) {
                let at = Name::new(&format!(
                    "{} ~> {}#{lambda}",
                    self.arena.name(from),
                    self.arena.name(to)
                ));
                recorded.append(self.instantiate_lambda(&at, &recorder, &lambda, params, *ret));
            }
            return recorded;
        }
        if from_c == to_c {
            for (key, from_part) in from_kind.parts() {
                let Some(&to_part) = to_kind.part(&key) else {
                    continue;
                };
                let relation = self.part_relation(&to_c, *from_part, to_part);
                recorded.push(self.record(relation, recorder.clone()));
            }
            return recorded;
        }
        if passing == Passing::Part {
            return recorded;
        }
        if let (
            Term::Ref(Mutability::Exclusive, from_target),
            Term::Ref(Mutability::Shared, to_target),
        ) = (&from_kind, &to_kind)
        {
            let relation = self.part_relation(&to_c, *from_target, *to_target);
            recorded.push(self.record(relation, recorder));
            return recorded;
        }
        let path = self
            .decls
            .tree()
            .path(&Node::Owned(from_c.clone()), &Node::Owned(to_c.clone()));
        let Some(path) = path else {
            return recorded;
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
                &mut recorded,
            );
            let result = self.instantiate(
                sig.result(),
                &mut vars,
                link(LinkAt::Result),
                &var_name,
                &recorder,
                &mut recorded,
            );
            let (Some((current_kind, _)), Some((param_kind, _))) = (
                self.classes.held(current).kind.clone(),
                self.classes.held(param).kind.clone(),
            ) else {
                return recorded;
            };
            let of = param_kind.constructor();
            for ((_, current_part), (_, param_part)) in
                current_kind.parts().into_iter().zip(param_kind.parts())
            {
                let relation = self.part_relation(&of, *current_part, *param_part);
                recorded.push(self.record(relation, recorder.clone()));
            }
            current = result;
        }
        recorded.append(self.split_parts(current, to, passing, by));
        recorded
    }
}
