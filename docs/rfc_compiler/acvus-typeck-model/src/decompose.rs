//! Decompositions: once an equivalence class takes a constructor, a relation
//! is recorded at each part of the two sides of each of its lower bounds, part
//! lower bounds and boundaries, as at the upper bound of the value's part: a
//! meeting where that upper bound is an unknown, recording a part lower bound
//! between the parts of a structural type and an equation between other parts,
//! and a boundary where it is not. Where the two sides' constructors differ,
//! the parts are linked by the same rule through the declared casts'
//! parameters and results, or between the targets where `&mut T` passes into
//! `&U`.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName, LinkAt, StructureSource};
use crate::decl::Node;
use crate::join::{ERROR, is_error};
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Mutability, Name, PartKey, Term};

/// How a relation's value passes into its other side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Passing {
    /// By a lower bound or a boundary: the value passes whole, so that two
    /// constructors are linked through the declared casts or a reborrow.
    Whole,
    /// By a part lower bound: a part passes, which takes no cast.
    Part,
}

/// What became of a relation of the log at a decomposition.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Decomposition {
    /// A lower bound, a part lower bound or a boundary whose two sides are one
    /// equivalence class: marked decomposed, nothing left to relate.
    OneClass,
    /// A lower bound, a part lower bound or a boundary whose two sides are
    /// solved: marked decomposed, with the relations recorded between their parts.
    Parts(Recording),
    /// A lower bound, a part lower bound or a boundary whose two sides are not
    /// both solved: left as it is.
    Unsolved,
    /// No lower bound, no part lower bound and no boundary: marked decomposed,
    /// having no parts to relate.
    NoParts,
}

impl State {
    /// Decomposes the relation at `i` in the log where its two sides are solved,
    /// and marks it decomposed where there is nothing left to decompose: two sides of
    /// one equivalence class, or a relation that is no lower bound, no part
    /// lower bound and no boundary.
    pub(crate) fn decompose_at(&mut self, i: usize) -> Decomposition {
        match self.log[i].relation {
            Relation::LowerBound { lower, upper } => {
                self.decompose_sides(i, lower, upper, Passing::Whole, upper)
            }
            Relation::PartLowerBound { lower, upper } => {
                self.decompose_sides(i, lower, upper, Passing::Part, upper)
            }
            Relation::Boundary { from, to } => {
                self.decompose_sides(i, from, to, Passing::Whole, from)
            }
            Relation::Wrap { from, to } => self.decompose_wrap(i, from, to),
            Relation::StructureConstraint(class, ref structure_constraint) => {
                let part = structure_constraint.class();
                self.decompose_structure_constraint(i, class, part)
            }
            Relation::Equation(..)
            | Relation::Structure(..)
            | Relation::Domain(..)
            | Relation::Values(..) => {
                self.decomposed.insert(i);
                Decomposition::NoParts
            }
        }
    }

    /// Decomposes the relation at `i`, whose value passes from `from` into `to`,
    /// recorded as `by` taking a constructor.
    fn decompose_sides(
        &mut self,
        i: usize,
        from: ClassId,
        to: ClassId,
        passing: Passing,
        by: ClassId,
    ) -> Decomposition {
        if self.root(from) == self.root(to) {
            self.decomposed.insert(i);
            Decomposition::OneClass
        } else if self.decomposes_now(from, to) {
            self.decomposed.insert(i);
            Decomposition::Parts(self.decompose_parts(from, to, passing, by))
        } else {
            Decomposition::Unsolved
        }
    }

    /// Whether the parts of a relation's two sides are related now:
    /// once both sides have taken a constructor.
    pub(crate) fn decomposes_now(&self, a: ClassId, b: ClassId) -> bool {
        self.is_solved(a) && self.is_solved(b)
    }

    /// Decomposes the structure constraint at `i` of `class`, naming `part`, once
    /// `class` takes a constructor: where that is `!`, which has no value and
    /// meets every structure constraint, the part it names is `!`; otherwise the
    /// class's structure holds the part already, and nothing is left to relate.
    fn decompose_structure_constraint(
        &mut self,
        i: usize,
        class: ClassId,
        part: ClassId,
    ) -> Decomposition {
        let root = self.root(class);
        if !self.is_solved(root) {
            return Decomposition::Unsolved;
        }
        self.decomposed.insert(i);
        if !matches!(self.classes.held(root).structure, Some((Term::Never, _))) {
            return Decomposition::NoParts;
        }
        let mut recorded = Recording::default();
        let relation = Relation::Structure(part, Term::Never, StructureSource::Written);
        recorded.push(self.record(relation, Recorder::Decomposition(root)));
        Decomposition::Parts(recorded)
    }

    /// Decomposes `e?`'s wrapping at `i` of the error value of the type `from`
    /// into `to`, once `from` takes a constructor: an `Error`
    /// or `!` is not wrapped and passes into `to` as it is; a type of a name
    /// `N` passes into the payload of `to`, an `Error` whose one member is
    /// `N`; and any other type into that of `to`, an `Error⟨..⟩`. Each payload
    /// is made anew, an unknown the value meets.
    fn decompose_wrap(&mut self, i: usize, from: ClassId, to: ClassId) -> Decomposition {
        let root = self.root(from);
        if !self.is_solved(root) {
            return Decomposition::Unsolved;
        }
        self.decomposed.insert(i);
        let recorder = Recorder::Decomposition(root);
        let mut recorded = Recording::default();
        let (term, _) = self
            .classes
            .held(root)
            .structure
            .clone()
            .expect("a solved class holds a structure");
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
            let structure = match named {
                Some(name) => Term::Enum(Name::new(ERROR), [(name, payload)].into()),
                None => Term::ErrorOf(payload),
            };
            let relation = Relation::Structure(to, structure, StructureSource::Written);
            recorded.push(self.record(relation, recorder.clone()));
            payload
        };
        let relation = Relation::LowerBound {
            lower: from,
            upper: into,
        };
        recorded.push(self.record(relation, recorder));
        Decomposition::Parts(recorded)
    }

    /// The relation recorded where the part `from` passes into the part `to`
    /// of a type of constructor `of`: where `to` is an unknown the
    /// upper bound is a meeting, recording a part lower bound where `of` is
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

    /// Records the relations between the parts of a relation's two sides.
    /// Where their constructors differ and the value passes whole,
    /// the parts are linked through the declared casts' parameters and
    /// results, or between the targets of a reborrow; a part lower bound's
    /// sides of two constructors relate nothing. Where a function value
    /// passes whole into a function type, an instance of each lambda its
    /// Values names links them, as though the type called it. Returns
    /// the relations recorded.
    fn decompose_parts(
        &mut self,
        from: ClassId,
        to: ClassId,
        passing: Passing,
        by: ClassId,
    ) -> Recording {
        let mut recorded = Recording::default();
        let recorder = Recorder::Decomposition(by);
        let (Some((from_structure, _)), Some((to_structure, _))) = (
            self.classes.held(from).structure.clone(),
            self.classes.held(to).structure.clone(),
        ) else {
            return recorded;
        };
        let (from_c, to_c) = (from_structure.constructor(), to_structure.constructor());
        if let (Term::FunctionValue { .. }, Term::Function { params, ret }) =
            (&from_structure, &to_structure)
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
            for (key, from_part) in from_structure.parts() {
                let Some(&to_part) = to_structure.part(&key) else {
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
        ) = (&from_structure, &to_structure)
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
            let (Some((current_structure, _)), Some((param_structure, _))) = (
                self.classes.held(current).structure.clone(),
                self.classes.held(param).structure.clone(),
            ) else {
                return recorded;
            };
            let of = param_structure.constructor();
            for ((_, current_part), (_, param_part)) in current_structure
                .parts()
                .into_iter()
                .zip(param_structure.parts())
            {
                let relation = self.part_relation(&of, *current_part, *param_part);
                recorded.push(self.record(relation, recorder.clone()));
            }
            current = result;
        }
        recorded.append(self.decompose_parts(current, to, passing, by));
        recorded
    }
}
