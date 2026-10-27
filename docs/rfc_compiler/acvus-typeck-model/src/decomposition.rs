//! Decompositions: once an equivalence class takes a constructor, a relation
//! is recorded at each part of the two sides of each of its lower bounds, part
//! lower bounds and coercion constraints, as at the upper bound of the value's part: a
//! lower bound where that upper bound is an unknown, recording a part lower bound
//! between the parts of a structural type and an equation between other parts,
//! and a coercion constraint where it is not. Where the two sides' nodes differ,
//! the parts are linked by the same rule through the declared casts'
//! parameters and results; two references whose targets are of one
//! constructor, or whose target is not yet known on the side passed into,
//! relate their targets, a reborrow among them. A structure constraint
//! "indexes `τ`" on an extern-defined container is linked to the slice's
//! element along the signature of its view.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Conflict, LinkAt, StructureConstraint, StructureSource};
use crate::decl::{DeclId, Node};
use crate::join::{ERROR, is_error};
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation, Role};
use crate::sig::SigTy;
use crate::solve::State;
use crate::ty::{Constructor, Mutability, Name, PartKey, Term};

/// How a relation's value passes into its other side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Passing {
    /// By a lower bound or a coercion constraint: the value passes whole, so that two
    /// constructors are linked through the declared casts or a reborrow.
    Whole,
    /// By a part lower bound: a part passes, which takes no cast.
    Part,
}

/// A relation being decomposed as the maker of the classes that link its two
/// sides through casts: each class it makes is labelled by the two sides,
/// the cast's step and what it is at, and takes that role in the relation.
struct Linker {
    item: usize,
    from: Rc<ClassName>,
    to: Rc<ClassName>,
}

impl Linker {
    fn at(&self, step: usize, at: LinkAt) -> Making {
        Making {
            item: self.item,
            role: Role::Link {
                step,
                at: at.clone(),
            },
            label: ClassName::Link {
                from: Rc::clone(&self.from),
                to: Rc::clone(&self.to),
                step,
                at,
            },
        }
    }
}

/// What became of a relation of the record at a decomposition.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Decomposition {
    /// A lower bound, a part lower bound or a coercion constraint whose two sides are one
    /// equivalence class: marked decomposed, nothing left to relate.
    OneClass,
    /// A lower bound, a part lower bound or a coercion constraint whose two sides are
    /// solved: marked decomposed, with the relations recorded between their parts.
    Parts(Recording),
    /// A lower bound, a part lower bound or a coercion constraint whose two sides are not
    /// both solved: left as it is.
    Unsolved,
    /// No lower bound, no part lower bound and no coercion constraint: marked decomposed,
    /// having no parts to relate.
    NoParts,
}

impl State {
    /// Decomposes the relation at `i` in the record where its two sides are solved,
    /// and marks it decomposed where there is nothing left to decompose: two sides of
    /// one equivalence class, or a relation that is no lower bound, no part
    /// lower bound and no coercion constraint.
    pub(crate) fn decompose_at(&mut self, i: usize) -> Decomposition {
        let Some(relation) = self.classes.items()[i].relation().cloned() else {
            self.classes.mark_decomposed(i);
            return Decomposition::NoParts;
        };
        match relation {
            Relation::LowerBound { lower, upper } => {
                self.decompose_sides(i, lower, upper, Passing::Whole, upper)
            }
            Relation::PartLowerBound { lower, upper } => {
                self.decompose_sides(i, lower, upper, Passing::Part, upper)
            }
            Relation::CoercionConstraint { from, to } => {
                self.decompose_sides(i, from, to, Passing::Whole, from)
            }
            Relation::Wrap { from, to } => self.decompose_wrap(i, from, to),
            Relation::StructureConstraint(class, structure_constraint) => {
                self.decompose_structure_constraint(i, class, &structure_constraint)
            }
            Relation::Equation(..)
            | Relation::Structure(..)
            | Relation::Domain(..)
            | Relation::Values(..)
            | Relation::Standing(..) => {
                self.classes.mark_decomposed(i);
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
            self.classes.mark_decomposed(i);
            Decomposition::OneClass
        } else if self.decomposes_now(from, to) {
            self.classes.mark_decomposed(i);
            Decomposition::Parts(self.decompose_parts(i, from, to, passing, by))
        } else {
            Decomposition::Unsolved
        }
    }

    /// Whether the parts of a relation's two sides are related now:
    /// once both sides have taken a constructor.
    pub(crate) fn decomposes_now(&self, a: ClassId, b: ClassId) -> bool {
        self.is_solved(a) && self.is_solved(b)
    }

    /// Decomposes the structure constraint at `i` of `class` once `class` takes a
    /// constructor: where that is `!`, which has no value and satisfies every
    /// structure constraint, the part it names is `!`; where it is an
    /// extern-defined type and the structure constraint is "indexes `τ`", it is
    /// linked along the container's view; otherwise the class's structure holds
    /// the part already, and nothing is left to relate.
    fn decompose_structure_constraint(
        &mut self,
        i: usize,
        class: ClassId,
        structure_constraint: &StructureConstraint,
    ) -> Decomposition {
        let root = self.root(class);
        let Some((structure, _)) = self.classes.held(root).structure.clone() else {
            return Decomposition::Unsolved;
        };
        self.classes.mark_decomposed(i);
        match (&structure, structure_constraint) {
            (Term::Never, _) => {
                let mut recorded = Recording::default();
                let part = structure_constraint.class();
                let relation = Relation::Structure(part, Term::Never, StructureSource::Solved);
                recorded.push(self.record(relation, Recorder::Decomposition(root)));
                Decomposition::Parts(recorded)
            }
            (Term::Extern { .. }, StructureConstraint::Element(element)) => {
                Decomposition::Parts(self.link_view(i, root, *element))
            }
            _ => Decomposition::NoParts,
        }
    }

    /// Decomposes `e?`'s wrapping at `i` of the error value of the type `from`
    /// into `to`, once `from` takes a constructor: an `Error`
    /// or `!` is not wrapped and passes into `to` as it is; a type of a name
    /// `N` passes into the payload of `to`, an `Error` whose one member is
    /// `N`; and any other type into that of `to`, an `Error⟨..⟩`. Each payload
    /// is made anew, an unknown the value enters.
    fn decompose_wrap(&mut self, i: usize, from: ClassId, to: ClassId) -> Decomposition {
        let root = self.root(from);
        if !self.is_solved(root) {
            return Decomposition::Unsolved;
        }
        self.classes.mark_decomposed(i);
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
            let relation = Relation::Structure(to, structure, StructureSource::Solved);
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
    /// upper bound is a lower bound, recording a part lower bound where `of` is
    /// structural and an equation otherwise, and elsewhere it is a coercion constraint.
    fn part_relation(&self, of: &Constructor, from: ClassId, to: ClassId) -> Relation {
        if !self.is_unknown(to) {
            Relation::CoercionConstraint { from, to }
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
        i: usize,
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
        if let (Term::Ref(from_m, from_target), Term::Ref(to_m, to_target)) =
            (&from_structure, &to_structure)
        {
            // Two references: where their targets are of one constructor, or
            // the target passed into has none yet, the targets are related,
            // a reference's target being invariant; where they differ, the
            // references are linked through the casts between their nodes,
            // a reborrow taken first where an exclusive reference passes
            // into a shared one.
            let constructor_of = |state: &State, id: ClassId| {
                state
                    .classes
                    .held(state.root(id))
                    .structure
                    .as_ref()
                    .map(|(structure, _)| structure.constructor())
            };
            let (from_tc, to_tc) = (constructor_of(self, *from_target), constructor_of(self, *to_target));
            let through_casts = passing == Passing::Whole
                && matches!((&from_tc, &to_tc), (Some(a), Some(b)) if a != b);
            if !through_casts {
                let relation = self.part_relation(&to_c, *from_target, *to_target);
                recorded.push(self.record(relation, recorder));
                return recorded;
            }
            let from_m = match (from_m, to_m) {
                (Mutability::Exclusive, Mutability::Shared) => Mutability::Shared,
                (from_m, _) => *from_m,
            };
            let (Some(from_tc), Some(to_tc)) = (from_tc, to_tc) else {
                unreachable!("both targets have a constructor");
            };
            let path = self
                .decls
                .tree()
                .path(&Node::Ref(from_m, from_tc), &Node::Ref(*to_m, to_tc));
            let Some(path) = path else {
                return recorded;
            };
            return self.link_through(i, from, to, passing, by, &path, recorded);
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
        let path = self
            .decls
            .tree()
            .path(&Node::Owned(from_c.clone()), &Node::Owned(to_c.clone()));
        let Some(path) = path else {
            return recorded;
        };
        self.link_through(i, from, to, passing, by, &path, recorded)
    }

    /// Links the parts of `from` to those of `to` through the casts of
    /// `path`: at each cast, the current type's parts pass into the cast's
    /// parameter's, and the last cast's result passes on into `to`.
    #[allow(clippy::too_many_arguments)]
    fn link_through(
        &mut self,
        i: usize,
        from: ClassId,
        to: ClassId,
        passing: Passing,
        by: ClassId,
        path: &[DeclId],
        mut recorded: Recording,
    ) -> Recording {
        let recorder = Recorder::Decomposition(by);
        let linker = Linker {
            item: i,
            from: Rc::new(self.class_name(from)),
            to: Rc::new(self.class_name(to)),
        };
        let mut current = from;
        for (step, cast) in path.iter().enumerate() {
            let sig = self.decls.decl(cast).sig.clone();
            let mut vars = BTreeMap::new();
            let var = |var| linker.at(step, LinkAt::Var(var));
            let param = self.instantiate(
                &sig.params()[0],
                &mut vars,
                linker.at(step, LinkAt::Param),
                &var,
                &recorder,
                &mut recorded,
            );
            let result = self.instantiate(
                sig.result(),
                &mut vars,
                linker.at(step, LinkAt::Result),
                &var,
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
        recorded.append(self.decompose_parts(i, current, to, passing, by));
        recorded
    }

    /// Links the structure constraint "indexes `τ`" at `recorded_at`, of the
    /// extern-defined container `root`, to the slice's element along the
    /// signature of its view: the container's parts equated with the view's
    /// parameter's and `τ` with the slice's element. A container with no view
    /// cannot have the structure constraint, which is refused.
    fn link_view(&mut self, i: usize, root: ClassId, element: ClassId) -> Recording {
        let mut recorded = Recording::default();
        let recorder = Recorder::Decomposition(root);
        let Some((structure, _)) = self.classes.held(root).structure.clone() else {
            return recorded;
        };
        let constructor = structure.constructor();
        let view = self
            .decls
            .tree()
            .up(&Node::Ref(Mutability::Shared, constructor.clone()))
            .cloned()
            .filter(|cast| {
                matches!(self.decls.decl(cast).sig.result(), SigTy::Term(term)
                    if matches!(&**term, Term::Ref(_, SigTy::Term(target))
                        if matches!(&**target, Term::Slice(_))))
            });
        let Some(view) = view else {
            let cause = self.conflict_cause(Conflict::StructureConstraint(
                StructureConstraint::Element(element),
                constructor,
            ));
            self.classes.refuse(root, cause);
            return recorded;
        };
        let sig = self.decls.decl(&view).sig.clone();
        let linker = Linker {
            item: i,
            from: Rc::new(self.class_name(root)),
            to: Rc::new(self.class_name(element)),
        };
        let mut vars = BTreeMap::new();
        let var = |var| linker.at(0, LinkAt::Var(var));
        let param = self.instantiate(
            &sig.params()[0],
            &mut vars,
            linker.at(0, LinkAt::Param),
            &var,
            &recorder,
            &mut recorded,
        );
        let result = self.instantiate(
            sig.result(),
            &mut vars,
            linker.at(0, LinkAt::Result),
            &var,
            &recorder,
            &mut recorded,
        );
        let target = |state: &State, id: ClassId| match &state.classes.held(id).structure {
            Some((Term::Ref(_, target), _)) => Some(*target),
            _ => None,
        };
        let (Some(container), Some(slice)) = (target(self, param), target(self, result)) else {
            return recorded;
        };
        let (Some((container_structure, _)), Some((Term::Slice(slice_element), _))) = (
            self.classes.held(container).structure.clone(),
            self.classes.held(slice).structure.clone(),
        ) else {
            return recorded;
        };
        for ((_, part), (_, param_part)) in structure
            .parts()
            .into_iter()
            .zip(container_structure.parts())
        {
            recorded.push(self.record(Relation::Equation(*part, *param_part), recorder.clone()));
        }
        recorded.push(self.record(Relation::Equation(element, slice_element), recorder));
        recorded
    }
}
