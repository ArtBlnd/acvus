//! Relations and what records them.
//!
//! Every relation is kept in a log with what recorded it, and the
//! equivalence classes are what the log makes of them. So the value a use
//! reads can be decided from the remaining relations alone.

use std::fmt;

use crate::class::{
    Arena, ClassId, NamedStructureConstraint, StructureConstraint, StructureSource,
};
use crate::domain::Domain;
use crate::ty::{Name, Term};

/// A relation recorded in equivalence classes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Relation {
    /// `τ ≤ α`, which a lower bound records, and a use's result type
    /// entering its result.
    LowerBound { lower: ClassId, upper: ClassId },
    /// `τ ≤ₚ α`, which a decomposition records between the parts of a structural type
    /// where the upper bound of the value's part is a lower bound.
    PartLowerBound { lower: ClassId, upper: ClassId },
    /// `T ⇝ U`, which a coercion constraint records.
    CoercionConstraint { from: ClassId, to: ClassId },
    /// An equation, which an invariant place records.
    Equation(ClassId, ClassId),
    /// A structure constraint, which a projection records.
    StructureConstraint(ClassId, StructureConstraint),
    /// A structure the class holds: a type the text or a declaration writes,
    /// or, in a copy of C, the structure the class held where its binding
    /// group was solved.
    Structure(ClassId, Term<ClassId>, StructureSource),
    /// The constructors the class's value may take, which a literal, a disjunctive constraint's
    /// generalization or a lend gives before the class takes one.
    Domain(ClassId, Domain),
    /// A lambda the Values of the class's function value names.
    Values(ClassId, Name),
    /// `e?`'s error value of the type `from` wrapped into `to`, which a decomposition
    /// relates once `from` takes a constructor.
    Wrap { from: ClassId, to: ClassId },
}

impl Relation {
    /// The classes the relation names, a structure's parts among them.
    pub fn classes(&self) -> Vec<ClassId> {
        match self {
            Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper } => {
                vec![*lower, *upper]
            }
            Relation::CoercionConstraint { from, to } | Relation::Wrap { from, to } => {
                vec![*from, *to]
            }
            Relation::Equation(a, b) => vec![*a, *b],
            Relation::StructureConstraint(class, structure_constraint) => {
                vec![*class, structure_constraint.class()]
            }
            Relation::Structure(class, term, _) => std::iter::once(*class)
                .chain(term.parts().into_iter().map(|(_, part)| *part))
                .collect(),
            Relation::Domain(class, _)
            | Relation::Values(class, _) => vec![*class],
        }
    }

    /// The relation with each class it names put through `f`.
    pub fn map<F>(self, mut f: F) -> Relation
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            Relation::LowerBound { lower, upper } => Relation::LowerBound {
                lower: f(lower),
                upper: f(upper),
            },
            Relation::PartLowerBound { lower, upper } => Relation::PartLowerBound {
                lower: f(lower),
                upper: f(upper),
            },
            Relation::CoercionConstraint { from, to } => Relation::CoercionConstraint {
                from: f(from),
                to: f(to),
            },
            Relation::Equation(a, b) => {
                let a = f(a);
                Relation::Equation(a, f(b))
            }
            Relation::StructureConstraint(class, structure_constraint) => {
                let class = f(class);
                Relation::StructureConstraint(class, structure_constraint.map(f))
            }
            Relation::Structure(class, term, source) => {
                let class = f(class);
                Relation::Structure(class, term.map(|_, part| f(part)), source)
            }
            Relation::Domain(class, domain) => Relation::Domain(f(class), domain),
            Relation::Values(class, lambda) => Relation::Values(f(class), lambda),
            Relation::Wrap { from, to } => Relation::Wrap {
                from: f(from),
                to: f(to),
            },
        }
    }
}

/// What recorded a relation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Recorder {
    /// The type the text gives a class.
    Origin,
    /// The body's text, at the item of the program at this index.
    Text(usize),
    /// The use, lend, projection or binding of this name, when it was
    /// decided.
    Disjunction(Name),
    /// The use, lend, projection or binding of this name before it was
    /// decided: the generalization of all it may become, which holds whatever
    /// it is decided as.
    Generalization(Name),
    /// The equivalence class of this class taking a constructor.
    Decomposition(ClassId),
    /// A cycle of lower bounds alone, merged into one equivalence class.
    Cycle,
    /// The function value `from` passing into the function type `to`, whose
    /// decomposition makes an instance of each lambda its Values names, as though the
    /// function type called it.
    FunctionType { from: ClassId, to: ClassId },
    /// The I\* check, making an instance of a lambda's C that no call made, in
    /// a view of a state it decides disjunctive constraints again in. Solving never records it.
    Check,
    /// A propagator, which reads where a value passes.
    Propagator(crate::propagator::Propagator),
    /// Fallback, once no step is left: the value it gives a class no step
    /// solves, a decision no other relation derives, which every read after
    /// it takes as given.
    Fallback,
    /// The instance `at`, made by `by`, the use of a function at its item or
    /// the call of a function value, recording in its binding group
    /// the copy of what `of` recorded in C.
    Instance {
        at: Name,
        by: Box<Recorder>,
        of: Box<Recorder>,
    },
}

impl Recorder {
    /// Whether solving derives it from the other relations, so that a value
    /// read derives it again from the relations that remain.
    pub fn is_derived(&self) -> bool {
        match self {
            Recorder::Decomposition(_)
            | Recorder::Cycle
            | Recorder::FunctionType { .. }
            | Recorder::Propagator(_) => true,
            Recorder::Instance { by, .. } => by.is_derived(),
            Recorder::Origin
            | Recorder::Text(_)
            | Recorder::Disjunction(_)
            | Recorder::Generalization(_)
            | Recorder::Check
            | Recorder::Fallback => false,
        }
    }

    /// The disjunctive constraints it is recorded by, a value read leaving any of them out
    /// leaving it out: a disjunctive constraint's own relation names that disjunctive constraint, and a
    /// copy in an instance names what made the instance and, by `copy_name`,
    /// the disjunctive constraints the relation it copies is recorded by. A disjunctive constraint's generalization
    /// names none: it is not recorded by the disjunctive constraint's being decided, so no
    /// read leaves it out.
    pub fn disjunctions(&self) -> Vec<Name> {
        match self {
            Recorder::Disjunction(name) => vec![name.clone()],
            Recorder::Instance { at, by, of } => by
                .disjunctions()
                .into_iter()
                .chain(
                    of.disjunctions()
                        .iter()
                        .map(|disjunction| copy_name(at, disjunction)),
                )
                .collect(),
            Recorder::Origin
            | Recorder::Text(_)
            | Recorder::Generalization(_)
            | Recorder::Decomposition(_)
            | Recorder::Cycle
            | Recorder::FunctionType { .. }
            | Recorder::Propagator(_)
            | Recorder::Check
            | Recorder::Fallback => Vec::new(),
        }
    }

    /// Whether the text, a disjunctive constraint, or a function value passing into a function
    /// type records it, so that a value it makes enter a type is checked,
    /// in a copy of C as where it was recorded.
    pub fn is_entering(&self) -> bool {
        match self {
            Recorder::Text(_)
            | Recorder::Disjunction(_)
            | Recorder::Generalization(_)
            | Recorder::FunctionType { .. } => true,
            Recorder::Instance { of, .. } => of.is_entering(),
            Recorder::Origin
            | Recorder::Decomposition(_)
            | Recorder::Cycle
            | Recorder::Propagator(_)
            | Recorder::Check
            | Recorder::Fallback => false,
        }
    }
}

/// The name, in the binding group an instance `at` is recorded in, of the
/// disjunctive constraint `disjunction` of the group instantiated.
pub fn copy_name(at: &Name, disjunction: &Name) -> Name {
    Name::new(&format!("{at}/{disjunction}"))
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Recorded {
    pub relation: Relation,
    pub recorder: Recorder,
}

/// A relation written with its classes' names.
pub struct Named<'a> {
    pub relation: &'a Relation,
    pub arena: &'a Arena,
}

impl fmt::Display for Named<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |id: &ClassId| self.arena.name(*id);
        match self.relation {
            Relation::LowerBound { lower, upper } => {
                write!(f, "{} <= {}", name(lower), name(upper))
            }
            Relation::PartLowerBound { lower, upper } => {
                write!(f, "{} <=p {}", name(lower), name(upper))
            }
            Relation::CoercionConstraint { from, to } => {
                write!(f, "{} ~> {}", name(from), name(to))
            }
            Relation::Equation(a, b) => write!(f, "{} = {}", name(a), name(b)),
            Relation::StructureConstraint(class, structure_constraint) => {
                let structure_constraint = NamedStructureConstraint {
                    structure_constraint,
                    arena: self.arena,
                };
                write!(f, "{}: {structure_constraint}", name(class))
            }
            Relation::Structure(class, term, _) => {
                let term = term.clone().map(|_, part| name(&part).clone());
                write!(f, "{}: {term}", name(class))
            }
            Relation::Domain(class, domain) => write!(f, "{}: {domain}", name(class)),
            Relation::Values(class, lambda) => {
                write!(f, "{}: Values contains {lambda}", name(class))
            }
            Relation::Wrap { from, to } => write!(f, "wrap {} into {}", name(from), name(to)),
        }
    }
}
