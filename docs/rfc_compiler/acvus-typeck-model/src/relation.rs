//! Relations, disjunctive constraints, and what records them.
//!
//! Every item is kept in the record with what recorded it, each class
//! holding its occurrences in them, and the equivalence classes are what the
//! record makes of them. An item is a relation, or a disjunctive constraint:
//! one of several alternatives, each a set of relations, which the judgment
//! decides reading the record.

use std::fmt;

use std::collections::BTreeSet;

use crate::class::{
    Arena, ClassId, ClassName, LinkAt, NamedStructureConstraint, Side, StructureConstraint,
    StructureSource,
};
use crate::disjunction::{Decision, Disjunction, DisjunctionAt};
use crate::domain::Domain;
use crate::program::Position;
use crate::ty::{Name, PartKey, Term};

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
    /// A structure the class holds: one the text, a declaration or a decided
    /// alternative writes, or one solving gives it, a decomposition's,
    /// fallback's or an identity made anew.
    Structure(ClassId, Term<ClassId>, StructureSource),
    /// The constructors the class's value may take, which a literal, a disjunctive constraint's
    /// generalization or a lend gives before the class takes one.
    Domain(ClassId, Domain),
    /// A lambda the Values of the class's function value names.
    Values(ClassId, Name),
    /// `e?`'s error value of the type `from` wrapped into `to`, which a decomposition
    /// relates once `from` takes a constructor.
    Wrap { from: ClassId, to: ClassId },
    /// The standing of a class in the group: what it is to the group before
    /// anything is solved.
    Standing(ClassId, Standing),
}

/// What a class is to the binding group whose record holds it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Standing {
    /// A class the text gives, by the name the program gives it: one of the
    /// outcome's.
    Text,
    /// The type of a parameter, which the use that makes an instance
    /// chooses: no unknown of the group.
    Universal,
    /// A class of an enclosing group, shared here and not made anew by an
    /// instance; `unknown` where it is an unknown of its own group, so that a
    /// value passing into it by name enters it.
    Shared { unknown: bool },
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
            | Relation::Values(class, _)
            | Relation::Standing(class, _) => vec![*class],
        }
    }

    /// The classes occurring in the relation, each with its role in it.
    pub fn occurrences(&self) -> Vec<(ClassId, Role)> {
        match self {
            Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper } => {
                vec![(*lower, Role::Lower), (*upper, Role::Upper)]
            }
            Relation::CoercionConstraint { from, to } | Relation::Wrap { from, to } => {
                vec![(*from, Role::Lower), (*to, Role::Upper)]
            }
            Relation::Equation(a, b) => vec![(*a, Role::Equated), (*b, Role::Equated)],
            Relation::StructureConstraint(class, structure_constraint) => {
                vec![
                    (*class, Role::Own),
                    (structure_constraint.class(), Role::Part(structure_constraint.key())),
                ]
            }
            Relation::Structure(class, term, _) => std::iter::once((*class, Role::Own))
                .chain(
                    term.parts()
                        .into_iter()
                        .map(|(key, part)| (*part, Role::Part(key))),
                )
                .collect(),
            Relation::Domain(class, _)
            | Relation::Values(class, _)
            | Relation::Standing(class, _) => vec![(*class, Role::Own)],
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
            Relation::Standing(class, standing) => Relation::Standing(f(class), standing),
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
    /// The judgment of the equivalence class of this class: the structure it
    /// takes at its join or its fill.
    Join(ClassId),
    /// The equivalence class of this class taking a constructor.
    Decomposition(ClassId),
    /// A cycle of lower bounds and part lower bounds alone, merged into one
    /// equivalence class.
    Cycle,
    /// The function value `from` passing into the function type `to`, whose
    /// decomposition makes an instance of each lambda its Values names, as though the
    /// function type called it.
    FunctionType { from: ClassId, to: ClassId },
    /// The I\* check, making an instance of a lambda's C that no call made, in
    /// a view of a state it decides disjunctive constraints again in. Solving never records it.
    Check,
    /// Fallback, at the fixpoint: the default it gives a class no judgment
    /// decided, which every judgment after it takes as given.
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
    /// The disjunctive constraints it is recorded by, a judgment leaving any of them out
    /// leaving it out: a disjunctive constraint's own relation names that disjunctive constraint, and a
    /// copy in an instance names what made the instance and, by `copy_name`,
    /// the disjunctive constraints the relation it copies is recorded by. A disjunctive constraint's generalization
    /// names none: it is not recorded by the disjunctive constraint's being decided, so no
    /// judgment leaves it out.
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
            | Recorder::Join(_)
            | Recorder::Decomposition(_)
            | Recorder::Cycle
            | Recorder::FunctionType { .. }
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
            | Recorder::Join(_)
            | Recorder::Decomposition(_)
            | Recorder::Cycle
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

/// An item of the record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    /// A relation: an item of one alternative.
    Relation(Relation),
    /// A disjunctive constraint of this name: a use, a lend, a projection, a
    /// binding or a call of a function value, whose alternatives are what it
    /// may become, and whose decision is the alternative it became, none
    /// while it is undecided. What the decided alternative records is
    /// recorded as relations of its own, by `Recorder::Disjunction`, and the
    /// classes it makes anew are the item's, by the role each is made in.
    Disjunction {
        name: Name,
        kind: Disjunction,
        decision: Option<Decision>,
    },
    /// An instance named `at`, made by `by`: the C of the binding group
    /// `group` made anew here, or, of a use whose constraints two groups
    /// record, the parts of C at the positions recorded here (`parts`). Each
    /// class made anew is the item's, in the role of the copy of the class
    /// it copies, named as that group's record names it.
    Instance {
        at: Name,
        by: Recorder,
        group: Name,
        parts: Option<BTreeSet<usize>>,
    },
    /// A lambda placed in a group instantiated, made anew here under the
    /// name `name` by the instance `at`, copying the lambda `of`: the classes
    /// of its environment, its parameters and then its result, are the
    /// item's, each in the role of its place.
    Lambda { name: Name, of: Name, at: Name },
}

impl Item {
    /// The classes occurring in the item as it is recorded, each with its
    /// role in it. The classes an item makes anew afterwards are attached to
    /// it as they are made (`Classes::attach`).
    pub fn occurrences(&self) -> Vec<(ClassId, Role)> {
        match self {
            Item::Relation(relation) => relation.occurrences(),
            Item::Disjunction { kind, .. } => kind.occurrences(),
            Item::Instance { .. } | Item::Lambda { .. } => Vec::new(),
        }
    }

    /// The relation, where the item is one.
    pub fn relation(&self) -> Option<&Relation> {
        match self {
            Item::Relation(relation) => Some(relation),
            Item::Disjunction { .. } | Item::Instance { .. } | Item::Lambda { .. } => None,
        }
    }
}

/// A class's role in an item of the record. The side of it the class is,
/// which the judgment reads, is a function of the role (`Role::side`).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Role {
    /// The value passes from it: the `τ` of `τ ≤ α` and of `τ ≤ₚ α`, the `T`
    /// of `T ⇝ U`, and the error value of a wrap.
    Lower,
    /// The value passes into it: the `α`, the `U`, and the class the error
    /// value is wrapped into.
    Upper,
    /// A side of an equation.
    Equated,
    /// The relation is on it: a structure, a structure constraint, a domain,
    /// a standing, or a lambda its Values names.
    Own,
    /// The part of this key a structure writes or a structure constraint
    /// names.
    Part(PartKey),
    /// A position of a use or a call of a function value.
    Position(Position),
    /// A lend's place.
    Place,
    /// A projection's operand.
    Operand,
    /// A binding's position.
    Bound,
    /// A binding's binding.
    Binding,
    /// A call's callee.
    Callee,
    /// A class a disjunctive constraint made anew, in this role.
    Made(DisjunctionAt),
    /// A class a decomposition made anew linking a relation's two sides
    /// through the cast at `step` of the path between them.
    Link { step: usize, at: LinkAt },
    /// A class an instance made anew, copying the class of this name of the
    /// group instantiated.
    Copy(ClassName),
    /// The class at this place of a lambda's environment: its parameter `i`,
    /// or, past the parameters, its result.
    Environment(usize),
}

impl Role {
    /// The side of the item the class is, as the judgment reads it: what the
    /// alternative taken records into a class is its upper side, what passes
    /// into the alternative its lower side, and what the item carries or
    /// reads of a class its own; a class made anew at a decision, a copy and
    /// an environment are the item's bookkeeping, and no side.
    pub fn side(&self) -> Option<Side> {
        Some(match self {
            Role::Lower => Side::Lower,
            Role::Upper => Side::Upper,
            Role::Equated => Side::Equated,
            Role::Own => Side::Own,
            Role::Part(_) => Side::Part,
            Role::Position(Position::Arg(_)) => Side::Lower,
            Role::Position(Position::Result | Position::Exit) => Side::Upper,
            Role::Place | Role::Operand | Role::Callee => Side::Own,
            Role::Bound => Side::Lower,
            Role::Binding => Side::Upper,
            Role::Made(DisjunctionAt::Generalized(_)) => Side::Upper,
            Role::Made(_) | Role::Link { .. } | Role::Copy(_) | Role::Environment(_) => {
                return None;
            }
        })
    }
}

/// An item of the record: what recorded it, and whether it is decomposed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Recorded {
    pub item: Item,
    pub recorder: Recorder,
    /// Whether its decomposition is recorded: the parts of its two sides
    /// related, or a structure constraint "indexes `τ`" linked along the view
    /// of its container. A copy of a decomposed relation is decomposed, since
    /// what its decomposition recorded is copied with it. A disjunctive
    /// constraint has no decomposition.
    pub decomposed: bool,
}

impl Recorded {
    /// The relation, where the item is one.
    pub fn relation(&self) -> Option<&Relation> {
        self.item.relation()
    }

    /// The disjunctive constraint's name, kind and decision, where the item
    /// is one.
    pub fn disjunction(&self) -> Option<(&Name, &Disjunction, Option<&Decision>)> {
        match &self.item {
            Item::Disjunction {
                name,
                kind,
                decision,
            } => Some((name, kind, decision.as_ref())),
            Item::Relation(_) | Item::Instance { .. } | Item::Lambda { .. } => None,
        }
    }
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
            Relation::Standing(class, standing) => write!(f, "{}: {standing:?}", name(class)),
        }
    }
}
