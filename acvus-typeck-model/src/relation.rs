//! Relations and what records them (l.228–238).
//!
//! Every relation is kept in a log with what recorded it, and the
//! equivalence classes are what the log makes of them. So the value a use
//! reads can be decided from the remaining relations alone (l.267).

use std::fmt;

use crate::class::{Arena, ClassId, Fact};
use crate::lit::LitVar;
use crate::ty::{Name, Term};

/// A relation recorded in equivalence classes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Relation {
    /// `τ ≤ α`, which a meeting records (l.234), and a use's result type
    /// entering its result (l.280).
    LowerBound { lower: ClassId, upper: ClassId },
    /// `T ⇝ U`, which a boundary records (l.235).
    Boundary { from: ClassId, to: ClassId },
    /// An equation, which an invariant place records (l.236).
    Equation(ClassId, ClassId),
    /// A kind known in part, which a projection records (l.237).
    Fact(ClassId, Fact),
    /// A type the text or a declaration writes, which the class holds (l.127).
    Kind(ClassId, Term<ClassId>),
    /// A literal's type (l.134).
    Lit(ClassId, LitVar),
    /// Whether the class is a reference, which a use records before it is
    /// decided (l.252).
    Reference(ClassId, bool),
}

/// What recorded a relation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Recorder {
    /// The type the text gives a class (l.127).
    Origin,
    /// The body's text, at the item of the program at this index (l.230).
    Text(usize),
    /// The use, lend, projection or binding of this name, when it was
    /// decided.
    Node(Name),
    /// The equivalence class of this class taking a constructor (l.284).
    Split(ClassId),
    /// A cycle of lower bounds alone, merged into one equivalence class
    /// (l.275).
    Cycle,
}

impl Recorder {
    /// Whether solving derives it from the other relations, so that a value
    /// read derives it again from the relations that remain (l.267).
    pub fn is_derived(&self) -> bool {
        matches!(self, Recorder::Split(_) | Recorder::Cycle)
    }
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
            Relation::LowerBound { lower, upper } => write!(f, "{} ≤ {}", name(lower), name(upper)),
            Relation::Boundary { from, to } => write!(f, "{} ⇝ {}", name(from), name(to)),
            Relation::Equation(a, b) => write!(f, "{} = {}", name(a), name(b)),
            Relation::Fact(class, fact) => write!(f, "{}: {fact:?}", name(class)),
            Relation::Kind(class, term) => {
                let term = term.clone().map(|_, part| name(&part).clone());
                write!(f, "{}: {term:?}", name(class))
            }
            Relation::Lit(class, lit) => write!(f, "{}: {lit}", name(class)),
            Relation::Reference(class, true) => write!(f, "{}: a reference", name(class)),
            Relation::Reference(class, false) => write!(f, "{}: no reference", name(class)),
        }
    }
}
