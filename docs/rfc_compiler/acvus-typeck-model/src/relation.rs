//! Relations and what records them (l.223–233).
//!
//! Every relation is kept in a log with what recorded it, and the
//! equivalence classes are what the log makes of them. So the value a use
//! reads can be decided from the remaining relations alone (l.263).

use std::fmt;

use crate::class::{Arena, ClassId, Fact, KindSource, NamedFact};
use crate::lit::LitVar;
use crate::ty::{Name, Term};

/// A relation recorded in equivalence classes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Relation {
    /// `τ ≤ α`, which a meeting records (l.229), and a use's result type
    /// entering its result (l.276).
    LowerBound { lower: ClassId, upper: ClassId },
    /// `τ ≤ₚ α`, which a split records between the parts of a structural type
    /// where the place the value's part passes into is a meeting (l.280).
    PartLowerBound { lower: ClassId, upper: ClassId },
    /// `T ⇝ U`, which a boundary records (l.230).
    Boundary { from: ClassId, to: ClassId },
    /// An equation, which an invariant place records (l.231).
    Equation(ClassId, ClassId),
    /// A kind known in part, which a projection records (l.232).
    Fact(ClassId, Fact),
    /// A kind the class holds: a type the text or a declaration writes
    /// (l.122), or, in a copy of C, the kind the class held where its binding
    /// group was solved (l.217).
    Kind(ClassId, Term<ClassId>, KindSource),
    /// A literal's type (l.129).
    Lit(ClassId, LitVar),
    /// Whether the class is a reference, which a use records before it is
    /// decided (l.248).
    Reference(ClassId, bool),
    /// A lambda the Values of the class's function value names (l.183).
    Values(ClassId, Name),
}

impl Relation {
    /// The classes the relation names, a kind's parts among them.
    pub fn classes(&self) -> Vec<ClassId> {
        match self {
            Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper } => {
                vec![*lower, *upper]
            }
            Relation::Boundary { from, to } => vec![*from, *to],
            Relation::Equation(a, b) => vec![*a, *b],
            Relation::Fact(class, fact) => vec![*class, fact.class()],
            Relation::Kind(class, term, _) => std::iter::once(*class)
                .chain(term.parts().into_iter().map(|(_, part)| *part))
                .collect(),
            Relation::Lit(class, _)
            | Relation::Reference(class, _)
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
            Relation::Boundary { from, to } => Relation::Boundary {
                from: f(from),
                to: f(to),
            },
            Relation::Equation(a, b) => {
                let a = f(a);
                Relation::Equation(a, f(b))
            }
            Relation::Fact(class, fact) => {
                let class = f(class);
                Relation::Fact(class, fact.map(f))
            }
            Relation::Kind(class, term, source) => {
                let class = f(class);
                Relation::Kind(class, term.map(|_, part| f(part)), source)
            }
            Relation::Lit(class, lit) => Relation::Lit(f(class), lit),
            Relation::Reference(class, reference) => Relation::Reference(f(class), reference),
            Relation::Values(class, lambda) => Relation::Values(f(class), lambda),
        }
    }
}

/// What recorded a relation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Recorder {
    /// The type the text gives a class (l.122).
    Origin,
    /// The body's text, at the item of the program at this index (l.225).
    Text(usize),
    /// The use, lend, projection or binding of this name, when it was
    /// decided.
    Node(Name),
    /// The equivalence class of this class taking a constructor (l.280).
    Split(ClassId),
    /// A cycle of lower bounds alone, merged into one equivalence class
    /// (l.271).
    Cycle,
    /// The I\* check, making an instance of a lambda's C that no call made, in
    /// a view of a state it decides nodes again in (`design/15-istar.md`,
    /// Domain). Solving never records it.
    Check,
    /// The instance `at`, made by `by`, the use of a function at its item or
    /// the call of a function value (l.198), recording in its binding group
    /// the copy of what `of` recorded in C (l.217).
    Instance {
        at: Name,
        by: Box<Recorder>,
        of: Box<Recorder>,
    },
}

impl Recorder {
    /// Whether solving derives it from the other relations, so that a value
    /// read derives it again from the relations that remain (l.263).
    pub fn is_derived(&self) -> bool {
        matches!(self, Recorder::Split(_) | Recorder::Cycle)
    }

    /// The nodes it is recorded by, a value read leaving any of them out
    /// leaving it out (l.263): a node's own relation names that node, and a
    /// copy in an instance names what made the instance and, by `copy_name`,
    /// the nodes the relation it copies is recorded by.
    pub fn nodes(&self) -> Vec<Name> {
        match self {
            Recorder::Node(name) => vec![name.clone()],
            Recorder::Instance { at, by, of } => by
                .nodes()
                .into_iter()
                .chain(of.nodes().iter().map(|node| copy_name(at, node)))
                .collect(),
            Recorder::Origin
            | Recorder::Text(_)
            | Recorder::Split(_)
            | Recorder::Cycle
            | Recorder::Check => Vec::new(),
        }
    }

    /// Whether the text or a node records it, so that a value it makes enter
    /// a type is checked (l.282), in a copy of C as where it was recorded.
    pub fn is_entering(&self) -> bool {
        match self {
            Recorder::Text(_) | Recorder::Node(_) => true,
            Recorder::Instance { of, .. } => of.is_entering(),
            Recorder::Origin | Recorder::Split(_) | Recorder::Cycle | Recorder::Check => false,
        }
    }
}

/// The name, in the binding group an instance `at` is recorded in, of the
/// node `node` of the group instantiated (l.217).
pub fn copy_name(at: &Name, node: &Name) -> Name {
    Name::new(&format!("{at}/{node}"))
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
            Relation::Boundary { from, to } => write!(f, "{} ~> {}", name(from), name(to)),
            Relation::Equation(a, b) => write!(f, "{} = {}", name(a), name(b)),
            Relation::Fact(class, fact) => {
                let fact = NamedFact {
                    fact,
                    arena: self.arena,
                };
                write!(f, "{}: {fact}", name(class))
            }
            Relation::Kind(class, term, _) => {
                let term = term.clone().map(|_, part| name(&part).clone());
                write!(f, "{}: {term}", name(class))
            }
            Relation::Lit(class, lit) => write!(f, "{}: {lit}", name(class)),
            Relation::Reference(class, true) => write!(f, "{}: a reference", name(class)),
            Relation::Reference(class, false) => write!(f, "{}: no reference", name(class)),
            Relation::Values(class, lambda) => {
                write!(f, "{}: Values contains {lambda}", name(class))
            }
        }
    }
}
