//! A domain: the constructors an equivalence class's value may take, given
//! before the class takes one, by a literal, by a lend's result, and by a
//! use's choice at a variable of its candidates' generalization.

use std::collections::BTreeSet;
use std::fmt;

use crate::lit::{Family, LitVar};
use crate::ty::{Constructor, FloatTy, IntTy, Mutability, Name, Scalar};

const INTS: [IntTy; 8] = [
    IntTy::I8,
    IntTy::I16,
    IntTy::I32,
    IntTy::I64,
    IntTy::U8,
    IntTy::U16,
    IntTy::U32,
    IntTy::U64,
];

const FLOATS: [FloatTy; 2] = [FloatTy::F32, FloatTy::F64];

/// The constructors a class's value may take. `!`, which has no value, is a
/// constructor like any other here: a domain admits it only where it says so.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Domain {
    /// These constructors alone; where none, no type's, refused where given.
    Only(BTreeSet<Constructor>),
    /// Every constructor but these.
    Except(BTreeSet<Constructor>),
}

impl Domain {
    /// A literal's: the scalars of its family that hold every value it holds.
    /// A literal has a value, so its domain admits no `!`.
    pub fn of_literal(lit: &LitVar) -> Domain {
        let scalars: BTreeSet<Constructor> = match lit.family() {
            Family::Int => INTS.iter().map(|&int| Scalar::Int(int)).collect::<Vec<_>>(),
            Family::Float => FLOATS
                .iter()
                .map(|&float| Scalar::Float(float))
                .collect::<Vec<_>>(),
        }
        .into_iter()
        .filter(|&scalar| lit.admits(scalar))
        .map(Constructor::Scalar)
        .collect();
        Domain::Only(scalars)
    }

    /// A lend's result: what every alternative of a lend shares (D1), a
    /// reference of either mutability, or `!`. It has no default
    /// (`fallback`), so it gives no class a type of its own.
    pub fn references() -> Domain {
        Domain::Only(
            [
                Constructor::Ref(Mutability::Shared),
                Constructor::Ref(Mutability::Exclusive),
                Constructor::Never,
            ]
            .into(),
        )
    }

    pub fn admits(&self, constructor: &Constructor) -> bool {
        match self {
            Domain::Only(constructors) => constructors.contains(constructor),
            Domain::Except(excluded) => !excluded.contains(constructor),
        }
    }

    /// The constructors both admit; none where they admit none in common.
    pub fn meet(&self, other: &Domain) -> Option<Domain> {
        let met = match (self, other) {
            (Domain::Only(a), Domain::Only(b)) => {
                Domain::Only(a.intersection(b).cloned().collect())
            }
            (Domain::Only(a), Domain::Except(b)) | (Domain::Except(b), Domain::Only(a)) => {
                Domain::Only(a.difference(b).cloned().collect())
            }
            (Domain::Except(a), Domain::Except(b)) => Domain::Except(a.union(b).cloned().collect()),
        };
        match &met {
            Domain::Only(constructors) if constructors.is_empty() => None,
            _ => Some(met),
        }
    }

    /// The constructor a class holding it alone falls back to: the family
    /// default, `i32` or `f64`, where it admits that of one family alone;
    /// otherwise the one constructor it admits other than `!`, where it
    /// admits one; none otherwise. `!` is no default: a class nothing reaches
    /// takes it with no domain.
    pub fn fallback(&self) -> Option<Constructor> {
        let defaults: Vec<Constructor> = [
            Constructor::Scalar(Scalar::Int(IntTy::I32)),
            Constructor::Scalar(Scalar::Float(FloatTy::F64)),
        ]
        .into_iter()
        .filter(|default| self.admits(default))
        .collect();
        if let [default] = defaults.as_slice() {
            return Some(default.clone());
        }
        let Domain::Only(constructors) = self else {
            return None;
        };
        match constructors
            .iter()
            .filter(|constructor| **constructor != Constructor::Never)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [one] => Some((*one).clone()),
            _ => None,
        }
    }
}

/// The domain a use's choice gives a variable of its candidates'
/// generalization, as the record holds it: the choice it reads, the
/// variable, and the constructor each candidate writes there. It is read
/// against the candidates that remain to the choice whenever the record is
/// read (`resolve`), never copied into the class.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GeneralizedDomain {
    /// The use whose choice it reads.
    pub choice: Name,
    /// The variable of the generalization.
    pub var: usize,
    /// The constructor each candidate writes at the variable, in the
    /// candidates' order; none where it writes a variable of its own.
    pub constructors: Vec<Option<Constructor>>,
}

impl GeneralizedDomain {
    /// The domain the candidates `remaining` give: every constructor where
    /// one of them writes a variable of its own there; otherwise the
    /// constructors they write, and `!`, which reaches every type.
    pub fn resolve(&self, remaining: &BTreeSet<usize>) -> Domain {
        if remaining.iter().any(|&i| self.constructors[i].is_none()) {
            return Domain::Except(BTreeSet::new());
        }
        Domain::Only(
            remaining
                .iter()
                .filter_map(|&i| self.constructors[i].clone())
                .chain([Constructor::Never])
                .collect(),
        )
    }
}

/// `f[u8, _]`: the use and each candidate's constructor, `_` where it writes
/// a variable of its own.
impl fmt::Display for GeneralizedDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = self
            .constructors
            .iter()
            .map(|constructor| {
                constructor
                    .as_ref()
                    .map_or("_".to_string(), constructor_name)
            })
            .collect();
        write!(f, "{}[{}]", self.choice, names.join(", "))
    }
}

fn constructor_name(constructor: &Constructor) -> String {
    match constructor {
        Constructor::Scalar(scalar) => scalar.to_string(),
        Constructor::Ref(Mutability::Shared) => "&".to_string(),
        Constructor::Ref(Mutability::Exclusive) => "&mut".to_string(),
        Constructor::Never => "!".to_string(),
        other => format!("{other:?}"),
    }
}

/// `{u8, i32}`, or `not {&, &mut}`.
impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (prefix, constructors) = match self {
            Domain::Only(constructors) => ("", constructors),
            Domain::Except(excluded) => ("not ", excluded),
        };
        let names: Vec<String> = constructors.iter().map(constructor_name).collect();
        write!(f, "{prefix}{{{}}}", names.join(", "))
    }
}
