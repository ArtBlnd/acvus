//! A domain: the constructors an equivalence class's value may take, given
//! before the class takes one, by a literal, the variable of a use's generalization
//! and a lend's result.

use std::collections::BTreeSet;
use std::fmt;

use crate::lit::{Family, LitVar};
use crate::ty::{Constructor, FloatTy, IntTy, Mutability, Scalar, Term};

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

    /// A reference of either mutability, or `!`, as at a generalization's variable
    /// every candidate writes a reference at, into which nothing enters where
    /// the use is decided `!`.
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

    /// Anything but a reference.
    pub fn no_references() -> Domain {
        Domain::Except(
            [
                Constructor::Ref(Mutability::Shared),
                Constructor::Ref(Mutability::Exclusive),
            ]
            .into(),
        )
    }

    /// Whether the domain admits a constructor that is no reference and not
    /// `!`.
    pub fn admits_no_reference(&self) -> bool {
        let reference_or_never = |constructor: &Constructor| {
            matches!(constructor, Constructor::Ref(_) | Constructor::Never)
        };
        match self {
            Domain::Only(admitted) => admitted.iter().any(|c| !reference_or_never(c)),
            Domain::Except(_) => true,
        }
    }

    pub fn admits(&self, constructor: &Constructor) -> bool {
        match self {
            Domain::Only(admitted) => admitted.contains(constructor),
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
            Domain::Only(admitted) if admitted.is_empty() => None,
            _ => Some(met),
        }
    }

    /// What a class holding it alone falls back to: its family's default,
    /// `i32` or `f64`, where it admits only scalars of one family, and `!`
    /// otherwise.
    pub fn fallback<P>(&self) -> Term<P> {
        let Domain::Only(admitted) = self else {
            return Term::Never;
        };
        let family = |constructor: &Constructor| match constructor {
            Constructor::Scalar(Scalar::Int(_)) => Some(Family::Int),
            Constructor::Scalar(Scalar::Float(_)) => Some(Family::Float),
            _ => None,
        };
        let families: BTreeSet<Option<Family>> = admitted.iter().map(family).collect();
        match families.into_iter().collect::<Vec<_>>().as_slice() {
            [Some(Family::Int)] => Term::Scalar(Scalar::Int(IntTy::I32)),
            [Some(Family::Float)] => Term::Scalar(Scalar::Float(FloatTy::F64)),
            _ => Term::Never,
        }
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
            Domain::Only(admitted) => ("", admitted),
            Domain::Except(excluded) => ("not ", excluded),
        };
        let names: Vec<String> = constructors.iter().map(constructor_name).collect();
        write!(f, "{prefix}{{{}}}", names.join(", "))
    }
}
