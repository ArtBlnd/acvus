//! A range: the constructors an equivalence class's value may take, given
//! before the class takes one, by a literal, the variable of a use's common
//! part and a lend's result.

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
/// constructor like any other here: a range admits it only where it says so.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Range {
    /// These constructors alone, at least one.
    Only(BTreeSet<Constructor>),
    /// Every constructor but these.
    Except(BTreeSet<Constructor>),
}

impl Range {
    /// A literal's: the scalars of its family that hold every value it holds.
    /// A literal has a value, so its range admits no `!`.
    pub fn of_literal(lit: &LitVar) -> Range {
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
        Range::Only(scalars)
    }

    /// A reference of either mutability, or `!`, as at a common part's variable
    /// every candidate writes a reference at, into which nothing enters where
    /// the use is decided `!`.
    pub fn references() -> Range {
        Range::Only(
            [
                Constructor::Ref(Mutability::Shared),
                Constructor::Ref(Mutability::Exclusive),
                Constructor::Never,
            ]
            .into(),
        )
    }

    /// Anything but a reference.
    pub fn no_references() -> Range {
        Range::Except(
            [
                Constructor::Ref(Mutability::Shared),
                Constructor::Ref(Mutability::Exclusive),
            ]
            .into(),
        )
    }

    pub fn admits(&self, constructor: &Constructor) -> bool {
        match self {
            Range::Only(admitted) => admitted.contains(constructor),
            Range::Except(excluded) => !excluded.contains(constructor),
        }
    }

    /// The constructors both admit; none where they admit none in common.
    pub fn meet(&self, other: &Range) -> Option<Range> {
        let met = match (self, other) {
            (Range::Only(a), Range::Only(b)) => Range::Only(a.intersection(b).cloned().collect()),
            (Range::Only(a), Range::Except(b)) | (Range::Except(b), Range::Only(a)) => {
                Range::Only(a.difference(b).cloned().collect())
            }
            (Range::Except(a), Range::Except(b)) => Range::Except(a.union(b).cloned().collect()),
        };
        match &met {
            Range::Only(admitted) if admitted.is_empty() => None,
            _ => Some(met),
        }
    }

    /// Whether every constructor it admits it admits too.
    pub fn within(&self, other: &Range) -> bool {
        self.meet(other).as_ref() == Some(self)
    }

    /// Whether it admits nothing but references and `!`.
    pub fn is_references(&self) -> bool {
        match self {
            Range::Only(admitted) => admitted
                .iter()
                .all(|constructor| matches!(constructor, Constructor::Ref(_) | Constructor::Never)),
            Range::Except(_) => false,
        }
    }

    /// The range a class this one is a lower bound of takes from it: itself,
    /// where it admits no `!` and only scalars, from which no cast goes up, so
    /// that a value reaching the class by `≤` is one it admits; none
    /// otherwise, a value of `!` reaching every type.
    pub fn upward(&self) -> Option<Range> {
        match self {
            Range::Only(admitted)
                if admitted
                    .iter()
                    .all(|constructor| matches!(constructor, Constructor::Scalar(_))) =>
            {
                Some(self.clone())
            }
            Range::Only(_) | Range::Except(_) => None,
        }
    }

    /// What a class holding it alone falls back to: its family's default,
    /// `i32` or `f64`, where it admits only scalars of one family, and `!`
    /// otherwise.
    pub fn fallback<P>(&self) -> Term<P> {
        let Range::Only(admitted) = self else {
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
impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (prefix, constructors) = match self {
            Range::Only(admitted) => ("", admitted),
            Range::Except(excluded) => ("not ", excluded),
        };
        let names: Vec<String> = constructors.iter().map(constructor_name).collect();
        write!(f, "{prefix}{{{}}}", names.join(", "))
    }
}
