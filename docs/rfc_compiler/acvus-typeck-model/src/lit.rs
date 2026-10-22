//! A literal's type: a LitVar `?N`, `N` the literal values it holds, of the
//! integer or the float family (l.127, l.162–163, l.288).

use std::collections::BTreeSet;
use std::fmt;

use crate::ty::{FloatTy, IntTy, Scalar};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Family {
    Int,
    Float,
}

/// A literal value. An integer literal is unsigned, a sign being an operator;
/// a float literal is held by its `f64` bits, so that literals are ordered.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Literal {
    Int(u128),
    Float(u64),
}

impl Literal {
    pub fn float(value: f64) -> Literal {
        Literal::Float(value.to_bits())
    }

    pub fn family(self) -> Family {
        match self {
            Literal::Int(_) => Family::Int,
            Literal::Float(_) => Family::Float,
        }
    }
}

/// `?N`: the values `N` of one family.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LitVar {
    family: Family,
    values: BTreeSet<Literal>,
}

impl LitVar {
    /// `?{n}`, the type of the literal `n`.
    pub fn of(literal: Literal) -> LitVar {
        LitVar {
            family: literal.family(),
            values: [literal].into(),
        }
    }

    pub fn family(&self) -> Family {
        self.family
    }

    pub fn values(&self) -> &BTreeSet<Literal> {
        &self.values
    }

    /// `?(N ∪ M)`, where the two are of one family (l.163).
    pub fn union(&self, other: &LitVar) -> Option<LitVar> {
        (self.family == other.family).then(|| LitVar {
            family: self.family,
            values: self.values.union(&other.values).copied().collect(),
        })
    }

    /// Whether `scalar` is of this family and holds every value in `N` (l.162).
    pub fn admits(&self, scalar: Scalar) -> bool {
        self.values.iter().all(|&literal| holds(scalar, literal))
    }

    /// The family's default (l.288).
    pub fn default_scalar(&self) -> Scalar {
        match self.family {
            Family::Int => Scalar::Int(IntTy::I32),
            Family::Float => Scalar::Float(FloatTy::F64),
        }
    }
}

/// `?{n, …}: int` or `?{x, …}: float`, as l.127 writes a LitVar.
impl fmt::Display for LitVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("?{")?;
        for (i, literal) in self.values.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match literal {
                Literal::Int(value) => write!(f, "{value}")?,
                Literal::Float(bits) => write!(f, "{:?}", f64::from_bits(*bits))?,
            }
        }
        match self.family {
            Family::Int => f.write_str("}: int"),
            Family::Float => f.write_str("}: float"),
        }
    }
}

fn holds(scalar: Scalar, literal: Literal) -> bool {
    match (scalar, literal) {
        (Scalar::Int(int), Literal::Int(value)) => value <= max(int),
        (Scalar::Float(FloatTy::F64), Literal::Float(_)) => true,
        (Scalar::Float(FloatTy::F32), Literal::Float(bits)) => {
            (f64::from_bits(bits) as f32).is_finite()
        }
        _ => false,
    }
}

/// The greatest value of an integer type.
fn max(int: IntTy) -> u128 {
    match int {
        IntTy::I8 => i8::MAX as u128,
        IntTy::I16 => i16::MAX as u128,
        IntTy::I32 => i32::MAX as u128,
        IntTy::I64 => i64::MAX as u128,
        IntTy::U8 => u8::MAX as u128,
        IntTy::U16 => u16::MAX as u128,
        IntTy::U32 => u32::MAX as u128,
        IntTy::U64 => u64::MAX as u128,
    }
}
