//! Domains.

use std::collections::BTreeSet;

use super::ty::Scalar;

/// A family of scalars that gives a fallback type. A class no
/// domain is registered in has no row here: it falls back to `!`, once every
/// class that has one has fallen back.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Family {
    /// The integers of 8 to 64 bits, and `usize`. Falls back to `i32`.
    Integer,
    /// `f32` and `f64`. Falls back to `f64`.
    Float,
}

impl Family {
    pub fn fallback(self) -> Scalar {
        match self {
            Family::Integer => Scalar::I32,
            Family::Float => Scalar::F64,
        }
    }

    fn holds(self, scalar: Scalar) -> bool {
        match scalar {
            Scalar::I8
            | Scalar::I16
            | Scalar::I32
            | Scalar::I64
            | Scalar::U8
            | Scalar::U16
            | Scalar::U32
            | Scalar::U64
            | Scalar::Usize => self == Family::Integer,
            Scalar::F32 | Scalar::F64 => self == Family::Float,
            Scalar::Bool | Scalar::Char => false,
        }
    }
}

/// The domain a literal with no suffix registers: the types that hold its
/// value. A class no domain is registered in may be every type, so every
/// type is written by registering nothing.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Domain(pub BTreeSet<Scalar>);

impl Domain {
    /// The types every one of `domains` holds, where there is a domain at all.
    pub fn intersection<'a, I>(domains: I) -> Option<Domain>
    where
        I: IntoIterator<Item = &'a Domain>,
    {
        domains
            .into_iter()
            .cloned()
            .reduce(|shared, domain| Domain(shared.0.intersection(&domain.0).copied().collect()))
    }

    /// The least row that holds this domain, whose fallback type it may not
    /// itself hold.
    pub fn family(&self) -> Option<Family> {
        [Family::Integer, Family::Float]
            .into_iter()
            .find(|family| self.0.iter().all(|scalar| family.holds(*scalar)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(scalars: &[Scalar]) -> Domain {
        Domain(scalars.iter().copied().collect())
    }

    #[test]
    fn an_intersection_holds_what_every_domain_holds() {
        let (small, large) = (of(&[Scalar::U8, Scalar::I32]), of(&[Scalar::I32, Scalar::I64]));
        assert_eq!(Domain::intersection([&small, &large]), Some(of(&[Scalar::I32])));
        assert_eq!(Domain::intersection([&small]), Some(small.clone()));
        assert_eq!(Domain::intersection([]), None);
    }

    #[test]
    fn a_domain_falls_back_by_the_least_row_that_holds_it() {
        assert_eq!(of(&[Scalar::U8, Scalar::Usize]).family(), Some(Family::Integer));
        assert_eq!(of(&[Scalar::F32]).family(), Some(Family::Float));
        assert_eq!(of(&[Scalar::U8, Scalar::F32]).family(), None);
        assert_eq!(of(&[Scalar::Bool]).family(), None);
        assert_eq!(Family::Integer.fallback(), Scalar::I32);
        assert_eq!(Family::Float.fallback(), Scalar::F64);
    }
}
