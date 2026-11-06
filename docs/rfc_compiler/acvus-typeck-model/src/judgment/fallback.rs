//! Fallback: rule 4 of the judgment, what each fallback candidate takes at
//! the `entry` group's fixpoint (D2).
//!
//! The orchestrator gives the candidates (`State::fallback_candidate`): a
//! class with no lower bound, recorded or pending, that holds a literal's
//! domain, with that domain; and a class with no lower bound and no domain,
//! on which no requirement may still be recorded, nor, where a requirement
//! is on it, an occurrence, with none. A literal's domain gives its default
//! (`Domain::fallback`), the family default or the one constructor it
//! admits, whatever the classes it passes into may still become: the
//! default replaces rule 3 there. `!` is given only to a class whose value
//! the text shows nowhere; a requirement witnesses that a value is used, so
//! a class with a requirement and nothing else gets no type, `no_value`. A
//! domain with no default gives no type, a conflict.
//!
//! This layer imports no `State` and no other layer.

use crate::class::ClassId;
use crate::domain::Domain;
use crate::ty::Constructor;

/// A fallback candidate (D2): the domain its own domains meet at, which a
/// literal's is among, or none for a class with no domain; and whether a
/// requirement is on it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Candidate {
    pub class: ClassId,
    pub domain: Option<Domain>,
    pub required: bool,
}

/// What a candidate takes at fallback.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Fallen {
    /// The constructor of its structure: a literal's domain's default, or `!`
    /// where the text shows its value nowhere.
    Default(Constructor),
    /// No type: its domain has no default, the values reaching it having no
    /// type in common or admitting several.
    NoType(Domain),
    /// No type: the text uses it as a value and shows none.
    NoValue,
}

/// What the candidates take (rule 4, D2), one entry per candidate that takes
/// something.
///
/// - Where a candidate has a domain, a literal's among its own, it takes the
///   domain's default (`Domain::fallback`); where the domain has none, no
///   type, `NoType`. Where at least one candidate has a domain, those
///   candidates alone take something: a default gives a head that may
///   decide other classes by rules 1 and 2, so the loop runs again before
///   anything takes `!`.
/// - Where no candidate has a domain, a candidate with a requirement takes
///   no type, `NoValue`, and every other candidate takes `!`.
///
/// It reads no occurrence, no disjunctive constraint and no choice, and it
/// must be given no class with a lower bound.
pub fn fall_back(candidates: &[Candidate]) -> Vec<(ClassId, Fallen)> {
    let by_domain: Vec<(ClassId, Fallen)> = candidates
        .iter()
        .filter_map(|candidate| {
            let domain = candidate.domain.as_ref()?;
            let fallen = match domain.fallback() {
                Some(default) => Fallen::Default(default),
                None => Fallen::NoType(domain.clone()),
            };
            Some((candidate.class, fallen))
        })
        .collect();
    if !by_domain.is_empty() {
        return by_domain;
    }
    candidates
        .iter()
        .map(|candidate| {
            let fallen = if candidate.required {
                Fallen::NoValue
            } else {
                Fallen::Default(Constructor::Never)
            };
            (candidate.class, fallen)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::class::{Arena, ClassName};
    use crate::lit::{LitVar, Literal};
    use crate::ty::{IntTy, Name, Scalar};

    fn classes(names: &[&str]) -> Vec<ClassId> {
        let mut arena = Arena::default();
        names
            .iter()
            .map(|name| arena.intern(ClassName::Text(Name::new(name))))
            .collect()
    }

    #[test]
    fn candidates_with_a_domain_take_their_default_first() {
        let ids = classes(&["c0", "c1"]);
        let int_lit = LitVar::of(Literal::Int(1));
        let candidates = vec![
            Candidate {
                class: ids[0],
                domain: Some(Domain::of_literal(&int_lit)),
                required: false,
            },
            Candidate {
                class: ids[1],
                domain: None,
                required: false,
            },
        ];
        assert_eq!(
            fall_back(&candidates),
            vec![(
                ids[0],
                Fallen::Default(Constructor::Scalar(Scalar::Int(IntTy::I32)))
            )]
        );
    }

    #[test]
    fn a_domain_admitting_nothing_gives_no_type() {
        let ids = classes(&["c0"]);
        let empty = Domain::Only(BTreeSet::new());
        let candidates = vec![Candidate {
            class: ids[0],
            domain: Some(empty.clone()),
            required: false,
        }];
        assert_eq!(fall_back(&candidates), vec![(ids[0], Fallen::NoType(empty))]);
    }

    #[test]
    fn a_domain_of_one_constructor_falls_back_to_it() {
        let ids = classes(&["c0"]);
        let one = Domain::Only(
            [
                Constructor::Scalar(Scalar::Int(IntTy::U8)),
                Constructor::Never,
            ]
            .into(),
        );
        let candidates = vec![Candidate {
            class: ids[0],
            domain: Some(one),
            required: false,
        }];
        assert_eq!(
            fall_back(&candidates),
            vec![(
                ids[0],
                Fallen::Default(Constructor::Scalar(Scalar::Int(IntTy::U8)))
            )]
        );
    }

    #[test]
    fn where_none_has_a_domain_a_requirement_gives_no_value_and_the_rest_take_never() {
        let ids = classes(&["c0", "c1"]);
        let candidates = vec![
            Candidate {
                class: ids[0],
                domain: None,
                required: true,
            },
            Candidate {
                class: ids[1],
                domain: None,
                required: false,
            },
        ];
        assert_eq!(
            fall_back(&candidates),
            vec![
                (ids[0], Fallen::NoValue),
                (ids[1], Fallen::Default(Constructor::Never))
            ]
        );
    }
}
