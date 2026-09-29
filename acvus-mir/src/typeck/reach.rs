//! The one reach rule (types §6), pure.
//!
//! Positions are read in order, an argument's and then a tuple's parts in
//! order: at each one, stepping up from its exact type along the cast tree,
//! the candidates that match at the first step where any match are kept.
//! Where more than one is left after them, the head the call's result is
//! taken at decides last, matched exactly. Exactly one must be left. Cost:
//! (positions × tree height + 1) × candidates.

use crate::decl::Head;

/// What a candidate's parameter, or its result, matches at one position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Match {
    /// A type variable: any head, at the exact step.
    Any,
    /// One head.
    Head(Head),
    /// A lifted impl's type variable (types §1): the head of any
    /// transparent structured type, at the exact step.
    Structured,
}

/// A candidate: what its parameter matches at each position, in the order
/// positions are read, and what its result matches.
pub(super) struct Candidate<'c> {
    pub params: &'c [Match],
    pub result: Match,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NoReach {
    None,
    Several(Vec<usize>),
}

/// The index of the one candidate reached by `positions`: at each, the
/// heads up the cast tree from its exact type, the exact one first; then,
/// where more than one is left, by `result`, the head the call's result is
/// taken at, where it is known.
pub(super) fn reach(candidates: &[Candidate<'_>], positions: &[Vec<Head>], result: Option<Head>) -> Result<usize, NoReach> {
    todo!("the positions as before, each `Match` at a step; then `result`, exactly, where more than one is left")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::Base;
    use crate::ty::{IntTy, Mutability};

    const I64: Head = Head::Base(Base::Int(IntTy::I64));
    const U8: Head = Head::Base(Base::Int(IntTy::U8));
    const SLICE: Head = Head::Ref(Mutability::Shared, Base::Slice);
    const STR: Head = Head::Ref(Mutability::Shared, Base::Str);
    const STRING: Head = Head::Ref(Mutability::Shared, Base::String);

    fn candidates(rows: &[Vec<Option<Head>>]) -> Vec<Candidate<'_>> {
        rows.iter().map(|row| Candidate { params: row }).collect()
    }

    /// types §6: at the exact type, the candidates that match are kept.
    #[test]
    fn the_exact_type_keeps_its_candidate() {
        let rows = vec![vec![Some(I64)], vec![Some(U8)]];
        assert_eq!(reach(&candidates(&rows), &[vec![U8]]), Ok(1));
    }

    /// types §6: where none matches at the exact type, the first step up
    /// where any matches decides.
    #[test]
    fn a_step_up_is_taken_only_where_the_exact_type_matches_none() {
        let rows = vec![vec![Some(STR)], vec![Some(SLICE)]];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]]), Ok(0));
        let rows = vec![vec![Some(STR)], vec![Some(STRING)]];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]]), Ok(1));
    }

    /// types §6: positions are read front to back; a later one narrows
    /// only what an earlier one kept.
    #[test]
    fn positions_narrow_in_order() {
        let rows = vec![vec![Some(I64), Some(U8)], vec![Some(I64), Some(I64)], vec![Some(U8), Some(U8)]];
        assert_eq!(reach(&candidates(&rows), &[vec![I64], vec![U8]]), Ok(0));
    }

    /// A type variable matches at the exact step.
    #[test]
    fn a_type_variable_matches_at_the_exact_step() {
        let rows = vec![vec![None], vec![Some(STR)]];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]]), Ok(0));
    }

    /// types §6: exactly one must be left.
    #[test]
    fn none_or_several_left_is_no_reach() {
        let rows = vec![vec![Some(I64)], vec![Some(I64)]];
        assert_eq!(reach(&candidates(&rows), &[vec![I64]]), Err(NoReach::Several(vec![0, 1])));
        assert_eq!(reach(&candidates(&rows), &[vec![U8]]), Err(NoReach::None));
    }

    /// A prefix of the positions may already leave one.
    #[test]
    fn a_prefix_may_decide() {
        let rows = vec![vec![Some(I64), Some(I64)], vec![Some(U8), Some(U8)]];
        assert_eq!(reach(&candidates(&rows), &[vec![U8]]), Ok(1));
    }
}
