//! The one reach rule (types §6), pure.
//!
//! Positions are read in order, an argument's and then a tuple's parts in
//! order: at each one, stepping up from its exact type along the cast tree,
//! the candidates that match at the first step where any match are kept.
//! Where more than one is left after them, the head the call's result is
//! taken at decides last, matched exactly. Exactly one must be left. Cost:
//! (positions × tree height + 1) × candidates.

use crate::decl::{Base, Head, Taking};

/// What a candidate's parameter, or its result, matches at one position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Match {
    /// A type variable: any head, at the exact step.
    Any,
    /// One head.
    Head(Head),
    /// A lifted impl's type variable (types §1), taken as the lift takes
    /// it: the head of any transparent structured type, or a reference of
    /// exactly that mutability to one, at the exact step.
    Structured(Taking),
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
    let mut left: Vec<usize> = (0..candidates.len()).collect();
    for (at, chain) in positions.iter().enumerate() {
        let matching = |step: &Head| -> Vec<usize> {
            left.iter().copied().filter(|candidate| matches(candidates[*candidate].params[at], *step)).collect()
        };
        let Some(kept) = chain.iter().map(matching).find(|kept| !kept.is_empty()) else {
            return Err(NoReach::None);
        };
        left = kept;
    }
    if left.len() > 1
        && let Some(result) = result
    {
        left.retain(|candidate| matches(candidates[*candidate].result, result));
    }
    match left[..] {
        [one] => Ok(one),
        [] => Err(NoReach::None),
        _ => Err(NoReach::Several(left)),
    }
}

/// Whether `matched` matches `head` at one step: a lifted impl's variable
/// at a transparent structured type taken as the lift takes it (types §1),
/// with no other mutability.
fn matches(matched: Match, head: Head) -> bool {
    let structured = |base: Base| matches!(base, Base::Tuple(_) | Base::Array | Base::Option | Base::Result | Base::Object | Base::Enum(_));
    match (matched, head) {
        (Match::Any, _) => true,
        (Match::Head(own), head) => own == head,
        (Match::Structured(Taking::Value), Head::Base(base)) => structured(base),
        (Match::Structured(Taking::Ref(taken)), Head::Ref(held, base)) => taken == held && structured(base),
        (Match::Structured(_), _) => false,
    }
}

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::Base;
    use crate::ty::{IntTy, Mutability};

    const I64: Head = Head::Base(Base::Int(IntTy::I64));
    const U8: Head = Head::Base(Base::Int(IntTy::U8));
    const BOOL: Head = Head::Base(Base::Bool);
    const PAIR: Head = Head::Base(Base::Tuple(2));
    const SLICE: Head = Head::Ref(Mutability::Shared, Base::Slice);
    const STR: Head = Head::Ref(Mutability::Shared, Base::Str);
    const STRING: Head = Head::Ref(Mutability::Shared, Base::String);
    const VALUE: Match = Match::Structured(Taking::Value);

    fn candidates<'c>(rows: &'c [(Vec<Match>, Match)]) -> Vec<Candidate<'c>> {
        rows.iter().map(|(params, result)| Candidate { params, result: *result }).collect()
    }

    fn head(head: Head) -> Match {
        Match::Head(head)
    }

    /// types §6: at the exact type, the candidates that match are kept.
    #[test]
    fn the_exact_type_keeps_its_candidate() {
        let rows = vec![(vec![head(I64)], Match::Any), (vec![head(U8)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![U8]], None), Ok(1));
    }

    /// types §6: where none matches at the exact type, the first step up
    /// where any matches decides.
    #[test]
    fn a_step_up_is_taken_only_where_the_exact_type_matches_none() {
        let rows = vec![(vec![head(STR)], Match::Any), (vec![head(SLICE)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]], None), Ok(0));
        let rows = vec![(vec![head(STR)], Match::Any), (vec![head(STRING)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]], None), Ok(1));
    }

    /// types §6: positions are read front to back; a later one narrows
    /// only what an earlier one kept.
    #[test]
    fn positions_narrow_in_order() {
        let rows = vec![
            (vec![head(I64), head(U8)], Match::Any),
            (vec![head(I64), head(I64)], Match::Any),
            (vec![head(U8), head(U8)], Match::Any),
        ];
        assert_eq!(reach(&candidates(&rows), &[vec![I64], vec![U8]], None), Ok(0));
    }

    /// A type variable matches at the exact step.
    #[test]
    fn a_type_variable_matches_at_the_exact_step() {
        let rows = vec![(vec![Match::Any], Match::Any), (vec![head(STR)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![STRING, STR]], None), Ok(0));
    }

    /// types §1: a lifted impl's variable matches a transparent structured
    /// type taken as the lift takes it, at the exact step, and nothing else:
    /// a lift taking `&T` is reached by neither a `&mut` nor a value.
    #[test]
    fn structured_matches_only_a_structure_taken_as_the_lift_takes_it() {
        let by_ref = Match::Structured(Taking::Ref(Mutability::Shared));
        let rows = vec![(vec![head(I64)], Match::Any), (vec![by_ref], VALUE)];
        let shared_pair = Head::Ref(Mutability::Shared, Base::Tuple(2));
        let mut_pair = Head::Ref(Mutability::Mut, Base::Tuple(2));
        assert_eq!(reach(&candidates(&rows), &[vec![shared_pair]], None), Ok(1));
        assert_eq!(reach(&candidates(&rows), &[vec![mut_pair]], None), Err(NoReach::None));
        assert_eq!(reach(&candidates(&rows), &[vec![PAIR]], None), Err(NoReach::None));
        assert_eq!(reach(&candidates(&rows), &[vec![I64]], None), Ok(0));
        let rows = vec![(vec![VALUE], VALUE)];
        assert_eq!(reach(&candidates(&rows), &[vec![PAIR]], None), Ok(0));
        assert_eq!(reach(&candidates(&rows), &[vec![shared_pair]], None), Err(NoReach::None));
        let names = Interner::new();
        let named = Head::Base(Base::Named { id: QualifiedRef::root(names.intern("S")), marked: false });
        assert_eq!(reach(&candidates(&rows), &[vec![named]], None), Err(NoReach::None));
    }

    /// types §6: exactly one must be left.
    #[test]
    fn none_or_several_left_is_no_reach() {
        let rows = vec![(vec![head(I64)], Match::Any), (vec![head(I64)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![I64]], None), Err(NoReach::Several(vec![0, 1])));
        assert_eq!(reach(&candidates(&rows), &[vec![U8]], None), Err(NoReach::None));
    }

    /// A prefix of the positions may already leave one.
    #[test]
    fn a_prefix_may_decide() {
        let rows = vec![(vec![head(I64), head(I64)], Match::Any), (vec![head(U8), head(U8)], Match::Any)];
        assert_eq!(reach(&candidates(&rows), &[vec![U8]], None), Ok(1));
    }

    /// types §6: where more than one is left after the arguments, the type
    /// the result is taken at decides last, matched exactly, with no step
    /// up the tree.
    #[test]
    fn the_result_decides_last_and_exactly() {
        let rows = vec![(vec![], head(I64)), (vec![], VALUE), (vec![], head(BOOL))];
        let rows = candidates(&rows);
        assert_eq!(reach(&rows, &[], Some(PAIR)), Ok(1));
        assert_eq!(reach(&rows, &[], Some(I64)), Ok(0));
        assert_eq!(reach(&rows, &[], Some(U8)), Err(NoReach::None), "no step up at the result");
        assert_eq!(reach(&rows, &[], None), Err(NoReach::Several(vec![0, 1, 2])));
    }

    /// types §6: the result is read only where the arguments leave more
    /// than one.
    #[test]
    fn the_result_is_not_read_where_the_arguments_leave_one() {
        let rows = vec![(vec![head(I64)], head(BOOL)), (vec![head(U8)], head(I64))];
        assert_eq!(reach(&candidates(&rows), &[vec![I64]], Some(I64)), Ok(0));
    }
}
