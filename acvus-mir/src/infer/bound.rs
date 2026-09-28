//! What a variable's bound admits (extern-fn §1, types §5).

use super::Bound;
use crate::ty::{Infer as InferPhase, IntTy, TyKind};

impl Bound {
    /// Whether a variable of this bound may be bound to a node of `kind`.
    pub(super) fn admits(&self, kind: &TyKind<InferPhase>) -> bool {
        match self {
            Bound::Any => true,
            Bound::Settled => !matches!(kind, TyKind::Never),
            Bound::Integer { signed, among } => {
                matches!(kind, TyKind::Int(width) if Self::integer_admits(*signed, among, *width))
            }
            Bound::Literal => matches!(kind, TyKind::Int(_)),
            Bound::OneOf(shapes) => todo!(
                "stop: when a OneOf bound of {} shapes admits a node is not settled: by head, by eq \
                 with one shape (binding what?), or only once the node is closed",
                shapes.len()
            ),
        }
    }

    /// The bound of a variable made one with another: what both admit.
    /// `None` when nothing is.
    pub(super) fn meet(&self, other: &Bound) -> Option<Bound> {
        match (self, other) {
            (Bound::Any, bound) | (bound, Bound::Any) => Some(bound.clone()),
            (Bound::OneOf(_), _) | (_, Bound::OneOf(_)) => todo!(
                "stop: the meet of a OneOf bound waits on what a OneOf bound admits"
            ),
            (Bound::Settled, bound) | (bound, Bound::Settled) => Some(bound.clone()),
            (Bound::Literal, Bound::Literal) => Some(Bound::Literal),
            (
                Bound::Integer {
                    signed: signed_a,
                    among: among_a,
                },
                Bound::Integer {
                    signed: signed_b,
                    among: among_b,
                },
            ) => {
                let signed = *signed_a && *signed_b;
                let among: Box<[IntTy]> = among_a
                    .iter()
                    .copied()
                    .filter(|width| among_b.contains(width) && Self::integer_admits(signed, among_a, *width))
                    .collect();
                (!among.is_empty()).then_some(Bound::Integer { signed, among })
            }
            (Bound::Literal, Bound::Integer { .. }) | (Bound::Integer { .. }, Bound::Literal) => todo!(
                "stop: a literal's variable made one with an Integer bound: Bound has no form for \
                 an integer bound that still defaults to i64 (types §5)"
            ),
        }
    }

    fn integer_admits(signed: bool, among: &[IntTy], width: IntTy) -> bool {
        among.contains(&width) && (signed || !width.signed())
    }
}
