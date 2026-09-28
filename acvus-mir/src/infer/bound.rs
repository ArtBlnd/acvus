//! What a variable's bound admits (extern-fn §1, types §5).

use super::Bound;
use crate::ty::{Infer as InferPhase, IntTy, TyKind};

impl Bound {
    /// Whether a variable of this bound may be bound to a node of `kind`.
    pub(super) fn admits(&self, kind: &TyKind<InferPhase>) -> bool {
        match self {
            Bound::Any => true,
            Bound::Settled => !matches!(kind, TyKind::Never),
            Bound::Integer { signed, among, .. } => {
                matches!(kind, TyKind::Int(width) if Self::integer_admits(*signed, among, *width))
            }
        }
    }

    /// The bound of a variable made one with another: what both admit.
    /// `None` when nothing is.
    pub(super) fn meet(&self, other: &Bound) -> Option<Bound> {
        match (self, other) {
            (Bound::Any, bound) | (bound, Bound::Any) => Some(bound.clone()),
            (Bound::Settled, bound) | (bound, Bound::Settled) => Some(bound.clone()),
            (
                Bound::Integer {
                    signed: signed_a,
                    among: among_a,
                    literal: literal_a,
                },
                Bound::Integer {
                    signed: signed_b,
                    among: among_b,
                    literal: literal_b,
                },
            ) => {
                let among: Box<[IntTy]> = among_a
                    .iter()
                    .copied()
                    .filter(|width| {
                        Self::integer_admits(*signed_a, among_a, *width)
                            && Self::integer_admits(*signed_b, among_b, *width)
                    })
                    .collect();
                (!among.is_empty()).then_some(Bound::Integer {
                    signed: *signed_a || *signed_b,
                    among,
                    literal: *literal_a && *literal_b,
                })
            }
        }
    }

    fn integer_admits(signed: bool, among: &[IntTy], width: IntTy) -> bool {
        among.contains(&width) && (signed || !width.signed())
    }
}
