//! The constant domain: bottom, one known value, or top; height 3, so its
//! widening is its join, stated. Abstract operations compute as the language
//! does (operators §1): an operation that would trap on the known operands
//! is top, since the trap is the run's, not a value.

use acvus_ast::Literal;

use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Bottom,
    Known(acvus_ast::Literal),
    Top,
}

/// Whether two literals spell one value of the language. An `f64` is its
/// bits, so `0.0` and `-0.0` are two values and a `NaN` is itself; the
/// suffix of an integer and the byte form of a byte string are the
/// literal's type, not its value.
fn same_value(a: &Literal, b: &Literal) -> bool {
    match (a.desugared(), b.desugared()) {
        (Literal::Float(x), Literal::Float(y)) => x.to_bits() == y.to_bits(),
        (Literal::List(xs), Literal::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(&ys).all(|(x, y)| same_value(x, y))
        }
        (a, b) => a == b,
    }
}

impl JoinSemiLattice for Constant {
    fn bottom() -> Self {
        Constant::Bottom
    }

    fn join(&mut self, other: &Self) -> bool {
        let next = match (&*self, other) {
            (_, Constant::Bottom) | (Constant::Top, _) => return false,
            (Constant::Known(a), Constant::Known(b)) if same_value(a, b) => return false,
            (Constant::Bottom, _) => other.clone(),
            (Constant::Known(_), Constant::Known(_) | Constant::Top) => Constant::Top,
        };
        *self = next;
        true
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Constant::Bottom, _) | (_, Constant::Top) => true,
            (Constant::Known(a), Constant::Known(b)) => same_value(a, b),
            (Constant::Known(_) | Constant::Top, Constant::Bottom) | (Constant::Top, Constant::Known(_)) => false,
        }
    }
}

/// The lattice has height three, so every ascending chain is finite and
/// the widening is the join.
impl Widen for Constant {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

pub struct ConstantDomain;

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<Constant> {
        vec![
            Constant::Bottom,
            Constant::Known(Literal::Int(0)),
            Constant::Known(Literal::Int(1)),
            Constant::Known(Literal::Bool(true)),
            Constant::Known(Literal::Float(0.0)),
            Constant::Known(Literal::Float(-0.0)),
            Constant::Known(Literal::Float(f64::NAN)),
            Constant::Top,
        ]
    }

    fn joined(a: &Constant, b: &Constant) -> Constant {
        let mut out = a.clone();
        out.join(b);
        out
    }

    /// Equality of lattice elements, which is mutual order; a `Known`
    /// `NaN` is not `==` to itself under `Literal`'s `PartialEq`.
    fn same(a: &Constant, b: &Constant) -> bool {
        a.le(b) && b.le(a)
    }

    #[test]
    fn join_is_associative_commutative_and_idempotent() {
        let all = samples();
        for a in &all {
            assert!(same(&joined(a, a), a), "{a:?} ⊔ {a:?}");
            for b in &all {
                assert!(same(&joined(a, b), &joined(b, a)), "{a:?} ⊔ {b:?}");
                for c in &all {
                    assert!(same(&joined(&joined(a, b), c), &joined(a, &joined(b, c))), "{a:?} {b:?} {c:?}");
                }
            }
        }
    }

    #[test]
    fn bottom_is_the_unit_of_join() {
        for a in &samples() {
            assert!(same(&joined(&Constant::bottom(), a), a));
            assert!(same(&joined(a, &Constant::bottom()), a));
        }
    }

    #[test]
    fn le_agrees_with_join() {
        let all = samples();
        for a in &all {
            for b in &all {
                assert_eq!(a.le(b), same(&joined(a, b), b), "{a:?} ≤ {b:?}");
            }
        }
    }

    #[test]
    fn join_reports_growth_exactly() {
        let all = samples();
        for a in &all {
            for b in &all {
                let mut grown = a.clone();
                let grew = grown.join(b);
                assert_eq!(grew, !same(&grown, a), "{a:?} ⊔ {b:?}");
            }
        }
    }

    #[test]
    fn zero_and_negative_zero_are_two_values() {
        let zero = Constant::Known(Literal::Float(0.0));
        let negative = Constant::Known(Literal::Float(-0.0));
        assert_eq!(joined(&zero, &negative), Constant::Top);
    }

    #[test]
    fn a_suffix_is_the_type_not_the_value() {
        let plain = Constant::Known(Literal::Int(7));
        let suffixed = Constant::Known(Literal::IntOf(acvus_ast::SuffixedInt {
            value: 7,
            width: acvus_ast::IntWidth::U8,
        }));
        assert!(same(&joined(&plain, &suffixed), &plain));
    }

    #[test]
    fn widening_terminates_on_an_ascending_chain() {
        let chain = (0..100).map(|n| Constant::Known(Literal::Int(n)));
        let mut fact = Constant::bottom();
        let mut rises = 0;
        for next in chain {
            if fact.widen(&next) {
                rises += 1;
            }
        }
        assert_eq!(fact, Constant::Top);
        assert!(rises <= 2, "a chain of height three rises at most twice, rose {rises}");
    }
}
