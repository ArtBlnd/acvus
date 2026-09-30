//! The constant domain: bottom, one known value, or top; height 3, so its
//! widening is its join, stated. Abstract operations compute as the language
//! does (operators §1): an operation that would trap on the known operands
//! is top, since the trap is the run's, not a value.

use std::cmp::Ordering;

use acvus_ast::Literal;

use crate::ir::{BinOp, Overflow, UnaryOp};
use crate::passes::domain::{
    Arithmetic, Comparison, JoinSemiLattice, Operation, Scalar, ValueDomain, Widen, not_an_operation,
};
use crate::ty::{CastTy, IntTy};

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

impl ValueDomain for ConstantDomain {
    type Value = Constant;

    fn top(&self) -> Constant {
        Constant::Top
    }

    fn constant(&self, literal: &Literal, ty: CastTy) -> Constant {
        Constant::Known(Scalar::of(literal, ty).literal())
    }

    fn binary(&self, op: BinOp, ty: CastTy, left: &Constant, right: &Constant) -> Constant {
        match (left, right) {
            (Constant::Bottom, _) | (_, Constant::Bottom) => Constant::Bottom,
            (Constant::Top, _) | (_, Constant::Top) => Constant::Top,
            (Constant::Known(left), Constant::Known(right)) => {
                let right = match op {
                    BinOp::Shl(_) | BinOp::Shr(_) => Scalar::amount(right),
                    _ => Scalar::of(right, ty),
                };
                outcome(binary(op, ty, Scalar::of(left, ty), right))
            }
        }
    }

    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &Constant) -> Constant {
        match operand {
            Constant::Bottom => Constant::Bottom,
            Constant::Top => Constant::Top,
            Constant::Known(operand) => outcome(unary(op, ty, Scalar::of(operand, ty))),
        }
    }

    fn cast(&self, from: CastTy, to: CastTy, operand: &Constant) -> Constant {
        match operand {
            Constant::Bottom => Constant::Bottom,
            Constant::Top => Constant::Top,
            Constant::Known(operand) => Constant::Known(cast(from, to, Scalar::of(operand, from)).literal()),
        }
    }
}

/// What an operation on known operands gives: its value, or a trap, which
/// belongs to the run and is no value.
enum Computed {
    Value(Scalar),
    Traps,
}

fn outcome(computed: Computed) -> Constant {
    match computed {
        Computed::Value(value) => Constant::Known(value.literal()),
        Computed::Traps => Constant::Top,
    }
}

/// The exact result where it fits `width`, else the trap.
fn fitting(exact: Option<i128>, width: IntTy) -> Computed {
    match exact.filter(|n| width.holds(*n)) {
        Some(n) => Computed::Value(Scalar::Int(n)),
        None => Computed::Traps,
    }
}

/// `n` modulo `2^bits`, read at `width`'s signedness. `n` is congruent to
/// the exact result modulo `2^128`, which `2^bits` divides.
fn wrapped(n: i128, width: IntTy) -> Computed {
    Computed::Value(Scalar::Int(width.read(n as u64)))
}

/// A trapping shift's amount: where it is the width or more, or negative,
/// the shift traps.
fn checked_amount(amount: i128, width: IntTy) -> Option<u32> {
    u32::try_from(amount).ok().filter(|amount| *amount < width.bits())
}

/// A wrapping shift's amount, modulo the width, as Rust's `wrapping_shl`.
fn wrapped_amount(amount: i128, width: IntTy) -> u32 {
    (amount as u32) & (width.bits() - 1)
}

/// `left op right` at `ty`, as the language computes it (operators §1-2,
/// §4).
fn binary(op: BinOp, ty: CastTy, left: Scalar, right: Scalar) -> Computed {
    let value = Computed::Value;
    let arithmetic = match Operation::of(op) {
        Operation::Compare(comparison) => return value(Scalar::Bool(compare(comparison, ty, left, right))),
        Operation::Compute(arithmetic) => arithmetic,
    };
    match (left, right) {
        (Scalar::Int(a), Scalar::Int(b)) => {
            let CastTy::Int(width) = ty else { not_an_operation(op, ty) };
            match arithmetic {
                Arithmetic::Add(Overflow::Trap) => fitting(a.checked_add(b), width),
                Arithmetic::Sub(Overflow::Trap) => fitting(a.checked_sub(b), width),
                Arithmetic::Mul(Overflow::Trap) => fitting(a.checked_mul(b), width),
                Arithmetic::Add(Overflow::Wrap) => wrapped(a.wrapping_add(b), width),
                Arithmetic::Sub(Overflow::Wrap) => wrapped(a.wrapping_sub(b), width),
                Arithmetic::Mul(Overflow::Wrap) => wrapped(a.wrapping_mul(b), width),
                Arithmetic::Div | Arithmetic::Mod if b == 0 => Computed::Traps,
                Arithmetic::Div => fitting(Some(a / b), width),
                Arithmetic::Mod if !width.holds(a / b) => Computed::Traps,
                Arithmetic::Mod => value(Scalar::Int(a % b)),
                Arithmetic::Shl(Overflow::Trap) => match checked_amount(b, width) {
                    Some(amount) => wrapped(a << amount, width),
                    None => Computed::Traps,
                },
                Arithmetic::Shl(Overflow::Wrap) => wrapped(a << wrapped_amount(b, width), width),
                Arithmetic::Shr(Overflow::Trap) => match checked_amount(b, width) {
                    Some(amount) => value(Scalar::Int(a >> amount)),
                    None => Computed::Traps,
                },
                Arithmetic::Shr(Overflow::Wrap) => value(Scalar::Int(a >> wrapped_amount(b, width))),
                Arithmetic::BitAnd => value(Scalar::Int(a & b)),
                Arithmetic::BitOr => value(Scalar::Int(a | b)),
                Arithmetic::Xor => value(Scalar::Int(a ^ b)),
                Arithmetic::Min => value(Scalar::Int(a.min(b))),
                Arithmetic::Max => value(Scalar::Int(a.max(b))),
                Arithmetic::And | Arithmetic::Or => not_an_operation(op, ty),
            }
        }
        (Scalar::F64(a), Scalar::F64(b)) => match arithmetic {
            Arithmetic::Add(_) => value(Scalar::F64(a + b)),
            Arithmetic::Sub(_) => value(Scalar::F64(a - b)),
            Arithmetic::Mul(_) => value(Scalar::F64(a * b)),
            Arithmetic::Div => value(Scalar::F64(a / b)),
            Arithmetic::Mod => value(Scalar::F64(a % b)),
            Arithmetic::And
            | Arithmetic::Or
            | Arithmetic::Xor
            | Arithmetic::BitAnd
            | Arithmetic::BitOr
            | Arithmetic::Shl(_)
            | Arithmetic::Shr(_)
            | Arithmetic::Min
            | Arithmetic::Max => not_an_operation(op, ty),
        },
        (Scalar::Bool(a), Scalar::Bool(b)) => match arithmetic {
            Arithmetic::And | Arithmetic::BitAnd => value(Scalar::Bool(a & b)),
            Arithmetic::Or | Arithmetic::BitOr => value(Scalar::Bool(a | b)),
            Arithmetic::Xor => value(Scalar::Bool(a ^ b)),
            Arithmetic::Add(_)
            | Arithmetic::Sub(_)
            | Arithmetic::Mul(_)
            | Arithmetic::Div
            | Arithmetic::Mod
            | Arithmetic::Shl(_)
            | Arithmetic::Shr(_)
            | Arithmetic::Min
            | Arithmetic::Max => not_an_operation(op, ty),
        },
        (Scalar::Int(_) | Scalar::F64(_) | Scalar::Char(_) | Scalar::Bool(_), _) => not_an_operation(op, ty),
    }
}

/// Whether `comparison` holds of two scalars of `ty`. Equality and ordering
/// are Rust's, but that an `f64` `NaN` equals itself (operators §4).
fn compare(comparison: Comparison, ty: CastTy, left: Scalar, right: Scalar) -> bool {
    let ordering = match (left, right) {
        (Scalar::Int(a), Scalar::Int(b)) => Some(a.cmp(&b)),
        (Scalar::F64(a), Scalar::F64(b)) => match comparison {
            Comparison::Eq | Comparison::Neq if a.is_nan() && b.is_nan() => Some(Ordering::Equal),
            _ => a.partial_cmp(&b),
        },
        (Scalar::Char(a), Scalar::Char(b)) => Some(a.cmp(&b)),
        (Scalar::Bool(a), Scalar::Bool(b)) => Some(a.cmp(&b)),
        (Scalar::Int(_) | Scalar::F64(_) | Scalar::Char(_) | Scalar::Bool(_), _) => {
            not_an_operation(comparison, ty)
        }
    };
    comparison.holds(ordering)
}

fn unary(op: UnaryOp, ty: CastTy, operand: Scalar) -> Computed {
    match (op, ty, operand) {
        (UnaryOp::Neg(Overflow::Trap), CastTy::Int(width), Scalar::Int(n)) if width.signed() => {
            fitting(n.checked_neg(), width)
        }
        (UnaryOp::Neg(Overflow::Wrap), CastTy::Int(width), Scalar::Int(n)) if width.signed() => {
            wrapped(n.wrapping_neg(), width)
        }
        (UnaryOp::Neg(_), CastTy::F64, Scalar::F64(x)) => Computed::Value(Scalar::F64(-x)),
        (UnaryOp::Not, CastTy::Bool, Scalar::Bool(b)) => Computed::Value(Scalar::Bool(!b)),
        _ => not_an_operation(op, ty),
    }
}

/// `value as to`, with Rust's values (operators §5): an integer wraps to
/// the target width, an `f64` saturates and `NaN` is `0` (Rust's float `as`
/// already gives both), and only a `u8` becomes a `char`.
fn cast(from: CastTy, to: CastTy, value: Scalar) -> Scalar {
    assert!(from.admits(to), "`{} as {}` is not a cast the language admits", from.name(), to.name());
    let wrap = |n: i128, width: IntTy| width.read(n as u64);
    match (value, to) {
        (Scalar::Int(n), CastTy::Int(width)) => Scalar::Int(wrap(n, width)),
        (Scalar::Int(n), CastTy::F64) => Scalar::F64(n as f64),
        (Scalar::Int(n), CastTy::Char) => Scalar::Char(char::from(n as u8)),
        (Scalar::F64(x), CastTy::Int(width)) => Scalar::Int((x as i128).clamp(width.min(), width.max())),
        (Scalar::F64(x), CastTy::F64) => Scalar::F64(x),
        (Scalar::Char(c), CastTy::Int(width)) => Scalar::Int(wrap(i128::from(u32::from(c)), width)),
        (Scalar::Bool(b), CastTy::Int(_)) => Scalar::Int(i128::from(b)),
        (value, to) => panic!("{value:?} as {} is admitted but has no value", to.name()),
    }
}

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

    // -- Value operations against the language's --------------------------

    const I8: CastTy = CastTy::Int(IntTy::I8);
    const U64: CastTy = CastTy::Int(IntTy::U64);
    const I64: CastTy = CastTy::Int(IntTy::I64);

    fn int(n: i128) -> Constant {
        Constant::Known(Literal::Int(n))
    }

    fn float(x: f64) -> Constant {
        Constant::Known(Literal::Float(x))
    }

    fn truth(b: bool) -> Constant {
        Constant::Known(Literal::Bool(b))
    }

    /// Rust's own result of the operation, or the trap as `Top`.
    fn rust(result: Option<i128>) -> Constant {
        result.map_or(Constant::Top, int)
    }

    #[test]
    fn integer_operations_compute_as_the_language_or_trap() {
        use BinOp::*;
        use Overflow::{Trap, Wrap};
        let max = i128::from(u64::MAX);
        let cases: Vec<(BinOp, CastTy, i128, i128, Constant)> = vec![
            (Add(Trap), I8, 100, 27, rust(100i8.checked_add(27).map(i128::from))),
            (Add(Trap), I8, 100, 28, rust(100i8.checked_add(28).map(i128::from))),
            (Add(Wrap), I8, 100, 28, int(i128::from(100i8.wrapping_add(28)))),
            (Sub(Trap), I8, -128, 1, rust((-128i8).checked_sub(1).map(i128::from))),
            (Sub(Wrap), I8, -128, 1, int(i128::from((-128i8).wrapping_sub(1)))),
            (Mul(Trap), I8, 16, 8, rust(16i8.checked_mul(8).map(i128::from))),
            (Mul(Wrap), I8, 16, 8, int(i128::from(16i8.wrapping_mul(8)))),
            (Div, I8, -128, -1, rust((-128i8).checked_div(-1).map(i128::from))),
            (Mod, I8, -128, -1, rust((-128i8).checked_rem(-1).map(i128::from))),
            (Div, I8, 7, 0, rust(7i8.checked_div(0).map(i128::from))),
            (Mod, I8, 7, 0, rust(7i8.checked_rem(0).map(i128::from))),
            (Div, I8, -7, 2, int(i128::from(-7i8 / 2))),
            (Mod, I8, -7, 2, int(i128::from(-7i8 % 2))),
            (Shl(Trap), I8, 1, 7, rust(1i8.checked_shl(7).map(i128::from))),
            (Shl(Trap), I8, 1, 8, rust(1i8.checked_shl(8).map(i128::from))),
            (Shl(Trap), I8, 1, -1, Constant::Top),
            (Shl(Wrap), I8, 1, 9, int(i128::from(1i8.wrapping_shl(9)))),
            (Shr(Trap), I8, -128, 7, rust((-128i8).checked_shr(7).map(i128::from))),
            (Shr(Trap), I8, -128, 8, rust((-128i8).checked_shr(8).map(i128::from))),
            (Shr(Wrap), I8, -128, 9, int(i128::from((-128i8).wrapping_shr(9)))),
            (Add(Trap), U64, max, 1, rust(u64::MAX.checked_add(1).map(i128::from))),
            (Add(Wrap), U64, max, 1, int(i128::from(u64::MAX.wrapping_add(1)))),
            (Sub(Wrap), U64, 0, 1, int(i128::from(0u64.wrapping_sub(1)))),
            (Mul(Trap), U64, max, max, rust(u64::MAX.checked_mul(u64::MAX).map(i128::from))),
            (Mul(Wrap), U64, max, max, int(i128::from(u64::MAX.wrapping_mul(u64::MAX)))),
            (Mul(Wrap), U64, max, 2, int(i128::from(u64::MAX.wrapping_mul(2)))),
            (Shl(Trap), U64, 1, 63, rust(1u64.checked_shl(63).map(i128::from))),
            (Shl(Trap), U64, 1, 64, rust(1u64.checked_shl(64).map(i128::from))),
            (Shl(Wrap), U64, 3, 64, int(i128::from(3u64.wrapping_shl(64)))),
            (Shr(Trap), U64, max, 63, rust(u64::MAX.checked_shr(63).map(i128::from))),
            (Div, U64, max, 2, int(i128::from(u64::MAX / 2))),
            (BitAnd, I8, -1, 5, int(i128::from(-1i8 & 5))),
            (BitOr, I8, -128, 1, int(i128::from(-128i8 | 1))),
            (Xor, U64, max, 1, int(i128::from(u64::MAX ^ 1))),
            (Min, I8, -3, 2, int(-3)),
            (Max, U64, max, 0, int(max)),
            (Lt, U64, max, 0, truth(false)),
            (Gte, I8, -1, -1, truth(true)),
            (Neq, I64, 4, 5, truth(true)),
        ];
        for (op, ty, a, b, expected) in cases {
            let got = ConstantDomain.binary(op, ty, &int(a), &int(b));
            assert_eq!(got, expected, "{a} {op:?} {b} at {}", ty.name());
        }
    }

    /// D79: a shift's amount keeps its own width, so an amount of a wider
    /// type is read as the integer it is: a trapping shift traps on it
    /// where it is at least the value's width, and a wrapping one takes it
    /// modulo that width, never truncated to the value's type first.
    #[test]
    fn a_shift_amount_is_read_at_its_own_width() {
        use BinOp::*;
        use Overflow::{Trap, Wrap};
        let wide = 1i128 << 40;
        assert_eq!(ConstantDomain.binary(Shl(Trap), CastTy::Int(IntTy::U8), &int(1), &int(wide)), Constant::Top);
        assert_eq!(ConstantDomain.binary(Shr(Trap), CastTy::Int(IntTy::U8), &int(128), &int(-1)), Constant::Top);
        assert_eq!(ConstantDomain.binary(Shl(Trap), CastTy::Int(IntTy::U8), &int(1), &int(7)), int(128));
        assert_eq!(ConstantDomain.binary(Shl(Wrap), CastTy::Int(IntTy::U8), &int(1), &int(wide + 1)), int(2));
    }

    #[test]
    fn negation_traps_at_the_minimum_only_when_trapping() {
        let domain = ConstantDomain;
        assert_eq!(domain.unary(UnaryOp::Neg(Overflow::Trap), I8, &int(-128)), Constant::Top);
        assert_eq!(domain.unary(UnaryOp::Neg(Overflow::Wrap), I8, &int(-128)), int(i128::from((-128i8).wrapping_neg())));
        assert_eq!(domain.unary(UnaryOp::Neg(Overflow::Trap), I8, &int(5)), int(-5));
        assert_eq!(domain.unary(UnaryOp::Not, CastTy::Bool, &truth(true)), truth(false));
    }

    #[test]
    fn float_operations_are_ieee_and_nan_equals_itself() {
        let f64_ty = CastTy::F64;
        let domain = ConstantDomain;
        let division = domain.binary(BinOp::Div, f64_ty, &float(1.0), &float(0.0));
        assert_eq!(division, float(f64::INFINITY), "division by zero is an infinity, not a trap");
        assert_eq!(domain.binary(BinOp::Eq, f64_ty, &float(f64::NAN), &float(f64::NAN)), truth(true));
        assert_eq!(domain.binary(BinOp::Neq, f64_ty, &float(f64::NAN), &float(f64::NAN)), truth(false));
        assert_eq!(domain.binary(BinOp::Lt, f64_ty, &float(f64::NAN), &float(1.0)), truth(false));
        assert_eq!(domain.binary(BinOp::Lte, f64_ty, &float(f64::NAN), &float(f64::NAN)), truth(false));
        assert_eq!(domain.binary(BinOp::Eq, f64_ty, &float(0.0), &float(-0.0)), truth(true));
    }

    #[test]
    fn casts_are_rusts() {
        let domain = ConstantDomain;
        let u8_ty = CastTy::Int(IntTy::U8);
        assert_eq!(domain.cast(I64, u8_ty, &int(300)), int(i128::from(300i64 as u8)));
        assert_eq!(domain.cast(I64, U64, &int(-1)), int(i128::from(-1i64 as u64)));
        assert_eq!(domain.cast(CastTy::F64, I8, &float(1e300)), int(i128::from(1e300f64 as i8)));
        assert_eq!(domain.cast(CastTy::F64, CastTy::Int(IntTy::I32), &float(f64::NAN)), int(0));
        assert_eq!(domain.cast(CastTy::F64, u8_ty, &float(-1.5)), int(i128::from(-1.5f64 as u8)));
        assert_eq!(domain.cast(CastTy::Char, u8_ty, &Constant::Known(Literal::Char('A'))), int(65));
        assert_eq!(domain.cast(u8_ty, CastTy::Char, &int(200)), Constant::Known(Literal::Char(200u8 as char)));
        assert_eq!(domain.cast(CastTy::Bool, I64, &truth(true)), int(1));
        assert_eq!(domain.cast(U64, CastTy::F64, &int(i128::from(u64::MAX))), float(u64::MAX as f64));
    }

    #[test]
    fn bottom_and_top_pass_through_an_operation() {
        let domain = ConstantDomain;
        let add = BinOp::Add(Overflow::Trap);
        assert_eq!(domain.binary(add, I64, &Constant::Bottom, &Constant::Top), Constant::Bottom);
        assert_eq!(domain.binary(add, I64, &int(1), &Constant::Top), Constant::Top);
        assert_eq!(domain.cast(I64, U64, &Constant::Bottom), Constant::Bottom);
    }

    #[test]
    #[should_panic(expected = "does not fit its type i8")]
    fn a_constant_that_does_not_fit_its_width_is_a_defect() {
        ConstantDomain.constant(&Literal::Int(200), I8);
    }

}
