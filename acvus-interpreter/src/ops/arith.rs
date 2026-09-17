//! Arithmetic and logic, one operation per operator at one operand type.
//!
//! An integer operation is the same Rust operator in a release build, at
//! the operand's width, panic messages included (RFC-0037).

use acvus_ast::{BinOp, UnaryOp};
use acvus_mir::ty::IntTy;

use crate::code::{Flow, Op, OpFn};
use crate::machine::Machine;
use crate::value::{Kind, Value};

/// Runs `$body` with `$t` the Rust integer type of an `IntTy`.
macro_rules! for_int_ty {
    ($k:expr, |$t:ident| $body:expr) => {
        match $k {
            IntTy::I8 => {
                type $t = i8;
                $body
            }
            IntTy::I16 => {
                type $t = i16;
                $body
            }
            IntTy::I32 => {
                type $t = i32;
                $body
            }
            IntTy::I64 => {
                type $t = i64;
                $body
            }
            IntTy::U8 => {
                type $t = u8;
                $body
            }
            IntTy::U16 => {
                type $t = u16;
                $body
            }
            IntTy::U32 => {
                type $t = u32;
                $body
            }
            IntTy::U64 => {
                type $t = u64;
                $body
            }
        }
    };
}

pub(crate) use for_int_ty;

/// One integer width, as the operations at that width read and write it.
pub trait Int: Copy + PartialOrd + 'static {
    const KIND: Kind;
    const SHIFT_MASK: u64;

    fn read(bits: u64) -> Self;
    fn word(self) -> u64;
    fn wide(self) -> i128;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn checked_div(self, other: Self) -> Option<Self>;
    fn checked_rem(self, other: Self) -> Option<Self>;
    fn wrapping_neg(self) -> Self;
    fn wrapping_shl(self, shift: u32) -> Self;
    fn wrapping_shr(self, shift: u32) -> Self;
    fn bitand(self, other: Self) -> Self;
    fn bitor(self, other: Self) -> Self;
    fn bitxor(self, other: Self) -> Self;
    fn is_zero(self) -> bool;
    fn eq(self, other: Self) -> bool;
}

macro_rules! impl_int {
    ($($t:ty => $k:ident),* $(,)?) => {
        $(impl Int for $t {
            const KIND: Kind = Kind::$k;
            const SHIFT_MASK: u64 = <$t>::BITS as u64 - 1;

            fn read(bits: u64) -> Self {
                IntTy::$k.read(bits) as $t
            }
            fn word(self) -> u64 {
                self as u64
            }
            fn wide(self) -> i128 {
                self as i128
            }
            fn wrapping_add(self, other: Self) -> Self {
                <$t>::wrapping_add(self, other)
            }
            fn wrapping_sub(self, other: Self) -> Self {
                <$t>::wrapping_sub(self, other)
            }
            fn wrapping_mul(self, other: Self) -> Self {
                <$t>::wrapping_mul(self, other)
            }
            fn checked_div(self, other: Self) -> Option<Self> {
                <$t>::checked_div(self, other)
            }
            fn checked_rem(self, other: Self) -> Option<Self> {
                <$t>::checked_rem(self, other)
            }
            fn wrapping_neg(self) -> Self {
                <$t>::wrapping_neg(self)
            }
            fn wrapping_shl(self, shift: u32) -> Self {
                <$t>::wrapping_shl(self, shift)
            }
            fn wrapping_shr(self, shift: u32) -> Self {
                <$t>::wrapping_shr(self, shift)
            }
            fn bitand(self, other: Self) -> Self {
                self & other
            }
            fn bitor(self, other: Self) -> Self {
                self | other
            }
            fn bitxor(self, other: Self) -> Self {
                self ^ other
            }
            fn is_zero(self) -> bool {
                self == 0
            }
            fn eq(self, other: Self) -> bool {
                self == other
            }
        })*
    };
}

impl_int! {
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
}

fn int<T>(value: T) -> Value
where
    T: Int,
{
    Value::inline(T::KIND, value.word())
}

const DIVIDE_BY_ZERO: &str = "attempt to divide by zero";
const REM_BY_ZERO: &str = "attempt to calculate the remainder with a divisor of zero";
const DIVIDE_OVERFLOW: &str = "attempt to divide with overflow";
const REM_OVERFLOW: &str = "attempt to calculate the remainder with overflow";

/// Rust's `<<` and `>>` take the amount modulo the width; `SHIFT_MASK` is
/// that modulus, so the masked amount is below `64` and reaches `u32`
/// whole, for an amount of either sign.
fn shift_amount<T>(bits: u64) -> u32
where
    T: Int,
{
    (T::read(bits).word() & T::SHIFT_MASK) as u32
}

/// The word-level integer operations: the operand words, read at `T`.
pub mod word {
    use super::*;

    macro_rules! int_words {
        ($( $name:ident ($a:ident, $b:ident) $body:block )*) => {
            $(
                #[inline]
                pub fn $name<T>(left: u64, right: u64) -> Value
                where
                    T: Int,
                {
                    let $a = T::read(left);
                    let $b = T::read(right);
                    $body
                }
            )*
        };
    }

    int_words! {
        add(a, b) { int(a.wrapping_add(b)) }
        sub(a, b) { int(a.wrapping_sub(b)) }
        mul(a, b) { int(a.wrapping_mul(b)) }
        div(a, b) {
            assert!(!b.is_zero(), "{DIVIDE_BY_ZERO}");
            int(a.checked_div(b).unwrap_or_else(|| panic!("{DIVIDE_OVERFLOW}")))
        }
        rem(a, b) {
            assert!(!b.is_zero(), "{REM_BY_ZERO}");
            int(a.checked_rem(b).unwrap_or_else(|| panic!("{REM_OVERFLOW}")))
        }
        eq(a, b) { Value::bool_(a.eq(b)) }
        neq(a, b) { Value::bool_(!a.eq(b)) }
        lt(a, b) { Value::bool_(a < b) }
        gt(a, b) { Value::bool_(a > b) }
        lte(a, b) { Value::bool_(a <= b) }
        gte(a, b) { Value::bool_(a >= b) }
        bit_and(a, b) { int(a.bitand(b)) }
        bit_or(a, b) { int(a.bitor(b)) }
        bit_xor(a, b) { int(a.bitxor(b)) }
    }

    #[inline]
    pub fn shl<T>(left: u64, right: u64) -> Value
    where
        T: Int,
    {
        int(T::read(left).wrapping_shl(shift_amount::<T>(right)))
    }

    #[inline]
    pub fn shr<T>(left: u64, right: u64) -> Value
    where
        T: Int,
    {
        int(T::read(left).wrapping_shr(shift_amount::<T>(right)))
    }

    #[inline]
    pub fn neg<T>(operand: u64) -> Value
    where
        T: Int,
    {
        int(T::read(operand).wrapping_neg())
    }
}

#[inline]
fn binary<F>(machine: &mut Machine<'_>, op: &Op, f: F) -> Flow
where
    F: FnOnce(u64, u64) -> Value,
{
    let left = machine.reg(op.b).bits();
    let right = machine.reg(op.c).bits();
    let value = f(left, right);
    machine.set(op.a, value);
    Flow::Next
}

#[inline]
fn unary<F>(machine: &mut Machine<'_>, op: &Op, f: F) -> Flow
where
    F: FnOnce(u64) -> Value,
{
    let operand = machine.reg(op.b).bits();
    let value = f(operand);
    machine.set(op.a, value);
    Flow::Next
}

macro_rules! int_ops {
    ($( $name:ident ),* $(,)?) => {
        $(
            pub fn $name<T>(machine: &mut Machine<'_>, op: &Op) -> Flow
            where
                T: Int,
            {
                binary(machine, op, word::$name::<T>)
            }
        )*
    };
}

int_ops!(
    add, sub, mul, div, rem, eq, neq, lt, gt, lte, gte, bit_and, bit_or, bit_xor, shl, shr
);

pub fn neg<T>(machine: &mut Machine<'_>, op: &Op) -> Flow
where
    T: Int,
{
    unary(machine, op, word::neg::<T>)
}

macro_rules! float_ops {
    ($( $name:ident ($a:ident, $b:ident) $body:block )*) => {
        /// The float operations on their operands.
        pub mod float_word {
            use super::*;

            $( #[inline] pub fn $name($a: f64, $b: f64) -> Value $body )*
        }

        $(
            pub fn $name(machine: &mut Machine<'_>, op: &Op) -> Flow {
                let left = machine.reg(op.b).as_float();
                let right = machine.reg(op.c).as_float();
                machine.set(op.a, float_word::$name(left, right));
                Flow::Next
            }
        )*
    };
}

float_ops! {
    add_f64(a, b) { Value::float(a + b) }
    sub_f64(a, b) { Value::float(a - b) }
    mul_f64(a, b) { Value::float(a * b) }
    div_f64(a, b) { Value::float(a / b) }
    rem_f64(a, b) { Value::float(a % b) }
    eq_f64(a, b) { Value::bool_(a.to_bits() == b.to_bits()) }
    neq_f64(a, b) { Value::bool_(a.to_bits() != b.to_bits()) }
    lt_f64(a, b) { Value::bool_(a.total_cmp(&b).is_lt()) }
    gt_f64(a, b) { Value::bool_(a.total_cmp(&b).is_gt()) }
    lte_f64(a, b) { Value::bool_(a.total_cmp(&b).is_le()) }
    gte_f64(a, b) { Value::bool_(a.total_cmp(&b).is_ge()) }
}

macro_rules! bool_ops {
    ($( $name:ident ($a:ident, $b:ident) $body:block )*) => {
        $(
            pub fn $name(machine: &mut Machine<'_>, op: &Op) -> Flow {
                let $a = machine.reg(op.b).as_bool();
                let $b = machine.reg(op.c).as_bool();
                machine.set(op.a, $body);
                Flow::Next
            }
        )*
    };
}

bool_ops! {
    and_bool(a, b) { Value::bool_(a && b) }
    or_bool(a, b) { Value::bool_(a || b) }
    eq_bool(a, b) { Value::bool_(a == b) }
    neq_bool(a, b) { Value::bool_(a != b) }
    xor_bool(a, b) { Value::bool_(a ^ b) }
}

pub fn neg_f64(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = Value::float(-machine.reg(op.b).as_float());
    machine.set(op.a, value);
    Flow::Next
}

pub fn not_bool(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = Value::bool_(!machine.reg(op.b).as_bool());
    machine.set(op.a, value);
    Flow::Next
}

fn int_op<T>(op: BinOp) -> OpFn
where
    T: Int,
{
    match op {
        BinOp::Add => add::<T>,
        BinOp::Sub => sub::<T>,
        BinOp::Mul => mul::<T>,
        BinOp::Div => div::<T>,
        BinOp::Mod => rem::<T>,
        BinOp::Eq => eq::<T>,
        BinOp::Neq => neq::<T>,
        BinOp::Lt => lt::<T>,
        BinOp::Gt => gt::<T>,
        BinOp::Lte => lte::<T>,
        BinOp::Gte => gte::<T>,
        BinOp::BitAnd => bit_and::<T>,
        BinOp::BitOr => bit_or::<T>,
        BinOp::Xor => bit_xor::<T>,
        BinOp::Shl => shl::<T>,
        BinOp::Shr => shr::<T>,
        BinOp::And | BinOp::Or => panic!("And/Or on an integer"),
    }
}

/// The operation a binary operator at an integer width prepares to.
pub fn int_binop(op: BinOp, k: IntTy) -> OpFn {
    for_int_ty!(k, |T| int_op::<T>(op))
}

/// The operation a binary operator at `Float` prepares to.
pub fn float_binop(op: BinOp) -> OpFn {
    match op {
        BinOp::Add => add_f64,
        BinOp::Sub => sub_f64,
        BinOp::Mul => mul_f64,
        BinOp::Div => div_f64,
        BinOp::Mod => rem_f64,
        BinOp::Eq => eq_f64,
        BinOp::Neq => neq_f64,
        BinOp::Lt => lt_f64,
        BinOp::Gt => gt_f64,
        BinOp::Lte => lte_f64,
        BinOp::Gte => gte_f64,
        other => panic!("unsupported float binop {other:?}"),
    }
}

/// The operation a binary operator at `Bool` prepares to.
pub fn bool_binop(op: BinOp) -> OpFn {
    match op {
        BinOp::And => and_bool,
        BinOp::Or => or_bool,
        BinOp::Eq => eq_bool,
        BinOp::Neq => neq_bool,
        BinOp::Xor => xor_bool,
        other => panic!("unsupported bool binop {other:?}"),
    }
}

/// The operation a unary operator at an integer width prepares to.
pub fn int_unaryop(op: UnaryOp, k: IntTy) -> OpFn {
    match op {
        UnaryOp::Neg => for_int_ty!(k, |T| neg::<T> as OpFn),
        other => panic!("unary {other:?} on an integer"),
    }
}

/// The operation a unary operator at `Float` prepares to.
pub fn float_unaryop(op: UnaryOp) -> OpFn {
    match op {
        UnaryOp::Neg => neg_f64,
        other => panic!("unary {other:?} on a float"),
    }
}

/// The operation a unary operator at `Bool` prepares to.
pub fn bool_unaryop(op: UnaryOp) -> OpFn {
    match op {
        UnaryOp::Not => not_bool,
        other => panic!("unary {other:?} on a bool"),
    }
}

#[cfg(test)]
mod primitive_operator_tests {
    use super::*;

    fn i64_op<F>(f: F, a: i64, b: i64) -> Value
    where
        F: FnOnce(u64, u64) -> Value,
    {
        f(a as u64, b as u64)
    }

    fn float<F>(f: F, a: f64, b: f64) -> bool
    where
        F: FnOnce(f64, f64) -> Value,
    {
        f(a, b).as_bool()
    }

    #[test]
    fn a_subtraction_within_the_width_is_the_difference() {
        assert_eq!(i64_op(word::sub::<i64>, 1, 2).as_int(), -1);
    }

    #[test]
    fn addition_subtraction_and_multiplication_past_the_width_wrap() {
        assert_eq!(i64_op(word::add::<i64>, i64::MAX, 1).as_int(), i64::MIN);
        assert_eq!(i64_op(word::sub::<i64>, i64::MIN, 1).as_int(), i64::MAX);
        assert_eq!(i64_op(word::mul::<i64>, i64::MIN, -1).as_int(), i64::MIN);
    }

    #[test]
    fn negation_of_the_minimum_is_the_minimum() {
        assert_eq!(word::neg::<i64>(i64::MIN as u64).as_int(), i64::MIN);
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn a_division_by_zero_panics() {
        i64_op(word::div::<i64>, 1, 0);
    }

    #[test]
    #[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
    fn a_remainder_by_zero_panics() {
        i64_op(word::rem::<i64>, 1, 0);
    }

    #[test]
    #[should_panic(expected = "attempt to divide with overflow")]
    fn dividing_the_minimum_by_minus_one_panics() {
        i64_op(word::div::<i64>, i64::MIN, -1);
    }

    #[test]
    #[should_panic(expected = "attempt to calculate the remainder with overflow")]
    fn the_remainder_of_the_minimum_by_minus_one_panics() {
        i64_op(word::rem::<i64>, i64::MIN, -1);
    }

    /// Each expected value is the measured output of a `rustc -O` build of
    /// the same expression; this test runs in debug, where `1i64 << -1`
    /// panics instead of masking.
    #[test]
    fn a_shift_takes_its_amount_modulo_the_width() {
        assert_eq!(i64_op(word::shl::<i64>, 1, -1).as_int(), i64::MIN);
        assert_eq!(i64_op(word::shl::<i64>, 1, 64).as_int(), 1);
        assert_eq!(i64_op(word::shl::<i64>, 1, 65).as_int(), 2);
        assert_eq!(i64_op(word::shl::<i64>, 1, i64::MIN).as_int(), 1);
        assert_eq!(i64_op(word::shr::<i64>, -1, -1).as_int(), -1);
        assert_eq!(i64_op(word::shr::<i64>, 256, 65).as_int(), 128);
        assert_eq!(i64_op(word::shl::<u8>, 1, 255).as_int(), 128);
    }

    #[test]
    fn float_equality_is_bit_identity() {
        assert!(float(float_word::eq_f64, f64::NAN, f64::NAN));
        assert!(float(float_word::neq_f64, 0.0, -0.0));
        assert!(float(float_word::eq_f64, 1.5, 1.5));
    }

    #[test]
    fn float_order_is_the_total_order() {
        assert!(float(float_word::lt_f64, -0.0, 0.0));
        assert!(float(float_word::lt_f64, f64::INFINITY, f64::NAN));
        assert!(float(float_word::gt_f64, -1.0, -f64::NAN));
        assert!(float(float_word::lte_f64, 2.0, 2.0));
    }

    #[test]
    fn an_integer_is_read_at_its_width() {
        assert_eq!(i64_op(word::add::<u8>, 200, 55).as_int(), 255);
        assert_eq!(<i8 as Int>::read(0xFF), -1);
    }

    #[test]
    fn an_addition_past_the_narrow_width_wraps() {
        assert_eq!(i64_op(word::add::<u8>, 200, 56).as_int(), 0);
        assert_eq!(i64_op(word::add::<u8>, 200, 57).as_int(), 1);
    }
}
