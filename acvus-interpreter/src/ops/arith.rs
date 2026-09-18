//! Arithmetic and logic, one operation per operator at one operand type.
//!
//! An integer operation is the same Rust operator in a release build, at
//! the operand's width, panic messages included (RFC-0037). Every operand
//! and every result here is a word, so a run is two `word` loads and one
//! `set_word` store, and the kind byte the frame wrote stands (RFC-0052 §5).

use std::marker::PhantomData;

use acvus_ast::{BinOp, UnaryOp};
use acvus_mir::ty::IntTy;

use crate::code::{Off, Op};
use crate::machine::Machine;
use crate::value::Kind;

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

/// The registers a two-operand operation names.
#[derive(Clone, Copy)]
pub struct Binary {
    pub dst: Off,
    pub l: Off,
    pub r: Off,
}

/// The registers a one-operand operation names.
#[derive(Clone, Copy)]
pub struct Unary {
    pub dst: Off,
    pub src: Off,
}

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

/// The word-level integer operations: the operand words, read at `T`, and
/// the result word a register holds under the kind its frame wrote.
pub mod word {
    use super::*;

    macro_rules! int_words {
        ($( $name:ident ($a:ident, $b:ident) $body:block )*) => {
            $(
                #[inline]
                pub fn $name<T>(left: u64, right: u64) -> u64
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

    macro_rules! int_compares {
        ($( $name:ident ($a:ident, $b:ident) $body:block )*) => {
            $(
                #[inline]
                pub fn $name<T>(left: u64, right: u64) -> bool
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
        add(a, b) { a.wrapping_add(b).word() }
        sub(a, b) { a.wrapping_sub(b).word() }
        mul(a, b) { a.wrapping_mul(b).word() }
        div(a, b) {
            assert!(!b.is_zero(), "{DIVIDE_BY_ZERO}");
            a.checked_div(b).unwrap_or_else(|| panic!("{DIVIDE_OVERFLOW}")).word()
        }
        rem(a, b) {
            assert!(!b.is_zero(), "{REM_BY_ZERO}");
            a.checked_rem(b).unwrap_or_else(|| panic!("{REM_OVERFLOW}")).word()
        }
        bit_and(a, b) { a.bitand(b).word() }
        bit_or(a, b) { a.bitor(b).word() }
        bit_xor(a, b) { a.bitxor(b).word() }
    }

    int_compares! {
        eq(a, b) { a.eq(b) }
        neq(a, b) { !a.eq(b) }
        lt(a, b) { a < b }
        gt(a, b) { a > b }
        lte(a, b) { a <= b }
        gte(a, b) { a >= b }
    }

    #[inline]
    pub fn shl<T>(left: u64, right: u64) -> u64
    where
        T: Int,
    {
        T::read(left).wrapping_shl(shift_amount::<T>(right)).word()
    }

    #[inline]
    pub fn shr<T>(left: u64, right: u64) -> u64
    where
        T: Int,
    {
        T::read(left).wrapping_shr(shift_amount::<T>(right)).word()
    }

    #[inline]
    pub fn neg<T>(operand: u64) -> u64
    where
        T: Int,
    {
        T::read(operand).wrapping_neg().word()
    }
}

/// The float operations on their operands.
pub mod float_word {
    #[inline]
    pub fn add_f64(a: f64, b: f64) -> f64 {
        a + b
    }
    #[inline]
    pub fn sub_f64(a: f64, b: f64) -> f64 {
        a - b
    }
    #[inline]
    pub fn mul_f64(a: f64, b: f64) -> f64 {
        a * b
    }
    #[inline]
    pub fn div_f64(a: f64, b: f64) -> f64 {
        a / b
    }
    #[inline]
    pub fn rem_f64(a: f64, b: f64) -> f64 {
        a % b
    }
    #[inline]
    pub fn neg_f64(a: f64) -> f64 {
        -a
    }

    #[inline]
    pub fn eq_f64(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }
    #[inline]
    pub fn neq_f64(a: f64, b: f64) -> bool {
        a.to_bits() != b.to_bits()
    }
    #[inline]
    pub fn lt_f64(a: f64, b: f64) -> bool {
        a.total_cmp(&b).is_lt()
    }
    #[inline]
    pub fn gt_f64(a: f64, b: f64) -> bool {
        a.total_cmp(&b).is_gt()
    }
    #[inline]
    pub fn lte_f64(a: f64, b: f64) -> bool {
        a.total_cmp(&b).is_le()
    }
    #[inline]
    pub fn gte_f64(a: f64, b: f64) -> bool {
        a.total_cmp(&b).is_ge()
    }
}

macro_rules! int_ops {
    ($( $op:ident = $f:ident -> $result:ident ),* $(,)?) => {
        $(
            pub struct $op<T>
            where
                T: Int,
            {
                slots: Binary,
                width: PhantomData<fn() -> T>,
            }

            impl<T> $op<T>
            where
                T: Int,
            {
                pub fn new(slots: Binary) -> $op<T> {
                    $op {
                        slots,
                        width: PhantomData,
                    }
                }
            }

            impl<T> Op for $op<T>
            where
                T: Int,
            {
                #[inline]
                fn run(&self, m: &mut Machine<'_>) {
                    let regs = m.regs();
                    let bits = word::$f::<T>(regs.word(self.slots.l), regs.word(self.slots.r));
                    regs.set_word(self.slots.dst, $result(bits));
                }
            }
        )*
    };
}

#[inline(always)]
fn as_word(bits: u64) -> u64 {
    bits
}

#[inline(always)]
fn as_bool_word(held: bool) -> u64 {
    held as u64
}

int_ops!(
    Add = add -> as_word,
    Sub = sub -> as_word,
    Mul = mul -> as_word,
    Div = div -> as_word,
    Rem = rem -> as_word,
    BitAnd = bit_and -> as_word,
    BitOr = bit_or -> as_word,
    BitXor = bit_xor -> as_word,
    Shl = shl -> as_word,
    Shr = shr -> as_word,
    Eq = eq -> as_bool_word,
    Neq = neq -> as_bool_word,
    Lt = lt -> as_bool_word,
    Gt = gt -> as_bool_word,
    Lte = lte -> as_bool_word,
    Gte = gte -> as_bool_word,
);

pub struct Neg<T>
where
    T: Int,
{
    slots: Unary,
    width: PhantomData<fn() -> T>,
}

impl<T> Neg<T>
where
    T: Int,
{
    pub fn new(slots: Unary) -> Neg<T> {
        Neg {
            slots,
            width: PhantomData,
        }
    }
}

impl<T> Op for Neg<T>
where
    T: Int,
{
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let bits = word::neg::<T>(regs.word(self.slots.src));
        regs.set_word(self.slots.dst, bits);
    }
}

macro_rules! float_ops {
    ($( $op:ident = $f:ident -> $result:ident ),* $(,)?) => {
        $(
            pub struct $op {
                pub slots: Binary,
            }

            impl Op for $op {
                #[inline]
                fn run(&self, m: &mut Machine<'_>) {
                    let regs = m.regs();
                    let left = f64::from_bits(regs.word(self.slots.l));
                    let right = f64::from_bits(regs.word(self.slots.r));
                    regs.set_word(self.slots.dst, $result(float_word::$f(left, right)));
                }
            }
        )*
    };
}

#[inline(always)]
fn as_float_word(held: f64) -> u64 {
    held.to_bits()
}

float_ops!(
    AddF64 = add_f64 -> as_float_word,
    SubF64 = sub_f64 -> as_float_word,
    MulF64 = mul_f64 -> as_float_word,
    DivF64 = div_f64 -> as_float_word,
    RemF64 = rem_f64 -> as_float_word,
    EqF64 = eq_f64 -> as_bool_word,
    NeqF64 = neq_f64 -> as_bool_word,
    LtF64 = lt_f64 -> as_bool_word,
    GtF64 = gt_f64 -> as_bool_word,
    LteF64 = lte_f64 -> as_bool_word,
    GteF64 = gte_f64 -> as_bool_word,
);

pub struct NegF64 {
    pub slots: Unary,
}

impl Op for NegF64 {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let operand = f64::from_bits(regs.word(self.slots.src));
        regs.set_word(self.slots.dst, float_word::neg_f64(operand).to_bits());
    }
}

macro_rules! bool_ops {
    ($( $op:ident ($a:ident, $b:ident) $body:block )*) => {
        $(
            pub struct $op {
                pub slots: Binary,
            }

            impl Op for $op {
                #[inline]
                fn run(&self, m: &mut Machine<'_>) {
                    let regs = m.regs();
                    let $a = regs.word(self.slots.l) != 0;
                    let $b = regs.word(self.slots.r) != 0;
                    regs.set_word(self.slots.dst, as_bool_word($body));
                }
            }
        )*
    };
}

bool_ops! {
    EqBool(a, b) { a == b }
    NeqBool(a, b) { a != b }
    XorBool(a, b) { a ^ b }
}

pub struct NotBool {
    pub slots: Unary,
}

impl Op for NotBool {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let held = regs.word(self.slots.src) != 0;
        regs.set_word(self.slots.dst, as_bool_word(!held));
    }
}

fn int_op<T>(op: BinOp, slots: Binary) -> Box<dyn Op>
where
    T: Int,
{
    match op {
        BinOp::Add => Box::new(Add::<T>::new(slots)),
        BinOp::Sub => Box::new(Sub::<T>::new(slots)),
        BinOp::Mul => Box::new(Mul::<T>::new(slots)),
        BinOp::Div => Box::new(Div::<T>::new(slots)),
        BinOp::Mod => Box::new(Rem::<T>::new(slots)),
        BinOp::Eq => Box::new(Eq::<T>::new(slots)),
        BinOp::Neq => Box::new(Neq::<T>::new(slots)),
        BinOp::Lt => Box::new(Lt::<T>::new(slots)),
        BinOp::Gt => Box::new(Gt::<T>::new(slots)),
        BinOp::Lte => Box::new(Lte::<T>::new(slots)),
        BinOp::Gte => Box::new(Gte::<T>::new(slots)),
        BinOp::BitAnd => Box::new(BitAnd::<T>::new(slots)),
        BinOp::BitOr => Box::new(BitOr::<T>::new(slots)),
        BinOp::Xor => Box::new(BitXor::<T>::new(slots)),
        BinOp::Shl => Box::new(Shl::<T>::new(slots)),
        BinOp::Shr => Box::new(Shr::<T>::new(slots)),
        other => panic!("unsupported int binop {other:?}"),
    }
}

/// The operation a binary operator at an integer width prepares to.
pub fn int_binop(op: BinOp, k: IntTy, slots: Binary) -> Box<dyn Op> {
    for_int_ty!(k, |T| int_op::<T>(op, slots))
}

/// The operation a binary operator at `Float` prepares to.
pub fn float_binop(op: BinOp, slots: Binary) -> Box<dyn Op> {
    match op {
        BinOp::Add => Box::new(AddF64 { slots }),
        BinOp::Sub => Box::new(SubF64 { slots }),
        BinOp::Mul => Box::new(MulF64 { slots }),
        BinOp::Div => Box::new(DivF64 { slots }),
        BinOp::Mod => Box::new(RemF64 { slots }),
        BinOp::Eq => Box::new(EqF64 { slots }),
        BinOp::Neq => Box::new(NeqF64 { slots }),
        BinOp::Lt => Box::new(LtF64 { slots }),
        BinOp::Gt => Box::new(GtF64 { slots }),
        BinOp::Lte => Box::new(LteF64 { slots }),
        BinOp::Gte => Box::new(GteF64 { slots }),
        other => panic!("unsupported float binop {other:?}"),
    }
}

/// The operation a binary operator at `Bool` prepares to.
pub fn bool_binop(op: BinOp, slots: Binary) -> Box<dyn Op> {
    match op {
        BinOp::Eq => Box::new(EqBool { slots }),
        BinOp::Neq => Box::new(NeqBool { slots }),
        BinOp::Xor => Box::new(XorBool { slots }),
        other => panic!("unsupported bool binop {other:?}"),
    }
}

/// The operation a unary operator at an integer width prepares to.
pub fn int_unaryop(op: UnaryOp, k: IntTy, slots: Unary) -> Box<dyn Op> {
    match op {
        UnaryOp::Neg => for_int_ty!(k, |T| Box::new(Neg::<T>::new(slots)) as Box<dyn Op>),
        other => panic!("unary {other:?} on an integer"),
    }
}

/// The operation a unary operator at `Float` prepares to.
pub fn float_unaryop(op: UnaryOp, slots: Unary) -> Box<dyn Op> {
    match op {
        UnaryOp::Neg => Box::new(NegF64 { slots }),
        other => panic!("unary {other:?} on a float"),
    }
}

/// The operation a unary operator at `Bool` prepares to.
pub fn bool_unaryop(op: UnaryOp, slots: Unary) -> Box<dyn Op> {
    match op {
        UnaryOp::Not => Box::new(NotBool { slots }),
        other => panic!("unary {other:?} on a bool"),
    }
}

#[cfg(test)]
mod primitive_operator_tests {
    use super::*;

    fn i64_op<F>(f: F, a: i64, b: i64) -> i64
    where
        F: FnOnce(u64, u64) -> u64,
    {
        f(a as u64, b as u64) as i64
    }

    #[test]
    fn a_subtraction_within_the_width_is_the_difference() {
        assert_eq!(i64_op(word::sub::<i64>, 1, 2), -1);
    }

    #[test]
    fn addition_subtraction_and_multiplication_past_the_width_wrap() {
        assert_eq!(i64_op(word::add::<i64>, i64::MAX, 1), i64::MIN);
        assert_eq!(i64_op(word::sub::<i64>, i64::MIN, 1), i64::MAX);
        assert_eq!(i64_op(word::mul::<i64>, i64::MIN, -1), i64::MIN);
    }

    #[test]
    fn negation_of_the_minimum_is_the_minimum() {
        assert_eq!(word::neg::<i64>(i64::MIN as u64) as i64, i64::MIN);
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
        assert_eq!(i64_op(word::shl::<i64>, 1, -1), i64::MIN);
        assert_eq!(i64_op(word::shl::<i64>, 1, 64), 1);
        assert_eq!(i64_op(word::shl::<i64>, 1, 65), 2);
        assert_eq!(i64_op(word::shl::<i64>, 1, i64::MIN), 1);
        assert_eq!(i64_op(word::shr::<i64>, -1, -1), -1);
        assert_eq!(i64_op(word::shr::<i64>, 256, 65), 128);
        assert_eq!(i64_op(word::shl::<u8>, 1, 255), 128);
    }

    #[test]
    fn float_equality_is_bit_identity() {
        assert!(float_word::eq_f64(f64::NAN, f64::NAN));
        assert!(float_word::neq_f64(0.0, -0.0));
        assert!(float_word::eq_f64(1.5, 1.5));
    }

    #[test]
    fn float_order_is_the_total_order() {
        assert!(float_word::lt_f64(-0.0, 0.0));
        assert!(float_word::lt_f64(f64::INFINITY, f64::NAN));
        assert!(float_word::gt_f64(-1.0, -f64::NAN));
        assert!(float_word::lte_f64(2.0, 2.0));
    }

    #[test]
    fn an_integer_is_read_at_its_width() {
        assert_eq!(i64_op(word::add::<u8>, 200, 55), 255);
        assert_eq!(<i8 as Int>::read(0xFF), -1);
    }

    #[test]
    fn an_addition_past_the_narrow_width_wraps() {
        assert_eq!(i64_op(word::add::<u8>, 200, 56), 0);
        assert_eq!(i64_op(word::add::<u8>, 200, 57), 1);
    }
}
