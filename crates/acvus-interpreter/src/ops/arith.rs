//! Arithmetic and logic, one operation per operator at one operand type.
//!
//! A trapping integer operation (`ir::Overflow::Trap`, the source's) is the
//! same Rust operator in a debug build, at the operand's width, panic
//! messages included: `+`, `-`, `*` and negation trap where the exact result
//! does not fit, and a shift where its amount is not below the width
//! (RFC-0037 rule 3). Each is one branch to the trap: `+`, `-` and `*` are
//! Rust's `overflowing_*`, their wrapped result stored and the flag tested
//! after the store (`int_trapping_ops`); negation and the shifts test before.
//! A wrapping one, which only a pass writes, is Rust's `wrapping_*`. `/` and
//! `%` trap as Rust's do at every build profile. Every operand and every
//! result here is a word, so a run is two `word` loads and one `set_word`
//! store, and the kind byte the frame wrote stands (RFC-0052 rule 5).

use std::marker::PhantomData;

use acvus_mir::ir::{BinOp, Checked, Overflow, UnaryOp};
use acvus_mir::ty::IntTy;

use crate::code::{Exit, Marked, Off, Op, successor};
use crate::machine::Machine;
use crate::ops::place::{self, BinaryAt, Place, UnaryAt, at_binary, at_unary};

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

/// The registers a one-operand operation names.
#[derive(Clone, Copy)]
pub struct Unary {
    pub dst: Marked,
    pub src: Marked,
}

/// One integer width, as the operations at that width read and write it.
pub trait Int: Copy + PartialOrd + 'static {
    const BITS: u64;
    const SHIFT_MASK: u64;

    fn read(bits: u64) -> Self;
    fn word(self) -> u64;
    fn wide(self) -> i128;
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn checked_mul(self, other: Self) -> Option<Self>;
    fn checked_neg(self) -> Option<Self>;
    fn overflowing_add(self, other: Self) -> (Self, bool);
    fn overflowing_sub(self, other: Self) -> (Self, bool);
    fn overflowing_mul(self, other: Self) -> (Self, bool);
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
            const BITS: u64 = <$t>::BITS as u64;
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
            fn checked_add(self, other: Self) -> Option<Self> {
                <$t>::checked_add(self, other)
            }
            fn checked_sub(self, other: Self) -> Option<Self> {
                <$t>::checked_sub(self, other)
            }
            fn checked_mul(self, other: Self) -> Option<Self> {
                <$t>::checked_mul(self, other)
            }
            fn checked_neg(self) -> Option<Self> {
                <$t>::checked_neg(self)
            }
            fn overflowing_add(self, other: Self) -> (Self, bool) {
                <$t>::overflowing_add(self, other)
            }
            fn overflowing_sub(self, other: Self) -> (Self, bool) {
                <$t>::overflowing_sub(self, other)
            }
            fn overflowing_mul(self, other: Self) -> (Self, bool) {
                <$t>::overflowing_mul(self, other)
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
const ADD_OVERFLOW: &str = "attempt to add with overflow";
const SUB_OVERFLOW: &str = "attempt to subtract with overflow";
const MUL_OVERFLOW: &str = "attempt to multiply with overflow";
const NEG_OVERFLOW: &str = "attempt to negate with overflow";
const SHL_OVERFLOW: &str = "attempt to shift left with overflow";
const SHR_OVERFLOW: &str = "attempt to shift right with overflow";

/// The trap of an operation that overflowed, out of the operation's line:
/// the operation is its checked form and one branch here (RFC-0037 rule 3).
#[cold]
#[inline(never)]
fn overflowed(text: &'static str) -> ! {
    panic!("{text}")
}

/// The value a checked operation gave, or its trap.
#[inline(always)]
fn or_trap<T>(held: Option<T>, text: &'static str) -> T {
    match held {
        Some(value) => value,
        None => overflowed(text),
    }
}

/// A trapping `+`, `-`, `*` and negation on values at their width: the
/// operations `word`'s and a chain's integer nodes both run, so the two
/// forms trap alike and with one text.
pub(crate) mod trapping {
    use super::*;

    #[inline(always)]
    pub(crate) fn add<T>(a: T, b: T) -> T
    where
        T: Int,
    {
        or_trap(a.checked_add(b), ADD_OVERFLOW)
    }

    #[inline(always)]
    pub(crate) fn sub<T>(a: T, b: T) -> T
    where
        T: Int,
    {
        or_trap(a.checked_sub(b), SUB_OVERFLOW)
    }

    #[inline(always)]
    pub(crate) fn mul<T>(a: T, b: T) -> T
    where
        T: Int,
    {
        or_trap(a.checked_mul(b), MUL_OVERFLOW)
    }

    #[inline(always)]
    pub(crate) fn neg<T>(a: T) -> T
    where
        T: Int,
    {
        or_trap(a.checked_neg(), NEG_OVERFLOW)
    }

    /// A trapping `+`, `-` or `*` whose trap is deferred to the caller: the
    /// wrapped value, and the operation's trap text where the exact result
    /// does not fit. A caller that runs the operation on a path the program
    /// may not take ends the run with `trap` only where the path is taken
    /// (RFC-0074 rule 2).
    pub(crate) type Deferred<T> = (T, Option<&'static str>);

    #[inline(always)]
    pub(crate) fn deferred_add<T>(a: T, b: T) -> Deferred<T>
    where
        T: Int,
    {
        let (value, overflowed) = a.overflowing_add(b);
        (value, overflowed.then_some(ADD_OVERFLOW))
    }

    #[inline(always)]
    pub(crate) fn deferred_sub<T>(a: T, b: T) -> Deferred<T>
    where
        T: Int,
    {
        let (value, overflowed) = a.overflowing_sub(b);
        (value, overflowed.then_some(SUB_OVERFLOW))
    }

    #[inline(always)]
    pub(crate) fn deferred_mul<T>(a: T, b: T) -> Deferred<T>
    where
        T: Int,
    {
        let (value, overflowed) = a.overflowing_mul(b);
        (value, overflowed.then_some(MUL_OVERFLOW))
    }

    /// The trap a deferred operation named, at the place the caller runs it.
    #[inline(always)]
    pub(crate) fn trap(text: &'static str) -> ! {
        overflowed(text)
    }
}

/// Rust's `wrapping_shl` and `wrapping_shr` take the amount modulo the
/// width; `SHIFT_MASK` is that modulus, so the masked amount is below `64`
/// and reaches `u32` whole, for an amount of either sign.
fn shift_amount<T>(bits: u64) -> u32
where
    T: Int,
{
    (T::read(bits).word() & T::SHIFT_MASK) as u32
}

/// The amount of a trapping shift, which Rust's `<<` and `>>` check
/// unsigned: an amount read at the width is below it, or the shift traps
/// with `text`. A negative amount's word is at least `2^63`, so it traps.
#[inline(always)]
fn checked_shift_amount<T>(bits: u64, text: &'static str) -> u32
where
    T: Int,
{
    let amount = T::read(bits).word();
    match amount < T::BITS {
        true => amount as u32,
        false => overflowed(text),
    }
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
        wrapping_add(a, b) { a.wrapping_add(b).word() }
        wrapping_sub(a, b) { a.wrapping_sub(b).word() }
        wrapping_mul(a, b) { a.wrapping_mul(b).word() }
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
        min(a, b) { if b < a { b.word() } else { a.word() } }
        max(a, b) { if b > a { b.word() } else { a.word() } }
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
        T::read(left)
            .wrapping_shl(checked_shift_amount::<T>(right, SHL_OVERFLOW))
            .word()
    }

    #[inline]
    pub fn shr<T>(left: u64, right: u64) -> u64
    where
        T: Int,
    {
        T::read(left)
            .wrapping_shr(checked_shift_amount::<T>(right, SHR_OVERFLOW))
            .word()
    }

    #[inline]
    pub fn wrapping_shl<T>(left: u64, right: u64) -> u64
    where
        T: Int,
    {
        T::read(left).wrapping_shl(shift_amount::<T>(right)).word()
    }

    #[inline]
    pub fn wrapping_shr<T>(left: u64, right: u64) -> u64
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
        trapping::neg(T::read(operand)).word()
    }

    #[inline]
    pub fn wrapping_neg<T>(operand: u64) -> u64
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
            pub struct $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                l: L::At,
                r: R::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (T, L, R, D)>,
            }

            impl<T, L, R, D> $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                pub fn new(at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> $op<T, L, R, D> {
                    $op {
                        l: at.l,
                        r: at.r,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<T, L, R, D> Op for $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let bits = word::$f::<T>(
                        L::read(regs, self.l, r0),
                        R::read(regs, self.r, r0),
                    );
                    let carried = D::write(regs, self.dst, $result(bits));
                    self.next.run(m, carried)
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

/// The trapping `+`, `-` and `*` of one operation per operator: the wrapped
/// result is written, and then the run ends with the operation's text where
/// it overflowed, before any operation reads the register (RFC-0037 rule
/// 3). The flag is tested after the store, next to the dispatch, and not
/// between the arithmetic and the store: with the test in between, one
/// never-taken branch cost `accum`'s `for range break` 19 % and `programs`'s
/// `bf table` 8.6 % on this machine; after the store it costs neither.
macro_rules! int_trapping_ops {
    ($( $op:ident = $f:ident ),* $(,)?) => {
        $(
            pub struct $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                l: L::At,
                r: R::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (T, L, R, D)>,
            }

            impl<T, L, R, D> $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                pub fn new(at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> $op<T, L, R, D> {
                    $op {
                        l: at.l,
                        r: at.r,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<T, L, R, D> Op for $op<T, L, R, D>
            where
                T: Int,
                L: Place,
                R: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let (value, trap) = trapping::$f(
                        T::read(L::read(regs, self.l, r0)),
                        T::read(R::read(regs, self.r, r0)),
                    );
                    let carried = D::write(regs, self.dst, value.word());
                    if let Some(text) = trap {
                        trapping::trap(text);
                    }
                    self.next.run(m, carried)
                }
            }
        )*
    };
}

int_trapping_ops!(Add = deferred_add, Sub = deferred_sub, Mul = deferred_mul);

int_ops!(
    WrappingAdd = wrapping_add -> as_word,
    WrappingSub = wrapping_sub -> as_word,
    WrappingMul = wrapping_mul -> as_word,
    Div = div -> as_word,
    Rem = rem -> as_word,
    BitAnd = bit_and -> as_word,
    BitOr = bit_or -> as_word,
    BitXor = bit_xor -> as_word,
    Shl = shl -> as_word,
    Shr = shr -> as_word,
    WrappingShl = wrapping_shl -> as_word,
    WrappingShr = wrapping_shr -> as_word,
    Min = min -> as_word,
    Max = max -> as_word,
    Eq = eq -> as_bool_word,
    Neq = neq -> as_bool_word,
    Lt = lt -> as_bool_word,
    Gt = gt -> as_bool_word,
    Lte = lte -> as_bool_word,
    Gte = gte -> as_bool_word,
);

macro_rules! int_unary_ops {
    ($( $op:ident = $f:ident ),* $(,)?) => {
        $(
            pub struct $op<T, S, D>
            where
                T: Int,
                S: Place,
                D: Place,
            {
                src: S::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (T, S, D)>,
            }

            impl<T, S, D> $op<T, S, D>
            where
                T: Int,
                S: Place,
                D: Place,
            {
                pub fn new(at: UnaryAt<S, D>, next: Box<dyn Op>) -> $op<T, S, D> {
                    $op {
                        src: at.src,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<T, S, D> Op for $op<T, S, D>
            where
                T: Int,
                S: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let bits = word::$f::<T>(S::read(regs, self.src, r0));
                    let carried = D::write(regs, self.dst, bits);
                    self.next.run(m, carried)
                }
            }
        )*
    };
}

int_unary_ops!(Neg = neg, WrappingNeg = wrapping_neg);

macro_rules! float_ops {
    ($( $op:ident = $f:ident -> $result:ident ),* $(,)?) => {
        $(
            pub struct $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                l: L::At,
                r: R::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (L, R, D)>,
            }

            impl<L, R, D> $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                pub fn new(at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> $op<L, R, D> {
                    $op {
                        l: at.l,
                        r: at.r,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<L, R, D> Op for $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let left = f64::from_bits(L::read(regs, self.l, r0));
                    let right = f64::from_bits(R::read(regs, self.r, r0));
                    let bits = $result(float_word::$f(left, right));
                    let carried = D::write(regs, self.dst, bits);
                    self.next.run(m, carried)
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

macro_rules! float_unary_ops {
    ($( $op:ident = $f:ident -> $result:ident ),* $(,)?) => {
        $(
            pub struct $op<S, D>
            where
                S: Place,
                D: Place,
            {
                src: S::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (S, D)>,
            }

            impl<S, D> $op<S, D>
            where
                S: Place,
                D: Place,
            {
                pub fn new(at: UnaryAt<S, D>, next: Box<dyn Op>) -> $op<S, D> {
                    $op {
                        src: at.src,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<S, D> Op for $op<S, D>
            where
                S: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let operand = f64::from_bits(S::read(regs, self.src, r0));
                    let bits = $result(float_word::$f(operand));
                    let carried = D::write(regs, self.dst, bits);
                    self.next.run(m, carried)
                }
            }
        )*
    };
}

float_unary_ops!(NegF64 = neg_f64 -> as_float_word);

macro_rules! bool_ops {
    ($( $op:ident ($a:ident, $b:ident) $body:block )*) => {
        $(
            pub struct $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                l: L::At,
                r: R::At,
                dst: D::At,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> (L, R, D)>,
            }

            impl<L, R, D> $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                pub fn new(at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> $op<L, R, D> {
                    $op {
                        l: at.l,
                        r: at.r,
                        dst: at.dst,
                        next,
                        at: PhantomData,
                    }
                }
            }

            impl<L, R, D> Op for $op<L, R, D>
            where
                L: Place,
                R: Place,
                D: Place,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    let $a = L::read(regs, self.l, r0) != 0;
                    let $b = R::read(regs, self.r, r0) != 0;
                    let carried = D::write(regs, self.dst, as_bool_word($body));
                    self.next.run(m, carried)
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

pub struct NotBool<S, D>
where
    S: Place,
    D: Place,
{
    src: S::At,
    dst: D::At,
    next: Box<dyn Op>,
    at: PhantomData<fn() -> (S, D)>,
}

impl<S, D> NotBool<S, D>
where
    S: Place,
    D: Place,
{
    pub fn new(at: UnaryAt<S, D>, next: Box<dyn Op>) -> NotBool<S, D> {
        NotBool {
            src: at.src,
            dst: at.dst,
            next,
            at: PhantomData,
        }
    }
}

impl<S, D> Op for NotBool<S, D>
where
    S: Place,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let held = S::read(regs, self.src, r0) != 0;
        let carried = D::write(regs, self.dst, as_bool_word(!held));
        self.next.run(m, carried)
    }
}

fn int_op<T, L, R, D>(op: BinOp, at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> Box<dyn Op>
where
    T: Int,
    L: Place,
    R: Place,
    D: Place,
{
    match op {
        BinOp::Add(Overflow::Trap) => Box::new(Add::<T, L, R, D>::new(at, next)),
        BinOp::Sub(Overflow::Trap) => Box::new(Sub::<T, L, R, D>::new(at, next)),
        BinOp::Mul(Overflow::Trap) => Box::new(Mul::<T, L, R, D>::new(at, next)),
        BinOp::Add(Overflow::Wrap) => Box::new(WrappingAdd::<T, L, R, D>::new(at, next)),
        BinOp::Sub(Overflow::Wrap) => Box::new(WrappingSub::<T, L, R, D>::new(at, next)),
        BinOp::Mul(Overflow::Wrap) => Box::new(WrappingMul::<T, L, R, D>::new(at, next)),
        BinOp::Div => Box::new(Div::<T, L, R, D>::new(at, next)),
        BinOp::Mod => Box::new(Rem::<T, L, R, D>::new(at, next)),
        BinOp::Eq => Box::new(Eq::<T, L, R, D>::new(at, next)),
        BinOp::Neq => Box::new(Neq::<T, L, R, D>::new(at, next)),
        BinOp::Lt => Box::new(Lt::<T, L, R, D>::new(at, next)),
        BinOp::Gt => Box::new(Gt::<T, L, R, D>::new(at, next)),
        BinOp::Lte => Box::new(Lte::<T, L, R, D>::new(at, next)),
        BinOp::Gte => Box::new(Gte::<T, L, R, D>::new(at, next)),
        BinOp::BitAnd => Box::new(BitAnd::<T, L, R, D>::new(at, next)),
        BinOp::BitOr => Box::new(BitOr::<T, L, R, D>::new(at, next)),
        BinOp::Xor => Box::new(BitXor::<T, L, R, D>::new(at, next)),
        BinOp::Shl(Overflow::Trap) => Box::new(Shl::<T, L, R, D>::new(at, next)),
        BinOp::Shr(Overflow::Trap) => Box::new(Shr::<T, L, R, D>::new(at, next)),
        BinOp::Shl(Overflow::Wrap) => Box::new(WrappingShl::<T, L, R, D>::new(at, next)),
        BinOp::Shr(Overflow::Wrap) => Box::new(WrappingShr::<T, L, R, D>::new(at, next)),
        BinOp::Min => Box::new(Min::<T, L, R, D>::new(at, next)),
        BinOp::Max => Box::new(Max::<T, L, R, D>::new(at, next)),
        other => panic!("unsupported int binop {other:?}"),
    }
}

/// The operation a binary operator at an integer width prepares to.
pub fn int_binop(op: BinOp, k: IntTy, places: place::Binary, next: Box<dyn Op>) -> Box<dyn Op> {
    at_binary!(places, |at| for_int_ty!(k, |T| int_op::<T, L, R, D>(
        op, at, next
    )))
}

fn float_op<L, R, D>(op: BinOp, at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> Box<dyn Op>
where
    L: Place,
    R: Place,
    D: Place,
{
    // At `f64` both `Overflow`s are the IEEE operation.
    match op {
        BinOp::Add(_) => Box::new(AddF64::<L, R, D>::new(at, next)),
        BinOp::Sub(_) => Box::new(SubF64::<L, R, D>::new(at, next)),
        BinOp::Mul(_) => Box::new(MulF64::<L, R, D>::new(at, next)),
        BinOp::Div => Box::new(DivF64::<L, R, D>::new(at, next)),
        BinOp::Mod => Box::new(RemF64::<L, R, D>::new(at, next)),
        BinOp::Eq => Box::new(EqF64::<L, R, D>::new(at, next)),
        BinOp::Neq => Box::new(NeqF64::<L, R, D>::new(at, next)),
        BinOp::Lt => Box::new(LtF64::<L, R, D>::new(at, next)),
        BinOp::Gt => Box::new(GtF64::<L, R, D>::new(at, next)),
        BinOp::Lte => Box::new(LteF64::<L, R, D>::new(at, next)),
        BinOp::Gte => Box::new(GteF64::<L, R, D>::new(at, next)),
        other => panic!("unsupported float binop {other:?}"),
    }
}

/// The operation a binary operator at `Float` prepares to.
pub fn float_binop(op: BinOp, places: place::Binary, next: Box<dyn Op>) -> Box<dyn Op> {
    at_binary!(places, |at| float_op::<L, R, D>(op, at, next))
}

fn bool_op<L, R, D>(op: BinOp, at: BinaryAt<L, R, D>, next: Box<dyn Op>) -> Box<dyn Op>
where
    L: Place,
    R: Place,
    D: Place,
{
    match op {
        BinOp::Eq => Box::new(EqBool::<L, R, D>::new(at, next)),
        BinOp::Neq => Box::new(NeqBool::<L, R, D>::new(at, next)),
        BinOp::Xor => Box::new(XorBool::<L, R, D>::new(at, next)),
        other => panic!("unsupported bool binop {other:?}"),
    }
}

/// The operation a binary operator at `Bool` prepares to.
pub fn bool_binop(op: BinOp, places: place::Binary, next: Box<dyn Op>) -> Box<dyn Op> {
    at_binary!(places, |at| bool_op::<L, R, D>(op, at, next))
}

/// The operation a unary operator at an integer width prepares to.
pub fn int_unaryop(op: UnaryOp, k: IntTy, places: place::Unary, next: Box<dyn Op>) -> Box<dyn Op> {
    let UnaryOp::Neg(overflow) = op else {
        panic!("unary {op:?} on an integer")
    };
    at_unary!(places, |at| for_int_ty!(k, |T| match overflow {
        Overflow::Trap => Box::new(Neg::<T, S, D>::new(at, next)) as Box<dyn Op>,
        Overflow::Wrap => Box::new(WrappingNeg::<T, S, D>::new(at, next)) as Box<dyn Op>,
    }))
}

/// Reads both operands from the frame: `reads_place` in `prepare` names no
/// check, so no operand of one rides.
pub fn int_check(op: Checked, k: IntTy, l: Off, r: Off, next: Box<dyn Op>) -> Box<dyn Op> {
    for_int_ty!(k, |T| match op {
        Checked::Add => Box::new(CheckAdd::<T> {
            l,
            r,
            next,
            at: PhantomData,
        }) as Box<dyn Op>,
        Checked::Sub => Box::new(CheckSub::<T> {
            l,
            r,
            next,
            at: PhantomData,
        }) as Box<dyn Op>,
        Checked::Mul => Box::new(CheckMul::<T> {
            l,
            r,
            next,
            at: PhantomData,
        }) as Box<dyn Op>,
    })
}

macro_rules! int_checks {
    ($( $op:ident = $f:ident ),* $(,)?) => {
        $(
            pub struct $op<T>
            where
                T: Int,
            {
                l: Off,
                r: Off,
                next: Box<dyn Op>,
                at: PhantomData<fn() -> T>,
            }

            impl<T> Op for $op<T>
            where
                T: Int,
            {
                successor!();

                #[inline]
                fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
                    let regs = m.regs();
                    trapping::$f(T::read(regs.word(self.l)), T::read(regs.word(self.r)));
                    self.next.run(m, r0)
                }
            }
        )*
    };
}

int_checks!(CheckAdd = add, CheckSub = sub, CheckMul = mul);

/// The three frame slots a `CheckSteps` reads: `from` and `step` at the
/// check's width, `count` a `u64`.
#[derive(Clone, Copy)]
pub struct Steps {
    pub from: Off,
    pub step: Off,
    pub count: Off,
}

/// Reads its operands from the frame, as `int_check`'s ops do.
pub fn int_check_steps(k: IntTy, steps: Steps, next: Box<dyn Op>) -> Box<dyn Op> {
    Box::new(CheckSteps { k, steps, next })
}

/// `from + count·step` over the integers: a width's values and a `u64`
/// count are below `2^64` in magnitude, so the product is below `2^128`
/// and is exact in `i128` wherever it does not overflow `i128`. Where it
/// does, its magnitude is at least `2^127`, and the sum, off by less than
/// `2^64`, fits no width: that is a trap too.
pub struct CheckSteps {
    k: IntTy,
    steps: Steps,
    next: Box<dyn Op>,
}

impl Op for CheckSteps {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let from = self.k.read(regs.word(self.steps.from));
        let step = self.k.read(regs.word(self.steps.step));
        let count = i128::from(regs.word(self.steps.count));
        let last = step
            .checked_mul(count)
            .and_then(|advanced| from.checked_add(advanced));
        if !last.is_some_and(|last| self.k.holds(last)) {
            overflowed(ADD_OVERFLOW);
        }
        self.next.run(m, r0)
    }
}

/// The operation a unary operator at `Float` prepares to. Both `Overflow`s
/// are the IEEE negation.
pub fn float_unaryop(op: UnaryOp, places: place::Unary, next: Box<dyn Op>) -> Box<dyn Op> {
    let UnaryOp::Neg(_) = op else {
        panic!("unary {op:?} on a float")
    };
    at_unary!(places, |at| Box::new(NegF64::<S, D>::new(at, next))
        as Box<dyn Op>)
}

/// The operation a unary operator at `Bool` prepares to.
pub fn bool_unaryop(op: UnaryOp, places: place::Unary, next: Box<dyn Op>) -> Box<dyn Op> {
    let UnaryOp::Not = op else {
        panic!("unary {op:?} on a bool")
    };
    at_unary!(places, |at| Box::new(NotBool::<S, D>::new(at, next))
        as Box<dyn Op>)
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

    /// What `Add`, `Sub` and `Mul` run on two words: the deferred operation,
    /// and its trap where it overflowed.
    fn trapping_word<T>(f: fn(T, T) -> trapping::Deferred<T>) -> impl FnOnce(u64, u64) -> u64
    where
        T: Int,
    {
        move |a, b| {
            let (value, trap) = f(T::read(a), T::read(b));
            if let Some(text) = trap {
                trapping::trap(text);
            }
            value.word()
        }
    }

    #[test]
    fn a_subtraction_within_the_width_is_the_difference() {
        assert_eq!(
            i64_op(trapping_word::<i64>(trapping::deferred_sub), 1, 2),
            -1
        );
    }

    #[test]
    fn wrapping_addition_subtraction_and_multiplication_past_the_width_wrap() {
        assert_eq!(i64_op(word::wrapping_add::<i64>, i64::MAX, 1), i64::MIN);
        assert_eq!(i64_op(word::wrapping_sub::<i64>, i64::MIN, 1), i64::MAX);
        assert_eq!(i64_op(word::wrapping_mul::<i64>, i64::MIN, -1), i64::MIN);
    }

    #[test]
    fn wrapping_negation_of_the_minimum_is_the_minimum() {
        assert_eq!(word::wrapping_neg::<i64>(i64::MIN as u64) as i64, i64::MIN);
    }

    #[test]
    fn trapping_arithmetic_within_the_width_is_the_result() {
        assert_eq!(
            i64_op(
                trapping_word::<i64>(trapping::deferred_add),
                i64::MAX - 1,
                1
            ),
            i64::MAX
        );
        assert_eq!(
            i64_op(
                trapping_word::<i64>(trapping::deferred_sub),
                i64::MIN + 1,
                1
            ),
            i64::MIN
        );
        assert_eq!(
            i64_op(
                trapping_word::<i64>(trapping::deferred_mul),
                i64::MIN / 2,
                2
            ),
            i64::MIN
        );
        assert_eq!(word::neg::<i64>((i64::MIN + 1) as u64) as i64, i64::MAX);
        assert_eq!(i64_op(word::shl::<u8>, 1, 7), 128);
        assert_eq!(i64_op(word::shr::<i64>, -8, 63), -1);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn a_trapping_addition_past_the_width_traps() {
        i64_op(trapping_word::<u8>(trapping::deferred_add), 255, 1);
    }

    #[test]
    #[should_panic(expected = "attempt to subtract with overflow")]
    fn a_trapping_subtraction_past_the_width_traps() {
        i64_op(trapping_word::<i64>(trapping::deferred_sub), i64::MIN, 1);
    }

    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn a_trapping_multiplication_past_the_width_traps() {
        i64_op(trapping_word::<i64>(trapping::deferred_mul), i64::MIN, -1);
    }

    #[test]
    #[should_panic(expected = "attempt to negate with overflow")]
    fn a_trapping_negation_of_the_minimum_traps() {
        word::neg::<i8>(i8::MIN as u64);
    }

    #[test]
    #[should_panic(expected = "attempt to shift left with overflow")]
    fn a_trapping_shift_left_by_the_width_traps() {
        i64_op(word::shl::<u8>, 1, 8);
    }

    #[test]
    #[should_panic(expected = "attempt to shift right with overflow")]
    fn a_trapping_shift_right_by_a_negative_amount_traps() {
        i64_op(word::shr::<i64>, 1, -1);
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
    /// the same expression, which is `wrapping_shl` and `wrapping_shr`.
    #[test]
    fn a_wrapping_shift_takes_its_amount_modulo_the_width() {
        assert_eq!(i64_op(word::wrapping_shl::<i64>, 1, -1), i64::MIN);
        assert_eq!(i64_op(word::wrapping_shl::<i64>, 1, 64), 1);
        assert_eq!(i64_op(word::wrapping_shl::<i64>, 1, 65), 2);
        assert_eq!(i64_op(word::wrapping_shl::<i64>, 1, i64::MIN), 1);
        assert_eq!(i64_op(word::wrapping_shr::<i64>, -1, -1), -1);
        assert_eq!(i64_op(word::wrapping_shr::<i64>, 256, 65), 128);
        assert_eq!(i64_op(word::wrapping_shl::<u8>, 1, 255), 128);
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
        assert_eq!(
            i64_op(trapping_word::<u8>(trapping::deferred_add), 200, 55),
            255
        );
        assert_eq!(<i8 as Int>::read(0xFF), -1);
    }

    #[test]
    fn a_wrapping_addition_past_the_narrow_width_wraps() {
        assert_eq!(i64_op(word::wrapping_add::<u8>, 200, 56), 0);
        assert_eq!(i64_op(word::wrapping_add::<u8>, 200, 57), 1);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn a_trapping_addition_past_the_narrow_width_traps() {
        i64_op(trapping_word::<u8>(trapping::deferred_add), 200, 56);
    }
}
