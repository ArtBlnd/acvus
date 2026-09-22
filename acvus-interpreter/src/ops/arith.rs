//! Arithmetic and logic, one operation per operator at one operand type.
//!
//! An integer operation is the same Rust operator in a release build, at
//! the operand's width, panic messages included (RFC-0037). Every operand
//! and every result here is a word, so a run is two `word` loads and one
//! `set_word` store, and the kind byte the frame wrote stands (RFC-0052 §5).

use std::marker::PhantomData;

use acvus_ast::{BinOp, UnaryOp};
use acvus_mir::ty::IntTy;

use crate::code::{Exit, Marked, Op, successor};
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

pub struct Neg<T, S, D>
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

impl<T, S, D> Neg<T, S, D>
where
    T: Int,
    S: Place,
    D: Place,
{
    pub fn new(at: UnaryAt<S, D>, next: Box<dyn Op>) -> Neg<T, S, D> {
        Neg {
            src: at.src,
            dst: at.dst,
            next,
            at: PhantomData,
        }
    }
}

impl<T, S, D> Op for Neg<T, S, D>
where
    T: Int,
    S: Place,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let bits = word::neg::<T>(S::read(regs, self.src, r0));
        let carried = D::write(regs, self.dst, bits);
        self.next.run(m, carried)
    }
}

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
        BinOp::Add => Box::new(Add::<T, L, R, D>::new(at, next)),
        BinOp::Sub => Box::new(Sub::<T, L, R, D>::new(at, next)),
        BinOp::Mul => Box::new(Mul::<T, L, R, D>::new(at, next)),
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
        BinOp::Shl => Box::new(Shl::<T, L, R, D>::new(at, next)),
        BinOp::Shr => Box::new(Shr::<T, L, R, D>::new(at, next)),
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
    match op {
        BinOp::Add => Box::new(AddF64::<L, R, D>::new(at, next)),
        BinOp::Sub => Box::new(SubF64::<L, R, D>::new(at, next)),
        BinOp::Mul => Box::new(MulF64::<L, R, D>::new(at, next)),
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
    let UnaryOp::Neg = op else {
        panic!("unary {op:?} on an integer")
    };
    at_unary!(places, |at| for_int_ty!(
        k,
        |T| Box::new(Neg::<T, S, D>::new(at, next)) as Box<dyn Op>
    ))
}

/// The operation a unary operator at `Float` prepares to.
pub fn float_unaryop(op: UnaryOp, places: place::Unary, next: Box<dyn Op>) -> Box<dyn Op> {
    let UnaryOp::Neg = op else {
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
