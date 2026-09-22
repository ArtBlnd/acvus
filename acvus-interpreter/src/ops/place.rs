//! Where one word an operation reads or writes lives (RFC-0052 rule 5).
//!
//! A word-typed SSA value with exactly one use, in the immediately following
//! operation of the same chain, never reaches the frame: the producer leaves
//! it in the argument register the tail call already passes, and the consumer
//! reads it from there. wasm3's `r0`, on `dyn Op`.
//!
//! The place is a type parameter, so no `run` tests where its operand is, and
//! the displacement exists only where there is one — `R0::At` is `()`, so an
//! operation whose operand rides holds no `Off` for it.

use crate::code::{Off, Where};
use crate::regs::{Cell, set_word_at, word_at};

/// One word's place, as the operations that name it are specialized on.
///
/// Obligation across artifacts: `acvus-interpreter-test/benches/asm_probe.rs`
/// reads the release binary, where `R0::read` is the identity on the argument
/// register and `R0::write` is the `mov` the tail call already needed —
/// neither lands an instruction of its own.
pub trait Place: Send + Sync + 'static {
    type At: Copy + Send + Sync + 'static;

    /// The word this place holds, given the word the operation was handed.
    ///
    /// # Safety
    /// As `regs::word_at`: `regs` is the base of the frame the operation runs
    /// in and `at` names a register that frame has.
    unsafe fn read(regs: *mut Cell, at: Self::At, r0: u64) -> u64;

    /// `bits` put where this place is, and handed on as the word the
    /// successor receives.
    ///
    /// Both places hand `bits` on, and that is sound because a word rides to
    /// the *immediately* following operation alone: no word rides past an
    /// operation that does not consume it, so there is nothing here to carry
    /// through.
    ///
    /// # Safety
    /// As `read`.
    unsafe fn write(regs: *mut Cell, at: Self::At, bits: u64) -> u64;

    /// The place read back, for the probes that walk a prepared body.
    #[cfg(any(debug_assertions, feature = "probe"))]
    fn whence(at: Self::At) -> Where;
}

/// A frame register, at the displacement the operation holds.
pub struct Slot;

impl Place for Slot {
    type At = Off;

    #[inline(always)]
    unsafe fn read(regs: *mut Cell, at: Off, _r0: u64) -> u64 {
        // SAFETY: the caller's contract.
        unsafe { word_at(regs, at) }
    }

    #[inline(always)]
    unsafe fn write(regs: *mut Cell, at: Off, bits: u64) -> u64 {
        // SAFETY: the caller's contract.
        unsafe { set_word_at(regs, at, bits) };
        bits
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn whence(at: Off) -> Where {
        Where::Frame(at)
    }
}

/// The argument register the tail call passes.
pub struct R0;

impl Place for R0 {
    type At = ();

    #[inline(always)]
    unsafe fn read(_regs: *mut Cell, _at: (), r0: u64) -> u64 {
        r0
    }

    #[inline(always)]
    unsafe fn write(_regs: *mut Cell, _at: (), bits: u64) -> u64 {
        bits
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn whence(_at: ()) -> Where {
        Where::Register
    }
}

/// The two operands and the result of a binary operation, as `prepare`
/// decided them.
#[derive(Clone, Copy)]
pub struct Binary {
    pub dst: Where,
    pub l: Where,
    pub r: Where,
}

/// The operand and the result of a unary operation, as `prepare` decided
/// them.
#[derive(Clone, Copy)]
pub struct Unary {
    pub dst: Where,
    pub src: Where,
}

/// The same three places as the operation holds them: each is the
/// displacement its place needs, and `R0` needs none. The three are one named
/// value so that an operation is never built from three bare `Off`s whose
/// order is all that separates the left operand from the right.
pub struct BinaryAt<L, R, D>
where
    L: Place,
    R: Place,
    D: Place,
{
    pub l: L::At,
    pub r: R::At,
    pub dst: D::At,
}

/// `BinaryAt` for an operation of one operand.
pub struct UnaryAt<S, D>
where
    S: Place,
    D: Place,
{
    pub src: S::At,
    pub dst: D::At,
}

/// Choose the three place types a `Binary` names and build the operation at
/// them: `$make` is written once and compiled at each reachable combination.
macro_rules! at_binary {
    ($places:expr, |$at:ident| $make:expr) => {{
        use $crate::code::Where;
        use $crate::ops::place::{Binary, BinaryAt, R0, Slot};
        match $places {
            Binary {
                l: Where::Frame(l),
                r: Where::Frame(r),
                dst: Where::Frame(dst),
            } => {
                type L = Slot;
                type R = Slot;
                type D = Slot;
                let $at = BinaryAt::<L, R, D> { l, r, dst };
                $make
            }
            Binary {
                l: Where::Frame(l),
                r: Where::Frame(r),
                dst: Where::Register,
            } => {
                type L = Slot;
                type R = Slot;
                type D = R0;
                let $at = BinaryAt::<L, R, D> { l, r, dst: () };
                $make
            }
            Binary {
                l: Where::Register,
                r: Where::Frame(r),
                dst: Where::Frame(dst),
            } => {
                type L = R0;
                type R = Slot;
                type D = Slot;
                let $at = BinaryAt::<L, R, D> { l: (), r, dst };
                $make
            }
            Binary {
                l: Where::Register,
                r: Where::Frame(r),
                dst: Where::Register,
            } => {
                type L = R0;
                type R = Slot;
                type D = R0;
                let $at = BinaryAt::<L, R, D> { l: (), r, dst: () };
                $make
            }
            Binary {
                l: Where::Frame(l),
                r: Where::Register,
                dst: Where::Frame(dst),
            } => {
                type L = Slot;
                type R = R0;
                type D = Slot;
                let $at = BinaryAt::<L, R, D> { l, r: (), dst };
                $make
            }
            Binary {
                l: Where::Frame(l),
                r: Where::Register,
                dst: Where::Register,
            } => {
                type L = Slot;
                type R = R0;
                type D = R0;
                let $at = BinaryAt::<L, R, D> { l, r: (), dst: () };
                $make
            }
            Binary {
                l: Where::Register,
                r: Where::Register,
                ..
            } => panic!(
                "both operands of one operation ride, which the ride analysis cannot \
                 produce: a riding value has exactly one use"
            ),
        }
    }};
}

/// `at_binary` for an operation of one operand.
macro_rules! at_unary {
    ($places:expr, |$at:ident| $make:expr) => {{
        use $crate::code::Where;
        use $crate::ops::place::{R0, Slot, Unary, UnaryAt};
        match $places {
            Unary {
                src: Where::Frame(src),
                dst: Where::Frame(dst),
            } => {
                type S = Slot;
                type D = Slot;
                let $at = UnaryAt::<S, D> { src, dst };
                $make
            }
            Unary {
                src: Where::Frame(src),
                dst: Where::Register,
            } => {
                type S = Slot;
                type D = R0;
                let $at = UnaryAt::<S, D> { src, dst: () };
                $make
            }
            Unary {
                src: Where::Register,
                dst: Where::Frame(dst),
            } => {
                type S = R0;
                type D = Slot;
                let $at = UnaryAt::<S, D> { src: (), dst };
                $make
            }
            Unary {
                src: Where::Register,
                dst: Where::Register,
            } => {
                type S = R0;
                type D = R0;
                let $at = UnaryAt::<S, D> { src: (), dst: () };
                $make
            }
        }
    }};
}

pub(crate) use {at_binary, at_unary};
