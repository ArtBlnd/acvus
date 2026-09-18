//! `expr as T` outside a chain, one operation per source-target pair
//! (RFC-0049).
//!
//! The value is Rust's `as` at the pair: an integer truncates or extends by
//! width, an integer to `f64` rounds to nearest, and an `f64` to an integer
//! saturates at the width's ends and maps `NaN` to zero.
//!
//! No conversion here can fail, so none of these operations has a trap
//! path; a cast the surface language writes runs wherever a pass puts it.
//! Inside a chain the same conversion is a leaf's read and no operation at
//! all — `chain::LeafRead`.

use std::marker::PhantomData;

use acvus_mir::ty::{IntTy, NumTy};

use crate::code::{Exit, Op, successor};
use crate::machine::Machine;
use crate::ops::arith::for_int_ty;
use crate::ops::chain::Num;
use crate::ops::place::{self, Place, UnaryAt, at_unary};

/// A numeric word read at its own type, at any other numeric type
/// (RFC-0049).
///
/// `Num`'s nine `of_*` constructors in `ops::chain` are the same nine Rust
/// `as` expressions seen from the target; this is the source's side of
/// them, and a pair added to one is a pair owed to the other.
pub trait AsNum: Num {
    fn as_num<T>(self) -> T
    where
        T: Num;
}

macro_rules! impl_as_num {
    ($($t:ty => $of:ident),* $(,)?) => {$(
        impl AsNum for $t {
            #[inline(always)]
            fn as_num<T>(self) -> T
            where
                T: Num,
            {
                T::$of(self)
            }
        }
    )*};
}

impl_as_num! {
    i8 => of_i8, i16 => of_i16, i32 => of_i32, i64 => of_i64,
    u8 => of_u8, u16 => of_u16, u32 => of_u32, u64 => of_u64,
    f64 => of_f64,
}

/// The two ends of one cast, as `prepare` read them off the instruction.
#[derive(Clone, Copy)]
pub struct Conversion {
    pub from: NumTy,
    pub into: NumTy,
}

pub struct Cast<F, T, S, D>
where
    F: AsNum,
    T: Num,
    S: Place,
    D: Place,
{
    src: S::At,
    dst: D::At,
    next: Box<dyn Op>,
    at: PhantomData<fn() -> (F, T, S, D)>,
}

impl<F, T, S, D> Cast<F, T, S, D>
where
    F: AsNum,
    T: Num,
    S: Place,
    D: Place,
{
    pub fn new(at: UnaryAt<S, D>, next: Box<dyn Op>) -> Cast<F, T, S, D> {
        Cast {
            src: at.src,
            dst: at.dst,
            next,
            at: PhantomData,
        }
    }
}

impl<F, T, S, D> Op for Cast<F, T, S, D>
where
    F: AsNum,
    T: Num,
    S: Place,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let held: T = F::read(S::read(regs, self.src, r0)).as_num();
        let carried = D::write(regs, self.dst, held.word());
        self.next.run(m, carried)
    }
}

fn into_ty<F, S, D>(into: NumTy, at: UnaryAt<S, D>, next: Box<dyn Op>) -> Box<dyn Op>
where
    F: AsNum,
    S: Place,
    D: Place,
{
    match into {
        NumTy::Int(k) => {
            for_int_ty!(k, |T| Box::new(Cast::<F, T, S, D>::new(at, next))
                as Box<dyn Op>)
        }
        NumTy::F64 => Box::new(Cast::<F, f64, S, D>::new(at, next)),
    }
}

pub fn cast_op(conversion: Conversion, places: place::Unary, next: Box<dyn Op>) -> Box<dyn Op> {
    let Conversion { from, into } = conversion;
    at_unary!(places, |at| match from {
        NumTy::Int(k) => for_int_ty!(k, |F| into_ty::<F, S, D>(into, at, next)),
        NumTy::F64 => into_ty::<f64, S, D>(into, at, next),
    })
}

/// Each expected value is the value of the Rust `as` expression written
/// beside it, which is what RFC-0049 rules the machine produces.
#[cfg(test)]
mod every_edge_is_rusts {
    use super::*;

    #[test]
    fn an_integer_narrows_and_widens_by_width() {
        assert_eq!(<i64 as AsNum>::as_num::<u64>(-1), -1i64 as u64);
        assert_eq!(<i64 as AsNum>::as_num::<u8>(300), 300i64 as u8);
        assert_eq!(<u8 as AsNum>::as_num::<i8>(200), 200u8 as i8);
        assert_eq!(<i8 as AsNum>::as_num::<i64>(-1), -1i8 as i64);
        assert_eq!(<u64 as AsNum>::as_num::<i32>(u64::MAX), u64::MAX as i32);
    }

    #[test]
    fn an_integer_to_a_float_rounds_to_nearest() {
        assert_eq!(<u64 as AsNum>::as_num::<f64>(u64::MAX), u64::MAX as f64);
        assert_eq!(<i64 as AsNum>::as_num::<f64>(i64::MIN), i64::MIN as f64);
        assert_eq!(<i64 as AsNum>::as_num::<f64>(1), 1i64 as f64);
    }

    #[test]
    fn a_float_to_an_integer_saturates_and_maps_nan_to_zero() {
        assert_eq!(<f64 as AsNum>::as_num::<i64>(f64::NAN), f64::NAN as i64);
        assert_eq!(<f64 as AsNum>::as_num::<i64>(1e30), 1e30f64 as i64);
        assert_eq!(<f64 as AsNum>::as_num::<i64>(-1e30), -1e30f64 as i64);
        assert_eq!(<f64 as AsNum>::as_num::<u8>(-1.0), -1.0f64 as u8);
        assert_eq!(<f64 as AsNum>::as_num::<u8>(300.5), 300.5f64 as u8);
        assert_eq!(<f64 as AsNum>::as_num::<i64>(-1.9), -1.9f64 as i64);
    }
}
