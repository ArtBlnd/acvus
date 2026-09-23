//! The debug evaluation of a declaration's postconditions at its return
//! (RFC-0082 rule 5). `#[extern_fn(ensures(..))]` emits a call of
//! `assert_postcondition` for each relation it states; a release build of
//! the extension compiles the call and never makes it.

use acvus_mir::laws::Relation;

use crate::len::Arr;
use crate::ty_arg::{Var, kind};

#[diagnostic::on_unimplemented(
    message = "`len` in a postcondition reads a slice or a container, and `{Self}` is neither",
    label = "not a slice or a container",
    note = "RFC-0082 rule 4: `len(x)` is the element count of a slice or a container parameter, or of `ret`"
)]
pub trait Length {
    fn length(&self) -> usize;
}

impl<T> Length for [T] {
    fn length(&self) -> usize {
        self.len()
    }
}

impl<T> Length for Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }
}

impl<T, N> Length for Arr<T, N>
where
    N: Var<kind::Length>,
{
    fn length(&self) -> usize {
        self.0.len()
    }
}

impl<T> Length for &T
where
    T: Length + ?Sized,
{
    fn length(&self) -> usize {
        (**self).length()
    }
}

impl<T> Length for &mut T
where
    T: Length + ?Sized,
{
    fn length(&self) -> usize {
        (**self).length()
    }
}

#[diagnostic::on_unimplemented(
    message = "a postcondition reads `{Self}` as a number, and it is not an integer",
    label = "not an integer",
    note = "RFC-0082 rule 4: a parameter or `ret` standing as a term is an integer"
)]
pub trait Integer {
    fn term(&self) -> Option<i128>;
}

macro_rules! integer_from {
    ($($t:ty),*) => {$(
        impl Integer for $t {
            fn term(&self) -> Option<i128> {
                Some(i128::from(*self))
            }
        }
    )*};
}

integer_from!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Integer for usize {
    fn term(&self) -> Option<i128> {
        i128::try_from(*self).ok()
    }
}

impl Integer for isize {
    fn term(&self) -> Option<i128> {
        i128::try_from(*self).ok()
    }
}

pub fn len_of<T>(subject: &T) -> Option<i128>
where
    T: Length + ?Sized,
{
    i128::try_from(subject.length()).ok()
}

pub fn add(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    a?.checked_add(b?)
}

pub fn sub(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    a?.checked_sub(b?)
}

pub fn mul(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    a?.checked_mul(b?)
}

pub fn max(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    Some(a?.max(b?))
}

/// One relation as its two sides evaluated at the return; a side outside
/// `i128` is `None`.
pub struct Evaluated {
    pub left: Option<i128>,
    pub relation: Relation,
    pub right: Option<i128>,
}

#[track_caller]
pub fn assert_postcondition(function: &str, stated: &str, evaluated: Evaluated) {
    let Evaluated {
        left,
        relation,
        right,
    } = evaluated;
    let (Some(left), Some(right)) = (left, right) else {
        panic!(
            "`{function}` returned, and its postcondition `{stated}` has a term outside i128, \
             so it cannot be checked"
        );
    };
    let holds = match relation {
        Relation::Eq => left == right,
        Relation::Le => left <= right,
        Relation::Lt => left < right,
    };
    if !holds {
        panic!(
            "`{function}` broke its postcondition `{stated}`: the left side is {left} and the \
             right side is {right}"
        );
    }
}
