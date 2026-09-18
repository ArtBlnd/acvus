//! The decision tree's tests: a scrutinee against a literal, and an
//! object against a key.

use std::marker::PhantomData;

use acvus_utils::Astr;

use crate::code::{Exit, Off, Op, successor};
use crate::machine::Machine;
use crate::ops::arith::{Int, Unary};
use crate::value::Value;

/// `THROUGH` is what the preparation read from the source's type: a
/// reference is read through, a value in place.
#[inline(always)]
fn place<const THROUGH: bool>(value: &Value) -> &Value {
    match THROUGH {
        // SAFETY: the type checker admits only a live reference here.
        true => unsafe { value.target() },
        false => value,
    }
}

/// The literal is held at the width it was written, which is wider than
/// the operand's for a literal the pattern can never match.
pub struct TestInt<T>
where
    T: Int,
{
    slots: Unary,
    want: i128,
    width: PhantomData<fn() -> T>,
    pub next: Box<dyn Op>,
}

impl<T> TestInt<T>
where
    T: Int,
{
    pub fn new(slots: Unary, want: i128, next: Box<dyn Op>) -> TestInt<T> {
        TestInt {
            slots,
            want,
            width: PhantomData,
            next,
        }
    }
}

impl<T> Op for TestInt<T>
where
    T: Int,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let matches = T::read(regs.word(self.slots.src)).wide() == self.want;
        regs.set_word(self.slots.dst, matches as u64);
        self.next.run(m, r0)
    }
}

pub struct TestFloat<const THROUGH: bool> {
    pub slots: Unary,
    pub want: f64,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for TestFloat<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let matches = place::<THROUGH>(regs.peek(self.slots.src)).as_float() == self.want;
        regs.set_word(self.slots.dst, matches as u64);
        self.next.run(m, r0)
    }
}

pub struct TestBool<const THROUGH: bool> {
    pub slots: Unary,
    pub want: bool,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for TestBool<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let matches = place::<THROUGH>(regs.peek(self.slots.src)).as_bool() == self.want;
        regs.set_word(self.slots.dst, matches as u64);
        self.next.run(m, r0)
    }
}

pub struct TestString<const THROUGH: bool> {
    pub slots: Unary,
    pub want: String,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for TestString<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let source = place::<THROUGH>(regs.peek(self.slots.src));
        // SAFETY: the type checker matches a string literal against a string.
        let matches = unsafe { source.as_str() } == self.want.as_str();
        regs.set_word(self.slots.dst, matches as u64);
        self.next.run(m, r0)
    }
}

/// A unit literal matches the one value of its type.
pub struct TestUnit {
    pub dst: Off,
    pub next: Box<dyn Op>,
}

impl Op for TestUnit {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().set_word(self.dst, true as u64);
        self.next.run(m, r0)
    }
}

pub struct TestObjectKey<const THROUGH: bool> {
    pub slots: Unary,
    pub key: Astr,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for TestObjectKey<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let source = place::<THROUGH>(regs.peek(self.slots.src));
        // SAFETY: is_object checked the vtable id.
        let has = source.is_object() && unsafe { source.as_object() }.contains_key(&self.key);
        regs.set_word(self.slots.dst, has as u64);
        self.next.run(m, r0)
    }
}
