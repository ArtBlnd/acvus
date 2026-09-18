//! Variants: the language's `Option` and `Result`, and the tagged union
//! every other variant type is.
//!
//! Which shape a constructor builds and which tag a test resolved to are
//! the destination's and the source's types as the preparation read them.

use acvus_extern::Owned;
use acvus_utils::Astr;

use crate::code::{Off, Op};
use crate::machine::Machine;
use crate::ops::arith::Unary;
use crate::value::{Kind, ResultValue, Value, VariantValue};

/// The place a variant test reads. A `Some` whose payload is a `None` has
/// no storage to point at, so a reference to it is the depth word itself
/// (RFC-0022) and `THROUGH` does not reach a target.
#[inline(always)]
pub(crate) fn scrutinee<const THROUGH: bool>(value: &Value) -> &Value {
    if !THROUGH || value.kind() == Kind::None {
        return value;
    }
    // SAFETY: the type checker admits only a live reference here.
    unsafe { value.target() }
}

/// `LARGE` is what the preparation read from the payload's type: an option
/// is its payload's own value (RFC-0022), so the option owns a `Large`
/// exactly when its payload does.
pub struct MakeSome<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for MakeSome<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let payload = regs.take::<LARGE>(self.slots.src);
        regs.define::<LARGE>(self.slots.dst, Value::some(payload));
    }
}

pub struct MakeNone {
    pub dst: Off,
}

impl Op for MakeNone {
    fn run(&self, m: &mut Machine<'_>) {
        m.regs().define::<false>(self.dst, Value::NONE);
    }
}

/// `Value::result` boxes either side, so the destination owns a `Large`
/// whatever `LARGE` — the payload's own ownership — says.
pub struct MakeOk<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for MakeOk<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let payload = Owned::from_value(regs.take::<LARGE>(self.slots.src));
        regs.define::<true>(self.slots.dst, Value::result(Ok(payload)));
    }
}

pub struct MakeErr<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for MakeErr<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let payload = Owned::from_value(regs.take::<LARGE>(self.slots.src));
        regs.define::<true>(self.slots.dst, Value::result(Err(payload)));
    }
}

pub struct MakeVariant<const LARGE: bool> {
    pub slots: Unary,
    pub tag: Astr,
}

impl<const LARGE: bool> Op for MakeVariant<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let payload = Owned::from_value(regs.take::<LARGE>(self.slots.src));
        let value = Value::variant(self.tag, Some(payload));
        regs.define::<true>(self.slots.dst, value);
    }
}

pub struct MakeUnitVariant {
    pub dst: Off,
    pub tag: Astr,
}

impl Op for MakeUnitVariant {
    fn run(&self, m: &mut Machine<'_>) {
        let value = Value::variant(self.tag, None);
        m.regs().define::<true>(self.dst, value);
    }
}

/// `SOME` is what the tag the arm tests for resolved to at preparation.
pub struct TestOption<const THROUGH: bool, const SOME: bool> {
    pub slots: Unary,
}

impl<const THROUGH: bool, const SOME: bool> Op for TestOption<THROUGH, SOME> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let is_some = !scrutinee::<THROUGH>(regs.peek(self.slots.src)).is_none();
        regs.set_word(self.slots.dst, (is_some == SOME) as u64);
    }
}

/// `OK` is what the tag the arm tests for resolved to at preparation.
pub struct TestResult<const THROUGH: bool, const OK: bool> {
    pub slots: Unary,
}

impl<const THROUGH: bool, const OK: bool> Op for TestResult<THROUGH, OK> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let source = scrutinee::<THROUGH>(regs.peek(self.slots.src));
        // SAFETY: the preparation read `Result` from the source's type.
        let is_ok = unsafe { source.as_result() }.is_ok();
        regs.set_word(self.slots.dst, (is_ok == OK) as u64);
    }
}

pub struct TestVariant<const THROUGH: bool> {
    pub slots: Unary,
    pub tag: Astr,
}

impl<const THROUGH: bool> Op for TestVariant<THROUGH> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let source = scrutinee::<THROUGH>(regs.peek(self.slots.src));
        // SAFETY: the preparation read an enum from the source's type.
        let matches = unsafe { source.as_variant() }.tag == self.tag;
        regs.set_word(self.slots.dst, matches as u64);
    }
}

/// `LARGE` is the payload's ownership, which is the option's own: an
/// unwrap moves it from one slot to another and the mark travels with it.
pub struct UnwrapOption<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for UnwrapOption<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let option = regs.take::<LARGE>(self.slots.src);
        regs.define::<LARGE>(self.slots.dst, Value::some_payload(option));
    }
}

/// The `Result` box is the frame's, whichever side it carries; `LARGE` is
/// the payload's ownership, which leaves the box for the destination.
pub struct UnwrapResult<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for UnwrapResult<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let result = regs.take::<true>(self.slots.src);
        // SAFETY: the preparation read `Result` from the source's type.
        let (Ok(payload) | Err(payload)) = unsafe { result.materialize::<ResultValue>() };
        regs.define::<LARGE>(self.slots.dst, payload.into_value());
    }
}

/// Whether the arm's variant carries a payload is not a type parameter
/// here, and that is a limit of the instruction rather than a choice: the
/// MIR's `UnwrapVariant` names a destination and a source and no tag, so
/// the preparation has no tag to resolve the arm against. Give the
/// instruction its tag and the `map_or_else` below becomes a second const.
pub struct UnwrapVariant<const LARGE: bool> {
    pub slots: Unary,
}

impl<const LARGE: bool> Op for UnwrapVariant<LARGE> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let variant = regs.take::<true>(self.slots.src);
        // SAFETY: the preparation read an enum from the source's type.
        let payload = unsafe { variant.materialize::<VariantValue>() }.payload;
        let value = payload.map_or_else(Value::unit, |held| (*held).into_value());
        regs.define::<LARGE>(self.slots.dst, value);
    }
}
