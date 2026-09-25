//! Variants: the language's `Option`, and the tagged union every other
//! variant type is.
//!
//! A `Result` reaches the variant operations below with the tag word `Ok` or
//! `Err`. It had its own family once — `MakeOk`, `MakeErr`, `TestResult`,
//! `UnwrapResult` over Rust's `Result<Owned, Owned>` — and RFC-0050 rule 8
//! removed it, because that family was the second layout one type had.
//! `prepare::runs`'s `a_heap_result_and_a_run_of_one_result_are_the_same_words`
//! is what holds the two to one word.
//!
//! Which shape a constructor builds and which tag a test resolved to are
//! the destination's and the source's types as the preparation read them.

use acvus_extern::Owned;

use crate::code::{Exit, Marked, Op, successor};
use crate::machine::Machine;
use crate::ops::arith::Unary;
use crate::value::{Kind, Value, VariantValue};

/// The place a variant test reads. A `Some` whose payload is a `None` has
/// no storage to point at, so a reference to it is the depth word itself
/// (RFC-0039 rule 6) and `THROUGH` does not reach a target.
#[inline(always)]
pub(crate) fn scrutinee<const THROUGH: bool>(value: &Value) -> &Value {
    if !THROUGH || value.kind() == Kind::None {
        return value;
    }
    // SAFETY: the type checker admits only a live reference here.
    unsafe { value.target() }
}

/// `LARGE` is what the preparation read from the payload's type: an option
/// is its payload's own value (RFC-0039 rule 6), so the option owns a `Large`
/// exactly when its payload does.
pub struct MakeSome<const LARGE: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for MakeSome<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let payload = regs.take::<LARGE>(self.slots.src);
        regs.define::<LARGE>(self.slots.dst, Value::some(payload));
        self.next.run(m, r0)
    }
}

/// `LARGE` is the option's type, as for `MakeSome`: a `None` of an option
/// whose payload owns a `Large` is that type's owned value, so the frame
/// claims its register as it claims a `Some`'s, and whatever takes the
/// register by the type finds the claim.
pub struct MakeNone<const LARGE: bool> {
    pub dst: Marked,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for MakeNone<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().define::<LARGE>(self.dst, Value::NONE);
        self.next.run(m, r0)
    }
}

/// `tag` is the register word the preparation resolved, not the name: what the
/// operation writes at run time it carries (RFC-0050 rule 8).
pub struct MakeVariant<const LARGE: bool> {
    pub slots: Unary,
    pub tag: Value,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for MakeVariant<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let value = Value::variant_with(self.tag, || {
            // SAFETY: `take` moved the word out of its register.
            unsafe { Owned::from_value(acvus_extern::Holding::new(), regs.take::<LARGE>(self.slots.src)) }
        });
        regs.define::<true>(self.slots.dst, value);
        self.next.run(m, r0)
    }
}

pub struct MakeUnitVariant {
    pub dst: Marked,
    pub tag: Value,
    pub next: Box<dyn Op>,
}

impl Op for MakeUnitVariant {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `UNDEF` owns nothing.
        let value = Value::variant_with(self.tag, || unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::UNDEF) });
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// `SOME` is what the tag the arm tests for resolved to at preparation.
pub struct TestOption<const THROUGH: bool, const SOME: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool, const SOME: bool> Op for TestOption<THROUGH, SOME> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let is_some = !scrutinee::<THROUGH>(regs.peek(self.slots.src.at())).is_none();
        regs.set_word(self.slots.dst.at(), (is_some == SOME) as u64);
        self.next.run(m, r0)
    }
}

pub struct TestVariant<const THROUGH: bool> {
    pub slots: Unary,
    pub tag: u64,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for TestVariant<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let source = scrutinee::<THROUGH>(regs.peek(self.slots.src.at()));
        // SAFETY: the preparation read an enum from the source's type.
        let matches = unsafe { source.as_variant() }.tag().bits() == self.tag;
        regs.set_word(self.slots.dst.at(), matches as u64);
        self.next.run(m, r0)
    }
}

/// `LARGE` is the payload's ownership, which is the option's own: an
/// unwrap moves it from one slot to another and the mark travels with it.
pub struct UnwrapOption<const LARGE: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for UnwrapOption<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let option = regs.take::<LARGE>(self.slots.src);
        regs.define::<LARGE>(self.slots.dst, Value::some_payload(option));
        self.next.run(m, r0)
    }
}

/// Whether the arm's variant carries a payload is not a type parameter
/// here, and that is a limit of the instruction rather than a choice: the
/// MIR's `UnwrapVariant` names a destination and a source and no tag, so
/// the preparation has no tag to resolve the arm against. Give the
/// instruction its tag and the kind test below becomes a second const.
pub struct UnwrapVariant<const LARGE: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for UnwrapVariant<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let variant = regs.take::<true>(self.slots.src);
        // SAFETY: the preparation read an enum from the source's type.
        let payload = unsafe { variant.materialize::<VariantValue>() }
            .into_payload()
            // SAFETY: the runtime moves the payload word into a register.
            .into_value(unsafe { acvus_extern::Holding::new() });
        let value = match payload.kind() {
            Kind::Undef => Value::unit(),
            _ => payload,
        };
        regs.define::<LARGE>(self.slots.dst, value);
        self.next.run(m, r0)
    }
}
