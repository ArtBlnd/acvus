//! Strings: a template's output, equality, and the clone the language's
//! `StringClone` instruction is (RFC-0020).

use acvus_extern::Release;

use crate::code::{ConcatPart, Off, Op};
use crate::machine::Machine;
use crate::ops::arith::{Binary, Unary};
use crate::value::Value;

/// `THROUGH` is what the preparation read from the source's type: a
/// `&String` is read through its reference, a `String` in place.
#[inline(always)]
fn place<const THROUGH: bool>(value: &Value) -> &Value {
    match THROUGH {
        // SAFETY: the type checker admits only a live reference here.
        true => unsafe { value.target() },
        false => value,
    }
}

pub struct CloneString<const THROUGH: bool> {
    pub slots: Unary,
}

impl<const THROUGH: bool> Op for CloneString<THROUGH> {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let source = place::<THROUGH>(regs.peek(self.slots.src));
        // SAFETY: the type checker admits only a `String` here.
        let text = unsafe { source.as_str() }.to_string();
        regs.define::<true>(self.slots.dst, Value::string(text));
    }
}

pub struct StringEq {
    pub slots: Binary,
}

impl Op for StringEq {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        // SAFETY: the type checker admits only live `&String`s here.
        let equal = unsafe {
            regs.peek(self.slots.l).target().as_str() == regs.peek(self.slots.r).target().as_str()
        };
        regs.set_word(self.slots.dst, equal as u64);
    }
}

pub struct Concat {
    pub dst: Off,
    pub parts: Box<[ConcatPart]>,
    /// Bit `i` is "the slot of part `i` owns a `Large`", as
    /// `composite::Elements`.
    pub owns_large: u64,
}

impl Op for Concat {
    fn run(&self, m: &mut Machine<'_>) {
        let regs = m.regs();
        let mut out = String::new();
        for part in &self.parts {
            let held = regs.read(part.slot);
            if part.through_reference {
                // SAFETY: the type checker admits only a live `&String` here.
                out.push_str(unsafe { held.target().as_str() });
                continue;
            }
            // SAFETY: the type checker admits only a `String` here.
            out.push_str(unsafe { held.as_str() });
            held.release();
        }
        regs.take_mask(self.owns_large);
        regs.define::<true>(self.dst, Value::string(out));
    }
}
