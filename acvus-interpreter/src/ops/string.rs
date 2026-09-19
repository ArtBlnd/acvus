//! Strings: a template's output, equality, and the clone the language's
//! `StringClone` instruction is (RFC-0020).

use acvus_extern::Release;

use crate::code::{ConcatPart, Exit, Marked, Op, successor};
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
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for CloneString<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let source = place::<THROUGH>(regs.peek(self.slots.src.at));
        // SAFETY: the type checker admits only a `String` here.
        let text = unsafe { source.as_str() }.to_string();
        regs.define::<true>(self.slots.dst, Value::string(text));
        self.next.run(m, r0)
    }
}

pub struct StringEq {
    pub slots: Binary,
    pub next: Box<dyn Op>,
}

impl Op for StringEq {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        // SAFETY: the type checker admits only live `&String`s here.
        let equal = unsafe {
            regs.peek(self.slots.l.at).target().as_str()
                == regs.peek(self.slots.r.at).target().as_str()
        };
        regs.set_word(self.slots.dst.at, equal as u64);
        self.next.run(m, r0)
    }
}

pub struct Concat {
    pub dst: Marked,
    pub parts: Box<[ConcatPart]>,
    /// Bit `i` is "the slot of part `i` owns a `Large`", as
    /// `composite::Elements`.
    pub owns_large: u64,
    pub next: Box<dyn Op>,
}

impl Op for Concat {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
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
        self.next.run(m, r0)
    }
}
