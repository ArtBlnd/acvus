//! Strings: a template's output, equality, and the clone the language's
//! `StringClone` instruction is (RFC-0020).

use acvus_extern::{Release, StrView, Words};

use crate::code::{ConcatPart, Exit, LentText, Marked, Off, Op, successor};
use crate::machine::Machine;
use crate::ops::arith::Unary;
use crate::regs::Regs;
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
    pub dst: Off,
    pub l: LentText,
    pub r: LentText,
    pub next: Box<dyn Op>,
}

impl Op for StringEq {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        // SAFETY: the checker keeps both operands live over this operation,
        // and each one's shape is what `prepare` read from its type.
        let equal = unsafe { lent(regs, self.l) == lent(regs, self.r) };
        regs.set_word(self.dst, equal as u64);
        self.next.run(m, r0)
    }
}

/// A string literal pattern against text the scrutinee lends: the bytes,
/// compared, with no value made (RFC-0062 Decision 2). `pattern::TestString`
/// is the same test where the scrutinee is a `String` in a register; this is
/// the one where it is the pair of a `&str`.
pub struct TestLentText {
    pub dst: Off,
    pub src: LentText,
    pub want: String,
    pub next: Box<dyn Op>,
}

impl Op for TestLentText {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        // SAFETY: the checker keeps the scrutinee live over this operation.
        let matches = unsafe { lent(regs, self.src) } == self.want.as_str();
        regs.set_word(self.dst, matches as u64);
        self.next.run(m, r0)
    }
}

/// # Safety
/// The run this shape names is live UTF-8 for the call.
#[inline]
unsafe fn lent<'a>(regs: &'a Regs<'_>, text: LentText) -> &'a str {
    match text {
        // SAFETY: the caller's contract, and the checker admits only a live
        // reference to a `String` at this shape.
        LentText::Through(slot) => unsafe { regs.peek(slot).target().as_str() },
        LentText::Pair(pair) => {
            let words = Words {
                ptr: regs.word(pair.ptr),
                len: regs.word(pair.len),
            };
            // SAFETY: the caller's contract for liveness, and the encoding is
            // the obligation `StrView::as_str` names.
            unsafe { StrView::from_words(words).as_str() }
        }
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
            match *part {
                // SAFETY: the checker keeps the part live over this
                // operation.
                ConcatPart::Lent(text) => out.push_str(unsafe { lent(regs, text) }),
                ConcatPart::Owned(slot) => {
                    let held = regs.read(slot);
                    // SAFETY: the type checker admits only a `String` here.
                    out.push_str(unsafe { held.as_str() });
                    held.release();
                }
            }
        }
        regs.take_mask(self.owns_large);
        regs.define::<true>(self.dst, Value::string(out));
        self.next.run(m, r0)
    }
}
