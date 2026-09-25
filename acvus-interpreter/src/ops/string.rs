//! Strings: a template's output, equality, and the clone the language's
//! `StringClone` instruction is (RFC-0020).

use acvus_extern::{Release, StrView, Words};

#[cfg(any(debug_assertions, feature = "probe"))]
use crate::code::OwnedOps;
use crate::code::{BlockId, ConcatPart, Exit, LentText, Marked, Off, Op, successor};
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
/// compared, with no value made (RFC-0062 rule 3). `pattern::TestString`
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
        // SAFETY: the caller's contract, and the checker admits only a
        // `String` at this shape.
        LentText::Own(slot) => unsafe { regs.peek(slot).as_str() },
        LentText::Through(slot) => unsafe { regs.peek(slot).target().as_str() },
        LentText::Pair(pair) => {
            // SAFETY: the pair is a `&str` pair, written from `into_pair` as
            // `ops::index::words` states; the caller's contract for liveness;
            // and the encoding is the obligation `StrView::as_str` names.
            unsafe {
                let words = Words::from_pair([regs.word(pair.ptr), regs.word(pair.len)]);
                StrView::from_words(words).as_str()
            }
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

/// A template's append: the part's bytes are written onto the end of the
/// `String` the `&mut` in `target` names, so the accumulator is extended
/// rather than rebuilt (RFC-0071).
pub struct Append {
    pub target: Off,
    pub part: ConcatPart,
    /// The part's mark bit where the part is a `String` this operation
    /// moves in, as `Concat`'s `owns_large` is.
    pub owns_large: u64,
    pub next: Box<dyn Op>,
}

impl Op for Append {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let reference = regs.read(self.target);
        // SAFETY: the checker admits only a live `&mut String` here, and
        // the loans keep the accumulator's storage live across this
        // operation.
        let out = unsafe { reference.target_mut().peek_mut::<String>() };
        match self.part {
            // SAFETY: the checker keeps the part live over this operation.
            ConcatPart::Lent(text) => out.push_str(unsafe { lent(regs, text) }),
            ConcatPart::Owned(slot) => {
                let held = regs.read(slot);
                // SAFETY: the type checker admits only a `String` here.
                out.push_str(unsafe { held.as_str() });
                held.release();
            }
        }
        regs.take_mask(self.owns_large);
        self.next.run(m, r0)
    }
}

/// One tested arm of a text dispatch: the string it names and the block the
/// machine enters for it.
pub struct StrArm {
    pub key: Box<str>,
    pub target: BlockId,
}

/// A `match` on string literals (RFC-0051): the text is read once and the
/// arms are scanned in the order the `match` wrote them.
///
/// Decision not to build: the sorted keys and the binary search RFC-0051 rejects
/// for a table. A string comparison starts with the length, so a
/// missing arm costs one word compare, and the arm counts here are the
/// handful a `match` on names is written with — the same measurement that
/// kept `Switch`'s scan over a hashed table at seven arms.
pub struct SwitchStr {
    pub src: LentText,
    pub arms: Box<[StrArm]>,
    pub default: BlockId,
}

impl Op for SwitchStr {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        // SAFETY: the checker keeps the scrutinee live over this operation.
        let text = unsafe { lent(m.regs(), self.src) };
        self.arms
            .iter()
            .find(|arm| &*arm.key == text)
            .map_or(self.default, |arm| arm.target)
            .into()
    }
}

/// One arm of a rejoining text dispatch: the string it names and the chain
/// the machine runs for it (RFC-0052 rule 3).
pub struct StrRegionArm {
    pub key: Box<str>,
    pub head: Box<dyn Op>,
}

/// The region form of [`SwitchStr`].
pub struct SwitchStrRegion {
    pub src: LentText,
    pub arms: Box<[StrRegionArm]>,
    pub default: Box<dyn Op>,
    pub next: Box<dyn Op>,
}

impl Op for SwitchStrRegion {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: the checker keeps the scrutinee live over this operation.
        let text = unsafe { lent(m.regs(), self.src) };
        let arm = self
            .arms
            .iter()
            .find(|arm| &*arm.key == text)
            .map_or(self.default.as_ref(), |arm| arm.head.as_ref());
        let word = arm.run(m, r0);
        self.next.run(m, word)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        self.arms
            .iter()
            .map(|arm| OwnedOps {
                part: "arm",
                head: arm.head.as_ref(),
            })
            .chain([OwnedOps {
                part: "default",
                head: self.default.as_ref(),
            }])
            .collect()
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        self.arms
            .iter_mut()
            .map(|arm| &mut arm.head)
            .chain([&mut self.default])
            .collect()
    }
}
