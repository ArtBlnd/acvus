//! Constants: the literal an operation defines its destination with.

use crate::code::{BlockId, Konst, Off, Op, successor};
use crate::machine::Machine;

/// The word of an integer, a float, a bool or unit. The slot's kind was
/// written when the frame was made (RFC-0052 §5), so the literal's type is
/// not in this operation at all.
pub struct Const {
    pub dst: Off,
    pub word: u64,
    pub next: Box<dyn Op>,
}

impl Op for Const {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        m.regs().set_word(self.dst, self.word);
        self.next.run(m, r0)
    }
}

/// A string or a list literal. The value is built at every run and owned by
/// the frame, not shared from the prepared body: the program may mutate what
/// a literal defined, and a shared one would need a copy at the first write.
pub struct ConstLarge {
    pub dst: Off,
    pub konst: Konst,
    pub next: Box<dyn Op>,
}

impl Op for ConstLarge {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let value = self.konst.value();
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}
