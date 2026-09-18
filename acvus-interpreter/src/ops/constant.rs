//! Constants: the literal an operation defines its destination with.

use crate::code::{Konst, Off, Op};
use crate::machine::Machine;

/// The word of an integer, a float, a bool or unit. The slot's kind was
/// written when the frame was made (RFC-0052 §5), so the literal's type is
/// not in this operation at all.
pub struct Const {
    pub dst: Off,
    pub word: u64,
}

impl Op for Const {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        m.regs().set_word(self.dst, self.word);
    }
}

/// A string or a list literal. The value is built at every run and owned by
/// the frame, not shared from the prepared body: the program may mutate what
/// a literal defined, and a shared one would need a copy at the first write.
pub struct ConstLarge {
    pub dst: Off,
    pub konst: Konst,
}

impl Op for ConstLarge {
    fn run(&self, m: &mut Machine<'_>) {
        let value = self.konst.value();
        m.regs().define::<true>(self.dst, value);
    }
}
