//! The buffer a loop lowered ahead keeps in one register of its frame
//! (RFC-0103 rule 2).
//!
//! Cross-artifact obligation: what the buffer still holds when it is
//! released, at a `break` or at the frame's end, includes spawn handles whose
//! jobs were never evaluated. Dropping a handle leaves its task counted in the
//! frame's tally (`flight::Unevaluated`), and that count is what keeps the
//! frame's cells alive until the job finishes (RFC-0046 rule 3).

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use acvus_extern::Release;

use crate::code::{Marked, Off};
use crate::regs::Regs;
use crate::value::Value;

#[derive(Clone, Copy, Debug)]
pub enum Crossing {
    Owning(Marked),
    Plain(Off),
}

pub(crate) struct Ring {
    bound: NonZeroUsize,
    next_issue: u64,
    stashes: VecDeque<Stash>,
}

impl Ring {
    pub(crate) fn new(bound: NonZeroUsize, first: u64) -> Ring {
        Ring {
            bound,
            next_issue: first,
            stashes: VecDeque::with_capacity(bound.get()),
        }
    }

    pub(crate) fn into_value(self) -> Value {
        // SAFETY: `Ring::held` is the one reader of the value, at `Ring`, and
        // the frame releases it through the vtable `erase` gave it.
        unsafe { Value::erase(self) }
    }

    /// # Safety
    /// `at` is the register `prepare` claimed for this loop's ring, which
    /// `ForAheadStart` defined with `into_value` on entering the loop and no
    /// operation outside the loop's header names.
    pub(crate) unsafe fn held<'r>(regs: &'r mut Regs<'_>, at: Off) -> &'r mut Ring {
        // SAFETY: the caller's contract: the register holds a `Ring`.
        unsafe { regs.peek_mut(at).peek_mut::<Ring>() }
    }

    pub(crate) fn issuable(&self) -> Option<u64> {
        (self.stashes.len() < self.bound.get()).then_some(self.next_issue)
    }

    pub(crate) fn push_issued(&mut self, stash: Stash, next_issue: u64) {
        debug_assert_eq!(
            stash.at, self.next_issue,
            "a stash was issued at another index than the one the ring had room for"
        );
        self.stashes.push_back(stash);
        self.next_issue = next_issue;
    }

    /// # Panics
    /// The oldest stash is not of `at`. Cross-artifact obligation: the header
    /// issues its own index before it asks, and `prepare` puts a `ForStep`
    /// that advances the counter by one on every edge back into the header,
    /// so the oldest stash is the header's own.
    pub(crate) fn take_oldest(&mut self, at: u64) -> Stash {
        let stash = self
            .stashes
            .pop_front()
            .unwrap_or_else(|| panic!("the header at index {at} holds no issued stash"));
        assert_eq!(
            stash.at, at,
            "the oldest issued stash is of another index than the header's"
        );
        stash
    }
}

pub(crate) struct Stash {
    at: u64,
    held: Vec<Held>,
}

enum Held {
    Owning(Marked, Value),
    Plain(Off, Value),
}

impl Stash {
    /// Cross-artifact obligation: an owning register's claim is dropped
    /// here, because `Regs::sweep` releases every register the frame still
    /// claims at its end, and the stash releases what it holds on its own.
    pub(crate) fn take(regs: &mut Regs<'_>, at: u64, crossing: &[Crossing]) -> Stash {
        let held = crossing
            .iter()
            .map(|register| match *register {
                Crossing::Owning(marked) => Held::Owning(marked, regs.take::<true>(marked)),
                Crossing::Plain(off) => Held::Plain(off, regs.read(off)),
            })
            .collect();
        Stash { at, held }
    }

    pub(crate) fn restore(mut self, regs: &mut Regs<'_>) {
        for held in std::mem::take(&mut self.held) {
            match held {
                Held::Owning(marked, value) => regs.define::<true>(marked, value),
                Held::Plain(off, value) => regs.put(off, value),
            }
        }
    }
}

impl Drop for Stash {
    fn drop(&mut self) {
        for held in std::mem::take(&mut self.held) {
            if let Held::Owning(_, value) = held {
                value.release();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use acvus_ast::Span;

    use crate::code::{Body, Literals};
    use crate::regs::{FrameSlot, MarkWords, Store, cells_for};

    const REGISTERS: u16 = 8;

    struct Counted {
        _alive: Arc<()>,
    }

    fn body() -> Body {
        Body {
            heads: Box::new([]),
            entry: 0,
            frame_len: REGISTERS,
            frame_cells: u16::try_from(cells_for(REGISTERS)).expect("a frame's cells fit a u16"),
            mark_words: MarkWords::of(REGISTERS),
            entry_konsts: Box::new([]),
            literals: Arc::new(Literals::of(std::iter::empty())),
            slot_kinds: Box::new([]),
            may_suspend: false,
            returns_a_view: false,
            params: Box::new([]),
            param_run: 0,
            param_marks: 0,
            captures: Box::new([]),
            order_param: None,
            span: Span::ZERO,
        }
    }

    fn counted(alive: &Arc<()>) -> Value {
        // SAFETY: the word is never materialized; `release` drops it as the
        // `Counted` it was erased from.
        unsafe {
            Value::erase(Counted {
                _alive: Arc::clone(alive),
            })
        }
    }

    fn marked(slot: u16) -> Marked {
        Marked::of(FrameSlot::of(slot))
    }

    const OWNING: u16 = 2;
    const PLAIN: u16 = 3;
    const RING: u16 = 5;

    /// A stash moved into the ring takes the frame's claim with it, so the
    /// frame's sweep releases the value once, through the ring.
    #[test]
    fn a_stash_left_in_the_ring_is_released_once_with_the_ring() {
        let body = body();
        let mut store = Store::new();
        let (mut regs, _) = store.bind(&body);
        regs.open_marks(body.mark_words);
        for slot in FrameSlot::first(usize::from(REGISTERS)) {
            regs.open(Off::of_below(slot), Value::UNDEF);
        }
        let alive = Arc::new(());
        let two = NonZeroUsize::new(2).expect("two is not zero");
        regs.assign::<true>(marked(RING), Ring::new(two, 0).into_value());
        regs.define::<true>(marked(OWNING), counted(&alive));
        regs.put(marked(PLAIN).at(), Value::int(7));

        let crossing = [Crossing::Owning(marked(OWNING)), Crossing::Plain(marked(PLAIN).at())];
        let stash = Stash::take(&mut regs, 0, &crossing);
        // SAFETY: the ring register was defined with `into_value` above.
        let ring = unsafe { Ring::held(&mut regs, marked(RING).at()) };
        assert_eq!(ring.issuable(), Some(0));
        ring.push_issued(stash, 1);
        assert_eq!(ring.issuable(), Some(1));
        assert_eq!(Arc::strong_count(&alive), 2);

        regs.sweep(body.mark_words);
        assert_eq!(Arc::strong_count(&alive), 1, "the stashed value is released once");
    }

    /// Taken back, the stash restores each register and the frame's claim
    /// on the owning one.
    #[test]
    fn a_stash_taken_back_restores_its_registers_and_their_claims() {
        let body = body();
        let mut store = Store::new();
        let (mut regs, _) = store.bind(&body);
        regs.open_marks(body.mark_words);
        for slot in FrameSlot::first(usize::from(REGISTERS)) {
            regs.open(Off::of_below(slot), Value::UNDEF);
        }
        let alive = Arc::new(());
        let one = NonZeroUsize::MIN;
        regs.assign::<true>(marked(RING), Ring::new(one, 4).into_value());
        regs.define::<true>(marked(OWNING), counted(&alive));
        regs.put(marked(PLAIN).at(), Value::int(7));

        let crossing = [Crossing::Owning(marked(OWNING)), Crossing::Plain(marked(PLAIN).at())];
        let stash = Stash::take(&mut regs, 4, &crossing);
        regs.put(marked(PLAIN).at(), Value::int(9));
        // SAFETY: the ring register was defined with `into_value` above.
        let ring = unsafe { Ring::held(&mut regs, marked(RING).at()) };
        ring.push_issued(stash, 5);
        assert_eq!(ring.issuable(), None, "a ring of one is full");
        let own = ring.take_oldest(4);
        own.restore(&mut regs);

        assert_eq!(regs.read(marked(PLAIN).at()).bits(), 7);
        assert_eq!(Arc::strong_count(&alive), 2);
        regs.sweep(body.mark_words);
        assert_eq!(Arc::strong_count(&alive), 1, "the restored value is the frame's again");
    }
}
