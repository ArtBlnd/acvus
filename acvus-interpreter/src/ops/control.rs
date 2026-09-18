//! Control flow: the terminators that choose a block, the two recognized
//! regions that run their own operations straight, and the instructions that
//! produce no control at all.
//!
//! Obligation across artifacts (RFC-0052 §3): a region's parts are operation
//! lists, so nothing here chooses a `BlockId` inside a region — and it is
//! `prepare::straight_run` that stops a region at the first instruction which
//! is not straight-line, so that no part ever needs one.
//!
//! Decided against: a flag a region tests for a `return` inside it. A `return`
//! is not straight-line, so the recognizer stops at it and the shape prepares
//! as the blocks it was (`a_while_that_returns_is_not_a_region`).

#[cfg(any(debug_assertions, feature = "probe"))]
use crate::code::OwnedOps;
use crate::code::{BlockId, Off, Op, RETURN, Terminator};
use crate::machine::Machine;
use crate::value::Value;

/// One move of a parallel move, as an operation of the block it belongs to
/// (RFC-0052 rule 1). `prepare` ordered the sequence, so no source is read
/// after it is overwritten, and `LARGE` is what the moved value owns, so no
/// `run` tests a `kind`: a `Large` move is the copy plus two ops on the
/// frame's mark word, a word move is the copy alone.
///
/// `WORD` is the same parameter `CallExtern1` carries: both registers were
/// opened with their kind when the frame was made, so the move is the word
/// (RFC-0052 rule 5).
pub struct Mov<const LARGE: bool, const WORD: bool> {
    pub dst: Off,
    pub src: Off,
}

impl<const LARGE: bool, const WORD: bool> Op for Mov<LARGE, WORD> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        const {
            assert!(
                !(LARGE && WORD),
                "a register whose kind the frame opened holds no Large"
            )
        }
        let regs = m.regs();
        match WORD {
            true => {
                let bits = regs.take_word(self.src);
                regs.set_word(self.dst, bits);
            }
            false => {
                let value = regs.take::<LARGE>(self.src);
                regs.define::<LARGE>(self.dst, value);
            }
        }
    }
}

/// The edge that carries no arguments, which is most of them.
pub struct Goto {
    pub target: BlockId,
}

impl Terminator for Goto {
    #[inline]
    fn next(&self, _: &mut Machine<'_>) -> BlockId {
        self.target
    }
}

/// The two edges of a conditional jump. Where an edge carries a parallel
/// move, `prepare` gives that edge a block of its own holding the `Mov`s, so
/// this terminator is one word load, one test and one `cmov`.
pub struct JumpIf {
    pub cond: Off,
    pub on_true: BlockId,
    pub on_false: BlockId,
}

impl Terminator for JumpIf {
    #[inline]
    fn next(&self, m: &mut Machine<'_>) -> BlockId {
        match m.regs().word(self.cond) != 0 {
            true => self.on_true,
            false => self.on_false,
        }
    }
}

/// The body's result, read at the width its register was written at
/// (RFC-0052 rule 5).
pub struct Return<const WORD: bool> {
    pub slot: Off,
}

impl<const WORD: bool> Terminator for Return<WORD> {
    #[inline]
    fn next(&self, m: &mut Machine<'_>) -> BlockId {
        let value = match WORD {
            true => {
                let regs = m.regs();
                let kind = regs.peek(self.slot).kind();
                Value::inline(kind, regs.take_word(self.slot))
            }
            false => m.regs().take::<true>(self.slot),
        };
        m.finish(value);
        RETURN
    }
}

/// The operations of one part of a region, run in order.
#[inline(always)]
fn run_part(ops: &[Box<dyn Op>], m: &mut Machine<'_>) {
    for op in ops {
        op.run(m);
    }
}

/// The `while` shape `prepare::recognize_loop` finds in the IR (RFC-0044,
/// stage 3), as one operation holding its two op lists.
///
/// Every move this shape used to interpret is an operation `prepare` placed:
/// the entering move before this operation, the move into the body at the head
/// of `body`, the back edge at the end of `body`, and the exiting move after
/// this operation.
pub struct Loop {
    pub head: Box<[Box<dyn Op>]>,
    pub cond: Off,
    pub body: Box<[Box<dyn Op>]>,
}

impl Op for Loop {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        loop {
            run_part(&self.head, m);
            if m.regs().word(self.cond) == 0 {
                return;
            }
            run_part(&self.body, m);
        }
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![
            OwnedOps {
                part: "head",
                ops: &self.head,
            },
            OwnedOps {
                part: "body",
                ops: &self.body,
            },
        ]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut [Box<dyn Op>]> {
        vec![&mut self.head, &mut self.body]
    }
}

/// The `if/else` shape `prepare::recognize_diamond` finds in the IR
/// (RFC-0044, stage 5), as one operation holding both arms. The moves the
/// join edge carries are the last operations of each arm.
pub struct Diamond {
    pub cond: Off,
    pub on_true: Box<[Box<dyn Op>]>,
    pub on_false: Box<[Box<dyn Op>]>,
}

impl Op for Diamond {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        let arm = match m.regs().word(self.cond) != 0 {
            true => &self.on_true,
            false => &self.on_false,
        };
        run_part(arm, m);
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![
            OwnedOps {
                part: "on_true",
                ops: &self.on_true,
            },
            OwnedOps {
                part: "on_false",
                ops: &self.on_false,
            },
        ]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut [Box<dyn Op>]> {
        vec![&mut self.on_true, &mut self.on_false]
    }
}

/// A call typed `!`. It is a terminator because nothing follows it: the
/// handler panics, and a block that held this as an operation would have a
/// successor no path reaches.
pub struct Diverge;

impl Terminator for Diverge {
    fn next(&self, _: &mut Machine<'_>) -> BlockId {
        panic!("a call typed `!` returned: its handler must panic")
    }
}

pub struct Merge {
    pub dst: Off,
}

impl Op for Merge {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        m.regs().define::<false>(self.dst, Value::unit());
    }
}

/// RFC-0052 §5 fixes a word-typed slot's kind at the frame's making; which
/// word sits under it here does not matter, `acvus_mir::ir::InstKind::Undef`
/// being UB to read as a concrete value.
pub struct Undef<const WORD: bool> {
    pub dst: Off,
}

impl Op for Undef<true> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        m.regs().set_word(self.dst, 0);
    }
}

impl Op for Undef<false> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        m.regs().define::<false>(self.dst, Value::UNDEF);
    }
}

/// Release whatever the register still owns (RFC-0041's drop instruction).
///
/// `prepare` emits this only where the value's type owns a `Large`, which is
/// why the take is `take::<true>`: `acvus_mir`'s drop insertion emits no drop
/// for a storage it saw emptied, and a take of a flat option's payload is one
/// such emptying — the payload is the option's whole value (RFC-0022). A drop
/// arriving at a slot the frame no longer marks is therefore a defect in the
/// lowering, and `Regs::take`'s debug assert is where it surfaces.
pub struct DropValue {
    pub slot: Off,
}

impl Op for DropValue {
    #[inline]
    fn run(&self, m: &mut Machine<'_>) {
        use acvus_extern::Release;
        m.regs().take::<true>(self.slot).release();
    }
}

/// A terminator for the same reason `Diverge` is: the lowering put it where
/// no path may arrive, so no block follows it.
pub struct Poison;

impl Terminator for Poison {
    fn next(&self, _: &mut Machine<'_>) -> BlockId {
        panic!("reached poison instruction")
    }
}
