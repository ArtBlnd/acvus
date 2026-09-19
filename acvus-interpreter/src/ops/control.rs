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
use std::marker::PhantomData;

use acvus_extern::Owned;

use crate::runtime::AcvusRuntime;

use crate::code::{BlockId, Exit, Marked, Off, Op, RETURN, SlicePair, successor};
use crate::machine::Machine;
use crate::ops::arith::Int;
use crate::ops::place::Place;
use crate::regs::Regs;
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
    pub dst: Marked,
    pub src: Marked,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool> Op for Mov<LARGE, WORD> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
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
                regs.set_word(self.dst.at, bits);
            }
            false => {
                let value = regs.take::<LARGE>(self.src);
                regs.define::<LARGE>(self.dst, value);
            }
        }
        self.next.run(m, r0)
    }
}

/// A slice's move: the two adjacent registers `prepare::assign_slots` gave
/// it (RFC-0047 amended, rule 4).
///
/// Decided against the narrower two `set_word`s: `prepare::order_moves`
/// routes a cycle through the scratch registers, whose kind bytes are
/// unopened for the reason stated there.
pub struct MovWide {
    pub dst: SlicePair,
    pub src: SlicePair,
    pub next: Box<dyn Op>,
}

impl Op for MovWide {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let ptr = regs.read(self.src.ptr);
        let len = regs.read(self.src.len);
        regs.put(self.dst.ptr, ptr);
        regs.put(self.dst.len, len);
        self.next.run(m, r0)
    }
}

/// The edge that carries no arguments, which is most of them.
pub struct Goto {
    pub target: BlockId,
}

impl Op for Goto {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        self.target.into()
    }
}

/// The two edges of a conditional jump. Where an edge carries a parallel
/// move, `prepare` gives that edge a block of its own holding the `Mov`s, so
/// this terminator is one word load, one test and one `cmov`.
pub struct JumpIf<C>
where
    C: Place,
{
    pub cond: C::At,
    pub on_true: BlockId,
    pub on_false: BlockId,
    pub at: PhantomData<fn() -> C>,
}

impl<C> Op for JumpIf<C>
where
    C: Place,
{
    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        match C::read(m.regs(), self.cond, r0) != 0 {
            true => self.on_true.into(),
            false => self.on_false.into(),
        }
    }
}

/// The body's result, read at the width its register was written at
/// (RFC-0052 rule 5).
pub struct Return<const WORD: bool> {
    pub slot: Marked,
}

impl<const WORD: bool> Op for Return<WORD> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let value = match WORD {
            true => {
                let regs = m.regs();
                let kind = regs.peek(self.slot.at).kind();
                Value::inline(kind, regs.take_word(self.slot))
            }
            false => m.regs().take::<true>(self.slot),
        };
        m.finish(value);
        RETURN
    }
}

/// The last node of a region's inner chain: it hands the word the chain
/// computed last to the region that owns the chain (RFC-0052 §3).
///
/// A part chooses no block, so what it hands back is a word and not a
/// `BlockId`; where the part's last operation wrote its result to the frame
/// instead, the word here is whatever rode into the chain, and the region
/// that reads a frame register — `Loop<Slot>` — never looks at it.
pub struct Yield;

impl Op for Yield {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, r0: u64) -> Exit {
        r0
    }
}

/// The `while` shape `prepare::recognize_loop` finds in the IR (RFC-0044,
/// stage 3), as one operation holding its two chains.
///
/// Every move this shape used to interpret is an operation `prepare` placed:
/// the entering move before this operation, the move into the body at the head
/// of `body`, the back edge at the end of `body`, and the exiting move after
/// this operation.
/// `C` is where the head's condition is: `place::R0` where the head chain's
/// last operation produced it — the word rides out of the part into the test
/// here, which is why the head is run for its return value — and
/// `place::Slot` where it does not, which is the head whose last word is not
/// the condition (a call tested later, an `Option` test). `prepare` chose
/// between them, so this `run` holds no test of its own.
pub struct Loop<C>
where
    C: Place,
{
    pub head: Box<dyn Op>,
    pub cond: C::At,
    pub body: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> C>,
}

impl<C> Op for Loop<C>
where
    C: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        loop {
            let word = self.head.run(m, r0);
            if C::read(m.regs(), self.cond, word) == 0 {
                break;
            }
            self.body.run(m, r0);
        }
        self.next.run(m, r0)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![
            OwnedOps {
                part: "head",
                head: self.head.as_ref(),
            },
            OwnedOps {
                part: "body",
                head: self.body.as_ref(),
            },
        ]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        vec![&mut self.head, &mut self.body]
    }
}

pub trait Source: Send + Sync + 'static {
    type Bound: Copy;

    fn bound(&self, regs: &Regs<'_>) -> Self::Bound;
    fn first(&self, regs: &Regs<'_>) -> u64;
    fn holds(at: u64, bound: Self::Bound) -> bool;
    fn step(at: u64) -> u64;

    /// Cross-artifact obligation: which of the body block's leading
    /// parameters this fills, and in which order, is decided in
    /// `acvus_mir::ir::ForSource::supplied_params` and `counter_param`. The
    /// body reads those registers by position, so an implementation that
    /// lays them in another order compiles here and reads the wrong register
    /// there.
    fn lay(&self, regs: &mut Regs<'_>, at: u64);
}

/// One operation for both `for x in &v` and `for x in &mut v`: the two heads
/// differ in the loan the terminator carries and not in what the machine does.
pub struct Slice {
    pub slice: SlicePair,
    pub elem: Off,
    pub index: Off,
}

impl Source for Slice {
    type Bound = u64;

    #[inline(always)]
    fn bound(&self, regs: &Regs<'_>) -> u64 {
        regs.word(self.slice.len)
    }

    #[inline(always)]
    fn first(&self, _regs: &Regs<'_>) -> u64 {
        0
    }

    #[inline(always)]
    fn holds(at: u64, bound: u64) -> bool {
        at < bound
    }

    #[inline(always)]
    fn step(at: u64) -> u64 {
        at + 1
    }

    #[inline(always)]
    fn lay(&self, regs: &mut Regs<'_>, at: u64) {
        // SAFETY: the slice holds its container's loan, and `at` is below the
        // length `bound` read — the terminator is the bound, so the read is
        // the unchecked one (RFC-0057 Decision 3).
        let target = unsafe { crate::ops::index::element::<false>(regs, self.slice, at) };
        regs.put(self.elem, Value::reference(target));
        regs.set_word(self.index, at);
    }
}

/// Cross-artifact obligation: this operation does not release the array, and
/// it must not. `acvus_mir::optimize::drop_insertion` emits a `Drop` of the
/// array on the loop's exit block, which is what releases the storage and the
/// slots the counter never reached; `acvus mir` over `for x in a { … }` prints
/// it as the `drop` above the exit's `return`. So the array's register keeps
/// its value and the frame keeps its claim on it for that `Drop` to take.
pub struct Array<const LARGE: bool, const WORD: bool> {
    pub array: Off,
    pub elem: Marked,
    pub index: Off,
}

impl<const LARGE: bool, const WORD: bool> Source for Array<LARGE, WORD> {
    type Bound = u64;

    #[inline(always)]
    fn bound(&self, regs: &Regs<'_>) -> u64 {
        // SAFETY: the preparation read `Array` off the source's type.
        unsafe { regs.peek(self.array).as_array() }.len() as u64
    }

    #[inline(always)]
    fn first(&self, _regs: &Regs<'_>) -> u64 {
        0
    }

    #[inline(always)]
    fn holds(at: u64, bound: u64) -> bool {
        at < bound
    }

    #[inline(always)]
    fn step(at: u64) -> u64 {
        at + 1
    }

    #[inline(always)]
    fn lay(&self, regs: &mut Regs<'_>, at: u64) {
        // SAFETY: as `bound`; `at` is below the length that read.
        let slot = unsafe { &mut regs.peek_mut(self.array).as_array_mut().0[at as usize] };
        let taken: Owned<AcvusRuntime> = std::mem::replace(slot, Owned::from_value(Value::UNDEF));
        regs.store::<LARGE, WORD>(self.elem, taken.into_value());
        regs.set_word(self.index, at);
    }
}

pub struct Range<T>
where
    T: Int,
{
    pub hi: Off,
    pub elem: Off,
    pub from: Off,
    pub width: PhantomData<fn() -> T>,
}

impl<T> Source for Range<T>
where
    T: Int,
{
    type Bound = T;

    #[inline(always)]
    fn bound(&self, regs: &Regs<'_>) -> T {
        T::read(regs.word(self.hi))
    }

    #[inline(always)]
    fn first(&self, regs: &Regs<'_>) -> u64 {
        regs.word(self.from)
    }

    #[inline(always)]
    fn holds(at: u64, bound: T) -> bool {
        T::read(at) < bound
    }

    #[inline(always)]
    fn step(at: u64) -> u64 {
        T::read(at).wrapping_add(T::read(1)).word()
    }

    #[inline(always)]
    fn lay(&self, regs: &mut Regs<'_>, at: u64) {
        regs.set_word(self.elem, at);
    }
}

/// The shape `prepare::recognize_for` finds in the IR (RFC-0057 Decision 3).
pub struct For<S>
where
    S: Source,
{
    pub src: S,
    pub body: Box<dyn Op>,
    pub next: Box<dyn Op>,
}

impl<S> Op for For<S>
where
    S: Source,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let bound = self.src.bound(m.regs());
        let mut at = self.src.first(m.regs());
        while S::holds(at, bound) {
            self.src.lay(m.regs(), at);
            self.body.run(m, r0);
            at = S::step(at);
        }
        self.next.run(m, r0)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![OwnedOps {
            part: "body",
            head: self.body.as_ref(),
        }]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        vec![&mut self.body]
    }
}

/// The `if/else` shape `prepare::recognize_diamond` finds in the IR
/// (RFC-0044, stage 5), as one operation holding both arms. The moves the
/// join edge carries are the last operations of each arm.
pub struct Diamond<C>
where
    C: Place,
{
    pub cond: C::At,
    pub on_true: Box<dyn Op>,
    pub on_false: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> C>,
}

impl<C> Op for Diamond<C>
where
    C: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let arm = match C::read(m.regs(), self.cond, r0) != 0 {
            true => self.on_true.as_ref(),
            false => self.on_false.as_ref(),
        };
        let word = arm.run(m, r0);
        self.next.run(m, word)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![
            OwnedOps {
                part: "on_true",
                head: self.on_true.as_ref(),
            },
            OwnedOps {
                part: "on_false",
                head: self.on_false.as_ref(),
            },
        ]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        vec![&mut self.on_true, &mut self.on_false]
    }
}

/// A call typed `!`. It is a terminator because nothing follows it: the
/// handler panics, and a block that held this as an operation would have a
/// successor no path reaches.
pub struct Diverge;

impl Op for Diverge {
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        panic!("a call typed `!` returned: its handler must panic")
    }
}

pub struct Merge {
    pub dst: Off,
    pub next: Box<dyn Op>,
}

impl Op for Merge {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().put(self.dst, Value::unit());
        self.next.run(m, r0)
    }
}

/// RFC-0052 §5 fixes a word-typed slot's kind at the frame's making; which
/// word sits under it here does not matter, `acvus_mir::ir::InstKind::Undef`
/// being UB to read as a concrete value.
pub struct Undef<const WORD: bool> {
    pub dst: Off,
    pub next: Box<dyn Op>,
}

impl Op for Undef<true> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().set_word(self.dst, 0);
        self.next.run(m, r0)
    }
}

impl Op for Undef<false> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().put(self.dst, Value::UNDEF);
        self.next.run(m, r0)
    }
}

/// The same for a slice's pair, whose two registers are word class.
pub struct UndefWide {
    pub dst: SlicePair,
    pub next: Box<dyn Op>,
}

impl Op for UndefWide {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        regs.set_word(self.dst.ptr, 0);
        regs.set_word(self.dst.len, 0);
        self.next.run(m, r0)
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
    pub slot: Marked,
    pub next: Box<dyn Op>,
}

impl Op for DropValue {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        use acvus_extern::Release;
        m.regs().take::<true>(self.slot).release();
        self.next.run(m, r0)
    }
}

/// A terminator for the same reason `Diverge` is: the lowering put it where
/// no path may arrive, so no block follows it.
pub struct Poison;

impl Op for Poison {
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        panic!("reached poison instruction")
    }
}
