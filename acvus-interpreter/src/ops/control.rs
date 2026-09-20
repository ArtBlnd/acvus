//! Control flow: the terminators that choose a block, the recognized regions
//! that run their own operations straight, and the instructions that produce
//! no control at all.
//!
//! Obligation across artifacts (RFC-0052 §3): a region's parts are operation
//! lists, so nothing here chooses a `BlockId` inside a region — and it is
//! `prepare::straight_run` that stops a region at the first instruction which
//! is not straight-line, so that no part ever needs one.
//!
//! Obligation across artifacts: a `Yield` hands back whatever its chain
//! computed last, which is any word at all. So `prepare` ends a chain whose
//! word a region reads as a verdict in `Fall`, `Break`, `Continue` or
//! `Return` instead, and the `Ending` parameter below is how it says which
//! chains those are.

#[cfg(any(debug_assertions, feature = "probe"))]
use crate::code::OwnedOps;
use std::marker::PhantomData;

use acvus_extern::Owned;

use crate::runtime::AcvusRuntime;

use crate::code::{
    AGAIN, BlockId, Exit, FALL, LEAVE, Marked, Off, Op, RETURN, SlicePair, successor,
};
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
///
/// Obligation across artifacts: under `PAIR`, `prepare::assign_slots` placed
/// the result in the two adjacent registers `SlicePair::at` derives, and the
/// caller's destination is a pair of the same shape (RFC-0062 Decision 1).
pub struct Return<const WORD: bool, const PAIR: bool> {
    pub slot: Marked,
}

impl<const WORD: bool, const PAIR: bool> Op for Return<WORD, PAIR> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        const {
            assert!(
                !(WORD && PAIR),
                "a view's two registers hold no kind the frame opened"
            )
        }
        match PAIR {
            true => {
                let pair = SlicePair::at(self.slot.at);
                let regs = m.regs();
                let (ptr, len) = (regs.word(pair.ptr), regs.word(pair.len));
                m.finish_pair(ptr, len);
            }
            false => {
                let value = match WORD {
                    true => {
                        let regs = m.regs();
                        let kind = regs.peek(self.slot.at).kind();
                        Value::inline(kind, regs.take_word(self.slot))
                    }
                    false => m.regs().take::<true>(self.slot),
                };
                m.finish(value);
            }
        }
        RETURN
    }
}

/// The last node of a region's inner chain: it hands the word the chain
/// computed last to the region that owns the chain (RFC-0052 §3).
///
/// A part chooses no block, so what it hands back is a word and not a
/// `BlockId`; where the part's last operation wrote its result to the frame
/// instead, the word here is whatever rode into the chain, and the region
/// that reads a frame register — `Loop<Slot, _>` — never looks at it.
pub struct Yield;

impl Op for Yield {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, r0: u64) -> Exit {
        r0
    }
}

pub struct Fall;

impl Op for Fall {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        FALL
    }
}

pub struct Break;

impl Op for Break {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        LEAVE
    }
}

pub struct Continue;

impl Op for Continue {
    #[inline]
    fn run(&self, _: &mut Machine<'_>, _: u64) -> Exit {
        AGAIN
    }
}

/// Obligation across artifacts: this is a type and not a `const` parameter so
/// that the demangled `Op::run` symbol names it. `benches/asm_probe.rs` reads
/// `Escapes` out of the symbol to know which operations have two ends — a
/// tail `jmp` to their successor and a `ret` carrying a verdict past it — and
/// a `bool` would reach the symbol as `true`, indistinguishable from any
/// other `true` an operation is monomorphized over.
pub trait Ending {
    const ESCAPES: bool;
}

pub struct Rejoins;

/// Some path through the chain ends at a `break`, a `continue`, a `?` or a
/// `return` inside it.
pub struct Escapes;

impl Ending for Rejoins {
    const ESCAPES: bool = false;
}

impl Ending for Escapes {
    const ESCAPES: bool = true;
}

/// What a loop does with the word its body chain handed back.
enum Handed {
    Iterate,
    Leave,
    /// The function is over: the word is `Return`'s, and it travels to the
    /// machine's loop through every region between.
    Over(Exit),
}

#[inline]
fn handed<E>(word: Exit) -> Handed
where
    E: Ending,
{
    match E::ESCAPES {
        false => Handed::Iterate,
        true => match word {
            FALL | AGAIN => Handed::Iterate,
            LEAVE => Handed::Leave,
            returned => Handed::Over(returned),
        },
    }
}

/// The `if` whose one arm ends in a `break`, a `continue`, a `?` or a
/// `return`, as `prepare::recognize_escape` finds it (RFC-0057 amended).
pub struct Escape<C, const ARM_ON: bool>
where
    C: Place,
{
    pub cond: C::At,
    pub arm: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> C>,
}

impl<C, const ARM_ON: bool> Op for Escape<C, ARM_ON>
where
    C: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        match (C::read(m.regs(), self.cond, r0) != 0) == ARM_ON {
            true => self.arm.run(m, r0),
            false => self.next.run(m, r0),
        }
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![OwnedOps {
            part: "arm",
            head: self.arm.as_ref(),
        }]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        vec![&mut self.arm]
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
///
/// `E` is the ending of `body`: over `Escapes` the body's chain ends in a
/// verdict node and this reads what it handed back, and over `Rejoins` the
/// body's word is the unread word `Yield` hands and there is no compare.
pub struct Loop<C, E>
where
    C: Place,
    E: Ending,
{
    pub head: Box<dyn Op>,
    pub cond: C::At,
    pub body: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (C, E)>,
}

impl<C, E> Op for Loop<C, E>
where
    C: Place,
    E: Ending,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        loop {
            let word = self.head.run(m, r0);
            if C::read(m.regs(), self.cond, word) == 0 {
                break;
            }
            match handed::<E>(self.body.run(m, r0)) {
                Handed::Iterate => {}
                Handed::Leave => break,
                Handed::Over(word) => return word,
            }
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
///
/// `E` is `Loop`'s parameter.
pub struct For<S, E>
where
    S: Source,
    E: Ending,
{
    pub src: S,
    pub body: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub ends: PhantomData<fn() -> E>,
}

impl<S, E> Op for For<S, E>
where
    S: Source,
    E: Ending,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let bound = self.src.bound(m.regs());
        let mut at = self.src.first(m.regs());
        while S::holds(at, bound) {
            self.src.lay(m.regs(), at);
            match handed::<E>(self.body.run(m, r0)) {
                Handed::Iterate => {}
                Handed::Leave => break,
                Handed::Over(word) => return word,
            }
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

pub struct ForStart<S>
where
    S: Source,
{
    pub src: S,
    pub counter: Off,
    pub next: Box<dyn Op>,
}

impl<S> Op for ForStart<S>
where
    S: Source,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let first = self.src.first(regs);
        regs.set_word(self.counter, first);
        self.next.run(m, r0)
    }
}

/// A `for` whose body does not rejoin keeps the counter and the bound in the
/// frame rather than in locals of one `run`, as `For<S>` does, and that is a
/// decision this operation's shape forces. The body's blocks hold a
/// terminator, so no region can own them; the header therefore has to be a
/// terminator too, and a terminator returns. What it costs is a load of the
/// counter and a read of the bound per iteration where the region pays
/// neither, and what it buys is a `break` and a `continue`, which are
/// ordinary edges out of and back into these blocks.
///
/// Cross-artifact obligation: an operation holding no successor ends its
/// chain in a `ret`, and `benches/asm_probe.rs` holds the list of the
/// families that may — `control::ForAt` is in `NO_SUCCESSOR` there.
pub struct ForAt<S>
where
    S: Source,
{
    pub src: S,
    pub counter: Off,
    pub body: BlockId,
    pub exit: BlockId,
}

impl<S> Op for ForAt<S>
where
    S: Source,
{
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let regs = m.regs();
        let at = regs.word(self.counter);
        let bound = self.src.bound(regs);
        match S::holds(at, bound) {
            true => {
                self.src.lay(regs, at);
                self.body.into()
            }
            false => self.exit.into(),
        }
    }
}

/// Cross-artifact obligation: which edges carry this is decided in
/// `prepare`'s `Jump` arm — every edge into a `for` header except the entry,
/// which carries `ForStart` instead. An edge that carries neither leaves the
/// counter where the last iteration left it and the loop does not terminate.
pub struct ForStep<S>
where
    S: Source,
{
    pub counter: Off,
    pub next: Box<dyn Op>,
    pub of: PhantomData<fn() -> S>,
}

impl<S> Op for ForStep<S>
where
    S: Source,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let at = regs.word(self.counter);
        regs.set_word(self.counter, S::step(at));
        self.next.run(m, r0)
    }
}

/// The `if/else` shape `prepare::recognize_diamond` finds in the IR
/// (RFC-0044, stage 5), as one operation holding both arms. The moves the
/// join edge carries are the last operations of each arm.
/// `E` is the ending of both arms: over `Escapes` an arm that reached a
/// `break`, a `continue` or a `return` hands its verdict past this operation
/// instead of rejoining, and the arm that did not hands `FALL`, which is why
/// no word rides out of this shape (`prepare::rides_from`).
pub struct Diamond<C, E>
where
    C: Place,
    E: Ending,
{
    pub cond: C::At,
    pub on_true: Box<dyn Op>,
    pub on_false: Box<dyn Op>,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (C, E)>,
}

impl<C, E> Op for Diamond<C, E>
where
    C: Place,
    E: Ending,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let arm = match C::read(m.regs(), self.cond, r0) != 0 {
            true => self.on_true.as_ref(),
            false => self.on_false.as_ref(),
        };
        let word = arm.run(m, r0);
        match (E::ESCAPES, word) {
            (true, FALL) | (false, _) => self.next.run(m, word),
            (true, verdict) => verdict,
        }
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
