//! Indexing a slice (RFC-0047 amended): the one element access the machine
//! does without knowing a container.
//!
//! A slice is a register pair — `ptr` then `len`, adjacent by
//! `prepare::assign_slots` — so reading element `i` is two register loads,
//! one compare and one dependent load. Nothing is boxed and nothing is
//! freed: a slice owns nothing (RFC-0048).

use acvus_extern::{Elements, Release, Words};
use acvus_mir::ir::IndexMode;

use crate::code::{Exit, Marked, Off, Op, SlicePair, successor};
use crate::machine::Machine;
use crate::regs::Regs;
use crate::runtime::AcvusRuntime;
use crate::value::{Kind, Value};

type Run = Elements<AcvusRuntime>;

/// The registers an indexed read names.
#[derive(Clone, Copy)]
pub struct Read {
    pub dst: Off,
    pub slice: SlicePair,
    pub index: Off,
}

/// The text Rust's own slice index gives; a test pins it byte for byte.
fn out_of_bounds(len: usize, index: u64) -> String {
    format!("index out of bounds: the len is {len} but the index is {index}")
}

/// The run the pair at `slice` holds.
///
/// # Safety
/// The container the slice borrows is live and unmoved, which the loan the
/// slice holds keeps true for as long as the pair is live (RFC-0018).
#[inline(always)]
unsafe fn run(regs: &Regs<'_>, slice: SlicePair) -> Run {
    let words = Words {
        ptr: regs.word(slice.ptr),
        len: regs.word(slice.len),
    };
    // SAFETY: the caller's contract, and `AsSlice` is what wrote the pair.
    unsafe { Run::from_words(words) }
}

/// The element position `index` names, checked against `run` where the
/// instruction carries no proof of the bound.
#[inline]
fn position<const CHECKED: bool>(run: &Run, index: u64) -> usize {
    if CHECKED && index >= run.len() as u64 {
        panic!("{}", out_of_bounds(run.len(), index));
    }
    // The bound above, or the interval pass's proof (RFC-0047 §7), puts
    // `index` below a length, which is a `usize`.
    index as usize
}

/// Element `index` of the run the pair at `slice` holds.
///
/// # Safety
/// As `run`.
#[inline]
unsafe fn element<'a, const CHECKED: bool>(
    regs: &Regs<'_>,
    slice: SlicePair,
    index: u64,
) -> &'a Value {
    // SAFETY: the caller's contract.
    let run = unsafe { run(regs, slice) };
    let at = position::<CHECKED>(&run, index);
    // SAFETY: the caller's contract, and `position` put `at` within the run.
    unsafe { run.at(at) }
}

pub struct IndexCopy<const CHECKED: bool> {
    pub read: Read,
    pub next: Box<dyn Op>,
}

impl<const CHECKED: bool> Op for IndexCopy<CHECKED> {
    successor!();

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn index_read(&self) -> Option<Read> {
        CHECKED.then_some(self.read)
    }

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let index = regs.word(self.read.index);
        // SAFETY: the slice holds its container's loan.
        let value = *unsafe { element::<CHECKED>(regs, self.read.slice, index) };
        debug_assert_ne!(
            value.kind(),
            Kind::Large,
            "an indexed copy leaves the container owning the element"
        );
        regs.put(self.read.dst, value);
        self.next.run(m, r0)
    }
}

pub struct IndexRef<const CHECKED: bool> {
    pub read: Read,
    pub next: Box<dyn Op>,
}

impl<const CHECKED: bool> Op for IndexRef<CHECKED> {
    successor!();

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn index_read(&self) -> Option<Read> {
        CHECKED.then_some(self.read)
    }

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let index = regs.word(self.read.index);
        // SAFETY: as `IndexCopy`.
        let target = unsafe { element::<CHECKED>(regs, self.read.slice, index) };
        regs.put(self.read.dst, Value::reference(target));
        self.next.run(m, r0)
    }
}

/// An element assignment releases what it overwrites (RFC-0045), which is
/// `LARGE`: what the preparation read from the element type.
pub struct IndexSet<const CHECKED: bool, const LARGE: bool> {
    pub slice: SlicePair,
    pub index: Off,
    pub value: Marked,
    pub next: Box<dyn Op>,
}

impl<const CHECKED: bool, const LARGE: bool> Op for IndexSet<CHECKED, LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let index = regs.word(self.index);
        let value = regs.take::<LARGE>(self.value);
        // SAFETY: the slice holds its container's loan.
        let slice = unsafe { run(regs, self.slice) };
        let at = position::<CHECKED>(&slice, index);
        // SAFETY: the operand is a `&mut [T]`, so its run is named once
        // here, and `position` put `at` within it.
        let slot = unsafe { slice.at_mut(at) };
        let overwritten = *slot;
        *slot = value;
        release_if::<LARGE>(overwritten);
        self.next.run(m, r0)
    }
}

#[inline(always)]
fn release_if<const LARGE: bool>(value: Value) {
    if LARGE {
        value.release();
    }
}

/// The operation an `Index` instruction prepares to.
pub fn checked(mode: IndexMode, read: Read, next: Box<dyn Op>) -> Box<dyn Op> {
    match mode {
        IndexMode::Copy => Box::new(IndexCopy::<true> { read, next }),
        IndexMode::Ref => Box::new(IndexRef::<true> { read, next }),
    }
}

/// The same operation without the bound check (RFC-0047 §7). Nothing in
/// `prepare` reaches it: the MIR holds no unchecked instruction, and until
/// the interval pass carries its own proof the only way to run one is for a
/// probe to substitute it into a prepared body.
#[cfg(any(test, feature = "probe"))]
pub fn unchecked(mode: IndexMode, read: Read, next: Box<dyn Op>) -> Box<dyn Op> {
    match mode {
        IndexMode::Copy => Box::new(IndexCopy::<false> { read, next }),
        IndexMode::Ref => Box::new(IndexRef::<false> { read, next }),
    }
}
