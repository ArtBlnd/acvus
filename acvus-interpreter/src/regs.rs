//! The register file (RFC-0048 §3, RFC-0052 §5 and rule 7).
//!
//! A frame is one cell: 15 registers and the one word that marks which of them
//! own a `Large`, 256 bytes, four cache lines, starting one. A `Store` is an
//! array of such cells, and a call's frame is the **next cell** — so a call
//! neither allocates nor computes an offset: it makes one capacity compare and
//! steps one cell up. A body whose frame does not fit a cell gets a `Heap`
//! frame of its own, and a call out of that frame roots a new `Store`.

use std::mem::MaybeUninit;

use acvus_extern::Release;

use crate::code::Off;
use crate::value::Value;

/// A frame that fits this many registers runs in one cell.
pub const INLINE_SLOTS: u16 = 15;

/// A chain this many frames deep never reaches the allocator.
pub const INLINE_CELLS: usize = 4;

/// One mark word covers a frame, so this is where `prepare` stops even with a
/// `Heap` frame in hand.
pub const MAX_FRAME_SLOTS: u16 = 64;

/// One frame: its registers and its mark word, in four cache lines.
#[repr(align(64))]
pub struct Cell {
    slots: [MaybeUninit<Value>; INLINE_SLOTS as usize],
    marked: u64,
}

impl Cell {
    const EMPTY: Cell = Cell {
        slots: [const { MaybeUninit::uninit() }; INLINE_SLOTS as usize],
        marked: 0,
    };
}

const _: () = assert!(
    size_of::<Cell>() == 256 && align_of::<Cell>() == 64,
    "a frame is four cache lines and starts one"
);
const _: () = assert!(
    INLINE_SLOTS <= MAX_FRAME_SLOTS,
    "one mark word covers a cell"
);
const _: () = assert!(
    MAX_FRAME_SLOTS == Off::MAX_INDEX + 1,
    "the widest frame and the widest byte displacement are the same bound"
);
const _: () = assert!(INLINE_CELLS > 0, "a store holds the frame it is made for");

/// A frame too wide for a cell. It has the cell's mark word and none of its
/// locality, and a call out of it roots a new `Store` rather than growing this
/// one; `prepare` says which bodies take one.
struct Big {
    slots: Box<[MaybeUninit<Value>]>,
    marked: u64,
}

pub struct Store {
    held: Held,
}

enum Held {
    Cells(Box<[Cell; INLINE_CELLS]>),
    Heap(Big),
}

impl Store {
    /// # Panics
    /// `slots` is above `MAX_FRAME_SLOTS`, which `prepare` must not emit.
    pub fn new(slots: u16) -> Store {
        assert!(
            slots <= MAX_FRAME_SLOTS,
            "a body of {slots} registers was prepared past the {MAX_FRAME_SLOTS} one frame's \
             mark word reaches"
        );
        let held = match slots <= INLINE_SLOTS {
            true => Held::Cells(Box::new([const { Cell::EMPTY }; INLINE_CELLS])),
            false => Held::Heap(Big {
                slots: (0..slots).map(|_| MaybeUninit::uninit()).collect(),
                marked: 0,
            }),
        };
        Store { held }
    }

    /// The first frame of the chain.
    pub fn borrow(&mut self) -> Regs<'_> {
        match &mut self.held {
            Held::Cells(cells) => {
                let [first, above @ ..] = &mut **cells;
                Regs::of(&mut first.slots, &mut first.marked, above)
            }
            Held::Heap(big) => Regs::of(&mut big.slots, &mut big.marked, &mut []),
        }
    }
}

/// One frame: its registers, its mark word, and the cells above it that a call
/// out of it takes its callee's frame from.
///
/// `prepare::check_assignment` proves every register an operation names is
/// below its body's `frame_len`, and `prepare`'s liveness proves every register
/// an operation reads was defined on the path that reached it. Those two proofs
/// are what the `unsafe` here stands on; `debug_assert!` re-checks the first in
/// a debug build, and nothing can check the second at run time, which is why
/// the registers are `MaybeUninit` rather than a readable sentinel.
pub struct Regs<'f> {
    slots: &'f mut [MaybeUninit<Value>],
    marked: &'f mut u64,
    above: &'f mut [Cell],
    /// The registers a callee's frame may have here: a cell's worth while a
    /// cell is left, none otherwise. One compare per call reads it.
    above_cap: u16,
}

impl<'f> Regs<'f> {
    fn of(
        slots: &'f mut [MaybeUninit<Value>],
        marked: &'f mut u64,
        above: &'f mut [Cell],
    ) -> Regs<'f> {
        let above_cap = match above.is_empty() {
            true => 0,
            false => INLINE_SLOTS,
        };
        Regs {
            slots,
            marked,
            above,
            above_cap,
        }
    }

    /// The one capacity compare a call makes: whether the cell above this
    /// frame holds a callee frame of `callee_len` registers.
    #[inline(always)]
    pub fn fits_above(&self, callee_len: u16) -> bool {
        callee_len <= self.above_cap
    }

    /// The callee's frame: the cell above this one.
    ///
    /// # Panics
    /// No cell is left, which `fits_above` answers before this is called.
    #[inline(always)]
    pub fn window(&mut self) -> Regs<'_> {
        let (first, above) = self
            .above
            .split_first_mut()
            .expect("a call took the cell above a frame whose `fits_above` said it has none");
        Regs::of(&mut first.slots, &mut first.marked, above)
    }

    /// The register's first byte. No arithmetic: the operation's field is
    /// already the byte displacement (RFC-0052 §5).
    #[inline(always)]
    fn at(&self, off: Off) -> *const Value {
        debug_assert!(
            off.index() < self.slots.len(),
            "an operation names register {}, which its body's frame does not have",
            off.index()
        );
        // SAFETY: the two proofs stated on `Regs`.
        unsafe {
            self.slots
                .as_ptr()
                .cast::<u8>()
                .add(off.byte())
                .cast::<Value>()
        }
    }

    #[inline(always)]
    fn at_mut(&mut self, off: Off) -> *mut Value {
        debug_assert!(
            off.index() < self.slots.len(),
            "an operation names register {}, which its body's frame does not have",
            off.index()
        );
        // SAFETY: the two proofs stated on `Regs`.
        unsafe {
            self.slots
                .as_mut_ptr()
                .cast::<u8>()
                .add(off.byte())
                .cast::<Value>()
        }
    }

    #[inline(always)]
    pub fn read(&self, off: Off) -> Value {
        // SAFETY: the two proofs stated on `Regs`.
        unsafe { self.at(off).read() }
    }

    #[inline(always)]
    pub fn peek(&self, off: Off) -> &Value {
        // SAFETY: the two proofs stated on `Regs`.
        unsafe { &*self.at(off) }
    }

    #[inline(always)]
    pub fn peek_mut(&mut self, off: Off) -> &mut Value {
        // SAFETY: the two proofs stated on `Regs`.
        unsafe { &mut *self.at_mut(off) }
    }

    /// One 8-byte load: the kind byte was written when the frame was made and
    /// no run of a word-typed register changes it (RFC-0052 §5).
    #[inline(always)]
    pub fn word(&self, off: Off) -> u64 {
        self.peek(off).bits()
    }

    /// One 8-byte store, the other half of `word`.
    #[inline(always)]
    pub fn set_word(&mut self, off: Off, bits: u64) {
        *self.peek_mut(off).bits_mut() = bits;
    }

    /// One store, and one `or` on the cell's mark word where the operation's
    /// type says the value owns a `Large` (RFC-0048 §4).
    #[inline(always)]
    pub fn define<const LARGE: bool>(&mut self, off: Off, value: Value) {
        // SAFETY: `check_assignment`, as stated on `Regs`. A register this
        // overwrites was released by a drop instruction or never owned
        // (RFC-0041, RFC-0048 §6), so no value is lost here.
        unsafe { self.at_mut(off).write(value) };
        if LARGE {
            *self.marked |= off.mark();
        }
    }

    /// The store a call's result takes (RFC-0052 §5).
    #[inline(always)]
    pub fn store<const LARGE: bool, const WORD: bool>(&mut self, off: Off, value: Value) {
        const {
            assert!(
                !(LARGE && WORD),
                "a register whose kind the frame opened holds no Large"
            )
        }
        match WORD {
            true => self.set_word(off, value.bits()),
            false => self.define::<LARGE>(off, value),
        }
    }

    /// The frame's first write of a word-typed register, which fixes its kind
    /// for every `set_word` after it (RFC-0052 §5).
    #[inline]
    pub fn open(&mut self, off: Off, value: Value) {
        // SAFETY: `check_assignment`, as stated on `Regs`; this is the
        // frame's first write of the register.
        unsafe { self.at_mut(off).write(value) };
    }

    /// The value, and — where its type owns a `Large` — the frame's claim on
    /// it dropped. A word operand touches neither register nor mark word
    /// (RFC-0052 §5).
    #[inline(always)]
    pub fn take<const LARGE: bool>(&mut self, off: Off) -> Value {
        let value = self.read(off);
        if LARGE {
            *self.marked &= !off.mark();
        }
        value
    }

    /// The batched form: one `and` of the constant mask of the registers this
    /// operation consumes (RFC-0048 §5).
    #[inline(always)]
    pub fn take_mask(&mut self, mask: u64) {
        debug_assert!(
            *self.marked & mask == mask,
            "an operation takes a register its frame does not own: a double take"
        );
        *self.marked &= !mask;
    }

    /// The dual of `take_mask`: the claim an entry hands the frame on the
    /// registers it fills, in one `or` (RFC-0052 rule 7).
    #[inline(always)]
    pub fn own_mask(&mut self, mask: u64) {
        *self.marked |= mask;
    }

    /// RFC-0045: the old value is released before the new one lands.
    pub fn assign<const LARGE: bool>(&mut self, off: Off, value: Value) {
        let one = off.mark();
        if *self.marked & one != 0 {
            self.read(off).release();
        }
        // SAFETY: `check_assignment`, as stated on `Regs`, with the previous
        // owner released just above.
        unsafe { self.at_mut(off).write(value) };
        match LARGE {
            true => *self.marked |= one,
            false => *self.marked &= !one,
        }
    }

    /// Leaving: the set bits of the frame's mark word, released, and the word
    /// cleared. It iterates the set bits, never the registers — the 16-slot
    /// kind scan at every return is what RFC-0048 §6 removed.
    pub fn sweep(&mut self) {
        let mut live = *self.marked;
        while live != 0 {
            let bit = live.trailing_zeros();
            debug_assert!(
                bit < u32::from(MAX_FRAME_SLOTS),
                "a set mark bit is a register index, which the mark word bounds"
            );
            self.read(Off::of(bit as u16)).release();
            live &= live - 1;
        }
        *self.marked = 0;
    }

    /// The registers an extern call's arguments sit in, lent to the handler
    /// (RFC-0044, stage 2b).
    ///
    /// # Panics
    /// The run leaves the frame, which means `prepare` placed a call's
    /// argument window outside the frame it sized.
    #[inline]
    pub fn run_of(&self, at: Off, arity: u16) -> &[Value] {
        let from = at.index();
        let to = from + usize::from(arity);
        assert!(
            to <= self.slots.len(),
            "an argument run of {arity} at register {from} leaves a frame of {} registers",
            self.slots.len()
        );
        // SAFETY: `prepare` allocated the run contiguously in this frame and
        // every register of it is defined at the call.
        unsafe {
            std::slice::from_raw_parts(
                self.slots.as_ptr().cast::<Value>().add(from),
                usize::from(arity),
            )
        }
    }

    /// The frame's first register, which a chain's pre-multiplied leaf offsets
    /// are byte displacements from.
    #[inline]
    pub fn as_ptr(&self) -> *const Value {
        self.slots.as_ptr().cast::<Value>()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}
