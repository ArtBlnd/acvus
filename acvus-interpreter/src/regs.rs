//! The register file (RFC-0048 §3, RFC-0052 §5, §6 and rule 7).
//!
//! A frame is a run of `Cell`s in one `Vec`: its registers, then the one
//! `Value`-wide slot that holds its mark word, then the window a call out of
//! it takes its callee's frame from. The caller owns the `Vec` and lends it
//! (RFC-0052 §6), so a call neither allocates nor frees: it computes one
//! displacement and steps past its own cells. An unbound frame is an empty
//! `Vec` — a stage whose closure is a `Code::Expr` never binds one and pays
//! nothing for it.

use std::mem::MaybeUninit;

use acvus_extern::Release;

use crate::code::{Body, Off, SlicePair};
use crate::value::Value;

/// The registers one cell holds: four cache lines of `Value`s.
pub const CELL_SLOTS: u16 = 16;

/// One mark word covers a frame, so this is where `prepare` stops.
pub const MAX_FRAME_SLOTS: u16 = 64;

/// The frame's mark word sits one `Value`-wide slot past its registers, so a
/// frame of `n` registers occupies the cells `n + 1` slots reach.
const MARK_SLOTS: u16 = 1;

/// The cells a bound frame keeps above itself, so that a call out of it runs
/// in this same `Vec`. Decision not to build: the `Vec` cannot grow while a
/// chain runs, because every frame below the growth point borrows from it, so
/// a chain deeper than this roots a `Store` of its own at the call
/// (`Machine::call_sync`).
const WINDOW_CELLS: usize = 3;

/// The cell a call lays its argument run in, which every bound frame keeps
/// above itself. Decision not to charge it to the callee: the run is written
/// before the caller knows whether the callee's whole frame fits above it, so
/// a frame that cannot offer this cell cannot host a call at all.
const ARG_CELLS: usize = 1;

/// One cell: four cache lines of registers, starting one. It holds no mark
/// word — a frame wider than a cell has to be one run of `Value`s, and an
/// interleaved word would break the displacement an `Off` already is.
#[repr(C, align(64))]
pub struct Cell {
    slots: [MaybeUninit<Value>; CELL_SLOTS as usize],
}

impl Cell {
    const fn uninit() -> Cell {
        Cell {
            slots: [const { MaybeUninit::uninit() }; CELL_SLOTS as usize],
        }
    }
}

const _: () = assert!(
    size_of::<Cell>() == 256 && align_of::<Cell>() == 64,
    "a cell is four cache lines and starts one"
);
const _: () = assert!(
    MAX_FRAME_SLOTS == Off::MAX_INDEX + 1,
    "the widest frame and the widest byte displacement are the same bound"
);
const _: () = assert!(
    ARG_CELLS <= WINDOW_CELLS,
    "a bound frame's window holds the cell its calls lay their arguments in"
);

/// The cells a frame of `slots` registers occupies, its mark word included.
#[inline(always)]
const fn cells_for(slots: u16) -> usize {
    (slots as usize + MARK_SLOTS as usize).div_ceil(CELL_SLOTS as usize)
}

/// The cells a frame of `MAX_FRAME_SLOTS` registers occupies: the most any
/// callee can ask of a window, and the cap `above_cap` is read against.
const MAX_FRAME_CELLS: usize = cells_for(MAX_FRAME_SLOTS);

/// Which body a frame is bound to, and so whether it already carries that
/// body's slot kinds and entry constants (`machine::open_frame`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoundTo(usize);

impl BoundTo {
    pub const NONE: BoundTo = BoundTo(0);

    /// Whether the frame was already bound to `body`, and bound to it now.
    #[inline]
    pub fn rebind(&mut self, body: &Body) -> bool {
        let key = BoundTo(std::ptr::from_ref(body).addr());
        debug_assert_ne!(key, BoundTo::NONE, "a body is not at address zero");
        std::mem::replace(self, key) == key
    }
}

/// The frame a call chain runs in: one `Vec`, owned by the caller and lent
/// per call (RFC-0052 §6).
pub struct Store {
    cells: Vec<Cell>,
    bound: BoundTo,
}

impl Store {
    /// A frame bound to no body: no cells, no allocation.
    pub fn new() -> Store {
        Store {
            cells: Vec::new(),
            bound: BoundTo::NONE,
        }
    }

    /// The frame `body` runs in, its mark word carrying `body`'s claim on its
    /// parameters, and whether the frame already carries `body`'s slot kinds
    /// and entry constants.
    ///
    /// Binding is what sizes the `Vec`, so no caller can borrow a frame
    /// narrower than the body it runs.
    ///
    /// # Panics
    /// `body.frame_len` is above `MAX_FRAME_SLOTS`, which `prepare` must not
    /// emit.
    #[inline]
    pub fn bind(&mut self, body: &Body) -> (Regs<'_>, bool) {
        let same = self.bound.rebind(body);
        if !same {
            self.widen(body.frame_len);
        }
        (Regs::of(&mut self.cells, body), same)
    }

    /// Room for a frame of `slots` registers and the window above it, taken
    /// once per binding and never given back.
    #[cold]
    fn widen(&mut self, slots: u16) {
        assert!(
            slots <= MAX_FRAME_SLOTS,
            "a body of {slots} registers was prepared past the {MAX_FRAME_SLOTS} one frame's \
             mark word reaches"
        );
        let need = cells_for(slots) + WINDOW_CELLS;
        if self.cells.len() < need {
            self.cells.resize_with(need, Cell::uninit);
        }
    }
}

impl Default for Store {
    fn default() -> Store {
        Store::new()
    }
}

/// One frame: the cells its registers and its mark word sit in, then the
/// cells a call out of it takes its callee's frame from.
///
/// `prepare::check_assignment` proves every register an operation names is
/// below its body's `frame_len`, and `prepare`'s liveness proves every
/// register an operation reads was defined on the path that reached it. Those
/// two proofs are what the `unsafe` here stands on; `debug_assert!` re-checks
/// the first in a debug build, and nothing can check the second at run time,
/// which is why the registers are `MaybeUninit` rather than a readable
/// sentinel.
pub struct Regs<'f> {
    cells: &'f mut [Cell],
    /// How many of `cells` are this frame's.
    own: u16,
    /// This frame's registers. Only the two bounds checks read it.
    len: u16,
    /// The cells left over for a callee's frame, capped at the widest frame
    /// `prepare` can emit. One compare per call reads it.
    above_cells: u16,
}

impl<'f> Regs<'f> {
    /// `cells` is the frame's own cells followed by its window. Making the
    /// frame is what writes its mark word, so no `Regs` exists whose mark
    /// word is unwritten: `body.param_marks` is the claim the entry hands it
    /// (RFC-0052 rule 7).
    ///
    /// # Panics
    /// `cells` is narrower than a frame of `body.frame_len` registers and the
    /// cell its calls lay their arguments in, which `Store::bind` and
    /// `fits_above` answer before this is reached.
    #[inline]
    fn of(cells: &'f mut [Cell], body: &Body) -> Regs<'f> {
        let len = body.frame_len;
        let own = cells_for(len);
        assert!(
            own + ARG_CELLS <= cells.len(),
            "a frame of {len} registers was borrowed from {} cells",
            cells.len()
        );
        let mut regs = Regs {
            own: own as u16,
            len,
            above_cells: (cells.len() - own).min(MAX_FRAME_CELLS) as u16,
            cells,
        };
        regs.mark(body.param_marks);
        regs
    }

    /// The one capacity compare a call makes: whether the cells above this
    /// frame hold a callee frame of `callee_len` registers and the cell that
    /// frame's own calls lay their arguments in.
    #[inline(always)]
    pub fn fits_above(&self, callee_len: u16) -> bool {
        cells_for(callee_len) + ARG_CELLS <= usize::from(self.above_cells)
    }

    /// The callee's frame: the cells above this one.
    ///
    /// # Panics
    /// Too few cells are left, which `fits_above` answers before this is
    /// called.
    #[inline(always)]
    pub fn window(&mut self, callee: &Body) -> Regs<'_> {
        Regs::of(&mut self.cells[usize::from(self.own)..], callee)
    }

    /// The register `at` of the frame the window above this one begins.
    #[inline(always)]
    fn above(&self, at: Off) -> *mut Value {
        debug_assert!(
            at.index() < CELL_SLOTS as usize * ARG_CELLS,
            "a call lays an argument in the callee's register {}, past the cell the window \
             begins with",
            at.index()
        );
        // SAFETY: `Regs::of` refuses a frame without the cell this writes, and
        // the debug assertion above holds the displacement inside it.
        unsafe {
            self.cells
                .as_ptr()
                .add(usize::from(self.own))
                .cast::<u8>()
                .add(at.byte())
                .cast::<Value>()
                .cast_mut()
        }
    }

    /// One argument of a call, written to the register the callee reads it
    /// from (RFC-0052 rule 7).
    #[inline(always)]
    pub fn lay(&mut self, at: Off, value: Value) {
        // SAFETY: as `above`. The callee's frame is unbound until the call, so
        // nothing owns what this overwrites.
        unsafe { self.above(at).write(value) };
    }

    /// The two registers of a slice argument (RFC-0047 amended, rule 4).
    #[inline(always)]
    pub fn lay_pair(&mut self, at: SlicePair, value: SlicePair) {
        let (ptr, len) = (self.read(value.ptr), self.read(value.len));
        self.lay(at.ptr, ptr);
        self.lay(at.len, len);
    }

    /// The argument run a caller laid, read where the callee's body is one
    /// chain and has no frame to read it from (RFC-0044, stage 4).
    #[inline]
    pub fn laid(&self, arity: u16) -> &[Value] {
        let first = self.above(Off::of(0));
        // SAFETY: `lay` wrote every register of the run, and `above` holds the
        // widest of them inside the cell the window begins with.
        unsafe { std::slice::from_raw_parts(first, usize::from(arity)) }
    }

    /// The register's first byte. No arithmetic: the operation's field is
    /// already the byte displacement (RFC-0052 §5).
    #[inline(always)]
    fn at(&self, off: Off) -> *const Value {
        debug_assert!(
            off.index() < usize::from(self.len),
            "an operation names register {}, which its body's frame does not have",
            off.index()
        );
        // SAFETY: the two proofs stated on `Regs`.
        unsafe {
            self.cells
                .as_ptr()
                .cast::<u8>()
                .add(off.byte())
                .cast::<Value>()
        }
    }

    #[inline(always)]
    fn at_mut(&mut self, off: Off) -> *mut Value {
        debug_assert!(
            off.index() < usize::from(self.len),
            "an operation names register {}, which its body's frame does not have",
            off.index()
        );
        // SAFETY: the two proofs stated on `Regs`.
        unsafe {
            self.cells
                .as_mut_ptr()
                .cast::<u8>()
                .add(off.byte())
                .cast::<Value>()
        }
    }

    /// The frame's mark word: the slot `cells_for` reserved just past its
    /// registers, written when the frame was made and never read before.
    #[inline(always)]
    fn mark_ptr(&self) -> *mut u64 {
        let byte = usize::from(self.len) * size_of::<Value>();
        // SAFETY: `cells_for(len)` reserved the slot at register index `len`,
        // and `Regs::of` wrote it before handing the frame out.
        unsafe {
            self.cells
                .as_ptr()
                .cast::<u8>()
                .add(byte)
                .cast::<u64>()
                .cast_mut()
        }
    }

    #[inline(always)]
    fn marked(&self) -> u64 {
        // SAFETY: as `mark_ptr`.
        unsafe { *self.mark_ptr() }
    }

    #[inline(always)]
    fn mark(&mut self, bits: u64) {
        // SAFETY: as `mark_ptr`.
        unsafe { *self.mark_ptr() = bits }
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

    /// The word of a word-typed register, with the frame's claim untouched:
    /// a register whose kind the frame opened holds no `Large`, so there is
    /// no mark bit to clear (RFC-0052 §5).
    #[inline(always)]
    pub fn take_word(&mut self, off: Off) -> u64 {
        debug_assert!(
            self.marked() & off.mark() == 0,
            "a word-typed register carries the frame's claim on a Large"
        );
        self.word(off)
    }

    /// One store, and one `or` on the frame's mark word where the operation's
    /// type says the value owns a `Large` (RFC-0048 §4).
    #[inline(always)]
    pub fn define<const LARGE: bool>(&mut self, off: Off, value: Value) {
        // SAFETY: `check_assignment`, as stated on `Regs`. A register this
        // overwrites was released by a drop instruction or never owned
        // (RFC-0041, RFC-0048 §6), so no value is lost here.
        unsafe { self.at_mut(off).write(value) };
        if LARGE {
            self.mark(self.marked() | off.mark());
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
            self.mark(self.marked() & !off.mark());
        }
        value
    }

    /// The batched form: one `and` of the constant mask of the registers this
    /// operation consumes (RFC-0048 §5).
    #[inline(always)]
    pub fn take_mask(&mut self, mask: u64) {
        let marked = self.marked();
        debug_assert!(
            marked & mask == mask,
            "an operation takes a register its frame does not own: a double take"
        );
        self.mark(marked & !mask);
    }

    /// RFC-0045: the old value is released before the new one lands.
    pub fn assign<const LARGE: bool>(&mut self, off: Off, value: Value) {
        let one = off.mark();
        let marked = self.marked();
        if marked & one != 0 {
            self.read(off).release();
        }
        // SAFETY: `check_assignment`, as stated on `Regs`, with the previous
        // owner released just above.
        unsafe { self.at_mut(off).write(value) };
        match LARGE {
            true => self.mark(marked | one),
            false => self.mark(marked & !one),
        }
    }

    /// Leaving: the set bits of the frame's mark word, released, and the word
    /// cleared. It iterates the set bits, never the registers — the 16-slot
    /// kind scan at every return is what RFC-0048 §6 removed.
    pub fn sweep(&mut self) {
        let mut live = self.marked();
        while live != 0 {
            let bit = live.trailing_zeros();
            debug_assert!(
                bit < u32::from(MAX_FRAME_SLOTS),
                "a set mark bit is a register index, which the mark word bounds"
            );
            self.read(Off::of(bit as u16)).release();
            live &= live - 1;
        }
        self.mark(0);
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
            to <= usize::from(self.len),
            "an argument run of {arity} at register {from} leaves a frame of {} registers",
            self.len
        );
        // SAFETY: `prepare` allocated the run contiguously in this frame and
        // every register of it is defined at the call.
        unsafe {
            std::slice::from_raw_parts(
                self.cells.as_ptr().cast::<Value>().add(from),
                usize::from(arity),
            )
        }
    }

    /// The frame's first register, which a chain's pre-multiplied leaf offsets
    /// are byte displacements from.
    #[inline]
    pub fn as_ptr(&self) -> *const Value {
        self.cells.as_ptr().cast::<Value>()
    }

    #[inline]
    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
