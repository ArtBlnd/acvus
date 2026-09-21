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
use std::ptr::NonNull;

use acvus_extern::Release;

use crate::code::{Body, Marked, Off};
use crate::value::Value;

/// The registers one cell holds: four cache lines of `Value`s.
pub const CELL_SLOTS: u16 = 16;

pub const MAX_FRAME_SLOTS: u16 = MAX_SCALAR_SLOTS + MAX_RUN_SLOTS;

/// `prepare::assign_slots` is what keeps this bound: it colours a scalar value
/// into the registers below it and nothing here can check that it did.
pub const MAX_SCALAR_SLOTS: u16 = 64;

pub const MAX_RUN_SLOTS: u16 = 256;

pub(crate) const MARK_WORD_SLOTS: u16 = 64;

const MARK_WORDS_PER_SLOT: u16 = 2;

/// `Regs::of` writes `Body::param_marks` into mark word 0 whatever a frame's
/// length is, so even a frame of no registers carries that one word.
const MARK_WORDS_AT_LEAST: u16 = 1;

const _: () = assert!(
    MARK_WORD_SLOTS as u32 == u64::BITS,
    "a mark word carries one bit per register"
);
const _: () = assert!(
    size_of::<Value>() == MARK_WORDS_PER_SLOT as usize * size_of::<u64>(),
    "a register slot holds a whole number of mark words"
);

#[inline(always)]
pub(crate) const fn mark_words(slots: u16) -> u16 {
    MARK_WORDS_AT_LEAST + slots.saturating_sub(1) / MARK_WORD_SLOTS
}

#[inline(always)]
const fn mark_slots(slots: u16) -> u16 {
    mark_words(slots).div_ceil(MARK_WORDS_PER_SLOT)
}

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

/// The arguments one call can lay in the window it opens: the cell a bound
/// frame keeps above itself, in registers.
pub const MAX_ARG_SLOTS: usize = CELL_SLOTS as usize * ARG_CELLS;

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

/// The cells a frame of `slots` registers occupies, its mark words included.
#[inline(always)]
pub(crate) const fn cells_for(slots: u16) -> usize {
    (slots as usize + mark_slots(slots) as usize).div_ceil(CELL_SLOTS as usize)
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
    root: FrameState,
}

impl Store {
    /// A frame bound to no body: no cells, no allocation.
    pub fn new() -> Store {
        Store {
            cells: Vec::new(),
            root: FrameState::UNBOUND,
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
        let same = self.root.bound.rebind(body);
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
            "a body of {slots} registers was prepared past the {MAX_FRAME_SLOTS} one frame holds"
        );
        let need = cells_for(slots) + WINDOW_CELLS;
        if self.cells.len() < need {
            self.cells.resize_with(need, Cell::uninit);
        }
    }
}

/// The cells of a `Store` whose root state has left it, with no operation on
/// them: it keeps the cells alive and offers no `bind` and no `widen`, so the
/// `FrameState` `new` hands back names cells that cannot move under it.
pub struct RootCells(Store);

/// A frame at the root of a call chain of its own: the state a call runs in,
/// and the cells that state names.
pub struct RootFrame {
    pub state: FrameState,
    pub cells: RootCells,
}

impl RootFrame {
    pub fn new() -> RootFrame {
        let mut store = Store::new();
        store.widen(MAX_FRAME_SLOTS);
        let mut state = std::mem::replace(&mut store.root, FrameState::UNBOUND);
        state.cells = NonNull::from(&mut store.cells[..]);
        RootFrame {
            state,
            cells: RootCells(store),
        }
    }
}

/// The cells a call takes its callee's frame from, and the body they are bound
/// to (RFC-0050 rule 6): the window above a running frame, or a `Store`'s own
/// cells at the root of a chain. A call lays its arguments in the registers the
/// callee reads them from and then `bind`s the callee there (RFC-0052 §7).
///
/// The frame below owns the state — inside the `Ctx` its `Machine` holds, or
/// inside the `Ctx` a `RootCells` pair was built with — and lends a handler
/// `&mut Ctx`, which is an address that already exists. Nothing is built on
/// the calling operation's stack, and `asm_probe` is where that shows: a `run`
/// whose local's address escapes into a callee loses its sibling call.
///
/// The cells are a raw slice, not a reference, because `Runtime::Frame<'a>`
/// has one lifetime parameter and `&mut` is invariant: a state that named its
/// cells by reference would carry that reference's lifetime as a second
/// parameter, and `&'a mut FrameState<'c>` does not shrink to
/// `&'a mut FrameState<'a>`. `Regs::take_window` is the only constructor of a
/// state over a running frame's cells, and it takes those cells out of the
/// `Regs` it names them in, so no two handles reach them.
pub struct FrameState {
    cells: NonNull<[Cell]>,
    bound: BoundTo,
}

// SAFETY: the state names cells a `&mut [Cell]` named before it, and `Cell` is
// `Send` and `Sync` — the assertion below is what keeps that true. Reaching
// the cells needs `&FrameState` or `&mut FrameState`, so a shared state hands
// out no more than a `&[Cell]` would.
unsafe impl Send for FrameState {}
// SAFETY: as `Send`.
unsafe impl Sync for FrameState {}

const _: fn() = || {
    fn cells_cross_threads<T>()
    where
        T: Send + Sync,
    {
    }
    cells_cross_threads::<Cell>();
};

impl FrameState {
    /// No cells and no body: what a `Store` holds until it is widened.
    pub const UNBOUND: FrameState = FrameState {
        cells: NonNull::slice_from_raw_parts(NonNull::dangling(), 0),
        bound: BoundTo::NONE,
    };

    /// # Safety
    /// `cells` are live and unmoved, and named by no other handle, for as long
    /// as this state is.
    #[inline(always)]
    unsafe fn of(cells: NonNull<[Cell]>) -> FrameState {
        FrameState {
            cells,
            bound: BoundTo::NONE,
        }
    }

    #[inline(always)]
    fn cells(&mut self) -> &mut [Cell] {
        // SAFETY: the state's own invariant: the cells are live, unmoved and
        // named by nothing else, and this borrows them for no longer than the
        // state.
        unsafe { self.cells.as_mut() }
    }

    #[inline(always)]
    pub fn fits(&self, callee: &Body) -> bool {
        usize::from(callee.frame_cells) + ARG_CELLS <= self.cells.len()
    }

    /// The frame `body` runs in, and whether the window already carries
    /// `body`'s slot kinds and entry constants (`machine::open_frame`): one
    /// `open_frame` per window and body, however many calls follow.
    ///
    /// # Panics
    /// The window is narrower than `body`, which `fits` answers before this
    /// is reached.
    #[inline]
    pub fn bind(&mut self, body: &Body) -> (Regs<'_>, bool) {
        let same = self.bound.rebind(body);
        (Regs::of(self.cells(), body), same)
    }

    /// The register `at` of the frame this window begins.
    #[inline(always)]
    fn at(&self, at: Off) -> *mut Value {
        debug_assert!(
            at.index() < CELL_SLOTS as usize * ARG_CELLS,
            "a call lays an argument in the callee's register {}, past the cell the window \
             begins with",
            at.index()
        );
        // SAFETY: `Store::widen` and `FrameState::fits` size every window with
        // the cell this writes before a frame is bound in it,
        // `Store::root_window` holds the widest frame there is, and the debug
        // assertion above holds the displacement inside that cell.
        unsafe {
            self.cells
                .as_ptr()
                .cast::<u8>()
                .add(at.byte())
                .cast::<Value>()
        }
    }

    /// One argument of a call, written to the register the callee reads it
    /// from (RFC-0052 rule 7).
    #[inline(always)]
    pub fn lay(&mut self, at: Off, value: Value) {
        // SAFETY: as `at`. The callee's frame is unbound until the call, so
        // nothing owns what this overwrites.
        unsafe { self.at(at).write(value) };
    }

    /// The argument run a caller laid, read where the callee's body is one
    /// chain and has no frame to read it from (RFC-0044, stage 4).
    #[inline]
    pub fn laid(&self, arity: u16) -> &[Value] {
        let first = self.at(Off::of(0));
        // SAFETY: `lay` wrote every register of the run, and `at` holds the
        // widest of them inside the cell the window begins with.
        unsafe { std::slice::from_raw_parts(first, usize::from(arity)) }
    }

    /// The callee's first `width` parameter registers, for a crossing that
    /// writes a call's arguments into them at their own widths (RFC-0059).
    ///
    /// `AcvusRuntime::call_now` proves the bound below at its own
    /// monomorphization, out of `IntoRun::WIDTH` (RFC-0059).
    #[inline]
    pub fn run_mut(&mut self, width: usize) -> &mut [Value] {
        debug_assert!(
            width <= MAX_ARG_SLOTS,
            "a call lays {width} arguments, past the {MAX_ARG_SLOTS} the cell a window begins \
             with holds"
        );
        let run = &mut self.cells()[0].slots[..width];
        for slot in run.iter_mut() {
            slot.write(Value::UNDEF);
        }
        // SAFETY: every slot of the run was written just above, and
        // `MaybeUninit<Value>` and `Value` are the same layout.
        unsafe { &mut *(run as *mut [MaybeUninit<Value>] as *mut [Value]) }
    }
}

impl Default for Store {
    fn default() -> Store {
        Store::new()
    }
}

/// # Safety
/// `cells` are one frame's own cells, `len` its registers, and every register
/// of the run is defined at the call.
#[inline]
unsafe fn run_in(cells: &[Cell], at: Off, arity: u16, len: u16) -> &[Value] {
    let from = at.index();
    let to = from + usize::from(arity);
    debug_assert!(
        to <= usize::from(len),
        "an argument run of {arity} at register {from} leaves a frame of {len} registers"
    );
    // SAFETY: the caller's contract.
    unsafe {
        std::slice::from_raw_parts(cells.as_ptr().cast::<Value>().add(from), usize::from(arity))
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
    /// `prepare` can emit.
    above_cells: u16,
}

impl<'f> Regs<'f> {
    /// `cells` is the frame's own cells followed by its window. Making the
    /// frame is what writes its mark word, so no `Regs` exists whose mark
    /// word is unwritten: `body.param_marks` is the claim the entry hands it
    /// (RFC-0052 rule 7).
    #[inline]
    fn of(cells: &'f mut [Cell], body: &Body) -> Regs<'f> {
        let len = body.frame_len;
        let own = usize::from(body.frame_cells);
        debug_assert_eq!(
            own,
            cells_for(len),
            "a body of {len} registers was prepared with {own} cells"
        );
        debug_assert!(
            own + ARG_CELLS <= cells.len(),
            "a frame of {len} registers was borrowed from {} cells, which `Store::widen` and \
             `FrameState::fits` answer before this is reached",
            cells.len()
        );
        let mut regs = Regs {
            own: own as u16,
            len,
            above_cells: (cells.len() - own).min(MAX_FRAME_CELLS) as u16,
            cells,
        };
        regs.mark(0, body.param_marks);
        regs
    }

    /// The cells above this frame, taken out of it as the handle its calls lend
    /// a handler. Taking is what keeps the two apart: afterwards this `Regs`
    /// reaches its registers and nothing above them, and a second take finds
    /// no cells left.
    #[inline(always)]
    pub fn take_window(&mut self) -> FrameState {
        let own = usize::from(self.own);
        let end = own + usize::from(self.above_cells);
        let cells = std::mem::take(&mut self.cells);
        let (mine, above) = cells[..end].split_at_mut(own);
        self.cells = mine;
        self.above_cells = 0;
        // SAFETY: `above` is a live `&mut [Cell]` that this `Regs` no longer
        // reaches and no other handle names, borrowed from the same cells the
        // frame below lent, which outlive the frame.
        unsafe { FrameState::of(NonNull::from(above)) }
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

    /// `word_byte` is `Marked::word_byte` of the register being marked, which
    /// `prepare` decided and moved into the operation.
    #[inline(always)]
    fn mark_ptr(&self, word_byte: usize) -> *mut u64 {
        debug_assert!(
            word_byte < usize::from(mark_words(self.len)) * size_of::<u64>(),
            "an operation marks the mark word at byte {word_byte}, which its body's frame does \
             not have"
        );
        let byte = usize::from(self.len) * size_of::<Value>() + word_byte;
        // SAFETY: `cells_for(len)` reserved `mark_slots(len)` slots at register
        // index `len`, and the assertion above holds `word_byte` inside them.
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
    fn marked(&self, word_byte: usize) -> u64 {
        // SAFETY: as `mark_ptr`.
        unsafe { *self.mark_ptr(word_byte) }
    }

    #[inline(always)]
    fn mark(&mut self, word_byte: usize, bits: u64) {
        // SAFETY: as `mark_ptr`.
        unsafe { *self.mark_ptr(word_byte) = bits }
    }

    /// The mark words above word 0, cleared. `Regs::of` does not clear them: it
    /// runs on every bind, and a frame of one mark word — every frame in the
    /// bench set — would pay for words it does not have on the one path a
    /// closure-heavy body takes per element. This runs from
    /// `machine::open_frame` instead, once per window and body, which is also
    /// where the kind bytes are written.
    pub fn open_marks(&mut self, mark_words: u16) {
        for word in 1..usize::from(mark_words) {
            self.mark(word * size_of::<u64>(), 0);
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

    #[inline(always)]
    pub fn projection(&mut self, off: Off) -> Value {
        Value::large_ref(self.at_mut(off))
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
    pub fn take_word(&mut self, at: Marked) -> u64 {
        debug_assert!(
            self.marked(at.word_byte()) & at.mask() == 0,
            "a word-typed register carries the frame's claim on a Large"
        );
        self.word(at.at)
    }

    /// One store, and one `or` on the frame's mark word where the operation's
    /// type says the value owns a `Large` (RFC-0048 §4).
    #[inline(always)]
    pub fn define<const LARGE: bool>(&mut self, at: Marked, value: Value) {
        // SAFETY: `check_assignment`, as stated on `Regs`. A register this
        // overwrites was released by a drop instruction or never owned
        // (RFC-0041, RFC-0048 §6), so no value is lost here.
        unsafe { self.at_mut(at.at).write(value) };
        if LARGE {
            let word = at.word_byte();
            self.mark(word, self.marked(word) | at.mask());
        }
    }

    /// A whole `Value` in a register the frame claims nothing in: the
    /// operation's type says the value owns no `Large`, so there is no mark
    /// bit to set and no mark word to name (RFC-0048 §4).
    #[inline(always)]
    pub fn put(&mut self, at: Off, value: Value) {
        // SAFETY: as `define`.
        unsafe { self.at_mut(at).write(value) };
    }

    /// The store a call's result takes (RFC-0052 §5).
    #[inline(always)]
    pub fn store<const LARGE: bool, const WORD: bool>(&mut self, at: Marked, value: Value) {
        const {
            assert!(
                !(LARGE && WORD),
                "a register whose kind the frame opened holds no Large"
            )
        }
        match WORD {
            true => self.set_word(at.at, value.bits()),
            false => self.define::<LARGE>(at, value),
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
    pub fn take<const LARGE: bool>(&mut self, at: Marked) -> Value {
        let value = self.read(at.at);
        if LARGE {
            let word = at.word_byte();
            self.mark(word, self.marked(word) & !at.mask());
        }
        value
    }

    /// The batched form: one `and` of the constant mask of the registers this
    /// operation consumes (RFC-0048 §5).
    ///
    /// `prepare::take_mask` and `prepare::window_take_mask` are what keep every
    /// register of a mask inside mark word 0; each asserts it where it builds
    /// one, and no structure here shows that they did.
    #[inline(always)]
    pub fn take_mask(&mut self, mask: u64) {
        let marked = self.marked(0);
        debug_assert!(
            marked & mask == mask,
            "an operation takes a register its frame does not own: a double take"
        );
        self.mark(0, marked & !mask);
    }

    /// RFC-0045: the old value is released before the new one lands.
    pub fn assign<const LARGE: bool>(&mut self, at: Marked, value: Value) {
        let one = at.mask();
        let word = at.word_byte();
        let marked = self.marked(word);
        if marked & one != 0 {
            self.read(at.at).release();
        }
        // SAFETY: `check_assignment`, as stated on `Regs`, with the previous
        // owner released just above.
        unsafe { self.at_mut(at.at).write(value) };
        match LARGE {
            true => self.mark(word, marked | one),
            false => self.mark(word, marked & !one),
        }
    }

    /// Leaving: the set bits of the frame's mark words, released, and the words
    /// cleared. It iterates the set bits, never the registers — the 16-slot
    /// kind scan at every return is what RFC-0048 §6 removed.
    /// The runtime word count costs `map cap | sum` 2.7 % against a build that
    /// sweeps word 0 alone, which is what this cost before frames grew past one
    /// mark word. Lifting word 0 out into an `#[inline(always)]` helper, so its
    /// displacement and bit base fold, was measured at **+10.8 %** instead:
    /// `sweep` inlines into every return site, and a second copy of the release
    /// path costs more than the count does.
    pub fn sweep(&mut self, mark_words: u16) {
        let mut word = 0usize;
        loop {
            let mut live = self.marked(word * size_of::<u64>());
            while live != 0 {
                let bit = live.trailing_zeros() as u16 + word as u16 * MARK_WORD_SLOTS;
                debug_assert!(
                    bit < self.len,
                    "a set mark bit is a register index, which its frame holds"
                );
                self.read(Off::of(bit)).release();
                live &= live - 1;
            }
            self.mark(word * size_of::<u64>(), 0);
            word += 1;
            if word >= usize::from(mark_words) {
                return;
            }
        }
    }

    /// The registers an extern call's arguments sit in, lent to the handler
    /// (RFC-0044, stage 2b).
    #[inline]
    pub fn run_of(&self, at: Off, arity: u16) -> &[Value] {
        // SAFETY: `prepare` allocated the run contiguously in this frame and
        // every register of it is defined at the call.
        unsafe { run_in(self.cells, at, arity, self.len) }
    }

    /// The registers an aggregate-returning call writes its components into,
    /// lent to the handler (RFC-0050 rules 5 and 6). The caller cleared every
    /// register of the run that held a `Large`, so a write here loses nothing.
    #[inline]
    pub fn run_of_mut(&mut self, at: Off, width: u16) -> &mut [Value] {
        let from = at.index();
        let to = from + usize::from(width);
        debug_assert!(
            to <= usize::from(self.len),
            "a destination run of {width} at register {from} leaves a frame of {} registers",
            self.len
        );
        // SAFETY: `prepare` placed the run contiguously in this frame, which
        // `prepare::check_assignment` proves against `Body::frame_len`.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.cells.as_mut_ptr().cast::<Value>().add(from),
                usize::from(width),
            )
        }
    }

    /// The frame's claim on a register whose value the frame did not write
    /// itself: an aggregate-returning call's handler wrote the `Large` into
    /// the lent run, and the frame takes ownership of it here (RFC-0048 §4).
    #[inline]
    pub fn claim(&mut self, at: Marked) {
        let word = at.word_byte();
        self.mark(word, self.marked(word) | at.mask());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_carries_one_mark_word_per_sixty_four_registers() {
        assert_eq!(mark_words(0), 1);
        assert_eq!(mark_words(1), 1);
        assert_eq!(mark_words(64), 1);
        assert_eq!(mark_words(65), 2);
        assert_eq!(mark_words(MAX_FRAME_SLOTS), 5);
    }

    /// A frame's own cells are the ones `take_window` withholds from the window,
    /// so a callee cannot reach a mark word of the frame below it.
    #[test]
    fn the_mark_words_lie_inside_the_cells_a_frame_keeps() {
        for len in 0..=MAX_FRAME_SLOTS {
            let last = usize::from(len) * size_of::<Value>()
                + usize::from(mark_words(len)) * size_of::<u64>();
            assert!(
                last <= cells_for(len) * size_of::<Cell>(),
                "a frame of {len} registers keeps {} cells, which its last mark word leaves",
                cells_for(len)
            );
        }
    }

    /// A frame of at most 64 registers occupies the cells it occupied before the
    /// runs: one mark slot, and `cells_for` the expression it was.
    #[test]
    fn a_scalar_frame_occupies_the_cells_it_did() {
        for len in 0..=MAX_SCALAR_SLOTS {
            assert_eq!(
                cells_for(len),
                (usize::from(len) + 1).div_ceil(CELL_SLOTS as usize),
                "a frame of {len} registers"
            );
        }
    }
}
