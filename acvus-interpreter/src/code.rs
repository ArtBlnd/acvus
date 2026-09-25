//! The prepared form of a body (RFC-0052).
//!
//! `prepare` and the `ops` module are two halves of one contract: every field
//! of an operation is a fact `prepare` decided and moved in, so a decision
//! that reaches a `run` as anything but a field or a type parameter is a
//! defect in `prepare`, not in the operation.

use std::sync::Arc;

use acvus_ast::Span;
use acvus_extern::{FieldAt, Owned, Words, repr};
use acvus_mir::ir::Label;
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;

use crate::machine::Machine;
use crate::regs::{FrameSlot, FrameState, MarkWords};
use crate::runtime::AcvusRuntime;
use crate::value::{Kind, Value};

/// `prepare::check_assignment` proves every slot an operation names is below
/// its body's `frame_len`, which is what makes a slot access unchecked in
/// release.
///
/// This is the language-level register index, and it lives in `prepare`: the
/// register assignment, the liveness and the parallel-move ordering are
/// written in it. An operation never holds one — it holds an `Off`.
pub type Slot = u16;

/// A register's place in its frame: `repr::Disp`, the byte displacement of
/// a slot of the frame's run of `Value`s (RFC-0080 rule 5).
///
/// `prepare` multiplies once, when it builds the operation, so no `run`
/// scales: `Add::<i64>::run`'s three `shl $4` are what this type removes
/// (RFC-0052 rule 5). `Slot` and `Off` are different types so that the index
/// and the displacement cannot be handed to each other's reader.
pub type Off = acvus_extern::repr::Disp<Value>;

/// A register and the frame's claim on the `Large` it may own: the mark word's
/// displacement inside the mark region and the register's bit in it, both
/// decided at `prepare` (RFC-0050 rule 2).
///
/// The mask is a field rather than a shift of `at` because the fourth build of
/// rule 2 derived the word and the bit from the displacement inside
/// `define::<true>`, `take::<true>` and `assign`. Its disassembly is the
/// measurement: `control::DropValue::run` went 31 → 39 instructions and
/// `storage::AssignVar<true>::run` 58 → 78, because a `{u16, u8, u8}` is one
/// dword load and three extracting shifts. Only the operations that mark carry
/// this, so an operation that writes a word — the whole `ops::chain` and
/// `ops::arith` half of the machine — still names its register with a bare
/// two-byte `Off`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Marked {
    mask: u64,
    word_byte: u32,
    at: repr::DispBelow<Value, { crate::regs::MAX_FRAME_SLOTS }>,
}

impl Marked {
    pub const fn of(slot: FrameSlot) -> Marked {
        let index = slot.get() as usize;
        Marked {
            mask: 1u64 << (index % crate::regs::MARK_WORD_SLOTS as usize),
            word_byte: (index / crate::regs::MARK_WORD_SLOTS as usize * size_of::<u64>()) as u32,
            at: repr::DispBelow::of(slot),
        }
    }

    #[inline(always)]
    pub const fn at(self) -> Off {
        self.at.disp()
    }

    /// The slice pair whose `ptr` is this register.
    #[inline(always)]
    pub const fn pair(self) -> SlicePair {
        SlicePair {
            ptr: self.at.disp(),
            len: self.at.after(),
        }
    }

    /// The displacement `Regs` adds to the first byte of its frame's mark
    /// region.
    #[inline(always)]
    pub const fn word_byte(self) -> usize {
        self.word_byte as usize
    }

    #[inline(always)]
    pub const fn mask(self) -> u64 {
        self.mask
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WordMask {
    pub word_byte: u32,
    pub mask: u64,
}

const _: () = assert!(
    u16::MAX as usize / size_of::<Value>() / crate::regs::MARK_WORD_SLOTS as usize
        * size_of::<u64>()
        <= u32::MAX as usize,
    "the mark word of the widest register an Off names lies within the displacement a Marked \
     carries"
);

const _: () = assert!(
    size_of::<Off>() == 2 && size_of::<Marked>() == 16 && align_of::<Marked>() == 8,
    "an operation that writes a word names its register in two bytes, and one that marks reads \
     its mask and its word with two aligned loads"
);

/// The two registers a slice occupies: `ptr` then `len`, adjacent
/// (RFC-0047 rule 6). An operation that holds one adds nothing to reach
/// `len`; one whose destination is a `Marked` derives it by `Marked::pair`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlicePair {
    pub ptr: Off,
    pub len: Off,
}

impl SlicePair {
    pub const fn at(ptr: FrameSlot) -> SlicePair {
        SlicePair {
            ptr: Off::of_below(ptr),
            len: Off::after(ptr),
        }
    }
}

/// Where `prepare` decided one word lives (RFC-0052 rule 5). It is what the
/// place *type* in `ops::place` is chosen from, at preparation; no `run`
/// holds one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Where {
    /// The frame register at this byte displacement.
    Frame(Off),
    /// The argument register, which the operation before this one in the same
    /// chain left the word in.
    Register,
}

pub type BlockId = u32;

/// The word a chain hands back when it ends (RFC-0052 rule 3).
///
/// A chain that ends at a joint hands back the block the machine enters
/// next; a region's part ends in `ops::control::Yield` and hands the region
/// that owns it the word its last operation computed. The two never meet:
/// `Machine::run` reads only the chains of `Body::heads`, and a region reads
/// only the chains it owns.
pub type Exit = u64;

/// The body returns what it left in `Machine::exit`.
pub const RETURN: Exit = BlockId::MAX as Exit;

/// The body left a future in `Machine::pending` for the driver.
pub const SUSPEND: Exit = RETURN - 1;

/// A `break`: the loop stops iterating and runs its successor.
pub const LEAVE: Exit = RETURN - 2;

/// A `continue`: the loop steps and tests again.
pub const AGAIN: Exit = RETURN - 3;

/// The chain ran to its end.
pub const FALL: Exit = RETURN - 4;

/// `Machine::run` is one compare against this per block.
pub const SENTINEL: Exit = FALL;

const _: () = assert!(
    RETURN >= SENTINEL && SUSPEND >= SENTINEL && LEAVE >= SENTINEL && AGAIN >= SENTINEL,
    "every word that is not a block leaves the machine's loop at its one compare"
);

/// Not on the release trait: no instance carries a name string outside
/// `oplist`, `asm_probe` and the loop-shape tests.
#[cfg(any(debug_assertions, feature = "probe"))]
pub trait Named {
    fn name(&self) -> &'static str;
}

#[cfg(any(debug_assertions, feature = "probe"))]
impl<T> Named for T
where
    T: ?Sized,
{
    fn name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

/// An operation holds its successor and ends by calling it; the operation
/// with no successor returns the `BlockId` the machine's loop reads.
///
/// Obligation across artifacts: that the call is a tail call is asserted by
/// `acvus-interpreter-test/benches/asm_probe.rs` on the release machine, and
/// that a `run` holds no `match`, no `let … else` and no `kind` test on a
/// fact its own type carries is RFC-0052 rule 1.
///
/// `r0` is the word the operation before it in the same chain produced
/// (RFC-0052 rule 5, `Place::R0`).
#[cfg(any(debug_assertions, feature = "probe"))]
pub trait Op: Named + Send + Sync {
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit;

    fn chain(&self) -> Option<ChainProbe<'_>> {
        None
    }

    /// The registers a **bound-checked** indexed read names, so that a probe
    /// can put the unchecked form of the same read in its place (RFC-0047
    /// rule 7). Nothing else answers it, and the unchecked form answers `None`,
    /// so a substitution cannot run twice.
    fn index_read(&self) -> Option<crate::ops::index::Read> {
        None
    }

    /// Answered by a region alone, for `oplist` and the loop-shape tests, so
    /// that they read the parts the recognizers produced rather than infer
    /// them. Each part is the head of one chain.
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        Vec::new()
    }

    /// The same heads, for `slice_ceiling`'s probe to substitute an operation
    /// inside a region.
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        Vec::new()
    }

    /// The successor, for a listing that walks a chain.
    fn successor(&self) -> Option<&dyn Op> {
        None
    }

    /// The successor, for a probe that walks a chain to a node of it.
    fn successor_mut(&mut self) -> Option<&mut Box<dyn Op>> {
        None
    }

    /// The successor, taken out: `substitute` is the one caller.
    fn take_successor(self: Box<Self>) -> Option<Box<dyn Op>> {
        None
    }
}

/// Put `make`'s node in `slot`'s place, carrying the successor the node
/// there held (RFC-0047 rule 7's probe, now that a successor is a field).
#[cfg(all(feature = "tooling", any(debug_assertions, feature = "probe")))]
pub fn substitute<F>(slot: &mut Box<dyn Op>, make: F)
where
    F: FnOnce(Box<dyn Op>) -> Box<dyn Op>,
{
    // SAFETY: `read` copies the one owning pointer out of `slot`;
    // `take_successor` consumes that copy, which is the old node's only
    // drop; `write` then stores the new node without dropping the copy. On
    // every path `slot` owns exactly one node.
    unsafe {
        let old = std::ptr::read(slot);
        let successor = old
            .take_successor()
            .expect("the node a probe substitutes holds a successor");
        std::ptr::write(slot, make(successor));
    }
}

#[cfg(any(debug_assertions, feature = "probe"))]
pub struct ChainProbe<'o> {
    pub dst: Where,
    pub plan: &'o crate::ops::chain::Plan,
}

#[cfg(any(debug_assertions, feature = "probe"))]
pub struct OwnedOps<'o> {
    pub part: &'static str,
    pub head: &'o dyn Op,
}

#[cfg(not(any(debug_assertions, feature = "probe")))]
pub trait Op: Send + Sync {
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit;
}

/// The one writer of the three probe methods, over the `next` field the tail
/// call reads.
macro_rules! successor {
    () => {
        #[cfg(any(debug_assertions, feature = "probe"))]
        fn successor(&self) -> Option<&dyn $crate::code::Op> {
            Some(self.next.as_ref())
        }

        #[cfg(any(debug_assertions, feature = "probe"))]
        fn successor_mut(&mut self) -> Option<&mut Box<dyn $crate::code::Op>> {
            Some(&mut self.next)
        }

        #[cfg(any(debug_assertions, feature = "probe"))]
        fn take_successor(self: Box<Self>) -> Option<Box<dyn $crate::code::Op>> {
            Some(self.next)
        }
    };
}

pub(crate) use successor;

/// An operation `prepare` has decided on and not yet linked: `prepare` emits
/// a body's operations in the order they run, and `chain` links them from the
/// end backwards.
pub(crate) struct Node(Box<dyn FnOnce(Box<dyn Op>) -> Box<dyn Op>>);

impl Node {
    #[inline]
    pub(crate) fn link(self, next: Box<dyn Op>) -> Box<dyn Op> {
        (self.0)(next)
    }
}

pub(crate) fn node<T, F>(make: F) -> Node
where
    F: FnOnce(Box<dyn Op>) -> T + 'static,
    T: Op + 'static,
{
    Node(Box::new(move |next| Box::new(make(next)) as Box<dyn Op>))
}

pub(crate) fn made<F>(make: F) -> Node
where
    F: FnOnce(Box<dyn Op>) -> Box<dyn Op> + 'static,
{
    Node(Box::new(make))
}

/// The head of the chain `nodes` make, ending at `end`.
pub(crate) fn chain<I>(nodes: I, end: Box<dyn Op>) -> Box<dyn Op>
where
    I: IntoIterator<Item = Node>,
    I::IntoIter: DoubleEndedIterator,
{
    nodes
        .into_iter()
        .rev()
        .fold(end, |next, node| node.link(next))
}

/// `owns_large` is the `Suspend` terminator's own type parameter, carried as a
/// field because the driver that stores the result is one `async fn` for every
/// suspension in the program. It costs one branch per suspension, and RFC-0052
/// rule 4 keeps every suspension out of a fused body.
pub enum Pending {
    Word {
        dst: Marked,
        owns_large: bool,
        fut: BoxFuture<'static, Value>,
    },
    Pair {
        dst: SlicePair,
        fut: BoxFuture<'static, Words>,
    },
    /// A wait that defines no register: a commit under waited access.
    Unit {
        fut: BoxFuture<'static, ()>,
    },
}

pub enum Konst {
    Word(Kind, u64),
    Str(String),
    List(Box<[Konst]>),
}

impl Konst {
    pub fn value(&self) -> Value {
        match self {
            Konst::Word(kind, bits) => Value::inline(*kind, *bits),
            Konst::Str(s) => Value::string(s.as_str()),
            Konst::List(items) => Value::array_with(|| {
                items
                    .iter()
                    // SAFETY: `value` makes a fresh word, which no other
                    // holder owns.
                    .map(|item| unsafe { Owned::from_value(acvus_extern::Holding::new(), item.value()) })
                    .collect()
            }),
        }
    }
}

/// Every distinct string literal of one module, copied once when it was
/// prepared: a `&str` constant is the pointer and length of one of these
/// runs (RFC-0062 rule 2), and the `Body` holding that constant holds an
/// `Arc` of this table, so the bytes outlive every operation naming them
/// whatever the module they were prepared from does.
pub struct Literals {
    runs: FxHashMap<Box<str>, Words>,
}

impl Literals {
    pub fn of<'a>(texts: impl Iterator<Item = &'a str>) -> Literals {
        let mut runs: FxHashMap<Box<str>, Words> = FxHashMap::default();
        for text in texts {
            if runs.contains_key(text) {
                continue;
            }
            let owned: Box<str> = Box::from(text);
            let words = Words::of_str(&owned);
            runs.insert(owned, words);
        }
        Literals { runs }
    }

    /// # Panics
    /// `text` is not one of the texts this table was built from, which
    /// `prepare_module` builds from the same bodies it then prepares.
    pub fn run(&self, text: &str) -> Words {
        *self
            .runs
            .get(text)
            .unwrap_or_else(|| panic!("literals: no run for {text:?}"))
    }
}

/// One part of a template's output, as the preparation read it from the
/// part's type (RFC-0062 rule 3).
#[derive(Clone, Copy)]
pub enum ConcatPart {
    /// A `String` in this register, moved into the output.
    Owned(Off),
    /// Text the operation only reads.
    Lent(LentText),
}

/// Text an operation reads without taking it: the two representations
/// RFC-0062 rule 3 admits, as the preparation read the operand's type.
#[derive(Clone, Copy)]
pub enum LentText {
    /// A `String` the register holds itself.
    Own(Off),
    /// A `&String`, read through the reference.
    Through(Off),
    /// A `&str`: the pair holding `(ptr, len)` of the bytes.
    Pair(SlicePair),
}

pub type Deref = fn(&Value) -> Value;

/// The MIR's `PathSeg::Payload` resolved to the shape the preparation read
/// from the type at that point. An option's payload step survives only where
/// the payload type is itself an option; anywhere else a `Some` is its
/// payload's own value and the step is dropped (RFC-0039 rule 6).
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Field(FieldAt),
    Index(usize),
    OptionPayload,
    VariantPayload,
}

/// `ops::chain` applies `Neg` and `WrappingNeg` to the left operand alone, and a
/// chain node carrying either is built with its right leaf unread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arith {
    /// At an integer width the trapping operation (`ir::Overflow::Trap`),
    /// as `Sub`, `Mul` and `Neg` are; at `f64` the IEEE one.
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    /// At an integer width the wrapping operation (`ir::Overflow::Wrap`), as
    /// `WrappingSub`, `WrappingMul` and `WrappingNeg` are; at `f64` the IEEE
    /// one.
    WrappingAdd,
    WrappingSub,
    WrappingMul,
    WrappingNeg,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compare {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Root {
    Num(Arith),
    Cmp(Compare),
}

/// RFC-0044 made the shape a type parameter of the chain operation. Under
/// RFC-0052 it is a field, and the chain family fell from 1404 instances per
/// entry to 351.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// `(a ∘ b)`
    NLL,
    /// `((a ∘ b) ∘ c)`
    NNLLL,
    /// `(a ∘ (b ∘ c))`
    NLNLL,
    /// `(((a ∘ b) ∘ c) ∘ d)`
    NNNLLLL,
    /// `((a ∘ (b ∘ c)) ∘ d)`
    NNLNLLL,
    /// `((a ∘ b) ∘ (c ∘ d))`
    NNLLNLL,
    /// `(a ∘ ((b ∘ c) ∘ d))`
    NLNNLLL,
    /// `(a ∘ (b ∘ (c ∘ d)))`
    NLNLNLL,
}

impl Shape {
    pub fn interior(self) -> usize {
        match self {
            Shape::NLL => 0,
            Shape::NNLLL | Shape::NLNLL => 1,
            Shape::NNNLLLL | Shape::NNLNLLL | Shape::NNLLNLL | Shape::NLNNLLL | Shape::NLNLNLL => 2,
        }
    }

    pub fn leaves(self) -> usize {
        self.interior() + 2
    }

    pub fn nodes(self) -> usize {
        self.interior() + 1
    }

    pub fn of_word(word: &str) -> Option<Shape> {
        let shape = match word {
            "NLL" => Shape::NLL,
            "NNLLL" => Shape::NNLLL,
            "NLNLL" => Shape::NLNLL,
            "NNNLLLL" => Shape::NNNLLLL,
            "NNLNLLL" => Shape::NNLNLLL,
            "NNLLNLL" => Shape::NNLLNLL,
            "NLNNLLL" => Shape::NLNNLLL,
            "NLNLNLL" => Shape::NLNLNLL,
            _ => return None,
        };
        Some(shape)
    }
}

pub struct ChainBounds;

impl ChainBounds {
    pub const MAX_NODES: usize = 3;
    pub const MAX_INTERIOR: usize = ChainBounds::MAX_NODES - 1;
    pub const MAX_LEAVES: usize = ChainBounds::MAX_NODES + 1;

    /// Pre-multiplied to the byte position of the register's **word**, which
    /// is what `ops::chain` adds to the frame pointer without a shift.
    pub fn byte_offset_of_word(at: Off) -> u16 {
        let off = at.byte() + Value::WORD_OFFSET;
        u16::try_from(off).expect("a chain reads a register whose offset fits a u16")
    }
}

/// One synchronous closure call. The arguments are already in the first
/// `arity` registers of `window`; `f` is the closure value the captures are
/// read from, taken by value — two registers — because the address of a
/// caller's local handed to an indirect callee is an address that callee may
/// keep, which costs the caller its tail call to its successor
/// (`benches/asm_probe.rs`).
///
/// # Safety
/// `f` is a closure of this `Code`: `Value::code_of` names this `Code` and
/// `Value::captures_of` names the captures its body reads.
pub type Entry = unsafe fn(
    code: &Code,
    f: Value,
    rt: &AcvusRuntime,
    window: &mut FrameState,
    arity: u16,
) -> Value;

/// The head of a `Code` is the function that enters it (RFC-0069 rule 5): a call
/// loads the word and calls it, with no test of the shape at the call.
pub struct Code {
    /// Chosen once where the `Code` is made: `entry_expr` for a chain,
    /// `entry_body` for a framed body. Private, because pairing it with the
    /// `body` below is what makes the entry's read of that body sound.
    entry: Entry,
    pub body: CodeBody,
}

/// `prepare` decides this once: a body that is exactly `params -> one chain
/// -> return` after register selection is an `Expr`, and the machine runs one
/// with no frame, no `Machine` and no dispatch loop (RFC-0044 rule 5).
pub enum CodeBody {
    Body(Arc<Body>),
    Expr(Arc<Expr>),
    Rust,
}

impl CodeBody {
    /// # Safety
    /// This is the body of a `Code` whose entry is `entry_body`, which
    /// `Code::body` is the one writer of.
    pub(crate) unsafe fn body_unchecked(&self) -> &Body {
        let CodeBody::Body(body) = self else {
            // SAFETY: the caller's contract.
            unsafe { std::hint::unreachable_unchecked() }
        };
        body
    }

    /// # Safety
    /// As `body_unchecked`, for `Code::expr` and `entry_expr`.
    pub(crate) unsafe fn expr_unchecked(&self) -> &Expr {
        let CodeBody::Expr(expr) = self else {
            // SAFETY: the caller's contract.
            unsafe { std::hint::unreachable_unchecked() }
        };
        expr
    }
}

/// A closure's code, named by address (RFC-0069 rule 2): one word, nothing
/// counted.
///
/// Obligation across artifacts: the `Code` is `RUST`, a `static`, or owned by
/// a `Prepared`, which the run's `InterpreterContext` owns, and a closure
/// value does not outlive its run — a space refuses a `Fn` (`layout.rs`), a
/// spawned run shares the context, and the checker refuses a closure in the
/// entry's result (`MirErrorKind::ClosureReturnedToTheHost`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CodeRef(std::ptr::NonNull<Code>);

// SAFETY: a `CodeRef` is a shared borrow of a `Code`, which is `Send + Sync`
// (the assertion below), held for a term the obligation above states.
unsafe impl Send for CodeRef {}
// SAFETY: as `Send`.
unsafe impl Sync for CodeRef {}

const _: fn() = || {
    fn shared_across_threads<T>()
    where
        T: Send + Sync,
    {
    }
    shared_across_threads::<Code>();
};

impl CodeRef {
    /// Minted where `prepare` holds the `Prepared`'s own `Arc`.
    pub(crate) fn of(code: &Arc<Code>) -> CodeRef {
        CodeRef(std::ptr::NonNull::from(code.as_ref()))
    }

    /// The `Code` for as long as the type's obligation holds: the borrow is
    /// unbounded because no value that holds a `CodeRef` outlives it.
    pub fn code<'c>(self) -> &'c Code {
        // SAFETY: the type's obligation.
        unsafe { &*self.0.as_ptr() }
    }

    pub fn address(self) -> *const () {
        self.0.as_ptr().cast()
    }

    /// # Safety
    /// `address` is `CodeRef::address` of a `CodeRef`.
    #[inline(always)]
    pub(crate) unsafe fn of_address(address: *const Code) -> CodeRef {
        // SAFETY: the caller's contract: a `CodeRef` is a `NonNull`.
        CodeRef(unsafe { std::ptr::NonNull::new_unchecked(address.cast_mut()) })
    }
}

static RUST: Code = Code {
    entry: crate::machine::entry_rust,
    body: CodeBody::Rust,
};

/// The one closure over `RUST`: its one capture is `callee`, which
/// `entry_rust` reads.
pub(crate) fn rust_fn(callee: acvus_extern::RustCallee<AcvusRuntime>) -> Value {
    // SAFETY: the one reader of this value is `entry_rust`, which reads it
    // as the `RustCallee<AcvusRuntime>` it was erased from.
    let erased = unsafe { Value::erase(callee) };
    // SAFETY: `erase` made the word just now, and nothing else holds it.
    let state = unsafe { Owned::from_value(acvus_extern::Holding::new(), erased) };
    Value::closure(CodeRef(std::ptr::NonNull::from(&RUST)), &mut std::iter::once(state))
}

impl Code {
    /// A framed body: the entry binds a frame in the caller's window.
    pub fn body(body: Arc<Body>) -> Code {
        Code {
            entry: crate::machine::entry_body,
            body: CodeBody::Body(body),
        }
    }

    /// One chain, or one argument handed back: the entry is the one the
    /// `Expr` carries, chosen by `prepare` from the shape it built.
    pub fn expr(expr: Arc<Expr>) -> Code {
        Code {
            entry: expr.entry,
            body: CodeBody::Expr(expr),
        }
    }

    /// The one synchronous entry.
    ///
    /// # Safety
    /// As `Entry`.
    #[inline]
    pub unsafe fn call(
        &self,
        f: Value,
        rt: &AcvusRuntime,
        window: &mut FrameState,
        arity: u16,
    ) -> Value {
        // SAFETY: the caller's contract, and the entry is the one this
        // `Code`'s constructor paired with its body.
        unsafe { (self.entry)(self, f, rt, window, arity) }
    }

    pub fn may_suspend(&self) -> bool {
        match &self.body {
            CodeBody::Body(body) => body.may_suspend,
            CodeBody::Expr(_) | CodeBody::Rust => false,
        }
    }
}

pub struct Expr {
    pub arity: u32,
    /// The entry `Code::expr` puts at the head of a `Code` over this `Expr`:
    /// `machine::entry_argument` for an `Argument`, one of
    /// `ops::chain`'s entries for a `Chain`. `prepare` writes it beside the
    /// `body` below, and that pairing is what makes each entry's unchecked
    /// read of the body sound.
    pub entry: Entry,
    pub body: ExprBody,
    pub span: Span,
}

pub enum ExprBody {
    Argument(u16),
    Chain(ExprChain),
}

impl Expr {
    /// # Safety
    /// This is the `Expr` of a `Code` whose entry is `entry_argument`, which
    /// `prepare::expression_body` writes beside this variant alone.
    pub(crate) unsafe fn argument_unchecked(&self) -> u16 {
        let ExprBody::Argument(at) = self.body else {
            // SAFETY: the caller's contract.
            unsafe { std::hint::unreachable_unchecked() }
        };
        at
    }

    /// # Safety
    /// As `argument_unchecked`, for the entries `ops::chain::chain_eval`
    /// picks.
    pub(crate) unsafe fn chain_unchecked(&self) -> &ExprChain {
        let ExprBody::Chain(chain) = &self.body else {
            // SAFETY: the caller's contract.
            unsafe { std::hint::unreachable_unchecked() }
        };
        chain
    }
}

pub struct ExprChain {
    pub plan: crate::ops::chain::Plan,
    /// The kind the root's word is: the operand type's for an arithmetic
    /// root, `Bool` for a comparison. A frameless call has no frame to have
    /// written it, so the chain carries it.
    pub kind: Kind,
    pub konsts: Box<[Value]>,
}

impl ExprChain {
    pub const MAX_OPERANDS: usize = 8;
}

pub struct EntryKonst {
    pub slot: Off,
    pub value: Value,
}

/// RFC-0052 rule 5: the frame writes this slot's kind once, when it is made, and
/// every operation that writes the slot afterwards writes the **word only**.
pub struct SlotKind {
    pub slot: Off,
    pub kind: Kind,
}

pub struct Body {
    /// One chain head per joint. `Machine::run` enters here and nowhere
    /// else: a straight run is a chain inside one of these.
    pub heads: Box<[Box<dyn Op>]>,
    pub entry: BlockId,
    /// A call out of this body takes its callee's frame from the window above
    /// this many slots (RFC-0052 rule 7).
    pub frame_len: u16,
    /// These two are fields rather than expressions over `frame_len` because
    /// three earlier attempts at RFC-0050 rule 2 computed them where they are
    /// read, and every one of them regressed the closure-binding benchmarks —
    /// `map cap | sum` by 7.6 % in the last of the three, which binds a frame
    /// per element. A bind, a window's `fits` and a sweep are the readers, and
    /// all three run per call.
    pub frame_cells: u16,
    pub mark_words: MarkWords,
    pub entry_konsts: Box<[EntryKonst]>,
    /// The module's literals, held here because an operation of this body
    /// names their bytes by address.
    pub literals: Arc<Literals>,
    pub slot_kinds: Box<[SlotKind]>,
    pub may_suspend: bool,
    /// The result is the register pair of RFC-0047 rule 6, which only a
    /// call's pair destination receives.
    pub returns_a_view: bool,
    pub params: Box<[Off]>,
    /// The registers a caller lays its arguments in, `params` at their widths.
    pub param_run: u16,
    /// The parameter slots whose type owns a `Large`: what the caller gave up
    /// with its own `take_mask`, claimed here in one store.
    pub param_marks: u64,
    pub captures: Box<[Off]>,
    pub order_param: Option<Off>,
    pub span: Span,
}

/// Each body here was prepared once, when the module was loaded (RFC-0044);
/// a `MakeClosure` copies an `Arc`.
pub struct Prepared {
    pub main: Arc<Body>,
    pub closures: FxHashMap<Label, Arc<Code>>,
    pub instances: crate::prepare::InstanceEntryStore,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An operation is reached through a `Box<dyn Op>` and its fields are read
    /// on the one path that runs it, so a family spread over two cache lines
    /// pays a second miss per operation. Carrying the mark word and bit as
    /// fields (RFC-0050 rule 2) costs fourteen bytes per register an operation
    /// marks, and this is what keeps that cost inside one line.
    #[test]
    fn no_operation_family_spans_two_cache_lines() {
        use crate::ops;

        const LINE: usize = 64;
        for (name, size) in [
            (
                "arith::Add<i64,Slot,Slot,Slot>",
                size_of::<ops::arith::Add<i64, ops::place::Slot, ops::place::Slot, ops::place::Slot>>(
                ),
            ),
            ("control::DropValue", size_of::<ops::control::DropValue>()),
            (
                "control::Mov<true,false>",
                size_of::<ops::control::Mov<true, false>>(),
            ),
            (
                "storage::AssignVar<true>",
                size_of::<ops::storage::AssignVar<true>>(),
            ),
            ("storage::Update", size_of::<ops::storage::Update>()),
            (
                "string::CloneString<false>",
                size_of::<ops::string::CloneString<false>>(),
            ),
            (
                "composite::MakeObject",
                size_of::<ops::composite::MakeObject>(),
            ),
        ] {
            assert!(
                size <= LINE,
                "{name} is {size} bytes, past the {LINE} one cache line holds"
            );
        }
    }

    /// # Safety
    /// `run` names live UTF-8, which is what this test is checking.
    unsafe fn text_of(run: acvus_extern::Words) -> &'static str {
        // SAFETY: the caller's contract.
        unsafe { run.str() }
    }

    #[test]
    fn a_run_outlives_the_text_the_table_was_built_from() {
        let literals = {
            let module_text = String::from("héllo");
            Literals::of(std::iter::once(module_text.as_str()))
        };
        let run = literals.run("héllo");
        assert_eq!(run.len(), 6);
        // SAFETY: the table owns the copy and is alive here, which is the
        // claim under test.
        assert_eq!(unsafe { text_of(run) }, "héllo");
    }

    #[test]
    fn one_text_twice_is_one_run() {
        let literals = Literals::of(["abc", "abc"].into_iter());
        assert_eq!(literals.run("abc"), literals.run("abc"));
    }
}
