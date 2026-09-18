//! The prepared form of a body (RFC-0052).
//!
//! `prepare` and the `ops` module are two halves of one contract: every field
//! of an operation is a fact `prepare` decided and moved in, so a decision
//! that reaches a `run` as anything but a field or a type parameter is a
//! defect in `prepare`, not in the operation.

use std::sync::Arc;

use acvus_ast::Span;
use acvus_extern::Owned;
use acvus_mir::ir::Label;
use acvus_utils::Astr;
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;

use crate::machine::Machine;
use crate::value::{Kind, Value};

/// `prepare::check_assignment` proves every slot an operation names is below
/// its body's `frame_len`, which is what makes a slot access unchecked in
/// release.
///
/// This is the language-level register index, and it lives in `prepare`: the
/// register assignment, the liveness and the parallel-move ordering are
/// written in it. An operation never holds one — it holds an `Off`.
pub type Slot = u16;

/// A register's byte displacement inside its frame.
///
/// `prepare` multiplies once, when it builds the operation, so no `run`
/// scales: `Add::<i64>::run`'s three `shl $4` are what this type removes
/// (RFC-0052 §5). `Slot` and `Off` are different types so that the index and
/// the displacement cannot be handed to each other's reader.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Off(u16);

impl Off {
    /// The widest register index an `Off` can stand for: the frame's mark word
    /// is 64 bits, so `mark` shifts by less than 64 and `of` cannot overflow.
    pub const MAX_INDEX: u16 = 63;

    /// # Panics
    /// `slot` is past `MAX_INDEX`, which `prepare`'s frame sizing must not
    /// emit — the same bound `regs::Store::new` asserts.
    pub const fn of(slot: Slot) -> Off {
        assert!(
            slot <= Off::MAX_INDEX,
            "an operation names a register past the 64 one frame's mark word reaches"
        );
        Off(slot * size_of::<Value>() as u16)
    }

    /// The displacement `Regs` adds to the frame's first byte.
    #[inline(always)]
    pub const fn byte(self) -> usize {
        self.0 as usize
    }

    /// The register index, for the diagnostics and the listings that speak in
    /// `Slot`; no `run` calls it.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize / size_of::<Value>()
    }

    /// The one `Off` that is not a displacement: the argument of a fused call
    /// that reads the call before it rather than a register (RFC-0044, stage
    /// 6). `ops::call::arg` matches it before anything reads it as one, and
    /// `of` cannot produce it.
    pub const PREVIOUS: Off = Off(u16::MAX);

    /// This register's bit of the frame's mark word. `of` bounds the shift.
    #[inline(always)]
    pub const fn mark(self) -> u64 {
        1u64 << (self.0 / size_of::<Value>() as u16)
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

/// The body returns what it left in `Machine::exit`.
pub const RETURN: BlockId = BlockId::MAX;

/// The body left a future in `Machine::pending` for the driver.
pub const SUSPEND: BlockId = BlockId::MAX - 1;

/// `Machine::run` is one compare against this per block.
pub const SENTINEL: BlockId = SUSPEND;

const _: () = assert!(
    RETURN >= SENTINEL && SUSPEND >= SENTINEL,
    "the machine's loop leaves on RETURN and on SUSPEND"
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
/// fact its own type carries is RFC-0052 §1.
///
/// `r0` is the word the operation before it in the same chain produced
/// (RFC-0052 rule 5, `Place::R0`).
#[cfg(any(debug_assertions, feature = "probe"))]
pub trait Op: Named + Send + Sync {
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId;

    fn chain(&self) -> Option<ChainProbe<'_>> {
        None
    }

    /// The registers a **bound-checked** indexed read names, so that a probe
    /// can put the unchecked form of the same read in its place (RFC-0047
    /// §7). Nothing else answers it, and the unchecked form answers `None`,
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
/// there held (RFC-0047 §7's probe, now that a successor is a field).
#[cfg(any(debug_assertions, feature = "probe"))]
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
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId;
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
pub struct Pending {
    pub dst: Off,
    pub owns_large: bool,
    pub fut: BoxFuture<'static, Value>,
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
            Konst::List(items) => Value::array(
                items
                    .iter()
                    .map(|item| Owned::from_value(item.value()))
                    .collect(),
            ),
        }
    }
}

pub struct ConcatPart {
    pub slot: Off,
    pub through_reference: bool,
}

pub struct FieldSlot {
    pub key: Astr,
    pub slot: Off,
}

pub type Deref = fn(&Value) -> Value;

/// The MIR's `PathSeg::Payload` resolved to the shape the preparation read
/// from the type at that point. An option's payload step survives only where
/// the payload type is itself an option; anywhere else a `Some` is its
/// payload's own value and the step is dropped (RFC-0022).
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Field(Astr),
    Index(usize),
    OptionPayload,
    ResultPayload,
    VariantPayload,
}

/// `ops::chain` applies `Neg` to the left operand alone, and a chain node
/// carrying it is built with its right leaf unread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
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

/// `prepare` decides this once: a body that is exactly `params -> one chain
/// -> return` after register selection is an `Expr`, and the machine runs one
/// with no frame, no `Machine` and no dispatch loop (RFC-0044, stage 4).
pub enum Code {
    Body(Arc<Body>),
    Expr(Arc<Expr>),
}

impl Code {
    pub fn may_suspend(&self) -> bool {
        match self {
            Code::Body(body) => body.may_suspend,
            Code::Expr(_) => false,
        }
    }
}

pub struct Expr {
    pub arity: u32,
    pub body: ExprBody,
    pub span: Span,
}

pub enum ExprBody {
    Argument(u16),
    Chain(ExprChain),
}

pub struct ExprChain {
    pub plan: crate::ops::chain::Plan,
    /// The kind the root's word is: the operand type's for an arithmetic
    /// root, `Bool` for a comparison. A frameless call has no frame to have
    /// written it, so the chain carries it.
    pub kind: Kind,
    pub eval: ExprFn,
    pub konsts: Box<[Value]>,
}

impl ExprChain {
    pub const MAX_OPERANDS: usize = 8;
}

pub type ExprFn = fn(&ExprChain, &[Value]) -> u64;

pub struct EntryKonst {
    pub slot: Off,
    pub value: Value,
}

/// RFC-0052 §5: the frame writes this slot's kind once, when it is made, and
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
    pub entry_konsts: Box<[EntryKonst]>,
    pub slot_kinds: Box<[SlotKind]>,
    pub may_suspend: bool,
    pub params: Box<[Off]>,
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
    pub main: Arc<Code>,
    pub closures: FxHashMap<Label, Arc<Code>>,
}
