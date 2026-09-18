//! The prepared form of a body: a linear array of fixed-size operations
//! (RFC-0044).
//!
//! A `Code` is what the machine runs. Every static fact an operation needs
//! — the operand slots, the type an arithmetic runs at, a literal's word, a
//! label's target index, an extern instance's handler, a closure's body —
//! is resolved when the `Code` is prepared. A `Code` is shared by `Arc`: a
//! `MakeClosure` copies a pointer.
//!
//! An operation is one `Op`: four `u32` words, each a register slot or a
//! small immediate, and one `p` word, either an immediate (an `f64`'s bits,
//! an integer, an index) or an index into the `Code`'s payload table, where
//! an operation that needs more than the inline words keeps its rest.

use std::sync::Arc;

use acvus_ast::Span;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::Label;
use acvus_utils::Astr;
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;

use crate::machine::Machine;
use crate::runtime::ExternHandler;
use crate::value::{Kind, Value};

/// The `a..d` word of an operation that names no register: a call with no
/// order edge, a variant constructor with no payload.
pub const NO_SLOT: u32 = u32::MAX;

pub type OpFn = fn(&mut Machine<'_>, &Op) -> Flow;

#[repr(C)]
pub struct Op {
    pub f: OpFn,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub p: usize,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Op>() == 32);

impl Op {
    pub fn new(f: OpFn) -> Self {
        Self {
            f,
            a: NO_SLOT,
            b: NO_SLOT,
            c: NO_SLOT,
            d: NO_SLOT,
            p: 0,
        }
    }

    pub fn a(mut self, a: u32) -> Self {
        self.a = a;
        self
    }

    pub fn b(mut self, b: u32) -> Self {
        self.b = b;
        self
    }

    pub fn c(mut self, c: u32) -> Self {
        self.c = c;
        self
    }

    pub fn d(mut self, d: u32) -> Self {
        self.d = d;
        self
    }

    pub fn p(mut self, p: usize) -> Self {
        self.p = p;
        self
    }
}

/// Where the machine goes after an operation.
pub enum Flow {
    Next,
    Jump(u32),
    /// Leave the loop with what the operation left in `Machine::exit`.
    Return,
    /// Hand a future up to the driver, which awaits it, stores the result
    /// in `dst`, and re-enters the loop at the next operation.
    Await(Pending),
}

pub struct Pending {
    pub dst: u32,
    pub fut: BoxFuture<'static, Value>,
}

/// A constant that is not one word.
pub enum Konst {
    /// A word and the `Kind` it carries.
    Word(Kind, u64),
    Str(String),
    List(Box<[Konst]>),
}

impl Konst {
    pub fn value(&self) -> Value {
        match self {
            Konst::Word(kind, bits) => Value::inline(*kind, *bits),
            Konst::Str(s) => Value::string(s.as_str()),
            Konst::List(items) => Value::array(items.iter().map(Konst::value).collect()),
        }
    }
}

/// One part of a template's output.
pub struct ConcatPart {
    pub slot: u32,
    pub through_reference: bool,
}

/// One field of an object constructor.
pub struct FieldSlot {
    pub key: Astr,
    pub slot: u32,
}

/// One move of a jump's parallel move.
#[derive(Clone, Copy)]
pub struct SlotMove {
    pub from: u32,
    pub to: u32,
}

/// A basic block without its terminator: the operation that owns one knows
/// where control goes after it.
pub struct BasicBlock(Box<[Op]>);

impl BasicBlock {
    pub fn new<I>(operations: I) -> Self
    where
        I: IntoIterator<Item = Op>,
    {
        Self(operations.into_iter().collect())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Op> {
        self.0.iter()
    }
}

/// An extern call site's arguments as the frame holds them: `arity`
/// contiguous registers from `at`, which the handler is lent and empties,
/// and the moves that fill the ones no argument was allocated into
/// (RFC-0044, stage 2b).
#[derive(Clone)]
pub struct ArgWindow {
    pub at: u32,
    pub arity: u32,
    pub moves: Box<[SlotMove]>,
}

/// An extern call site, resolved: which handler runs, where its `Order`
/// lands, and how its arguments reach it.
pub struct ExternCall {
    pub handler: ExternHandler,
    /// `NO_SLOT` for a pure call. It lives here rather than in an `Op`
    /// word because a by-value call spends all three words on arguments.
    pub order: u32,
    pub args: ExternArgs,
}

#[derive(Clone)]
pub enum ExternArgs {
    /// The operation's `b`, `c`, `d` words are the argument slots, in the
    /// declaration's order, as many as the arity: what `prepare` writes
    /// there is what `ops::call::call_extern_0..3` reads.
    ByValue,
    Window(ArgWindow),
}

/// The `while` shape `prepare::recognize_loop` finds in the IR and
/// `control::while_loop` runs (RFC-0044, stage 3).
pub struct LoopBody {
    pub enter: Box<[SlotMove]>,
    pub head: BasicBlock,
    pub cond_slot: u32,
    pub into_body: Box<[SlotMove]>,
    pub body: BasicBlock,
    pub back: Box<[SlotMove]>,
    pub exit: Box<[SlotMove]>,
}

pub struct DiamondArm {
    pub block: BasicBlock,
    pub join: Box<[SlotMove]>,
}

/// The `if/else` shape `prepare::recognize_diamond` finds in the IR and
/// `control::diamond` runs (RFC-0044, stage 5).
pub struct Diamond {
    pub on_true: DiamondArm,
    pub on_false: DiamondArm,
}

/// The operator of one node of a chain, applied by `Arith::apply` in
/// `ops::chain`. `Neg` ignores its right operand.
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

/// What a chain's root node does with the two values below it, and so
/// what kind of value the chain produces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Root {
    Num(Arith),
    Cmp(Compare),
}

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Shape2 {
    NNLLL,
    NLNLL,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Shape3 {
    NNNLLLL,
    NNLNLLL,
    NNLLNLL,
    NLNNLLL,
    NLNLNLL,
}

impl Shape2 {
    pub const fn read(raw: u8) -> Shape2 {
        match raw {
            0 => Shape2::NNLLL,
            1 => Shape2::NLNLL,
            _ => panic!("a two-node chain instance was parameterized by no shape"),
        }
    }
}

impl Shape3 {
    pub const fn read(raw: u8) -> Shape3 {
        match raw {
            0 => Shape3::NNNLLLL,
            1 => Shape3::NNLNLLL,
            2 => Shape3::NNLLNLL,
            3 => Shape3::NLNNLLL,
            4 => Shape3::NLNLNLL,
            _ => panic!("a three-node chain instance was parameterized by no shape"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arity {
    One,
    Two(Shape2),
    Three(Shape3),
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

    pub fn slots(self) -> usize {
        self.interior() + 1
    }

    pub fn arity(self) -> Arity {
        match self {
            Shape::NLL => Arity::One,
            Shape::NNLLL => Arity::Two(Shape2::NNLLL),
            Shape::NLNLL => Arity::Two(Shape2::NLNLL),
            Shape::NNNLLLL => Arity::Three(Shape3::NNNLLLL),
            Shape::NNLNLLL => Arity::Three(Shape3::NNLNLLL),
            Shape::NNLLNLL => Arity::Three(Shape3::NNLLNLL),
            Shape::NLNNLLL => Arity::Three(Shape3::NLNNLLL),
            Shape::NLNLNLL => Arity::Three(Shape3::NLNLNLL),
        }
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

/// A run of arithmetic as one operation (RFC-0044, stage 4).
///
/// The chain is a pure expression. Where its value goes is the consumer's:
/// an operation in a body writes it to `op.a`, and a `Code::Expr` returns
/// it.
pub struct Chain {
    pub shape: Shape,
    pub post_order_ops: [Arith; Chain::MAX_INTERIOR],
    pub root: Root,
    pub leaf_offsets: [u16; Chain::MAX_LEAVES],
}

impl Chain {
    pub const MAX_NODES: usize = 3;
    pub const MAX_INTERIOR: usize = Chain::MAX_NODES - 1;
    pub const MAX_LEAVES: usize = Chain::MAX_NODES + 1;

    pub fn offset(slot: u32) -> u16 {
        let off = slot as usize * size_of::<Value>() + Value::WORD_OFFSET;
        u16::try_from(off).expect("a chain reads a register whose offset fits a u16")
    }
}

/// A path step, with the MIR's `PathSeg::Payload` resolved to the shape
/// the preparation read from the type at that point. An option's payload
/// step survives only where the payload type is itself an option; anywhere
/// else a `Some` is its payload's own value and the step is dropped
/// (RFC-0022).
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Field(Astr),
    Index(usize),
    OptionPayload,
    ResultPayload,
    VariantPayload,
}

/// What an operation keeps outside its inline words. The `Code` owns the
/// table; an operation reads it by index.
pub enum Payload {
    Path(Box<[Step]>),
    /// Register slots in order: a composite's elements, a call's arguments.
    Slots(Box<[u32]>),
    Parts(Box<[ConcatPart]>),
    Fields(Box<[FieldSlot]>),
    /// A jump's moves, in the order `prepare::order_moves` put them.
    Moves(Box<[SlotMove]>),
    Loop(LoopBody),
    Diamond(Diamond),
    /// A variant's tag, or an object's key.
    Name(Astr),
    PageKey(Box<str>),
    /// The string a pattern test compares against.
    Text(String),
    /// The integer a pattern test compares against, at the width the
    /// literal was written, which no inline word carries.
    Wide(i128),
    Konst(Konst),
    Extern(ExternCall),
    Chain(Box<Chain>),
    Direct {
        callee: QualifiedRef,
        args: Box<[u32]>,
    },
    Closure {
        code: Arc<Code>,
        captures: Box<[u32]>,
    },
}

/// The variant name a payload mismatch reports.
pub fn payload_name(payload: &Payload) -> &'static str {
    match payload {
        Payload::Path(_) => "Path",
        Payload::Slots(_) => "Slots",
        Payload::Parts(_) => "Parts",
        Payload::Fields(_) => "Fields",
        Payload::Moves(_) => "Moves",
        Payload::Loop(_) => "Loop",
        Payload::Diamond(_) => "Diamond",
        Payload::Name(_) => "Name",
        Payload::PageKey(_) => "PageKey",
        Payload::Text(_) => "Text",
        Payload::Wide(_) => "Wide",
        Payload::Konst(_) => "Konst",
        Payload::Chain(_) => "Chain",
        Payload::Extern(_) => "Extern",
        Payload::Direct { .. } => "Direct",
        Payload::Closure { .. } => "Closure",
    }
}

/// One prepared body.
///
/// A body that is exactly `params -> one chain -> return` after register
/// selection is an `Expr`, which runs with no frame, no `Machine` and no
/// dispatch loop; every other body is a `Body`. The distinction is decided
/// once, at preparation, and every caller matches on it (RFC-0044,
/// stage 4).
pub enum Code {
    Body(Body),
    Expr(Expr),
}

impl Code {
    /// Whether any operation of this body can return `Flow::Await`.
    pub fn may_suspend(&self) -> bool {
        match self {
            Code::Body(body) => body.may_suspend,
            Code::Expr(_) => false,
        }
    }

    /// The span an ICE about this body names it by: a `Code` carries no
    /// name.
    pub fn site(&self) -> Span {
        match self {
            Code::Body(body) => body.spans.first().copied().unwrap_or(Span::ZERO),
            Code::Expr(expr) => expr.span,
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
    pub chain: Chain,
    pub eval: ExprFn,
    pub konsts: Box<[Value]>,
}

impl ExprChain {
    pub const MAX_OPERANDS: usize = 8;
}

pub type ExprFn = fn(&Chain, &[Value]) -> Value;

pub struct EntryKonst {
    pub slot: u32,
    pub value: Value,
}

/// A prepared body the machine runs operation by operation.
pub struct Body {
    pub ops: Box<[Op]>,
    /// The span of the instruction each operation came from, read when an
    /// error is raised.
    pub spans: Box<[Span]>,
    pub payloads: Box<[Payload]>,
    /// The registers a run of this body needs: one per `ValueId`, the
    /// scratch slot where a jump's moves needed one, and one per
    /// `entry_konsts` entry.
    pub frame_len: u32,
    pub entry_konsts: Box<[EntryKonst]>,
    /// Whether any operation of this body can return `Flow::Await`. The
    /// preparation sets it; `ops::call`'s synchronous path asserts it is
    /// false on the callee it is about to run.
    pub may_suspend: bool,
    pub params: Box<[u32]>,
    pub captures: Box<[u32]>,
    pub order_param: Option<u32>,
}

/// A module as the machine runs it: its entry body and every closure body
/// it makes, each prepared once.
pub struct Prepared {
    pub main: Arc<Code>,
    pub closures: FxHashMap<Label, Arc<Code>>,
}
