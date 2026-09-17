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
use acvus_mir::ir::{Label, PathSeg};
use acvus_utils::Astr;
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;

use crate::error::RuntimeError;
use crate::machine::Machine;
use crate::runtime::ExternHandler;
use crate::value::{Tag, Value};

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
    pub fut: BoxFuture<'static, Result<Value, RuntimeError>>,
}

/// A constant that is not one word.
pub enum Konst {
    /// A tagged word, as `Value::Small` holds it.
    Word(Tag, u64),
    Str(String),
    List(Box<[Konst]>),
}

impl Konst {
    pub fn value(&self) -> Value {
        match self {
            Konst::Word(tag, bits) => Value::Small(*tag, *bits),
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
pub struct SlotMove {
    pub from: u32,
    pub to: u32,
}

/// What an operation keeps outside its inline words. The `Code` owns the
/// table; an operation reads it by index.
pub enum Payload {
    Path(Box<[PathSeg]>),
    /// Register slots in order: a composite's elements, a call's arguments.
    Slots(Box<[u32]>),
    Parts(Box<[ConcatPart]>),
    Fields(Box<[FieldSlot]>),
    /// A jump's moves, in the order `prepare::order_moves` put them.
    Moves(Box<[SlotMove]>),
    /// A variant's tag, or an object's key.
    Name(Astr),
    PageKey(Box<str>),
    /// The string a pattern test compares against.
    Text(String),
    /// The integer a pattern test compares against, at the width the
    /// literal was written, which no inline word carries.
    Wide(i128),
    Konst(Konst),
    Extern {
        handler: ExternHandler,
        args: Box<[u32]>,
    },
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
        Payload::Name(_) => "Name",
        Payload::PageKey(_) => "PageKey",
        Payload::Text(_) => "Text",
        Payload::Wide(_) => "Wide",
        Payload::Konst(_) => "Konst",
        Payload::Extern { .. } => "Extern",
        Payload::Direct { .. } => "Direct",
        Payload::Closure { .. } => "Closure",
    }
}

/// One prepared body.
pub struct Code {
    pub ops: Box<[Op]>,
    /// The span of the instruction each operation came from, read when an
    /// error is raised.
    pub spans: Box<[Span]>,
    pub payloads: Box<[Payload]>,
    /// The registers a run of this body needs: one per `ValueId`, and the
    /// scratch slot where a jump's moves needed one.
    pub frame_len: u32,
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
