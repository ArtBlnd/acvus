//! The IR printer: a body or a module as text, in notation alone. It is
//! never read back.
//!
//! What it prints is a `View`, a set of items each on or off; `View::CFG`
//! and `View::FULL` are two of them.
//! - Always: each block by its parameters, instructions and terminator; each
//!   value with its type (`ty::print`); the body's entry.
//! - `header`: the body's parameters, captures, order parameter and slots,
//!   each with its name and type, and its result type.
//! - `spans`: each instruction's and slot's span, `@start..end`.
//! - `order`: the order tokens each instruction consumes and yields.
//! - `structure`: a `Diamond`'s and a `Switch`'s join, a loop's stages and
//!   exit.
//! - `origins`: each value's `ValOrigin`.
//! - `canonical`: values, blocks and instructions renumbered in the order
//!   they print, blocks in reverse postorder from the entry, blocks no path
//!   reaches after them by their ids; off, each prints by its own id.
//!
//! Annotations are what an analysis says at a point, given by `Annotate`
//! and printed on `#` lines above the instruction, the block or the edge
//! they are at, each `key: value`, keys in order. The printer knows no
//! analysis.
//!
//! Notation only: an instruction prints as its operation's name and its
//! operands, a place as its root and path (`slot0.a`, `*v3.b`, `v4[v5]`), a
//! name by the interner, a type by `ty::print`. Nothing else is written.
//!
//! Deterministic: the same module and view print the same text. Nothing is
//! printed in a hash map's order; every set and map is printed sorted.

use acvus_utils::Interner;

use super::{BlockId, BodyId, InstId, Module};

/// The items a print shows beyond blocks, instructions, terminators and
/// types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub header: bool,
    pub spans: bool,
    pub order: bool,
    pub structure: bool,
    pub origins: bool,
    pub canonical: bool,
}

impl View {
    /// The control flow and the types.
    pub const CFG: View = View { header: false, spans: false, order: false, structure: false, origins: false, canonical: false };
    /// Everything the body holds.
    pub const FULL: View = View { header: true, spans: true, order: true, structure: true, origins: true, canonical: false };
}

/// What an analysis says at points of one body, as `key: value` pairs in
/// notation. A point it says nothing of gives none.
pub trait Annotate {
    /// At a block's start.
    fn block(&self, block: BlockId) -> Vec<(&'static str, String)>;
    /// Just before an instruction.
    fn before(&self, inst: InstId) -> Vec<(&'static str, String)>;
    /// On an edge, from the block whose terminator takes it.
    fn edge(&self, from: BlockId, to: BlockId) -> Vec<(&'static str, String)>;
}

/// `body` of `module` as `view` shows it, with each of `annotations`.
pub fn body(module: &Module, id: BodyId, names: &Interner, view: View, annotations: &[&dyn Annotate]) -> String {
    todo!("{:?} {id:?} {:?} {view:?} {}", std::ptr::from_ref(module), std::ptr::from_ref(names), annotations.len())
}

/// Every body of `module`, by `BodyId`, as `view` shows it. Annotations are
/// of one body each, so a module prints none.
pub fn module(module: &Module, names: &Interner, view: View) -> String {
    todo!("{:?} {:?} {view:?}", std::ptr::from_ref(module), std::ptr::from_ref(names))
}
