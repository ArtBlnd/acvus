//! The graph: the units compiled together (programs), each a function at a
//! name the embedding gives, and the host's declarations beside them.
//!
//! Concepts:
//! - A `Unit` is a script or a template (a module is to-be): one function
//!   at a `QualifiedRef` the embedding gives. Its inputs `$name` are its
//!   parameters, passed by name (programs §1, §2, §4).
//! - Its `Interface` is its scheme, generalized exactly as a function value
//!   bound by `let` is (functions §2): a unit is a function whose
//!   environment is empty, so the two are one generalization. It is all
//!   another unit sees of it: its function type over what it generalizes,
//!   its parameters the inputs by name, and every decision its body could
//!   not make as a requirement made again at each call. Its types are
//!   inferred within the unit alone, and a call only instantiates them
//!   (transparency, types §1). It is kept in a store of its own, since a
//!   unit is checked apart from its callers.
//! - An `Instantiation` is a unit at types for its parameters and result:
//!   the scheme instantiated.
//! - An entry is a host-declared function a run may start at, with the
//!   types of its inputs and result (host; `decl::EntryDecl`).
//!
//! Relations:
//! - A unit reaches another only by its whole name, never by a bare one
//!   (programs); a bare name reaches only host declarations (types §6).
//!   The calls between units, read from each unit's syntax by those names,
//!   are the call graph. A cycle in it is refused: only a module's `fn`
//!   recurses (functions §4).
//! - A unit's interface is checked after the interfaces of the units it
//!   calls: `typeck` with its inputs open, generalized at the end
//!   (`typeck::Inputs::Open`). The host's declarations and the callees'
//!   interfaces are what its calls reach.
//! - An instantiation is checked again with its inputs fixed at its types
//!   (`typeck::Inputs::Fixed`), then lowered, taken through the required
//!   steps and the checks, and optimized: one body. The instantiations are
//!   gathered from the entries down: each body's calls of another unit ask
//!   for that unit's instantiation at the call's types.
//! - One source, one `Module`: every body of a unit's instantiations.
//! - Inline, per module, after every instantiation is compiled: a call of
//!   another unit's instantiation may be replaced by that body, its types
//!   carried into the calling module's table.
//!
//! Incremental:
//! - An interface is checked again only where its source, the host's
//!   declarations or an interface it read changed; where the result equals
//!   the one before, no unit that read it is checked again (early cutoff).
//! - An instantiation is compiled again only where its unit's interface
//!   inputs changed or its types did.
//! - Inline is done again for a module only where it or a module it inlines
//!   from changed.
//!
//! Deterministic: units, instantiations and bodies are taken in the order
//! of their names and types, never a hash map's.

use acvus_utils::QualifiedRef;

use crate::decl::Declarations;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::pipeline::Level;

/// A unit's source.
pub enum Source {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

pub struct Unit {
    pub name: QualifiedRef,
    pub source: Source,
}

pub use crate::typeck::{Interface, Interfaces};

/// The graph a host compiles, kept across compiles.
pub struct Graph {
    // The host's declarations, verified once, the units by name, and each
    // step's cached result with its key.
}

/// A unit's instantiation's body in its module.
pub struct Entry {
    pub module: QualifiedRef,
    pub body: BodyId,
}

/// What a compile gives: each unit's module, and each host-declared entry's
/// body; or every refusal, by unit.
pub struct Compiled<'g> {
    pub modules: Vec<(QualifiedRef, &'g Module)>,
    pub entries: Vec<(QualifiedRef, Entry)>,
}

pub struct Refused {
    pub unit: QualifiedRef,
    pub refusals: Vec<Refusal>,
}

/// A refusal of a unit: its parse, its checker's (typeck's or check's), or
/// the graph's (a cycle of calls through no `fn`, an entry whose declared
/// types the unit's interface does not admit, a call of no unit).
pub enum Refusal {
    Parse(acvus_ast::ParseError),
    Typeck(crate::typeck::Refusal),
    Check(crate::check::Refusal),
    Cycle { through: Vec<QualifiedRef> },
    EntryDoesNotFit { entry: QualifiedRef },
}

impl Graph {
    pub fn new(declarations: Declarations) -> Graph {
        todo!("{:?}", std::ptr::from_ref(&declarations))
    }

    /// Adds or replaces the unit at `unit.name`.
    pub fn set_unit(&mut self, unit: Unit) {
        todo!("{:?}", unit.name)
    }

    pub fn remove_unit(&mut self, name: QualifiedRef) {
        todo!("{name:?}")
    }

    /// Compiles what changed since the last compile, at `level`.
    pub fn compile(&mut self, level: Level) -> Result<Compiled<'_>, Vec<Refused>> {
        todo!("{level:?}")
    }
}
