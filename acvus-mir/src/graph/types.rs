//! Type definitions for the compilation graph.
//!
//! Functions and Contexts are identified by `QualifiedRef` (namespace + name).
//! No opaque IDs — the name IS the identity.
//!
//! MIR receives **parsed ASTs**, not source strings. Parsing happens outside.

use acvus_utils::Freeze;

use crate::ty::{EffectConstraint, Hint, Ty};

// ── Identifiers ─────────────────────────────────────────────────────

acvus_utils::declare_id!(pub VersionId);
acvus_utils::declare_id!(pub ScopeId);

// Re-export from acvus-utils.
pub use acvus_utils::QualifiedRef;

// ── Constraint ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type inferred from source.
    Inferred,
    /// Exact declared type.
    Exact(Ty),
    /// Type derived from a function's output type.
    DerivedFnOutput(QualifiedRef, TypeTransform),
    /// Type derived from a context's type.
    DerivedContext(QualifiedRef, TypeTransform),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTransform {
    Identity,
    ElemOf,
}

// ── Function ────────────────────────────────────────────────────────

/// A callable signature: named, typed parameters.
#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<crate::ty::Param>,
}

/// Output + optional call signature.
#[derive(Debug, Clone)]
pub struct FnConstraint {
    /// Parameter types.
    pub signature: Option<Signature>,
    /// Output type constraint.
    pub output: Constraint,
    /// Effect upper bound. `None` = no constraint (anything allowed).
    pub effect: Option<EffectConstraint>,
    /// Hint for the function. `None` = no hint.
    pub hint: Option<Hint>,
}

/// Per-function metadata used by optimizer passes.
/// Bundles type information and scheduling hints into a single lookup.
#[derive(Debug, Clone)]
pub struct FnMetadata {
    /// Resolved function type (Ty::Fn).
    pub ty: Ty,
    /// Scheduling hint from ExternFn declaration.
    pub hint: Option<Hint>,
}

#[derive(Debug, Clone)]
pub enum FnKind {
    /// Has a parsed AST. MIR typechecks and compiles.
    Local(ParsedAst),
    /// Black box. Runtime provides the value.
    /// Effect information lives in the function's type (`Ty::Fn { effect }`).
    Extern,
}

/// Parsed AST for local functions.
#[derive(Debug, Clone)]
pub enum ParsedAst {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

/// An executable entity in the graph. Identified by `QualifiedRef`.
#[derive(Debug, Clone)]
pub struct Function {
    /// Unique identity = namespace + name.
    pub qref: QualifiedRef,
    pub kind: FnKind,
    pub constraint: FnConstraint,
}

// ── Context ──────────────────────────────────────────────────────────

/// A loadable value in the graph. Injected externally or derived from a function.
/// Identified by `QualifiedRef` (namespace + name).
#[derive(Debug, Clone)]
pub struct Context {
    /// Unique identity = namespace + name.
    pub qref: QualifiedRef,
    pub constraint: Constraint,
}

// ── Context policy ──────────────────────────────────────────────────

/// External constraints on a context, injected by the orchestration layer.
///
/// Analogous to memory page permissions:
/// - `volatile`: loads/stores must not be elided by SSA (externally observable).
/// - `read_only`: stores are forbidden (compile error).
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextPolicy {
    pub volatile: bool,
    pub read_only: bool,
}

// ── Compilation graph ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
}
