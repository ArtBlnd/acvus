//! Type definitions for the compilation graph.
//!
//! Functions and Contexts are identified by `QualifiedRef` (namespace + name).
//! No opaque IDs — the name IS the identity.
//!
//! MIR receives **parsed ASTs**, not source strings. Parsing happens outside.

use acvus_utils::Freeze;

use crate::ty::{EffectConstraint, PolyTy};

// ── Identifiers ─────────────────────────────────────────────────────

acvus_utils::declare_id!(pub VersionId);
acvus_utils::declare_id!(pub ScopeId);

// Re-export from acvus-utils.
pub use acvus_utils::QualifiedRef;

// ── Function ────────────────────────────────────────────────────────

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
///
/// `ty` is a `PolyTy` — typically `TyTerm::Fn { params, ret, captures, effect, hint }`.
/// Unresolved parts use `Var(n)` placeholders (inferred by the solver).
#[derive(Debug, Clone)]
pub struct Function {
    /// Unique identity = namespace + name.
    pub qref: QualifiedRef,
    pub kind: FnKind,
    /// The function's polymorphic type (Fn { params, ret, captures, effect, hint }).
    /// `Var` placeholders are inferred by the solver.
    pub ty: PolyTy,
    /// Effect upper bound. `None` = no constraint (anything allowed).
    /// Checked post-inference: body effect must not exceed this bound.
    pub effect_constraint: Option<EffectConstraint>,
}

// ── Context ──────────────────────────────────────────────────────────

/// A loadable value in the graph. Injected externally or derived from a function.
/// Identified by `QualifiedRef` (namespace + name).
///
/// `ty` is a `PolyTy`. If the type is unknown (to be inferred), use a `Var` placeholder.
#[derive(Debug, Clone)]
pub struct Context {
    /// Unique identity = namespace + name.
    pub qref: QualifiedRef,
    /// The context's polymorphic type. `Var` = to be inferred.
    pub ty: PolyTy,
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
