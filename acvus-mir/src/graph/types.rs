//! Type definitions for the new compilation graph.
//!
//! Key differences from `graph::types`:
//! - Entity split into Function (executable) and Context (loadable).
//! - FnConstraint bundles signature + output.
//! - No membership — PHI/SSA handles type unification.
//! - Function has name instead of name_to_id — graph owns the name→id mapping.

use acvus_utils::Astr;
use acvus_utils::Freeze;

use crate::ty::Ty;

// ── Identifiers ─────────────────────────────────────────────────────

acvus_utils::declare_id!(pub ContextId);
acvus_utils::declare_id!(pub FunctionId);
acvus_utils::declare_id!(pub VersionId);
acvus_utils::declare_id!(pub ScopeId);
acvus_utils::declare_id!(pub NamespaceId);

// ── Source ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Script,
    Template,
}

#[derive(Debug, Clone)]
pub struct SourceCode {
    pub name: Astr,
    pub source: Astr,
    pub kind: SourceKind,
}

// ── Constraint ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type inferred from source.
    Inferred,
    /// Exact declared type.
    Exact(Ty),
    /// Type derived from a function's output type.
    DerivedFnOutput(FunctionId, TypeTransform),
    /// Type derived from a context's type.
    DerivedContext(ContextId, TypeTransform),
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
}

/// Declared dependency of an extern function.
/// The compiler trusts these hints for effect analysis and scheduling.
/// Violation at runtime (accessing undeclared dependencies) is a soundness
/// violation and must panic.
#[derive(Debug, Clone)]
pub enum DependencyHint {
    /// Depends on another function's output.
    Function(FunctionId, Ty),
    /// Reads a context value.
    ContextRead(ContextId, Ty),
    /// Writes a context value.
    ContextWrite(ContextId, Ty),
}

#[derive(Debug, Clone)]
pub enum FnKind {
    /// Has source code. Graph engine typechecks and compiles.
    Local(SourceCode),
    /// Black box. Runtime provides the value.
    /// `deps` declares dependencies for effect analysis. Undeclared access = panic.
    Extern { deps: Freeze<Vec<DependencyHint>> },
}

/// An executable entity in the graph.
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub name: Astr,
    pub kind: FnKind,
    pub constraint: FnConstraint,
}

// ── Context ──────────────────────────────────────────────────────────

/// A loadable value in the graph. Injected externally or derived from a function.
#[derive(Debug, Clone)]
pub struct Context {
    pub id: ContextId,
    pub name: Astr,
    pub constraint: Constraint,
}

// ── Compilation graph ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
}
