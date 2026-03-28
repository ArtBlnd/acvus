//! Type definitions for the new compilation graph.
//!
//! Key differences from `graph::types`:
//! - Entity split into Function (executable) and Context (loadable).
//! - FnConstraint bundles signature + output.
//! - No membership — PHI/SSA handles type unification.
//! - Function has name instead of name_to_id — graph owns the name→id mapping.

use acvus_utils::Astr;
use acvus_utils::Freeze;

use crate::ty::{EffectConstraint, Ty};

// ── Identifiers ─────────────────────────────────────────────────────

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
}

#[derive(Debug, Clone)]
pub enum FnKind {
    /// Has source code. Graph engine typechecks and compiles.
    Local(SourceCode),
    /// Has pre-parsed AST. Skips parsing — used by lowerer-generated code.
    LocalAst(ParsedAst),
    /// Black box. Runtime provides the value.
    /// Effect information lives in the function's type (`Ty::Fn { effect }`).
    Extern,
}

/// Pre-parsed AST for lowerer-generated functions.
#[derive(Debug, Clone)]
pub enum ParsedAst {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

/// An executable entity in the graph.
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub name: Astr,
    /// Namespace this function belongs to. `None` = root (global).
    pub namespace: Option<NamespaceId>,
    pub kind: FnKind,
    pub constraint: FnConstraint,
}

// Re-export from acvus-utils.
pub use acvus_utils::QualifiedRef;

// ── Context ──────────────────────────────────────────────────────────

/// A loadable value in the graph. Injected externally or derived from a function.
/// Identity = QualifiedRef (namespace + name). No separate ContextId.
#[derive(Debug, Clone)]
pub struct Context {
    pub name: Astr,
    /// Namespace this context belongs to. `None` = root (global).
    pub namespace: Option<NamespaceId>,
    pub constraint: Constraint,
}

impl Context {
    /// The qualified reference that uniquely identifies this context.
    pub fn qualified_ref(&self) -> QualifiedRef {
        // Note: namespace here is NamespaceId, but QualifiedRef uses Astr.
        // The graph must resolve NamespaceId → Astr for this.
        // For root contexts (namespace=None), this is straightforward.
        QualifiedRef {
            namespace: None,
            name: self.name,
        }
    }
}

// ── Namespace ────────────────────────────────────────────────────────

/// A flat scope for qualified access.
/// - Unqualified `@name` / `func()` → resolves in root only.
/// - Qualified `@ns::name` / `ns::func()` → resolves in the named namespace only.
#[derive(Debug, Clone)]
pub struct Namespace {
    pub id: NamespaceId,
    pub name: Astr,
}

// ── Compilation graph ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub namespaces: Freeze<Vec<Namespace>>,
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
}
