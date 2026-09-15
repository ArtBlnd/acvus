//! Type definitions for the compilation graph.
//!
//! Functions and Contexts are identified by `QualifiedRef` (namespace + name).
//! No opaque IDs - the name IS the identity.
//!
//! MIR receives **parsed ASTs**, not source strings. Parsing happens outside.

use acvus_utils::Freeze;

use crate::ty::PolyTy;

// -- Identifiers -----------------------------------------------------

acvus_utils::declare_id!(pub VersionId);
acvus_utils::declare_id!(pub ScopeId);

// Re-export from acvus-utils.
pub use acvus_utils::QualifiedRef;

// -- Function --------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FnKind {
    /// Has a parsed AST. MIR typechecks and compiles.
    Local(ParsedAst),
    /// Black box. Runtime provides the value. `bounds[i]` is the declared
    /// bound of variable `i` of the function's type; `instances` are the
    /// types of a shared signature's instances (RFC-0027).
    Extern {
        bounds: Vec<crate::ty::TyVarBound>,
        instances: Vec<crate::ty::PolyTy>,
    },
}

/// Parsed AST for local functions.
#[derive(Debug, Clone)]
pub enum ParsedAst {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

/// An executable entity in the graph. Identified by `QualifiedRef`.
///
/// `ty` is a `PolyTy` - typically `TyTerm::Fn { params, ret, captures, effect }`.
/// Unresolved parts use `Var(n)` placeholders (inferred by the solver).
#[derive(Debug, Clone)]
pub struct Function {
    /// Unique identity = namespace + name.
    pub qref: QualifiedRef,
    pub kind: FnKind,
    /// The function's polymorphic type (Fn { params, ret, captures, effect }).
    /// `Var` placeholders are inferred by the solver.
    pub ty: PolyTy,
}

// -- Context ----------------------------------------------------------

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

// -- Context policy --------------------------------------------------

// -- Compilation graph -----------------------------------------------

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
}
