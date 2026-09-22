//! Type definitions for the compilation graph.
//!
//! Functions and Contexts are identified by `QualifiedRef` (namespace + name).
//! No opaque IDs - the name IS the identity.
//!
//! MIR receives **parsed ASTs**, not source strings. Parsing happens outside.

use acvus_ast::Literal;
use acvus_utils::{Astr, Freeze};
use rustc_hash::FxHashMap;

use crate::ty::{Mutability, PolyTy, Ty, TyTerm, TypeArg};

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
    /// bound of variable `i` of the function's type.
    Extern {
        bounds: Vec<crate::ty::TyVarBound>,
        instances: crate::ty::Instances,
        requires: Vec<crate::ty::RequirementSig>,
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

// -- Inputs -----------------------------------------------------------

/// A name a host must inject for this compilation to run: a context `@name`,
/// or a `$name` no binding fixed.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextInfo {
    pub name: QualifiedRef,
    pub ty: Ty,
}

// -- Bindings ---------------------------------------------------------

/// The `$` names a host fixed before this compilation: each is a constant in
/// every body that reads it, and none of them is an input the host must still
/// supply (RFC-0071 Decision 5).
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    by_name: FxHashMap<Astr, Literal>,
}

impl Bindings {
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn bind(&mut self, name: Astr, value: Literal) {
        self.by_name.insert(name, value);
    }

    pub fn unbind(&mut self, name: Astr) {
        self.by_name.remove(&name);
    }

    pub fn get(&self, name: Astr) -> Option<&Literal> {
        self.by_name.get(&name)
    }

    /// By name, so that what two runs of one compilation write is one
    /// program: the order this yields is the order the constants enter the
    /// body.
    pub fn iter(&self) -> impl Iterator<Item = (Astr, &Literal)> {
        let mut held: Vec<(Astr, &Literal)> = self
            .by_name
            .iter()
            .map(|(name, value)| (*name, value))
            .collect();
        held.sort_by_key(|(name, _)| name.bits());
        held.into_iter()
    }
}

/// The type a bound `$` is checked at. Text is `&str` rather than `String`
/// because the constant lowering writes for it is the one a string literal
/// writes, so a bound `$mode` stands wherever `"review"` stands.
pub fn bound_input_ty(value: &Literal) -> Option<Ty> {
    match value.desugared() {
        Literal::Int(_) => Some(Ty::I64),
        Literal::Float(_) => Some(Ty::Float),
        Literal::Bool(_) => Some(Ty::Bool),
        Literal::Char(_) => Some(Ty::Char),
        Literal::String(_) => Some(Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(TyTerm::Str)),
        )),
        Literal::IntOf(_) | Literal::Bytes(_) | Literal::List(_) | Literal::Unit => None,
    }
}

// -- Context policy --------------------------------------------------

// -- Compilation graph -----------------------------------------------

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
    pub bindings: Bindings,
    /// Obligation across artifacts: this body's result crosses to the host
    /// as one `Value` read by kind, which `acvus_interpreter::Interpreter::
    /// execute` reads and RFC-0054 fixes. `None` is a graph no host starts,
    /// compiled to be lowered, printed or diagnosed.
    pub entry: Option<QualifiedRef>,
}
