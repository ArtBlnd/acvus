//! Type definitions for the compilation graph.
//!
//! Functions and Contexts are identified by `QualifiedRef` (namespace + name).
//! No opaque IDs - the name IS the identity.
//!
//! MIR receives **parsed ASTs**, not source strings. Parsing happens outside.

use std::convert::Infallible;

use acvus_utils::{Astr, Freeze};
use rustc_hash::FxHashMap;

use super::bind::{BindingRefused, BoundValue};
use crate::ty::{EffectTerm, IdentityTerm, Poly, PolyTy, Ty, TypeRegistry};

// -- Identifiers -----------------------------------------------------

acvus_utils::declare_id!(pub VersionId);
acvus_utils::declare_id!(pub ScopeId);

// Re-export from acvus-utils.
pub use acvus_utils::QualifiedRef;

// -- Function --------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FnKind {
    /// Has a parsed AST. MIR typechecks and compiles.
    Local(ParsedAst, Inputs),
    /// Black box. Runtime provides the value. `bounds[i]` is the declared
    /// bound of type variable `i` of the function's type, `effect_bounds[i]`
    /// of effect variable `i`.
    Extern {
        bounds: Vec<crate::ty::TyVarBound>,
        effect_bounds: Vec<crate::ty::EffectVarBound>,
        instances: crate::ty::Instances,
        requires: Vec<crate::ty::RequirementSig>,
    },
}

/// What a `$` a local function's body reads may name, beside the parameters
/// its type declares (RFC-0054 rule 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inputs {
    /// A declared parameter or a binding, and nothing else; a type that
    /// declares no parameter declares none.
    Declared,
    /// Also any other name, which becomes one more parameter, after the
    /// declared ones and in the order the body first reads it.
    FromReads,
}

/// Parsed AST for local functions.
#[derive(Debug, Clone)]
pub enum ParsedAst {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
    /// A source that did not parse, as the parse recovered it. It is checked
    /// for what parsed and never lowered (RFC-0078); its parse errors are
    /// the caller's to report.
    Recovered(RecoveredAst),
}

#[derive(Debug, Clone)]
pub enum RecoveredAst {
    Script(acvus_ast::Script<acvus_ast::ErrorNode>),
    Template(acvus_ast::Template<acvus_ast::ErrorNode>),
}

/// A source as its parse left it. The batch path and an editor both build
/// a function's AST through this, so both report the same errors for the
/// same source (RFC-0078 rule 7).
#[derive(Debug, Clone)]
pub struct Parsed {
    pub ast: ParsedAst,
    pub errors: Vec<acvus_ast::ParseError>,
}

impl Parsed {
    pub fn script(
        parsed: Result<
            acvus_ast::Script,
            acvus_ast::Recovered<acvus_ast::Script<acvus_ast::ErrorNode>>,
        >,
    ) -> Self {
        match parsed {
            Ok(script) => Self {
                ast: ParsedAst::Script(script),
                errors: Vec::new(),
            },
            Err(recovered) => Self {
                ast: ParsedAst::Recovered(RecoveredAst::Script(recovered.tree)),
                errors: recovered.errors,
            },
        }
    }

    pub fn template(
        parsed: Result<
            acvus_ast::Template,
            acvus_ast::Recovered<acvus_ast::Template<acvus_ast::ErrorNode>>,
        >,
    ) -> Self {
        match parsed {
            Ok(template) => Self {
                ast: ParsedAst::Template(template),
                errors: Vec::new(),
            },
            Err(recovered) => Self {
                ast: ParsedAst::Recovered(RecoveredAst::Template(recovered.tree)),
                errors: recovered.errors,
            },
        }
    }
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
    /// The local function whose result is the context's first value
    /// (RFC-0090 rule 1). `infer` checks its body against `ty` with every
    /// identity open, so the source the init makes is the value's.
    pub init: Option<QualifiedRef>,
}

impl Context {
    /// Whether the graph solves this context's type (RFC-0090 rule 1): its
    /// type holds a type, length or representation variable. An identity or
    /// effect variable is not one, since a declared type is lifted with a
    /// variable at every identity (`lift_declaration`).
    pub fn is_open(&self) -> bool {
        self.ty
            .try_map::<Poly, ()>(
                &mut |_| Err(()),
                &mut |var| Ok(IdentityTerm::Var(var)),
                &mut |var| Ok(EffectTerm::Var(var)),
                &mut |_| Err(()),
                &mut |_| Err(()),
                &mut |never: Infallible| match never {},
            )
            .is_err()
    }
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
/// supply (RFC-0071 rule 5). Every value held here types on its own
/// (RFC-0087 rule 3).
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    by_name: FxHashMap<Astr, BoundValue>,
}

impl Bindings {
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Refuses a value that has no type, before any body is checked
    /// (RFC-0087 rule 3).
    pub fn bind(&mut self, name: Astr, value: BoundValue) -> Result<(), BindingRefused> {
        super::bind::admit(&value)?;
        self.by_name.insert(name, value);
        Ok(())
    }

    pub fn unbind(&mut self, name: Astr) {
        self.by_name.remove(&name);
    }

    pub fn get(&self, name: Astr) -> Option<&BoundValue> {
        self.by_name.get(&name)
    }

    /// By name, so that what two runs of one compilation write is one
    /// program: the order this yields is the order the constants enter the
    /// body.
    pub fn iter(&self) -> impl Iterator<Item = (Astr, &BoundValue)> {
        let mut held: Vec<(Astr, &BoundValue)> = self
            .by_name
            .iter()
            .map(|(name, value)| (*name, value))
            .collect();
        held.sort_by_key(|(name, _)| name.bits());
        held.into_iter()
    }
}

// -- Context policy --------------------------------------------------

// -- Compilation graph -----------------------------------------------

/// How a host reaches the storage behind its contexts, declared when it
/// compiles (RFC-0090 rule 3). Under `Async` every access of a context may
/// wait, so a body that touches one is `Async` (RFC-0046); under `Sync` an
/// access returns at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Sync,
    Async,
}

#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub functions: Freeze<Vec<Function>>,
    pub contexts: Freeze<Vec<Context>>,
    pub types: Freeze<TypeRegistry>,
    pub bindings: Bindings,
    pub access: Access,
    /// Obligation across artifacts: each of these bodies' results crosses to
    /// the host, which `acvus_interpreter::Interpreter::execute` and
    /// `acvus_interpreter::Output` read, and RFC-0054 fixes each one's
    /// declared return. A host that runs an initializer and the scripts that
    /// read what it stored compiles them into one graph with an entry each
    /// (RFC-0090 rule 1). No entry is a graph no host starts, compiled to be
    /// lowered, printed or diagnosed.
    pub entries: Vec<QualifiedRef>,
}
