//! Type definitions for the compilation graph.

use rustc_hash::FxHashMap;
use acvus_utils::Astr;

use crate::ty::Ty;

// ── Identifiers ──────────────────────────────────────────────────────

/// Global unique identifier for a context variable.
/// Assigned by lowering. MIR only sees the Id, never the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(pub u32);

/// Global unique identifier for a compilation unit or extern declaration.
/// Assigned by lowering. Shared ID space between Unit and Extern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitId(pub u32);

/// Identifier for a scope within the compilation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

/// Identifier for a namespace in the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamespaceId(pub u32);

// ── Compilation units ────────────────────────────────────────────────

/// Source kind: script (expression) or template (text with interpolation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Script,
    Template,
}

/// A compilation unit — has source, gets compiled, produces a Coroutine at runtime.
#[derive(Debug, Clone)]
pub struct CompilationUnit {
    pub id: UnitId,
    pub source: Astr,
    pub kind: SourceKind,
    /// AST @name → ContextId mapping for this unit's namespace.
    pub name_to_id: FxHashMap<Astr, ContextId>,
    /// If this unit's output should unify with a ScopeLocal variable,
    /// this is that variable's ContextId. Lowering declares this.
    /// e.g., init → @self, bind → @self.
    pub output_binding: Option<ContextId>,
}

/// An extern declaration — no source, declared input/output types.
/// Runtime provides the value (e.g., LLM API call).
/// Produces a Coroutine at runtime, just like a Unit.
#[derive(Debug, Clone)]
pub struct ExternDecl {
    pub id: UnitId,
    /// Units this extern consumes, with expected types.
    /// Used to validate that consumed unit output types match expectations.
    /// (e.g., assert unit → Bool, message unit → String)
    pub inputs: Vec<(UnitId, Ty)>,
    /// The type this extern produces.
    pub output_ty: Ty,
}

// ── Scopes and bindings ──────────────────────────────────────────────

/// A scope groups units that share a namespace.
/// The graph engine discovers SCCs within scopes.
#[derive(Debug, Clone)]
pub struct Scope {
    pub id: ScopeId,
    pub units: Vec<UnitId>,
    pub bindings: Vec<ContextBinding>,
}

/// A binding: ContextId → where does the type come from + optional constraint.
#[derive(Debug, Clone)]
pub struct ContextBinding {
    pub id: ContextId,
    pub source: ContextSource,
    /// Structural constraint on this binding (e.g., Sequence<β, O, Pure>).
    /// Applied before SCC unification. Without it, types may widen to
    /// the lattice top (e.g., Iterator instead of Sequence).
    pub constraint: Option<Ty>,
}

/// Where a context variable's type comes from.
#[derive(Debug, Clone)]
pub enum ContextSource {
    /// From another unit/extern's output, with optional type transform.
    Derived(UnitId, TypeTransform),
    /// Shared within this scope — graph engine determines SCC membership.
    ScopeLocal,
    /// Dynamic external — runtime provides the value. Always unknown for reachability.
    External,
}

/// Type transform applied to a unit's output type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTransform {
    /// No transform — use the output type as-is.
    Identity,
    /// Extract element type from a collection. Delegates to Ty::elem_of().
    ElemOf,
}

// ── External types ───────────────────────────────────────────────────

/// Type of an external (dynamic) context variable.
#[derive(Debug, Clone)]
pub enum ExternalType {
    /// User specified the type explicitly.
    Known(Ty),
    /// Type unknown — analysis pass may infer it.
    Infer,
}

// ── Namespace ────────────────────────────────────────────────────────

/// A namespace in the hierarchy. Names are resolved within namespaces.
#[derive(Debug, Clone)]
pub struct Namespace {
    pub id: NamespaceId,
    pub parent: Option<NamespaceId>,
    pub path: Astr,
}

/// Maps ContextId → (NamespaceId, Name) for display/error purposes.
#[derive(Debug, Clone)]
pub struct ContextIdTable {
    entries: Vec<(NamespaceId, Astr)>,
}

impl ContextIdTable {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn insert(&mut self, id: ContextId, namespace: NamespaceId, name: Astr) {
        let idx = id.0 as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, (NamespaceId(0), name));
        }
        self.entries[idx] = (namespace, name);
    }

    pub fn get(&self, id: ContextId) -> Option<(NamespaceId, Astr)> {
        self.entries.get(id.0 as usize).copied()
    }
}

// ── Compilation graph ────────────────────────────────────────────────

/// The full compilation graph — input to the graph engine.
/// Produced by lowering. Contains no orchestration concepts.
#[derive(Debug, Clone)]
pub struct CompilationGraph {
    pub units: Vec<CompilationUnit>,
    pub externs: Vec<ExternDecl>,
    pub scopes: Vec<Scope>,
    pub externals: FxHashMap<ContextId, ExternalType>,
    pub id_table: ContextIdTable,
}
