use acvus_mir::context_registry::{ContextTypeRegistry, RegistryConflictError};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};

use crate::spec::NodeKind;

// ── Scope key constants ────────────────────────────────────────────
// These are the canonical names for node-local context keys.
// Using constants instead of inline string literals prevents typo-driven bugs
// and makes rename-refactoring possible.

/// `@self` — accumulated persistent value.
pub const KEY_SELF: &str = "self";
/// `@raw` — raw output of the body computation.
pub const KEY_RAW: &str = "raw";
/// `@bind` — bind key value (for IfModified).
pub const KEY_BIND: &str = "bind";
/// `@turn_index` — current turn number.
pub const KEY_TURN_INDEX: &str = "turn_index";
/// `@item` — current element in iterator.
pub const KEY_ITEM: &str = "item";
/// `@index` — current iteration index.
pub const KEY_INDEX: &str = "index";

// ── Message field constants ────────────────────────────────────────
// Canonical field names for the message Object { role, content, content_type }.

pub const MSG_ROLE: &str = "role";
pub const MSG_CONTENT: &str = "content";
pub const MSG_CONTENT_TYPE: &str = "content_type";

/// The canonical Ty for a single message element: `{ role: String, content: String, content_type: String }`.
///
/// This is the **single source of truth** for the message Object schema.
/// Used by LlmSpec::output_ty, content_to_value, and ApiKind::message_elem_ty.
pub fn message_elem_ty(interner: &Interner) -> Ty {
    Ty::Object(rustc_hash::FxHashMap::from_iter([
        (interner.intern(MSG_ROLE), Ty::String),
        (interner.intern(MSG_CONTENT), Ty::String),
        (interner.intern(MSG_CONTENT_TYPE), Ty::String),
    ]))
}

/// Strategy — groups execution, persistency, initial_value, retry, and assert.
#[derive(Debug, Clone)]
pub struct Strategy {
    pub execution: Execution,
    pub persistency: Persistency,
    /// Optional initial state. When Some, `@self` is available in the node body.
    pub initial_value: Option<Astr>,
    /// Maximum retry count on RuntimeError. 0 = no retry.
    pub retry: u32,
    /// Assert script (must evaluate to Bool). If false, triggers retry.
    pub assert: Option<Astr>,
}

/// A function parameter with name, type, and optional description.
#[derive(Debug, Clone)]
pub struct FnParam {
    pub name: Astr,
    pub ty: Ty,
    pub description: Option<Astr>,
}

/// Node specification — pure compilation input, no Serde.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub name: Astr,
    pub kind: NodeKind,
    pub strategy: Strategy,
    /// Whether this node is a function node.
    pub is_function: bool,
    /// Function parameters.
    pub fn_params: Vec<FnParam>,
}

/// Which script scope is being compiled/typechecked.
/// Determines which local variables (@self, @raw) are injected.
#[derive(Debug, Clone, Copy)]
pub enum ContextScope {
    /// initial_value: no @self, no @raw
    InitialValue,
    /// Node body (messages, expr, assert): @self if initial_value exists
    Body,
    /// Bind script (Sequence/Patch): @self if initial_value exists, + @raw
    Bind,
}

impl NodeSpec {
    /// Build the context type registry visible inside this node's scripts.
    ///
    /// Starts from `base` and adds fn_params, @self, @raw as scoped types
    /// based on `scope`.
    ///
    /// All node-internal context assembly MUST go through this method
    /// to prevent locals from being accidentally omitted or duplicated.
    pub fn build_node_context(
        &self,
        interner: &Interner,
        base: &ContextTypeRegistry,
        scope: ContextScope,
        locals: Option<&NodeLocalTypes>,
    ) -> Result<ContextTypeRegistry, RegistryConflictError> {
        let mut extra: Vec<(Astr, Ty)> = Vec::new();
        if self.is_function {
            for p in &self.fn_params {
                extra.push((p.name, p.ty.clone()));
            }
        }
        if let Some(locals) = locals {
            match scope {
                ContextScope::InitialValue => {
                    // No @self, no @raw
                }
                ContextScope::Body => {
                    // @self only if initial_value exists
                    if self.strategy.initial_value.is_some() {
                        extra.push((interner.intern(KEY_SELF), locals.self_ty.clone()));
                    }
                }
                ContextScope::Bind => {
                    // @self if initial_value exists + @raw always
                    if self.strategy.initial_value.is_some() {
                        extra.push((interner.intern(KEY_SELF), locals.self_ty.clone()));
                    }
                    extra.push((interner.intern(KEY_RAW), locals.raw_ty.clone()));
                }
            }
        }
        base.with_extra_scoped(extra)
    }
}

/// Per-node local types (@raw, @self) — computed once, used everywhere.
#[derive(Debug, Clone)]
pub struct NodeLocalTypes {
    /// @raw — raw output of the node (before bind)
    pub raw_ty: Ty,
    /// @self — determined by initial_value's return type
    pub self_ty: Ty,
}

/// Execution strategy — determines execution timing and @self storage location.
///
/// Context hierarchy:
///   turn_context  — per-turn. Empty at turn start, discarded at turn end.
///   storage       — persistent. Survives across turns.
#[derive(Debug, Clone, Default)]
pub enum Execution {
    /// Execute every invocation. @self stored in turn_context, overwritten each time.
    Always,
    /// Execute once per turn. @self stored in storage (persistent).
    /// Next turn can reference previous @self.
    #[default]
    OncePerTurn,
    /// Execute only when key changes. @self stored in storage (persistent).
    /// Unchanged key → previous @self retained.
    IfModified { key: Astr },
}

/// Persistency mode — determines how node output is persisted.
///
/// Every persistent mode requires a `bind` script that transforms the raw
/// node output (`@raw`) into the stored value. The bind result is diffed
/// against the previous value via `PatchDiff::compute` for history tracking.
///
/// - Simple overwrite: `Patch { bind: "@raw" }` (identity — stores @raw as-is)
/// - Accumulation: `Sequence { bind: "@self | chain(@raw | iter)" }`
/// - Partial update: `Patch { bind: "{count: @self.count + 1, ..@self}" }`
#[derive(Debug, Clone, Default)]
pub enum Persistency {
    /// Don't persist to storage.
    #[default]
    Ephemeral,
    /// Tracked sequence with diff-based updates. `bind` script transforms @raw → stored value.
    Sequence { bind: Astr },
    /// Recursive value patch. `bind` script transforms @raw → stored value.
    /// The result is diffed against the previous value for history tracking.
    Patch { bind: Astr },
}

/// A message entry: either a template block or an iterator over a context key.
#[derive(Debug, Clone)]
pub enum MessageSpec {
    Block {
        role: Astr,
        source: String,
    },
    Iterator {
        key: Astr,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        slice: Option<Vec<i64>>,
        /// Override the role for all messages from this iterator.
        role: Option<Astr>,
        /// Token budget for this iterator.
        token_budget: Option<TokenBudget>,
    },
}

/// Token budget for a single iterator.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Lower = fills first (0 is highest priority).
    pub priority: u32,
    /// Minimum guaranteed tokens (reserved from the shared pool).
    pub min: Option<u32>,
    /// Maximum tokens this iterator may use.
    pub max: Option<u32>,
}
