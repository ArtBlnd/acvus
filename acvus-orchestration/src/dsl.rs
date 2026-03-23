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

/// Strategy — execution + persistency + retry + assert.
#[derive(Debug, Clone)]
pub struct Strategy {
    pub execution: Execution,
    pub persistency: Persistency,
    /// Maximum retry count on RuntimeError. 0 = no retry.
    pub retry: u32,
    /// Assert script (must evaluate to Bool). If false, triggers retry.
    pub assert: Astr,
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
            let has_self = matches!(
                self.strategy.persistency,
                Persistency::Sequence { .. } | Persistency::Patch { .. }
            );
            match scope {
                ContextScope::InitialValue => {
                    // No @self, no @raw
                }
                ContextScope::Body => {
                    if has_self {
                        extra.push((interner.intern(KEY_SELF), locals.self_ty.clone()));
                    }
                }
                ContextScope::Bind => {
                    if has_self {
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
}

/// Persistency mode — determines @self existence and storage location.
///
/// - Ephemeral: no @self, no bind, no init. Stateless.
/// - Sequence/Patch: @self exists. bind + initial_value required (in variant).
///   Storage persists across turns.
#[derive(Debug, Clone, Default)]
pub enum Persistency {
    /// No @self. Stateless.
    #[default]
    Ephemeral,
    /// @self in storage. Append semantics.
    Sequence { initial_value: Astr, bind: Astr },
    /// @self in storage. Diff-based overwrite.
    Patch { initial_value: Astr, bind: Astr },
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
