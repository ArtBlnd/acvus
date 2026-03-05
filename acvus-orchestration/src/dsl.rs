use crate::kind::NodeKind;

/// Self specification — how to transform raw output into stored value.
///
/// Inside `self_bind`:
///   `@self` = previous stored value (or initial_value on first run)
///   `@raw`  = raw output from node kind
/// Result = new @self = value exposed as @name externally.
#[derive(Debug, Clone)]
pub struct SelfSpec {
    /// Script: @self(previous) + @raw(raw output) → new @self.
    pub self_bind: String,
    /// Script to produce the initial @self before any execution.
    pub initial_value: String,
}

/// Node specification — pure compilation input, no Serde.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub name: String,
    pub kind: NodeKind,
    pub self_spec: SelfSpec,
    pub strategy: Strategy,
}

/// Execution strategy — determines execution timing and @self storage location.
///
/// Context hierarchy:
///   turn_context  — per-turn. Empty at turn start, discarded at turn end.
///   storage       — persistent. Survives across turns.
#[derive(Debug, Clone, Default)]
pub enum Strategy {
    /// Execute every invocation. @self stored in turn_context, overwritten each time.
    Always,
    /// Execute once per turn. @self stored in storage (persistent).
    /// Next turn can reference previous @self.
    #[default]
    OncePerTurn,
    /// Execute only when key changes. @self stored in storage (persistent).
    /// Unchanged key → previous @self retained.
    IfModified { key: String },
    /// Execute once per turn. @self stored in storage (persistent).
    /// Evaluates history_bind (@self + other context → entry) and appends to @history.{name}.
    History { history_bind: String },
}

/// A message entry: either a template block or an iterator over a context key.
#[derive(Debug, Clone)]
pub enum MessageSpec {
    Block {
        role: String,
        source: String,
    },
    Iterator {
        key: String,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        slice: Option<Vec<i64>>,
        /// Override the role for all messages from this iterator.
        role: Option<String>,
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
