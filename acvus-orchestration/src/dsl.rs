use crate::kind::NodeKind;

/// History specification for a node.
#[derive(Debug, Clone)]
pub struct HistorySpec {
    /// Script expression to evaluate and store each turn.
    pub store: String,
}

/// Node specification — pure compilation input, no Serde.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub name: String,
    pub kind: NodeKind,
    pub strategy: Strategy,
    pub history: Option<HistorySpec>,
}

#[derive(Debug, Clone, Default)]
pub enum StrategyMode {
    #[default]
    Always,
    IfModified,
}

#[derive(Debug, Clone, Default)]
pub struct Strategy {
    pub mode: StrategyMode,
    /// Script source for bind value (if-modified only).
    pub bind_source: Option<String>,
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
