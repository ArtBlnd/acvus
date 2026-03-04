use std::collections::{BTreeMap, HashMap};

use acvus_mir::ty::Ty;

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain {
        source: String,
    },
    Llm {
        provider: String,
        model: String,
        messages: Vec<MessageSpec>,
        tools: Vec<ToolBinding>,
        generation: GenerationParams,
        cache_key: Option<String>,
        /// Total input token budget shared across budgeted iterators.
        max_tokens: Option<u32>,
    },
    LlmCache {
        provider: String,
        model: String,
        messages: Vec<MessageSpec>,
        /// TTL string, e.g. "300s", "1h".
        ttl: String,
        /// Provider-specific cache config (e.g. display_name for Gemini).
        cache_config: HashMap<String, serde_json::Value>,
    },
}

impl NodeKind {
    /// The output type produced by this node kind.
    pub fn output_ty(&self) -> Ty {
        match self {
            NodeKind::Plain { .. } => Ty::String,
            NodeKind::Llm { .. } => Ty::Object(BTreeMap::from([
                ("role".into(), Ty::String),
                ("content".into(), Ty::String),
                ("content_type".into(), Ty::String),
            ])),
            NodeKind::LlmCache { .. } => Ty::String,
        }
    }
}

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

/// Generation parameters for model calls.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub grounding: bool,
}

/// Tool binding — binds a tool name to a target node with typed parameters.
#[derive(Debug, Clone)]
pub struct ToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: HashMap<String, String>,
}
