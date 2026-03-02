use std::collections::HashMap;

use serde::Deserialize;

/// A context reference like `"@node-name"`. Strips the `@` prefix on deserialization.
#[derive(Debug, Clone)]
pub struct ContextRef(pub String);

impl<'de> Deserialize<'de> for ContextRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.strip_prefix('@') {
            Some(name) => Ok(ContextRef(name.to_string())),
            None => Err(serde::de::Error::custom(format!(
                "context reference must start with '@', got: {s}"
            ))),
        }
    }
}

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum NodeKind {
    Llm,
    LlmCache {
        /// TTL string, e.g. "300s", "1h".
        ttl: String,
        /// Provider-specific cache config (e.g. display_name for Gemini).
        #[serde(default)]
        cache_config: HashMap<String, serde_json::Value>,
    },
}

/// Node specification parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    #[serde(flatten)]
    pub kind: NodeKind,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    #[serde(default)]
    pub messages: Vec<MessageSpec>,
    pub strategy: Strategy,
    #[serde(default)]
    pub generation: GenerationParams,
    pub cache_key: Option<ContextRef>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Strategy {
    #[serde(default)]
    pub mode: StrategyMode,
    /// Template file for cache key (if-modified only).
    pub key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyMode {
    #[default]
    Always,
    IfModified,
}

/// A message entry: either a template block or an iterator over a storage key.
///
/// Iterator is tried first so that `{iterator, role, template}` matches Iterator, not Block.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageSpec {
    Iterator {
        iterator: ContextRef,
        template: Option<String>,
        inline_template: Option<String>,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        #[serde(default)]
        slice: Option<Vec<i64>>,
        /// Bind each item to this context name (e.g. `bind = "msg"` → `@msg`).
        /// Without bind, the legacy `@type`/`@text` injection is used.
        bind: Option<String>,
        /// Override the role for all messages from this iterator.
        role: Option<String>,
    },
    Block {
        role: String,
        template: Option<String>,
        inline_template: Option<String>,
    },
}

/// Generation parameters for model calls.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
}

/// Tool declaration.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDecl {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, String>,
}
