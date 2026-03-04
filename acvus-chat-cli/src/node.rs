use std::collections::HashMap;
use std::path::Path;

use acvus_orchestration::{
    ApiKind, GenerationParams, HistorySpec, LlmCacheSpec, LlmSpec, MaxTokens, MessageSpec,
    NodeKind, NodeSpec, PlainSpec, Strategy, StrategyMode, TokenBudget, ToolBinding,
};
use serde::Deserialize;

/// TOML-deserializable node definition.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum NodeKindDef {
    Plain,
    Llm,
    LlmCache {
        ttl: String,
        #[serde(default)]
        cache_config: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeDef {
    pub name: String,
    #[serde(flatten)]
    kind: NodeKindDef,
    // Plain node: top-level template
    template: Option<String>,
    inline_template: Option<String>,
    // Llm/LlmCache (None for Plain)
    provider: Option<String>,
    model: Option<String>,
    #[serde(default)]
    max_tokens: MaxTokensDef,
    #[serde(default)]
    tools: Vec<ToolBindingDef>,
    #[serde(default)]
    messages: Vec<MessageDef>,
    #[serde(default)]
    strategy: StrategyDef,
    #[serde(default)]
    generation: GenerationParamsDef,
    cache_key: Option<String>,
    history: Option<HistoryDef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HistoryDef {
    pub store: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
enum StrategyModeDef {
    #[default]
    Always,
    IfModified,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct StrategyDef {
    #[serde(default)]
    mode: StrategyModeDef,
    bind: Option<String>,
}

/// Serde message entry — tried as Iterator first (untagged).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum MessageDef {
    Iterator {
        iterator: String,
        #[serde(default)]
        slice: Option<Vec<i64>>,
        role: Option<String>,
        token_budget: Option<TokenBudgetDef>,
    },
    Block {
        role: String,
        template: Option<String>,
        inline_template: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct TokenBudgetDef {
    priority: u32,
    min: Option<u32>,
    max: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct MaxTokensDef {
    input: Option<u32>,
    output: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct GenerationParamsDef {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    #[serde(default)]
    grounding: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolBindingDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    node: String,
    #[serde(default)]
    params: HashMap<String, String>,
}

/// Resolve a template: file path or inline source → source string.
fn resolve_template(
    base_dir: &Path,
    file: Option<&str>,
    inline: Option<&str>,
) -> Result<String, String> {
    match (file, inline) {
        (Some(path), _) => {
            let full = base_dir.join(path);
            std::fs::read_to_string(&full)
                .map_err(|e| format!("failed to load template '{}': {e}", full.display()))
        }
        (None, Some(src)) => Ok(src.to_string()),
        (None, None) => Err("no template or inline_template provided".into()),
    }
}

/// Convert a TOML `NodeDef` into a pure `NodeSpec`, reading template files from `base_dir`.
pub fn resolve_node(
    def: NodeDef,
    base_dir: &Path,
    provider_apis: &HashMap<String, ApiKind>,
) -> Result<NodeSpec, String> {
    let mut messages = Vec::new();
    for (i, msg) in def.messages.into_iter().enumerate() {
        match msg {
            MessageDef::Block {
                role,
                template,
                inline_template,
            } => {
                let source =
                    resolve_template(base_dir, template.as_deref(), inline_template.as_deref())
                        .map_err(|e| format!("message {i}: {e}"))?;
                messages.push(MessageSpec::Block { role, source });
            }
            MessageDef::Iterator {
                iterator,
                slice,
                role,
                token_budget,
            } => {
                messages.push(MessageSpec::Iterator {
                    key: iterator,
                    slice,
                    role,
                    token_budget: token_budget.map(|tb| TokenBudget {
                        priority: tb.priority,
                        min: tb.min,
                        max: tb.max,
                    }),
                });
            }
        }
    }

    let strategy = Strategy {
        mode: match def.strategy.mode {
            StrategyModeDef::Always => StrategyMode::Always,
            StrategyModeDef::IfModified => StrategyMode::IfModified,
        },
        bind_source: def
            .strategy
            .bind
            .map(|path| {
                let full = base_dir.join(&path);
                std::fs::read_to_string(&full)
                    .map_err(|e| format!("failed to load strategy bind '{}': {e}", full.display()))
            })
            .transpose()?,
    };

    let generation = GenerationParams {
        temperature: def.generation.temperature,
        top_p: def.generation.top_p,
        top_k: def.generation.top_k,
        grounding: def.generation.grounding,
    };

    let max_tokens = MaxTokens {
        input: def.max_tokens.input,
        output: def.max_tokens.output,
    };

    let tools: Vec<ToolBinding> = def
        .tools
        .into_iter()
        .map(|t| ToolBinding {
            name: t.name,
            description: t.description,
            node: t.node,
            params: t.params,
        })
        .collect();

    let cache_key = def.cache_key;

    let kind = match def.kind {
        NodeKindDef::Plain => {
            let source = resolve_template(
                base_dir,
                def.template.as_deref(),
                def.inline_template.as_deref(),
            )
            .map_err(|e| format!("plain node '{}': {e}", def.name))?;
            NodeKind::Plain(PlainSpec { source })
        }
        NodeKindDef::Llm => {
            let provider = def
                .provider
                .ok_or_else(|| format!("node '{}': llm requires 'provider'", def.name))?;
            let model = def
                .model
                .ok_or_else(|| format!("node '{}': llm requires 'model'", def.name))?;
            let api = provider_apis
                .get(&provider)
                .ok_or_else(|| format!("node '{}': unknown provider '{provider}'", def.name))?
                .clone();
            NodeKind::Llm(LlmSpec {
                api,
                provider,
                model,
                messages,
                tools,
                generation,
                cache_key,
                max_tokens,
            })
        }
        NodeKindDef::LlmCache { ttl, cache_config } => {
            let provider = def
                .provider
                .ok_or_else(|| format!("node '{}': llm-cache requires 'provider'", def.name))?;
            let model = def
                .model
                .ok_or_else(|| format!("node '{}': llm-cache requires 'model'", def.name))?;
            let api = provider_apis
                .get(&provider)
                .ok_or_else(|| format!("node '{}': unknown provider '{provider}'", def.name))?
                .clone();
            NodeKind::LlmCache(LlmCacheSpec {
                api,
                provider,
                model,
                messages,
                ttl,
                cache_config,
            })
        }
    };

    Ok(NodeSpec {
        name: def.name,
        kind,
        strategy,
        history: def.history.map(|h| HistorySpec { store: h.store }),
    })
}
