use acvus_orchestration::{
    ApiKind, GenerationParams, LlmSpec, MaxTokens, MessageSpec, NodeKind, NodeSpec, PlainSpec,
    SelfSpec, Strategy, TokenBudget, ToolBinding,
};
use acvus_utils::Interner;
use rust_decimal::Decimal;
use serde::Deserialize;

/// JSON-deserializable node definition from the web UI.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebNode {
    pub name: String,
    pub kind: String,
    pub api: String,
    pub model: String,
    pub temperature: Decimal,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub grounding: bool,
    pub max_tokens: WebMaxTokens,
    pub self_spec: WebSelfSpec,
    pub strategy: WebStrategy,
    pub retry: u32,
    pub assert: String,
    pub messages: Vec<WebMessage>,
    pub tools: Vec<WebToolBinding>,
}

#[derive(Deserialize)]
pub struct WebMaxTokens {
    pub input: u32,
    pub output: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSelfSpec {
    pub initial_value: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum WebStrategy {
    Always,
    #[serde(rename = "once-per-turn")]
    OncePerTurn,
    #[serde(rename = "if-modified", rename_all = "camelCase")]
    IfModified {
        key: String,
    },
    #[serde(rename = "history", rename_all = "camelCase")]
    History {
        history_bind: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum WebMessage {
    #[serde(rename_all = "camelCase")]
    Block { role: String, template: String },
    #[serde(rename_all = "camelCase")]
    Iterator {
        iterator: String,
        role: Option<String>,
        slice: Option<Vec<i64>>,
        token_budget: Option<WebTokenBudget>,
    },
}

#[derive(Deserialize)]
pub struct WebTokenBudget {
    pub priority: u32,
    pub min: Option<u32>,
    pub max: Option<u32>,
}

#[derive(Deserialize)]
pub struct WebToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: Vec<WebToolParam>,
}

#[derive(Deserialize)]
pub struct WebToolParam {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
}

pub fn convert_node(interner: &Interner, web: &WebNode) -> Result<NodeSpec, String> {
    let kind = match web.kind.as_str() {
        "llm" => NodeKind::Llm(LlmSpec {
            // Fallback to OpenAI when api is empty/unknown — ApiKind is only used at
            // runtime for LLM calls, not during typechecking. This allows nodes with
            // an unset provider to still participate in type analysis.
            api: ApiKind::parse(&web.api).unwrap_or(ApiKind::OpenAI),
            provider: String::new(),
            model: web.model.clone(),
            messages: web
                .messages
                .iter()
                .map(|m| match m {
                    WebMessage::Block { role, template } => MessageSpec::Block {
                        role: interner.intern(role),
                        source: template.clone(),
                    },
                    WebMessage::Iterator {
                        iterator,
                        role,
                        slice,
                        token_budget,
                    } => MessageSpec::Iterator {
                        key: interner.intern(iterator),
                        slice: slice.clone(),
                        role: role.as_ref().map(|r| interner.intern(r)),
                        token_budget: token_budget.as_ref().map(|tb| TokenBudget {
                            priority: tb.priority,
                            min: tb.min,
                            max: tb.max,
                        }),
                    },
                })
                .collect(),
            tools: web
                .tools
                .iter()
                .map(|t| ToolBinding {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    node: t.node.clone(),
                    params: t
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.ty.clone()))
                        .collect(),
                })
                .collect(),
            generation: GenerationParams {
                temperature: Some(web.temperature),
                top_p: web.top_p,
                top_k: web.top_k,
                grounding: web.grounding,
            },
            cache_key: None,
            max_tokens: MaxTokens {
                input: Some(web.max_tokens.input),
                output: Some(web.max_tokens.output),
            },
        }),
        _ => NodeKind::Plain(PlainSpec {
            source: web
                .messages
                .iter()
                .filter_map(|m| match m {
                    WebMessage::Block { template, .. } => Some(template.as_str()),
                    _ => None,
                })
                .next()
                .unwrap_or("")
                .to_string(),
        }),
    };

    let strategy = match &web.strategy {
        WebStrategy::Always => Strategy::Always,
        WebStrategy::OncePerTurn => Strategy::OncePerTurn,
        WebStrategy::IfModified { key } => Strategy::IfModified {
            key: interner.intern(key),
        },
        WebStrategy::History { history_bind } => Strategy::History {
            history_bind: interner.intern(history_bind),
        },
    };

    Ok(NodeSpec {
        name: interner.intern(&web.name),
        kind,
        self_spec: SelfSpec {
            initial_value: web
                .self_spec
                .initial_value
                .as_ref()
                .map(|s| interner.intern(s)),
        },
        strategy,
        retry: web.retry,
        assert: if web.assert.trim().is_empty() {
            None
        } else {
            Some(interner.intern(&web.assert))
        },
    })
}
