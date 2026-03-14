use acvus_orchestration::{
    ApiKind, ExprSpec, Execution, GenerationParams, LlmSpec, MaxTokens, MessageSpec, NodeKind,
    FnParam, NodeSpec, Persistency, PlainSpec, Strategy, ThinkingConfig, TokenBudget, ToolBinding,
    ToolParamInfo,
};
use acvus_utils::Interner;
use rust_decimal::Decimal;
use serde::Deserialize;

/// JSON-deserializable strategy definition from the web UI.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebStrategy {
    pub execution: WebExecution,
    #[serde(default)]
    pub persistency: WebPersistency,
    #[serde(default)]
    pub initial_value: Option<String>,
    #[serde(default)]
    pub retry: u32,
    #[serde(default)]
    pub assert: String,
}

/// JSON-deserializable node definition from the web UI.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebNode {
    pub name: String,
    pub strategy: WebStrategy,
    #[serde(default)]
    pub is_function: bool,
    #[serde(default)]
    pub fn_params: Vec<WebFnParam>,
    #[serde(flatten)]
    pub kind: WebNodeKind,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum WebNodeKind {
    #[serde(rename = "llm", rename_all = "camelCase")]
    Llm {
        #[serde(default)]
        api: Option<ApiKind>,
        model: String,
        temperature: Decimal,
        top_p: Option<Decimal>,
        top_k: Option<u32>,
        #[serde(default)]
        grounding: bool,
        thinking: Option<ThinkingConfig>,
        max_tokens: WebMaxTokens,
        #[serde(default)]
        messages: Vec<WebMessage>,
        #[serde(default)]
        tools: Vec<WebToolBinding>,
    },
    #[serde(rename = "expr", rename_all = "camelCase")]
    Expr {
        #[serde(default)]
        expr_source: String,
    },
    #[serde(rename = "plain")]
    Plain {},
}

#[derive(Deserialize)]
pub struct WebMaxTokens {
    pub input: u32,
    pub output: u32,
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum WebExecution {
    Always,
    #[serde(rename = "once-per-turn")]
    OncePerTurn,
    #[serde(rename = "if-modified", rename_all = "camelCase")]
    IfModified {
        key: String,
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
    pub description: Option<String>,
}

#[derive(Deserialize)]
pub struct WebFnParam {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub description: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum WebPersistency {
    #[default]
    Ephemeral,
    Snapshot,
    Deque { bind: String },
    Diff { bind: String },
}

impl WebNode {
    pub fn into_node(&self, interner: &Interner) -> Result<NodeSpec, String> {
        let initial_value = self.strategy.initial_value.as_ref().map(|s| interner.intern(s));

        let kind = match &self.kind {
            WebNodeKind::Llm {
                api,
                model,
                temperature,
                top_p,
                top_k,
                grounding,
                thinking,
                max_tokens,
                messages,
                tools,
            } => NodeKind::Llm(LlmSpec {
                // Fallback to OpenAI when unset — ApiKind is only used at runtime
                // for LLM calls, not during typechecking. Allows nodes with no
                // provider to still participate in type analysis.
                api: api.clone().unwrap_or(ApiKind::OpenAI),
                provider: String::new(),
                model: model.clone(),
                messages: messages
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
                tools: tools
                    .iter()
                    .map(|t| ToolBinding {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        node: t.node.clone(),
                        params: t
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), ToolParamInfo {
                                ty: p.ty.clone(),
                                description: p.description.clone(),
                            }))
                            .collect(),
                    })
                    .collect(),
                generation: GenerationParams {
                    temperature: Some(*temperature),
                    top_p: *top_p,
                    top_k: *top_k,
                    grounding: *grounding,
                    thinking: thinking.clone(),
                },
                cache_key: None,
                max_tokens: MaxTokens {
                    input: Some(max_tokens.input),
                    output: Some(max_tokens.output),
                },
            }),
            WebNodeKind::Expr {
                expr_source,
            } => NodeKind::Expr(ExprSpec {
                source: expr_source.clone(),
                output_ty: acvus_mir::ty::Ty::Infer,
            }),
            WebNodeKind::Plain {} => NodeKind::Plain(PlainSpec {
                source: String::new(),
            }),
        };

        let execution = match &self.strategy.execution {
            WebExecution::Always => Execution::Always,
            WebExecution::OncePerTurn => Execution::OncePerTurn,
            WebExecution::IfModified { key } => Execution::IfModified {
                key: interner.intern(key),
            },
        };

        let persistency = match &self.strategy.persistency {
            WebPersistency::Ephemeral => Persistency::Ephemeral,
            WebPersistency::Snapshot => Persistency::Snapshot,
            WebPersistency::Deque { bind } => Persistency::Deque { bind: interner.intern(bind) },
            WebPersistency::Diff { bind } => Persistency::Diff { bind: interner.intern(bind) },
        };

        Ok(NodeSpec {
            name: interner.intern(&self.name),
            kind,
            strategy: Strategy {
                execution,
                persistency,
                initial_value,
                retry: self.strategy.retry,
                assert: if self.strategy.assert.trim().is_empty() {
                    None
                } else {
                    Some(interner.intern(&self.strategy.assert))
                },
            },
            is_function: self.is_function,
            fn_params: self
                .fn_params
                .iter()
                .map(|p| {
                    let ty = crate::parse_type_string(&interner, &p.ty);
                    FnParam {
                        name: interner.intern(&p.name),
                        ty,
                        description: p.description.as_ref().map(|d| interner.intern(d)),
                    }
                })
                .collect(),
        })
    }
}
