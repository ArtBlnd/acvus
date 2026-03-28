use acvus_orchestration::{
    AnthropicSpec, Execution, ExpressionSpec, FnParam, GoogleAICacheSpec, GoogleAISpec, MaxTokens,
    MessageSpec, NodeKind, NodeSpec, OpenAICompatibleSpec, Persistency, PlainSpec, Strategy,
    ThinkingConfig, TokenBudget, ToolBinding, ToolParamInfo,
};
use acvus_utils::Interner;
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Config deserialization (JSON from JS → ChatSession.create)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct SessionConfig {
    pub nodes: Vec<NodeConfig>,
    pub providers: FxHashMap<String, ProviderConfigJson>,
    pub entrypoint: String,
    #[serde(default)]
    pub context: FxHashMap<String, ContextDecl>,
    #[serde(default)]
    pub asset_store_name: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct ProviderConfigJson {
    pub api: ApiKindJson,
    pub endpoint: String,
    pub api_key: String,
}

/// Provider API kind — local to pomollu-engine for deserializing config.
/// Used to dispatch which NodeKind variant to create.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiKindJson {
    OpenAI,
    Anthropic,
    Google,
}

#[derive(Deserialize)]
pub(crate) struct ContextDecl {
    #[serde(rename = "type")]
    pub ty: Option<crate::schema::TypeDesc>,
}

#[derive(Deserialize)]
pub(crate) struct StrategyConfig {
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub persistency: PersistencyConfig,
    pub initial_value: Option<String>,
    #[serde(default)]
    pub retry: u32,
    #[serde(default)]
    pub assert_script: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct NodeConfig {
    pub name: String,
    pub strategy: StrategyConfig,
    #[serde(default)]
    pub is_function: bool,
    #[serde(default)]
    pub fn_params: Vec<FnParamConfig>,
    #[serde(flatten)]
    pub kind: NodeKindConfig,
}

#[derive(Deserialize)]
pub(crate) struct FnParamConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub description: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub(crate) enum NodeKindConfig {
    #[serde(rename = "llm")]
    Llm {
        provider: String,
        api: ApiKindJson,
        model: String,
        temperature: Option<Decimal>,
        top_p: Option<Decimal>,
        top_k: Option<u32>,
        #[serde(default)]
        grounding: bool,
        thinking: Option<ThinkingConfig>,
        max_tokens: Option<MaxTokensJson>,
        messages: Vec<MessageConfig>,
        #[serde(default)]
        tools: Vec<ToolConfig>,
    },
    #[serde(rename = "plain")]
    Plain { template: String },
    #[serde(rename = "expr")]
    Expr {
        template: String,
        output_ty: Option<crate::schema::TypeDesc>,
    },
    #[serde(rename = "iterator")]
    Iterator {
        sources: Vec<IteratorSourceConfig>,
        #[serde(default)]
        unordered: bool,
    },
}

#[derive(Deserialize)]
pub(crate) struct IteratorSourceConfig {
    pub name: String,
    pub expr: String,
    #[serde(default)]
    pub entries: Vec<IteratorEntryConfig>,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct IteratorEntryConfig {
    #[serde(default)]
    pub condition: Option<String>,
    pub transform: TransformConfig,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub(crate) enum TransformConfig {
    Template { source: String },
    Script { source: String },
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ToolConfig {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: Vec<ToolParamConfigEntry>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ToolParamConfigEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub description: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct MaxTokensJson {
    pub input: Option<u32>,
    pub output: Option<u32>,
}

#[derive(Deserialize)]
pub(crate) struct MessageConfig {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub inline_template: Option<String>,
    #[serde(default)]
    pub iterator: Option<String>,
    #[serde(default)]
    pub slice: Option<Vec<i64>>,
    #[serde(default)]
    pub token_budget: Option<TokenBudgetConfig>,
}

#[derive(Deserialize)]
pub(crate) struct TokenBudgetConfig {
    pub priority: u32,
    #[serde(default)]
    pub min: Option<u32>,
    #[serde(default)]
    pub max: Option<u32>,
}

#[derive(Deserialize, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub(crate) enum PersistencyConfig {
    #[default]
    Ephemeral,
    Sequence {
        bind: String,
    },
    Patch {
        bind: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "mode")]
pub(crate) enum ExecutionConfig {
    #[serde(rename = "always")]
    Always,
    #[serde(rename = "once-per-turn")]
    OncePerTurn,
}

// ---------------------------------------------------------------------------
// NodeConfig → NodeSpec conversion
// ---------------------------------------------------------------------------

pub(crate) fn convert_node(
    interner: &Interner,
    cfg: &NodeConfig,
    providers: &FxHashMap<String, ProviderConfigJson>,
) -> Result<NodeSpec, String> {
    let kind = match &cfg.kind {
        NodeKindConfig::Llm {
            provider,
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
        } => {
            let provider_cfg = providers
                .get(provider)
                .ok_or_else(|| format!("node '{}': unknown provider '{provider}'", cfg.name))?;
            let endpoint = provider_cfg.endpoint.clone();
            let api_key = provider_cfg.api_key.clone();

            let messages: Vec<MessageSpec> = messages
                .iter()
                .filter_map(|m| {
                    if let Some(iter) = &m.iterator {
                        Some(MessageSpec::Iterator {
                            key: interner.intern(iter),
                            slice: m.slice.clone(),
                            role: m.role.as_ref().map(|r| interner.intern(r)),
                            token_budget: m.token_budget.as_ref().map(|tb| TokenBudget {
                                priority: tb.priority,
                                min: tb.min,
                                max: tb.max,
                            }),
                        })
                    } else {
                        let source = m.inline_template.as_ref().or(m.template.as_ref())?.clone();
                        Some(MessageSpec::Block {
                            role: m
                                .role
                                .as_ref()
                                .map(|r| interner.intern(r))
                                .unwrap_or_else(|| interner.intern("user")),
                            source,
                        })
                    }
                })
                .collect();

            let compiled_tools: Vec<ToolBinding> = tools
                .iter()
                .map(|t| ToolBinding {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    node: t.node.clone(),
                    params: t
                        .params
                        .iter()
                        .map(|p| {
                            (
                                p.name.clone(),
                                ToolParamInfo {
                                    ty: p.ty.clone(),
                                    description: p.description.clone(),
                                },
                            )
                        })
                        .collect(),
                })
                .collect();

            let max_tokens_val = max_tokens
                .as_ref()
                .map(|mt| MaxTokens {
                    input: mt.input,
                    output: mt.output,
                })
                .unwrap_or_default();

            match api {
                ApiKindJson::OpenAI => NodeKind::OpenAICompatible(OpenAICompatibleSpec {
                    endpoint,
                    api_key,
                    model: model.clone(),
                    messages,
                    tools: compiled_tools,
                    temperature: *temperature,
                    top_p: *top_p,
                    cache_key: None,
                    max_tokens: max_tokens_val,
                }),
                ApiKindJson::Anthropic => NodeKind::Anthropic(AnthropicSpec {
                    endpoint,
                    api_key,
                    model: model.clone(),
                    messages,
                    tools: compiled_tools,
                    temperature: *temperature,
                    top_p: *top_p,
                    top_k: *top_k,
                    max_tokens: max_tokens_val,
                    thinking: thinking.clone(),
                    cache_key: None,
                }),
                ApiKindJson::Google => NodeKind::GoogleAI(GoogleAISpec {
                    endpoint,
                    api_key,
                    model: model.clone(),
                    messages,
                    tools: compiled_tools,
                    temperature: *temperature,
                    top_p: *top_p,
                    top_k: *top_k,
                    max_tokens: max_tokens_val,
                    thinking: thinking.clone(),
                    grounding: *grounding,
                    cache_key: None,
                }),
            }
        }
        NodeKindConfig::Expr {
            template,
            output_ty,
        } => {
            let output_ty = output_ty
                .as_ref()
                .map(|desc| crate::desc_to_ty(interner, desc));
            NodeKind::Expression(ExpressionSpec {
                source: template.clone(),
                output_ty,
            })
        }
        NodeKindConfig::Plain { template } => NodeKind::Plain(PlainSpec {
            source: template.clone(),
        }),
        NodeKindConfig::Iterator { sources, unordered } => {
            NodeKind::Iterator(acvus_orchestration::IteratorSpec {
                sources: sources
                    .iter()
                    .map(|s| acvus_orchestration::IteratorSource {
                        name: s.name.clone(),
                        expr: interner.intern(&s.expr),
                        entries: s
                            .entries
                            .iter()
                            .map(|e| acvus_orchestration::IteratorEntry {
                                condition: e.condition.as_ref().map(|c| interner.intern(c)),
                                transform: match &e.transform {
                                    TransformConfig::Template { source } => {
                                        acvus_orchestration::SourceTransform::Template(
                                            interner.intern(source),
                                        )
                                    }
                                    TransformConfig::Script { source } => {
                                        acvus_orchestration::SourceTransform::Script(
                                            interner.intern(source),
                                        )
                                    }
                                },
                            })
                            .collect(),
                        start: s.start.as_ref().map(|v| interner.intern(v)),
                        end: s.end.as_ref().map(|v| interner.intern(v)),
                    })
                    .collect(),
                unordered: *unordered,
            })
        }
    };

    let execution = match &cfg.strategy.execution {
        ExecutionConfig::Always => Execution::Always,
        ExecutionConfig::OncePerTurn => Execution::OncePerTurn,
    };

    let persistency = match &cfg.strategy.persistency {
        PersistencyConfig::Ephemeral => Persistency::Ephemeral,
        PersistencyConfig::Sequence { bind } => Persistency::Sequence {
            bind: interner.intern(bind),
        },
        PersistencyConfig::Patch { bind } => Persistency::Patch {
            bind: interner.intern(bind),
        },
    };

    Ok(NodeSpec {
        name: interner.intern(&cfg.name),
        kind,
        strategy: Strategy {
            execution,
            persistency,
            initial_value: cfg
                .strategy
                .initial_value
                .as_ref()
                .map(|s| interner.intern(s)),
            retry: cfg.strategy.retry,
            assert: cfg
                .strategy
                .assert_script
                .as_ref()
                .map(|s| interner.intern(s)),
        },
        is_function: cfg.is_function,
        fn_params: cfg
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
