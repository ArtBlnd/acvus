use acvus_mir::ty::Ty;
use acvus_orchestration::{
    ApiKind, ExprSpec, Execution, GenerationParams, LlmSpec, MaxTokens, MessageSpec, NodeKind,
    FnParam, NodeSpec, Persistency, PlainSpec, Strategy, ThinkingConfig, TokenBudget, ToolBinding,
    ToolParamInfo,
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
    pub side_effects: Vec<String>,
    #[serde(default)]
    pub asset_store_name: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct ProviderConfigJson {
    pub api: ApiKind,
    pub endpoint: String,
    pub api_key: String,
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
        api: ApiKind,
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
    #[serde(rename = "display")]
    Display {
        iterator: String,
        template: String,
    },
    #[serde(rename = "display_static")]
    DisplayStatic {
        template: String,
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
    pub node: String,
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
    Snapshot,
    Sequence { bind: String },
    Patch { bind: String },
}

#[derive(Deserialize)]
#[serde(tag = "mode")]
pub(crate) enum ExecutionConfig {
    #[serde(rename = "always")]
    Always,
    #[serde(rename = "once-per-turn")]
    OncePerTurn,
    #[serde(rename = "if-modified")]
    IfModified { key: String },
}

// ---------------------------------------------------------------------------
// NodeConfig → NodeSpec conversion
// ---------------------------------------------------------------------------

pub(crate) fn convert_node(interner: &Interner, cfg: &NodeConfig) -> Result<NodeSpec, String> {
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

            NodeKind::Llm(LlmSpec {
                api: api.clone(),
                provider: provider.clone(),
                model: model.clone(),
                messages,
                tools: tools
                    .iter()
                    .map(|t| ToolBinding {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        node: t.node.clone(),
                        params: t.params.iter().map(|p| (p.name.clone(), ToolParamInfo {
                            ty: p.ty.clone(),
                            description: p.description.clone(),
                        })).collect(),
                    })
                    .collect(),
                generation: GenerationParams {
                    temperature: *temperature,
                    top_p: *top_p,
                    top_k: *top_k,
                    grounding: *grounding,
                    thinking: thinking.clone(),
                },
                cache_key: None,
                max_tokens: max_tokens
                    .as_ref()
                    .map(|mt| MaxTokens {
                        input: mt.input,
                        output: mt.output,
                    })
                    .unwrap_or_default(),
            })
        }
        NodeKindConfig::Expr {
            template,
            output_ty,
        } => {
            let output_ty = output_ty
                .as_ref()
                .map(|desc| crate::desc_to_ty(interner, desc))
                .unwrap_or(Ty::Infer);
            NodeKind::Expr(ExprSpec {
                source: template.clone(),
                output_ty,
            })
        }
        NodeKindConfig::Plain { template } => NodeKind::Plain(PlainSpec {
            source: template.clone(),
        }),
        NodeKindConfig::Display { iterator, template } => {
            NodeKind::Display(acvus_orchestration::DisplaySpec {
                iterator: iterator.clone(),
                template: template.clone(),
            })
        }
        NodeKindConfig::DisplayStatic { template } => {
            NodeKind::DisplayStatic(acvus_orchestration::DisplayStaticSpec {
                template: template.clone(),
            })
        }
        NodeKindConfig::Iterator { sources, unordered } => {
            NodeKind::Iterator(acvus_orchestration::IteratorSpec {
                sources: sources.iter().map(|s| (s.name.clone(), interner.intern(&s.node))).collect(),
                unordered: *unordered,
            })
        }
    };

    let execution = match &cfg.strategy.execution {
        ExecutionConfig::Always => Execution::Always,
        ExecutionConfig::OncePerTurn => Execution::OncePerTurn,
        ExecutionConfig::IfModified { key } => Execution::IfModified {
            key: interner.intern(key),
        },
    };

    let persistency = match &cfg.strategy.persistency {
        PersistencyConfig::Ephemeral => Persistency::Ephemeral,
        PersistencyConfig::Snapshot => Persistency::Snapshot,
        PersistencyConfig::Sequence { bind } => Persistency::Sequence { bind: interner.intern(bind) },
        PersistencyConfig::Patch { bind } => Persistency::Patch { bind: interner.intern(bind) },
    };

    Ok(NodeSpec {
        name: interner.intern(&cfg.name),
        kind,
        strategy: Strategy {
            execution,
            persistency,
            initial_value: cfg.strategy.initial_value.as_ref().map(|s| interner.intern(s)),
            retry: cfg.strategy.retry,
            assert: cfg.strategy.assert_script.as_ref().map(|s| interner.intern(s)),
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
