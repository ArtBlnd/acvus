use std::sync::Arc;

use acvus_interpreter::{PureValue, Value};
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    ApiKind, DisplayEntrySpec, ExprSpec, GenerationParams, HttpRequest, IterableDisplaySpec,
    LlmSpec, MaxTokens, MessageSpec, NodeKind, NodeSpec, PlainSpec, ProviderConfig, Resolved,
    StaticDisplaySpec, Storage, Strategy, TokenBudget, ToolBinding,
    compile_iterable_display, compile_static_display, render_display, render_display_with_idx,
};
use acvus_utils::{Astr, Interner};
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;
use wasm_bindgen_futures::JsFuture;

use crate::build_registry;
use crate::schema::*;


// ---------------------------------------------------------------------------
// UnsafeSend -- WASM is single-threaded, safe to mark as Send
// ---------------------------------------------------------------------------

struct UnsafeSend<T>(T);
unsafe impl<T> Send for UnsafeSend<T> {}

impl<T> Future for UnsafeSend<T>
where
    T: Future,
{
    type Output = T::Output;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: WASM is single-threaded, no concurrent access.
        unsafe { self.map_unchecked_mut(|s| &mut s.0).poll(cx) }
    }
}

// ---------------------------------------------------------------------------
// WebFetch -- browser fetch API
// ---------------------------------------------------------------------------

pub struct WebFetch;

impl acvus_orchestration::Fetch for WebFetch {
    fn fetch(
        &self,
        request: &HttpRequest,
    ) -> impl Future<Output = Result<serde_json::Value, String>> + Send {
        let url = request.url.clone();
        let headers = request.headers.clone();
        let body = request.body.clone();

        UnsafeSend(async move {
            use web_sys::{Headers as WHeaders, Request, RequestInit, Response};

            let opts = RequestInit::new();
            opts.set_method("POST");

            let h = WHeaders::new().map_err(|e| format!("{e:?}"))?;
            h.set("Content-Type", "application/json")
                .map_err(|e| format!("{e:?}"))?;
            for (k, v) in &headers {
                h.set(k, v).map_err(|e| format!("{e:?}"))?;
            }
            opts.set_headers(&h);

            let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;
            opts.set_body(&JsValue::from_str(&body_str));

            let req = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;

            let window = web_sys::window().ok_or("no window")?;
            let resp_val = JsFuture::from(window.fetch_with_request(&req))
                .await
                .map_err(|e| format!("{e:?}"))?;
            let resp: Response = resp_val.dyn_into().map_err(|e| format!("{e:?}"))?;

            let json_promise = resp.json().map_err(|e| format!("{e:?}"))?;
            let json_val = JsFuture::from(json_promise)
                .await
                .map_err(|e| format!("{e:?}"))?;

            let json_str = js_sys::JSON::stringify(&json_val)
                .map_err(|e| format!("{e:?}"))?
                .as_string()
                .ok_or_else(|| "JSON.stringify returned non-string".to_string())?;
            serde_json::from_str(&json_str).map_err(|e| e.to_string())
        })
    }
}

// ---------------------------------------------------------------------------
// SessionStorage -- Rust cache + JS callback for real-time sync
// ---------------------------------------------------------------------------

/// SAFETY: WASM is single-threaded. js_sys::Function is !Send+!Sync but
/// there is no concurrent access.
struct SendSyncFunction(js_sys::Function);
unsafe impl Send for SendSyncFunction {}
unsafe impl Sync for SendSyncFunction {}

pub struct SessionStorage {
    entries: FxHashMap<String, Arc<Value>>,
    on_change: Option<SendSyncFunction>,
    interner: Interner,
}

impl Default for SessionStorage {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
            on_change: None,
            interner: Interner::new(),
        }
    }
}

impl SessionStorage {
    pub fn with_callback(on_change: js_sys::Function, interner: &Interner) -> Self {
        Self {
            entries: FxHashMap::default(),
            on_change: Some(SendSyncFunction(on_change)),
            interner: interner.clone(),
        }
    }

    fn notify(&self, key: &str, value: &Value) {
        let Some(ref cb) = self.on_change else { return };
        let js_key = JsValue::from_str(key);
        let cv = value.clone().into_pure().to_concrete(&self.interner);
        let jcv: JsConcreteValue = cv.into();
        let js_val = jcv.into_ts().unwrap().js_value();
        let _ = cb.0.call2(&JsValue::NULL, &js_key, &js_val);
    }

    fn notify_remove(&self, key: &str) {
        let Some(ref cb) = self.on_change else { return };
        let js_key = JsValue::from_str(key);
        // Use UNDEFINED (not NULL) so JS can distinguish removal from Unit values.
        let _ = cb.0.call2(&JsValue::NULL, &js_key, &JsValue::UNDEFINED);
    }
}

impl Storage for SessionStorage {
    fn get(&self, key: &str) -> Option<Arc<Value>> {
        self.entries.get(key).cloned()
    }

    fn set(&mut self, key: String, value: Value) {
        self.notify(&key, &value);
        self.entries.insert(key, Arc::new(value));
    }

    fn remove(&mut self, key: &str) {
        if self.entries.remove(key).is_some() {
            self.notify_remove(key);
        }
    }
}

impl SessionStorage {
    fn to_snapshot(&self) -> StorageSnapshot {
        StorageSnapshot(
            self.entries
                .iter()
                .map(|(k, v)| {
                    let concrete = v.as_ref().clone().into_pure().to_concrete(&self.interner);
                    (k.clone(), JsConcreteValue::from(concrete))
                })
                .collect(),
        )
    }

    pub fn export(&self) -> Ts<StorageSnapshot> {
        self.to_snapshot().into_ts().unwrap()
    }

    pub fn import(
        js: Ts<StorageSnapshot>,
        on_change: Option<js_sys::Function>,
        interner: &Interner,
    ) -> Self {
        let snapshot = js.to_rust().unwrap();

        let entries: FxHashMap<String, Arc<Value>> = snapshot
            .0
            .into_iter()
            .map(|(k, jcv)| {
                let cv: acvus_interpreter::ConcreteValue = jcv.into();
                let pure = PureValue::from_concrete(&cv, interner);
                (k, Arc::new(Value::from_pure(pure)))
            })
            .collect();

        Self {
            entries,
            on_change: on_change.map(SendSyncFunction),
            interner: interner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// ChatSession -- wasm_bindgen wrapper around ChatEngine
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
#[serde(default)]
struct DisplayEntryJson {
    name: String,
    condition: String,
    template: String,
}

#[wasm_bindgen]
pub struct ChatSession {
    engine: acvus_chat::ChatEngine<SessionStorage>,
    /// Types of values stored in storage (node self types + @turn).
    /// Used for display compilation.
    storage_types: FxHashMap<Astr, Ty>,
    interner: Interner,
}

// ---------------------------------------------------------------------------
// Config deserialization (JSON from JS)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SessionConfig {
    nodes: Vec<NodeConfig>,
    providers: FxHashMap<String, ProviderConfigJson>,
    entrypoint: String,
    #[serde(default)]
    context: FxHashMap<String, ContextDecl>,
    #[serde(default)]
    side_effects: Vec<String>,
}

#[derive(Deserialize)]
struct ProviderConfigJson {
    api: ApiKind,
    endpoint: String,
    api_key: String,
}

#[derive(Deserialize)]
struct ContextDecl {
    #[serde(rename = "type")]
    ty: Option<crate::TypeDesc>,
}

#[derive(Deserialize)]
struct NodeConfig {
    name: String,
    initial_value: Option<String>,
    strategy: StrategyConfig,
    #[serde(default)]
    retry: u32,
    assert_script: Option<String>,
    #[serde(default)]
    is_function: bool,
    #[serde(default)]
    fn_params: Vec<FnParamConfig>,
    #[serde(flatten)]
    kind: NodeKindConfig,
}

#[derive(Deserialize)]
struct FnParamConfig {
    name: String,
    #[serde(rename = "type")]
    ty: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum NodeKindConfig {
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
        output_ty: Option<crate::TypeDesc>,
    },
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct ToolConfig {
    name: String,
    description: String,
    node: String,
    params: FxHashMap<String, String>,
}

#[derive(Deserialize)]
struct MaxTokensJson {
    input: Option<u32>,
    output: Option<u32>,
}

#[derive(Deserialize)]
struct MessageConfig {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    inline_template: Option<String>,
    #[serde(default)]
    iterator: Option<String>,
    #[serde(default)]
    slice: Option<Vec<i64>>,
    #[serde(default)]
    token_budget: Option<TokenBudgetConfig>,
}

#[derive(Deserialize)]
struct TokenBudgetConfig {
    priority: u32,
    #[serde(default)]
    min: Option<u32>,
    #[serde(default)]
    max: Option<u32>,
}

#[derive(Deserialize)]
#[serde(tag = "mode")]
enum StrategyConfig {
    #[serde(rename = "always")]
    Always,
    #[serde(rename = "once-per-turn")]
    OncePerTurn,
    #[serde(rename = "if-modified")]
    IfModified { key: String },
    #[serde(rename = "history")]
    History { history_bind: String },
}

fn convert_node(interner: &Interner, cfg: &NodeConfig) -> Result<NodeSpec, String> {
    let kind = match &cfg.kind {
        NodeKindConfig::Llm {
            provider,
            api,
            model,
            temperature,
            top_p,
            top_k,
            grounding,
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
                        params: t.params.clone(),
                    })
                    .collect(),
                generation: GenerationParams {
                    temperature: *temperature,
                    top_p: *top_p,
                    top_k: *top_k,
                    grounding: *grounding,
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
                initial_value: cfg.initial_value.clone(),
            })
        }
        NodeKindConfig::Plain { template } => NodeKind::Plain(PlainSpec {
            source: template.clone(),
        }),
    };

    let strategy = match &cfg.strategy {
        StrategyConfig::Always => Strategy::Always,
        StrategyConfig::OncePerTurn => Strategy::OncePerTurn,
        StrategyConfig::IfModified { key } => Strategy::IfModified {
            key: interner.intern(key),
        },
        StrategyConfig::History { history_bind } => Strategy::History {
            history_bind: interner.intern(history_bind),
        },
    };

    // Non-Expr nodes must not have initial_value
    if !matches!(kind, NodeKind::Expr(_)) && cfg.initial_value.is_some() {
        return Err(format!(
            "node '{}': initial_value is only supported for Expr nodes",
            cfg.name
        ));
    }

    Ok(NodeSpec {
        name: interner.intern(&cfg.name),
        kind,
        strategy,
        retry: cfg.retry,
        assert: cfg.assert_script.as_ref().map(|s| interner.intern(s)),
        is_function: cfg.is_function,
        fn_params: cfg.fn_params.iter().map(|p| {
            let ty = crate::parse_type_string(&interner, &p.ty);
            (interner.intern(&p.name), ty)
        }).collect(),
    })
}

#[wasm_bindgen]
impl ChatSession {
    /// Create a new chat session from JSON config + optional persisted storage.
    ///
    /// `config_json`: `{ nodes, providers, entrypoint, context? }`
    /// `storage_js`:  previously exported storage (or null for fresh session)
    /// `on_storage_change`: JS callback `(key: string, value: any | null) => void`
    ///   called on every storage set (value = JSON) or remove (value = null).
    pub async fn create(
        config_json: &str,
        storage_js: Option<Ts<StorageSnapshot>>,
        on_storage_change: Option<js_sys::Function>,
    ) -> Result<ChatSession, JsValue> {
        let interner = Interner::new();

        let config: SessionConfig =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Build context types from config
        let mut context_types: FxHashMap<Astr, Ty> = FxHashMap::default();
        for (name, decl) in &config.context {
            let desc = decl
                .ty
                .as_ref()
                .ok_or_else(|| JsValue::from_str(&format!("context '{name}': missing type")))?;
            let ty = crate::desc_to_ty(&interner, desc);
            context_types.insert(interner.intern(name), ty);
        }

        // Convert nodes -- collect all errors
        let specs: Vec<NodeSpec> = config
            .nodes
            .iter()
            .map(|n| convert_node(&interner, n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e))?;

        // Compile -- also compute full context_types (including node-derived @turn, @nodeName, etc.)
        let registry = build_registry(&interner, context_types)
            .map_err(|e| {
                let key_name = interner.resolve(e.key);
                JsValue::from_str(&format!(
                    "context type conflict: @{key_name} exists in both {} and {} tier",
                    e.tier_a, e.tier_b
                ))
            })?;
        let env = acvus_orchestration::compute_external_context_env(
            &interner,
            &specs,
            registry,
        )
        .map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.display(&interner).to_string())
                .collect::<Vec<_>>()
                .join("\n");
            JsValue::from_str(&msg)
        })?;
        let storage_types = env.storage_types.clone();

        let compiled =
            acvus_orchestration::compile_nodes_with_env(&interner, &specs, env)
                .map_err(|errs| {
                let msg = errs
                    .iter()
                    .map(|e| e.display(&interner).to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                JsValue::from_str(&msg)
            })?;

        // Providers
        let providers: FxHashMap<String, ProviderConfig> = config
            .providers
            .into_iter()
            .map(|(name, p)| {
                (
                    name,
                    ProviderConfig {
                        api: p.api,
                        endpoint: p.endpoint,
                        api_key: p.api_key,
                    },
                )
            })
            .collect();

        // Storage: restore from IndexedDB or fresh, with real-time JS callback
        let storage = match storage_js {
            Some(snapshot) => SessionStorage::import(snapshot, on_storage_change, &interner),
            None => match on_storage_change {
                Some(cb) => SessionStorage::with_callback(cb, &interner),
                None => SessionStorage {
                    interner: interner.clone(),
                    ..SessionStorage::default()
                },
            },
        };

        let engine = acvus_chat::ChatEngine::new(
            compiled,
            providers,
            WebFetch,
            storage,
            &config.entrypoint,
            &config.side_effects,
            &interner,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(ChatSession {
            engine,
            storage_types,
            interner,
        })
    }

    /// Run one turn. `resolve_fn` is called when the engine needs an external
    /// context value (e.g. @input). It receives a key name string and must
    /// return a Promise<string>.
    pub async fn turn(&mut self, resolve_fn: &js_sys::Function) -> Result<JsValue, JsError> {
        // SAFETY: WASM is single-threaded -- js_sys::Function is !Send+!Sync
        // but there is no concurrent access.
        struct SendSyncFn(js_sys::Function);
        unsafe impl Send for SendSyncFn {}
        unsafe impl Sync for SendSyncFn {}

        let wrapped = SendSyncFn(resolve_fn.clone());
        let interner = self.interner.clone();

        let resolver = move |key: Astr| {
            let resolve_fn = wrapped.0.clone();
            let interner = interner.clone();
            UnsafeSend(async move {
                let this = JsValue::NULL;
                let key_str = interner.resolve(key);
                let js_key = JsValue::from_str(key_str);
                let result = resolve_fn
                    .call1(&this, &js_key)
                    .unwrap_or(JsValue::UNDEFINED);

                let value = if result.has_type::<js_sys::Promise>() {
                    let promise: js_sys::Promise = result.unchecked_into();
                    JsFuture::from(promise).await.unwrap_or(JsValue::UNDEFINED)
                } else {
                    result
                };

                let s: String = value.as_string().unwrap_or_default();
                Resolved::Turn(Value::String(s))
            })
        };

        let extern_handler = {
            let interner = self.interner.clone();
            move |name: Astr, args: Vec<acvus_interpreter::Value>| {
                let interner = interner.clone();
                UnsafeSend(async move {
                    acvus_ext::regex_call(&interner, name, args).await
                })
            }
        };

        let result = self
            .engine
            .turn(&resolver, &extern_handler)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let concrete = result.into_pure().to_concrete(&self.interner);
        let jcv: JsConcreteValue = concrete.into();
        Ok(jcv.into_ts()?.js_value())
    }

    /// Export storage as a typed snapshot (for IndexedDB persistence / display rendering).
    pub fn export_storage(&self) -> Ts<StorageSnapshot> {
        self.engine.state.storage.export()
    }

    /// Current turn count.
    pub fn turn_count(&self) -> usize {
        self.engine.history_len()
    }

    /// Evaluate an iterator script against storage and return the list length.
    pub async fn display_list_len(&self, iterator_script: &str) -> Result<usize, JsError> {
        use acvus_interpreter::Interpreter;
        use acvus_utils::Stepped;

        let (compiled, _) = acvus_orchestration::compile_script(
            &self.interner,
            iterator_script,
            &self.storage_types,
        )
        .map_err(|e| JsError::new(&e.display(&self.interner).to_string()))?;

        let interp = Interpreter::new(
            &self.interner,
            compiled.module.clone(),
        );
        let mut coroutine = interp.execute();
        loop {
            match coroutine.resume().await {
                Stepped::Emit(value) => {
                    let Value::List(items) = value else {
                        return Ok(0);
                    };
                    return Ok(items.len());
                }
                Stepped::NeedContext(request) => {
                    let name = request.name();
                    let Some(value) = self.engine.state.storage.get(self.interner.resolve(name))
                    else {
                        return Ok(0);
                    };
                    request.resolve(value);
                }
                Stepped::NeedExternCall(_) => {
                    panic!("unexpected extern call in display_list_len");
                }
                Stepped::Done => return Ok(0),
                Stepped::Error(e) => return Err(JsError::new(&format!("display error: {e}"))),
            }
        }
    }

    /// Render one index of an iterable display.
    /// Evaluates `iterator_script` against storage, takes the element at `index`,
    /// then runs each entry's condition/template with `@item` and `@index` injected.
    ///
    /// `entries_json`: `[{"condition": "...", "template": "..."}]`
    pub async fn render_display(
        &self,
        iterator_script: &str,
        entries_json: &str,
        index: usize,
    ) -> Result<JsValue, JsError> {
        let entries: Vec<DisplayEntryJson> = serde_json::from_str(entries_json)
            .map_err(|e| JsError::new(&format!("invalid entries JSON: {e}")))?;
        let spec = IterableDisplaySpec {
            iterator: iterator_script.to_string(),
            entries: entries
                .into_iter()
                .map(|e| DisplayEntrySpec {
                    name: e.name,
                    condition: e.condition,
                    template: e.template,
                })
                .collect(),
        };
        let compiled =
            compile_iterable_display(&self.interner, &spec, &self.storage_types)
                .map_err(|errs| {
                    let msg = errs
                        .iter()
                        .map(|e| e.display(&self.interner).to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    JsError::new(&msg)
                })?;
        let result = render_display_with_idx(
            &self.interner,
            &compiled,
            &self.engine.state.storage,
            index,
        )
        .await;
        Ok(DisplayRenderResult(result.into_iter().map(Into::into).collect()).into_ts()?.js_value())
    }

    /// Render a static display template.
    pub async fn render_static(&self, template: &str) -> Result<JsValue, JsError> {
        let spec = StaticDisplaySpec {
            template: template.to_string(),
        };
        let compiled =
            compile_static_display(&self.interner, &spec, &self.storage_types).map_err(
                |errs| {
                    let msg = errs
                        .iter()
                        .map(|e| e.display(&self.interner).to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    JsError::new(&msg)
                },
            )?;
        let result = render_display(
            &self.interner,
            &compiled,
            &self.engine.state.storage,
        )
        .await;
        Ok(DisplayRenderResult(result.into_iter().map(Into::into).collect()).into_ts()?.js_value())
    }
}
