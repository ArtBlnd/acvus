use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, PureValue, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    ApiKind, DisplayEntrySpec, ExprSpec, GenerationParams, HttpRequest, IterableDisplaySpec,
    LlmSpec, MaxTokens, MessageSpec, NodeKind, NodeSpec, PlainSpec, ProviderConfig, Resolved,
    SelfSpec, StaticDisplaySpec, Storage, Strategy, TokenBudget, ToolBinding,
    compile_iterable_display, compile_static_display, render_display,
    render_display_with_idx,
};
use acvus_utils::{Astr, Interner};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::default_registry;

// ---------------------------------------------------------------------------
// PureValue -> serde_json::Value (plain JS-compatible representation)
// ---------------------------------------------------------------------------

pub(crate) fn pure_to_json(interner: &Interner, v: &PureValue) -> serde_json::Value {
    match v {
        PureValue::Unit => serde_json::Value::Null,
        PureValue::Int(n) => serde_json::json!(*n),
        PureValue::Float(f) => serde_json::json!(*f),
        PureValue::String(s) => serde_json::Value::String(s.clone()),
        PureValue::Bool(b) => serde_json::Value::Bool(*b),
        PureValue::Byte(b) => serde_json::json!(*b),
        PureValue::Range {
            start,
            end,
            inclusive,
        } => serde_json::json!({"start": start, "end": end, "inclusive": inclusive}),
        PureValue::List(items) => {
            serde_json::Value::Array(items.iter().map(|i| pure_to_json(interner, i)).collect())
        }
        PureValue::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(k, v)| (interner.resolve(*k).to_string(), pure_to_json(interner, v)))
                .collect(),
        ),
        PureValue::Tuple(items) => {
            serde_json::Value::Array(items.iter().map(|i| pure_to_json(interner, i)).collect())
        }
        PureValue::Variant { tag, payload } => {
            let tag_str = interner.resolve(*tag).to_string();
            match payload {
                Some(p) => serde_json::json!({tag_str: pure_to_json(interner, p)}),
                None => serde_json::Value::String(tag_str),
            }
        }
    }
}

/// serde_json::Value -> PureValue (reverse of pure_to_json)
fn json_to_pure(interner: &Interner, v: &serde_json::Value) -> Option<PureValue> {
    match v {
        serde_json::Value::Null => Some(PureValue::Unit),
        serde_json::Value::Bool(b) => Some(PureValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(PureValue::Int(i))
            } else {
                Some(PureValue::Float(n.as_f64()?))
            }
        }
        serde_json::Value::String(s) => Some(PureValue::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Option<Vec<PureValue>> = arr.iter().map(|i| json_to_pure(interner, i)).collect();
            Some(PureValue::List(items?))
        }
        serde_json::Value::Object(map) => {
            let items: Option<HashMap<Astr, PureValue>> = map
                .iter()
                .map(|(k, v)| json_to_pure(interner, v).map(|pv| (interner.intern(k), pv)))
                .collect();
            Some(PureValue::Object(items?))
        }
    }
}

// ---------------------------------------------------------------------------
// UnsafeSend -- WASM is single-threaded, safe to mark as Send
// ---------------------------------------------------------------------------

struct UnsafeSend<T>(T);
unsafe impl<T> Send for UnsafeSend<T> {}

impl<T> std::future::Future for UnsafeSend<T>
where
    T: std::future::Future,
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
    ) -> impl std::future::Future<Output = Result<serde_json::Value, String>> + Send {
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

            let req =
                Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;

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
    entries: HashMap<String, Arc<Value>>,
    on_change: Option<SendSyncFunction>,
    interner: Interner,
}

impl Default for SessionStorage {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            on_change: None,
            interner: Interner::new(),
        }
    }
}

impl SessionStorage {
    pub fn with_callback(on_change: js_sys::Function, interner: &Interner) -> Self {
        Self {
            entries: HashMap::new(),
            on_change: Some(SendSyncFunction(on_change)),
            interner: interner.clone(),
        }
    }

    fn notify(&self, key: &str, value: &Value) {
        let Some(ref cb) = self.on_change else { return };
        let js_key = JsValue::from_str(key);
        let json_str = serde_json::to_string(&pure_to_json(&self.interner, &value.clone().into_pure()))
            .unwrap_or_default();

        let js_val = js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL);
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
    pub fn export(&self) -> JsValue {
        let map: HashMap<&str, serde_json::Value> = self
            .entries
            .iter()
            .map(|(k, v)| (k.as_str(), pure_to_json(&self.interner, &v.as_ref().clone().into_pure())))
            .collect();
        let json_str = serde_json::to_string(&map).unwrap_or_default();
        js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL)
    }

    pub fn export_json(&self) -> JsValue {
        self.export()
    }

    pub fn import(js: JsValue, on_change: Option<js_sys::Function>, interner: &Interner) -> Self {
        let json_str = js_sys::JSON::stringify(&js)
            .ok()
            .and_then(|s| s.as_string())
            .unwrap_or_default();

        let map: HashMap<String, serde_json::Value> =
            serde_json::from_str(&json_str).unwrap_or_default();

        // Convert JSON values back to PureValue -> Value
        let entries: HashMap<String, Arc<Value>> = map
            .into_iter()
            .filter_map(|(k, v)| {
                let pure = json_to_pure(interner, &v)?;
                let value = Value::from_pure(pure);
                Some((k, Arc::new(value)))
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

#[derive(Deserialize)]
struct DisplayEntryJson {
    #[serde(default)]
    name: String,
    #[serde(default)]
    condition: String,
    #[serde(default)]
    template: String,
}

#[wasm_bindgen]
pub struct ChatSession {
    engine: acvus_chat::ChatEngine<SessionStorage>,
    /// Types of values stored in storage (node self types + @turn).
    /// Used for display compilation.
    storage_types: HashMap<Astr, Ty>,
    interner: Interner,
}

// ---------------------------------------------------------------------------
// Config deserialization (JSON from JS)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SessionConfig {
    nodes: Vec<NodeConfig>,
    providers: HashMap<String, ProviderConfigJson>,
    entrypoint: String,
    #[serde(default)]
    context: HashMap<String, ContextDecl>,
    #[serde(default)]
    side_effects: Vec<String>,
}

#[derive(Deserialize)]
struct ProviderConfigJson {
    api: String,
    endpoint: String,
    api_key: String,
}

#[derive(Deserialize)]
struct ContextDecl {
    #[serde(rename = "type")]
    ty: Option<String>,
}

#[derive(Deserialize)]
struct NodeConfig {
    name: String,
    kind: String,
    #[serde(default)]
    initial_value: Option<String>,
    strategy: StrategyConfig,
    #[serde(default)]
    retry: u32,
    #[serde(default)]
    assert_script: Option<String>,

    // LLM-specific
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<MaxTokensJson>,
    #[serde(default)]
    messages: Option<Vec<MessageConfig>>,
    #[serde(default)]
    tools: Option<Vec<ToolConfig>>,

    // Plain/Expr-specific
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    output_ty: Option<String>,
}

#[derive(Deserialize)]
struct ToolConfig {
    name: String,
    #[serde(default)]
    description: String,
    node: String,
    #[serde(default)]
    params: HashMap<String, String>,
}

#[derive(Deserialize)]
struct MaxTokensJson {
    #[serde(default)]
    input: Option<u32>,
    #[serde(default)]
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
    let kind = match cfg.kind.as_str() {
        "llm" => {
            let api_str = cfg.api.as_ref()
                .ok_or_else(|| format!("node '{}': llm requires 'api'", cfg.name))?;
            let api = ApiKind::parse(api_str)
                .ok_or_else(|| format!("node '{}': unknown api '{api_str}'", cfg.name))?;
            let provider = cfg.provider.as_ref()
                .ok_or_else(|| format!("node '{}': llm requires 'provider'", cfg.name))?;
            let model = cfg.model.as_ref()
                .ok_or_else(|| format!("node '{}': llm requires 'model'", cfg.name))?;
            let messages_cfg = cfg.messages.as_ref()
                .ok_or_else(|| format!("node '{}': llm requires 'messages'", cfg.name))?;

            let messages: Vec<MessageSpec> = messages_cfg
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
                        let source = m
                            .inline_template
                            .as_ref()
                            .or(m.template.as_ref())?
                            .clone();
                        Some(MessageSpec::Block {
                            role: m.role.as_ref().map(|r| interner.intern(r)).unwrap_or_else(|| interner.intern("user")),
                            source,
                        })
                    }
                })
                .collect();

            let tools: Vec<ToolBinding> = cfg
                .tools
                .as_ref()
                .map(|ts| {
                    ts.iter()
                        .map(|t| ToolBinding {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            node: t.node.clone(),
                            params: t.params.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();

            NodeKind::Llm(LlmSpec {
                api,
                provider: provider.clone(),
                model: model.clone(),
                messages,
                tools,
                generation: GenerationParams {
                    temperature: cfg.temperature,
                    ..Default::default()
                },
                cache_key: None,
                max_tokens: cfg
                    .max_tokens
                    .as_ref()
                    .map(|mt| MaxTokens {
                        input: mt.input,
                        output: mt.output,
                    })
                    .unwrap_or_default(),
            })
        }
        "expr" => {
            let source = cfg.template.as_ref()
                .ok_or_else(|| format!("node '{}': expr requires 'template'", cfg.name))?;
            let output_ty = cfg
                .output_ty
                .as_deref()
                .and_then(|s| crate::parse_ty(interner, s))
                .unwrap_or(Ty::Infer);
            NodeKind::Expr(ExprSpec {
                source: source.clone(),
                output_ty,
            })
        }
        "plain" => {
            let source = cfg.template.as_ref()
                .ok_or_else(|| format!("node '{}': plain requires 'template'", cfg.name))?;
            NodeKind::Plain(PlainSpec {
                source: source.clone(),
            })
        }
        other => return Err(format!("node '{}': unknown kind '{other}'", cfg.name)),
    };

    let strategy = match &cfg.strategy {
        StrategyConfig::Always => Strategy::Always,
        StrategyConfig::OncePerTurn => Strategy::OncePerTurn,
        StrategyConfig::IfModified { key } => Strategy::IfModified { key: interner.intern(key) },
        StrategyConfig::History { history_bind } => Strategy::History { history_bind: interner.intern(history_bind) },
    };

    Ok(NodeSpec {
        name: interner.intern(&cfg.name),
        kind,
        self_spec: SelfSpec {
            initial_value: cfg.initial_value.as_ref().map(|s| interner.intern(s)),
        },
        strategy,
        retry: cfg.retry,
        assert: cfg.assert_script.as_ref().map(|s| interner.intern(s)),
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
        storage_js: JsValue,
        on_storage_change: Option<js_sys::Function>,
    ) -> Result<ChatSession, JsValue> {
        let interner = Interner::new();

        let config: SessionConfig =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Build context types from config
        let mut context_types: HashMap<Astr, Ty> = HashMap::new();
        for (name, decl) in &config.context {
            let ty_str = decl.ty.as_ref()
                .ok_or_else(|| JsValue::from_str(&format!("context '{name}': missing type")))?;
            let ty = crate::parse_ty(&interner, ty_str)
                .ok_or_else(|| JsValue::from_str(&format!("context '{name}': invalid type '{ty_str}'")))?;
            context_types.insert(interner.intern(name), ty);
        }

        // Convert nodes -- collect all errors
        let specs: Vec<NodeSpec> = config.nodes.iter()
            .map(|n| convert_node(&interner, n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e))?;

        // Compile -- also compute full context_types (including node-derived @turn, @nodeName, etc.)
        let extern_registry = default_registry(&interner);
        let env = acvus_orchestration::compute_external_context_env(
            &interner,
            &specs,
            &context_types,
            &extern_registry,
        )
        .map_err(|errs| {
            let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
            JsValue::from_str(&msg)
        })?;
        let storage_types = env.storage_types.clone();

        let compiled = acvus_orchestration::compile_nodes_with_env(&interner, &specs, &extern_registry, env)
            .map_err(|errs| {
                let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
                JsValue::from_str(&msg)
            })?;

        // Providers
        let providers: HashMap<String, ProviderConfig> = config
            .providers
            .into_iter()
            .filter_map(|(name, p)| {
                let api = ApiKind::parse(&p.api)?;
                Some((
                    name,
                    ProviderConfig {
                        api,
                        endpoint: p.endpoint,
                        api_key: p.api_key,
                    },
                ))
            })
            .collect();

        // Storage: restore from IndexedDB or fresh, with real-time JS callback
        let storage = if storage_js.is_null() || storage_js.is_undefined() {
            match on_storage_change {
                Some(cb) => SessionStorage::with_callback(cb, &interner),
                None => SessionStorage { interner: interner.clone(), ..SessionStorage::default() },
            }
        } else {
            SessionStorage::import(storage_js, on_storage_change, &interner)
        };

        let mut extern_fns = ExternFnRegistry::new(&interner);
        let regex_mod = acvus_ext::regex_module(&interner, &mut extern_fns);
        let mut extern_reg = ExternRegistry::new();
        extern_reg.register(&regex_mod);
        drop(extern_reg); // only needed for compile, not runtime

        let engine = acvus_chat::ChatEngine::new(
            compiled,
            providers,
            WebFetch,
            extern_fns,
            storage,
            &config.entrypoint,
            &config.side_effects,
            &interner,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(ChatSession { engine, storage_types, interner })
    }

    /// Run one turn. `resolve_fn` is called when the engine needs an external
    /// context value (e.g. @input). It receives a key name string and must
    /// return a Promise<string>.
    pub async fn turn(&mut self, resolve_fn: &js_sys::Function) -> Result<JsValue, JsValue> {
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

        let result = self
            .engine
            .turn(&resolver)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let json = pure_to_json(&self.interner, &result.into_pure());
        let json_str = serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&json_str).map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }

    /// Export storage as JSON (for IndexedDB persistence).
    /// Returns a plain JS object with PureValue-compatible structure.
    pub fn export_storage(&self) -> JsValue {
        self.engine.state.storage.export()
    }

    /// Export storage as plain JSON (for display rendering).
    /// Returns JSON where PureValue enums are flattened to native JS types.
    pub fn export_storage_json(&self) -> JsValue {
        self.engine.state.storage.export_json()
    }

    /// Current turn count.
    pub fn turn_count(&self) -> usize {
        self.engine.history_len()
    }

    /// Evaluate an iterator script against storage and return the list length.
    pub async fn display_list_len(&self, iterator_script: &str) -> Result<usize, JsValue> {
        use acvus_utils::Stepped;
        use acvus_interpreter::Interpreter;

        let registry = default_registry(&self.interner);
        let (compiled, _) = acvus_orchestration::compile_script(
            &self.interner,
            iterator_script,
            &self.storage_types,
            &registry,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let interp = Interpreter::new(&self.interner, compiled.module.clone(), self.engine.extern_fns());
        let (mut coroutine, mut key) = interp.execute();
        loop {
            match coroutine.resume(key).await {
                Stepped::Emit(emit) => {
                    let (value, _) = emit.into_parts();
                    let Value::List(items) = value else {
                        return Ok(0);
                    };
                    return Ok(items.len());
                }
                Stepped::NeedContext(need) => {
                    let name = need.name();
                    let Some(value) = self.engine.state.storage.get(self.interner.resolve(name)) else {
                        return Ok(0);
                    };
                    key = need.into_key(value);
                }
                Stepped::Done => return Ok(0),
                Stepped::Error(e) => return Err(JsValue::from_str(&format!("display error: {e}"))),
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
    ) -> Result<JsValue, JsValue> {
        let entries: Vec<DisplayEntryJson> =
            serde_json::from_str(entries_json).unwrap_or_default();
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
        let registry = default_registry(&self.interner);
        let compiled = compile_iterable_display(&self.interner, &spec, &self.storage_types, &registry)
            .map_err(|errs| {
                let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
                JsValue::from_str(&msg)
            })?;
        let result =
            render_display_with_idx(&self.interner, &compiled, &self.engine.state.storage, self.engine.extern_fns(), index)
                .await;
        let json_str = serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&json_str).map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }

    /// Render a static display template.
    pub async fn render_static(&self, template: &str) -> Result<JsValue, JsValue> {
        let spec = StaticDisplaySpec {
            template: template.to_string(),
        };
        let registry = default_registry(&self.interner);
        let compiled = compile_static_display(&self.interner, &spec, &self.storage_types, &registry)
            .map_err(|errs| {
                let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
                JsValue::from_str(&msg)
            })?;
        let result =
            render_display(&self.interner, &compiled, &self.engine.state.storage, self.engine.extern_fns()).await;
        let json_str = serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&json_str).map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}
