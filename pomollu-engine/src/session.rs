use std::sync::Arc;

use acvus_interpreter::{IntoValue, TypedValue, Value};
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    BlobStore, BlobStoreJournal, EntryRef, Journal,
    Resolved,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use tsify::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;
use wasm_bindgen_futures::JsFuture;

use crate::build_registry;
use crate::config::{self, SessionConfig};
use crate::fetch::{UnsafeSend, WebFetch};
use crate::idb::IdbAssetStore;
use crate::idb::IdbBlobStore;
use crate::schema::*;

// ---------------------------------------------------------------------------
// Cursor persistence helpers
// ---------------------------------------------------------------------------

async fn save_cursor(journal: &mut BlobStoreJournal<IdbBlobStore>, cursor: Uuid) {
    let bytes = cursor.as_bytes().to_vec();
    let hash = journal.store_mut().put(bytes).await;
    let current = journal.store().ref_get("cursor").await;
    let _ = journal.store_mut().ref_cas("cursor", current, hash).await;
}

async fn load_cursor(journal: &BlobStoreJournal<IdbBlobStore>) -> Option<Uuid> {
    let hash = journal.store().ref_get("cursor").await?;
    let bytes = journal.store().get(&hash).await?;
    Uuid::from_slice(&bytes).ok()
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------

async fn dispatch_extern(
    interner: &Interner,
    asset_store: &Option<std::sync::Arc<IdbAssetStore>>,
    name: Astr,
    args: Vec<TypedValue>,
) -> Result<TypedValue, acvus_interpreter::RuntimeError> {
    let name_str = interner.resolve(name);
    match name_str {
        "asset_url" => {
            let Value::Pure(acvus_interpreter::PureValue::String(ref path)) = *args[0].value() else {
                return Err(acvus_interpreter::RuntimeError::unexpected_type(
                    "asset_url", &[acvus_interpreter::ValueKind::String], args[0].value().kind(),
                ));
            };
            acvus_interpreter::set_interner_ctx(interner);
            let option_string_ty = Ty::Option(Box::new(Ty::String));
            let Some(store) = asset_store else {
                let result: Option<String> = None;
                return Ok(TypedValue::new(result.into_value(), option_string_ty));
            };
            let exists = store.exists(path).await;
            if !exists {
                let result: Option<String> = None;
                return Ok(TypedValue::new(result.into_value(), option_string_ty));
            }
            let version = store.version().await;
            let db_name = store.db_name();
            let result: Option<String> = Some(format!("/asset/{db_name}/{path}?v={version}"));
            Ok(TypedValue::new(result.into_value(), option_string_ty))
        }
        _ => {
            let plain_args: Vec<&Value> = args.iter().map(|tv| tv.value()).collect();
            let value = acvus_ext::regex_call(interner, name, plain_args).await?;
            Ok(TypedValue::new(value, Ty::String))
        }
    }
}

#[wasm_bindgen]
pub struct ChatSession {
    engine: acvus_chat::ChatEngine<BlobStoreJournal<IdbBlobStore>>,
    /// Context type registry for compilation (extern fns + system/storage + user).
    compile_registry: ContextTypeRegistry,
    interner: Interner,
    asset_store: Option<std::sync::Arc<IdbAssetStore>>,
    entrypoint_name: String,
    /// Whether current evaluation is no_execute (for flush decision).
    no_execute: bool,
}

#[wasm_bindgen]
impl ChatSession {
    /// Create or resume a chat session.
    ///
    /// If IndexedDB already has state for `session_id`, the session is resumed.
    /// Otherwise a fresh session is created.
    pub async fn create(
        config_json: &str,
        session_id: &str,
    ) -> Result<ChatSession, JsValue> {
        let interner = Interner::new();

        let cfg: SessionConfig =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Build context types from config
        let mut context_types: FxHashMap<Astr, Ty> = FxHashMap::default();
        for (name, decl) in &cfg.context {
            let desc = decl
                .ty
                .as_ref()
                .ok_or_else(|| JsValue::from_str(&format!("context '{name}': missing type")))?;
            let ty = crate::desc_to_ty(&interner, desc);
            context_types.insert(interner.intern(name), ty);
        }

        // Convert nodes
        let specs: Vec<acvus_orchestration::NodeSpec> = cfg
            .nodes
            .iter()
            .map(|n| config::convert_node(&interner, n, &cfg.providers))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e))?;

        // Compile
        let registry = build_registry(&interner, context_types).map_err(|e| {
            let key_name = interner.resolve(e.key);
            JsValue::from_str(&format!(
                "context type conflict: @{key_name} exists in both {} and {} tier",
                e.tier_a, e.tier_b
            ))
        })?;
        let env =
            acvus_orchestration::compute_external_context_env(&interner, &specs, registry)
                .map_err(|errs| {
                    let msg = errs
                        .iter()
                        .map(|e| e.display(&interner).to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    JsValue::from_str(&msg)
                })?;
        let compile_registry = env.registry.to_full();

        let mut compiled =
            acvus_orchestration::compile_nodes_with_env(&interner, &specs, env).map_err(
                |errs| {
                    let msg = errs
                        .iter()
                        .map(|e| e.display(&interner).to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    JsValue::from_str(&msg)
                },
            )?;


        // Open IDB store and journal
        let session_id_owned = session_id.to_string();
        let store = IdbBlobStore::open(session_id_owned.clone()).await;
        let (journal, cursor) = match BlobStoreJournal::open(store, interner.clone()).await
            .map_err(|e| JsError::new(&format!("journal open: {e}")))?
        {
            Some(journal) => {
                let cursor = load_cursor(&journal)
                    .await
                    .expect("cursor ref missing in existing journal");
                (journal, cursor)
            }
            None => {
                let store = IdbBlobStore::open(session_id_owned).await;
                BlobStoreJournal::new(store, interner.clone()).await
                    .map_err(|e| JsError::new(&format!("journal new: {e}")))?
            }
        };

        // Open asset store if configured
        let asset_store = match &cfg.asset_store_name {
            Some(name) => Some(Arc::new(IdbAssetStore::open(name).await)),
            None => None,
        };

        let engine = acvus_chat::ChatEngine::new(
            compiled,
            WebFetch,
            journal,
            cursor,
            &cfg.entrypoint,
            &interner,
        )
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let entrypoint_name = cfg.entrypoint.clone();

        Ok(ChatSession {
            engine,
            compile_registry,
            interner,
            asset_store,
            entrypoint_name,
            no_execute: false,
        })
    }

    /// Returns the full tree structure and current cursor.
    pub fn tree(&self) -> Result<JsValue, JsError> {
        let nodes: Vec<TurnNode> = self
            .engine
            .journal
            .tree_nodes()
            .into_iter()
            .map(|(uuid, parent, depth)| TurnNode {
                uuid: uuid.to_string(),
                parent: parent.map(|p| p.to_string()),
                depth,
            })
            .collect();
        let view = TreeView {
            nodes,
            cursor: self.engine.cursor.to_string(),
        };
        Ok(view.into_ts()?.js_value())
    }

    /// Start an evaluation. Resolves the named node and prepares streaming.
    ///
    /// - `no_execute=false`: creates a new journal branch.
    /// - `no_execute=true`: reads from current cursor (no branch).
    pub async fn start_evaluate(
        &mut self,
        node_name: &str,
        no_execute: bool,
        resolve_fn: &js_sys::Function,
    ) -> Result<(), JsError> {
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
                Resolved::Turn(TypedValue::string(s))
            })
        };

        let extern_handler = {
            let interner = self.interner.clone();
            let asset_store = self.asset_store.clone();
            move |name: Astr, args: Vec<acvus_interpreter::TypedValue>| {
                let interner = interner.clone();
                let asset_store = asset_store.clone();
                UnsafeSend(async move { dispatch_extern(&interner, &asset_store, name, args).await })
            }
        };

        self.no_execute = no_execute;
        self.engine
            .start_evaluate(node_name, no_execute, &resolver, &extern_handler)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(())
    }

    /// Pull the next item from the current evaluation.
    ///
    /// Returns a JsValue (ConcreteValue) or null when done.
    /// Handles NeedContext/NeedExternCall internally.
    pub async fn evaluate_next(
        &mut self,
        resolve_fn: &js_sys::Function,
    ) -> Result<JsValue, JsError> {
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
                Resolved::Turn(TypedValue::string(s))
            })
        };

        let extern_handler = {
            let interner = self.interner.clone();
            let asset_store = self.asset_store.clone();
            move |name: Astr, args: Vec<acvus_interpreter::TypedValue>| {
                let interner = interner.clone();
                let asset_store = asset_store.clone();
                UnsafeSend(async move { dispatch_extern(&interner, &asset_store, name, args).await })
            }
        };

        match self.engine.evaluate_next(&resolver, &extern_handler).await {
            Ok(Some(value)) => {
                let concrete = value.to_concrete(&self.interner);
                let jcv: JsConcreteValue = concrete.into();
                Ok(jcv.into_ts()?.js_value())
            }
            Ok(None) => {
                // Flush if this was an executing evaluation
                if !self.no_execute {
                    self.engine.journal.flush_tree().await;
                    save_cursor(&mut self.engine.journal, self.engine.cursor).await;
                }
                Ok(JsValue::NULL)
            }
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Cancel an in-progress evaluation. Rolls back cursor if executing.
    pub fn cancel_evaluate(&mut self) {
        self.engine.cancel_evaluate();
    }

    /// Current turn count.
    pub async fn turn_count(&self) -> Result<usize, JsError> {
        self.engine.history_len().await
            .map_err(|e| JsError::new(&format!("turn_count: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Navigation API
// ---------------------------------------------------------------------------

#[wasm_bindgen]
impl ChatSession {
    /// Undo: move cursor to parent entry.
    pub async fn undo(&mut self) -> Result<(), JsError> {
        let parent = self
            .engine
            .journal
            .parent_of(self.engine.cursor)
            .ok_or_else(|| JsError::new("already at root \u{2014} cannot undo"))?;
        self.engine.cursor = parent;
        save_cursor(&mut self.engine.journal, self.engine.cursor).await;
        Ok(())
    }

    /// Navigate to a specific entry by UUID.
    pub async fn goto(&mut self, id: &str) -> Result<(), JsError> {
        let uuid = Uuid::parse_str(id).map_err(|e| JsError::new(&e.to_string()))?;
        if !self.engine.journal.contains(uuid) {
            return Err(JsError::new("entry not found in history tree"));
        }
        self.engine.cursor = uuid;
        save_cursor(&mut self.engine.journal, self.engine.cursor).await;
        Ok(())
    }

    /// Get storage state at a specific entry.
    pub async fn state_at(&self, id: &str) -> Result<JsValue, JsError> {
        let uuid = Uuid::parse_str(id).map_err(|e| JsError::new(&e.to_string()))?;
        if !self.engine.journal.contains(uuid) {
            return Err(JsError::new("entry not found in history tree"));
        }
        let entry = self.engine.journal.entry(uuid).await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let view = crate::history::StorageView {
            cursor: uuid.to_string(),
            depth: entry.depth(),
            entries: entry
                .entries()
                .into_iter()
                .map(|(k, v)| {
                    let concrete = v.to_concrete(&self.interner);
                    (k, JsConcreteValue::from(concrete))
                })
                .collect(),
        };
        Ok(view.into_ts()?.js_value())
    }

    /// Get visible state at the current cursor.
    pub async fn visible_state(&self) -> Result<JsValue, JsError> {
        let cursor = self.engine.cursor;
        let entry = self.engine.journal.entry(cursor).await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let view = crate::history::StorageView {
            cursor: cursor.to_string(),
            depth: entry.depth(),
            entries: entry
                .entries()
                .into_iter()
                .map(|(k, v)| {
                    let concrete = v.to_concrete(&self.interner);
                    (k, JsConcreteValue::from(concrete))
                })
                .collect(),
        };
        Ok(view.into_ts()?.js_value())
    }
}
