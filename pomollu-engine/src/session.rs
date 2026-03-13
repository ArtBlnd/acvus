use acvus_interpreter::Value;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    BlobStore, BlobStoreJournal, DisplayEntrySpec, EntryRef, IterableDisplaySpec, Journal,
    ProviderConfig, Resolved, StaticDisplaySpec, compile_iterable_display,
    compile_static_display, render_display, render_display_with_idx,
};
use acvus_utils::{Astr, Interner, Stepped};
use rustc_hash::FxHashMap;
use serde::Deserialize;
use tsify::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;
use wasm_bindgen_futures::JsFuture;

use crate::build_registry;
use crate::config::{self, SessionConfig};
use crate::fetch::{UnsafeSend, WebFetch};
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

#[derive(Deserialize, Default)]
#[serde(default)]
struct DisplayEntryJson {
    name: String,
    condition: String,
    template: String,
}

#[wasm_bindgen]
pub struct ChatSession {
    engine: acvus_chat::ChatEngine<BlobStoreJournal<IdbBlobStore>>,
    storage_types: FxHashMap<Astr, Ty>,
    interner: Interner,
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
            .map(|n| config::convert_node(&interner, n))
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
        let storage_types = env.storage_types.clone();

        let compiled =
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

        // Providers
        let providers: FxHashMap<String, ProviderConfig> = cfg
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

        // Open IDB store and journal
        let session_id_owned = session_id.to_string();
        let store = IdbBlobStore::open(session_id_owned.clone()).await;
        let (journal, cursor) = match BlobStoreJournal::open(store, interner.clone()).await {
            Some(journal) => {
                let cursor = load_cursor(&journal)
                    .await
                    .expect("cursor ref missing in existing journal");
                (journal, cursor)
            }
            None => {
                let store = IdbBlobStore::open(session_id_owned).await;
                BlobStoreJournal::new(store, interner.clone()).await
            }
        };

        let engine = acvus_chat::ChatEngine::new(
            compiled,
            providers,
            WebFetch,
            journal,
            cursor,
            &cfg.entrypoint,
            &cfg.side_effects,
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

    /// Run one turn.
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
                UnsafeSend(async move { acvus_ext::regex_call(&interner, name, args).await })
            }
        };

        let (result, _new_cursor) = self
            .engine
            .turn(&resolver, &extern_handler)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        // Persist tree + cursor to IDB
        self.engine.journal.flush_tree().await;
        save_cursor(&mut self.engine.journal, self.engine.cursor).await;

        let concrete = result.into_pure().to_concrete(&self.interner);
        let jcv: JsConcreteValue = concrete.into();

        // Build TurnNode for the new cursor
        let cursor = self.engine.cursor;
        let parent = self.engine.journal.parent_of(cursor);
        let entry = self.engine.journal.entry(cursor).await;
        let turn_node = TurnNode {
            uuid: cursor.to_string(),
            parent: parent.map(|p| p.to_string()),
            depth: entry.depth(),
        };

        let turn_result = TurnResult {
            value: jcv,
            turn: turn_node,
        };
        Ok(turn_result.into_ts()?.js_value())
    }

    /// Current turn count.
    pub async fn turn_count(&self) -> usize {
        self.engine.history_len().await
    }

    /// Evaluate an iterator script against storage and return the list length.
    pub async fn display_list_len(&self, iterator_script: &str) -> Result<usize, JsError> {
        use acvus_interpreter::Interpreter;

        let (compiled, _) = acvus_orchestration::compile_script(
            &self.interner,
            iterator_script,
            &self.storage_types,
        )
        .map_err(|e| JsError::new(&e.display(&self.interner).to_string()))?;

        let interp = Interpreter::new(&self.interner, compiled.module.clone());
        let mut coroutine = interp.execute();
        loop {
            match coroutine.resume().await {
                Stepped::Emit(value) => {
                    let len = match value {
                        Value::List(items) => items.len(),
                        Value::Deque(deque) => deque.len(),
                        _ => return Ok(0),
                    };
                    return Ok(len);
                }
                Stepped::NeedContext(request) => {
                    let name = request.name();
                    let cursor = self.engine.cursor;
                    let entry = self.engine.journal.entry(cursor).await;
                    let Some(value) = entry.get(self.interner.resolve(name)) else {
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
        let compiled = compile_iterable_display(&self.interner, &spec, &self.storage_types)
            .map_err(|errs| {
                let msg = errs
                    .iter()
                    .map(|e| e.display(&self.interner).to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                JsError::new(&msg)
            })?;
        let cursor = self.engine.cursor;
        let entry = self.engine.journal.entry(cursor).await;
        let result = render_display_with_idx(&self.interner, &compiled, &entry, index).await;
        Ok(
            DisplayRenderResult(result.into_iter().map(Into::into).collect())
                .into_ts()?
                .js_value(),
        )
    }

    /// Render a static display template.
    pub async fn render_static(&self, template: &str) -> Result<JsValue, JsError> {
        let spec = StaticDisplaySpec {
            template: template.to_string(),
        };
        let compiled =
            compile_static_display(&self.interner, &spec, &self.storage_types).map_err(|errs| {
                let msg = errs
                    .iter()
                    .map(|e| e.display(&self.interner).to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                JsError::new(&msg)
            })?;
        let cursor = self.engine.cursor;
        let entry = self.engine.journal.entry(cursor).await;
        let result = render_display(&self.interner, &compiled, &entry).await;
        Ok(
            DisplayRenderResult(result.into_iter().map(Into::into).collect())
                .into_ts()?
                .js_value(),
        )
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
        let entry = self.engine.journal.entry(uuid).await;
        let view = crate::history::StorageView {
            cursor: uuid.to_string(),
            depth: entry.depth(),
            entries: entry
                .entries()
                .into_iter()
                .map(|(k, v)| {
                    let concrete = v.as_ref().clone().into_pure().to_concrete(&self.interner);
                    (k, JsConcreteValue::from(concrete))
                })
                .collect(),
        };
        Ok(view.into_ts()?.js_value())
    }

    /// Get visible state at the current cursor.
    pub async fn visible_state(&self) -> Result<JsValue, JsError> {
        let cursor = self.engine.cursor;
        let entry = self.engine.journal.entry(cursor).await;
        let view = crate::history::StorageView {
            cursor: cursor.to_string(),
            depth: entry.depth(),
            entries: entry
                .entries()
                .into_iter()
                .map(|(k, v)| {
                    let concrete = v.as_ref().clone().into_pure().to_concrete(&self.interner);
                    (k, JsConcreteValue::from(concrete))
                })
                .collect(),
        };
        Ok(view.into_ts()?.js_value())
    }
}
