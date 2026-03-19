use acvus_lsp::{DocId, LspError, LspErrorCategory, LspSession, ScriptMode};
use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry};
use acvus_mir::analysis::reachable_context::KnownValue;
use rustc_hash::FxHashMap;
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

use crate::convert::WebNode;
use crate::error::{EngineError, ErrorCategory};
use crate::schema::*;
use crate::{asset_context_types, build_registry, convert_context_types, try_extract_known};

// ---------------------------------------------------------------------------
// WASM LanguageSession — exposes full LSP document-centric API
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct LanguageSession {
    inner: LspSession,
}

#[wasm_bindgen]
impl LanguageSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: LspSession::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    /// Open a new document with source, mode, and scope (context types).
    /// Returns the document ID (u32).
    pub fn open(
        &mut self,
        source: &str,
        mode: &str,
        scope: Ts<WasmScope>,
    ) -> Result<u32, JsError> {
        let scope = scope.to_rust()?;
        let interner = self.inner.interner();
        let registry = self.build_scope(interner, &scope)?;
        let mode = parse_mode(mode)?;
        let id = self.inner.open(source.to_string(), mode, registry);
        Ok(id.raw())
    }

    /// Update a document's source. Invalidates caches.
    pub fn update_source(&mut self, doc_id: u32, source: &str) {
        self.inner.update_source(DocId::from_raw(doc_id), source.to_string());
    }

    /// Update a document's scope. Invalidates caches.
    pub fn update_scope(
        &mut self,
        doc_id: u32,
        scope: Ts<WasmScope>,
    ) -> Result<(), JsError> {
        let scope = scope.to_rust()?;
        let interner = self.inner.interner();
        let registry = self.build_scope(interner, &scope)?;
        self.inner.update_scope(DocId::from_raw(doc_id), registry);
        Ok(())
    }

    /// Update both source and scope atomically.
    pub fn update(
        &mut self,
        doc_id: u32,
        source: &str,
        scope: Ts<WasmScope>,
    ) -> Result<(), JsError> {
        let scope = scope.to_rust()?;
        let interner = self.inner.interner();
        let registry = self.build_scope(interner, &scope)?;
        self.inner.update(DocId::from_raw(doc_id), source.to_string(), registry);
        Ok(())
    }

    /// Close a document.
    pub fn close(&mut self, doc_id: u32) {
        self.inner.close(DocId::from_raw(doc_id));
    }

    /// Bind a document to a node field (inference unit).
    /// Diagnostics for bound documents come from inference (shared subst).
    pub fn bind_doc_to_node(
        &mut self,
        doc_id: u32,
        node_name: &str,
        field: &str,
        field_index: Option<usize>,
    ) {
        let node_field = match (field, field_index) {
            ("initialValue", _) => acvus_lsp::NodeField::InitialValue,
            ("bind", _) => acvus_lsp::NodeField::Bind,
            ("assert", _) => acvus_lsp::NodeField::Assert,
            ("ifModifiedKey", _) => acvus_lsp::NodeField::IfModifiedKey,
            ("exprSource", _) => acvus_lsp::NodeField::ExprSource,
            ("message", Some(i)) => acvus_lsp::NodeField::Message(i),
            ("iteratorExpr", Some(i)) => acvus_lsp::NodeField::IteratorExpr(i),
            ("iteratorTmpl", Some(i)) => acvus_lsp::NodeField::IteratorTmpl(i),
            _ => return,
        };
        self.inner.bind_doc_to_node(DocId::from_raw(doc_id), node_name, node_field);
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Get diagnostics for a document.
    pub fn diagnostics(&mut self, doc_id: u32) -> Result<Ts<DiagnosticsResult>, JsError> {
        let errors = self.inner.diagnostics(DocId::from_raw(doc_id));
        Ok(DiagnosticsResult {
            ok: errors.is_empty(),
            errors: convert_lsp_errors(&errors),
        }
        .into_ts()?)
    }

    /// Discover context keys for a document.
    pub fn context_keys(
        &mut self,
        doc_id: u32,
        known_values: Ts<WasmKnownValues>,
    ) -> Result<Ts<ContextKeysResult>, JsError> {
        let known_input = known_values.to_rust()?;
        let interner = self.inner.interner();

        // Build known values from static scripts
        let known = self.build_known_values(interner, &known_input);

        let keys = self.inner.context_keys(DocId::from_raw(doc_id), &known);
        let interner = self.inner.interner();

        let result_keys: Vec<ContextKey> = keys
            .iter()
            .map(|k| ContextKey {
                name: interner.resolve(k.name).to_string(),
                ty: ty_to_desc(interner, &k.ty),
                status: match k.status {
                    acvus_lsp::ContextKeyStatus::Eager => ContextKeyStatus::Eager,
                    acvus_lsp::ContextKeyStatus::Lazy => ContextKeyStatus::Lazy,
                    acvus_lsp::ContextKeyStatus::Pruned => ContextKeyStatus::Pruned,
                },
            })
            .collect();

        Ok(ContextKeysResult {
            keys: result_keys,
        }
        .into_ts()?)
    }

    /// Get completions at cursor position.
    pub fn completions(
        &self,
        doc_id: u32,
        cursor: usize,
    ) -> Result<Ts<CompletionResult>, JsError> {
        let items: Vec<CompletionItem> = self
            .inner
            .completions(DocId::from_raw(doc_id), cursor)
            .into_iter()
            .map(|i| CompletionItem {
                label: i.label,
                kind: match i.kind {
                    acvus_lsp::CompletionKind::Context => CompletionKind::Context,
                    acvus_lsp::CompletionKind::Builtin => CompletionKind::Builtin,
                    acvus_lsp::CompletionKind::Keyword => CompletionKind::Keyword,
                },
                detail: i.detail,
                insert_text: i.insert_text,
            })
            .collect();
        Ok(CompletionResult { items }.into_ts()?)
    }

    // -----------------------------------------------------------------------
    // Node-level rebuild
    // -----------------------------------------------------------------------

    /// Full node-level rebuild — typecheck all nodes.
    pub fn rebuild_nodes(
        &mut self,
        options: Ts<TypecheckNodesOptions>,
    ) -> Result<Ts<TypecheckNodesResult>, JsError> {
        let options = options.to_rust()?;
        let interner = self.inner.interner();

        let user_types = convert_context_types(interner, &options.injected_types);
        let registry = match build_registry(interner, user_types) {
            Ok(r) => r,
            Err(e) => {
                let key_name = interner.resolve(e.key);
                return Ok(TypecheckNodesResult::fail(vec![EngineError::general(
                    ErrorCategory::Type,
                    format!(
                        "context type conflict: @{key_name} in {} and {}",
                        e.tier_a, e.tier_b
                    ),
                )])
                .into_ts()?);
            }
        };

        let specs: Vec<acvus_orchestration::NodeSpec> = match options
            .nodes
            .iter()
            .map(|w| w.into_node(interner))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(s) => s,
            Err(e) => {
                return Ok(
                    TypecheckNodesResult::fail(vec![EngineError::general(ErrorCategory::Parse, e)])
                        .into_ts()?,
                );
            }
        };

        drop(interner);
        let result = self.inner.rebuild_nodes(specs, registry);
        let interner = self.inner.interner();

        let context_types: FxHashMap<String, TypeDesc> = result
            .context_types
            .iter()
            .map(|(k, v)| (interner.resolve(*k).to_string(), ty_to_desc(interner, v)))
            .collect();

        let node_locals: FxHashMap<String, NodeLocalTypes> = result
            .node_locals
            .iter()
            .map(|(k, v)| {
                (
                    interner.resolve(*k).to_string(),
                    NodeLocalTypes {
                        raw: ty_to_desc(interner, &v.raw_ty),
                        self_ty: ty_to_desc(interner, &v.self_ty),
                    },
                )
            })
            .collect();

        let node_errors: FxHashMap<String, NodeErrors> = result
            .node_errors
            .iter()
            .map(|(k, v)| (interner.resolve(*k).to_string(), convert_node_errors(v)))
            .collect();

        Ok(TypecheckNodesResult {
            env_errors: convert_lsp_errors(&result.env_errors),
            context_types,
            node_locals,
            node_errors,
        }
        .into_ts()?)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_scope(
        &self,
        interner: &acvus_utils::Interner,
        scope: &WasmScope,
    ) -> Result<ContextTypeRegistry, JsError> {
        let mut extern_fns = acvus_ext::regex_context_types(interner);
        extern_fns.extend(asset_context_types(interner));
        let system = convert_context_types(interner, &scope.provided);
        let user = convert_context_types(interner, &scope.user);
        let partial = match PartialContextTypeRegistry::new(extern_fns, system, user) {
            Ok(r) => r,
            Err(e) => {
                let key_name = interner.resolve(e.key);
                return Err(JsError::new(&format!(
                    "context type conflict: @{key_name} in {} and {}",
                    e.tier_a, e.tier_b
                )));
            }
        };
        Ok(partial.to_full())
    }

    fn build_known_values(
        &self,
        interner: &acvus_utils::Interner,
        input: &WasmKnownValues,
    ) -> FxHashMap<acvus_utils::Astr, KnownValue> {
        // We need a registry for try_extract_known — use an empty one.
        // Known values are simple constants, they don't need context types.
        let empty_reg = PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        )
        .unwrap()
        .to_full();

        let mut known = FxHashMap::default();
        for (name, script) in &input.values {
            if let Some(kv) = try_extract_known(interner, script, &empty_reg) {
                known.insert(interner.intern(name), kv);
            }
        }
        known
    }
}

// ---------------------------------------------------------------------------
// WASM input/output types
// ---------------------------------------------------------------------------

/// Scope for a document — the types visible in that document's context.
#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct WasmScope {
    /// Provided types — engine-provided, NOT user params.
    /// Goes into system/scoped tier → context_keys excludes these.
    #[serde(default)]
    pub provided: FxHashMap<String, TypeDesc>,
    /// User-declared types — param types declared by the user.
    /// Goes into user tier → context_keys includes these.
    #[serde(default)]
    pub user: FxHashMap<String, TypeDesc>,
}

/// Known values for context key pruning.
#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct WasmKnownValues {
    /// Map of param name → static script source.
    #[serde(default)]
    pub values: FxHashMap<String, String>,
}

/// Diagnostics result.
#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticsResult {
    pub ok: bool,
    pub errors: Vec<EngineError>,
}

/// A discovered context key.
#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ContextKey {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: TypeDesc,
    pub status: ContextKeyStatus,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub enum ContextKeyStatus {
    Eager,
    Lazy,
    Pruned,
}

/// Context keys discovery result.
#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ContextKeysResult {
    pub keys: Vec<ContextKey>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_mode(mode: &str) -> Result<ScriptMode, JsError> {
    match mode {
        "script" => Ok(ScriptMode::Script),
        "template" => Ok(ScriptMode::Template),
        other => Err(JsError::new(&format!("invalid mode: {other}"))),
    }
}

fn convert_lsp_errors(errs: &[LspError]) -> Vec<EngineError> {
    errs.iter()
        .map(|e| {
            let category = match e.category {
                LspErrorCategory::Parse => ErrorCategory::Parse,
                LspErrorCategory::Type => ErrorCategory::Type,
            };
            EngineError::from_lsp(category, &e.message, e.span)
        })
        .collect()
}

fn convert_node_errors(src: &acvus_lsp::NodeErrors) -> NodeErrors {
    NodeErrors {
        env: convert_lsp_errors(&src.env),
        initial_value: convert_lsp_errors(&src.initial_value),
        bind: convert_lsp_errors(&src.bind),
        if_modified_key: convert_lsp_errors(&src.if_modified_key),
        assert: convert_lsp_errors(&src.assert),
        messages: src
            .messages
            .iter()
            .map(|(k, v)| (k.to_string(), convert_lsp_errors(v)))
            .collect(),
        expr_source: convert_lsp_errors(&src.expr_source),
    }
}

use serde::{Deserialize, Serialize};
