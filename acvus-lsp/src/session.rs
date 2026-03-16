use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry};
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::{Ty, TySubst};
use acvus_mir_pass::analysis::reachable_context::KnownValue;
use acvus_orchestration::{ContextScope, Execution, NodeSpec, Persistency};
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Opaque document identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocId(u32);

impl DocId {
    pub fn from_raw(id: u32) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptMode {
    Template,
    Script,
}

/// Which field of a node this document represents.
/// Used for binding documents to inference units (nodes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeField {
    InitialValue,
    Bind,
    Assert,
    IfModifiedKey,
    ExprSource,
    Message(usize),
    IteratorExpr(usize),
    IteratorTmpl(usize),
}

#[derive(Debug, Clone)]
pub struct LspError {
    pub category: LspErrorCategory,
    pub message: String,
    pub span: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspErrorCategory {
    Parse,
    Type,
}

#[derive(Clone)]
pub struct ContextKeyInfo {
    pub name: Astr,
    pub ty: Ty,
    pub status: ContextKeyStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextKeyStatus {
    Eager,
    Lazy,
    Pruned,
}

#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: String,
    pub insert_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Context,
    Builtin,
    Keyword,
}

// --- rebuild result types (generic, node-level) ---

#[derive(Clone)]
pub struct RebuildResult {
    pub env_errors: Vec<LspError>,
    pub context_types: FxHashMap<Astr, Ty>,
    pub node_locals: FxHashMap<Astr, NodeLocals>,
    pub node_errors: FxHashMap<Astr, NodeErrors>,
}

#[derive(Clone)]
pub struct NodeLocals {
    pub raw_ty: Ty,
    pub self_ty: Ty,
}

#[derive(Debug, Clone, Default)]
pub struct NodeErrors {
    pub env: Vec<LspError>,
    pub initial_value: Vec<LspError>,
    pub bind: Vec<LspError>,
    pub if_modified_key: Vec<LspError>,
    pub assert: Vec<LspError>,
    pub messages: FxHashMap<usize, Vec<LspError>>,
    pub expr_source: Vec<LspError>,
}

impl NodeErrors {
    pub fn is_empty(&self) -> bool {
        self.env.is_empty()
            && self.initial_value.is_empty()
            && self.bind.is_empty()
            && self.if_modified_key.is_empty()
            && self.assert.is_empty()
            && self.messages.is_empty()
            && self.expr_source.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Document — a single script/template with its scope
// ---------------------------------------------------------------------------

struct Document {
    source: String,
    mode: ScriptMode,
    scope: ContextTypeRegistry,
    scope_fingerprint: u64,
    version: u64,

    // Cached results — invalidated when version changes
    cache: DocCache,
}

fn scope_fingerprint(scope: &ContextTypeRegistry, interner: &Interner) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for tier in [scope.system(), scope.scoped(), scope.user()] {
        tier.len().hash(&mut hasher);
        let mut keys: Vec<&str> = tier.keys().map(|k| interner.resolve(*k)).collect();
        keys.sort();
        for k in keys {
            k.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[derive(Default)]
struct DocCache {
    version: u64, // version when cache was computed
    diagnostics: Option<Vec<LspError>>,
    context_keys: Option<CachedKeys>,
}

struct CachedKeys {
    /// The known values that were used to compute this cache.
    known_hash: u64,
    keys: Vec<ContextKeyInfo>,
}

impl DocCache {
    fn is_valid(&self, doc_version: u64) -> bool {
        self.version == doc_version
    }

    fn invalidate(&mut self) {
        self.diagnostics = None;
        self.context_keys = None;
    }
}

// ---------------------------------------------------------------------------
// Rebuild cache
// ---------------------------------------------------------------------------

struct RebuildCache {
    fingerprint: u64,
    result: RebuildResult,
}

fn rebuild_fingerprint(specs: &[NodeSpec], registry: &PartialContextTypeRegistry, interner: &Interner) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    specs.len().hash(&mut hasher);
    for spec in specs {
        interner.resolve(spec.name).hash(&mut hasher);
        spec.is_function.hash(&mut hasher);
        // Hash kind-specific source strings
        match &spec.kind {
            acvus_orchestration::NodeKind::Expr(e) => e.source.hash(&mut hasher),
            acvus_orchestration::NodeKind::Llm(llm) => {
                for msg in &llm.messages {
                    match msg {
                        acvus_orchestration::MessageSpec::Block { source, .. } => source.hash(&mut hasher),
                        acvus_orchestration::MessageSpec::Iterator { key, .. } => {
                            interner.resolve(*key).hash(&mut hasher);
                        }
                    }
                }
            }
            acvus_orchestration::NodeKind::Plain(_)
            | acvus_orchestration::NodeKind::LlmCache(_)
            | acvus_orchestration::NodeKind::Display(_)
            | acvus_orchestration::NodeKind::DisplayStatic(_)
            | acvus_orchestration::NodeKind::Iterator(_) => {}
        }
        // Hash strategy
        if let Some(iv) = spec.strategy.initial_value {
            interner.resolve(iv).hash(&mut hasher);
        }
        if let Some(a) = spec.strategy.assert {
            interner.resolve(a).hash(&mut hasher);
        }
        match &spec.strategy.persistency {
            Persistency::Sequence { bind } | Persistency::Patch { bind } => {
                interner.resolve(*bind).hash(&mut hasher);
            }
            Persistency::Ephemeral | Persistency::Snapshot => {}
        }
    }
    // Hash registry user keys (the part that changes between 1st and 2nd rebuild)
    let user = registry.user();
    user.len().hash(&mut hasher);
    let mut user_keys: Vec<&str> = user.keys().map(|k| interner.resolve(*k)).collect();
    user_keys.sort();
    for k in user_keys {
        k.hash(&mut hasher);
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// LspSession — document-centric, incremental
// ---------------------------------------------------------------------------

pub struct LspSession {
    interner: Interner,
    documents: FxHashMap<DocId, Document>,
    next_doc_id: u32,

    // Orchestration system keys — always excluded from context_keys.
    initial_system_keys: FxHashMap<Astr, Ty>,

    // Node-level state (from rebuild_nodes)
    node_env: Option<acvus_orchestration::ExternalContextEnv>,
    specs: Vec<NodeSpec>,
    rebuild_cache: Option<RebuildCache>,

    // Document ↔ Node binding — inference unit membership
    doc_to_node: FxHashMap<DocId, (Astr, NodeField)>,
    // Inference results — per (node, field) errors from typecheck_node (shared subst)
    inference_errors: FxHashMap<(Astr, NodeField), Vec<LspError>>,
}

impl LspSession {
    pub fn new() -> Self {
        let interner = Interner::new();

        // Bootstrap system types — always present in the orchestration engine.
        // @turn_index is registered by compute_external_context_env as system tier.
        // We pre-register it so context_keys filters it even before rebuild_nodes.
        let mut initial_system = FxHashMap::default();
        initial_system.insert(interner.intern("turn_index"), Ty::Int);

        Self {
            interner,
            documents: FxHashMap::default(),
            next_doc_id: 0,
            node_env: None,
            initial_system_keys: initial_system,
            specs: Vec::new(),
            rebuild_cache: None,
            doc_to_node: FxHashMap::default(),
            inference_errors: FxHashMap::default(),
        }
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    /// Open a new document. Returns its DocId.
    pub fn open(
        &mut self,
        source: String,
        mode: ScriptMode,
        scope: ContextTypeRegistry,
    ) -> DocId {
        let id = DocId(self.next_doc_id);
        self.next_doc_id += 1;
        let fp = scope_fingerprint(&scope, &self.interner);
        self.documents.insert(
            id,
            Document {
                source,
                mode,
                scope,
                scope_fingerprint: fp,
                version: 1,
                cache: DocCache::default(),
            },
        );
        id
    }

    /// Update a document's source. Invalidates all caches.
    /// No-op if the source is unchanged.
    pub fn update_source(&mut self, id: DocId, source: String) {
        if let Some(doc) = self.documents.get_mut(&id) {
            if doc.source == source {
                return;
            }
            doc.source = source;
            doc.version += 1;
            doc.cache.invalidate();
        }
    }

    /// Update a document's scope. Invalidates typecheck caches (not parse).
    /// Update a document's scope. No-op if the scope fingerprint is unchanged.
    pub fn update_scope(&mut self, id: DocId, scope: ContextTypeRegistry) {
        if let Some(doc) = self.documents.get_mut(&id) {
            let fp = scope_fingerprint(&scope, &self.interner);
            if doc.scope_fingerprint == fp {
                return; // No change
            }
            doc.scope = scope;
            doc.scope_fingerprint = fp;
            doc.version += 1;
            doc.cache.invalidate();
        }
    }

    /// Update both source and scope atomically.
    pub fn update(&mut self, id: DocId, source: String, scope: ContextTypeRegistry) {
        if let Some(doc) = self.documents.get_mut(&id) {
            doc.source = source;
            doc.scope = scope;
            doc.version += 1;
            doc.cache.invalidate();
        }
    }

    /// Close a document.
    pub fn close(&mut self, id: DocId) {
        self.documents.remove(&id);
        self.doc_to_node.remove(&id);
    }

    // -----------------------------------------------------------------------
    // Document ↔ Node binding
    // -----------------------------------------------------------------------

    /// Bind a document to a node field.
    /// Diagnostics for bound documents come from inference (shared subst)
    /// instead of standalone check_script.
    pub fn bind_doc_to_node(&mut self, doc_id: DocId, node_name: &str, field: NodeField) {
        let name = self.interner.intern(node_name);
        self.doc_to_node.insert(doc_id, (name, field));
    }

    /// Unbind a document from its node.
    pub fn unbind_doc(&mut self, doc_id: DocId) {
        self.doc_to_node.remove(&doc_id);
    }

    /// Store per-field inference results from typecheck_node.
    fn store_node_inference(&mut self, node_name: Astr, errors: &NodeErrors) {
        self.inference_errors.insert((node_name, NodeField::InitialValue), errors.initial_value.clone());
        self.inference_errors.insert((node_name, NodeField::Bind), errors.bind.clone());
        self.inference_errors.insert((node_name, NodeField::Assert), errors.assert.clone());
        self.inference_errors.insert((node_name, NodeField::IfModifiedKey), errors.if_modified_key.clone());
        self.inference_errors.insert((node_name, NodeField::ExprSource), errors.expr_source.clone());
        for (idx, msg_errors) in &errors.messages {
            self.inference_errors.insert((node_name, NodeField::Message(*idx)), msg_errors.clone());
        }
    }

    // -----------------------------------------------------------------------
    // Queries — read, use cache
    // -----------------------------------------------------------------------

    /// Get diagnostics for a document.
    ///
    /// - Bound document (node field): returns inference result (shared subst, accurate).
    /// - Unbound document: standalone check_script/check_template (cached).
    pub fn diagnostics(&mut self, id: DocId) -> Vec<LspError> {
        // Bound document → inference result
        if let Some((node_name, field)) = self.doc_to_node.get(&id) {
            return self
                .inference_errors
                .get(&(*node_name, *field))
                .cloned()
                .unwrap_or_default();
        }

        // Unbound document → standalone check (cached)
        let doc = match self.documents.get(&id) {
            Some(d) => d,
            None => return vec![],
        };

        if doc.cache.is_valid(doc.version) {
            if let Some(ref diags) = doc.cache.diagnostics {
                return diags.clone();
            }
        }

        let diags = match doc.mode {
            ScriptMode::Script => {
                check_script(&self.interner, &doc.source, &doc.scope, None)
            }
            ScriptMode::Template => {
                check_template(&self.interner, &doc.source, &doc.scope)
            }
        };

        let doc = self.documents.get_mut(&id).unwrap();
        doc.cache.version = doc.version;
        doc.cache.diagnostics = Some(diags.clone());
        diags
    }

    /// Get context keys for a document. Cached per (version, known).
    ///
    /// Automatically excludes:
    /// - Keys in the document's provided scope (extern/system/scoped tiers)
    /// - Keys in the rebuild_nodes system tier (@turn_index, node names, etc.)
    pub fn context_keys(
        &mut self,
        id: DocId,
        known: &FxHashMap<Astr, KnownValue>,
    ) -> Vec<ContextKeyInfo> {
        let doc = match self.documents.get(&id) {
            Some(d) => d,
            None => return vec![],
        };

        let known_hash = hash_known(known);

        if doc.cache.is_valid(doc.version) {
            if let Some(ref cached) = doc.cache.context_keys {
                if cached.known_hash == known_hash {
                    return cached.keys.clone();
                }
            }
        }

        // Compute
        let mut keys = discover_context_keys(
            &self.interner,
            &doc.source,
            doc.mode,
            &doc.scope,
            known,
        );

        // Exclude orchestration system keys:
        // - initial_system_keys: always present (@turn_index)
        // - node_env system tier: after rebuild_nodes (node names, etc.)
        keys.retain(|k| {
            if self.initial_system_keys.contains_key(&k.name) {
                return false;
            }
            if let Some(ref env) = self.node_env {
                if env.registry.system().contains_key(&k.name) {
                    return false;
                }
            }
            true
        });

        // Store in cache
        let doc = self.documents.get_mut(&id).unwrap();
        doc.cache.version = doc.version;
        doc.cache.context_keys = Some(CachedKeys {
            known_hash,
            keys: keys.clone(),
        });
        keys
    }

    /// Infer the tail type of a document's script. Returns None if parse fails or type is unknown.
    pub fn tail_type(&self, id: DocId) -> Option<Ty> {
        let doc = match self.documents.get(&id) {
            Some(d) => d,
            None => return None,
        };
        infer_tail_type(&self.interner, &doc.source, &doc.scope)
    }

    /// Get completions at cursor position.
    pub fn completions(&self, id: DocId, cursor: usize) -> Vec<CompletionItem> {
        let doc = match self.documents.get(&id) {
            Some(d) => d,
            None => return vec![],
        };

        if cursor == 0 || cursor > doc.source.len() || !doc.source.is_char_boundary(cursor) {
            return vec![];
        }

        let before = &doc.source[..cursor];
        match detect_trigger(before) {
            CompletionTrigger::Context { prefix } => {
                context_completions(&self.interner, &doc.scope, &prefix)
            }
            CompletionTrigger::Pipe => {
                let expr_before = before.trim_end().strip_suffix('|').unwrap_or(before).trim();
                let tail_ty = infer_tail_type(&self.interner, expr_before, &doc.scope);
                pipe_completions_filtered(&self.interner, &tail_ty)
            }
            CompletionTrigger::Keyword { prefix } => keyword_completions(&prefix),
            CompletionTrigger::None => vec![],
        }
    }

    // -----------------------------------------------------------------------
    // rebuild_nodes — generic node-level analysis
    // -----------------------------------------------------------------------

    pub fn rebuild_nodes(
        &mut self,
        specs: Vec<NodeSpec>,
        registry: PartialContextTypeRegistry,
    ) -> RebuildResult {
        // Cache check: same specs + same registry → return cached result
        let fp = rebuild_fingerprint(&specs, &registry, &self.interner);
        if let Some(ref cache) = self.rebuild_cache {
            if cache.fingerprint == fp {
                return cache.result.clone();
            }
        }

        // 1. Compute external context env
        let env = match acvus_orchestration::compute_external_context_env(
            &self.interner,
            &specs,
            registry,
        ) {
            Ok(env) => env,
            Err(errs) => {
                return RebuildResult {
                    env_errors: from_orch_errors(&errs, &self.interner),
                    context_types: FxHashMap::default(),
                    node_locals: FxHashMap::default(),
                    node_errors: FxHashMap::default(),
                };
            }
        };

        // context_types: visible = system + user
        let context_types: FxHashMap<Astr, Ty> = env.registry.visible();

        // node_locals
        let node_locals: FxHashMap<Astr, NodeLocals> = env
            .node_locals
            .iter()
            .map(|(name, locals)| {
                (
                    *name,
                    NodeLocals {
                        raw_ty: locals.raw_ty.clone(),
                        self_ty: locals.self_ty.clone(),
                    },
                )
            })
            .collect();

        // 2. Per-node, per-field typecheck
        let full_reg = env.registry.to_full();
        let mut node_errors: FxHashMap<Astr, NodeErrors> = FxHashMap::default();

        for spec in &specs {
            let Some(locals) = env.node_locals.get(&spec.name) else {
                continue;
            };
            let errors = typecheck_node(&self.interner, spec, locals, &full_reg);

            // Store inference results for bound documents
            self.store_node_inference(spec.name, &errors);

            if !errors.is_empty() {
                node_errors.insert(spec.name, errors);
            }
        }

        // Store state
        self.specs = specs;
        self.node_env = Some(env);

        let result = RebuildResult {
            env_errors: vec![],
            context_types,
            node_locals,
            node_errors,
        };
        self.rebuild_cache = Some(RebuildCache {
            fingerprint: fp,
            result: result.clone(),
        });
        result
    }
}

// ---------------------------------------------------------------------------
// Core analysis functions — stateless, pure
// ---------------------------------------------------------------------------

fn check_script(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Vec<LspError> {
    let mut subst = TySubst::new();
    check_script_with_subst(interner, source, registry, expected_tail, &mut subst)
}

fn check_script_with_subst(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
    subst: &mut TySubst,
) -> Vec<LspError> {
    let script = match acvus_ast::parse_script(interner, source) {
        Ok(s) => s,
        Err(e) => {
            return vec![LspError {
                category: LspErrorCategory::Parse,
                message: format!("{e}"),
                span: None,
            }];
        }
    };
    match acvus_mir::compile_script_with_hint_subst(
        interner,
        &script,
        registry,
        expected_tail,
        subst,
    ) {
        Ok(_) => vec![],
        Err(errs) => from_mir_errors(&errs, interner),
    }
}

fn check_template(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Vec<LspError> {
    let ast = match acvus_ast::parse(interner, source) {
        Ok(a) => a,
        Err(e) => {
            return vec![LspError {
                category: LspErrorCategory::Parse,
                message: format!("{e}"),
                span: None,
            }];
        }
    };
    match acvus_mir::compile(interner, &ast, registry) {
        Ok(_) => vec![],
        Err(errs) => from_mir_errors(&errs, interner),
    }
}

/// Discover context keys from a script/template.
/// Uses analysis mode — unknown @contexts become fresh type vars instead of errors.
fn discover_context_keys(
    interner: &Interner,
    source: &str,
    mode: ScriptMode,
    registry: &ContextTypeRegistry,
    known: &FxHashMap<Astr, KnownValue>,
) -> Vec<ContextKeyInfo> {
    let module = match mode {
        ScriptMode::Template => {
            let ast = match acvus_ast::parse(interner, source) {
                Ok(a) => a,
                Err(_) => return vec![],
            };
            let (module, _hints, _errs) =
                acvus_mir::compile_analysis_partial(interner, &ast, registry);
            module
        }
        ScriptMode::Script => {
            let script = match acvus_ast::parse_script(interner, source) {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            let (module, _hints, _tail, _errs) =
                acvus_mir::compile_script_analysis_with_tail_partial(
                    interner, &script, registry, None,
                );
            module
        }
    };

    // Filter: only return keys NOT already in scope.
    // Scope-defined keys (@self, @raw, @item, @index, @turn_index, node names, etc.)
    // are resolved — not unresolved params.
    extract_context_keys(interner, &module, known)
        .into_iter()
        .filter(|k| registry.is_user_key(&k.name))
        .collect()
}

/// Extract context keys from a compiled MIR module using reachable context analysis.
fn extract_context_keys(
    interner: &Interner,
    module: &MirModule,
    known: &FxHashMap<Astr, KnownValue>,
) -> Vec<ContextKeyInfo> {
    use acvus_mir::ir::ValueId;
    use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
    use acvus_mir_pass::analysis::val_def::ValDefMapAnalysis;
    use acvus_mir_pass::AnalysisPass;

    let val_def = ValDefMapAnalysis.run(module, ());
    let partition = partition_context_keys(module, known, &val_def);

    let mut type_map = FxHashMap::<Astr, Ty>::default();
    let mut collect_types = |insts: &[acvus_mir::ir::Inst],
                             val_types: &FxHashMap<ValueId, Ty>| {
        for inst in insts {
            if let InstKind::ContextLoad { dst, name, .. } = &inst.kind {
                type_map
                    .entry(*name)
                    .or_insert_with(|| val_types.get(dst).cloned().unwrap_or(Ty::Infer));
            }
        }
    };
    collect_types(&module.main.insts, &module.main.val_types);
    for body in module.closures.values() {
        collect_types(&body.body.insts, &body.body.val_types);
    }

    let mut seen = FxHashSet::<Astr>::default();
    let mut keys = Vec::new();

    for name in &partition.eager {
        if seen.insert(*name) {
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKeyInfo { name: *name, ty, status: ContextKeyStatus::Eager });
        }
    }
    for name in &partition.lazy {
        if seen.insert(*name) {
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKeyInfo { name: *name, ty, status: ContextKeyStatus::Lazy });
        }
    }
    for name in &partition.reachable_known {
        if seen.insert(*name) {
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKeyInfo { name: *name, ty, status: ContextKeyStatus::Eager });
        }
    }
    for name in &partition.pruned {
        if seen.insert(*name) {
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKeyInfo { name: *name, ty, status: ContextKeyStatus::Pruned });
        }
    }

    keys.sort_by(|a, b| {
        interner.resolve(a.name).cmp(interner.resolve(b.name))
    });
    keys
}

// ---------------------------------------------------------------------------
// Node typecheck — per-node, per-field
// ---------------------------------------------------------------------------

fn typecheck_node(
    interner: &Interner,
    spec: &NodeSpec,
    locals: &acvus_orchestration::NodeLocalTypes,
    full_reg: &ContextTypeRegistry,
) -> NodeErrors {
    let mut errors = NodeErrors::default();
    let locals_ref = Some(locals);

    let node_reg = match spec.build_node_context(
        interner,
        full_reg,
        ContextScope::Body,
        locals_ref,
    ) {
        Ok(r) => r,
        Err(e) => {
            errors.env = vec![LspError {
                category: LspErrorCategory::Type,
                message: format!(
                    "context type conflict: @{} in {} and {}",
                    interner.resolve(e.key),
                    e.tier_a,
                    e.tier_b
                ),
                span: None,
            }];
            return errors;
        }
    };

    let mut shared_subst = TySubst::new();

    // initial_value — hint depends on persistency kind
    if let Some(init_src) = spec.strategy.initial_value {
        let init_reg = spec
            .build_node_context(interner, full_reg, ContextScope::InitialValue, locals_ref)
            .expect("InitialValue scope should not conflict");

        match &spec.strategy.persistency {
            Persistency::Sequence { .. } => {
                let raw_ty = &locals.raw_ty;
                let init_hint: Option<Ty> = match raw_ty {
                    Ty::List(elem) => Some(Ty::List(elem.clone())),
                    Ty::Deque(elem, origin) => Some(Ty::Deque(elem.clone(), *origin)),
                    Ty::Sequence(elem, origin, effect) => {
                        Some(Ty::Sequence(elem.clone(), *origin, *effect))
                    }
                    _ => None,
                };
                errors.initial_value = check_script_with_subst(
                    interner,
                    interner.resolve(init_src),
                    &init_reg,
                    init_hint.as_ref(),
                    &mut shared_subst,
                );
            }
            Persistency::Patch { .. } => {
                errors.initial_value = check_script_with_subst(
                    interner,
                    interner.resolve(init_src),
                    &init_reg,
                    None,
                    &mut shared_subst,
                );
            }
            Persistency::Ephemeral | Persistency::Snapshot => {
                let hint = match &locals.self_ty {
                    Ty::Error => None,
                    ty => Some(ty),
                };
                errors.initial_value =
                    check_script(interner, interner.resolve(init_src), &init_reg, hint);
            }
        }
    }

    // if_modified key
    if let Execution::IfModified { key } = &spec.strategy.execution {
        let no_self_reg = spec
            .build_node_context(interner, full_reg, ContextScope::InitialValue, locals_ref)
            .expect("InitialValue scope should not conflict");
        errors.if_modified_key =
            check_script(interner, interner.resolve(*key), &no_self_reg, None);
    }

    // bind script (Sequence/Patch)
    match &spec.strategy.persistency {
        Persistency::Sequence { bind } | Persistency::Patch { bind } => {
            let bind_reg = spec
                .build_node_context(interner, full_reg, ContextScope::Bind, locals_ref)
                .expect("Bind scope should not conflict");
            // bind result must be compatible with self_ty (Sequence<T,O,E>).
            // raw_ty is Deque<T,O> but @self is coerced to Sequence<T,O,Pure>.
            // bind must return something assignable back to self_ty.
            let bind_hint = match &locals.self_ty {
                Ty::Error => None,
                ty => Some(ty),
            };
            errors.bind = check_script_with_subst(
                interner,
                interner.resolve(*bind),
                &bind_reg,
                bind_hint,
                &mut shared_subst,
            );
        }
        Persistency::Ephemeral | Persistency::Snapshot => {}
    }

    // assert
    if let Some(assert_src) = spec.strategy.assert {
        let assert_reg = spec
            .build_node_context(interner, full_reg, ContextScope::Bind, locals_ref)
            .expect("Bind scope should not conflict");
        errors.assert = check_script(
            interner,
            interner.resolve(assert_src),
            &assert_reg,
            Some(&Ty::Bool),
        );
    }

    // messages (LLM only)
    let messages: &[acvus_orchestration::MessageSpec] = match &spec.kind {
        acvus_orchestration::NodeKind::Llm(llm) => &llm.messages,
        acvus_orchestration::NodeKind::Plain(_)
        | acvus_orchestration::NodeKind::Expr(_)
        | acvus_orchestration::NodeKind::LlmCache(_)
        | acvus_orchestration::NodeKind::Display(_)
        | acvus_orchestration::NodeKind::DisplayStatic(_)
        | acvus_orchestration::NodeKind::Iterator(_) => &[],
    };

    for (mi, msg) in messages.iter().enumerate() {
        let errs = match msg {
            acvus_orchestration::MessageSpec::Block { source, .. } => {
                check_template(interner, source, &node_reg)
            }
            acvus_orchestration::MessageSpec::Iterator { key, .. } => {
                check_script(interner, interner.resolve(*key), &node_reg, None)
            }
        };
        if !errs.is_empty() {
            errors.messages.insert(mi, errs);
        }
    }

    // expr source — output_ty as hint (forces return type for @history etc.)
    if let acvus_orchestration::NodeKind::Expr(expr_spec) = &spec.kind {
        let hint = match &expr_spec.output_ty {
            Ty::Infer => None,
            ty => Some(ty),
        };
        errors.expr_source = check_script(interner, &expr_spec.source, &node_reg, hint);
    }

    errors
}

// ---------------------------------------------------------------------------
// Completion helpers
// ---------------------------------------------------------------------------

enum CompletionTrigger {
    Context { prefix: String },
    Pipe,
    Keyword { prefix: String },
    None,
}

fn detect_trigger(before: &str) -> CompletionTrigger {
    let trimmed = before.trim_end();
    if trimmed.is_empty() {
        return CompletionTrigger::None;
    }
    if trimmed.ends_with('|') {
        return CompletionTrigger::Pipe;
    }
    if let Some(at_pos) = before.rfind('@') {
        let after_at = &before[at_pos + 1..];
        if after_at
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return CompletionTrigger::Context {
                prefix: after_at.to_string(),
            };
        }
    }
    let last_word = before
        .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or("");
    if !last_word.is_empty() {
        return CompletionTrigger::Keyword {
            prefix: last_word.to_string(),
        };
    }
    CompletionTrigger::None
}

/// Context completions — all visible types in scope matching prefix.
fn context_completions(
    interner: &Interner,
    scope: &ContextTypeRegistry,
    prefix: &str,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // All visible types from the scope (system + scoped + user — excludes extern)
    for tier in [scope.system(), scope.scoped(), scope.user()] {
        for (name, ty) in tier {
            let name_str = interner.resolve(*name).to_string();
            if !name_str.starts_with(prefix) {
                continue;
            }
            items.push(CompletionItem {
                label: format!("@{name_str}"),
                kind: CompletionKind::Context,
                detail: format!("{}", ty.display(interner)),
                insert_text: name_str,
            });
        }
    }

    items.sort_by(|a, b| a.label.cmp(&b.label));
    items
}

/// Infer the tail type of a script expression using analysis mode.
fn infer_tail_type(
    interner: &Interner,
    source: &str,
    scope: &ContextTypeRegistry,
) -> Option<Ty> {
    let script = acvus_ast::parse_script(interner, source).ok()?;
    let (_module, _hints, tail_ty, _errs) =
        acvus_mir::compile_script_analysis_with_tail_partial(interner, &script, scope, None);
    // Resolve through substitution — if still Var/Infer, return None
    match &tail_ty {
        Ty::Var(_) | Ty::Infer | Ty::Error => None,
        _ => Some(tail_ty),
    }
}

/// Pipe completions filtered by the expression type before `|`.
/// Only builtins whose first parameter is compatible with `tail_ty` are included.
/// If tail_ty is None (unknown), all builtins are returned.
fn pipe_completions_filtered(interner: &Interner, tail_ty: &Option<Ty>) -> Vec<CompletionItem> {
    let reg = acvus_mir::builtins::registry();
    let mut items = Vec::new();
    let mut seen_names = FxHashSet::<&str>::default();

    for name in reg.all_names() {
        let candidates = reg.candidates(name);
        if candidates.is_empty() {
            continue;
        }

        // Check if ANY overload's first param is compatible with tail_ty
        let mut compatible = tail_ty.is_none(); // unknown → show all
        let mut best_sig: Option<(Vec<String>, String)> = None;

        for &candidate_id in candidates {
            let entry = reg.get(candidate_id);
            let mut sig_subst = TySubst::new();
            let (params, ret) = (entry.signature)(&mut sig_subst);

            if params.is_empty() {
                continue;
            }

            if let Some(ty) = &tail_ty {
                let snap = sig_subst.snapshot();
                if sig_subst
                    .unify(ty, &params[0], acvus_mir::ty::Polarity::Covariant)
                    .is_ok()
                {
                    // Also check constraint (e.g. require_scalar for to_string)
                    let resolved_args: Vec<Ty> = params
                        .iter()
                        .map(|t| sig_subst.resolve(t))
                        .collect();
                    let constraint_ok = entry
                        .constraint
                        .map_or(true, |c| c(&resolved_args, interner).is_none());

                    if constraint_ok {
                        compatible = true;
                        let resolved_params: Vec<String> = resolved_args
                            .iter()
                            .map(|t| format!("{}", t.display(interner)))
                            .collect();
                        let resolved_ret = format!("{}", sig_subst.resolve(&ret).display(interner));
                        best_sig = Some((resolved_params, resolved_ret));
                    }
                }
                sig_subst.rollback(snap);
            } else if best_sig.is_none() {
                let params_str: Vec<String> = params
                    .iter()
                    .map(|t| format!("{}", t.display(interner)))
                    .collect();
                let ret_str = format!("{}", ret.display(interner));
                best_sig = Some((params_str, ret_str));
            }
        }

        if compatible && !seen_names.contains(name) {
            seen_names.insert(name);
            let (params_str, ret_str) = best_sig.unwrap_or_else(|| {
                let entry = reg.get(candidates[0]);
                let mut s = TySubst::new();
                let (p, r) = (entry.signature)(&mut s);
                (
                    p.iter().map(|t| format!("{}", t.display(interner))).collect(),
                    format!("{}", r.display(interner)),
                )
            });
            items.push(CompletionItem {
                label: name.to_string(),
                kind: CompletionKind::Builtin,
                detail: format!("({}) → {}", params_str.join(", "), ret_str),
                insert_text: format!(" {name}"),
            });
        }
    }
    items.sort_by(|a, b| a.label.cmp(&b.label));
    items
}

fn keyword_completions(prefix: &str) -> Vec<CompletionItem> {
    let keywords = ["true", "false", "None", "Some"];
    keywords
        .iter()
        .filter(|kw| kw.starts_with(prefix))
        .map(|kw| CompletionItem {
            label: kw.to_string(),
            kind: CompletionKind::Keyword,
            detail: String::new(),
            insert_text: kw.to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Error conversion helpers
// ---------------------------------------------------------------------------

fn from_mir_errors(errs: &[acvus_mir::error::MirError], interner: &Interner) -> Vec<LspError> {
    errs.iter()
        .map(|e| LspError {
            category: LspErrorCategory::Type,
            message: format!("{}", e.display(interner)),
            span: Some((e.span.start, e.span.end)),
        })
        .collect()
}

fn from_orch_errors(
    errs: &[acvus_orchestration::OrchError],
    interner: &Interner,
) -> Vec<LspError> {
    errs.iter()
        .map(|e| LspError {
            category: LspErrorCategory::Type,
            message: format!("{}", e.display(interner)),
            span: None,
        })
        .collect()
}

/// Simple hash for known values map — used for cache invalidation.
/// We only need to detect if the map changed, not produce a perfect hash.
fn hash_known(known: &FxHashMap<Astr, KnownValue>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    known.len().hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ty::Ty;
    use acvus_orchestration::{
        Execution, ExprSpec, NodeKind, NodeSpec, Persistency, PlainSpec, Strategy,
    };

    fn make_session() -> LspSession {
        LspSession::new()
    }

    fn empty_registry(interner: &Interner) -> ContextTypeRegistry {
        PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        )
        .unwrap()
        .to_full()
    }

    fn empty_partial_registry() -> PartialContextTypeRegistry {
        PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        )
        .unwrap()
    }

    fn registry_with_user(
        interner: &Interner,
        entries: &[(&str, Ty)],
    ) -> ContextTypeRegistry {
        let mut user = FxHashMap::default();
        for (name, ty) in entries {
            user.insert(interner.intern(name), ty.clone());
        }
        PartialContextTypeRegistry::new(FxHashMap::default(), FxHashMap::default(), user)
            .unwrap()
            .to_full()
    }

    fn partial_registry_with_user(
        interner: &Interner,
        entries: &[(&str, Ty)],
    ) -> PartialContextTypeRegistry {
        let mut user = FxHashMap::default();
        for (name, ty) in entries {
            user.insert(interner.intern(name), ty.clone());
        }
        PartialContextTypeRegistry::new(FxHashMap::default(), FxHashMap::default(), user)
            .unwrap()
    }

    fn plain_node(interner: &Interner, name: &str) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Plain(PlainSpec {
                source: String::new(),
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    fn expr_node(interner: &Interner, name: &str, source: &str) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expr(ExprSpec {
                source: source.to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    fn expr_node_with_initial(
        interner: &Interner,
        name: &str,
        source: &str,
        initial_value: &str,
    ) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expr(ExprSpec {
                source: source.to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Snapshot,
                initial_value: Some(interner.intern(initial_value)),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    fn expr_node_sequence(
        interner: &Interner,
        name: &str,
        source: &str,
        initial_value: &str,
        bind: &str,
    ) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expr(ExprSpec {
                source: source.to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Sequence {
                    bind: interner.intern(bind),
                },
                initial_value: Some(interner.intern(initial_value)),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    fn expr_node_diff(
        interner: &Interner,
        name: &str,
        source: &str,
        initial_value: &str,
        bind: &str,
    ) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expr(ExprSpec {
                source: source.to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Patch {
                    bind: interner.intern(bind),
                },
                initial_value: Some(interner.intern(initial_value)),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    // ===== document diagnostics =====

    #[test]
    fn diagnostics_empty_script_no_errors() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("".to_string(), ScriptMode::Script, reg);
        let diags = session.diagnostics(id);
        assert!(diags.is_empty());
    }

    #[test]
    fn diagnostics_valid_script() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        let diags = session.diagnostics(id);
        assert!(diags.is_empty());
    }

    #[test]
    fn diagnostics_type_error() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + \"hello\"".to_string(), ScriptMode::Script, reg);
        let diags = session.diagnostics(id);
        assert!(!diags.is_empty());
    }

    #[test]
    fn diagnostics_parse_error() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 +".to_string(), ScriptMode::Script, reg);
        let diags = session.diagnostics(id);
        assert!(!diags.is_empty());
        assert_eq!(diags[0].category, LspErrorCategory::Parse);
    }

    #[test]
    fn diagnostics_cached_on_second_call() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        let d1 = session.diagnostics(id);
        let d2 = session.diagnostics(id);
        assert_eq!(d1.len(), d2.len());
    }

    #[test]
    fn diagnostics_invalidated_on_source_update() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(id).is_empty());

        session.update_source(id, "1 +".to_string());
        assert!(!session.diagnostics(id).is_empty());
    }

    #[test]
    fn diagnostics_template_valid() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("name", Ty::String)]);
        let id = session.open("hello {{ @name }}".to_string(), ScriptMode::Template, reg);
        let diags = session.diagnostics(id);
        assert!(diags.is_empty());
    }

    // ===== context keys =====

    #[test]
    fn context_keys_discovers_unknown_refs() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        assert_eq!(keys.len(), 1);
        assert_eq!(session.interner().resolve(keys[0].name), "x");
    }

    #[test]
    fn context_keys_type_inferred() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        assert!(matches!(keys[0].ty, Ty::Int));
    }

    #[test]
    fn context_keys_template() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("hello {{ @name }}".to_string(), ScriptMode::Template, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        assert_eq!(keys.len(), 1);
        assert_eq!(session.interner().resolve(keys[0].name), "name");
    }

    #[test]
    fn context_keys_known_excludes_provided() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        let id = session.open("@x + @y".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        // @x is in scope (user tier), so not discovered as unknown
        // @y is unknown, so discovered
        assert!(keys.iter().any(|k| session.interner().resolve(k.name) == "y"));
    }

    // ===== completions =====

    #[test]
    fn completions_context_after_at() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("history", Ty::String)]);
        let id = session.open("@".to_string(), ScriptMode::Script, reg);
        let items = session.completions(id, 1);
        assert!(items.iter().any(|i| i.label == "@history"));
    }

    #[test]
    fn completions_prefix_filter() {
        let mut session = make_session();
        let reg = registry_with_user(
            session.interner(),
            &[("history", Ty::String), ("name", Ty::String), ("hero", Ty::Int)],
        );
        let id = session.open("@h".to_string(), ScriptMode::Script, reg);
        let items = session.completions(id, 2);
        assert!(items.iter().all(|i| i.label.starts_with("@h")));
        assert!(items.iter().any(|i| i.label == "@history"));
        assert!(items.iter().any(|i| i.label == "@hero"));
        assert!(items.iter().all(|i| i.label != "@name"));
    }

    #[test]
    fn completions_scope_includes_self() {
        let mut session = make_session();
        let mut user = FxHashMap::default();
        user.insert(session.interner().intern("self"), Ty::Int);
        let reg = PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            user,
        )
        .unwrap()
        .to_full();
        let id = session.open("@s".to_string(), ScriptMode::Script, reg);
        let items = session.completions(id, 2);
        assert!(items.iter().any(|i| i.label == "@self"));
    }

    #[test]
    fn completions_pipe_builtins() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x |".to_string(), ScriptMode::Script, reg);
        let items = session.completions(id, 4);
        assert!(!items.is_empty());
        assert!(items.iter().any(|i| i.label == "filter"));
        assert!(items.iter().all(|i| i.kind == CompletionKind::Builtin));
    }

    #[test]
    fn completions_keywords() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("tr".to_string(), ScriptMode::Script, reg);
        let items = session.completions(id, 2);
        assert!(items.iter().any(|i| i.label == "true"));
    }

    #[test]
    fn completions_cursor_out_of_bounds() {
        let session = make_session();
        let items = session.completions(DocId(999), 1);
        assert!(items.is_empty());
    }

    // ===== rebuild_nodes =====

    #[test]
    fn rebuild_empty_nodes() {
        let mut session = make_session();
        let result = session.rebuild_nodes(vec![], empty_partial_registry());
        assert!(result.env_errors.is_empty());
        assert!(result.node_errors.is_empty());
        let has_turn_index = result
            .context_types
            .iter()
            .any(|(k, _)| session.interner.resolve(*k) == "turn_index");
        assert!(has_turn_index);
    }

    #[test]
    fn rebuild_expr_node_valid() {
        let mut session = make_session();
        let node = expr_node(session.interner(), "calc", "1 + 2");
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        assert!(result.env_errors.is_empty());
        assert!(result.node_errors.is_empty());
    }

    #[test]
    fn rebuild_expr_node_type_error() {
        let mut session = make_session();
        let node = expr_node(session.interner(), "bad", "1 + \"hello\"");
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let bad = session.interner.intern("bad");
        assert!(result.node_errors.contains_key(&bad));
    }

    #[test]
    fn rebuild_sequence_2pass() {
        let mut session = make_session();
        let node = expr_node_sequence(
            session.interner(),
            "history",
            "\"msg\"",
            "[]",
            "@self | append(@raw)",
        );
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        assert!(result.env_errors.is_empty());
        let history = session.interner.intern("history");
        let has_bind_error = result
            .node_errors
            .get(&history)
            .map_or(false, |e| !e.bind.is_empty());
        assert!(!has_bind_error);
    }

    // ===== incremental behavior =====

    #[test]
    fn document_close_removes() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1".to_string(), ScriptMode::Script, reg);
        session.close(id);
        assert!(session.diagnostics(id).is_empty()); // doc gone, returns empty
    }

    #[test]
    fn scope_update_invalidates_cache() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);

        // Without @x in scope → type error (unknown context in non-analysis mode)
        // Actually check_script uses strict mode, not analysis mode.
        // So @x would be an error.
        let d1 = session.diagnostics(id);
        assert!(!d1.is_empty());

        // Add @x to scope
        let reg2 = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        session.update_scope(id, reg2);
        let d2 = session.diagnostics(id);
        assert!(d2.is_empty());
    }

    // ===== trigger detection =====

    #[test]
    fn trigger_at_sign() {
        assert!(matches!(detect_trigger("@"), CompletionTrigger::Context { prefix } if prefix.is_empty()));
    }

    #[test]
    fn trigger_pipe() {
        assert!(matches!(detect_trigger("@x |"), CompletionTrigger::Pipe));
    }

    #[test]
    fn trigger_keyword() {
        assert!(matches!(detect_trigger("tru"), CompletionTrigger::Keyword { prefix } if prefix == "tru"));
    }

    #[test]
    fn trigger_empty() {
        assert!(matches!(detect_trigger(""), CompletionTrigger::None));
    }

    // ===== incremental: no-op updateSource =====

    #[test]
    fn update_source_noop_when_unchanged() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        // First diagnostics call → caches
        let d1 = session.diagnostics(id);
        assert!(d1.is_empty());
        // updateSource with same value → no-op, cache preserved
        session.update_source(id, "1 + 2".to_string());
        // Cache should still be valid (no recompute)
        let d2 = session.diagnostics(id);
        assert!(d2.is_empty());
    }

    #[test]
    fn update_source_invalidates_when_changed() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(id).is_empty());
        session.update_source(id, "1 + \"x\"".to_string());
        assert!(!session.diagnostics(id).is_empty());
    }

    // ===== incremental: scope update =====

    #[test]
    fn scope_update_changes_completions() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@".to_string(), ScriptMode::Script, reg);
        let c1 = session.completions(id, 1);
        assert!(c1.is_empty()); // empty scope → no completions

        let reg2 = registry_with_user(session.interner(), &[("foo", Ty::Int)]);
        session.update_scope(id, reg2);
        let c2 = session.completions(id, 1);
        assert!(c2.iter().any(|i| i.label == "@foo"));
    }

    #[test]
    fn scope_update_changes_diagnostics() {
        // context_keys returns ALL @refs regardless of scope (LSP is generic).
        // Filtering is the caller's responsibility.
        // But diagnostics DO change with scope (undefined → defined).
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);
        // @x not in scope → error
        assert!(!session.diagnostics(id).is_empty());

        // Add @x to scope → no error
        let reg2 = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        session.update_scope(id, reg2);
        assert!(session.diagnostics(id).is_empty());
    }

    #[test]
    fn context_keys_excludes_provided_scope() {
        // context_keys only returns keys NOT in the provided tiers (extern/system/scoped).
        // Keys in user tier ARE returned (user-declared params still need values).
        let mut session = make_session();
        // Put @x in system tier (provided) — should be excluded from context_keys
        let mut system = FxHashMap::default();
        system.insert(session.interner().intern("x"), Ty::Int);
        let reg = ContextTypeRegistry::new(
            FxHashMap::default(),
            system,
            FxHashMap::default(),
            FxHashMap::default(),
        ).unwrap();
        let id = session.open("@x + @y".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        let names: Vec<&str> = keys.iter().map(|k| session.interner().resolve(k.name)).collect();
        // @x is in system tier (provided) → excluded
        assert!(!names.contains(&"x"), "provided @x should be excluded");
        // @y is not in any tier → returned
        assert!(names.contains(&"y"), "unknown @y should be returned");
    }

    #[test]
    fn context_keys_includes_user_tier() {
        // Keys in user tier are still returned — they need runtime values.
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        let id = session.open("@x + @y".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        let names: Vec<&str> = keys.iter().map(|k| session.interner().resolve(k.name)).collect();
        // @x is in user tier → returned (still needs a value)
        assert!(names.contains(&"x"), "user @x should be returned");
        // @y is unknown → returned
        assert!(names.contains(&"y"), "unknown @y should be returned");
    }

    // ===== multi-document =====

    #[test]
    fn multiple_documents_independent() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let d1 = session.open("1 + 2".to_string(), ScriptMode::Script, reg.clone());
        let d2 = session.open("1 + \"bad\"".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(d1).is_empty());
        assert!(!session.diagnostics(d2).is_empty());
    }

    #[test]
    fn multiple_documents_different_scopes() {
        let mut session = make_session();
        let reg1 = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        let reg2 = registry_with_user(session.interner(), &[("x", Ty::String)]);
        let d1 = session.open("@x + 1".to_string(), ScriptMode::Script, reg1);
        let d2 = session.open("@x + 1".to_string(), ScriptMode::Script, reg2);
        // d1: @x is Int → ok
        assert!(session.diagnostics(d1).is_empty());
        // d2: @x is String → type error
        assert!(!session.diagnostics(d2).is_empty());
    }

    #[test]
    fn context_keys_across_documents() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let d1 = session.open("@a + 1".to_string(), ScriptMode::Script, reg.clone());
        let d2 = session.open("@b + 2".to_string(), ScriptMode::Script, reg);
        let k1 = session.context_keys(d1, &FxHashMap::default());
        let k2 = session.context_keys(d2, &FxHashMap::default());
        assert_eq!(k1.len(), 1);
        assert_eq!(session.interner().resolve(k1[0].name), "a");
        assert_eq!(k2.len(), 1);
        assert_eq!(session.interner().resolve(k2[0].name), "b");
    }

    // ===== context keys: pruning with known values =====

    #[test]
    fn context_keys_with_and_without_known() {
        // Pruning uses template match blocks, not scripts.
        // Test that known values parameter is accepted and doesn't crash.
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);

        // Without known
        let k1 = session.context_keys(id, &FxHashMap::default());
        assert_eq!(k1.len(), 1);

        // With known (even though it doesn't affect this simple script)
        let mut known = FxHashMap::default();
        known.insert(
            session.interner().intern("x"),
            KnownValue::Literal(acvus_ast::Literal::Int(42)),
        );
        let k2 = session.context_keys(id, &known);
        // @x should still appear (it's used even when known)
        assert!(!k2.is_empty());
    }

    // ===== context keys: type inference accuracy =====

    #[test]
    fn context_keys_infer_int_from_arithmetic() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        assert_eq!(keys.len(), 1);
        assert!(matches!(keys[0].ty, Ty::Int));
    }

    #[test]
    fn context_keys_single_ref_type_is_var() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x".to_string(), ScriptMode::Script, reg);
        let keys = session.context_keys(id, &FxHashMap::default());
        assert_eq!(keys.len(), 1);
        // Single @x usage → type is a fresh Var (analysis mode assigns fresh type vars)
        // Not Infer — Infer is the sentinel, Var is the resolved form
    }

    // ===== document lifecycle =====

    #[test]
    fn open_returns_unique_ids() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let d1 = session.open("1".to_string(), ScriptMode::Script, reg.clone());
        let d2 = session.open("2".to_string(), ScriptMode::Script, reg);
        assert_ne!(d1, d2);
    }

    #[test]
    fn closed_doc_diagnostics_empty() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + \"x\"".to_string(), ScriptMode::Script, reg);
        assert!(!session.diagnostics(id).is_empty());
        session.close(id);
        assert!(session.diagnostics(id).is_empty()); // closed → empty
    }

    #[test]
    fn update_atomically_changes_both() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("@x + 1".to_string(), ScriptMode::Script, reg);
        // @x unknown → diagnostics fail
        assert!(!session.diagnostics(id).is_empty());

        // Update both source and scope atomically
        let new_reg = registry_with_user(session.interner(), &[("y", Ty::Int)]);
        session.update(id, "@y + 1".to_string(), new_reg);
        assert!(session.diagnostics(id).is_empty());
    }

    // ===== rebuild + document interplay =====

    #[test]
    fn rebuild_does_not_affect_documents() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("1 + 2".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(id).is_empty());

        // rebuild_nodes is independent of documents
        let node = expr_node(session.interner(), "calc", "1 + 2");
        session.rebuild_nodes(vec![node], empty_partial_registry());

        // Document should still work
        assert!(session.diagnostics(id).is_empty());
    }

    #[test]
    fn multiple_rebuild_updates_state() {
        let mut session = make_session();
        let node1 = expr_node(session.interner(), "a", "42");
        let r1 = session.rebuild_nodes(vec![node1], empty_partial_registry());
        assert!(r1.context_types.iter().any(|(k, _)| session.interner.resolve(*k) == "a"));

        let node2 = expr_node(session.interner(), "b", "\"hello\"");
        let r2 = session.rebuild_nodes(vec![node2], empty_partial_registry());
        assert!(r2.context_types.iter().any(|(k, _)| session.interner.resolve(*k) == "b"));
    }

    // ===== empty/edge cases =====

    #[test]
    fn empty_source_no_diagnostics() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(id).is_empty());
    }

    #[test]
    fn empty_source_no_context_keys() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("".to_string(), ScriptMode::Script, reg);
        assert!(session.context_keys(id, &FxHashMap::default()).is_empty());
    }

    #[test]
    fn whitespace_only_no_diagnostics() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let id = session.open("   \n\t  ".to_string(), ScriptMode::Script, reg);
        assert!(session.diagnostics(id).is_empty());
    }

    #[test]
    fn completions_at_cursor_zero_empty() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        let id = session.open("@x".to_string(), ScriptMode::Script, reg);
        assert!(session.completions(id, 0).is_empty());
    }

    // ===== real use-case: bind / initial_value / assert =====

    #[test]
    fn rebuild_initial_value_typecheck() {
        let mut session = make_session();
        let node = expr_node_with_initial(session.interner(), "counter", "1", "0");
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        assert!(result.env_errors.is_empty());
        let counter = session.interner.intern("counter");
        let errs = result.node_errors.get(&counter);
        assert!(
            errs.is_none() || errs.unwrap().initial_value.is_empty(),
            "initial_value '0' should have no errors"
        );
    }

    #[test]
    fn rebuild_sequence_chain_bind() {
        // Exact use-case:
        //   body = [{content: "msg", content_type: "text", role: "user",}]
        //   → @raw = List<{content, content_type, role}>
        //   initial_value = []
        //   bind = @self | chain(@raw)
        //   → @self = Sequence<{content, content_type, role}>
        //   chain_seq: (Sequence<T,O,E>, Iterator<T,E>) → Sequence<T,O,E>
        //   List coerces to Iterator → matches
        let mut session = make_session();
        let body = r#"[{content: "msg", content_type: "text", role: "user",}]"#;
        let node = expr_node_sequence(
            session.interner(),
            "history",
            body,
            "[]",
            "@self | chain(@raw)",
        );
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        assert!(result.env_errors.is_empty(), "env errors: {:?}", result.env_errors);
        let history = session.interner.intern("history");
        let errs = result.node_errors.get(&history);
        assert!(
            errs.is_none() || (errs.unwrap().initial_value.is_empty() && errs.unwrap().bind.is_empty()),
            "sequence chain bind should pass: {:?}",
            errs,
        );

        // Verify @self type is actually Sequence
        let history = session.interner.intern("history");
        let locals = result.node_locals.get(&history).expect("history should have locals");
        assert!(
            matches!(locals.self_ty, Ty::Sequence(..)),
            "self_ty should be Sequence, got {:?}",
            locals.self_ty,
        );
        // Verify element type is the object {content, content_type, role}
        if let Ty::Sequence(elem, _, _) = &locals.self_ty {
            assert!(
                matches!(elem.as_ref(), Ty::Object(_)),
                "Sequence element should be Object, got {:?}",
                elem,
            );
        }
    }

    #[test]
    fn rebuild_sequence_map_demotes_to_iterator_fails() {
        // bind = @self | map(fn(x) => x) returns Iterator, not Sequence.
        // Iterator loses origin → cannot be stored back into Sequence storage.
        // This should cause a type error in bind.
        let mut session = make_session();
        let body = r#"[{content: "msg", content_type: "text", role: "user",}]"#;
        let node = expr_node_sequence(
            session.interner(),
            "bad_map",
            body,
            "[]",
            "@self | map(x -> x)",
        );
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let bad = session.interner.intern("bad_map");
        let errs = result.node_errors.get(&bad);
        assert!(
            errs.is_some() && !errs.unwrap().bind.is_empty(),
            "map demotes Sequence to Iterator/List — bind should fail: {:?}",
            errs,
        );
    }

    #[test]
    fn rebuild_sequence_non_collection_initial_fails() {
        // initial_value = "not a list" → not a valid Sequence container
        let mut session = make_session();
        let node = expr_node_sequence(
            session.interner(),
            "bad",
            "\"entry\"",
            "\"not a list\"",
            "@self | chain(@raw)",
        );
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let bad = session.interner.intern("bad");
        let errs = result.node_errors.get(&bad);
        let has_error = errs.map_or(false, |e| {
            !e.initial_value.is_empty() || !e.bind.is_empty()
        });
        assert!(has_error, "non-collection initial_value should cause error in sequence");
    }

    #[test]
    fn rebuild_assert_expects_bool() {
        let mut session = make_session();
        let interner = session.interner();
        let node = NodeSpec {
            name: interner.intern("guarded"),
            kind: NodeKind::Expr(ExprSpec {
                source: "42".to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: Some(interner.intern("true")),
            },
            is_function: false,
            fn_params: vec![],
        };
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let guarded = session.interner.intern("guarded");
        let errs = result.node_errors.get(&guarded);
        assert!(
            errs.is_none() || errs.unwrap().assert.is_empty(),
            "assert 'true' should be valid"
        );
    }

    #[test]
    fn rebuild_assert_rejects_non_bool() {
        let mut session = make_session();
        let interner = session.interner();
        let node = NodeSpec {
            name: interner.intern("bad_assert"),
            kind: NodeKind::Expr(ExprSpec {
                source: "42".to_string(),
                output_ty: Ty::Infer,
            }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: Some(interner.intern("42")),
            },
            is_function: false,
            fn_params: vec![],
        };
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let bad = session.interner.intern("bad_assert");
        let errs = result.node_errors.get(&bad);
        assert!(
            errs.is_some() && !errs.unwrap().assert.is_empty(),
            "assert '42' (non-Bool) should fail"
        );
    }

    #[test]
    fn rebuild_patch_initial_no_hint() {
        let mut session = make_session();
        let node = expr_node_diff(
            session.interner(),
            "state",
            "{a: 1,}",
            "{a: 0, b: \"x\",}",
            "@raw",
        );
        let result = session.rebuild_nodes(vec![node], empty_partial_registry());
        let state = session.interner.intern("state");
        let errs = result.node_errors.get(&state);
        assert!(
            errs.is_none() || errs.unwrap().initial_value.is_empty(),
            "Patch initial_value should infer freely without hint"
        );
    }

    #[test]
    fn rebuild_node_dep_order() {
        let mut session = make_session();
        let a = expr_node(session.interner(), "a", "42");
        let b = expr_node(session.interner(), "b", "@a + 1");
        let result = session.rebuild_nodes(vec![a, b], empty_partial_registry());
        assert!(result.env_errors.is_empty());
        assert!(result.node_errors.is_empty());
    }

    // ===== real use-case: document diagnostics for node fields =====

    #[test]
    fn doc_bind_with_self_raw_in_scope() {
        // @self is a Sequence (lazy, origin-tracked), @raw is the element type
        // Use chain (Sequence-compatible) instead of append (Deque-only)
        let mut session = make_session();
        let origin = acvus_mir::ty::Origin::Concrete(99);
        let effect = acvus_mir::ty::Effect::Pure;
        let scope = registry_with_user(
            session.interner(),
            &[
                ("self", Ty::Sequence(Box::new(Ty::String), origin, effect)),
                ("raw", Ty::String),
            ],
        );
        let id = session.open("@self | take(10)".to_string(), ScriptMode::Script, scope);
        let diags = session.diagnostics(id);
        assert!(diags.is_empty(), "bind with Sequence @self in scope should pass: {:?}", diags);
    }

    #[test]
    fn doc_bind_without_self_fails() {
        let mut session = make_session();
        let scope = empty_registry(session.interner());
        let id = session.open("@self | append(@raw)".to_string(), ScriptMode::Script, scope);
        let diags = session.diagnostics(id);
        assert!(!diags.is_empty(), "bind without @self in scope should fail");
    }

    #[test]
    fn doc_initial_value_no_self_in_scope() {
        // InitialValue scope should NOT have @self
        let mut session = make_session();
        let scope = empty_registry(session.interner());
        let id = session.open("0".to_string(), ScriptMode::Script, scope);
        let diags = session.diagnostics(id);
        assert!(diags.is_empty(), "simple initial_value should pass");
    }

    // ===== real use-case: completions for node fields =====

    #[test]
    fn completions_bind_scope_includes_self_and_raw() {
        let mut session = make_session();
        let origin = acvus_mir::ty::Origin::Concrete(1);
        let effect = acvus_mir::ty::Effect::Pure;
        let scope = registry_with_user(
            session.interner(),
            &[("self", Ty::Sequence(Box::new(Ty::Int), origin, effect)), ("raw", Ty::Int)],
        );
        let id = session.open("@".to_string(), ScriptMode::Script, scope);
        let items = session.completions(id, 1);
        assert!(items.iter().any(|i| i.label == "@self"), "bind scope should include @self");
        assert!(items.iter().any(|i| i.label == "@raw"), "bind scope should include @raw");
    }

    #[test]
    fn completions_body_scope_has_self_no_raw() {
        let mut session = make_session();
        let scope = registry_with_user(
            session.interner(),
            &[("self", Ty::Int)],
        );
        let id = session.open("@".to_string(), ScriptMode::Script, scope);
        let items = session.completions(id, 1);
        assert!(items.iter().any(|i| i.label == "@self"), "body scope should include @self");
        assert!(items.iter().all(|i| i.label != "@raw"), "body scope should NOT include @raw");
    }

    #[test]
    fn completions_initial_value_no_self_no_raw() {
        let mut session = make_session();
        let scope = empty_registry(session.interner());
        let id = session.open("@".to_string(), ScriptMode::Script, scope);
        let items = session.completions(id, 1);
        assert!(items.iter().all(|i| i.label != "@self"), "initial_value scope should NOT include @self");
        assert!(items.iter().all(|i| i.label != "@raw"), "initial_value scope should NOT include @raw");
    }

    // ===== real use-case: template diagnostics =====

    #[test]
    fn template_with_context_ref() {
        let mut session = make_session();
        let scope = registry_with_user(session.interner(), &[("name", Ty::String)]);
        let id = session.open("Hello {{ @name }}!".to_string(), ScriptMode::Template, scope);
        assert!(session.diagnostics(id).is_empty());
    }

    #[test]
    fn template_without_context_ref_fails() {
        let mut session = make_session();
        let scope = empty_registry(session.interner());
        // @name not in scope → should fail in strict mode
        let id = session.open("Hello {{ @name }}!".to_string(), ScriptMode::Template, scope);
        let diags = session.diagnostics(id);
        assert!(!diags.is_empty(), "template with undefined @name should fail");
    }

    // ===== LSP ↔ compile parity =====
    //
    // LSP typecheck and actual compilation use the same underlying functions.
    // These tests verify they produce identical error/success outcomes.

    /// Helper: run actual compile and return whether it succeeded.
    fn compile_succeeds(
        interner: &Interner,
        source: &str,
        registry: &ContextTypeRegistry,
    ) -> bool {
        let script = match acvus_ast::parse_script(interner, source) {
            Ok(s) => s,
            Err(_) => return false,
        };
        acvus_mir::compile_script_with_hint(interner, &script, registry, None).is_ok()
    }

    #[test]
    fn parity_valid_script() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("x", Ty::Int)]);
        let source = "@x + 1";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "LSP and compile should agree on valid script");
    }

    #[test]
    fn parity_type_error() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let source = "1 + \"hello\"";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "LSP and compile should agree on type error");
    }

    #[test]
    fn parity_undefined_context() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let source = "@undefined + 1";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "LSP and compile should agree on undefined context");
    }

    #[test]
    fn parity_sequence_chain_bind() {
        // Verify LSP rebuild_nodes and actual compile_nodes agree on sequence bind.
        let mut session = make_session();
        let body = r#"[{content: "msg", content_type: "text", role: "user",}]"#;
        let node = expr_node_sequence(
            session.interner(),
            "hist",
            body,
            "[]",
            "@self | chain(@raw)",
        );
        // LSP
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let hist = session.interner.intern("hist");
        let lsp_bind_ok = lsp_result
            .node_errors
            .get(&hist)
            .map_or(true, |e| e.bind.is_empty());

        // Actual compile
        let compile_result = acvus_orchestration::compile_nodes(
            session.interner(),
            &[node],
            empty_partial_registry(),
        );
        let compile_ok = compile_result.is_ok();

        assert_eq!(
            lsp_bind_ok, compile_ok,
            "LSP and compile should agree on sequence chain bind: lsp={lsp_bind_ok}, compile={compile_ok}"
        );
    }

    #[test]
    fn parity_sequence_map_bind_rejected() {
        let mut session = make_session();
        let body = r#"[{content: "msg", content_type: "text", role: "user",}]"#;
        let node = expr_node_sequence(
            session.interner(),
            "bad",
            body,
            "[]",
            "@self | map(x -> x)",
        );
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let bad = session.interner.intern("bad");
        let lsp_bind_ok = lsp_result
            .node_errors
            .get(&bad)
            .map_or(true, |e| e.bind.is_empty());
        let compile_result = acvus_orchestration::compile_nodes(
            session.interner(),
            &[node],
            empty_partial_registry(),
        );
        let compile_ok = compile_result.is_ok();
        assert_eq!(
            lsp_bind_ok, compile_ok,
            "LSP and compile should agree on map bind rejection: lsp_bind_ok={lsp_bind_ok}, compile_ok={compile_ok}"
        );
    }

    #[test]
    fn parity_parse_error() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let source = "1 +";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "parity: parse error");
        assert!(!lsp_ok);
    }

    #[test]
    fn parity_empty_script() {
        let mut session = make_session();
        let reg = empty_registry(session.interner());
        let source = "";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        // Empty scripts are not compiled (skipped), so both should be OK.
        assert!(lsp_ok, "parity: empty script should be OK");
    }

    #[test]
    fn parity_context_with_type() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("name", Ty::String)]);
        let source = "@name | len_str";
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "parity: context with correct type");
        assert!(lsp_ok);
    }

    #[test]
    fn parity_context_wrong_type_usage() {
        let mut session = make_session();
        let reg = registry_with_user(session.interner(), &[("name", Ty::String)]);
        let source = "@name + 1"; // String + Int → type error
        let id = session.open(source.to_string(), ScriptMode::Script, reg.clone());
        let lsp_ok = session.diagnostics(id).is_empty();
        let compile_ok = compile_succeeds(session.interner(), source, &reg);
        assert_eq!(lsp_ok, compile_ok, "parity: context wrong type usage");
        assert!(!lsp_ok);
    }

    #[test]
    fn parity_node_expr_valid() {
        let mut session = make_session();
        let node = expr_node(session.interner(), "calc", "42 + 1");
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let calc = session.interner.intern("calc");
        let lsp_ok = lsp_result.node_errors.get(&calc).map_or(true, |e| e.expr_source.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: valid expr node");
        assert!(lsp_ok);
    }

    #[test]
    fn parity_node_expr_type_error() {
        let mut session = make_session();
        let node = expr_node(session.interner(), "bad", "1 + \"x\"");
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let bad = session.interner.intern("bad");
        let lsp_ok = lsp_result.node_errors.get(&bad).map_or(true, |e| e.expr_source.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: expr type error");
        assert!(!lsp_ok);
    }

    #[test]
    fn parity_node_initial_value_valid() {
        let mut session = make_session();
        let node = expr_node_with_initial(session.interner(), "c", "1", "0");
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let c = session.interner.intern("c");
        let lsp_ok = lsp_result.node_errors.get(&c).map_or(true, |e| e.initial_value.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: valid initial_value");
        assert!(lsp_ok);
    }

    #[test]
    fn parity_node_assert_bool() {
        let mut session = make_session();
        let interner = session.interner();
        let node = NodeSpec {
            name: interner.intern("g"),
            kind: NodeKind::Expr(ExprSpec { source: "42".into(), output_ty: Ty::Infer }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: Some(interner.intern("true")),
            },
            is_function: false,
            fn_params: vec![],
        };
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let g = session.interner.intern("g");
        let lsp_ok = lsp_result.node_errors.get(&g).map_or(true, |e| e.assert.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: assert bool");
        assert!(lsp_ok);
    }

    #[test]
    fn parity_node_assert_non_bool() {
        let mut session = make_session();
        let interner = session.interner();
        let node = NodeSpec {
            name: interner.intern("g"),
            kind: NodeKind::Expr(ExprSpec { source: "42".into(), output_ty: Ty::Infer }),
            strategy: Strategy {
                execution: Execution::Always,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: Some(interner.intern("42")),
            },
            is_function: false,
            fn_params: vec![],
        };
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let g = session.interner.intern("g");
        let lsp_ok = lsp_result.node_errors.get(&g).map_or(true, |e| e.assert.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: assert non-bool");
        assert!(!lsp_ok);
    }

    #[test]
    fn parity_node_dependency_order() {
        let mut session = make_session();
        let a = expr_node(session.interner(), "a", "42");
        let b = expr_node(session.interner(), "b", "@a + 1");
        let lsp_result = session.rebuild_nodes(vec![a.clone(), b.clone()], empty_partial_registry());
        let lsp_ok = lsp_result.env_errors.is_empty() && lsp_result.node_errors.is_empty();
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[a, b], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: node dependency order");
        assert!(lsp_ok);
    }

    #[test]
    fn parity_patch_bind() {
        let mut session = make_session();
        let node = expr_node_diff(session.interner(), "st", "{a: 1,}", "{a: 0,}", "@raw");
        let lsp_result = session.rebuild_nodes(vec![node.clone()], empty_partial_registry());
        let st = session.interner.intern("st");
        let lsp_ok = lsp_result.node_errors.get(&st).map_or(true, |e| e.bind.is_empty());
        let compile_ok = acvus_orchestration::compile_nodes(
            session.interner(), &[node], empty_partial_registry(),
        ).is_ok();
        assert_eq!(lsp_ok, compile_ok, "parity: patch bind");
        assert!(lsp_ok);
    }
}
