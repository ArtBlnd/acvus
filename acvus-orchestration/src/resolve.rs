use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{Interpreter, RuntimeError, Stepped, Value};
use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
use acvus_utils::{Astr, ContextRequest, Coroutine, ExternCallRequest, Interner};
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, info, warn};

use crate::compile::{CompiledMessage, CompiledNode, CompiledScript, CompiledStrategy};
use crate::node::Node;
use crate::storage::Storage;

// ---------------------------------------------------------------------------
// ResolveState — bundled mutable context
// ---------------------------------------------------------------------------

/// Mutable state that flows through the resolver.
///
/// Groups the three stores that always travel together:
/// - `storage`: persistent cross-turn values (generic backend)
/// - `turn_context`: values valid for this turn only
/// - `bind_cache`: IfModified key→output cache
pub struct ResolveState<S> {
    pub storage: S,
    pub turn_context: FxHashMap<Astr, Arc<Value>>,
    pub bind_cache: FxHashMap<Astr, Vec<(Value, Arc<Value>)>>,
    /// Buffered history entries for the current turn. Flushed once at turn end.
    pub history_entries: FxHashMap<Astr, Value>,
}

impl<S> ResolveState<S>
where
    S: Storage,
{
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            turn_context: FxHashMap::default(),
            bind_cache: FxHashMap::default(),
            history_entries: FxHashMap::default(),
        }
    }

    /// Check if a name is already resolved in turn_context or storage.
    fn is_available(&self, name: Astr, interner: &Interner) -> bool {
        self.turn_context.contains_key(&name) || self.storage.get(interner.resolve(name)).is_some()
    }
}

// ---------------------------------------------------------------------------
// External resolver result
// ---------------------------------------------------------------------------

/// External resolver result with lifetime hint.
pub enum Resolved {
    /// Valid for this request only. Not cached.
    Once(Value),
    /// Valid for this turn. Cached in turn_context, discarded at turn end.
    Turn(Value),
    /// Persistent. Stored in storage, survives across turns.
    Persist(Value),
}

// ---------------------------------------------------------------------------
// Event loop types
// ---------------------------------------------------------------------------

/// Output of one coroutine step from FuturesUnordered.
struct StepOutput {
    node_idx: usize,
    coroutine: Coroutine<Value, RuntimeError>,
    stepped: Stepped<Value, RuntimeError>,
    local: FxHashMap<Astr, Arc<Value>>,
}

enum PendingRequest {
    Context(ContextRequest<Value>),
    ExternCall(ExternCallRequest<Value>),
}

/// A parked coroutine waiting for a dependency to be resolved.
struct PendingWork {
    node_idx: usize,
    coroutine: Coroutine<Value, RuntimeError>,
    local: FxHashMap<Astr, Arc<Value>>,
    request: PendingRequest,
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

/// Dependency-aware node resolver.
///
/// Uses a flat FuturesUnordered event loop to drive coroutines and resolve
/// dependencies without recursive Box::pin calls.
pub struct Resolver<'a, R, EH> {
    pub nodes: &'a [CompiledNode],
    pub node_table: &'a [Arc<dyn Node>],
    pub name_to_idx: &'a FxHashMap<Astr, usize>,
    pub resolver: &'a R,
    pub extern_handler: &'a EH,
    pub interner: &'a Interner,
}

impl<'a, R, EH> Resolver<'a, R, EH>
where
    R: AsyncFn(Astr) -> Resolved + Sync,
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    pub async fn resolve_node<S>(
        &self,
        idx: usize,
        state: &mut ResolveState<S>,
        local: FxHashMap<Astr, Arc<Value>>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let max_retries = self.nodes[idx].retry;
        let mut attempt = 0u32;
        loop {
            match self.resolve_node_impl(idx, state, local.clone()).await {
                Ok(()) => return Ok(()),
                Err(ResolveError::Runtime {
                    ref node,
                    ref error,
                }) => {
                    if attempt < max_retries {
                        attempt += 1;
                        warn!(
                            node = %node,
                            attempt = attempt,
                            max = max_retries,
                            error = %error,
                            "retrying node after runtime error",
                        );
                        continue;
                    }
                    return Err(ResolveError::Runtime {
                        node: node.clone(),
                        error: error.clone(),
                    });
                }
                Err(e) => return Err(e),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Node lifecycle: prepare → drive → finalize
    // -----------------------------------------------------------------------

    async fn resolve_node_impl<S>(
        &self,
        idx: usize,
        state: &mut ResolveState<S>,
        mut local: FxHashMap<Astr, Arc<Value>>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);
        info!(node = %node_name_str, "resolve node start");

        // === PRE-PHASES ===

        // IfModified: evaluate key, check cache
        if let CompiledStrategy::IfModified { key } = &node.strategy {
            let key_value = self.drive_script(key, &FxHashMap::default(), state).await?;
            if let Some(entries) = state.bind_cache.get(&node.name)
                && let Some((_, cached_output)) = entries.iter().find(|(v, _)| v == &key_value)
            {
                debug!(node = %node_name_str, "if_modified cache hit, skipping execution");
                state.storage.set(
                    interner.resolve(node.name).to_string(),
                    Value::clone(cached_output),
                );
                return Ok(());
            }
            debug!(node = %node_name_str, "if_modified cache miss, will execute");
            local.insert(interner.intern("bind"), Arc::new(key_value));
        }

        // Dependencies are resolved lazily via NeedContext in drive_with_deps.

        // initial_value: load @self
        if let Some(ref init_script) = node.self_spec.initial_value {
            let name_str = interner.resolve(node.name);
            let prev_self = if let Some(arc) = state.storage.get(name_str) {
                Value::clone(&arc)
            } else if let Some(arc) = state.turn_context.get(&node.name) {
                Value::clone(arc)
            } else {
                debug!(node = %node_name_str, "evaluating initial_value (first run)");
                self.drive_script(init_script, &FxHashMap::default(), state)
                    .await?
            };
            local.insert(interner.intern("self"), Arc::new(prev_self));
        }

        // === MAIN COROUTINE (flat event loop) ===

        debug!(node = %node_name_str, "spawning coroutine");
        let coroutine = self.node_table[idx].spawn(local.clone());
        let new_self = self
            .drive_with_deps(idx, coroutine, &local, state)
            .await?;

        // === POST-PHASES ===

        // Assert
        if let Some(ref assert_script) = node.assert {
            let mut bind_local = FxHashMap::default();
            bind_local.insert(interner.intern("self"), Arc::new(new_self.clone()));
            debug!(node = %node_name_str, "evaluating assert");
            let result = self.drive_script(assert_script, &bind_local, state).await?;
            let Value::Bool(passed) = result else {
                return Err(ResolveError::Runtime {
                    node: interner.resolve(node.name).to_string(),
                    error: RuntimeError::type_mismatch("assert", "bool", &format!("{result:?}")),
                });
            };
            if !passed {
                info!(node = %node_name_str, "assert failed, triggering retry");
                return Err(ResolveError::Runtime {
                    node: interner.resolve(node.name).to_string(),
                    error: RuntimeError::other("assert failed"),
                });
            }
        }

        // IfModified: cache
        if matches!(node.strategy, CompiledStrategy::IfModified { .. })
            && let Some(bind_val) = local.get(&interner.intern("bind"))
        {
            state
                .bind_cache
                .entry(node.name)
                .or_default()
                .push(((**bind_val).clone(), Arc::new(new_self.clone())));
        }

        // Store + history
        let name_str = interner.resolve(node.name).to_string();
        match &node.strategy {
            CompiledStrategy::Always => {
                state.turn_context.insert(node.name, Arc::new(new_self));
            }
            CompiledStrategy::OncePerTurn | CompiledStrategy::IfModified { .. } => {
                state.storage.set(name_str, new_self);
            }
            CompiledStrategy::History { history_bind } => {
                state.storage.set(name_str, new_self.clone());
                let mut hist_local = FxHashMap::default();
                hist_local.insert(interner.intern("self"), Arc::new(new_self));

                debug!(node = %node_name_str, "evaluating history_bind");
                let entry = self.drive_script(history_bind, &hist_local, state).await?;
                state.history_entries.insert(node.name, entry);
            }
        }

        info!(node = %node_name_str, "resolve node complete");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Flat event loop — FuturesUnordered
    // -----------------------------------------------------------------------

    /// Drive the root coroutine to completion, resolving dependencies via
    /// a flat FuturesUnordered event loop. No recursive Box::pin.
    async fn drive_with_deps<S>(
        &self,
        root_idx: usize,
        root_coroutine: Coroutine<Value, RuntimeError>,
        root_local: &FxHashMap<Astr, Arc<Value>>,
        state: &mut ResolveState<S>,
    ) -> Result<Value, ResolveError>
    where
        S: Storage,
    {
        let mut futs: FuturesUnordered<
            Pin<Box<dyn Future<Output = StepOutput> + Send + '_>>,
        > = FuturesUnordered::new();
        let mut pending: FxHashMap<Astr, Vec<PendingWork>> = FxHashMap::default();
        let mut in_flight: FxHashSet<usize> = FxHashSet::default();

        // Start root coroutine
        enqueue_step(&mut futs, root_idx, root_coroutine, root_local.clone());

        while let Some(output) = futs.next().await {
            let StepOutput {
                node_idx,
                coroutine,
                stepped,
                local,
            } = output;

            match stepped {
                Stepped::Emit(value) => {
                    if node_idx == root_idx {
                        return Ok(value);
                    }
                    // Dependency node completed — finalize and wake waiters
                    self.finalize_dep(node_idx, &value, state);
                    let arc = Arc::new(value);
                    let name = self.nodes[node_idx].name;
                    if let Some(waiters) = pending.remove(&name) {
                        for w in waiters {
                            match w.request {
                                PendingRequest::Context(req) => req.resolve(Arc::clone(&arc)),
                                PendingRequest::ExternCall(req) => req.resolve(Arc::clone(&arc)),
                            }
                            enqueue_step(&mut futs, w.node_idx, w.coroutine, w.local);
                        }
                    }
                }
                Stepped::NeedContext(request) => {
                    self.handle_need_context(
                        node_idx,
                        coroutine,
                        local,
                        request,
                        state,
                        &mut futs,
                        &mut pending,
                        &mut in_flight,
                    )
                    .await?;
                }
                Stepped::NeedExternCall(request) => {
                    self.handle_need_extern_call(
                        node_idx,
                        coroutine,
                        local,
                        request,
                        state,
                        &mut futs,
                        &mut pending,
                        &mut in_flight,
                    )
                    .await?;
                }
                Stepped::Done => {
                    if node_idx == root_idx {
                        warn!("root coroutine finished without emit");
                        return Ok(Value::Unit);
                    }
                    // Dep finished without emit — treat as Unit
                    let name = self.nodes[node_idx].name;
                    self.finalize_dep(node_idx, &Value::Unit, state);
                    let arc = Arc::new(Value::Unit);
                    if let Some(waiters) = pending.remove(&name) {
                        for w in waiters {
                            match w.request {
                                PendingRequest::Context(req) => req.resolve(Arc::clone(&arc)),
                                PendingRequest::ExternCall(req) => req.resolve(Arc::clone(&arc)),
                            }
                            enqueue_step(&mut futs, w.node_idx, w.coroutine, w.local);
                        }
                    }
                }
                Stepped::Error(e) => {
                    return Err(ResolveError::Runtime {
                        node: self
                            .interner
                            .resolve(self.nodes[node_idx].name)
                            .to_string(),
                        error: e,
                    });
                }
            }
        }

        // All futures drained without root emitting
        Err(ResolveError::UnresolvedContext(
            "event loop exhausted without root node completing".to_string(),
        ))
    }

    /// Handle a NeedContext from a coroutine in the event loop.
    async fn handle_need_context<S>(
        &self,
        node_idx: usize,
        coroutine: Coroutine<Value, RuntimeError>,
        local: FxHashMap<Astr, Arc<Value>>,
        request: ContextRequest<Value>,
        state: &mut ResolveState<S>,
        futs: &mut FuturesUnordered<Pin<Box<dyn Future<Output = StepOutput> + Send + '_>>>,
        pending: &mut FxHashMap<Astr, Vec<PendingWork>>,
        in_flight: &mut FxHashSet<usize>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let name = request.name();
        let name_str = self.interner.resolve(name);

        // 1. Function nodes always return ExternFn handles (before any caching).
        // Actual execution happens via NeedExternCall → ClosureCall.
        if let Some(&dep_idx) = self.name_to_idx.get(&name) {
            if self.nodes[dep_idx].is_function {
                request.resolve(Arc::new(Value::ExternFn(name)));
                enqueue_step(futs, node_idx, coroutine, local);
                return Ok(());
            }
        }

        // 2. Try local context
        if let Some(arc) = local.get(&name) {
            debug!(context = %name_str, "resolved from local");
            request.resolve(Arc::clone(arc));
            enqueue_step(futs, node_idx, coroutine, local);
            return Ok(());
        }

        // 3. Try turn_context
        if let Some(arc) = state.turn_context.get(&name).cloned() {
            debug!(context = %name_str, "resolved from turn_context");
            request.resolve(arc);
            enqueue_step(futs, node_idx, coroutine, local);
            return Ok(());
        }

        // 4. Try storage
        if let Some(arc) = state.storage.get(name_str) {
            debug!(context = %name_str, "resolved from storage");
            request.resolve(arc);
            enqueue_step(futs, node_idx, coroutine, local);
            return Ok(());
        }

        // 5. Is it a node?
        if let Some(&dep_idx) = self.name_to_idx.get(&name) {

            let needs_spawn = !in_flight.contains(&dep_idx) && self.needs_resolve(dep_idx, state);

            if needs_spawn {
                debug!(context = %name_str, "spawning dependency node");
                if let Some(dep_co) = self.prepare_dep(dep_idx, state, FxHashMap::default()).await? {
                    in_flight.insert(dep_idx);
                    enqueue_step(futs, dep_idx, dep_co, FxHashMap::default());
                } else {
                    let value = self.lookup(name, state).await?;
                    request.resolve(value);
                    enqueue_step(futs, node_idx, coroutine, local);
                    return Ok(());
                }
            }

            pending.entry(name).or_default().push(PendingWork {
                node_idx,
                coroutine,
                local,
                request: PendingRequest::Context(request),
            });
            return Ok(());
        }

        // 5. External resolver
        debug!(context = %name_str, "calling external resolver");
        let value = self.lookup(name, state).await?;
        request.resolve(value);
        enqueue_step(futs, node_idx, coroutine, local);
        Ok(())
    }

    /// Handle a NeedExternCall from a coroutine in the event loop.
    async fn handle_need_extern_call<S>(
        &self,
        node_idx: usize,
        coroutine: Coroutine<Value, RuntimeError>,
        local: FxHashMap<Astr, Arc<Value>>,
        request: ExternCallRequest<Value>,
        state: &mut ResolveState<S>,
        futs: &mut FuturesUnordered<Pin<Box<dyn Future<Output = StepOutput> + Send + '_>>>,
        pending: &mut FxHashMap<Astr, Vec<PendingWork>>,
        in_flight: &mut FxHashSet<usize>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let name = request.name();
        let name_str = self.interner.resolve(name);

        // 1. Is it a node? (tool call — args passed as Object in first arg)
        if let Some(&dep_idx) = self.name_to_idx.get(&name) {
            let args = request.args().to_vec();
            let node = &self.nodes[dep_idx];
            let dep_local: FxHashMap<Astr, Arc<Value>> = if node.is_function {
                // Function call: positional args → param names
                node.fn_params.iter()
                    .zip(args.into_iter())
                    .map(|((name, _), val)| (*name, Arc::new(val)))
                    .collect()
            } else {
                // Tool call: first arg as Object
                if let Some(Value::Object(obj)) = args.first() {
                    obj.iter().map(|(k, v)| (*k, Arc::new(v.clone()))).collect()
                } else {
                    FxHashMap::default()
                }
            };

            debug!(context = %name_str, "spawning tool node via extern call");
            if let Some(dep_co) = self.prepare_dep(dep_idx, state, dep_local.clone()).await? {
                enqueue_step(futs, dep_idx, dep_co, dep_local);
            } else {
                // IfModified cache hit
                let value = self.lookup(name, state).await?;
                request.resolve(value);
                enqueue_step(futs, node_idx, coroutine, local);
                return Ok(());
            }

            pending.entry(name).or_default().push(PendingWork {
                node_idx,
                coroutine,
                local,
                request: PendingRequest::ExternCall(request),
            });
            return Ok(());
        }

        // 2. extern_handler (regex etc.)
        let args = request.args().to_vec();
        debug!(context = %name_str, "calling extern_handler");
        match (self.extern_handler)(name, args).await {
            Ok(value) => {
                request.resolve(Arc::new(value));
                enqueue_step(futs, node_idx, coroutine, local);
            }
            Err(e) => {
                return Err(ResolveError::Runtime {
                    node: self.interner.resolve(self.nodes[node_idx].name).to_string(),
                    error: e,
                });
            }
        }
        Ok(())
    }

    /// Prepare a dependency node (pre-phases only). Returns the spawned
    /// coroutine, or None if IfModified cache hit resolved it.
    async fn prepare_dep<S>(
        &self,
        idx: usize,
        state: &mut ResolveState<S>,
        mut local: FxHashMap<Astr, Arc<Value>>,
    ) -> Result<Option<Coroutine<Value, RuntimeError>>, ResolveError>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);

        // IfModified: check cache
        if let CompiledStrategy::IfModified { key } = &node.strategy {
            let key_value = self.drive_script(key, &FxHashMap::default(), state).await?;
            if let Some(entries) = state.bind_cache.get(&node.name)
                && let Some((_, cached_output)) = entries.iter().find(|(v, _)| v == &key_value)
            {
                debug!(node = %node_name_str, "dep if_modified cache hit");
                state.storage.set(
                    interner.resolve(node.name).to_string(),
                    Value::clone(cached_output),
                );
                return Ok(None);
            }
            local.insert(interner.intern("bind"), Arc::new(key_value));
        }

        // initial_value
        if let Some(ref init_script) = node.self_spec.initial_value {
            let name_str = interner.resolve(node.name);
            let prev_self = if let Some(arc) = state.storage.get(name_str) {
                Value::clone(&arc)
            } else if let Some(arc) = state.turn_context.get(&node.name) {
                Value::clone(arc)
            } else {
                self.drive_script(init_script, &FxHashMap::default(), state)
                    .await?
            };
            local.insert(interner.intern("self"), Arc::new(prev_self));
        }

        Ok(Some(self.node_table[idx].spawn(local)))
    }

    /// Finalize a dependency node after it emits: store result.
    fn finalize_dep<S>(&self, idx: usize, value: &Value, state: &mut ResolveState<S>)
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let name_str = interner.resolve(node.name).to_string();

        match &node.strategy {
            CompiledStrategy::Always => {
                state
                    .turn_context
                    .insert(node.name, Arc::new(value.clone()));
            }
            _ => {
                state.storage.set(name_str, value.clone());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Script driver (flat, non-recursive)
    // -----------------------------------------------------------------------

    /// Drive a compiled script to completion.
    /// Resolves NeedContext from local → turn_context → storage → external.
    /// Does NOT resolve nodes — scripts must reference already-available values.
    async fn drive_script<S>(
        &self,
        script: &CompiledScript,
        local: &FxHashMap<Astr, Arc<Value>>,
        state: &mut ResolveState<S>,
    ) -> Result<Value, ResolveError>
    where
        S: Storage,
    {
        let interp = Interpreter::new(self.interner, script.module.clone());
        let mut coroutine = interp.execute();
        loop {
            match coroutine.resume().await {
                Stepped::Emit(value) => return Ok(value),
                Stepped::NeedContext(request) => {
                    let name = request.name();
                    if let Some(arc) = local.get(&name) {
                        request.resolve(Arc::clone(arc));
                    } else {
                        let value = self.lookup(name, state).await?;
                        request.resolve(value);
                    }
                }
                Stepped::NeedExternCall(request) => {
                    let name = request.name();
                    let args = request.args().to_vec();
                    match (self.extern_handler)(name, args).await {
                        Ok(value) => request.resolve(Arc::new(value)),
                        Err(e) => return Err(ResolveError::Runtime { node: String::new(), error: e }),
                    }
                }
                Stepped::Done => return Ok(Value::Unit),
                Stepped::Error(e) => {
                    return Err(ResolveError::Runtime {
                        node: String::new(),
                        error: e,
                    })
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Eager dependency resolution
    // -----------------------------------------------------------------------

    /// Compute node indices that are *definitely* needed by `nodes[idx]`.
    fn eager_node_deps<S>(&self, idx: usize, storage: &S) -> Vec<usize>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let known = node.known_from_storage(self.interner, storage);
        let mut eager = FxHashSet::default();

        for msg in node.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                let p = partition_context_keys(&block.module, &known, &block.val_def);
                eager.extend(p.eager);
            }
        }

        if node.kind.messages().is_empty() {
            match &node.kind {
                crate::kind::CompiledNodeKind::Plain(plain) => {
                    let p =
                        partition_context_keys(&plain.block.module, &known, &plain.block.val_def);
                    eager.extend(p.eager);
                }
                crate::kind::CompiledNodeKind::Expr(expr) => {
                    let p =
                        partition_context_keys(&expr.script.module, &known, &expr.script.val_def);
                    eager.extend(p.eager);
                }
                _ => {}
            }
        }

        if let Some(ref iv) = node.self_spec.initial_value
            && storage.get(self.interner.resolve(node.name)).is_none()
        {
            eager.extend(iv.context_keys.iter().copied());
        }
        match &node.strategy {
            CompiledStrategy::History { history_bind } => {
                eager.extend(history_bind.context_keys.iter().copied());
            }
            CompiledStrategy::IfModified { key } => {
                eager.extend(key.context_keys.iter().copied());
            }
            _ => {}
        }

        eager
            .iter()
            .filter_map(|name| self.name_to_idx.get(name).copied())
            .filter(|&i| i != idx)
            .collect()
    }

    /// Check if a node needs resolution.
    fn needs_resolve<S>(&self, idx: usize, state: &ResolveState<S>) -> bool
    where
        S: Storage,
    {
        let name = self.nodes[idx].name;
        match &self.nodes[idx].strategy {
            CompiledStrategy::Always => true,
            _ => !state.is_available(name, self.interner),
        }
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Look up a value: turn_context → storage → external resolver.
    async fn lookup<S>(
        &self,
        name: Astr,
        state: &mut ResolveState<S>,
    ) -> Result<Arc<Value>, ResolveError>
    where
        S: Storage,
    {
        let name_str = self.interner.resolve(name);
        if let Some(arc) = state.turn_context.get(&name) {
            debug!(name = %name_str, source = "turn_context", "lookup hit");
            return Ok(Arc::clone(arc));
        }
        if let Some(arc) = state.storage.get(name_str) {
            debug!(name = %name_str, source = "storage", "lookup hit");
            return Ok(arc);
        }
        info!(name = %name_str, "calling external resolver");
        match (self.resolver)(name).await {
            Resolved::Once(value) => {
                debug!(name = %name_str, kind = "once", "external resolver returned");
                Ok(Arc::new(value))
            }
            Resolved::Turn(value) => {
                debug!(name = %name_str, kind = "turn", "external resolver returned");
                let arc = Arc::new(value);
                state.turn_context.insert(name, Arc::clone(&arc));
                Ok(arc)
            }
            Resolved::Persist(value) => {
                debug!(name = %name_str, kind = "persist", "external resolver returned");
                let arc = Arc::new(value);
                state.storage.set(name_str.to_string(), Value::clone(&arc));
                Ok(arc)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: enqueue a step into FuturesUnordered
// ---------------------------------------------------------------------------

fn enqueue_step<'a>(
    futs: &mut FuturesUnordered<Pin<Box<dyn Future<Output = StepOutput> + Send + 'a>>>,
    node_idx: usize,
    coroutine: Coroutine<Value, RuntimeError>,
    local: FxHashMap<Astr, Arc<Value>>,
) {
    futs.push(Box::pin(async move {
        let (coroutine, stepped) = coroutine.step().await;
        StepOutput {
            node_idx,
            coroutine,
            stepped,
            local,
        }
    }));
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ResolveError {
    UnresolvedContext(String),
    Runtime { node: String, error: RuntimeError },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnresolvedContext(name) => write!(f, "unresolved context: @{name}"),
            ResolveError::Runtime { node, error } => {
                if node.is_empty() {
                    write!(f, "runtime error: {error}")
                } else {
                    write!(f, "runtime error in node '{node}': {error}")
                }
            }
        }
    }
}

impl std::error::Error for ResolveError {}
