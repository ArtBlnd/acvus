use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, RuntimeError, Stepped, Value};
use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use tracing::{debug, info, warn};

use crate::compile::{CompiledMessage, CompiledNode, CompiledScript, CompiledStrategy};
use crate::node::Node;
use crate::storage::Storage;

type Fut<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

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
// Resolver
// ---------------------------------------------------------------------------

/// Dependency-aware node resolver.
///
/// Before spawning a node's coroutine, eagerly pre-resolves dependencies
/// that are *definitely* needed (on unconditionally reachable code paths).
/// As each dependency resolves, the known-value set grows, potentially
/// pruning dead branches and revealing new eager dependencies.
///
/// Lazy dependencies (behind unknown branch conditions) are still resolved
/// on-demand when the coroutine emits `NeedContext`.
pub struct Resolver<'a, R> {
    pub nodes: &'a [CompiledNode],
    pub node_table: &'a [Arc<dyn Node>],
    pub name_to_idx: &'a FxHashMap<Astr, usize>,
    pub extern_fns: &'a ExternFnRegistry,
    pub resolver: &'a R,
    pub interner: &'a Interner,
}

impl<'a, R> Resolver<'a, R>
where
    R: AsyncFn(Astr) -> Resolved + Sync,
{
    pub fn resolve_node<S>(
        &'a self,
        idx: usize,
        state: &'a mut ResolveState<S>,
        local: FxHashMap<Astr, Arc<Value>>,
    ) -> Fut<'a, Result<(), ResolveError>>
    where
        S: Storage,
    {
        Box::pin(async move {
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
        })
    }

    async fn resolve_node_impl<S>(
        &'a self,
        idx: usize,
        state: &'a mut ResolveState<S>,
        mut local: FxHashMap<Astr, Arc<Value>>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);
        info!(node = %node_name_str, "resolve node start");

        // IfModified: evaluate key, check cache
        if let CompiledStrategy::IfModified { key } = &node.strategy {
            let key_value = self.eval_script(key, &FxHashMap::default(), state).await?;
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

        // Prefetch: iteratively resolve eager deps until stable.
        // Each resolved dep may reveal new eager deps via branch pruning.
        // Always-strategy nodes are skipped — they re-execute every
        // invocation and are handled on-demand via NeedContext.
        loop {
            let eager = self.eager_node_deps(idx, &state.storage);
            let unresolved: Vec<usize> = eager
                .into_iter()
                .filter(|&i| {
                    !matches!(self.nodes[i].strategy, CompiledStrategy::Always)
                        && !state.is_available(self.nodes[i].name, interner)
                })
                .collect();
            if unresolved.is_empty() {
                break;
            }
            let dep_names: Vec<&str> = unresolved
                .iter()
                .map(|&i| interner.resolve(self.nodes[i].name))
                .collect();
            debug!(
                node = %node_name_str,
                deps = ?dep_names,
                "prefetching eager dependencies",
            );
            for dep_idx in unresolved {
                self.resolve_node(dep_idx, state, FxHashMap::default())
                    .await?;
            }
        }

        // If initial_value is Some, load @self and inject into local context
        if let Some(ref init_script) = node.self_spec.initial_value {
            let name_str = interner.resolve(node.name);
            let prev_self = if let Some(arc) = state.storage.get(name_str) {
                Value::clone(&arc)
            } else if let Some(arc) = state.turn_context.get(&node.name) {
                Value::clone(arc)
            } else {
                debug!(node = %node_name_str, "evaluating initial_value (first run)");
                self.eval_script(init_script, &FxHashMap::default(), state)
                    .await?
            };
            local.insert(interner.intern("self"), Arc::new(prev_self));
        }

        // Spawn via Node trait
        debug!(node = %node_name_str, "spawning coroutine");
        let (mut coroutine, first_key) = self.node_table[idx].spawn(local.clone());
        let new_self = self
            .eval_coroutine(&mut coroutine, first_key, &FxHashMap::default(), state)
            .await?;

        // Assert: evaluate after new_self (= raw output), before storing.
        if let Some(ref assert_script) = node.assert {
            let mut bind_local = FxHashMap::default();
            bind_local.insert(interner.intern("self"), Arc::new(new_self.clone()));
            debug!(node = %node_name_str, "evaluating assert");
            let result = self.eval_script(assert_script, &bind_local, state).await?;
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
                let entry = self.eval_script(history_bind, &hist_local, state).await?;
                // Buffer entry — flushed to @turn.history at turn end.
                state.history_entries.insert(node.name, entry);
            }
        }

        info!(node = %node_name_str, "resolve node complete");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Eager dependency resolution
    // -----------------------------------------------------------------------

    /// Compute node indices that are *definitely* needed by `nodes[idx]`.
    ///
    /// Uses dead-branch analysis: context keys behind unknown branch
    /// conditions are excluded (they will be resolved lazily).
    fn eager_node_deps<S>(&self, idx: usize, storage: &S) -> Vec<usize>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let known = node.known_from_storage(self.interner, storage);
        let mut eager = HashSet::new();

        // Message blocks (Llm/LlmCache): proper eager/lazy partition
        for msg in node.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                let p = partition_context_keys(&block.module, &known, &block.val_def);
                eager.extend(p.eager);
            }
        }

        // Plain/Expr (no messages): partition via the node's main module.
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

        // initial_value/strategy scripts: always eager
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

        // Filter to node indices, exclude self
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
    // Context resolution
    // -----------------------------------------------------------------------

    /// Resolve a context value by name.
    /// Resolution: node → turn_context → storage → external resolver.
    fn resolve_context<S>(
        &'a self,
        name: Astr,
        bindings: FxHashMap<Astr, Value>,
        state: &'a mut ResolveState<S>,
    ) -> Fut<'a, Result<Arc<Value>, ResolveError>>
    where
        S: Storage,
    {
        Box::pin(async move {
            let ctx_name_str = self.interner.resolve(name);
            // Tool call: resolve target node with bindings as local context
            if !bindings.is_empty() {
                debug!(context = %ctx_name_str, "resolving context with bindings (tool call)");
                if let Some(&idx) = self.name_to_idx.get(&name) {
                    let local = bindings
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    self.resolve_node(idx, state, local).await?;
                }
                return self.lookup(name, state).await;
            }

            // Node: resolve if needed
            if let Some(&idx) = self.name_to_idx.get(&name) {
                if !self.needs_resolve(idx, state) {
                    debug!(context = %ctx_name_str, "context already available");
                    return self.lookup(name, state).await;
                }
                debug!(context = %ctx_name_str, "resolving context on-demand");
                self.resolve_node(idx, state, FxHashMap::default()).await?;
            }

            self.lookup(name, state).await
        })
    }

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

    // -----------------------------------------------------------------------
    // Coroutine evaluation
    // -----------------------------------------------------------------------

    /// Drive any coroutine to completion. The single core loop.
    /// Resolution: local → resolve_context (turn_context → storage → external).
    fn eval_coroutine<S>(
        &'a self,
        coroutine: &'a mut acvus_utils::Coroutine<Value, RuntimeError>,
        first_key: acvus_utils::ResumeKey<Value>,
        local: &'a FxHashMap<Astr, Arc<Value>>,
        state: &'a mut ResolveState<S>,
    ) -> Fut<'a, Result<Value, ResolveError>>
    where
        S: Storage,
    {
        Box::pin(async move {
            let mut key = first_key;
            loop {
                match coroutine.resume(key).await {
                    Stepped::Emit(emit) => {
                        let (value, _) = emit.into_parts();
                        return Ok(value);
                    }
                    Stepped::NeedContext(need) => {
                        let name = need.name();
                        let name_str = self.interner.resolve(name);
                        if let Some(arc) = local.get(&name) {
                            debug!(context = %name_str, "need_context resolved from local");
                            key = need.into_key(Arc::clone(arc));
                        } else {
                            debug!(context = %name_str, "need_context delegating to resolve_context");
                            let bindings = need.bindings().clone();
                            let value = self.resolve_context(name, bindings, state).await?;
                            key = need.into_key(value);
                        }
                    }
                    Stepped::Done => {
                        warn!("coroutine finished without emit");
                        return Ok(Value::Unit);
                    }
                    Stepped::Error(e) => {
                        return Err(ResolveError::Runtime {
                            node: String::new(),
                            error: e,
                        });
                    }
                }
            }
        })
    }

    /// Run a compiled script. Convenience over eval_coroutine.
    fn eval_script<S>(
        &'a self,
        script: &'a CompiledScript,
        local: &'a FxHashMap<Astr, Arc<Value>>,
        state: &'a mut ResolveState<S>,
    ) -> Fut<'a, Result<Value, ResolveError>>
    where
        S: Storage,
    {
        Box::pin(async move {
            let interp = Interpreter::new(self.interner, script.module.clone(), self.extern_fns);
            let (mut coroutine, key) = interp.execute();
            self.eval_coroutine(&mut coroutine, key, local, state).await
        })
    }
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
