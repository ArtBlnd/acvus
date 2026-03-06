use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, Stepped, Value};
use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
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
    pub turn_context: HashMap<String, Arc<Value>>,
    pub bind_cache: HashMap<String, Vec<(Value, Arc<Value>)>>,
}

impl<S> ResolveState<S>
where
    S: Storage,
{
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            turn_context: HashMap::new(),
            bind_cache: HashMap::new(),
        }
    }

    /// Check if a name is already resolved in turn_context or storage.
    fn is_available(&self, name: &str) -> bool {
        self.turn_context.contains_key(name) || self.storage.get(name).is_some()
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
    pub name_to_idx: &'a HashMap<String, usize>,
    pub extern_fns: &'a ExternFnRegistry,
    pub resolver: &'a R,
}

impl<'a, R> Resolver<'a, R>
where
    R: AsyncFn(String) -> Resolved + Sync,
{
    pub fn resolve_node<S>(
        &'a self,
        idx: usize,
        state: &'a mut ResolveState<S>,
        local: HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<(), ResolveError>>
    where
        S: Storage,
    {
        Box::pin(self.resolve_node_impl(idx, state, local))
    }

    async fn resolve_node_impl<S>(
        &'a self,
        idx: usize,
        state: &'a mut ResolveState<S>,
        mut local: HashMap<String, Arc<Value>>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        info!(node = %node.name, "resolve node start");

        // IfModified: evaluate key, check cache
        if let CompiledStrategy::IfModified { key } = &node.strategy {
            let key_value = self.eval_script(key, &HashMap::new(), state).await?;
            if let Some(entries) = state.bind_cache.get(&node.name)
                && let Some((_, cached_output)) = entries.iter().find(|(v, _)| v == &key_value)
            {
                debug!(node = %node.name, "if_modified cache hit, skipping execution");
                state
                    .storage
                    .set(node.name.clone(), Value::clone(cached_output));
                return Ok(());
            }
            debug!(node = %node.name, "if_modified cache miss, will execute");
            local.insert("bind".into(), Arc::new(key_value));
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
                        && !state.is_available(&self.nodes[i].name)
                })
                .collect();
            if unresolved.is_empty() {
                break;
            }
            let dep_names: Vec<&str> = unresolved
                .iter()
                .map(|&i| self.nodes[i].name.as_str())
                .collect();
            debug!(
                node = %node.name,
                deps = ?dep_names,
                "prefetching eager dependencies",
            );
            for dep_idx in unresolved {
                self.resolve_node(dep_idx, state, HashMap::new()).await?;
            }
        }

        // Spawn via Node trait
        debug!(node = %node.name, "spawning coroutine");
        let (mut coroutine, first_key) = self.node_table[idx].spawn(local.clone());
        let raw_output = self
            .eval_coroutine(&mut coroutine, first_key, &HashMap::new(), state)
            .await?;

        // Build bind local context
        let mut bind_local = HashMap::new();
        bind_local.insert("raw".into(), Arc::new(raw_output));

        let new_self = if matches!(node.strategy, CompiledStrategy::Always) {
            debug!(node = %node.name, "evaluating self_bind (always)");
            self.eval_script(&node.self_spec.self_bind, &bind_local, state)
                .await?
        } else {
            // Load previous @self (or initial_value on first run)
            let prev_self = if let Some(arc) = state.storage.get(&node.name) {
                Value::clone(&arc)
            } else if let Some(arc) = state.turn_context.get(&node.name) {
                Value::clone(arc)
            } else {
                debug!(node = %node.name, "evaluating initial_value (first run)");
                self.eval_script(&node.self_spec.initial_value, &HashMap::new(), state)
                    .await?
            };
            bind_local.insert("self".into(), Arc::new(prev_self));
            debug!(node = %node.name, "evaluating self_bind");
            self.eval_script(&node.self_spec.self_bind, &bind_local, state)
                .await?
        };

        // IfModified: cache
        if matches!(node.strategy, CompiledStrategy::IfModified { .. })
            && let Some(bind_val) = local.get("bind")
        {
            state
                .bind_cache
                .entry(node.name.clone())
                .or_default()
                .push(((**bind_val).clone(), Arc::new(new_self.clone())));
        }

        // Store + history
        match &node.strategy {
            CompiledStrategy::Always => {
                state
                    .turn_context
                    .insert(node.name.clone(), Arc::new(new_self));
            }
            CompiledStrategy::OncePerTurn | CompiledStrategy::IfModified { .. } => {
                state.storage.set(node.name.clone(), new_self);
            }
            CompiledStrategy::History { history_bind } => {
                state.storage.set(node.name.clone(), new_self.clone());
                bind_local.insert("self".into(), Arc::new(new_self));

                debug!(node = %node.name, "evaluating history_bind");
                let entry = self
                    .eval_script(history_bind, &bind_local, state)
                    .await?;
                if let Some(Value::Object(obj)) = state.storage.get_mut("history")
                    && let Some(Value::List(list)) = obj.get_mut(&node.name)
                {
                    list.push(entry);
                }
            }
        }

        info!(node = %node.name, "resolve node complete");
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
        let known = node.known_from_storage(storage);
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
                    let p = partition_context_keys(
                        &plain.block.module,
                        &known,
                        &plain.block.val_def,
                    );
                    eager.extend(p.eager);
                }
                crate::kind::CompiledNodeKind::Expr(expr) => {
                    let p = partition_context_keys(
                        &expr.script.module,
                        &known,
                        &expr.script.val_def,
                    );
                    eager.extend(p.eager);
                }
                _ => {}
            }
        }

        // self_bind/initial_value/strategy scripts: always eager
        eager.extend(node.self_spec.self_bind.context_keys.iter().cloned());
        if storage.get(&node.name).is_none() {
            eager.extend(
                node.self_spec
                    .initial_value
                    .context_keys
                    .iter()
                    .cloned(),
            );
        }
        match &node.strategy {
            CompiledStrategy::History { history_bind } => {
                eager.extend(history_bind.context_keys.iter().cloned());
            }
            CompiledStrategy::IfModified { key } => {
                eager.extend(key.context_keys.iter().cloned());
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
        let name = &self.nodes[idx].name;
        match &self.nodes[idx].strategy {
            CompiledStrategy::Always => true,
            _ => !state.is_available(name),
        }
    }

    // -----------------------------------------------------------------------
    // Context resolution
    // -----------------------------------------------------------------------

    /// Resolve a context value by name.
    /// Resolution: node → turn_context → storage → external resolver.
    fn resolve_context<S>(
        &'a self,
        name: &str,
        bindings: HashMap<String, Value>,
        state: &'a mut ResolveState<S>,
    ) -> Fut<'a, Result<Arc<Value>, ResolveError>>
    where
        S: Storage,
    {
        let name = name.to_string();
        Box::pin(async move {
            // Tool call: resolve target node with bindings as local context
            if !bindings.is_empty() {
                debug!(context = %name, "resolving context with bindings (tool call)");
                if let Some(&idx) = self.name_to_idx.get(&name) {
                    let local = bindings
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    self.resolve_node(idx, state, local).await?;
                }
                return self.lookup(&name, state).await;
            }

            // Node: resolve if needed
            if let Some(&idx) = self.name_to_idx.get(&name) {
                if !self.needs_resolve(idx, state) {
                    debug!(context = %name, "context already available");
                    return self.lookup(&name, state).await;
                }
                debug!(context = %name, "resolving context on-demand");
                self.resolve_node(idx, state, HashMap::new()).await?;
            }

            self.lookup(&name, state).await
        })
    }

    /// Look up a value: turn_context → storage → external resolver.
    async fn lookup<S>(
        &self,
        name: &str,
        state: &mut ResolveState<S>,
    ) -> Result<Arc<Value>, ResolveError>
    where
        S: Storage,
    {
        if let Some(arc) = state.turn_context.get(name) {
            debug!(name = %name, source = "turn_context", "lookup hit");
            return Ok(Arc::clone(arc));
        }
        if let Some(arc) = state.storage.get(name) {
            debug!(name = %name, source = "storage", "lookup hit");
            return Ok(arc);
        }
        info!(name = %name, "calling external resolver");
        match (self.resolver)(name.to_string()).await {
            Resolved::Once(value) => {
                debug!(name = %name, kind = "once", "external resolver returned");
                Ok(Arc::new(value))
            }
            Resolved::Turn(value) => {
                debug!(name = %name, kind = "turn", "external resolver returned");
                let arc = Arc::new(value);
                state
                    .turn_context
                    .insert(name.to_string(), Arc::clone(&arc));
                Ok(arc)
            }
            Resolved::Persist(value) => {
                debug!(name = %name, kind = "persist", "external resolver returned");
                let arc = Arc::new(value);
                state.storage.set(name.to_string(), Value::clone(&arc));
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
        coroutine: &'a mut acvus_coroutine::Coroutine<Value>,
        first_key: acvus_coroutine::ResumeKey<Value>,
        local: &'a HashMap<String, Arc<Value>>,
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
                        let name = need.name().to_string();
                        if let Some(arc) = local.get(&name) {
                            debug!(context = %name, "need_context resolved from local");
                            key = need.into_key(Arc::clone(arc));
                        } else {
                            debug!(context = %name, "need_context delegating to resolve_context");
                            let bindings = need.bindings().clone();
                            let value = self
                                .resolve_context(&name, bindings, state)
                                .await?;
                            key = need.into_key(value);
                        }
                    }
                    Stepped::Done => {
                        warn!("coroutine finished without emit");
                        return Ok(Value::Unit);
                    }
                }
            }
        })
    }

    /// Run a compiled script. Convenience over eval_coroutine.
    fn eval_script<S>(
        &'a self,
        script: &'a CompiledScript,
        local: &'a HashMap<String, Arc<Value>>,
        state: &'a mut ResolveState<S>,
    ) -> Fut<'a, Result<Value, ResolveError>>
    where
        S: Storage,
    {
        Box::pin(async move {
            let interp = Interpreter::new(script.module.clone(), self.extern_fns);
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
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnresolvedContext(name) => write!(f, "unresolved context: @{name}"),
        }
    }
}

impl std::error::Error for ResolveError {}
