use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, Stepped, Value};

use crate::compile::{CompiledNode, CompiledScript, CompiledStrategy};
use crate::node::Node;
use crate::storage::{HashMapStorage, Storage};

type Fut<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// External resolver result with lifetime hint.
pub enum Resolved {
    /// Valid for this request only. Not cached.
    Once(Value),
    /// Valid for this turn. Cached in turn_context, discarded at turn end.
    Turn(Value),
    /// Persistent. Stored in storage, survives across turns.
    Persist(Value),
}

/// Demand-driven node resolver. Starts from a target node and recursively
/// resolves dependencies via NeedContext.
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
    pub fn resolve_node(
        &'a self,
        idx: usize,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_context: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<(), ResolveError>> {
        Box::pin(self.resolve_node_impl(idx, storage, local, bind_cache, turn_context))
    }

    async fn resolve_node_impl(
        &'a self,
        idx: usize,
        storage: &'a mut HashMapStorage,
        mut local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_context: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<(), ResolveError> {
        let node = &self.nodes[idx];

        // IfModified: evaluate key, check cache
        if let CompiledStrategy::IfModified { key } = &node.strategy {
            let key_value = self
                .eval_script(key, &HashMap::new(), storage, bind_cache, turn_context)
                .await?;
            if let Some(entries) = bind_cache.get(&node.name)
                && let Some((_, cached_output)) = entries.iter().find(|(v, _)| v == &key_value)
            {
                storage.set(node.name.clone(), Value::clone(cached_output));
                return Ok(());
            }
            local.insert("bind".into(), Arc::new(key_value));
        }

        // Spawn via Node trait
        let (mut coroutine, first_key) = self.node_table[idx].spawn(local.clone());
        let raw_output = self
            .eval_coroutine(
                &mut coroutine,
                first_key,
                &HashMap::new(),
                storage,
                bind_cache,
                turn_context,
            )
            .await?;

        // Build bind local context
        let mut bind_local = HashMap::new();
        bind_local.insert("raw".into(), Arc::new(raw_output));

        let new_self = if matches!(node.strategy, CompiledStrategy::Always) {
            // Always: no @self, no initial_value — just @raw transform
            self.eval_script(
                &node.self_spec.self_bind,
                &bind_local,
                storage,
                bind_cache,
                turn_context,
            )
            .await?
        } else {
            // Load previous @self (or initial_value on first run)
            let prev_self = if let Some(arc) = storage.get(&node.name) {
                Value::clone(&arc)
            } else if let Some(arc) = turn_context.get(&node.name) {
                Value::clone(arc)
            } else {
                self.eval_script(
                    &node.self_spec.initial_value,
                    &HashMap::new(),
                    storage,
                    bind_cache,
                    turn_context,
                )
                .await?
            };
            bind_local.insert("self".into(), Arc::new(prev_self));
            self.eval_script(
                &node.self_spec.self_bind,
                &bind_local,
                storage,
                bind_cache,
                turn_context,
            )
            .await?
        };

        // IfModified: cache
        if matches!(node.strategy, CompiledStrategy::IfModified { .. })
            && let Some(bind_val) = local.get("bind")
        {
            bind_cache
                .entry(node.name.clone())
                .or_default()
                .push(((**bind_val).clone(), Arc::new(new_self.clone())));
        }

        // Store + history
        match &node.strategy {
            CompiledStrategy::Always => {
                turn_context.insert(node.name.clone(), Arc::new(new_self));
            }
            CompiledStrategy::OncePerTurn | CompiledStrategy::IfModified { .. } => {
                storage.set(node.name.clone(), new_self);
            }
            CompiledStrategy::History { history_bind } => {
                storage.set(node.name.clone(), new_self.clone());
                bind_local.insert("self".into(), Arc::new(new_self));

                let entry = self
                    .eval_script(history_bind, &bind_local, storage, bind_cache, turn_context)
                    .await?;
                if let Some(Value::Object(obj)) = storage.get_mut("history")
                    && let Some(Value::List(list)) = obj.get_mut(&node.name)
                {
                    list.push(entry);
                }
            }
        }

        Ok(())
    }

    /// Resolve a context value by name.
    /// Resolution: node → turn_context → storage → external resolver.
    fn resolve_context(
        &'a self,
        name: &str,
        bindings: HashMap<String, Value>,
        storage: &'a mut HashMapStorage,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_context: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<Arc<Value>, ResolveError>> {
        let name = name.to_string();
        Box::pin(async move {
            // Tool call: resolve target node with bindings as local context
            if !bindings.is_empty() {
                if let Some(&idx) = self.name_to_idx.get(&name) {
                    let local = bindings
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    self.resolve_node(idx, storage, local, bind_cache, turn_context)
                        .await?;
                }
                return self.lookup(&name, storage, turn_context).await;
            }

            // Node: resolve if needed
            if let Some(&idx) = self.name_to_idx.get(&name) {
                let needs_resolve = match &self.nodes[idx].strategy {
                    // Always: re-execute every invocation
                    CompiledStrategy::Always => true,
                    // Others: resolve only if not yet available
                    _ => storage.get(&name).is_none() && !turn_context.contains_key(&name),
                };
                if !needs_resolve {
                    return self.lookup(&name, storage, turn_context).await;
                }
                self.resolve_node(idx, storage, HashMap::new(), bind_cache, turn_context)
                    .await?;
            }

            self.lookup(&name, storage, turn_context).await
        })
    }

    /// Look up a value: turn_context → storage → external resolver.
    async fn lookup(
        &self,
        name: &str,
        storage: &mut HashMapStorage,
        turn_context: &mut HashMap<String, Arc<Value>>,
    ) -> Result<Arc<Value>, ResolveError> {
        if let Some(arc) = turn_context.get(name) {
            return Ok(Arc::clone(arc));
        }
        if let Some(arc) = storage.get(name) {
            return Ok(arc);
        }
        match (self.resolver)(name.to_string()).await {
            Resolved::Once(value) => Ok(Arc::new(value)),
            Resolved::Turn(value) => {
                let arc = Arc::new(value);
                turn_context.insert(name.to_string(), Arc::clone(&arc));
                Ok(arc)
            }
            Resolved::Persist(value) => {
                let arc = Arc::new(value);
                storage.set(name.to_string(), Value::clone(&arc));
                Ok(arc)
            }
        }
    }

    /// Drive any coroutine to completion. The single core loop.
    /// Resolution: local → resolve_context (turn_context → storage → external).
    fn eval_coroutine(
        &'a self,
        coroutine: &'a mut acvus_coroutine::Coroutine<Value>,
        first_key: acvus_coroutine::ResumeKey<Value>,
        local: &'a HashMap<String, Arc<Value>>,
        storage: &'a mut HashMapStorage,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_context: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<Value, ResolveError>> {
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
                            key = need.into_key(Arc::clone(arc));
                        } else {
                            let bindings = need.bindings().clone();
                            let value = self
                                .resolve_context(&name, bindings, storage, bind_cache, turn_context)
                                .await?;
                            key = need.into_key(value);
                        }
                    }
                    Stepped::Done => return Ok(Value::Unit),
                }
            }
        })
    }

    /// Run a compiled script. Convenience over eval_coroutine.
    fn eval_script(
        &'a self,
        script: &'a CompiledScript,
        local: &'a HashMap<String, Arc<Value>>,
        storage: &'a mut HashMapStorage,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_context: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<Value, ResolveError>> {
        Box::pin(async move {
            let interp = Interpreter::new(script.module.clone(), self.extern_fns);
            let (mut coroutine, key) = interp.execute();
            self.eval_coroutine(
                &mut coroutine,
                key,
                local,
                storage,
                bind_cache,
                turn_context,
            )
            .await
        })
    }
}

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
