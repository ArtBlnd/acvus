use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{Interpreter, RuntimeError, Stepped, Value};
use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
use acvus_utils::{Astr, ContextRequest, Coroutine, ExternCallRequest, Interner};
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, info, warn};

use crate::CompiledNodeKind;
use crate::compile::{CompiledMessage, CompiledNode, CompiledScript, CompiledStrategy};

/// Expr 노드의 initial_value 접근 헬퍼.
fn expr_initial_value(node: &CompiledNode) -> Option<&CompiledScript> {
    match &node.kind {
        CompiledNodeKind::Expr(e) => e.initial_value.as_ref(),
        _ => None,
    }
}
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
///
/// ## Context resolution priority (highest → lowest)
///
/// 1. **Local** — function params, @self, @bind (per-task)
/// 2. **turn_context** — Always / OncePerTurn results (per-turn, merged to storage at turn end)
/// 3. **Node execution** — spawn & run the node if its strategy says so
/// 4. **Storage** — persistent cross-turn values (only for names with no pending node)
/// 5. **External resolver** — fallback for names not produced by any node
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

type TaskId = usize;

/// Output of one coroutine step from FuturesUnordered.
struct StepResult {
    task_id: TaskId,
    coroutine: Coroutine<Value, RuntimeError>,
    stepped: Stepped<Value, RuntimeError>,
}

/// What a task is doing — side table keyed by TaskId.
enum TaskMeta {
    Node {
        node_idx: usize,
        local: FxHashMap<Astr, Arc<Value>>,
        is_root: bool,
    },
    Script {
        purpose: ScriptPurpose,
        local: FxHashMap<Astr, Arc<Value>>,
    },
}

enum ScriptPurpose {
    IfModifiedKey { node_idx: usize },
    InitialValue { node_idx: usize },
    Assert { node_idx: usize, value: Value },
    HistoryBind { node_idx: usize, value: Value },
}

enum PendingRequest {
    Context(ContextRequest<Value>),
    ExternCall(ExternCallRequest<Value>),
}

impl PendingRequest {
    fn resolve(self, value: Arc<Value>) {
        match self {
            PendingRequest::Context(r) => r.resolve(value),
            PendingRequest::ExternCall(r) => r.resolve(value),
        }
    }
}

struct Parked {
    task_id: TaskId,
    coroutine: Coroutine<Value, RuntimeError>,
    request: PendingRequest,
}

// ---------------------------------------------------------------------------
// LoopState — all mutable loop bookkeeping in one place
// ---------------------------------------------------------------------------

struct LoopState<'a> {
    next_task_id: TaskId,
    meta: FxHashMap<TaskId, TaskMeta>,
    futs: FuturesUnordered<Pin<Box<dyn Future<Output = StepResult> + Send + 'a>>>,
    dep_waiters: FxHashMap<Astr, Vec<Parked>>,
    in_flight: FxHashSet<usize>,
    remaining_roots: FxHashSet<usize>,
    retry_state: FxHashMap<usize, (u32, u32, FxHashMap<Astr, Arc<Value>>)>,
    /// Always + initial_value Expr 노드의 직렬화 큐.
    ///
    /// 이런 노드는 @self를 읽고 갱신하므로, 동시에 여러 인스턴스가 실행되면
    /// 같은 @self를 읽어 lost update가 발생한다.
    /// 이미 in_flight인 경우 여기에 park하고, 완료 시 하나씩 깨워 재실행한다.
    serialized_queue: FxHashMap<usize, VecDeque<Parked>>,
}

impl<'a> LoopState<'a> {
    fn new() -> Self {
        Self {
            next_task_id: 0,
            meta: FxHashMap::default(),
            futs: FuturesUnordered::new(),
            dep_waiters: FxHashMap::default(),
            in_flight: FxHashSet::default(),
            remaining_roots: FxHashSet::default(),
            retry_state: FxHashMap::default(),
            serialized_queue: FxHashMap::default(),
        }
    }

    fn alloc_id(&mut self) -> TaskId {
        let id = self.next_task_id;
        self.next_task_id += 1;
        id
    }

    fn enqueue_step(&mut self, task_id: TaskId, coroutine: Coroutine<Value, RuntimeError>) {
        self.futs.push(Box::pin(async move {
            let (coroutine, stepped) = coroutine.step().await;
            StepResult {
                task_id,
                coroutine,
                stepped,
            }
        }));
    }

    fn local(&self, task_id: TaskId) -> &FxHashMap<Astr, Arc<Value>> {
        match self.meta.get(&task_id) {
            Some(TaskMeta::Node { local, .. }) | Some(TaskMeta::Script { local, .. }) => local,
            None => {
                static EMPTY: std::sync::LazyLock<FxHashMap<Astr, Arc<Value>>> =
                    std::sync::LazyLock::new(FxHashMap::default);
                &EMPTY
            }
        }
    }

    fn is_node_task(&self, task_id: TaskId) -> bool {
        matches!(self.meta.get(&task_id), Some(TaskMeta::Node { .. }))
    }

    fn wake_waiters(&mut self, name: Astr, value: Value) {
        let arc = Arc::new(value);
        if let Some(waiters) = self.dep_waiters.remove(&name) {
            for w in waiters {
                w.request.resolve(Arc::clone(&arc));
                self.enqueue_step(w.task_id, w.coroutine);
            }
        }
    }

    fn park_for_dep(
        &mut self,
        name: Astr,
        task_id: TaskId,
        coroutine: Coroutine<Value, RuntimeError>,
        request: PendingRequest,
    ) {
        self.dep_waiters.entry(name).or_default().push(Parked {
            task_id,
            coroutine,
            request,
        });
    }
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

/// Dependency-aware node resolver.
///
/// Uses a flat FuturesUnordered event loop to drive coroutines and resolve
/// dependencies without recursive Box::pin calls.
/// Script phases (IfModified, InitialValue, Assert, HistoryBind) are driven
/// as first-class coroutine tasks, identical to node coroutines.
pub struct Resolver<'a, R, EH> {
    pub nodes: &'a [CompiledNode],
    pub node_table: &'a [Arc<dyn Node>],
    pub name_to_idx: &'a FxHashMap<Astr, usize>,
    pub resolver: &'a R,
    pub extern_handler: &'a EH,
    pub interner: &'a Interner,
    pub rdeps: &'a [FxHashSet<usize>],
}

impl<'a, R, EH> Resolver<'a, R, EH>
where
    R: AsyncFn(Astr) -> Resolved + Sync,
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    // -----------------------------------------------------------------------
    // Public entry points
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
        self.resolve_nodes(vec![(idx, local)], state).await
    }

    pub async fn resolve_nodes<S>(
        &self,
        roots: Vec<(usize, FxHashMap<Astr, Arc<Value>>)>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        if roots.is_empty() {
            return Ok(());
        }

        let mut lp = LoopState::new();

        for (idx, local) in roots {
            let max_retries = self.nodes[idx].retry;
            lp.retry_state.insert(idx, (max_retries, 0, local.clone()));
            lp.remaining_roots.insert(idx);
            self.start_prepare(idx, local, true, &mut lp, state);
        }

        while let Some(step) = lp.futs.next().await {
            let StepResult {
                task_id,
                coroutine,
                stepped,
            } = step;

            match stepped {
                Stepped::Emit(value) => {
                    self.handle_emit(task_id, value, &mut lp, state)?;
                }
                Stepped::Done => {
                    if let Some(TaskMeta::Node { node_idx, .. }) = lp.meta.get(&task_id) {
                        warn!(
                            node = %self.interner.resolve(self.nodes[*node_idx].name),
                            "coroutine finished without emit"
                        );
                    }
                    self.handle_emit(task_id, Value::Unit, &mut lp, state)?;
                }
                Stepped::NeedContext(request) => {
                    self.handle_need_context(task_id, coroutine, request, &mut lp, state)
                        .await?;
                }
                Stepped::NeedExternCall(request) => {
                    self.handle_need_extern_call(task_id, coroutine, request, &mut lp, state)
                        .await?;
                }
                Stepped::Error(e) => {
                    self.handle_error(task_id, e, &mut lp, state)?;
                }
            }

            if lp.remaining_roots.is_empty() {
                return Ok(());
            }
        }

        Err(ResolveError::UnresolvedContext(
            "event loop exhausted without all root nodes completing".to_string(),
        ))
    }

    // -----------------------------------------------------------------------
    // Emit (unified Emit + Done)
    // -----------------------------------------------------------------------

    fn handle_emit<S>(
        &self,
        task_id: TaskId,
        value: Value,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        match lp.meta.remove(&task_id) {
            Some(TaskMeta::Node {
                node_idx,
                local,
                is_root,
            }) => {
                if matches!(self.nodes[node_idx].strategy, CompiledStrategy::Always) {
                    lp.in_flight.remove(&node_idx);
                }
                if is_root {
                    self.start_root_finalize(node_idx, value, local, lp, state)?;
                } else {
                    self.apply_store(node_idx, &value, state);
                    self.finish_dep_wake(node_idx, value, lp, state);
                }
            }
            Some(TaskMeta::Script { purpose, local }) => {
                self.handle_script_emit(purpose, value, local, lp, state)?;
            }
            None => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Error
    // -----------------------------------------------------------------------

    fn handle_error<S>(
        &self,
        task_id: TaskId,
        error: RuntimeError,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let task_meta = lp.meta.remove(&task_id);
        let node_idx = match &task_meta {
            Some(TaskMeta::Node { node_idx, .. }) => *node_idx,
            Some(TaskMeta::Script { purpose, .. }) => script_purpose_node_idx(purpose),
            None => {
                return Err(ResolveError::Runtime {
                    node: String::new(),
                    error,
                });
            }
        };

        let is_root = matches!(&task_meta, Some(TaskMeta::Node { is_root: true, .. }));
        let node_name = self
            .interner
            .resolve(self.nodes[node_idx].name)
            .to_string();

        if is_root && self.try_retry(node_idx, &node_name, &error, lp, state) {
            return Ok(());
        }

        Err(ResolveError::Runtime {
            node: node_name,
            error,
        })
    }

    // -----------------------------------------------------------------------
    // NeedContext
    // -----------------------------------------------------------------------

    /// Resolve a context request.
    ///
    /// Priority: Local → turn_context → Node (execution) → Storage → External resolver
    async fn handle_need_context<S>(
        &self,
        task_id: TaskId,
        coroutine: Coroutine<Value, RuntimeError>,
        request: ContextRequest<Value>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let name = request.name();
        let name_str = self.interner.resolve(name);

        // 1. Local context (highest priority)
        if let Some(arc) = lp.local(task_id).get(&name) {
            debug!(context = %name_str, "resolved from local");
            request.resolve(Arc::clone(arc));
            lp.enqueue_step(task_id, coroutine);
            return Ok(());
        }

        // 2. turn_context (already resolved this turn)
        if let Some(arc) = state.turn_context.get(&name).cloned() {
            debug!(context = %name_str, "resolved from turn_context");
            request.resolve(arc);
            lp.enqueue_step(task_id, coroutine);
            return Ok(());
        }

        // 3. Known node — check before storage so Node > Storage
        if let Some(&dep_idx) = self.name_to_idx.get(&name) {
            // 3a. Function node → ExternFn handle
            if self.nodes[dep_idx].is_function {
                request.resolve(Arc::new(Value::ExternFn(name)));
                lp.enqueue_step(task_id, coroutine);
                return Ok(());
            }

            // 3b. Node task → spawn dep if strategy says so
            if lp.is_node_task(task_id) {
                if self.needs_resolve(dep_idx, state) {
                    // Serialized node already in flight → park in serialized_queue
                    if self.needs_serialized(dep_idx) && lp.in_flight.contains(&dep_idx) {
                        lp.serialized_queue.entry(dep_idx).or_default().push_back(Parked {
                            task_id, coroutine,
                            request: PendingRequest::Context(request),
                        });
                        return Ok(());
                    }
                    if !lp.in_flight.contains(&dep_idx) {
                        debug!(context = %name_str, "spawning dependency node");
                        self.start_prepare(dep_idx, FxHashMap::default(), false, lp, state);
                    }
                    lp.park_for_dep(name, task_id, coroutine, PendingRequest::Context(request));
                    return Ok(());
                }

                // 3c. Node doesn't need resolve → serve from storage
                if let Some(arc) = state.storage.get(name_str) {
                    debug!(context = %name_str, "resolved from storage");
                    request.resolve(arc);
                    lp.enqueue_step(task_id, coroutine);
                    return Ok(());
                }

                // Node exists but not in storage either → park for dep
                if !lp.in_flight.contains(&dep_idx) {
                    self.start_prepare(dep_idx, FxHashMap::default(), false, lp, state);
                }
                lp.park_for_dep(name, task_id, coroutine, PendingRequest::Context(request));
                return Ok(());
            }
        }

        // 4. Storage (for non-node names)
        if let Some(arc) = state.storage.get(name_str) {
            debug!(context = %name_str, "resolved from storage");
            request.resolve(arc);
            lp.enqueue_step(task_id, coroutine);
            return Ok(());
        }

        // 5. External resolver (lowest priority)
        debug!(context = %name_str, "calling external resolver");
        let value = self.resolve_external(name, state).await?;
        request.resolve(value);
        lp.enqueue_step(task_id, coroutine);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // NeedExternCall
    // -----------------------------------------------------------------------

    async fn handle_need_extern_call<S>(
        &self,
        task_id: TaskId,
        coroutine: Coroutine<Value, RuntimeError>,
        request: ExternCallRequest<Value>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let name = request.name();

        // 1. Node task → spawn dep node
        if lp.is_node_task(task_id) {
            if let Some(&dep_idx) = self.name_to_idx.get(&name) {
                let args = request.args().to_vec();
                let node = &self.nodes[dep_idx];
                let dep_local: FxHashMap<Astr, Arc<Value>> = if node.is_function {
                    node.fn_params
                        .iter()
                        .zip(args.into_iter())
                        .map(|((name, _), val)| (*name, Arc::new(val)))
                        .collect()
                } else if let Some(Value::Object(obj)) = args.first() {
                    obj.iter()
                        .map(|(k, v)| (*k, Arc::new(v.clone())))
                        .collect()
                } else {
                    FxHashMap::default()
                };

                // Serialized node already in flight → park in serialized_queue
                if self.needs_serialized(dep_idx) && lp.in_flight.contains(&dep_idx) {
                    lp.serialized_queue.entry(dep_idx).or_default().push_back(Parked {
                        task_id, coroutine,
                        request: PendingRequest::ExternCall(request),
                    });
                    return Ok(());
                }

                debug!(
                    context = %self.interner.resolve(name),
                    "spawning tool node via extern call"
                );
                self.start_prepare(dep_idx, dep_local, false, lp, state);
                lp.park_for_dep(name, task_id, coroutine, PendingRequest::ExternCall(request));
                return Ok(());
            }
        }

        // 2. extern_handler
        let args = request.args().to_vec();
        debug!(context = %self.interner.resolve(name), "calling extern_handler");
        match (self.extern_handler)(name, args).await {
            Ok(value) => {
                request.resolve(Arc::new(value));
                lp.enqueue_step(task_id, coroutine);
            }
            Err(e) => {
                let node_idx = match lp.meta.get(&task_id) {
                    Some(TaskMeta::Node { node_idx, .. }) => *node_idx,
                    Some(TaskMeta::Script { purpose, .. }) => script_purpose_node_idx(purpose),
                    None => {
                        return Err(ResolveError::Runtime {
                            node: String::new(),
                            error: e,
                        });
                    }
                };
                return Err(ResolveError::Runtime {
                    node: self
                        .interner
                        .resolve(self.nodes[node_idx].name)
                        .to_string(),
                    error: e,
                });
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Script phase transitions
    // -----------------------------------------------------------------------

    fn handle_script_emit<S>(
        &self,
        purpose: ScriptPurpose,
        value: Value,
        local: FxHashMap<Astr, Arc<Value>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        match purpose {
            ScriptPurpose::IfModifiedKey { node_idx } => {
                let node = &self.nodes[node_idx];
                let node_name_str = self.interner.resolve(node.name);

                // Cache hit → skip execution
                if let Some(entries) = state.bind_cache.get(&node.name)
                    && let Some((_, cached)) = entries.iter().find(|(v, _)| v == &value)
                {
                    debug!(node = %node_name_str, "if_modified cache hit, skipping execution");
                    state
                        .turn_context
                        .insert(node.name, Arc::new(Value::clone(cached)));
                    if lp.remaining_roots.contains(&node_idx) {
                        lp.remaining_roots.remove(&node_idx);
                    }
                    lp.wake_waiters(node.name, Value::clone(cached));
                    return Ok(());
                }

                debug!(node = %node_name_str, "if_modified cache miss, will execute");
                let mut new_local = local;
                new_local.insert(self.interner.intern("bind"), Arc::new(value));
                self.start_after_if_modified(node_idx, new_local, lp, state);
            }

            ScriptPurpose::InitialValue { node_idx } => {
                let mut new_local = local;
                new_local.insert(self.interner.intern("self"), Arc::new(value));
                let is_root = lp.remaining_roots.contains(&node_idx);
                self.spawn_node_task(node_idx, new_local, is_root, lp);
            }

            ScriptPurpose::Assert {
                node_idx,
                value: node_value,
            } => {
                let node_name_str = self.interner.resolve(self.nodes[node_idx].name);

                let Value::Bool(passed) = value else {
                    return Err(ResolveError::Runtime {
                        node: node_name_str.to_string(),
                        error: RuntimeError::type_mismatch(
                            "assert",
                            "bool",
                            &format!("{value:?}"),
                        ),
                    });
                };

                if !passed {
                    info!(node = %node_name_str, "assert failed, triggering retry");
                    let node_name = node_name_str.to_string();
                    let error = RuntimeError::other("assert failed");
                    if !self.try_retry(node_idx, &node_name, &error, lp, state) {
                        return Err(ResolveError::Runtime {
                            node: node_name,
                            error,
                        });
                    }
                    return Ok(());
                }

                self.apply_root_finalize(node_idx, &node_value, &local, lp, state);
            }

            ScriptPurpose::HistoryBind {
                node_idx,
                value: node_value,
            } => {
                let node = &self.nodes[node_idx];
                state.history_entries.insert(node.name, value);

                // turn_context already set by apply_root_finalize — no redundant storage.set
                info!(node = %self.interner.resolve(node.name), "resolve node complete");

                lp.remaining_roots.remove(&node_idx);
                self.finish_dep_wake(node_idx, node_value, lp, state);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Root finalize: assert → cache → store → history
    // -----------------------------------------------------------------------

    fn start_root_finalize<S>(
        &self,
        node_idx: usize,
        value: Value,
        local: FxHashMap<Astr, Arc<Value>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> Result<(), ResolveError>
    where
        S: Storage,
    {
        let node = &self.nodes[node_idx];

        if let Some(ref assert_script) = node.assert {
            let mut bind_local = local;
            bind_local.insert(self.interner.intern("self"), Arc::new(value.clone()));
            debug!(node = %self.interner.resolve(node.name), "evaluating assert");
            self.spawn_script_task(
                assert_script,
                bind_local,
                ScriptPurpose::Assert { node_idx, value },
                lp,
            );
        } else {
            self.apply_root_finalize(node_idx, &value, &local, lp, state);
        }
        Ok(())
    }

    fn apply_root_finalize<S>(
        &self,
        node_idx: usize,
        value: &Value,
        local: &FxHashMap<Astr, Arc<Value>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) where
        S: Storage,
    {
        let node = &self.nodes[node_idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);

        // IfModified: cache update
        if matches!(node.strategy, CompiledStrategy::IfModified { .. })
            && let Some(bind_val) = local.get(&interner.intern("bind"))
        {
            state
                .bind_cache
                .entry(node.name)
                .or_default()
                .push(((**bind_val).clone(), Arc::new(value.clone())));
        }

        // Store in turn_context (merged to storage at turn end)
        state
            .turn_context
            .insert(node.name, Arc::new(value.clone()));

        // History: spawn history_bind script before completing
        if let CompiledStrategy::History { history_bind } = &node.strategy {
            let mut hist_local = FxHashMap::default();
            hist_local.insert(interner.intern("self"), Arc::new(value.clone()));
            debug!(node = %node_name_str, "evaluating history_bind");
            self.spawn_script_task(
                history_bind,
                hist_local,
                ScriptPurpose::HistoryBind {
                    node_idx,
                    value: value.clone(),
                },
                lp,
            );
            return; // HistoryBind will remove from remaining_roots
        }

        info!(node = %node_name_str, "resolve node complete");
        lp.remaining_roots.remove(&node_idx);
        self.finish_dep_wake(node_idx, value.clone(), lp, state);
    }

    // -----------------------------------------------------------------------
    // Prepare phase: IfModified → InitialValue → Node spawn
    // -----------------------------------------------------------------------

    /// Wake dep waiters after a node completes. For serialized nodes,
    /// wakes one waiter at a time via re-execution to avoid lost updates.
    fn finish_dep_wake<S>(
        &self,
        node_idx: usize,
        value: Value,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) where
        S: Storage,
    {
        let node_name = self.nodes[node_idx].name;
        if self.needs_serialized(node_idx) {
            if let Some(queue) = lp.serialized_queue.get_mut(&node_idx)
                && let Some(parked) = queue.pop_front()
            {
                // 다음 waiter를 dep_waiters로 이동 → 재실행 결과를 받게 됨
                lp.dep_waiters.entry(node_name).or_default().push(parked);
                // 노드 재실행 (갱신된 @self from turn_context)
                self.start_prepare(node_idx, FxHashMap::default(), false, lp, state);
            } else {
                // 직렬화 큐 비었음 → 일반 wake
                lp.wake_waiters(node_name, value);
                self.try_eager_schedule(node_idx, lp, state);
            }
        } else {
            lp.wake_waiters(node_name, value);
            self.try_eager_schedule(node_idx, lp, state);
        }
    }

    fn start_prepare<S>(
        &self,
        idx: usize,
        local: FxHashMap<Astr, Arc<Value>>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);
        info!(node = %node_name_str, "prepare node");

        // IfModified: spawn key script
        if let CompiledStrategy::IfModified { ref key } = node.strategy {
            self.spawn_script_task(
                key,
                local,
                ScriptPurpose::IfModifiedKey { node_idx: idx },
                lp,
            );
            lp.in_flight.insert(idx);
            return;
        }

        // InitialValue (Expr only)
        if let Some(init_script) = expr_initial_value(node) {
            if let Some(prev) = self.load_self_value(idx, state) {
                let mut new_local = local;
                new_local.insert(interner.intern("self"), Arc::new(prev));
                debug!(node = %node_name_str, "spawning coroutine");
                self.spawn_node_task(idx, new_local, is_root, lp);
            } else {
                debug!(node = %node_name_str, "evaluating initial_value (first run)");
                self.spawn_script_task(
                    init_script,
                    local,
                    ScriptPurpose::InitialValue { node_idx: idx },
                    lp,
                );
                lp.in_flight.insert(idx);
            }
            return;
        }

        debug!(node = %node_name_str, "spawning coroutine");
        self.spawn_node_task(idx, local, is_root, lp);
    }

    /// After IfModified key resolved (cache miss): check initial_value, then spawn node.
    fn start_after_if_modified<S>(
        &self,
        node_idx: usize,
        local: FxHashMap<Astr, Arc<Value>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) where
        S: Storage,
    {
        let node = &self.nodes[node_idx];

        if let Some(init_script) = expr_initial_value(node) {
            if let Some(prev) = self.load_self_value(node_idx, state) {
                let mut new_local = local;
                new_local.insert(self.interner.intern("self"), Arc::new(prev));
                let is_root = lp.remaining_roots.contains(&node_idx);
                self.spawn_node_task(node_idx, new_local, is_root, lp);
            } else {
                self.spawn_script_task(
                    init_script,
                    local,
                    ScriptPurpose::InitialValue { node_idx },
                    lp,
                );
            }
        } else {
            let is_root = lp.remaining_roots.contains(&node_idx);
            self.spawn_node_task(node_idx, local, is_root, lp);
        }
    }

    /// Try to load the existing @self value from storage or turn_context.
    fn load_self_value<S>(&self, idx: usize, state: &ResolveState<S>) -> Option<Value>
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        let name_str = self.interner.resolve(node.name);
        if let Some(arc) = state.storage.get(name_str) {
            return Some(Value::clone(&arc));
        }
        if let Some(arc) = state.turn_context.get(&node.name) {
            return Some(Value::clone(arc));
        }
        None
    }

    // -----------------------------------------------------------------------
    // Store (common for root and dep finalize)
    // -----------------------------------------------------------------------

    /// Store a dep node's result. All strategies go to turn_context;
    /// merged to storage at turn end by ChatEngine.
    fn apply_store<S>(&self, idx: usize, value: &Value, state: &mut ResolveState<S>)
    where
        S: Storage,
    {
        let node = &self.nodes[idx];
        state
            .turn_context
            .insert(node.name, Arc::new(value.clone()));
    }

    // -----------------------------------------------------------------------
    // Retry
    // -----------------------------------------------------------------------

    fn try_retry<S>(
        &self,
        idx: usize,
        node_name: &str,
        error: &RuntimeError,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) -> bool
    where
        S: Storage,
    {
        let Some((max_retries, attempt, local)) = lp.retry_state.get_mut(&idx) else {
            return false;
        };
        if *attempt >= *max_retries {
            return false;
        }
        *attempt += 1;
        warn!(
            node = %node_name,
            attempt = *attempt,
            max = *max_retries,
            error = %error,
            "retrying node after runtime error",
        );
        let local_clone = local.clone();
        self.start_prepare(idx, local_clone, true, lp, state);
        true
    }

    // -----------------------------------------------------------------------
    // External resolver
    // -----------------------------------------------------------------------

    async fn resolve_external<S>(
        &self,
        name: Astr,
        state: &mut ResolveState<S>,
    ) -> Result<Arc<Value>, ResolveError>
    where
        S: Storage,
    {
        let name_str = self.interner.resolve(name);
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
    // Eager dependency scheduling
    // -----------------------------------------------------------------------

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
                CompiledNodeKind::Plain(plain) => {
                    let p =
                        partition_context_keys(&plain.block.module, &known, &plain.block.val_def);
                    eager.extend(p.eager);
                }
                CompiledNodeKind::Expr(expr) => {
                    let p =
                        partition_context_keys(&expr.script.module, &known, &expr.script.val_def);
                    eager.extend(p.eager);
                }
                _ => {}
            }
        }

        if let Some(iv) = expr_initial_value(node)
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

    /// Always 전략 + initial_value를 가진 Expr 노드인지.
    /// 이 조건이면 동시 실행 시 @self lost update가 발생하므로 직렬화 필요.
    fn needs_serialized(&self, idx: usize) -> bool {
        let node = &self.nodes[idx];
        matches!(node.strategy, CompiledStrategy::Always)
            && matches!(&node.kind, CompiledNodeKind::Expr(e) if e.initial_value.is_some())
    }

    fn needs_resolve<S>(&self, idx: usize, state: &ResolveState<S>) -> bool
    where
        S: Storage,
    {
        let name = self.nodes[idx].name;
        match &self.nodes[idx].strategy {
            // Always: re-execute on every reference within a turn.
            CompiledStrategy::Always => true,
            // All other strategies: execute once per turn.
            // Only turn_context counts — storage values from previous turns
            // do NOT satisfy this. The node must re-run each turn.
            _ => !state.turn_context.contains_key(&name),
        }
    }

    fn try_eager_schedule<S>(
        &self,
        completed_idx: usize,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<S>,
    ) where
        S: Storage,
    {
        if completed_idx >= self.rdeps.len() {
            return;
        }
        for &candidate in &self.rdeps[completed_idx] {
            if lp.in_flight.contains(&candidate) || !self.needs_resolve(candidate, state) {
                continue;
            }
            let eager_deps = self.eager_node_deps(candidate, &state.storage);
            if eager_deps
                .iter()
                .all(|&dep| !self.needs_resolve(dep, state))
            {
                debug!(
                    node = %self.interner.resolve(self.nodes[candidate].name),
                    "eager scheduling"
                );
                self.start_prepare(candidate, FxHashMap::default(), false, lp, state);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Task spawning
    // -----------------------------------------------------------------------

    fn spawn_node_task(
        &self,
        node_idx: usize,
        local: FxHashMap<Astr, Arc<Value>>,
        is_root: bool,
        lp: &mut LoopState<'_>,
    ) {
        let tid = lp.alloc_id();
        let coroutine = self.node_table[node_idx].spawn(local.clone());
        lp.meta.insert(
            tid,
            TaskMeta::Node {
                node_idx,
                local,
                is_root,
            },
        );
        lp.in_flight.insert(node_idx);
        lp.enqueue_step(tid, coroutine);
    }

    fn spawn_script_task(
        &self,
        script: &CompiledScript,
        local: FxHashMap<Astr, Arc<Value>>,
        purpose: ScriptPurpose,
        lp: &mut LoopState<'_>,
    ) {
        let tid = lp.alloc_id();
        let interp = Interpreter::new(self.interner, script.module.clone());
        let coroutine = interp.execute();
        lp.meta.insert(tid, TaskMeta::Script { purpose, local });
        lp.enqueue_step(tid, coroutine);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

fn script_purpose_node_idx(purpose: &ScriptPurpose) -> usize {
    match purpose {
        ScriptPurpose::IfModifiedKey { node_idx }
        | ScriptPurpose::InitialValue { node_idx }
        | ScriptPurpose::Assert { node_idx, .. }
        | ScriptPurpose::HistoryBind { node_idx, .. } => *node_idx,
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
