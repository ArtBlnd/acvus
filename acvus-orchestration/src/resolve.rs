use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{Interpreter, LazyValue, PureValue, RuntimeError, Stepped, TypedValue, Value};
use acvus_mir::ty::Ty;
use acvus_mir_pass::analysis::reachable_context::partition_context_keys;
use acvus_utils::{Astr, ContextRequest, Coroutine, ExternCallRequest, Interner, TrackedDeque};
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, info, warn};

use crate::CompiledNodeKind;
use crate::compile::{CompiledExecution, CompiledMessage, CompiledNode, CompiledPersistency, CompiledScript};

/// Access a node's initial_value script.
fn node_initial_value(node: &CompiledNode) -> Option<&CompiledScript> {
    node.strategy.initial_value.as_ref()
}
use crate::node::Node;
use crate::storage::{EntryMut, EntryRef, PatchDiff, StoragePatch};

// ---------------------------------------------------------------------------
// ResolveState — bundled mutable context
// ---------------------------------------------------------------------------

/// Mutable state that flows through the resolver.
///
/// Groups the three stores that always travel together:
/// - `entry`: mutable entry into the storage tree
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
pub struct ResolveState<E> {
    pub entry: E,
    pub turn_context: FxHashMap<Astr, Arc<TypedValue>>,
    pub bind_cache: FxHashMap<Astr, Vec<(TypedValue, Arc<TypedValue>)>>,
}

impl<E> ResolveState<E> {
    /// Cache a value in turn_context. This is the **single entry point**
    /// for all turn_context insertions.
    pub fn cache(&mut self, name: Astr, value: Arc<TypedValue>) {
        self.turn_context.insert(name, value);
    }

    /// Whether a value is already cached for this turn.
    pub fn is_cached(&self, name: &Astr) -> bool {
        self.turn_context.contains_key(name)
    }

    /// Get a cached value from turn_context.
    pub fn get_cached(&self, name: &Astr) -> Option<Arc<TypedValue>> {
        self.turn_context.get(name).cloned()
    }
}

impl<'j, E: EntryMut<'j>> ResolveState<E> {
    /// Persist a value to storage. This is the **single entry point**
    /// for all storage writes.
    ///
    /// The type checker guarantees that persistent nodes have storable
    /// output types at compile time. This debug_assert is defense-in-depth.
    pub fn persist(&mut self, key: &str, diff: StoragePatch) {
        self.entry.apply(key, diff);
    }

    /// Load a value by name: turn_context first (most recent), then storage.
    pub fn load(&self, name: Astr, name_str: &str) -> Option<Arc<TypedValue>> {
        self.turn_context
            .get(&name)
            .cloned()
            .or_else(|| self.entry.get(name_str))
    }

    /// Load @self for a node: turn_context first (this turn's update),
    /// then storage (previous turn's value).
    pub fn load_self(&self, name: Astr, name_str: &str) -> Option<Arc<TypedValue>> {
        self.turn_context
            .get(&name)
            .cloned()
            .or_else(|| self.entry.get(name_str))
    }
}

// ---------------------------------------------------------------------------
// External resolver result
// ---------------------------------------------------------------------------

/// External resolver result with lifetime hint.
pub enum Resolved {
    /// Valid for this request only. Not cached.
    Once(TypedValue),
    /// Valid for this turn. Cached in turn_context, discarded at turn end.
    Turn(TypedValue),
    /// Persistent. Stored in storage, survives across turns.
    Persist(TypedValue),
}

// ---------------------------------------------------------------------------
// Event loop types
// ---------------------------------------------------------------------------

type TaskId = usize;

/// Output of one coroutine step from FuturesUnordered.
struct StepResult {
    task_id: TaskId,
    coroutine: Coroutine<TypedValue, RuntimeError>,
    stepped: Stepped<TypedValue, RuntimeError>,
}

/// What a task is doing — side table keyed by TaskId.
enum TaskMeta {
    Node {
        node_idx: usize,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        is_root: bool,
    },
    Script {
        purpose: ScriptPurpose,
        local: FxHashMap<Astr, Arc<TypedValue>>,
    },
}

enum ScriptPurpose {
    IfModifiedKey { node_idx: usize },
    InitialValue { node_idx: usize },
    Assert { node_idx: usize, value: TypedValue },
    BindScript {
        node_idx: usize,
        value: TypedValue,
        origin: Option<TrackedDeque<Value>>,
    },
}

enum PendingRequest {
    Context(ContextRequest<TypedValue>),
    ExternCall(ExternCallRequest<TypedValue>),
}

impl PendingRequest {
    fn resolve(self, value: Arc<TypedValue>) {
        match self {
            PendingRequest::Context(r) => r.resolve(value),
            PendingRequest::ExternCall(r) => r.resolve(value),
        }
    }
}

struct Parked {
    task_id: TaskId,
    coroutine: Coroutine<TypedValue, RuntimeError>,
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
    retry_state: FxHashMap<usize, (u32, u32, FxHashMap<Astr, Arc<TypedValue>>)>,
    /// Always + initial_value Expr 노드의 직렬화 큐.
    ///
    /// 이런 노드는 @self를 읽고 갱신하므로, 동시에 여러 인스턴스가 실행되면
    /// 같은 @self를 읽어 lost update가 발생한다.
    /// 이미 in_flight인 경우 여기에 park하고, 완료 시 하나씩 깨워 재실행한다.
    serialized_queue: FxHashMap<usize, VecDeque<Parked>>,
    /// When true, dependency nodes are NOT executed — served from storage only.
    /// Root (entrypoint) nodes still execute normally.
    no_execute: bool,
}

impl<'a> LoopState<'a> {
    fn new(no_execute: bool) -> Self {
        Self {
            next_task_id: 0,
            meta: FxHashMap::default(),
            futs: FuturesUnordered::new(),
            dep_waiters: FxHashMap::default(),
            in_flight: FxHashSet::default(),
            remaining_roots: FxHashSet::default(),
            retry_state: FxHashMap::default(),
            serialized_queue: FxHashMap::default(),
            no_execute,
        }
    }

    fn alloc_id(&mut self) -> TaskId {
        let id = self.next_task_id;
        self.next_task_id += 1;
        id
    }

    fn enqueue_step(&mut self, task_id: TaskId, coroutine: Coroutine<TypedValue, RuntimeError>) {
        self.futs.push(Box::pin(async move {
            let (coroutine, stepped) = coroutine.step().await;
            StepResult {
                task_id,
                coroutine,
                stepped,
            }
        }));
    }

    fn local(&self, task_id: TaskId) -> &FxHashMap<Astr, Arc<TypedValue>> {
        match self.meta.get(&task_id) {
            Some(TaskMeta::Node { local, .. }) | Some(TaskMeta::Script { local, .. }) => local,
            None => {
                static EMPTY: std::sync::LazyLock<FxHashMap<Astr, Arc<TypedValue>>> =
                    std::sync::LazyLock::new(FxHashMap::default);
                &EMPTY
            }
        }
    }

    fn is_node_task(&self, task_id: TaskId) -> bool {
        matches!(self.meta.get(&task_id), Some(TaskMeta::Node { .. }))
    }

    fn wake_waiters(&mut self, name: Astr, value: TypedValue) {
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
        coroutine: Coroutine<TypedValue, RuntimeError>,
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
/// Script phases (IfModified, InitialValue, Assert, BindScript) are driven
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
    EH: AsyncFn(Astr, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync,
{
    // -----------------------------------------------------------------------
    // Public entry points
    // -----------------------------------------------------------------------

    pub async fn resolve_node<'j, E>(
        &self,
        idx: usize,
        state: &mut ResolveState<E>,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        no_execute: bool,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        self.resolve_nodes(vec![(idx, local)], state, no_execute).await
    }

    pub async fn resolve_nodes<'j, E>(
        &self,
        roots: Vec<(usize, FxHashMap<Astr, Arc<TypedValue>>)>,
        state: &mut ResolveState<E>,
        no_execute: bool,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        if roots.is_empty() {
            return Ok(());
        }

        let mut lp = LoopState::new(no_execute);

        for (idx, local) in roots {
            let max_retries = self.nodes[idx].strategy.retry;
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
                    self.handle_emit(task_id, TypedValue::unit(), &mut lp, state)?;
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

        let stuck_roots: Vec<String> = lp.remaining_roots.iter().map(|&idx| {
            self.interner.resolve(self.nodes[idx].name).to_string()
        }).collect();
        let dep_waiters: Vec<String> = lp.dep_waiters.keys().map(|&name| {
            self.interner.resolve(name).to_string()
        }).collect();
        let serialized_queue: Vec<String> = lp.serialized_queue.keys().map(|&idx| {
            self.interner.resolve(self.nodes[idx].name).to_string()
        }).collect();
        let turn_context_keys: Vec<String> = state.turn_context.keys().map(|&name| {
            self.interner.resolve(name).to_string()
        }).collect();
        let in_flight: Vec<String> = lp.in_flight.iter().map(|&idx| {
            self.interner.resolve(self.nodes[idx].name).to_string()
        }).collect();
        let mut parked: Vec<ParkedDiag> = Vec::new();
        // dep_waiters
        for (name, waiters) in &lp.dep_waiters {
            let waiting_for = self.interner.resolve(*name).to_string();
            for w in waiters {
                let task = match lp.meta.get(&w.task_id) {
                    Some(TaskMeta::Node { node_idx, .. }) =>
                        self.interner.resolve(self.nodes[*node_idx].name).to_string(),
                    Some(TaskMeta::Script { purpose, .. }) => {
                        let idx = script_purpose_node_idx(purpose);
                        format!("script:{}", self.interner.resolve(self.nodes[idx].name))
                    }
                    None => format!("task#{}", w.task_id),
                };
                parked.push(ParkedDiag { task, waiting_for: waiting_for.clone() });
            }
        }
        // serialized_queue
        for (&idx, queue) in &lp.serialized_queue {
            let waiting_for = self.interner.resolve(self.nodes[idx].name).to_string();
            for w in queue {
                let task = match lp.meta.get(&w.task_id) {
                    Some(TaskMeta::Node { node_idx, .. }) =>
                        self.interner.resolve(self.nodes[*node_idx].name).to_string(),
                    Some(TaskMeta::Script { purpose, .. }) => {
                        let idx = script_purpose_node_idx(purpose);
                        format!("script:{}", self.interner.resolve(self.nodes[idx].name))
                    }
                    None => format!("task#{}", w.task_id),
                };
                parked.push(ParkedDiag { task, waiting_for: format!("serial:{}", waiting_for) });
            }
        }

        Err(ResolveError::Deadlock { stuck_roots, dep_waiters, serialized_queue, parked, turn_context_keys, in_flight })
    }

    // -----------------------------------------------------------------------
    // Emit (unified Emit + Done)
    // -----------------------------------------------------------------------

    fn handle_emit<'j, E>(
        &self,
        task_id: TaskId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        match lp.meta.remove(&task_id) {
            Some(TaskMeta::Node {
                node_idx,
                local,
                is_root,
            }) => {
                if matches!(self.nodes[node_idx].strategy.execution, CompiledExecution::Always) {
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

    fn handle_error<'j, E>(
        &self,
        task_id: TaskId,
        error: RuntimeError,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
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
    async fn handle_need_context<'j, E>(
        &self,
        task_id: TaskId,
        coroutine: Coroutine<TypedValue, RuntimeError>,
        request: ContextRequest<TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
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
        if let Some(arc) = state.get_cached(&name) {
            debug!(context = %name_str, "resolved from turn_context");
            request.resolve(arc);
            lp.enqueue_step(task_id, coroutine);
            return Ok(());
        }

        // 3. Known node — check before storage so Node > Storage
        if let Some(&dep_idx) = self.name_to_idx.get(&name) {
            // 3a. Function node → ExternFn handle
            if self.nodes[dep_idx].is_function {
                request.resolve(Arc::new(TypedValue::new(Arc::new(Value::extern_fn(name)), Ty::Infer)));
                lp.enqueue_step(task_id, coroutine);
                return Ok(());
            }

            // 3b. Node task → spawn dep if strategy says so
            if lp.is_node_task(task_id) {
                if self.needs_resolve(dep_idx, state, lp.no_execute) {
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
                if let Some(arc) = state.entry.get(name_str) {
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
        if let Some(arc) = state.entry.get(name_str) {
            debug!(context = %name_str, "resolved from storage");
            request.resolve(arc);
            lp.enqueue_step(task_id, coroutine);
            return Ok(());
        }

        // 4b. @turn_index — derived from tree depth
        if name == self.interner.intern("turn_index") {
            let depth = state.entry.depth();
            request.resolve(Arc::new(TypedValue::int(depth as i64)));
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

    async fn handle_need_extern_call<'j, E>(
        &self,
        task_id: TaskId,
        coroutine: Coroutine<TypedValue, RuntimeError>,
        request: ExternCallRequest<TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        let name = request.name();

        // 1. Node task → spawn dep node
        if lp.is_node_task(task_id) {
            if let Some(&dep_idx) = self.name_to_idx.get(&name) {
                let args = request.args().to_vec();
                let node = &self.nodes[dep_idx];
                let dep_local: FxHashMap<Astr, Arc<TypedValue>> = if node.is_function {
                    // Tool call args come as a single Object — unpack fields by name.
                    if let Some(tv) = args.first()
                        && let Value::Lazy(LazyValue::Object(obj)) = tv.value()
                    {
                        node.fn_params
                            .iter()
                            .filter_map(|p| {
                                let val = obj.get(&p.name)?.clone();
                                Some((p.name, Arc::new(TypedValue::new(Arc::new(val), Ty::Infer))))
                            })
                            .collect()
                    } else {
                        // Positional fallback (e.g. script-level @fn(a, b) calls)
                        node.fn_params
                            .iter()
                            .zip(args.into_iter())
                            .map(|(p, val)| (p.name, Arc::new(val)))
                            .collect()
                    }
                } else if let Some(tv) = args.first()
                    && let Value::Lazy(LazyValue::Object(obj)) = tv.value()
                {
                    obj.iter()
                        .map(|(k, v)| (*k, Arc::new(TypedValue::new(Arc::new(v.clone()), Ty::Infer))))
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

    fn handle_script_emit<'j, E>(
        &self,
        purpose: ScriptPurpose,
        value: TypedValue,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        match purpose {
            ScriptPurpose::IfModifiedKey { node_idx } => {
                let node = &self.nodes[node_idx];
                let node_name_str = self.interner.resolve(node.name);

                // Cache hit → skip execution
                if let Some(entries) = state.bind_cache.get(&node.name)
                    && let Some((_, cached)) = entries.iter().find(|(v, _)| v == &value)
                {
                    let cached_value = TypedValue::clone(cached);
                    debug!(node = %node_name_str, "if_modified cache hit, skipping execution");
                    state.cache(node.name, Arc::new(cached_value.clone()));
                    if lp.remaining_roots.contains(&node_idx) {
                        lp.remaining_roots.remove(&node_idx);
                    }
                    lp.wake_waiters(node.name, cached_value);
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

                let Value::Pure(PureValue::Bool(passed)) = value.value() else {
                    return Err(ResolveError::Runtime {
                        node: node_name_str.to_string(),
                        error: RuntimeError::type_mismatch(
                            "assert",
                            "bool",
                            &format!("{value:?}"),
                        ),
                    });
                };

                if !*passed {
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

            ScriptPurpose::BindScript {
                node_idx,
                value: _raw_value,
                origin: _,
            } => {
                let node = &self.nodes[node_idx];
                let node_name_str = self.interner.resolve(node.name);

                // Apply the bind result to storage based on mode.
                // The bind result (`value`) is the accumulated @self — this is
                // what @node_name should resolve to for other nodes.
                let stored_value = match &node.strategy.persistency {
                    CompiledPersistency::Sequence { .. } => {
                        // Pass the value as-is to storage. apply() handles
                        // collect (if Sequence) and diff computation.
                        let stored = value.clone();
                        state.persist(
                            node_name_str,
                            StoragePatch::Snapshot(value),
                        );
                        stored
                    }
                    CompiledPersistency::Patch { .. } => {
                        // Deep recursive diff: compare bind result against previous @self.
                        // If they differ, store only the diff (history accumulates).
                        // If no previous value exists, snapshot the entire value.
                        let old_value = state.load(node.name, node_name_str);
                        let patch = old_value
                            .as_ref()
                            .and_then(|old| PatchDiff::compute(old.value(), value.value()));
                        let stored = value.clone();
                        match patch {
                            Some(diff) => {
                                state.persist(node_name_str, StoragePatch::Patch(diff));
                            }
                            None => {
                                // First write or identical value → snapshot
                                state.persist(node_name_str, StoragePatch::Snapshot(value));
                            }
                        }
                        stored
                    }
                    _ => unreachable!("BindScript only spawned for Sequence/Patch modes"),
                };

                // Update turn_context with the bind result (not raw).
                // @node_name must reflect the accumulated value (@self view).
                state.cache(node.name, Arc::new(stored_value.clone()));

                info!(node = %node_name_str, "resolve node complete");

                lp.remaining_roots.remove(&node_idx);
                self.finish_dep_wake(node_idx, stored_value, lp, state);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bind-scope locals: @self + @raw injection (single source of truth)
    // -----------------------------------------------------------------------

    /// Prepare locals for Bind scope (@self + @raw).
    /// Returns `(locals, origin)` where `origin` is the Deque checkpoint
    /// needed by bind to compute diffs. Assert callers can ignore `origin`.
    fn prepare_bind_locals<'j, E>(
        &self,
        node_idx: usize,
        raw_value: &TypedValue,
        state: &ResolveState<E>,
    ) -> (FxHashMap<Astr, Arc<TypedValue>>, Option<TrackedDeque<Value>>)
    where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[node_idx];
        let interner = self.interner;
        let mut locals = FxHashMap::default();
        let mut origin: Option<TrackedDeque<Value>> = None;

        // @self = previous stored value (or initial_value on first run)
        if let Some(prev) = self.load_self_value(node_idx, state) {
            let self_val = match &node.strategy.persistency {
                CompiledPersistency::Sequence { .. } => {
                    let deque = match prev.value() {
                        Value::Lazy(LazyValue::Deque(d)) => d.clone(),
                        Value::Lazy(LazyValue::List(items)) => TrackedDeque::from_vec(items.clone()),
                        _ => panic!("sequence mode @self: expected Deque or List"),
                    };
                    origin = Some(deque.clone());
                    let mut working = deque;
                    working.checkpoint();
                    TypedValue::new(Arc::new(Value::deque(working)), Ty::Infer)
                }
                _ => prev,
            };
            locals.insert(interner.intern("self"), Arc::new(self_val));
        } else if matches!(&node.strategy.persistency, CompiledPersistency::Sequence { .. }) {
            // First run with no stored value — inject empty sequence with checkpoint
            let deque = TrackedDeque::new();
            origin = Some(deque.clone());
            let mut working = deque;
            working.checkpoint();
            locals.insert(interner.intern("self"), Arc::new(TypedValue::new(Arc::new(Value::deque(working)), Ty::Infer)));
        }

        // @raw = this turn's raw output. Strictly bind-internal — never
        // exposed via turn_context or dep wake.
        locals.insert(interner.intern("raw"), Arc::new(raw_value.clone()));

        (locals, origin)
    }

    // -----------------------------------------------------------------------
    // Root finalize: assert → cache → store → bind
    // -----------------------------------------------------------------------

    fn start_root_finalize<'j, E>(
        &self,
        node_idx: usize,
        value: TypedValue,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[node_idx];

        if let Some(ref assert_script) = node.strategy.assert {
            let (assert_local, _origin) = self.prepare_bind_locals(node_idx, &value, state);
            debug!(node = %self.interner.resolve(node.name), "evaluating assert");
            self.spawn_script_task(
                assert_script,
                assert_local,
                ScriptPurpose::Assert { node_idx, value },
                lp,
            );
        } else {
            self.apply_root_finalize(node_idx, &value, &local, lp, state);
        }
        Ok(())
    }

    fn apply_root_finalize<'j, E>(
        &self,
        node_idx: usize,
        value: &TypedValue,
        local: &FxHashMap<Astr, Arc<TypedValue>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[node_idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);

        // IfModified: cache update
        if matches!(node.strategy.execution, CompiledExecution::IfModified { .. })
            && let Some(bind_val) = local.get(&interner.intern("bind"))
        {
            state
                .bind_cache
                .entry(node.name)
                .or_default()
                .push(((**bind_val).clone(), Arc::new(value.clone())));
        }

        // Mode: Sequence/Patch → spawn bind script before completing.
        //
        // For Deque/Patch modes, turn_context is NOT set here with the raw value.
        // It will be set to the bind result (accumulated @self) when the bind
        // script completes. This ensures @node_name === @self (not @raw).
        //
        // @raw is strictly internal to the bind script — it is injected as a
        // local variable and MUST NOT leak to turn_context or dep wakers.
        // External consumers of @node_name always see the accumulated value.
        match &node.strategy.persistency {
            CompiledPersistency::Sequence { bind } | CompiledPersistency::Patch { bind } => {
                let (bind_local, origin) = self.prepare_bind_locals(node_idx, &value, state);
                debug!(node = %node_name_str, "evaluating bind script");
                self.spawn_script_task(
                    bind,
                    bind_local,
                    ScriptPurpose::BindScript {
                        node_idx,
                        value: value.clone(),
                        origin,
                    },
                    lp,
                );
                return; // BindScript will remove from remaining_roots
            }
            CompiledPersistency::Snapshot => {
                state.persist(
                    node_name_str,
                    StoragePatch::Snapshot(value.clone()),
                );
                state.cache(node.name, Arc::new(value.clone()));
            }
            CompiledPersistency::Ephemeral => {
                state.cache(node.name, Arc::new(value.clone()));
            }
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
    fn finish_dep_wake<'j, E>(
        &self,
        node_idx: usize,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
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

    fn start_prepare<'j, E>(
        &self,
        idx: usize,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[idx];
        let interner = self.interner;
        let node_name_str = interner.resolve(node.name);
        info!(node = %node_name_str, "prepare node");

        // IfModified: spawn key script
        if let CompiledExecution::IfModified { ref key } = node.strategy.execution {
            self.spawn_script_task(
                key,
                local,
                ScriptPurpose::IfModifiedKey { node_idx: idx },
                lp,
            );
            lp.in_flight.insert(idx);
            return;
        }

        // InitialValue
        if let Some(init_script) = node_initial_value(node) {
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
    fn start_after_if_modified<'j, E>(
        &self,
        node_idx: usize,
        local: FxHashMap<Astr, Arc<TypedValue>>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[node_idx];

        if let Some(init_script) = node_initial_value(node) {
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

    /// Try to load the existing @self value from turn_context (this turn's
    /// update) first, then storage (previous turn's value).
    fn load_self_value<'j, E>(&self, idx: usize, state: &ResolveState<E>) -> Option<TypedValue>
    where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[idx];
        let name_str = self.interner.resolve(node.name);
        state.load_self(node.name, name_str).map(|arc| TypedValue::clone(&arc))
    }

    // -----------------------------------------------------------------------
    // Store (common for root and dep finalize)
    // -----------------------------------------------------------------------

    /// Store a dep node's result. All strategies go to turn_context;
    /// merged to storage at turn end by ChatEngine.
    fn apply_store<'j, E>(&self, idx: usize, value: &TypedValue, state: &mut ResolveState<E>)
    where
        E: EntryMut<'j>,
    {
        let node = &self.nodes[idx];
        state.cache(node.name, Arc::new(value.clone()));
    }

    // -----------------------------------------------------------------------
    // Retry
    // -----------------------------------------------------------------------

    fn try_retry<'j, E>(
        &self,
        idx: usize,
        node_name: &str,
        error: &RuntimeError,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> bool
    where
        E: EntryMut<'j>,
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

    async fn resolve_external<'j, E>(
        &self,
        name: Astr,
        state: &mut ResolveState<E>,
    ) -> Result<Arc<TypedValue>, ResolveError>
    where
        E: EntryMut<'j>,
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
                state.cache(name, Arc::clone(&arc));
                Ok(arc)
            }
            Resolved::Persist(value) => {
                debug!(name = %name_str, kind = "persist", "external resolver returned");
                let arc = Arc::new(value);
                state.persist(name_str, StoragePatch::Snapshot(TypedValue::clone(&arc)));
                Ok(arc)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Eager dependency scheduling
    // -----------------------------------------------------------------------

    fn eager_node_deps(&self, idx: usize, entry: &dyn EntryRef<'_>) -> Vec<usize> {
        let node = &self.nodes[idx];
        let known = node.known_from_entry(self.interner, entry);
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

        if let Some(iv) = node_initial_value(node)
            && entry.get(self.interner.resolve(node.name)).is_none()
        {
            eager.extend(iv.context_keys.iter().copied());
        }
        match &node.strategy.execution {
            CompiledExecution::IfModified { key } => {
                eager.extend(key.context_keys.iter().copied());
            }
            _ => {}
        }
        match &node.strategy.persistency {
            CompiledPersistency::Sequence { bind } | CompiledPersistency::Patch { bind } => {
                eager.extend(bind.context_keys.iter().copied());
            }
            _ => {}
        }

        eager
            .iter()
            .filter_map(|name| self.name_to_idx.get(name).copied())
            .filter(|&i| i != idx)
            .collect()
    }

    /// Always 전략 + initial_value를 가진 노드인지.
    /// 이 조건이면 동시 실행 시 @self lost update가 발생하므로 직렬화 필요.
    fn needs_serialized(&self, idx: usize) -> bool {
        let node = &self.nodes[idx];
        matches!(node.strategy.execution, CompiledExecution::Always)
            && node.strategy.initial_value.is_some()
    }

    fn needs_resolve<E>(&self, idx: usize, state: &ResolveState<E>, no_execute: bool) -> bool {
        if no_execute { return false; }
        let name = self.nodes[idx].name;
        match &self.nodes[idx].strategy.execution {
            // Always: re-execute on every reference within a turn.
            CompiledExecution::Always => true,
            // All other strategies: execute once per turn.
            // Only turn_context counts — storage values from previous turns
            // do NOT satisfy this. The node must re-run each turn.
            _ => !state.is_cached(&name),
        }
    }

    fn try_eager_schedule<'j, E>(
        &self,
        completed_idx: usize,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
    {
        if completed_idx >= self.rdeps.len() {
            return;
        }
        for &candidate in &self.rdeps[completed_idx] {
            if lp.in_flight.contains(&candidate) || !self.needs_resolve(candidate, state, lp.no_execute) {
                continue;
            }
            let eager_deps = {
                let entry_ref = state.entry.as_ref();
                self.eager_node_deps(candidate, &entry_ref)
            };
            if eager_deps
                .iter()
                .all(|&dep| !self.needs_resolve(dep, state, lp.no_execute))
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
        local: FxHashMap<Astr, Arc<TypedValue>>,
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
        local: FxHashMap<Astr, Arc<TypedValue>>,
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
        | ScriptPurpose::BindScript { node_idx, .. } => *node_idx,
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ParkedDiag {
    /// The task that is parked (node or script-for-node name).
    pub task: String,
    /// The `@name` it is waiting for.
    pub waiting_for: String,
}

#[derive(Debug)]
pub enum ResolveError {
    /// A context key could not be resolved at all.
    UnresolvedContext(String),
    /// The event loop drained but some root nodes never completed (deadlock).
    Deadlock {
        stuck_roots: Vec<String>,
        dep_waiters: Vec<String>,
        serialized_queue: Vec<String>,
        turn_context_keys: Vec<String>,
        in_flight: Vec<String>,
        parked: Vec<ParkedDiag>,
    },
    Runtime { node: String, error: RuntimeError },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnresolvedContext(name) => write!(f, "unresolved context: @{name}"),
            ResolveError::Deadlock { stuck_roots, dep_waiters, serialized_queue, turn_context_keys, in_flight, parked } => {
                write!(f, "deadlock: roots [{}]", stuck_roots.join(", "))?;
                if !dep_waiters.is_empty() {
                    write!(f, ", dep_waiters [{}]", dep_waiters.join(", "))?;
                }
                if !serialized_queue.is_empty() {
                    write!(f, ", serial_q [{}]", serialized_queue.join(", "))?;
                }
                if !in_flight.is_empty() {
                    write!(f, ", in_flight [{}]", in_flight.join(", "))?;
                }
                write!(f, ", turn_ctx [{}]", turn_context_keys.join(", "))?;
                for p in parked {
                    write!(f, "; {} -> @{}", p.task, p.waiting_for)?;
                }
                Ok(())
            }
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
