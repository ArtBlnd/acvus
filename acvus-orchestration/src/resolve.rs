use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{Interpreter, IterHandle, LazyValue, PureValue, RuntimeError, Stepped, TypedValue, Value, ValueKind};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::Ty;
use acvus_mir::analysis::reachable_context::partition_context_keys;
use acvus_utils::{Astr, ContextRequest, Coroutine, ExternCallRequest, Interner, TrackedDeque};
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, info, warn};

use crate::CompiledNodeKind;
use crate::compile::{CompiledExecution, CompiledMessage, CompiledNode, CompiledNodeGraph, CompiledScript, NodeId, NodeRole, PersistMode};
use crate::dsl::{KEY_SELF, KEY_RAW, KEY_BIND, KEY_TURN_INDEX};

/// Access a node's initial_value script.
fn node_initial_value(node: &CompiledNode) -> Option<&CompiledScript> {
    node.strategy.initial_value.as_ref()
}
use crate::node::Node;
use crate::storage::{EntryMut, EntryRef, PatchDiff};

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
    pub turn_context: FxHashMap<Astr, TypedValue>,
    pub bind_cache: FxHashMap<Astr, Vec<(TypedValue, TypedValue)>>,
}

impl<E> ResolveState<E> {
    /// Cache a value in turn_context. This is the **single entry point**
    /// for all turn_context insertions.
    pub fn cache(&mut self, name: Astr, value: TypedValue) {
        self.turn_context.insert(name, value);
    }

    /// Whether a value is already cached for this turn.
    pub fn is_cached(&self, name: &Astr) -> bool {
        self.turn_context.contains_key(name)
    }

    /// Get a cached value from turn_context.
    pub fn get_cached(&self, name: &Astr) -> Option<TypedValue> {
        self.turn_context.get(name).cloned()
    }
}

impl<'j, E: EntryMut<'j>> ResolveState<E> {
    /// Persist a patch to storage.
    pub fn persist_patch(&mut self, key: &str, diff: PatchDiff, ty: Ty) {
        self.entry.record_patch(key, diff, ty);
    }

    /// Persist a patch and cache the stored form.
    pub fn persist_patch_and_cache(&mut self, name: Astr, key: &str, diff: PatchDiff, ty: Ty) -> TypedValue {
        self.entry.record_patch(key, diff, ty);
        let stored = self.entry.get(key)
            .expect("value must be in storage after persist_patch");
        self.turn_context.insert(name, stored.clone());
        stored
    }

    /// Persist a sequence diff. Consumes the working TrackedDeque,
    /// returns the new deque with evolved checksum.
    pub fn persist_sequence_diff(&mut self, key: &str, working: TrackedDeque<Value>, ty: Ty) {
        self.entry.record_sequence_diff(key, working, ty);
    }

    /// Persist a sequence diff and cache the stored form.
    pub fn persist_sequence_diff_and_cache(
        &mut self,
        name: Astr,
        key: &str,
        working: TrackedDeque<Value>,
        ty: Ty,
    ) -> TypedValue {
        let _new_deque = self.entry.record_sequence_diff(key, working, ty);
        let stored = self.entry.get(key)
            .expect("value must be in storage after persist_sequence_diff");
        self.turn_context.insert(name, stored.clone());
        stored
    }

    /// Load a value by name: turn_context first (most recent), then storage.
    pub fn load(&self, name: Astr, name_str: &str) -> Option<TypedValue> {
        self.turn_context
            .get(&name)
            .cloned()
            .or_else(|| self.entry.get(name_str))
    }

    /// Load @self for a node: turn_context first (this turn's update),
    /// then storage (previous turn's value).
    pub fn load_self(&self, name: Astr, name_str: &str) -> Option<TypedValue> {
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
        node_id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
    },
    Script {
        phase: Phase,
        local: FxHashMap<Astr, TypedValue>,
    },
}

// ---------------------------------------------------------------------------
// NextStep / Phase — intent-based lifecycle state machine
// ---------------------------------------------------------------------------

/// What to do next in a node's lifecycle.
/// Handlers return this to declare their intent. The event loop executes it.
enum NextStep {
    /// Evaluate a script/coroutine asynchronously, then route the result to a Phase handler.
    Eval {
        _node_id: NodeId,
        source: EvalSource,
        local: FxHashMap<Astr, TypedValue>,
        then: Phase,
    },
    /// Inject @self (role-dependent) and spawn the node's coroutine.
    Spawn {
        node_id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
    },
    /// Node is done. Execute common completion: cache + mark_complete + wake deps.
    /// The exact behavior depends on NodeRole (handled inside execute).
    Propagate {
        node_id: NodeId,
        value: TypedValue,
    },
    /// Nothing to do.
    Noop,
}

/// Source for Eval — compiled script or raw coroutine.
enum EvalSource {
    Script(CompiledScript),
    Coroutine(Coroutine<TypedValue, RuntimeError>),
}

/// Async evaluation completion point. Each variant's handler returns NextStep.
enum Phase {
    /// IfModified key evaluated → check bind cache for hit.
    IfModifiedKey { node_id: NodeId },
    /// InitialValue evaluated → persist and spawn or propagate.
    InitialValue { node_id: NodeId },
    /// Assert evaluated → if passed, finalize; if failed, retry or error.
    Assert { node_id: NodeId, value: TypedValue },
    /// Sequence collect completed → persist via record_sequence_diff, propagate.
    PersistSequence { node_id: NodeId },
}

enum PendingRequest {
    Context(ContextRequest<TypedValue>),
    ExternCall(ExternCallRequest<TypedValue>),
}

impl PendingRequest {
    fn resolve(self, value: TypedValue) {
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

/// Tracks node lifecycle: which nodes are in-flight, who is waiting for
/// whom, and the serialization queue for nodes that must not run concurrently.
struct NodeTracker {
    in_flight: FxHashSet<NodeId>,
    dep_waiters: FxHashMap<NodeId, Vec<Parked>>,
    /// Always + initial_value Expr 노드의 직렬화 큐.
    ///
    /// 이런 노드는 @self를 읽고 갱신하므로, 동시에 여러 인스턴스가 실행되면
    /// 같은 @self를 읽어 lost update가 발생한다.
    /// 이미 in_flight인 경우 여기에 park하고, 완료 시 하나씩 깨워 재실행한다.
    serialized_queue: FxHashMap<NodeId, VecDeque<Parked>>,
}

impl NodeTracker {
    fn take_waiters(&mut self, id: &NodeId) -> Vec<Parked> {
        self.dep_waiters.remove(id).unwrap_or_default()
    }
}

/// Progress state for streaming root nodes.
struct RootProgress {
    /// Last value emitted by the root node (for finalization on Done).
    last_emit: Option<TypedValue>,
    /// For streaming roots: the original sequence value before unpack.
    /// Used for storage finalization after unpack Done.
    finalize_value: Option<TypedValue>,
}

pub struct LoopState<'a> {
    next_task_id: TaskId,
    meta: FxHashMap<TaskId, TaskMeta>,
    futs: FuturesUnordered<Pin<Box<dyn Future<Output = StepResult> + Send + 'a>>>,
    tracker: NodeTracker,
    pub remaining_roots: FxHashSet<NodeId>,
    pub retry_state: FxHashMap<NodeId, (u32, u32, FxHashMap<Astr, TypedValue>)>,
    /// When true, dependency nodes are NOT executed — served from storage only.
    /// Root (entrypoint) nodes still execute normally.
    no_execute: bool,
    /// When true, only evaluate initial_value scripts and store results.
    /// Node body execution is skipped entirely.
    initial_value_only: bool,
    root: RootProgress,
}

impl<'a> LoopState<'a> {
    pub fn new(no_execute: bool) -> Self {
        Self {
            next_task_id: 0,
            meta: FxHashMap::default(),
            futs: FuturesUnordered::new(),
            tracker: NodeTracker {
                in_flight: FxHashSet::default(),
                dep_waiters: FxHashMap::default(),
                serialized_queue: FxHashMap::default(),
            },
            remaining_roots: FxHashSet::default(),
            retry_state: FxHashMap::default(),
            no_execute,
            initial_value_only: false,
            root: RootProgress {
                last_emit: None,
                finalize_value: None,
            },
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

    fn local(&self, task_id: TaskId) -> &FxHashMap<Astr, TypedValue> {
        match self.meta.get(&task_id) {
            Some(TaskMeta::Node { local, .. }) | Some(TaskMeta::Script { local, .. }) => local,
            None => {
                static EMPTY: std::sync::LazyLock<FxHashMap<Astr, TypedValue>> =
                    std::sync::LazyLock::new(FxHashMap::default);
                &EMPTY
            }
        }
    }

    fn is_node_task(&self, task_id: TaskId) -> bool {
        matches!(self.meta.get(&task_id), Some(TaskMeta::Node { .. }))
    }

    fn wake_waiters(&mut self, id: NodeId, value: TypedValue) {
        for w in self.tracker.take_waiters(&id) {
            w.request.resolve(value.clone());
            self.enqueue_step(w.task_id, w.coroutine);
        }
    }

    fn park_for_dep(
        &mut self,
        id: NodeId,
        task_id: TaskId,
        coroutine: Coroutine<TypedValue, RuntimeError>,
        request: PendingRequest,
    ) {
        self.tracker.dep_waiters.entry(id).or_default().push(Parked {
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
    pub graph: &'a CompiledNodeGraph,
    pub node_table: &'a [Arc<dyn Node>],
    pub resolver: &'a R,
    pub extern_handler: &'a EH,
    pub interner: &'a Interner,
    pub rdeps: &'a [FxHashSet<NodeId>],
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
        id: NodeId,
        state: &mut ResolveState<E>,
        local: FxHashMap<Astr, TypedValue>,
        no_execute: bool,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        self.resolve_nodes(vec![(id, local)], state, no_execute).await
    }

    pub async fn resolve_nodes<'j, E>(
        &self,
        roots: Vec<(NodeId, FxHashMap<Astr, TypedValue>)>,
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

        for (id, local) in roots {
            let max_retries = self.graph.node(id).strategy.retry;
            lp.retry_state.insert(id, (max_retries, 0, local.clone()));
            lp.remaining_roots.insert(id);
            let step = self.on_prepare(id, local, true, &mut lp, state)?;
            self.execute(step, &mut lp, state)?;
        }

        while let Some(sr) = lp.futs.next().await {
            let StepResult {
                task_id,
                coroutine,
                stepped,
            } = sr;

            match stepped {
                Stepped::Emit(value) => {
                    let next = self.on_emit(task_id, value, &mut lp, state)?;
                    self.execute(next, &mut lp, state)?;
                }
                Stepped::Done => {
                    if let Some(TaskMeta::Node { node_id, .. }) = lp.meta.get(&task_id) {
                        warn!(
                            node = %self.interner.resolve(self.graph.node(*node_id).name),
                            "coroutine finished without emit"
                        );
                    }
                    let next = self.on_emit(task_id, TypedValue::unit(), &mut lp, state)?;
                    self.execute(next, &mut lp, state)?;
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

        let stuck_roots: Vec<String> = lp.remaining_roots.iter().map(|&id| {
            self.interner.resolve(self.graph.node(id).name).to_string()
        }).collect();
        let dep_waiters: Vec<String> = lp.tracker.dep_waiters.keys().map(|&id| {
            self.interner.resolve(self.graph.node(id).name).to_string()
        }).collect();
        let serialized_queue: Vec<String> = lp.tracker.serialized_queue.keys().map(|&id| {
            self.interner.resolve(self.graph.node(id).name).to_string()
        }).collect();
        let turn_context_keys: Vec<String> = state.turn_context.keys().map(|&name| {
            self.interner.resolve(name).to_string()
        }).collect();
        let in_flight: Vec<String> = lp.tracker.in_flight.iter().map(|&id| {
            self.interner.resolve(self.graph.node(id).name).to_string()
        }).collect();
        let mut parked: Vec<ParkedDiag> = Vec::new();
        // dep_waiters
        for (id, waiters) in &lp.tracker.dep_waiters {
            let waiting_for = self.interner.resolve(self.graph.node(*id).name).to_string();
            for w in waiters {
                let task = match lp.meta.get(&w.task_id) {
                    Some(TaskMeta::Node { node_id, .. }) =>
                        self.interner.resolve(self.graph.node(*node_id).name).to_string(),
                    Some(TaskMeta::Script { phase, .. }) => {
                        let id = phase_node_id(phase);
                        format!("script:{}", self.interner.resolve(self.graph.node(id).name))
                    }
                    None => format!("task#{}", w.task_id),
                };
                parked.push(ParkedDiag { task, waiting_for: waiting_for.clone() });
            }
        }
        // serialized_queue
        for (&id, queue) in &lp.tracker.serialized_queue {
            let waiting_for = self.interner.resolve(self.graph.node(id).name).to_string();
            for w in queue {
                let task = match lp.meta.get(&w.task_id) {
                    Some(TaskMeta::Node { node_id, .. }) =>
                        self.interner.resolve(self.graph.node(*node_id).name).to_string(),
                    Some(TaskMeta::Script { phase, .. }) => {
                        let id = phase_node_id(phase);
                        format!("script:{}", self.interner.resolve(self.graph.node(id).name))
                    }
                    None => format!("task#{}", w.task_id),
                };
                parked.push(ParkedDiag { task, waiting_for: format!("serial:{}", waiting_for) });
            }
        }

        Err(ResolveError::Deadlock { stuck_roots, dep_waiters, serialized_queue, parked, turn_context_keys, in_flight })
    }

    // -----------------------------------------------------------------------
    // Streaming loop — drives the resolver one step at a time
    // -----------------------------------------------------------------------

    /// Resume the resolver loop until the ROOT node yields or finishes.
    ///
    /// - ROOT `Emit` → returns `Some(value)`. Coroutine is re-enqueued for next yield.
    /// - ROOT `Done` → returns `None`. Root is finalized.
    /// - Dep `Emit`/`Done` → handled internally (finalize, wake deps).
    /// - `NeedContext`/`NeedExternCall` → handled internally (spawn dep, call resolver).
    ///
    /// The caller invokes this repeatedly to stream items from the root node.
    pub async fn resume_loop<'j, E>(
        &self,
        lp: &mut LoopState<'static>,
        state: &mut ResolveState<E>,
    ) -> Result<Option<TypedValue>, ResolveError>
    where
        E: EntryMut<'j>,
    {
        while let Some(step) = lp.futs.next().await {
            let StepResult {
                task_id,
                coroutine,
                stepped,
            } = step;

            match stepped {
                Stepped::Emit(value) => {
                    let is_root = matches!(
                        lp.meta.get(&task_id),
                        Some(TaskMeta::Node { is_root: true, .. })
                    );
                    if is_root {
                        // ROOT yield — check if the value is iterable.
                        // If so, replace the ROOT coroutine with an unpack
                        // coroutine that pulls items via exec_next and
                        // yield_vals each one. This preserves lazy evaluation.
                        if is_iterable(&value) {
                            // Save original for storage finalization
                            lp.root.finalize_value = Some(value.clone());
                            // Drop the original coroutine (it already emitted).
                            // Spawn an unpack coroutine in its place.
                            let meta = lp.meta.remove(&task_id).unwrap();
                            let unpack = unpack_coroutine(value, self.interner.clone());
                            let new_tid = lp.alloc_id();
                            lp.meta.insert(new_tid, meta);
                            lp.enqueue_step(new_tid, unpack);
                            // Continue the loop — next step yields the first unpacked item.
                            continue;
                        }

                        // Scalar — return directly to caller.
                        lp.root.last_emit = Some(value.clone());
                        lp.enqueue_step(task_id, coroutine);
                        return Ok(Some(value));
                    } else {
                        let next = self.on_emit(task_id, value, lp, state)?;
                        self.execute(next, lp, state)?;
                    }
                }
                Stepped::Done => {
                    let is_root = matches!(
                        lp.meta.get(&task_id),
                        Some(TaskMeta::Node { is_root: true, .. })
                    );
                    if is_root {
                        // ROOT done → finalize with last emitted value.
                        // Prefer root.finalize_value (original iterable before unpack),
                        // then root.last_emit, then unit.
                        let final_value = lp.root.finalize_value.take()
                            .or_else(|| lp.root.last_emit.take())
                            .unwrap_or_else(TypedValue::unit);
                        let next = self.on_emit(task_id, final_value, lp, state)?;
                        self.execute(next, lp, state)?;
                        if lp.remaining_roots.is_empty() {
                            return Ok(None);
                        }
                    } else {
                        if lp.meta.contains_key(&task_id) {
                            warn!("dep coroutine finished without emit");
                        }
                        let next = self.on_emit(task_id, TypedValue::unit(), lp, state)?;
                        self.execute(next, lp, state)?;
                    }
                }
                Stepped::NeedContext(request) => {
                    self.handle_need_context(task_id, coroutine, request, lp, state)
                        .await?;
                }
                Stepped::NeedExternCall(request) => {
                    self.handle_need_extern_call(task_id, coroutine, request, lp, state)
                        .await?;
                }
                Stepped::Error(e) => {
                    self.handle_error(task_id, e, lp, state)?;
                }
            }
        }

        // Futs exhausted without root completing
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // Pre-populate initial values
    // -----------------------------------------------------------------------

    /// Evaluate initial_value scripts for non-ephemeral nodes that have no
    /// stored value yet, and persist the results to storage.
    ///
    /// Must be called before `resolve_node` / `resolve_nodes` when a
    /// `no_execute` pass may reference non-ephemeral nodes. Ensures that
    /// every non-ephemeral node with an `initial_value` has a value in
    /// storage before the display pass runs.
    pub async fn populate_initial_values<'j, E>(
        &self,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        let needs_init: Vec<NodeId> = self
            .graph.nodes
            .iter()
            .filter(|node| {
                node.strategy.initial_value.is_some()
                    && matches!(node.role, NodeRole::Persistent { .. })
                    && self.load_self_value(node.id, state).is_none()
            })
            .map(|node| node.id)
            .collect();

        if needs_init.is_empty() {
            return Ok(());
        }

        let mut lp = LoopState::new(false);
        lp.initial_value_only = true;

        for id in needs_init {
            lp.remaining_roots.insert(id);
            let step = self.on_prepare(id, FxHashMap::default(), true, &mut lp, state)?;
            self.execute(step, &mut lp, state)?;
        }

        while let Some(sr) = lp.futs.next().await {
            let StepResult {
                task_id,
                coroutine,
                stepped,
            } = sr;

            match stepped {
                Stepped::Emit(value) => {
                    let next = self.on_emit(task_id, value, &mut lp, state)?;
                    self.execute(next, &mut lp, state)?;
                }
                Stepped::Done => {
                    let next = self.on_emit(task_id, TypedValue::unit(), &mut lp, state)?;
                    self.execute(next, &mut lp, state)?;
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

        Ok(())
    }

    // -----------------------------------------------------------------------
    // NextStep execute dispatcher
    // -----------------------------------------------------------------------

    fn execute<'j, E>(
        &self,
        step: NextStep,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        match step {
            NextStep::Eval { _node_id: _, source, local, then } => {
                let tid = lp.alloc_id();
                let coroutine = match source {
                    EvalSource::Script(script) => {
                        Interpreter::new(self.interner, script.module.clone()).execute()
                    }
                    EvalSource::Coroutine(c) => c,
                };
                lp.meta.insert(tid, TaskMeta::Script { phase: then, local });
                lp.enqueue_step(tid, coroutine);
                Ok(())
            }
            NextStep::Spawn { node_id, local, is_root } => {
                self.execute_spawn(node_id, local, is_root, lp, state);
                Ok(())
            }
            NextStep::Propagate { node_id, value } => {
                self.execute_propagate(node_id, value, lp, state)
            }
            NextStep::Noop => Ok(()),
        }
    }

    /// Role-dependent @self injection + spawn the node's coroutine.
    fn execute_spawn<'j, E>(
        &self,
        node_id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    )
    where
        E: EntryMut<'j>,
    {
        let mut local = local;

        // @self injection — role determines how
        match &self.graph.node(node_id).role {
            NodeRole::Body { bind_id } => {
                if let Some(prev) = self.load_self_value(*bind_id, state) {
                    local.insert(self.interner.intern(KEY_SELF), prev);
                }
            }
            NodeRole::Persistent { .. } => {
                self.prepare_bind_self(node_id, &mut local, lp, state);
            }
            NodeRole::Standalone => {
                if let Some(prev) = self.load_self_value(node_id, state) {
                    local.insert(self.interner.intern(KEY_SELF), prev);
                }
            }
        }

        // Spawn coroutine
        let tid = lp.alloc_id();
        let coroutine = self.node_table[node_id.index()].spawn(local.clone());
        lp.meta.insert(tid, TaskMeta::Node { node_id, local, is_root });
        lp.tracker.in_flight.insert(node_id);
        lp.enqueue_step(tid, coroutine);
    }

    /// Role-dependent completion: cache + mark_complete + wake deps.
    fn execute_propagate<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        match &self.graph.node(node_id).role {
            NodeRole::Body { .. } => {
                // Body: just wake @raw waiters, no cache/persist
                lp.tracker.in_flight.remove(&node_id);
                lp.wake_waiters(node_id, value);
            }
            NodeRole::Persistent { .. } | NodeRole::Standalone => {
                // Cache in turn_context
                let node = self.graph.node(node_id);
                let node_name_str = self.interner.resolve(node.name);
                state.cache(node.name, value.clone());
                info!(node = %node_name_str, "resolve node complete");
                lp.remaining_roots.remove(&node_id);
                self.finish_dep_wake(node_id, value, lp, state)?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // on_emit — unified dispatch for Emit/Done
    // -----------------------------------------------------------------------

    fn on_emit<'j, E>(
        &self,
        task_id: TaskId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        match lp.meta.remove(&task_id) {
            Some(TaskMeta::Node { node_id, local, is_root: _ }) => {
                self.on_node_emit(node_id, value, local, lp, state)
            }
            Some(TaskMeta::Script { phase, local }) => {
                self.on_phase_complete(phase, value, local, lp, state)
            }
            None => {
                panic!(
                    "on_emit: task {} emitted but has no meta — \
                     possible second emit from a node whose meta was already consumed",
                    task_id,
                );
            }
        }
    }

    /// Handle emit from a Node task — dispatch by role.
    fn on_node_emit<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        local: FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(node_id);

        // Always strategy: remove from in_flight on every emit
        if matches!(node.strategy.execution, CompiledExecution::Always) {
            lp.tracker.in_flight.remove(&node_id);
        }

        match &node.role {
            NodeRole::Body { .. } => {
                // Body: check assert, then propagate (wake @raw waiters)
                if let Some(assert_script) = &node.strategy.assert {
                    debug!(node = %self.interner.resolve(node.name), "evaluating body assert");
                    let mut assert_local = FxHashMap::default();
                    assert_local.insert(self.interner.intern(KEY_RAW), value.clone());
                    Ok(NextStep::Eval {
                        _node_id: node_id,
                        source: EvalSource::Script(assert_script.clone()),
                        local: assert_local,
                        then: Phase::Assert { node_id, value },
                    })
                } else {
                    Ok(NextStep::Propagate { node_id, value })
                }
            }
            NodeRole::Persistent { .. } => {
                // IfModified bind cache update
                if matches!(node.strategy.execution, CompiledExecution::IfModified { .. }) {
                    if let Some(bind_val) = local.get(&self.interner.intern(KEY_BIND)) {
                        state.bind_cache.entry(node.name).or_default()
                            .push((bind_val.clone(), value.clone()));
                    }
                }
                // Assert check, then finalize
                if let Some(assert_script) = &node.strategy.assert {
                    debug!(node = %self.interner.resolve(node.name), "evaluating bind assert");
                    let mut assert_local = FxHashMap::default();
                    assert_local.insert(self.interner.intern(KEY_RAW), value.clone());
                    Ok(NextStep::Eval {
                        _node_id: node_id,
                        source: EvalSource::Script(assert_script.clone()),
                        local: assert_local,
                        then: Phase::Assert { node_id, value },
                    })
                } else {
                    self.finalize_persistent(node_id, &value, lp, state)
                }
            }
            NodeRole::Standalone => {
                // IfModified bind cache update
                if matches!(node.strategy.execution, CompiledExecution::IfModified { .. }) {
                    if let Some(bind_val) = local.get(&self.interner.intern(KEY_BIND)) {
                        state.bind_cache.entry(node.name).or_default()
                            .push((bind_val.clone(), value.clone()));
                    }
                }
                // Assert check, then propagate
                if let Some(assert_script) = &node.strategy.assert {
                    debug!(node = %self.interner.resolve(node.name), "evaluating assert");
                    let mut assert_local = FxHashMap::default();
                    assert_local.insert(self.interner.intern(KEY_RAW), value.clone());
                    Ok(NextStep::Eval {
                        _node_id: node_id,
                        source: EvalSource::Script(assert_script.clone()),
                        local: assert_local,
                        then: Phase::Assert { node_id, value },
                    })
                } else {
                    Ok(NextStep::Propagate { node_id, value })
                }
            }
        }
    }

    /// Finalize a persistent (Sequence/Patch) node — returns NextStep.
    fn finalize_persistent<'j, E>(
        &self,
        node_id: NodeId,
        value: &TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(node_id);
        let node_name_str = self.interner.resolve(node.name);

        match &node.role {
            NodeRole::Persistent { mode: PersistMode::Sequence, .. } => {
                let collect_coroutine = collect_sequence_coroutine(value.clone());
                Ok(NextStep::Eval {
                    _node_id: node_id,
                    source: EvalSource::Coroutine(collect_coroutine),
                    local: FxHashMap::default(),
                    then: Phase::PersistSequence { node_id },
                })
            }
            NodeRole::Persistent { mode: PersistMode::Patch, .. } => {
                let old_value = state.load(node.name, node_name_str);
                let diff = old_value
                    .as_ref()
                    .and_then(|old| PatchDiff::compute(old.value(), value.value()))
                    .unwrap_or_else(|| PatchDiff::set(value.clone()));
                let ty = node.output_ty.clone();
                let stored = state.persist_patch_and_cache(node.name, node_name_str, diff, ty);
                Ok(NextStep::Propagate { node_id, value: stored })
            }
            _ => {
                unreachable!("finalize_persistent called on non-Persistent node")
            }
        }
    }

    /// Handle completion of a Phase (script) task — dispatch by phase variant.
    fn on_phase_complete<'j, E>(
        &self,
        phase: Phase,
        value: TypedValue,
        local: FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        match phase {
            Phase::IfModifiedKey { node_id } => {
                self.on_if_modified_key(node_id, value, local, lp, state)
            }
            Phase::InitialValue { node_id } => {
                self.on_initial_value(node_id, value, local, lp, state)
            }
            Phase::Assert { node_id, value: original_value } => {
                self.on_assert(node_id, value, original_value, local, lp, state)
            }
            Phase::PersistSequence { node_id } => {
                self.on_persist_sequence(node_id, value, lp, state)
            }
        }
    }

    /// IfModified key evaluated → check bind cache for hit, else spawn.
    fn on_if_modified_key<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        local: FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(node_id);
        let node_name_str = self.interner.resolve(node.name);

        // Cache hit check
        if let Some(entries) = state.bind_cache.get(&node.name)
            && let Some((_, cached)) = entries.iter().find(|(v, _)| v == &value)
        {
            let cached_value = cached.clone();
            debug!(node = %node_name_str, "if_modified cache hit, skipping execution");
            state.cache(node.name, cached_value.clone());
            if lp.remaining_roots.contains(&node_id) {
                lp.remaining_roots.remove(&node_id);
            }
            lp.wake_waiters(node_id, cached_value);
            return Ok(NextStep::Noop);
        }

        debug!(node = %node_name_str, "if_modified cache miss, will execute");
        let mut new_local = local;
        new_local.insert(self.interner.intern(KEY_BIND), value);

        // Check initial_value
        if let Some(init_script) = node_initial_value(node) {
            if self.load_self_value(node_id, state).is_some() {
                // Has stored value → spawn directly (execute_spawn handles @self)
                let is_root = lp.remaining_roots.contains(&node_id);
                Ok(NextStep::Spawn { node_id, local: new_local, is_root })
            } else {
                Ok(NextStep::Eval {
                    _node_id: node_id,
                    source: EvalSource::Script(init_script.clone()),
                    local: new_local,
                    then: Phase::InitialValue { node_id },
                })
            }
        } else {
            // No initial_value → spawn directly
            let is_root = lp.remaining_roots.contains(&node_id);
            Ok(NextStep::Spawn { node_id, local: new_local, is_root })
        }
    }

    /// InitialValue evaluated → persist and spawn or propagate.
    fn on_initial_value<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        local: FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        if lp.initial_value_only {
            self.persist_initial_value(node_id, &value, state);
            lp.tracker.in_flight.remove(&node_id);
            lp.remaining_roots.remove(&node_id);
            // Must still go through finish_dep_wake to wake dependents
            self.finish_dep_wake(node_id, value, lp, state)?;
            return Ok(NextStep::Noop);
        }

        // Normal mode: persist initial value, then spawn
        self.persist_initial_value(node_id, &value, state);
        let is_root = lp.remaining_roots.contains(&node_id);
        Ok(NextStep::Spawn {
            node_id,
            local,
            is_root,
        })
    }

    /// Assert evaluated → if passed, finalize; if failed, retry or error.
    fn on_assert<'j, E>(
        &self,
        node_id: NodeId,
        assert_result: TypedValue,
        original_value: TypedValue,
        _local: FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(node_id);
        let node_name = self.interner.resolve(node.name).to_string();

        let Value::Pure(PureValue::Bool(passed)) = assert_result.value() else {
            return Err(ResolveError::Runtime {
                node: node_name,
                error: RuntimeError::unexpected_type(
                    "assert",
                    &[ValueKind::Bool],
                    assert_result.value().kind(),
                ),
            });
        };

        if !*passed {
            info!(node = %node_name, "assert failed, triggering retry");
            let error = RuntimeError::assert_failed();
            if !self.try_retry(node_id, &node_name, &error, lp, state)? {
                return Err(ResolveError::Runtime { node: node_name, error });
            }
            return Ok(NextStep::Noop);
        }

        // Assert passed — finalize based on role
        match &node.role {
            NodeRole::Body { .. } => {
                Ok(NextStep::Propagate { node_id, value: original_value })
            }
            NodeRole::Persistent { .. } => {
                self.finalize_persistent(node_id, &original_value, lp, state)
            }
            NodeRole::Standalone => {
                Ok(NextStep::Propagate { node_id, value: original_value })
            }
        }
    }

    /// Sequence collect completed → compute diff, persist, propagate.
    fn on_persist_sequence<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        _lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(node_id);
        let node_name_str = self.interner.resolve(node.name);
        let ty = node.output_ty.clone();

        let working_deque = extract_deque(&value, "PersistSequence result");

        let stored_value = state.persist_sequence_diff_and_cache(
            node.name,
            node_name_str,
            working_deque,
            ty,
        );

        info!(node = %node_name_str, "sequence collect + persist complete");

        Ok(NextStep::Propagate { node_id, value: stored_value })
    }

    // -----------------------------------------------------------------------
    // on_prepare — replaces start_prepare
    // -----------------------------------------------------------------------

    /// Public entry point: prepare and execute a node.
    /// This is the replacement for the old `start_prepare` method.
    pub fn start_prepare<'j, E>(
        &self,
        id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        let step = self.on_prepare(id, local, is_root, lp, state)?;
        self.execute(step, lp, state)
    }

    fn on_prepare<'j, E>(
        &self,
        id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(id);
        info!(node = %self.interner.resolve(node.name), "prepare node");

        match &node.role {
            NodeRole::Body { .. } => {
                // Body: just spawn (execute_spawn injects @self from bind storage)
                Ok(NextStep::Spawn { node_id: id, local, is_root })
            }
            NodeRole::Persistent { .. } | NodeRole::Standalone => {
                self.on_prepare_lifecycle(id, local, is_root, lp, state)
            }
        }
    }

    /// Unified prepare for Persistent + Standalone nodes.
    /// @self injection is handled by execute_spawn (role-dependent).
    fn on_prepare_lifecycle<'j, E>(
        &self,
        id: NodeId,
        local: FxHashMap<Astr, TypedValue>,
        is_root: bool,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<NextStep, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(id);
        let node_name_str = self.interner.resolve(node.name);

        // initial_value_only mode
        if lp.initial_value_only {
            if let Some(init_script) = node_initial_value(node) {
                if let Some(prev) = self.load_self_value(id, state) {
                    if is_root { lp.remaining_roots.remove(&id); }
                    self.finish_dep_wake(id, prev, lp, state)?;
                    return Ok(NextStep::Noop);
                }

                debug!(node = %node_name_str, "evaluating initial_value (populate)");
                lp.tracker.in_flight.insert(id);
                return Ok(NextStep::Eval {
                    _node_id: id,
                    source: EvalSource::Script(init_script.clone()),
                    local,
                    then: Phase::InitialValue { node_id: id },
                });
            }
            
            if is_root { lp.remaining_roots.remove(&id); }
            return Ok(NextStep::Noop);
        }

        // IfModified
        if let CompiledExecution::IfModified { key } = &node.strategy.execution {
            lp.tracker.in_flight.insert(id);
            return Ok(NextStep::Eval {
                _node_id: id,
                source: EvalSource::Script(key.clone()),
                local,
                then: Phase::IfModifiedKey { node_id: id },
            });
        }

        // InitialValue
        if let Some(init_script) = node_initial_value(node) {
            if self.load_self_value(id, state).is_some() {
                // Has stored value → spawn directly (execute_spawn injects @self)
                debug!(node = %node_name_str, "spawning coroutine");
                return Ok(NextStep::Spawn { node_id: id, local, is_root });
            } else {
                // First run → evaluate initial_value
                debug!(node = %node_name_str, "evaluating initial_value (first run)");
                lp.tracker.in_flight.insert(id);
                return Ok(NextStep::Eval {
                    _node_id: id,
                    source: EvalSource::Script(init_script.clone()),
                    local,
                    then: Phase::InitialValue { node_id: id },
                });
            }
        }

        // Default: just spawn
        debug!(node = %node_name_str, "spawning coroutine");
        Ok(NextStep::Spawn { node_id: id, local, is_root })
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
        let node_id = match &task_meta {
            Some(TaskMeta::Node { node_id, .. }) => *node_id,
            Some(TaskMeta::Script { phase, .. }) => phase_node_id(phase),
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
            .resolve(self.graph.node(node_id).name)
            .to_string();

        if is_root && self.try_retry(node_id, &node_name, &error, lp, state)? {
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
        if let Some(val) = lp.local(task_id).get(&name) {
            debug!(context = %name_str, "resolved from local");
            request.resolve(val.clone());
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

        // 2b. Scope resolution: bind node requesting @raw → spawn body on demand
        if let Some(TaskMeta::Node { node_id, .. }) = lp.meta.get(&task_id) {
            if let NodeRole::Persistent { body_id, .. } = &self.graph.node(*node_id).role {
                let raw_name = self.interner.intern(KEY_RAW);
                if name == raw_name {
                    let body_id = *body_id;
                    // @raw → spawn body node and park until it emits
                    if !lp.tracker.in_flight.contains(&body_id) {
                        debug!(context = %name_str, "bind requesting @raw, spawning body node");
                        let step = self.on_prepare(body_id, FxHashMap::default(), false, lp, state)?;
                        self.execute(step, lp, state)?;
                    }
                    lp.park_for_dep(body_id, task_id, coroutine, PendingRequest::Context(request));
                    return Ok(());
                }
            }
        }

        // 3. Known node — check before storage so Node > Storage
        if let Some(&dep_id) = self.graph.name_to_primary.get(&name) {
            // 3a. Function node → ExternFn handle
            if self.graph.node(dep_id).is_function {
                let node = self.graph.node(dep_id);
                let fn_ty = Ty::Fn {
                    params: node.fn_params.iter().map(|p| p.ty.clone()).collect(),
                    ret: Box::new(node.output_ty.clone()),
                    kind: acvus_mir::ty::FnKind::Extern,
                    captures: vec![],
                    effect: acvus_mir::ty::Effect::Effectful,
                };
                request.resolve(TypedValue::new(Value::extern_fn(name), fn_ty));
                lp.enqueue_step(task_id, coroutine);
                return Ok(());
            }

            // 3b. Node task → spawn dep if strategy says so
            if lp.is_node_task(task_id) {
                let needs = if lp.initial_value_only {
                    self.needs_resolve_initial_value_only(dep_id, state)
                } else {
                    self.needs_resolve(dep_id, state, lp.no_execute)?
                };
                if needs {
                    // Serialized node already in flight → park in serialized_queue
                    if self.needs_serialized(dep_id) && lp.tracker.in_flight.contains(&dep_id) {
                        lp.tracker.serialized_queue.entry(dep_id).or_default().push_back(Parked {
                            task_id, coroutine,
                            request: PendingRequest::Context(request),
                        });
                        return Ok(());
                    }
                    if !lp.tracker.in_flight.contains(&dep_id) {
                        debug!(context = %name_str, "spawning dependency node");
                        let step = self.on_prepare(dep_id, FxHashMap::default(), false, lp, state)?;
                        self.execute(step, lp, state)?;
                    }
                    lp.park_for_dep(dep_id, task_id, coroutine, PendingRequest::Context(request));
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
                if !lp.tracker.in_flight.contains(&dep_id) {
                    let step = self.on_prepare(dep_id, FxHashMap::default(), false, lp, state)?;
                    self.execute(step, lp, state)?;
                }
                lp.park_for_dep(dep_id, task_id, coroutine, PendingRequest::Context(request));
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
        if name == self.interner.intern(KEY_TURN_INDEX) {
            let depth = state.entry.depth();
            request.resolve(TypedValue::int(depth as i64));
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
            if let Some(&dep_id) = self.graph.name_to_primary.get(&name) {
                let args = request.args().to_vec();
                let node = self.graph.node(dep_id);
                let dep_local: FxHashMap<Astr, TypedValue> = if node.is_function {
                    // Tool call args come as a single Object — unpack fields by name.
                    if let Some(tv) = args.first()
                        && let Value::Lazy(LazyValue::Object(obj)) = tv.value()
                    {
                        node.fn_params
                            .iter()
                            .filter_map(|p| {
                                let val = obj.get(&p.name)?.clone();
                                Some((p.name, TypedValue::new(val, p.ty.clone())))
                            })
                            .collect()
                    } else {
                        // Positional fallback (e.g. script-level @fn(a, b) calls)
                        node.fn_params
                            .iter()
                            .zip(args.into_iter())
                            .map(|(p, val)| (p.name, val))
                            .collect()
                    }
                } else if let Some(tv) = args.first()
                    && let Value::Lazy(LazyValue::Object(obj)) = tv.value()
                {
                    // Extract field types from the Object's Ty.
                    let field_types = match tv.ty() {
                        Ty::Object(fields) => Some(fields),
                        _ => None,
                    };
                    obj.iter()
                        .map(|(k, v)| {
                            let field_ty = field_types
                                .and_then(|ft| ft.get(k))
                                .cloned()
                                .unwrap_or_else(Ty::error);
                            (*k, TypedValue::new(v.clone(), field_ty))
                        })
                        .collect()
                } else {
                    FxHashMap::default()
                };

                // Serialized node already in flight → park in serialized_queue
                if self.needs_serialized(dep_id) && lp.tracker.in_flight.contains(&dep_id) {
                    lp.tracker.serialized_queue.entry(dep_id).or_default().push_back(Parked {
                        task_id, coroutine,
                        request: PendingRequest::ExternCall(request),
                    });
                    return Ok(());
                }

                debug!(
                    context = %self.interner.resolve(name),
                    "spawning tool node via extern call"
                );
                let step = self.on_prepare(dep_id, dep_local, false, lp, state)?;
                self.execute(step, lp, state)?;
                lp.park_for_dep(dep_id, task_id, coroutine, PendingRequest::ExternCall(request));
                return Ok(());
            }
        }

        // 2. extern_handler
        let args = request.args().to_vec();
        debug!(context = %self.interner.resolve(name), "calling extern_handler");
        match (self.extern_handler)(name, args).await {
            Ok(value) => {
                request.resolve(value);
                lp.enqueue_step(task_id, coroutine);
            }
            Err(e) => {
                let node_id = match lp.meta.get(&task_id) {
                    Some(TaskMeta::Node { node_id, .. }) => *node_id,
                    Some(TaskMeta::Script { phase, .. }) => phase_node_id(phase),
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
                        .resolve(self.graph.node(node_id).name)
                        .to_string(),
                    error: e,
                });
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bind-scope locals: @self injection for bind nodes
    // -----------------------------------------------------------------------

    /// Prepare @self local for a bind node at spawn time.
    /// For Sequence mode: clones deque, checkpoints, stores origin in `lp.bind_origins`.
    fn prepare_bind_self<'j, E>(
        &self,
        bind_id: NodeId,
        local: &mut FxHashMap<Astr, TypedValue>,
        lp: &mut LoopState<'_>,
        state: &ResolveState<E>,
    )
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(bind_id);
        let interner = self.interner;

        // @self = previous stored value (or initial_value on first run)
        if let Some(prev) = self.load_self_value(bind_id, state) {
            let self_val = match &node.role {
                NodeRole::Persistent { mode: PersistMode::Sequence, .. } => {
                    let deque = extract_deque(&prev, "sequence mode @self");
                    let mut working = deque;
                    working.checkpoint();
                    let sc = acvus_interpreter::SequenceChain::new(working);
                    TypedValue::new(Value::sequence(sc), node.output_ty.clone())
                }
                _ => prev,
            };
            local.insert(interner.intern(KEY_SELF), self_val);
        } else if node.strategy.initial_value.is_some() {
            // initial_value exists but no stored value → populate_initial_values was not called or bug
            let name_str = self.interner.resolve(node.name);
            panic!(
                "node '{}' has initial_value but no stored value in prepare_bind_self — \
                 populate_initial_values must be called before resolve",
                name_str,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Dep wake + scheduling
    // -----------------------------------------------------------------------

    /// Wake dep waiters after a node completes. For serialized nodes,
    /// wakes one waiter at a time via re-execution to avoid lost updates.
    fn finish_dep_wake<'j, E>(
        &self,
        node_id: NodeId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        if self.needs_serialized(node_id) {
            if let Some(queue) = lp.tracker.serialized_queue.get_mut(&node_id)
                && let Some(parked) = queue.pop_front()
            {
                // 다음 waiter를 dep_waiters로 이동 → 재실행 결과를 받게 됨
                lp.tracker.dep_waiters.entry(node_id).or_default().push(parked);
                // 노드 재실행 (갱신된 @self from turn_context)
                let step = self.on_prepare(node_id, FxHashMap::default(), false, lp, state)?;
                self.execute(step, lp, state)?;
            } else {
                // 직렬화 큐 비었음 → 일반 wake
                lp.wake_waiters(node_id, value);
                self.try_eager_schedule(node_id, lp, state)?;
            }
        } else {
            lp.wake_waiters(node_id, value);
            self.try_eager_schedule(node_id, lp, state)?;
        }
        Ok(())
    }



    /// Try to load the existing @self value from turn_context (this turn's
    /// update) first, then storage (previous turn's value).
    fn load_self_value<'j, E>(&self, id: NodeId, state: &ResolveState<E>) -> Option<TypedValue>
    where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(id);
        let name_str = self.interner.resolve(node.name);
        state.load_self(node.name, name_str)
    }

    // -----------------------------------------------------------------------
    // persist_initial_value — store initial_value result via PatchDiff::set
    // -----------------------------------------------------------------------

    /// Persist an initial_value result to storage using `PatchDiff::set`.
    ///
    /// Uses a full `set` for all modes because:
    /// - This is a first write (no previous value to diff against)
    /// - `PatchDiff::set` stores the TypedValue as-is
    /// - `prepare_bind_locals` already handles List→Deque coercion
    fn persist_initial_value<'j, E>(
        &self,
        id: NodeId,
        value: &TypedValue,
        state: &mut ResolveState<E>,
    ) where
        E: EntryMut<'j>,
    {
        let node = self.graph.node(id);
        let name_str = self.interner.resolve(node.name);
        let ty = node.output_ty.clone();
        match &node.role {
            NodeRole::Persistent { mode: PersistMode::Sequence, .. } => {
                // Initial persist: extract deque and let storage handle it.
                let items = extract_deque(value, "Sequence persistency initial value");
                state.persist_sequence_diff(name_str, items, ty);
            }
            NodeRole::Persistent { mode: PersistMode::Patch, .. } => {
                state.persist_patch(name_str, PatchDiff::set(value.clone()), ty);
            }
            _ => {
                // Standalone with initial_value: persist as Patch
                state.persist_patch(name_str, PatchDiff::set(value.clone()), ty);
            }
        }
    }


    // -----------------------------------------------------------------------
    // Retry
    // -----------------------------------------------------------------------

    fn try_retry<'j, E>(
        &self,
        id: NodeId,
        node_name: &str,
        error: &RuntimeError,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<bool, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let Some((max_retries, attempt, local)) = lp.retry_state.get_mut(&id) else {
            return Ok(false);
        };
        if *attempt >= *max_retries {
            return Ok(false);
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
        let step = self.on_prepare(id, local_clone, true, lp, state)?;
        self.execute(step, lp, state)?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // External resolver
    // -----------------------------------------------------------------------

    async fn resolve_external<'j, E>(
        &self,
        name: Astr,
        state: &mut ResolveState<E>,
    ) -> Result<TypedValue, ResolveError>
    where
        E: EntryMut<'j>,
    {
        let name_str = self.interner.resolve(name);
        info!(name = %name_str, "calling external resolver");
        match (self.resolver)(name).await {
            Resolved::Once(value) => {
                debug!(name = %name_str, kind = "once", "external resolver returned");
                Ok(value)
            }
            Resolved::Turn(value) => {
                debug!(name = %name_str, kind = "turn", "external resolver returned");
                state.cache(name, value.clone());
                Ok(value)
            }
            Resolved::Persist(value) => {
                debug!(name = %name_str, kind = "persist", "external resolver returned");
                let ty = value.ty().clone();
                state.persist_patch(name_str, PatchDiff::set(value.clone()), ty);
                Ok(value)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Eager dependency scheduling
    // -----------------------------------------------------------------------

    fn eager_node_deps(&self, id: NodeId, entry: &dyn EntryRef<'_>) -> Vec<NodeId> {
        let node = self.graph.node(id);
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
                CompiledNodeKind::Expression(expr) => {
                    let p =
                        partition_context_keys(&expr.script.module, &known, &expr.script.val_def);
                    eager.extend(p.eager);
                }
                CompiledNodeKind::OpenAICompatible(_)
                | CompiledNodeKind::Anthropic(_)
                | CompiledNodeKind::GoogleAI(_)
                | CompiledNodeKind::GoogleAICache(_)
                | CompiledNodeKind::Iterator { .. } => {
                    // LLM/Cache/Iterator: no script-level context_keys to partition
                }
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
            CompiledExecution::Always | CompiledExecution::OncePerTurn => {}
        }
        if let NodeRole::Persistent { bind, .. } = &node.role {
            eager.extend(bind.context_keys.iter().copied());
        }

        eager
            .iter()
            .filter_map(|name| self.graph.name_to_primary.get(name).copied())
            .filter(|&dep_id| dep_id != id)
            .collect()
    }

    /// Always 전략 + initial_value를 가진 노드인지.
    /// 이 조건이면 동시 실행 시 @self lost update가 발생하므로 직렬화 필요.
    fn needs_serialized(&self, id: NodeId) -> bool {
        let node = self.graph.node(id);
        matches!(node.strategy.execution, CompiledExecution::Always)
            && node.strategy.initial_value.is_some()
    }

    fn needs_resolve_initial_value_only<'j, E>(&self, id: NodeId, state: &ResolveState<E>) -> bool
    where
        E: EntryMut<'j>,
    {
        let name = self.graph.node(id).name;
        if state.is_cached(&name) {
            return false;
        }
        let name_str = self.interner.resolve(name);
        if state.entry.get(name_str).is_some() {
            return false;
        }
        // Need to evaluate if this node has an initial_value
        node_initial_value(self.graph.node(id)).is_some()
    }

    fn needs_resolve<'j, E>(&self, id: NodeId, state: &ResolveState<E>, no_execute: bool) -> Result<bool, ResolveError>
    where
        E: EntryMut<'j>,
    {
        if no_execute {
            match &self.graph.node(id).role {
                NodeRole::Persistent { .. } => {
                    let name = self.graph.node(id).name;
                    let name_str = self.interner.resolve(name);
                    // Already resolved this turn or persisted in storage → no resolve needed.
                    if state.is_cached(&name) || state.entry.get(name_str).is_some() {
                        return Ok(false);
                    }
                    // Non-ephemeral, not in turn_context, not in storage → error.
                    return Err(ResolveError::UnresolvedContext(name_str.to_string()));
                }
                NodeRole::Standalone | NodeRole::Body { .. } => return Ok(true),
            }
        }
        let name = self.graph.node(id).name;
        Ok(match self.graph.node(id).strategy.execution {
            // Always: re-execute on every reference within a turn.
            CompiledExecution::Always => true,
            // OncePerTurn / IfModified: execute once per turn.
            // Only turn_context counts — storage values from previous turns
            // do NOT satisfy this. The node must re-run each turn.
            CompiledExecution::OncePerTurn | CompiledExecution::IfModified { .. } => !state.is_cached(&name),
        })
    }

    fn try_eager_schedule<'j, E>(
        &self,
        completed_id: NodeId,
        lp: &mut LoopState<'_>,
        state: &mut ResolveState<E>,
    ) -> Result<(), ResolveError>
    where
        E: EntryMut<'j>,
    {
        if completed_id.index() >= self.rdeps.len() {
            return Ok(());
        }
        for &candidate in &self.rdeps[completed_id.index()] {
            let candidate_needs = if lp.initial_value_only {
                self.needs_resolve_initial_value_only(candidate, state)
            } else {
                self.needs_resolve(candidate, state, lp.no_execute)?
            };
            if lp.tracker.in_flight.contains(&candidate) || !candidate_needs {
                continue;
            }
            let eager_deps = {
                let entry_ref = state.entry.as_ref();
                self.eager_node_deps(candidate, &entry_ref)
            };
            let all_deps_ready = if lp.initial_value_only {
                eager_deps.iter().all(|&dep| !self.needs_resolve_initial_value_only(dep, state))
            } else {
                // If any dep errors, treat as not ready (don't propagate error from eager check).
                eager_deps.iter().all(|&dep| self.needs_resolve(dep, state, lp.no_execute).unwrap_or(true) == false)
            };
            if all_deps_ready
            {
                debug!(
                    node = %self.interner.resolve(self.graph.node(candidate).name),
                    "eager scheduling"
                );
                let step = self.on_prepare(candidate, FxHashMap::default(), false, lp, state)?;
                self.execute(step, lp, state)?;
            }
        }
        Ok(())
    }

}

// ---------------------------------------------------------------------------
// Helpers: value → TrackedDeque extraction
// ---------------------------------------------------------------------------

/// Extract a `TrackedDeque<Value>` from a value that should be Deque, List, or Sequence.
///
/// Used at storage boundaries when converting runtime values to the tracked deque
/// representation for Sequence persistency mode.
fn extract_deque(value: &TypedValue, context: &str) -> TrackedDeque<Value> {
    match value.value() {
        Value::Lazy(LazyValue::Deque(d)) => d.clone(),
        Value::Lazy(LazyValue::Sequence(sc)) => sc.origin().clone(),
        other => panic!("{context}: expected Deque or Sequence, got {other:?}"),
    }
}

/// Extract the element type from a Sequence type.
fn sequence_elem_ty(ty: &Ty) -> &Ty {
    match ty {
        Ty::Sequence(elem, ..) => elem,
        other => unreachable!("expected Sequence type, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Whether a value should be unpacked by the streaming layer.
fn is_iterable(value: &TypedValue) -> bool {
    matches!(
        value.value(),
        Value::Lazy(LazyValue::Iterator(_))
            | Value::Lazy(LazyValue::Sequence(_))
            | Value::Lazy(LazyValue::List(_))
            | Value::Lazy(LazyValue::Deque(_))
    )
}

/// Create a coroutine that unpacks an iterable value, yielding items one by one.
/// For Iterator/Sequence: uses exec_next (lazy — one item per step).
/// For List/Deque: yields each element.
///
/// Element type is extracted from the value's Ty:
/// `Iterator(elem, _)` / `Sequence(elem, _, _)` / `List(elem)` / `Deque(elem, _)` → elem.
fn unpack_coroutine(value: TypedValue, interner: Interner) -> Coroutine<TypedValue, RuntimeError> {
    // Extract element type from the container's Ty.
    let elem_ty = match value.ty() {
        Ty::Iterator(elem, _) | Ty::Sequence(elem, ..) | Ty::List(elem) | Ty::Deque(elem, _) => {
            (**elem).clone()
        }
        _ => panic!("unpack_coroutine: expected iterable type (Iterator/Sequence/List/Deque), got {:?}", value.ty()),
    };

    acvus_utils::coroutine(move |handle| async move {
        match value.value() {
            Value::Lazy(LazyValue::Iterator(_)) | Value::Lazy(LazyValue::Sequence(_)) => {
                let effect = match value.ty() {
                    Ty::Iterator(_, e) | Ty::Sequence(_, _, e) => *e,
                    _ => acvus_mir::ty::Effect::Pure,
                };
                let ih = match value.into_inner() {
                    Value::Lazy(LazyValue::Iterator(ih)) => ih,
                    Value::Lazy(LazyValue::Sequence(sc)) => sc.into_iter_handle(effect),
                    _ => unreachable!(),
                };
                let mut interp = Interpreter::new(&interner, MirModule::default());
                let mut current = ih;
                loop {
                    let result;
                    (interp, result) =
                        Interpreter::exec_next(interp, current, &handle).await?;
                    match result {
                        Some((item, rest)) => {
                            current = rest;
                            handle
                                .yield_val(TypedValue::new(item, elem_ty.clone()))
                                .await;
                        }
                        None => break,
                    }
                }
            }
            Value::Lazy(LazyValue::List(_)) | Value::Lazy(LazyValue::Deque(_)) => {
                let items = match value.into_inner() {
                    Value::Lazy(LazyValue::List(items)) => items,
                    Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                    _ => unreachable!(),
                };
                for item in items {
                    handle
                        .yield_val(TypedValue::new(item, elem_ty.clone()))
                        .await;
                }
            }
            _ => unreachable!("is_iterable returned true but value is not iterable"),
        }
        Ok(())
    })
}

/// Coroutine that collects a Sequence(Pure) into a Deque value.
///
/// Uses `SequenceChain::collect` to preserve checksum lineage —
/// the result can be diffed against the origin via `TrackedDeque::into_diff`.
fn collect_sequence_coroutine(seq_value: TypedValue) -> Coroutine<TypedValue, RuntimeError> {
    acvus_utils::coroutine(move |handle| async move {
        let ty = seq_value.ty().clone();

        let sc = match seq_value.into_inner() {
            Value::Lazy(LazyValue::Sequence(sc)) => sc,
            other => panic!("collect_sequence_coroutine: expected Sequence, got {other:?}"),
        };

        let deque = sc.collect(&handle).await?;
        // Collected Deque — CollectSequence handler extracts the TrackedDeque
        // The result is passed to record_sequence_diff.
        // Use Deque type derived from the Sequence's element type + origin.
        let deque_ty = match &ty {
            Ty::Sequence(elem, origin, _effect) => Ty::Deque(elem.clone(), *origin),
            _ => panic!("collect_sequence_coroutine: expected Sequence ty, got {ty:?}"),
        };
        handle
            .yield_val(TypedValue::new(Value::deque(deque), deque_ty))
            .await;
        Ok(())
    })
}

fn phase_node_id(phase: &Phase) -> NodeId {
    match phase {
        Phase::IfModifiedKey { node_id }
        | Phase::InitialValue { node_id }
        | Phase::Assert { node_id, .. }
        | Phase::PersistSequence { node_id, .. } => *node_id,
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
