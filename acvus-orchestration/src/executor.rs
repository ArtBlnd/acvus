use std::collections::VecDeque;
use std::pin::Pin;

use acvus_interpreter::{RuntimeError, Stepped, TypedValue};
use acvus_mir::graph::ContextId;
use acvus_utils::{ContextRequest, Coroutine, ExternCallRequest};
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, warn};

use crate::{dsl::{Execution, Persistency}, lower::LowerResult};

// ExecutionState — bundled mutable context

pub struct ExecutionState<'a> {
    pub turn_context: FxHashMap<ContextId, TypedValue>,
    pub units: &'a LowerResult,
}

impl<'a> ExecutionState<'a> {
    pub fn new(units: &'a LowerResult) -> Self {
        Self {
            turn_context: FxHashMap::default(),
            units,
        }
    }

    pub fn cache(&mut self, id: ContextId, value: TypedValue) {
        self.turn_context.insert(id, value);
    }

    pub fn is_cached(&self, id: ContextId) -> bool {
        self.turn_context.contains_key(&id)
    }

    pub fn get_cached(&self, id: ContextId) -> Option<TypedValue> {
        self.turn_context.get(&id).cloned()
    }
}

// External resolver result

/// External context provider result.
pub enum Resolved {
    /// Valid for this request only. Not cached.
    Once(TypedValue),
    /// Valid for this turn. Cached in turn_context, discarded at turn end.
    Turn(TypedValue),
    /// Persistent. Stored in storage, survives across turns.
    Persist(TypedValue),
    /// No value found. Unit receives None via try_request_context.
    NotFound,
}

// Event loop types

struct StepResult {
    unit_id: ContextId,
    coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
    stepped: Stepped<TypedValue, RuntimeError, ContextId>,
}

enum PendingRequest {
    Context(ContextRequest<TypedValue, ContextId>),
    ExternCall(ExternCallRequest<TypedValue, ContextId>),
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
    unit_id: ContextId,
    coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
    request: PendingRequest,
}

// LoopState -- all mutable loop bookkeeping in one place

struct NodeTracker {
    in_flight: FxHashSet<ContextId>,
    dep_waiters: FxHashMap<ContextId, Vec<Parked>>,
    serialized_queue: FxHashMap<ContextId, VecDeque<Parked>>,
}

impl NodeTracker {
    fn take_waiters(&mut self, id: &ContextId) -> Vec<Parked> {
        self.dep_waiters.remove(id).unwrap_or_default()
    }
}

pub struct LoopState<'a> {
    futs: FuturesUnordered<Pin<Box<dyn Future<Output = StepResult> + Send + 'a>>>,
    tracker: NodeTracker,
    remaining_roots: FxHashSet<ContextId>,
    no_execute: bool,
}

impl<'a> LoopState<'a> {
    pub fn new(no_execute: bool) -> Self {
        Self {
            futs: FuturesUnordered::new(),
            tracker: NodeTracker {
                in_flight: FxHashSet::default(),
                dep_waiters: FxHashMap::default(),
                serialized_queue: FxHashMap::default(),
            },
            remaining_roots: FxHashSet::default(),
            no_execute,
        }
    }

    fn enqueue_step(&mut self, unit_id: ContextId, coroutine: Coroutine<TypedValue, RuntimeError, ContextId>) {
        self.futs.push(Box::pin(async move {
            let (coroutine, stepped) = coroutine.step().await;
            StepResult {
                unit_id,
                coroutine,
                stepped,
            }
        }));
    }

    fn wake_waiters(&mut self, id: ContextId, value: TypedValue) {
        for w in self.tracker.take_waiters(&id) {
            w.request.resolve(value.clone());
            self.enqueue_step(w.unit_id, w.coroutine);
        }
    }

    fn park_for_dep(
        &mut self,
        dep_id: ContextId,
        unit_id: ContextId,
        coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
        request: PendingRequest,
    ) {
        self.tracker
            .dep_waiters
            .entry(dep_id)
            .or_default()
            .push(Parked {
                unit_id,
                coroutine,
                request,
            });
    }
}

// Executor

/// Drives unit coroutines and handles NeedContext/NeedExternCall dependencies.
pub struct Executor<'a, R, EH> {
    pub units: &'a LowerResult,
    pub context_provider: &'a R,
    pub extern_handler: &'a EH,
}

impl<'a, R, EH> Executor<'a, R, EH>
where
    R: AsyncFn(ContextId) -> Resolved + Sync,
    EH: AsyncFn(ContextId, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync,
{
    pub async fn execute(
        &self,
        id: ContextId,
        state: &mut ExecutionState<'_>,
        no_execute: bool,
    ) -> Result<(), ExecutionError> {
        self.execute_all(vec![id], state, no_execute).await
    }

    pub async fn execute_all(
        &self,
        roots: Vec<ContextId>,
        state: &mut ExecutionState<'_>,
        no_execute: bool,
    ) -> Result<(), ExecutionError> {
        if roots.is_empty() {
            return Ok(());
        }

        let mut lp = LoopState::new(no_execute);

        for id in roots {
            lp.remaining_roots.insert(id);
            self.spawn(id, &mut lp);
        }

        while let Some(sr) = lp.futs.next().await {
            let StepResult {
                unit_id,
                coroutine,
                stepped,
            } = sr;

            match stepped {
                Stepped::Emit(value) => {
                    let meta = self.units.meta(unit_id).unwrap();
                    if matches!(meta.policy.execution, Execution::Always) {
                        lp.tracker.in_flight.remove(&unit_id);
                    }
                    self.propagate(unit_id, value, &mut lp, state)?;
                }
                Stepped::Done => {
                    warn!(unit = ?unit_id, "coroutine finished without emit");
                    self.propagate(unit_id, TypedValue::unit(), &mut lp, state)?;
                }
                Stepped::NeedContext(request) => {
                    self.handle_need_context(unit_id, coroutine, request, &mut lp, state)
                        .await?;
                }
                Stepped::NeedExternCall(request) => {
                    self.handle_need_extern_call(unit_id, coroutine, request, &mut lp, state)
                        .await?;
                }
                Stepped::StoreContext(id, value) => {
                    state.cache(id, value);
                    lp.enqueue_step(unit_id, coroutine);
                }
                Stepped::Error(e) => {
                    return Err(ExecutionError::Runtime {
                        unit: unit_id,
                        error: e,
                    });
                }
            }

            if lp.remaining_roots.is_empty() {
                return Ok(());
            }
        }

        Err(ExecutionError::deadlock(&lp, state))
    }

    fn spawn(&self, unit_id: ContextId, lp: &mut LoopState<'_>) {
        let coroutine = self.units.unit(unit_id).spawn(FxHashMap::default());
        lp.tracker.in_flight.insert(unit_id);
        lp.enqueue_step(unit_id, coroutine);
    }

    fn propagate(
        &self,
        unit_id: ContextId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
        state: &mut ExecutionState<'_>,
    ) -> Result<(), ExecutionError> {
        state.cache(unit_id, value.clone());
        lp.remaining_roots.remove(&unit_id);
        self.finish_dep_wake(unit_id, value, lp)
    }

    // NeedContext

    async fn handle_need_context(
        &self,
        unit_id: ContextId,
        coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
        request: ContextRequest<TypedValue, ContextId>,
        lp: &mut LoopState<'_>,
        state: &mut ExecutionState<'_>,
    ) -> Result<(), ExecutionError> {
        let id = request.key();

        // 1. Spawnable unit — needs_execute decides.
        if self.units.has(id) {
            if self.needs_execute(id, state, lp.no_execute)? {
                self.spawn_or_park(id, unit_id, coroutine, PendingRequest::Context(request), lp);
                return Ok(());
            }
            let val = state
                .get_cached(id)
                .ok_or(ExecutionError::UnresolvedContext { id })?;
            debug!(context = ?id, "resolved from cache");
            request.resolve(val);
            lp.enqueue_step(unit_id, coroutine);
            return Ok(());
        }

        // 2. Non-unit cached values (external contexts, storage values)
        if let Some(val) = state.get_cached(id) {
            debug!(context = ?id, "resolved from turn_context");
            request.resolve(val);
            lp.enqueue_step(unit_id, coroutine);
            return Ok(());
        }

        // 3. External context provider (lowest priority)
        debug!(context = ?id, "calling context_provider");
        match self.resolve_external(id, state).await {
            Some(value) => {
                request.resolve(value);
            }
            None => {
                debug!(context = ?id, "not found");
                request.resolve_not_found();
            }
        }
        lp.enqueue_step(unit_id, coroutine);
        Ok(())
    }

    // NeedExternCall

    async fn handle_need_extern_call(
        &self,
        unit_id: ContextId,
        coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
        request: ExternCallRequest<TypedValue, ContextId>,
        lp: &mut LoopState<'_>,
        _state: &mut ExecutionState<'_>,
    ) -> Result<(), ExecutionError> {
        let id = request.key();

        if self.units.has(id) {
            self.spawn_or_park(
                id,
                unit_id,
                coroutine,
                PendingRequest::ExternCall(request),
                lp,
            );
            return Ok(());
        }

        let args = request.args().to_vec();
        debug!(context = ?id, "calling extern_handler");
        let value =
            (self.extern_handler)(id, args)
                .await
                .map_err(|error| ExecutionError::Runtime {
                    unit: unit_id,
                    error,
                })?;
        request.resolve(value);
        lp.enqueue_step(unit_id, coroutine);
        Ok(())
    }

    // Spawn-or-park

    fn spawn_or_park(
        &self,
        dep_id: ContextId,
        unit_id: ContextId,
        coroutine: Coroutine<TypedValue, RuntimeError, ContextId>,
        request: PendingRequest,
        lp: &mut LoopState<'_>,
    ) {
        if self.needs_serialized(dep_id) && lp.tracker.in_flight.contains(&dep_id) {
            lp.tracker
                .serialized_queue
                .entry(dep_id)
                .or_default()
                .push_back(Parked {
                    unit_id,
                    coroutine,
                    request,
                });
            return;
        }
        if !lp.tracker.in_flight.contains(&dep_id) {
            self.spawn(dep_id, lp);
        }
        lp.park_for_dep(dep_id, unit_id, coroutine, request);
    }

    // Dep wake + scheduling

    fn finish_dep_wake(
        &self,
        unit_id: ContextId,
        value: TypedValue,
        lp: &mut LoopState<'_>,
    ) -> Result<(), ExecutionError> {
        if self.needs_serialized(unit_id) {
            if let Some(queue) = lp.tracker.serialized_queue.get_mut(&unit_id)
                && let Some(parked) = queue.pop_front()
            {
                lp.tracker
                    .dep_waiters
                    .entry(unit_id)
                    .or_default()
                    .push(parked);
                self.spawn(unit_id, lp);
                return Ok(());
            }
        }

        lp.wake_waiters(unit_id, value);
        Ok(())
    }

    // External context provider

    async fn resolve_external(
        &self,
        id: ContextId,
        state: &mut ExecutionState<'_>,
    ) -> Option<TypedValue> {
        let resolved = (self.context_provider)(id).await;
        match resolved {
            Resolved::Once(value) => Some(value),
            Resolved::Turn(value) | Resolved::Persist(value) => {
                state.cache(id, value.clone());
                Some(value)
            }
            Resolved::NotFound => None,
        }
    }

    fn needs_serialized(&self, id: ContextId) -> bool {
        self.units
            .meta(id)
            .expect("needs_serialized: ContextId must have meta")
            .serialized()
    }

    fn needs_execute(
        &self,
        id: ContextId,
        state: &ExecutionState<'_>,
        no_execute: bool,
    ) -> Result<bool, ExecutionError> {
        let meta = self.units.meta(id).unwrap();
        if no_execute {
            match &meta.policy.persistency {
                Persistency::Sequence { .. } | Persistency::Patch { .. } => {
                    if state.get_cached(id).is_some() {
                        return Ok(false);
                    }
                    return Err(ExecutionError::UnresolvedContext { id });
                }
                Persistency::Ephemeral => return Ok(true),
            }
        }
        Ok(match meta.policy.execution {
            Execution::Always => true,
            Execution::OncePerTurn => !state.is_cached(id),
        })
    }
}

// Error

#[derive(Debug)]
pub struct ParkedDiag {
    pub task: ContextId,
    pub waiting_for: ContextId,
}

#[derive(Debug)]
pub enum ExecutionError {
    UnresolvedContext { id: ContextId },
    Deadlock {
        stuck_roots: Vec<ContextId>,
        dep_waiters: Vec<ContextId>,
        serialized_queue: Vec<ContextId>,
        turn_context_keys: Vec<ContextId>,
        in_flight: Vec<ContextId>,
        parked: Vec<ParkedDiag>,
    },
    Runtime { unit: ContextId, error: RuntimeError },
}

impl ExecutionError {
    fn deadlock(lp: &LoopState<'_>, state: &ExecutionState<'_>) -> Self {
        let stuck_roots: Vec<ContextId> = lp.remaining_roots.iter().copied().collect();
        let dep_waiters: Vec<ContextId> = lp.tracker.dep_waiters.keys().copied().collect();
        let serialized_queue: Vec<ContextId> = lp.tracker.serialized_queue.keys().copied().collect();
        let in_flight: Vec<ContextId> = lp.tracker.in_flight.iter().copied().collect();
        let turn_context_keys: Vec<ContextId> = state.turn_context.keys().copied().collect();
        let mut parked = Vec::new();
        for (&id, waiters) in &lp.tracker.dep_waiters {
            for w in waiters {
                parked.push(ParkedDiag {
                    task: w.unit_id,
                    waiting_for: id,
                });
            }
        }
        for (&id, queue) in &lp.tracker.serialized_queue {
            for w in queue {
                parked.push(ParkedDiag {
                    task: w.unit_id,
                    waiting_for: id,
                });
            }
        }
        ExecutionError::Deadlock {
            stuck_roots,
            dep_waiters,
            serialized_queue,
            turn_context_keys,
            in_flight,
            parked,
        }
    }
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::UnresolvedContext { id } => write!(f, "unresolved context: {id:?}"),
            ExecutionError::Deadlock {
                stuck_roots,
                dep_waiters,
                serialized_queue,
                turn_context_keys,
                in_flight,
                parked,
            } => {
                write!(f, "deadlock: roots {stuck_roots:?}")?;
                if !dep_waiters.is_empty() {
                    write!(f, ", dep_waiters {dep_waiters:?}")?;
                }
                if !serialized_queue.is_empty() {
                    write!(f, ", serial_q {serialized_queue:?}")?;
                }
                if !in_flight.is_empty() {
                    write!(f, ", in_flight {in_flight:?}")?;
                }
                write!(f, ", turn_ctx {turn_context_keys:?}")?;
                for p in parked {
                    write!(f, "; {:?} -> {:?}", p.task, p.waiting_for)?;
                }
                Ok(())
            }
            ExecutionError::Runtime { unit, error } => {
                write!(f, "runtime error in {unit:?}: {error}")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}
