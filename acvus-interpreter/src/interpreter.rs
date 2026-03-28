use std::sync::Arc;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::graph::FunctionId;
use acvus_mir::ir::{Callee, CastKind, Inst, InstKind, Label, MirBody, MirModule, ValueId};
use acvus_mir::ty::{Effect, Ty};
use acvus_utils::{Astr, Freeze, Interner, LocalFactory, LocalVec, TrackedDeque};
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stackfuture::StackFuture;

use crate::error::RuntimeError;
use crate::value::{FnValue, Value};

// ── Builtin dispatch ─────────────────────────────────────────────────

/// Args passed to builtin functions. Stack-allocated for ≤4 args.
pub type Args = SmallVec<[Value; 4]>;

/// Max size for stack-allocated async futures.
pub const ASYNC_FUTURE_SIZE: usize = 1024;

/// Sync builtin — no future overhead.
pub type SyncBuiltinFn = fn(Args, &Interner) -> Result<Value, RuntimeError>;

/// Async builtin — stack-allocated future, receives &mut Interpreter.
pub type AsyncBuiltinFn =
    for<'x> fn(
        Args,
        &'x mut Interpreter,
    ) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE>;

/// Builtin handler — sync or async.
pub enum BuiltinHandler {
    Sync(SyncBuiltinFn),
    Async(AsyncBuiltinFn),
}

// ── Frame ────────────────────────────────────────────────────────────

/// Register file. Stores one `Value` per SSA ValueId.
/// No Arc wrapping — values live directly in the frame.
struct Frame {
    regs: LocalVec<ValueId, Value>,
    label_map: FxHashMap<Label, usize>,
}

impl Frame {
    fn new(val_factory: &LocalFactory<ValueId>, label_map: FxHashMap<Label, usize>) -> Self {
        Self {
            regs: val_factory.build_vec(|| Value::Empty),
            label_map,
        }
    }

    /// Write a value into a register.
    #[inline]
    fn set(&mut self, id: ValueId, value: Value) {
        self.regs[id] = value;
    }

    /// Borrow a register. Panics on Empty (moved-out).
    #[inline]
    fn get(&self, id: ValueId) -> &Value {
        let v = &self.regs[id];
        assert!(!v.is_empty(), "get: register {id:?} already moved");
        v
    }

    /// Move a value out, leaving Empty behind. For move-only values.
    #[inline]
    fn take(&mut self, id: ValueId) -> Value {
        let v = self.regs[id].take();
        assert!(!v.is_empty(), "take: register {id:?} already moved");
        v
    }

    /// Share a value (clone for inline/Arc, panic for move-only).
    #[inline]
    fn share(&self, id: ValueId) -> Value {
        self.get(id).share()
    }

    /// Move-aware value extraction: take if move-only type, share otherwise.
    /// Soundness: move_check guarantees move-only values are used at most once.
    #[inline]
    fn use_val(&mut self, id: ValueId, val_types: &FxHashMap<ValueId, Ty>) -> Value {
        if let Some(ty) = val_types.get(&id)
            && acvus_mir::validate::move_check::is_move_only(ty) == Some(true)
        {
            return self.take(id);
        }
        self.share(id)
    }

    // ── Control flow ─────────────────────────────────────────────

    fn jump(&mut self, insts: &[Inst], label: &Label, args: &[ValueId]) -> usize {
        let target = self.resolve_label(label);
        self.bind_block_params(insts, target, args);
        target
    }

    fn jump_if(
        &mut self,
        insts: &[Inst],
        cond: ValueId,
        then: (&Label, &[ValueId]),
        else_: (&Label, &[ValueId]),
    ) -> usize {
        let cond_val = match self.get(cond) {
            Value::Bool(b) => *b,
            other => panic!("jump_if: expected Bool, got {other:?}"),
        };
        let (label, args) = if cond_val { then } else { else_ };
        self.jump(insts, label, args)
    }

    fn resolve_label(&self, label: &Label) -> usize {
        *self
            .label_map
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}"))
    }

    fn bind_block_params(&mut self, insts: &[Inst], target: usize, args: &[ValueId]) {
        if let InstKind::BlockLabel { params, .. } = &insts[target].kind {
            // Collect values first to avoid borrow issues.
            let values: Vec<Value> = args.iter().map(|a| self.share(*a)).collect();
            for (param, val) in params.iter().zip(values) {
                self.set(*param, val);
            }
        }
    }
}

// ── Label map ────────────────────────────────────────────────────────

fn build_label_map(body: &MirBody) -> FxHashMap<Label, usize> {
    build_label_map_from_insts(&body.insts)
}

fn build_label_map_from_insts(insts: &[Inst]) -> FxHashMap<Label, usize> {
    insts
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some((*label, i)),
            _ => None,
        })
        .collect()
}

/// Control flow after executing one instruction.
enum Flow {
    Next,
    Jump(usize),
    Return(Value),
}

/// Result of applying ops pipeline to a single element.
enum ApplyResult {
    Emit(Value),
    Skip,
    Expand(Vec<Value>),
}

use crate::journal::{ContextOverlay, ContextWrite, EntryMut, EntryRef};
use acvus_mir::graph::QualifiedRef;

// ── Interpreter ──────────────────────────────────────────────────────

/// Result of execution — return value + context mutations.
pub struct ExecResult {
    pub value: Value,
    /// Legacy: context writes from Module/overlay path.
    pub writes: Vec<ContextWrite>,
    /// ExternFn: context defs as raw Values (ordered by context_defs declaration).
    pub defs: Vec<Value>,
}

/// A single executable unit — MIR module, builtin handler, or extern function.
pub enum Executable {
    Module(MirModule),
    Builtin(BuiltinHandler),
    /// External function with type-safe handler (uses/defs aware).
    Extern(crate::extern_fn::ExternHandler),
}

impl Executable {
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Module(_) => "Module",
            Self::Builtin(_) => "Builtin",
            Self::Extern(_) => "Extern",
        }
    }
}

/// Readonly shared state — clone is cheap (Freeze/Arc internally).
#[derive(Clone)]
pub struct InterpreterContext {
    pub interner: Interner,
    pub functions: Freeze<FxHashMap<FunctionId, Executable>>,
    pub fn_types: Freeze<FxHashMap<FunctionId, Ty>>,
    pub context_names: Freeze<FxHashMap<QualifiedRef, Astr>>,
    pub executor: Arc<dyn crate::executor::Executor>,
}

impl InterpreterContext {
    pub fn new(
        interner: &Interner,
        functions: FxHashMap<FunctionId, Executable>,
        executor: Arc<dyn crate::executor::Executor>,
    ) -> Self {
        Self {
            interner: interner.clone(),
            functions: Freeze::new(functions),
            fn_types: Freeze::new(FxHashMap::default()),
            context_names: Freeze::new(FxHashMap::default()),
            executor,
        }
    }

    pub fn with_fn_types(mut self, fn_types: FxHashMap<FunctionId, Ty>) -> Self {
        self.fn_types = Freeze::new(fn_types);
        self
    }

    pub fn with_context_names(mut self, context_names: FxHashMap<QualifiedRef, Astr>) -> Self {
        self.context_names = Freeze::new(context_names);
        self
    }
}

/// Per-execution mutable state.
pub struct Interpreter {
    shared: InterpreterContext,
    entry: FunctionId,
    overlay: ContextOverlay,
    variables: FxHashMap<Astr, Value>,
    /// Arguments passed to this interpreter via Spawn.
    /// Bound to MirBody.param_regs when execute_function runs.
    spawn_args: Vec<Value>,
}

impl Interpreter {
    pub fn new(shared: InterpreterContext, entry: FunctionId, overlay: ContextOverlay) -> Self {
        Self {
            shared,
            entry,
            overlay,
            variables: FxHashMap::default(),
            spawn_args: Vec::new(),
        }
    }

    /// Resolve context name for a QualifiedRef. Returns owned String to avoid borrow conflicts.
    fn resolve_context_key(&self, qref: &QualifiedRef) -> Result<String, RuntimeError> {
        let name = self
            .shared
            .context_names
            .get(qref)
            .ok_or_else(|| RuntimeError::internal(format!("no context name for {qref:?}")))?;
        Ok(self.shared.interner.resolve(*name).to_string())
    }

    /// Collect context uses: SSA hint (from frame) or fallback (from overlay via type).
    fn collect_context_uses(
        &self,
        context_uses: &[(QualifiedRef, ValueId)],
        fn_id: &FunctionId,
        frame: &Frame,
    ) -> Result<Vec<Value>, RuntimeError> {
        if !context_uses.is_empty() {
            Ok(context_uses
                .iter()
                .map(|(_, vid)| frame.share(*vid))
                .collect())
        } else {
            self.collect_uses_from_type(fn_id)
        }
    }

    /// Fallback: collect context uses from function type (overlay path, no SSA hint).
    fn collect_uses_from_type(&self, fn_id: &FunctionId) -> Result<Vec<Value>, RuntimeError> {
        let Some(fn_ty) = self.shared.fn_types.get(fn_id) else {
            return Ok(vec![]);
        };
        let Ty::Fn {
            effect: Effect::Resolved(eff),
            ..
        } = fn_ty
        else {
            return Ok(vec![]);
        };
        eff.reads
            .iter()
            .map(|qref| {
                let key = self.resolve_context_key(qref)?;
                self.overlay
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| RuntimeError::internal(format!("undefined context '{key}'")))
            })
            .collect()
    }

    /// Apply context defs: SSA hint (frame + overlay) or fallback (overlay only via type).
    fn apply_context_defs(
        &mut self,
        context_defs: &[(QualifiedRef, ValueId)],
        fn_id: &FunctionId,
        defs: Vec<Value>,
        frame: &mut Frame,
        projection_map: &mut FxHashMap<ValueId, QualifiedRef>,
    ) -> Result<(), RuntimeError> {
        if !context_defs.is_empty() {
            for ((ctx_id, vid), def_value) in context_defs.iter().zip(defs) {
                let key = self.resolve_context_key(ctx_id)?;
                let for_overlay = def_value.share();
                frame.set(*vid, def_value);
                self.overlay.apply_field(&key, &[], for_overlay);
                projection_map.insert(*vid, *ctx_id);
            }
        } else {
            self.apply_defs_from_type(fn_id, defs)?;
        }
        Ok(())
    }

    /// Fallback: apply context defs from function type (overlay path, no SSA hint).
    fn apply_defs_from_type(
        &mut self,
        fn_id: &FunctionId,
        defs: Vec<Value>,
    ) -> Result<(), RuntimeError> {
        let Some(fn_ty) = self.shared.fn_types.get(fn_id) else {
            return Ok(());
        };
        let Ty::Fn {
            effect: Effect::Resolved(eff),
            ..
        } = fn_ty
        else {
            return Ok(());
        };
        for (qref, def_value) in eff.writes.iter().zip(defs) {
            let key = self.resolve_context_key(qref)?;
            self.overlay.apply_field(&key, &[], def_value);
        }
        Ok(())
    }

    /// Fork for spawn — shared state is cheap clone, overlay forks, args carried.
    pub fn fork(&self, entry: FunctionId, args: Vec<Value>) -> Self {
        Self {
            shared: self.shared.clone(),
            entry,
            overlay: self.overlay.spawn_fork(),
            variables: FxHashMap::default(),
            spawn_args: args,
        }
    }

    fn function(&self, id: &FunctionId) -> &Executable {
        self.shared
            .functions
            .get(id)
            .unwrap_or_else(|| panic!("no function for #{}", id.index()))
    }

    fn module(&self, id: &FunctionId) -> &MirModule {
        match self.function(id) {
            Executable::Module(m) => m,
            other => panic!(
                "expected Module for #{}, got {}",
                id.index(),
                other.variant_name()
            ),
        }
    }

    /// Execute the entry module. Returns value + accumulated context writes.
    pub async fn execute(&mut self) -> Result<ExecResult, RuntimeError> {
        let entry = self.entry;
        let args = std::mem::take(&mut self.spawn_args);
        let value = self.execute_function(&entry, &args).await?;
        let overlay = std::mem::replace(
            &mut self.overlay,
            ContextOverlay::new(
                Arc::new(std::collections::HashMap::new()),
                self.shared.interner.clone(),
            ),
        );
        let writes = overlay.into_patches();
        Ok(ExecResult {
            value,
            writes,
            defs: Vec::new(),
        })
    }

    /// Execute a specific function by FunctionId with explicit args.
    async fn execute_function(
        &mut self,
        id: &FunctionId,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        let m = self.module(id);
        let insts: Arc<[Inst]> = m.main.insts.clone().into();
        let closures = m.closures.clone();
        let param_regs = m.main.param_regs.clone();
        let val_types = m.main.val_types.clone();
        let label_map = build_label_map(&m.main);
        let mut frame = Frame::new(&m.main.val_factory, label_map);
        // Bind args to param_regs.
        for (reg, val) in param_regs.iter().zip(args.iter()) {
            frame.set(*reg, val.clone());
        }
        let mut projection_map: FxHashMap<ValueId, QualifiedRef> = FxHashMap::default();
        self.run_loop(
            &insts,
            &closures,
            &mut frame,
            &mut projection_map,
            &val_types,
        )
        .await
    }

    /// Shared execution loop — used by both execute and closure calls.
    /// BoxFuture wrapper for async recursion (closure → run_loop → closure).
    fn run_loop<'s>(
        &'s mut self,
        insts: &'s [Inst],
        closures: &'s FxHashMap<Label, MirBody>,
        frame: &'s mut Frame,
        projection_map: &'s mut FxHashMap<ValueId, QualifiedRef>,
        val_types: &'s FxHashMap<ValueId, Ty>,
    ) -> BoxFuture<'s, Result<Value, RuntimeError>> {
        Box::pin(self.run_loop_inner(insts, closures, frame, projection_map, val_types))
    }

    async fn run_loop_inner(
        &mut self,
        insts: &[Inst],
        closures: &FxHashMap<Label, MirBody>,
        frame: &mut Frame,
        projection_map: &mut FxHashMap<ValueId, QualifiedRef>,
        val_types: &FxHashMap<ValueId, Ty>,
    ) -> Result<Value, RuntimeError> {
        let mut pc = 0;
        while pc < insts.len() {
            match self
                .execute_inst(insts, closures, pc, frame, projection_map, val_types)
                .await?
            {
                Flow::Next => pc += 1,
                Flow::Jump(target) => pc = target,
                Flow::Return(val) => return Ok(val),
            }
        }
        Ok(Value::Unit)
    }

    /// Execute a single instruction. Returns control flow directive.
    async fn execute_inst(
        &mut self,
        insts: &[Inst],
        closures: &FxHashMap<Label, MirBody>,
        pc: usize,
        frame: &mut Frame,
        projection_map: &mut FxHashMap<ValueId, QualifiedRef>,
        val_types: &FxHashMap<ValueId, Ty>,
    ) -> Result<Flow, RuntimeError> {
        match &insts[pc].kind {
            // ── Constants ────────────────────────────────────
            InstKind::Const { dst, value } => {
                frame.set(*dst, literal_to_value(value));
            }

            // ── Variables ────────────────────────────────────
            InstKind::VarLoad { dst, name } => {
                let val = self
                    .variables
                    .get(name)
                    .unwrap_or_else(|| {
                        panic!(
                            "undefined variable ${}",
                            self.shared.interner.resolve(*name)
                        )
                    })
                    .share();
                frame.set(*dst, val);
            }
            InstKind::VarStore { name, src } => {
                let val = frame.share(*src);
                self.variables.insert(*name, val);
            }
            InstKind::ParamLoad { dst, name } => {
                let val = self
                    .variables
                    .get(name)
                    .unwrap_or_else(|| {
                        panic!("undefined param ${}", self.shared.interner.resolve(*name))
                    })
                    .share();
                frame.set(*dst, val);
            }

            // ── Context ───────────────────────────────────────
            InstKind::ContextProject { dst, ctx, .. } => {
                projection_map.insert(*dst, *ctx);
                frame.set(*dst, Value::Unit);
            }
            InstKind::ContextLoad { dst, src } => {
                let ctx_id = projection_map[src];
                let name = self
                    .shared
                    .context_names
                    .get(&ctx_id)
                    .unwrap_or_else(|| panic!("context load: no name for {:?}", ctx_id));
                let key = self.shared.interner.resolve(*name);
                let val = self
                    .overlay
                    .get(key)
                    .unwrap_or_else(|| panic!("context load: undefined context '{}'", key))
                    .clone();
                frame.set(*dst, val);
            }
            InstKind::ContextStore { dst, value } => {
                let ctx_id = projection_map[dst];
                let name = self
                    .shared
                    .context_names
                    .get(&ctx_id)
                    .unwrap_or_else(|| panic!("context store: no name for {:?}", ctx_id));
                let key = self.shared.interner.resolve(*name);
                let val = frame.share(*value);
                self.overlay.apply_field(key, &[], val);
            }

            // ── Arithmetic / Logic ───────────────────────────
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let result = eval_binop(*op, frame.get(*left), frame.get(*right))?;
                frame.set(*dst, result);
            }
            InstKind::UnaryOp { dst, op, operand } => {
                let result = eval_unaryop(*op, frame.get(*operand))?;
                frame.set(*dst, result);
            }

            // ── Field / Index access ─────────────────────────
            InstKind::FieldGet { dst, object, field } => {
                let val = match frame.get(*object) {
                    Value::Object(obj) => obj
                        .get(field)
                        .unwrap_or_else(|| {
                            panic!("missing field {}", self.shared.interner.resolve(*field))
                        })
                        .share(),
                    other => panic!("FieldGet on non-object: {other:?}"),
                };
                frame.set(*dst, val);
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                let val = match frame.get(*tuple) {
                    Value::Tuple(t) => t[*index].share(),
                    other => panic!("TupleIndex on non-tuple: {other:?}"),
                };
                frame.set(*dst, val);
            }

            // ── Constructors ─────────────────────────────────
            InstKind::MakeDeque { dst, elements } => {
                let items: Vec<Value> = elements.iter().map(|e| frame.share(*e)).collect();
                frame.set(*dst, Value::deque(TrackedDeque::from_vec(items)));
            }
            InstKind::MakeObject { dst, fields } => {
                let obj: FxHashMap<Astr, Value> =
                    fields.iter().map(|(k, v)| (*k, frame.share(*v))).collect();
                frame.set(*dst, Value::object(obj));
            }
            InstKind::MakeRange {
                dst,
                start,
                end,
                kind,
            } => {
                let s = frame.get(*start).as_int();
                let e = frame.get(*end).as_int();
                frame.set(
                    *dst,
                    Value::range(s, e, matches!(kind, RangeKind::InclusiveEnd)),
                );
            }
            InstKind::MakeTuple { dst, elements } => {
                let items: Vec<Value> = elements.iter().map(|e| frame.share(*e)).collect();
                frame.set(*dst, Value::tuple(items));
            }
            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => {
                let captured: Vec<Value> = captures.iter().map(|c| frame.share(*c)).collect();
                let closure_body = closures
                    .get(body)
                    .unwrap_or_else(|| panic!("closure body not found: {body:?}"));
                frame.set(
                    *dst,
                    Value::closure(FnValue {
                        body: Arc::new(closure_body.clone()),
                        captures: captured.into(),
                    }),
                );
            }

            // ── Variant ──────────────────────────────────────
            InstKind::MakeVariant { dst, tag, payload } => {
                let p = payload.map(|v| frame.share(v));
                frame.set(*dst, Value::variant(*tag, p));
            }
            InstKind::TestVariant { dst, src, tag } => {
                let matches = match frame.get(*src) {
                    Value::Variant { tag: t, .. } => t == tag,
                    _ => false,
                };
                frame.set(*dst, Value::bool_(matches));
            }
            InstKind::UnwrapVariant { dst, src } => {
                let val = match frame.take(*src) {
                    Value::Variant {
                        payload: Some(p), ..
                    } => Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share()),
                    Value::Variant { payload: None, .. } => Value::Unit,
                    other => panic!("UnwrapVariant on non-variant: {other:?}"),
                };
                frame.set(*dst, val);
            }

            // ── Pattern testing ──────────────────────────────
            InstKind::TestLiteral { dst, src, value } => {
                let matches = match (frame.get(*src), value) {
                    (Value::Int(a), Literal::Int(b)) => *a == *b,
                    (Value::Float(a), Literal::Float(b)) => *a == *b,
                    (Value::Bool(a), Literal::Bool(b)) => *a == *b,
                    (Value::String(a), Literal::String(b)) => a.as_ref() == b.as_str(),
                    _ => false,
                };
                frame.set(*dst, Value::bool_(matches));
            }
            InstKind::TestListLen {
                dst,
                src,
                min_len,
                exact,
            } => {
                let len = match frame.get(*src) {
                    Value::List(l) => l.len(),
                    Value::Deque(d) => d.len(),
                    _ => 0,
                };
                let matches = if *exact {
                    len == *min_len
                } else {
                    len >= *min_len
                };
                frame.set(*dst, Value::bool_(matches));
            }
            InstKind::TestObjectKey { dst, src, key } => {
                let has = match frame.get(*src) {
                    Value::Object(o) => o.contains_key(key),
                    _ => false,
                };
                frame.set(*dst, Value::bool_(has));
            }
            InstKind::TestRange {
                dst,
                src,
                start,
                end,
                kind,
            } => {
                let inclusive = matches!(kind, RangeKind::InclusiveEnd);
                let in_range = match frame.get(*src) {
                    Value::Int(n) => *n >= *start && if inclusive { *n <= *end } else { *n < *end },
                    _ => false,
                };
                frame.set(*dst, Value::bool_(in_range));
            }

            // ── Cast ─────────────────────────────────────────
            InstKind::Cast { dst, src, kind } => {
                let val = eval_cast(*kind, frame.share(*src));
                frame.set(*dst, val);
            }

            // ── Iterator ─────────────────────────────────────
            InstKind::IterStep {
                dst,
                iter_src,
                iter_dst,
                done,
                done_args,
            } => {
                let mut iter = frame.take(*iter_src).into_iterator();
                match self.exec_next(&mut iter).await? {
                    Some(val) => {
                        frame.set(*dst, val);
                        frame.set(*iter_dst, Value::iterator(*iter));
                    }
                    None => {
                        let target = frame.jump(insts, done, done_args);
                        return Ok(Flow::Jump(target));
                    }
                }
            }

            // ── Control flow ─────────────────────────────────
            InstKind::BlockLabel { .. } => {}
            InstKind::Jump { label, args } => {
                let target = frame.jump(insts, label, args);
                return Ok(Flow::Jump(target));
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let target = frame.jump_if(
                    insts,
                    *cond,
                    (then_label, then_args),
                    (else_label, else_args),
                );
                return Ok(Flow::Jump(target));
            }
            InstKind::Return(val) => {
                return Ok(Flow::Return(frame.take(*val)));
            }
            InstKind::Nop => {}
            InstKind::Poison { .. } => {
                panic!("reached poison instruction");
            }

            // ── Functions ─────────────────────────────────────
            InstKind::LoadFunction { dst, id } => {
                frame.set(*dst, Value::Int(id.index() as i64));
            }
            InstKind::FunctionCall {
                dst,
                callee,
                args,
                context_uses,
                context_defs,
            } => {
                let result = match callee {
                    Callee::Direct(id) => {
                        let is_extern = matches!(self.function(id), Executable::Extern(_));
                        if is_extern {
                            let handler = match self.function(id) {
                                Executable::Extern(h) => h.clone(),
                                _ => unreachable!(),
                            };
                            let arg_vals: Vec<Value> =
                                args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                            let uses = self.collect_context_uses(context_uses, id, &frame)?;

                            let output = match &handler {
                                crate::extern_fn::ExternHandler::Sync(f) => {
                                    f(arg_vals, uses, &self.shared.interner)?
                                }
                                crate::extern_fn::ExternHandler::Async(f) => {
                                    let interner = self.shared.interner.clone();
                                    f(arg_vals, uses, interner).await?
                                }
                            };

                            self.apply_context_defs(
                                context_defs,
                                id,
                                output.defs,
                                frame,
                                projection_map,
                            )?;

                            output.rets.into_iter().next().unwrap_or(Value::Unit)
                        } else {
                            let arg_vals: Args =
                                args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                            self.dispatch_call(id, arg_vals).await?
                        }
                    }
                    Callee::Indirect(val_id) => {
                        let fv = frame.take(*val_id).into_fn();
                        match args.len() {
                            1 => {
                                self.call_closure(&fv, frame.use_val(args[0], val_types))
                                    .await?
                            }
                            2 => {
                                self.call_closure_2(
                                    &fv,
                                    frame.use_val(args[0], val_types),
                                    frame.use_val(args[1], val_types),
                                )
                                .await?
                            }
                            _ => {
                                let arg = if args.is_empty() {
                                    Value::Unit
                                } else {
                                    frame.use_val(args[0], val_types)
                                };
                                self.call_closure(&fv, arg).await?
                            }
                        }
                    }
                };
                frame.set(*dst, result);
            }
            InstKind::Spawn {
                dst,
                callee,
                args,
                context_uses,
            } => {
                let callee_id = match callee {
                    Callee::Direct(id) => *id,
                    Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
                };
                let is_extern = matches!(self.function(&callee_id), Executable::Extern(_));
                let handle = if is_extern {
                    let spawn_args: Vec<Value> = args.iter().map(|a| frame.share(*a)).collect();
                    let uses = self.collect_context_uses(context_uses, &callee_id, &frame)?;
                    let handler = match self.function(&callee_id) {
                        Executable::Extern(h) => h.clone(),
                        _ => unreachable!(),
                    };
                    let interner = self.shared.interner.clone();
                    // Spawn is pure — handler executes at eval time.
                    match &handler {
                        crate::extern_fn::ExternHandler::Sync(f) => {
                            let f = Arc::clone(f);
                            self.shared.executor.spawn_blocking(Box::new(move || {
                                let output = f(spawn_args, uses, &interner)?;
                                Ok(ExecResult {
                                    value: output.rets.into_iter().next().unwrap_or(Value::Unit),
                                    writes: Vec::new(),
                                    defs: output.defs,
                                })
                            }))
                        }
                        crate::extern_fn::ExternHandler::Async(f) => {
                            let f = Arc::clone(f);
                            self.shared.executor.spawn_async(Box::pin(async move {
                                let output = f(spawn_args, uses, interner).await?;
                                Ok(ExecResult {
                                    value: output.rets.into_iter().next().unwrap_or(Value::Unit),
                                    writes: Vec::new(),
                                    defs: output.defs,
                                })
                            }))
                        }
                    }
                } else {
                    // Existing path: fork interpreter for Module/Builtin.
                    let spawn_args: Vec<Value> = args.iter().map(|a| frame.share(*a)).collect();
                    let child = self.fork(callee_id, spawn_args);
                    self.shared.executor.spawn_interpreter(child)
                };
                frame.set(*dst, Value::Handle(Box::new(handle)));
            }
            InstKind::Eval {
                dst,
                src,
                context_defs,
            } => {
                let handle = match frame.take(*src) {
                    Value::Handle(h) => *h,
                    other => panic!("eval: expected Handle, got {other:?}"),
                };
                let result = self.shared.executor.eval(handle).await?;

                // Legacy Module path: merge overlay patches.
                if !result.writes.is_empty() {
                    self.overlay.merge_patches(result.writes);
                }

                // ExternFn path: apply defs to overlay via context_defs mapping.
                let defs = result.defs;
                let defs_count = defs.len();
                for ((ctx_id, vid), def_value) in context_defs.iter().zip(defs) {
                    let key = self.resolve_context_key(ctx_id)?;
                    let for_overlay = def_value.share();
                    frame.set(*vid, def_value);
                    self.overlay.apply_field(&key, &[], for_overlay);
                    projection_map.insert(*vid, *ctx_id);
                }

                // Module path: register remaining SSA names not covered by defs.
                for (ctx_id, vid) in context_defs.iter().skip(defs_count) {
                    projection_map.insert(*vid, *ctx_id);
                }

                frame.set(*dst, result.value);
            }

            // ── Object/List dynamic access ───────────────────
            InstKind::ObjectGet { dst, object, key } => {
                let val = match frame.get(*object) {
                    Value::Object(obj) => obj
                        .get(key)
                        .unwrap_or_else(|| panic!("ObjectGet: missing key"))
                        .share(),
                    other => panic!("ObjectGet on non-object: {other:?}"),
                };
                frame.set(*dst, val);
            }
            InstKind::ListIndex { dst, list, index } => {
                let val = match frame.get(*list) {
                    Value::List(l) => {
                        let idx = if *index < 0 {
                            (l.len() as i32 + *index) as usize
                        } else {
                            *index as usize
                        };
                        l[idx].share()
                    }
                    other => panic!("ListIndex on non-list: {other:?}"),
                };
                frame.set(*dst, val);
            }
            InstKind::ListGet { dst, list, index } => {
                let idx = frame.get(*index).as_int() as usize;
                let val = match frame.get(*list) {
                    Value::List(l) => l[idx].share(),
                    other => panic!("ListGet on non-list: {other:?}"),
                };
                frame.set(*dst, val);
            }
            InstKind::ListSlice {
                dst,
                list,
                skip_head,
                skip_tail,
            } => {
                let val = match frame.take(*list) {
                    Value::List(l) => {
                        let len = l.len();
                        let start = *skip_head;
                        let end = len.saturating_sub(*skip_tail);
                        if start >= end {
                            Value::list(vec![])
                        } else {
                            Value::list(l[start..end].iter().map(|v| v.share()).collect())
                        }
                    }
                    other => panic!("ListSlice on non-list: {other:?}"),
                };
                frame.set(*dst, val);
            }
        }
        Ok(Flow::Next)
    }

    // ── Iterator pulling ───────────────────────────────────────────

    /// Pull one element from an iterator.
    ///
    /// **Pure path**: first call collects all elements through ops pipeline
    /// (calling closures via interpreter), stores in shared Arc. Subsequent
    /// calls just index. No lock after init.
    ///
    /// **Effectful path**: lazy pull, one element at a time.
    pub async fn exec_next(
        &mut self,
        iter: &mut crate::iter::IterHandle,
    ) -> Result<Option<Value>, RuntimeError> {
        use crate::iter::{EffectfulState, IterHandle};

        match iter {
            IterHandle::Pure { items, init, index } => {
                // Fast path: already collected.
                if let Some(collected) = items.lock().unwrap().as_ref() {
                    if *index < collected.len() {
                        let val = collected[*index].clone();
                        *index += 1;
                        return Ok(Some(val));
                    } else {
                        return Ok(None);
                    }
                }

                // First access: collect through ops pipeline.
                let pinit = init
                    .lock()
                    .unwrap()
                    .take()
                    .expect("pure iterator: init already consumed but items not set");
                let collected = self.collect_through_ops(pinit.source, &pinit.ops).await?;
                let arc: Arc<[Value]> = collected.into();
                *items.lock().unwrap() = Some(Arc::clone(&arc));

                if *index < arc.len() {
                    let val = arc[*index].clone();
                    *index += 1;
                    Ok(Some(val))
                } else {
                    Ok(None)
                }
            }
            IterHandle::Effectful { state, .. } => {
                match state {
                    EffectfulState::Done => Ok(None),
                    EffectfulState::Suspended {
                        source,
                        elem_ops,
                        offset,
                        take_remaining,
                    } => {
                        // Take limit reached?
                        if let Some(0) = take_remaining {
                            *state = EffectfulState::Done;
                            return Ok(None);
                        }

                        while *offset < source.len() {
                            let val = source[*offset].clone();
                            *offset += 1;

                            let result = self.apply_ops(val, elem_ops).await?;

                            match result {
                                ApplyResult::Emit(v) => {
                                    if let Some(rem) = take_remaining {
                                        *rem -= 1;
                                    }
                                    return Ok(Some(v));
                                }
                                ApplyResult::Skip => continue,
                                ApplyResult::Expand(items) => {
                                    if let Some(first) = items.into_iter().next() {
                                        if let Some(rem) = take_remaining {
                                            *rem -= 1;
                                        }
                                        return Ok(Some(first));
                                    }
                                    continue;
                                }
                            }
                        }
                        *state = EffectfulState::Done;
                        Ok(None)
                    }
                    EffectfulState::Generator {
                        next_fn,
                        elem_ops,
                        take_remaining,
                    } => {
                        if let Some(0) = take_remaining {
                            *state = EffectfulState::Done;
                            return Ok(None);
                        }

                        while let Some(val) = next_fn.get_mut()() {
                            let result = self.apply_ops(val, elem_ops).await?;

                            match result {
                                ApplyResult::Emit(v) => {
                                    if let Some(rem) = take_remaining {
                                        *rem -= 1;
                                    }
                                    return Ok(Some(v));
                                }
                                ApplyResult::Skip => continue,
                                ApplyResult::Expand(items) => {
                                    if let Some(first) = items.into_iter().next() {
                                        if let Some(rem) = take_remaining {
                                            *rem -= 1;
                                        }
                                        return Ok(Some(first));
                                    }
                                    continue;
                                }
                            }
                        }
                        *state = EffectfulState::Done;
                        Ok(None)
                    }
                }
            }
        }
    }

    /// Collect all source elements through an ops pipeline.
    async fn collect_through_ops(
        &mut self,
        source: Vec<Value>,
        ops: &[crate::iter::IterOp],
    ) -> Result<Vec<Value>, RuntimeError> {
        use crate::iter::IterOp;

        let mut items = source;

        for op in ops {
            match op {
                IterOp::Map(f) => {
                    let mut mapped = Vec::with_capacity(items.len());
                    for item in items {
                        let result = self.call_closure(f, item).await?;
                        mapped.push(result);
                    }
                    items = mapped;
                }
                IterOp::Filter(f) => {
                    let mut filtered = Vec::with_capacity(items.len());
                    for item in items {
                        let keep = self.call_closure(f, item.clone()).await?;
                        if keep.as_bool() {
                            filtered.push(item);
                        }
                    }
                    items = filtered;
                }
                IterOp::Take(n) => {
                    items.truncate(*n);
                }
                IterOp::Skip(n) => {
                    if *n < items.len() {
                        items = items.split_off(*n);
                    } else {
                        items.clear();
                    }
                }
                IterOp::Chain(extra) => {
                    items.extend(extra.iter().cloned());
                }
                IterOp::Flatten => {
                    let mut flat = Vec::new();
                    for item in items {
                        match item {
                            Value::List(l) => flat.extend(l.iter().cloned()),
                            other => flat.push(other),
                        }
                    }
                    items = flat;
                }
                IterOp::FlatMap(f) => {
                    let mut flat = Vec::new();
                    for item in items {
                        let result = self.call_closure(f, item).await?;
                        match result {
                            Value::List(l) => flat.extend(l.iter().cloned()),
                            other => flat.push(other),
                        }
                    }
                    items = flat;
                }
            }
        }

        Ok(items)
    }

    /// Apply ops pipeline to a single element (effectful path).
    async fn apply_ops(
        &mut self,
        mut val: Value,
        ops: &[crate::iter::IterOp],
    ) -> Result<ApplyResult, RuntimeError> {
        use crate::iter::IterOp;

        for op in ops {
            match op {
                IterOp::Map(f) => {
                    val = self.call_closure(f, val).await?;
                }
                IterOp::Filter(f) => {
                    let keep = self.call_closure(f, val.clone()).await?;
                    if !keep.as_bool() {
                        return Ok(ApplyResult::Skip);
                    }
                }
                IterOp::Take(_) | IterOp::Skip(_) | IterOp::Chain(_) => {
                    unreachable!("iterator-level ops resolved at construction time")
                }
                IterOp::Flatten => {
                    return match val {
                        Value::List(l) => Ok(ApplyResult::Expand(l.iter().cloned().collect())),
                        other => Ok(ApplyResult::Emit(other)),
                    };
                }
                IterOp::FlatMap(f) => {
                    let result = self.call_closure(f, val).await?;
                    return match result {
                        Value::List(l) => Ok(ApplyResult::Expand(l.iter().cloned().collect())),
                        other => Ok(ApplyResult::Emit(other)),
                    };
                }
            }
        }
        Ok(ApplyResult::Emit(val))
    }

    /// Intern a name (convenience for builtins creating variants).
    pub fn intern_name(&self, name: &str) -> Astr {
        self.shared.interner.intern(name)
    }

    /// Call a closure with a single argument.
    pub async fn call_closure(&mut self, f: &FnValue, arg: Value) -> Result<Value, RuntimeError> {
        let body = &f.body;
        let label_map = build_label_map_from_insts(&body.insts);
        let mut frame = Frame::new(&body.val_factory, label_map);
        let mut projection_map = FxHashMap::default();

        for (reg, cap) in body.capture_regs.iter().zip(f.captures.iter()) {
            frame.set(*reg, cap.clone());
        }
        if let Some(&param_reg) = body.param_regs.first() {
            frame.set(param_reg, arg);
        }

        let empty_closures = FxHashMap::default();
        self.run_loop(
            &body.insts,
            &empty_closures,
            &mut frame,
            &mut projection_map,
            &body.val_types,
        )
        .await
    }

    /// Call a closure with two arguments (for reduce, fold).
    pub async fn call_closure_2(
        &mut self,
        f: &FnValue,
        arg1: Value,
        arg2: Value,
    ) -> Result<Value, RuntimeError> {
        let body = &f.body;
        let label_map = build_label_map_from_insts(&body.insts);
        let mut frame = Frame::new(&body.val_factory, label_map);
        let mut projection_map = FxHashMap::default();

        for (reg, cap) in body.capture_regs.iter().zip(f.captures.iter()) {
            frame.set(*reg, cap.clone());
        }
        let mut params = body.param_regs.iter();
        if let Some(&r) = params.next() {
            frame.set(r, arg1);
        }
        if let Some(&r) = params.next() {
            frame.set(r, arg2);
        }

        let empty_closures = FxHashMap::default();
        self.run_loop(
            &body.insts,
            &empty_closures,
            &mut frame,
            &mut projection_map,
            &body.val_types,
        )
        .await
    }

    /// Dispatch a direct function call by FunctionId.
    async fn dispatch_call(&mut self, id: &FunctionId, args: Args) -> Result<Value, RuntimeError> {
        match self.function(id) {
            Executable::Builtin(BuiltinHandler::Sync(f)) => {
                let f = *f;
                f(args, &self.shared.interner)
            }
            Executable::Builtin(BuiltinHandler::Async(f)) => {
                let f = *f;
                f(args, self).await
            }
            Executable::Module(_) => {
                let arg_values: Vec<Value> = args.into_vec();
                self.execute_function(id, &arg_values).await
            }
            Executable::Extern(_) => {
                // ExternHandler dispatch requires uses/defs from Spawn/Eval path.
                // Direct FunctionCall to an Extern is not supported — the compiler
                // should emit Spawn+Eval for extern functions.
                panic!(
                    "extern function #{} called via FunctionCall; \
                     use Spawn+Eval for extern functions with uses/defs",
                    id.index()
                )
            }
        }
    }
}

// ── Literal → Value ──────────────────────────────────────────────────

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::string(s.as_str()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Byte(b) => Value::Byte(*b),
        Literal::List(items) => Value::list(items.iter().map(literal_to_value).collect()),
    }
}

// ── BinOp ────────────────────────────────────────────────────────────

fn eval_binop(op: BinOp, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        // Int × Int
        (Value::Int(a), Value::Int(b)) => Ok(match op {
            BinOp::Add => Value::Int(a.wrapping_add(*b)),
            BinOp::Sub => Value::Int(a.wrapping_sub(*b)),
            BinOp::Mul => Value::Int(a.wrapping_mul(*b)),
            BinOp::Div => {
                if *b == 0 {
                    return Err(RuntimeError::division_by_zero());
                }
                Value::Int(a / b)
            }
            BinOp::Mod => {
                if *b == 0 {
                    return Err(RuntimeError::division_by_zero());
                }
                Value::Int(a % b)
            }
            BinOp::Eq => Value::Bool(a == b),
            BinOp::Neq => Value::Bool(a != b),
            BinOp::Lt => Value::Bool(a < b),
            BinOp::Gt => Value::Bool(a > b),
            BinOp::Lte => Value::Bool(a <= b),
            BinOp::Gte => Value::Bool(a >= b),
            BinOp::BitAnd => Value::Int(a & b),
            BinOp::BitOr => Value::Int(a | b),
            BinOp::Xor => Value::Int(a ^ b),
            BinOp::Shl => Value::Int(a << b),
            BinOp::Shr => Value::Int(a >> b),
            BinOp::And | BinOp::Or => panic!("And/Or on Int"),
        }),
        // Float × Float
        (Value::Float(a), Value::Float(b)) => Ok(match op {
            BinOp::Add => Value::Float(a + b),
            BinOp::Sub => Value::Float(a - b),
            BinOp::Mul => Value::Float(a * b),
            BinOp::Div => Value::Float(a / b),
            BinOp::Mod => Value::Float(a % b),
            BinOp::Eq => Value::Bool(a == b),
            BinOp::Neq => Value::Bool(a != b),
            BinOp::Lt => Value::Bool(a < b),
            BinOp::Gt => Value::Bool(a > b),
            BinOp::Lte => Value::Bool(a <= b),
            BinOp::Gte => Value::Bool(a >= b),
            _ => panic!("unsupported float binop {op:?}"),
        }),
        // String + String
        (Value::String(a), Value::String(b)) => match op {
            BinOp::Add => {
                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);
                Ok(Value::string(s))
            }
            BinOp::Eq => Ok(Value::Bool(a == b)),
            BinOp::Neq => Ok(Value::Bool(a != b)),
            _ => panic!("unsupported string binop {op:?}"),
        },
        // Bool × Bool
        (Value::Bool(a), Value::Bool(b)) => Ok(match op {
            BinOp::And => Value::Bool(*a && *b),
            BinOp::Or => Value::Bool(*a || *b),
            BinOp::Eq => Value::Bool(a == b),
            BinOp::Neq => Value::Bool(a != b),
            BinOp::Xor => Value::Bool(a ^ b),
            _ => panic!("unsupported bool binop {op:?}"),
        }),
        _ => Err(RuntimeError::bin_op_mismatch(op, left.kind(), right.kind())),
    }
}

// ── UnaryOp ──────────────────────────────────────────────────────────

fn eval_unaryop(op: UnaryOp, val: &Value) -> Result<Value, RuntimeError> {
    match (op, val) {
        (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
        (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
        (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        _ => Err(RuntimeError::unary_op_mismatch(op, val.kind())),
    }
}

// ── Cast ─────────────────────────────────────────────────────────────

fn eval_cast(kind: CastKind, val: Value) -> Value {
    use crate::iter::{IterHandle, SequenceChain};
    use acvus_mir::ty::Effect;

    match kind {
        CastKind::DequeToList => match val {
            Value::Deque(d) => Value::list(d.as_slice().to_vec()),
            other => panic!("DequeToList on {other:?}"),
        },
        CastKind::ListToIterator => match val {
            Value::List(l) => {
                let items =
                    std::sync::Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone());
                Value::iterator(IterHandle::from_list(items, Effect::pure()))
            }
            other => panic!("ListToIterator on {other:?}"),
        },
        CastKind::DequeToIterator => match val {
            Value::Deque(d) => {
                let items = std::sync::Arc::try_unwrap(d)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .into_vec();
                Value::iterator(IterHandle::from_list(items, Effect::pure()))
            }
            other => panic!("DequeToIterator on {other:?}"),
        },
        CastKind::RangeToIterator => match val {
            Value::Range(r) => {
                let items: Vec<Value> = if r.inclusive {
                    (r.start..=r.end).map(Value::Int).collect()
                } else {
                    (r.start..r.end).map(Value::Int).collect()
                };
                Value::iterator(IterHandle::from_list(items, Effect::pure()))
            }
            other => panic!("RangeToIterator on {other:?}"),
        },
        CastKind::DequeToSequence => match val {
            Value::Deque(d) => {
                let td = std::sync::Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone());
                Value::sequence(SequenceChain::from_stored(td, Effect::pure()))
            }
            other => panic!("DequeToSequence on {other:?}"),
        },
        CastKind::SequenceToIterator => match val {
            Value::Sequence(sc) => Value::iterator(sc.into_iter_handle()),
            other => panic!("SequenceToIterator on {other:?}"),
        },
    }
}
