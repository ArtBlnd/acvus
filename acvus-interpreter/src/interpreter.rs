use std::sync::Arc;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{Callee, CastKind, Inst, InstKind, Label, MirBody, MirModule, ValueId};
use acvus_mir::ty::{Effect, Ty};
use acvus_utils::{Astr, Freeze, Interner, LocalFactory, LocalVec, TrackedDeque};
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::error::RuntimeError;
use crate::value::{FnValue, Value};

// ── Builtin dispatch ─────────────────────────────────────────────────

/// Args passed to builtin functions. Stack-allocated for ≤4 args.
pub type Args = SmallVec<[Value; 4]>;

/// Sync builtin — no future overhead.
pub type SyncBuiltinFn = fn(Args, &Interner) -> Result<Value, RuntimeError>;

/// Async builtin — standalone, no Interpreter needed.
pub type AsyncBuiltinFn =
    fn(Args, Interner) -> BoxFuture<'static, Result<Value, RuntimeError>>;

/// Builtin handler — sync or async.
pub enum BuiltinHandler {
    Sync(SyncBuiltinFn),
    Async(AsyncBuiltinFn),
}

// ── Frame ────────────────────────────────────────────────────────────

/// Register file. Stores one `Value` per SSA ValueId.
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

    #[inline]
    fn set(&mut self, id: ValueId, value: Value) {
        self.regs[id] = value;
    }

    #[inline]
    fn get(&self, id: ValueId) -> &Value {
        let v = &self.regs[id];
        assert!(!v.is_empty(), "get: register {id:?} already moved");
        v
    }

    #[inline]
    fn take(&mut self, id: ValueId) -> Value {
        let v = self.regs[id].take();
        assert!(!v.is_empty(), "take: register {id:?} already moved");
        v
    }

    #[inline]
    fn share(&self, id: ValueId) -> Value {
        self.get(id).share()
    }

    #[inline]
    fn use_val(&mut self, id: ValueId, val_types: &FxHashMap<ValueId, Ty>) -> Value {
        if let Some(ty) = val_types.get(&id)
            && acvus_mir::validate::move_check::is_move_only(ty) == Some(true)
        {
            return self.take(id);
        }
        self.share(id)
    }

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

// ── Public types ────────────────────────────────────────────────────

/// Result of execution — return value + context mutations.
pub struct ExecResult {
    pub value: Value,
    pub writes: Vec<ContextWrite>,
    pub defs: Vec<Value>,
}

/// A single executable unit — MIR module, builtin handler, or extern function.
pub enum Executable {
    Module(MirModule),
    Builtin(BuiltinHandler),
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
    pub functions: Freeze<FxHashMap<QualifiedRef, Executable>>,
    pub fn_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
    pub context_names: Freeze<FxHashMap<QualifiedRef, Astr>>,
    pub executor: Arc<dyn crate::executor::Executor>,
}

impl InterpreterContext {
    pub fn new(
        interner: &Interner,
        functions: FxHashMap<QualifiedRef, Executable>,
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

    pub fn with_fn_types(mut self, fn_types: FxHashMap<QualifiedRef, Ty>) -> Self {
        self.fn_types = Freeze::new(fn_types);
        self
    }

    pub fn with_context_names(mut self, context_names: FxHashMap<QualifiedRef, Astr>) -> Self {
        self.context_names = Freeze::new(context_names);
        self
    }
}

// ── RunContext — mutable state bundle for run_loop ───────────────────

/// Per-execution mutable state passed through the run_loop call chain.
struct RunContext {
    shared: InterpreterContext,
    overlay: ContextOverlay,
    variables: FxHashMap<Astr, Value>,
}

// ── Standalone helpers (extracted from Interpreter methods) ──────────

fn resolve_context_key(
    shared: &InterpreterContext,
    qref: &QualifiedRef,
) -> Result<String, RuntimeError> {
    let name = shared
        .context_names
        .get(qref)
        .ok_or_else(|| RuntimeError::internal(format!("no context name for {qref:?}")))?;
    Ok(shared.interner.resolve(*name).to_string())
}

fn collect_context_uses(
    shared: &InterpreterContext,
    overlay: &ContextOverlay,
    context_uses: &[(QualifiedRef, ValueId)],
    fn_id: &QualifiedRef,
    frame: &Frame,
) -> Result<Vec<Value>, RuntimeError> {
    if !context_uses.is_empty() {
        Ok(context_uses
            .iter()
            .map(|(_, vid)| frame.share(*vid))
            .collect())
    } else {
        collect_uses_from_type(shared, overlay, fn_id)
    }
}

fn collect_uses_from_type(
    shared: &InterpreterContext,
    overlay: &ContextOverlay,
    fn_id: &QualifiedRef,
) -> Result<Vec<Value>, RuntimeError> {
    let Some(fn_ty) = shared.fn_types.get(fn_id) else {
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
        .filter_map(|t| match t {
            acvus_mir::ty::EffectTarget::Context(qref) => Some(qref),
            acvus_mir::ty::EffectTarget::Token(_) => None,
        })
        .map(|qref| {
            let key = resolve_context_key(shared, qref)?;
            overlay
                .get(&key)
                .cloned()
                .ok_or_else(|| RuntimeError::internal(format!("undefined context '{key}'")))
        })
        .collect()
}

fn apply_context_defs(
    shared: &InterpreterContext,
    overlay: &mut ContextOverlay,
    context_defs: &[(QualifiedRef, ValueId)],
    fn_id: &QualifiedRef,
    defs: Vec<Value>,
    frame: &mut Frame,
    projection_map: &mut FxHashMap<ValueId, QualifiedRef>,
) -> Result<(), RuntimeError> {
    if !context_defs.is_empty() {
        for ((ctx_id, vid), def_value) in context_defs.iter().zip(defs) {
            let key = resolve_context_key(shared, ctx_id)?;
            let for_overlay = def_value.share();
            frame.set(*vid, def_value);
            overlay.apply_field(&key, &[], for_overlay);
            projection_map.insert(*vid, *ctx_id);
        }
    } else {
        apply_defs_from_type(shared, overlay, fn_id, defs)?;
    }
    Ok(())
}

fn apply_defs_from_type(
    shared: &InterpreterContext,
    overlay: &mut ContextOverlay,
    fn_id: &QualifiedRef,
    defs: Vec<Value>,
) -> Result<(), RuntimeError> {
    let Some(fn_ty) = shared.fn_types.get(fn_id) else {
        return Ok(());
    };
    let Ty::Fn {
        effect: Effect::Resolved(eff),
        ..
    } = fn_ty
    else {
        return Ok(());
    };
    for (target, def_value) in eff.writes.iter().zip(defs) {
        let qref = match target {
            acvus_mir::ty::EffectTarget::Context(qref) => qref,
            acvus_mir::ty::EffectTarget::Token(_) => continue,
        };
        let key = resolve_context_key(shared, qref)?;
        overlay.apply_field(&key, &[], def_value);
    }
    Ok(())
}

fn lookup_function<'a>(shared: &'a InterpreterContext, id: &QualifiedRef) -> &'a Executable {
    shared.functions.get(id).unwrap_or_else(|| {
        let name = shared.interner.resolve(id.name);
        panic!("no function for {id:?} (name={name:?})")
    })
}

fn lookup_module<'a>(shared: &'a InterpreterContext, id: &QualifiedRef) -> &'a MirModule {
    match lookup_function(shared, id) {
        Executable::Module(m) => m,
        other => panic!(
            "expected Module for {id:?}, got {}",
            other.variant_name()
        ),
    }
}

// ── run_loop — standalone execution engine ──────────────────────────

fn run_loop<'s>(
    ctx: &'s mut RunContext,
    insts: &'s [Inst],
    closures: &'s FxHashMap<Label, MirBody>,
    frame: &'s mut Frame,
    projection_map: &'s mut FxHashMap<ValueId, QualifiedRef>,
    val_types: &'s FxHashMap<ValueId, Ty>,
) -> BoxFuture<'s, Result<Value, RuntimeError>> {
    Box::pin(run_loop_inner(
        ctx,
        insts,
        closures,
        frame,
        projection_map,
        val_types,
    ))
}

async fn run_loop_inner(
    ctx: &mut RunContext,
    insts: &[Inst],
    closures: &FxHashMap<Label, MirBody>,
    frame: &mut Frame,
    projection_map: &mut FxHashMap<ValueId, QualifiedRef>,
    val_types: &FxHashMap<ValueId, Ty>,
) -> Result<Value, RuntimeError> {
    let mut pc = 0;
    while pc < insts.len() {
        match execute_inst(ctx, insts, closures, pc, frame, projection_map, val_types).await? {
            Flow::Next => pc += 1,
            Flow::Jump(target) => pc = target,
            Flow::Return(val) => return Ok(val),
        }
    }
    Ok(Value::Unit)
}

/// Execute a single instruction. Returns control flow directive.
async fn execute_inst(
    ctx: &mut RunContext,
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
            let val = ctx
                .variables
                .get(name)
                .unwrap_or_else(|| {
                    panic!(
                        "undefined variable ${}",
                        ctx.shared.interner.resolve(*name)
                    )
                })
                .share();
            frame.set(*dst, val);
        }
        InstKind::VarStore { name, src } => {
            let val = frame.share(*src);
            ctx.variables.insert(*name, val);
        }
        InstKind::ParamLoad { dst, name } => {
            let val = ctx
                .variables
                .get(name)
                .unwrap_or_else(|| {
                    panic!(
                        "undefined param ${}",
                        ctx.shared.interner.resolve(*name)
                    )
                })
                .share();
            frame.set(*dst, val);
        }

        // ── Context ───────────────────────────────────────
        InstKind::ContextProject { dst, ctx: ctx_ref, .. } => {
            projection_map.insert(*dst, *ctx_ref);
            frame.set(*dst, Value::Unit);
        }
        InstKind::ContextLoad { dst, src } => {
            let ctx_id = projection_map[src];
            let name = ctx
                .shared
                .context_names
                .get(&ctx_id)
                .unwrap_or_else(|| panic!("context load: no name for {:?}", ctx_id));
            let key = ctx.shared.interner.resolve(*name);
            let val = ctx
                .overlay
                .get(key)
                .unwrap_or_else(|| panic!("context load: undefined context '{}'", key))
                .clone();
            frame.set(*dst, val);
        }
        InstKind::ContextStore { dst, value } => {
            let ctx_id = projection_map[dst];
            let name = ctx
                .shared
                .context_names
                .get(&ctx_id)
                .unwrap_or_else(|| panic!("context store: no name for {:?}", ctx_id));
            let key = ctx.shared.interner.resolve(*name);
            let val = frame.share(*value);
            ctx.overlay.apply_field(key, &[], val);
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
                        panic!("missing field {}", ctx.shared.interner.resolve(*field))
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
                    shared: ctx.shared.clone(),
                    overlay: ctx.overlay.spawn_fork(),
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

        // ── List iteration ───────────────────────────────
        InstKind::ListStep {
            dst,
            list,
            index_src,
            index_dst,
            done,
            done_args,
        } => {
            let idx = frame.get(*index_src).as_int();
            let len = match frame.get(*list) {
                Value::List(l) => l.len() as i64,
                Value::Deque(d) => d.len() as i64,
                other => panic!("ListStep: expected List or Deque, got {other:?}"),
            };
            if idx >= len {
                let target = frame.jump(insts, done, done_args);
                return Ok(Flow::Jump(target));
            }
            let val = match frame.get(*list) {
                Value::List(l) => l[idx as usize].share(),
                Value::Deque(d) => d.as_slice()[idx as usize].share(),
                _ => unreachable!(),
            };
            frame.set(*dst, val);
            frame.set(*index_dst, Value::Int(idx + 1));
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
        InstKind::Undef { dst } => {
            frame.set(*dst, Value::Undef);
        }
        InstKind::Poison { .. } => {
            panic!("reached poison instruction");
        }

        // ── Functions ─────────────────────────────────────
        InstKind::LoadFunction { dst: _, id: _ } => {
            todo!("LoadFunction: graph-level function references not yet supported at runtime");
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
                    let is_extern = matches!(lookup_function(&ctx.shared, id), Executable::Extern(_));
                    if is_extern {
                        let handler = match lookup_function(&ctx.shared, id) {
                            Executable::Extern(h) => h.clone(),
                            _ => unreachable!(),
                        };
                        let arg_vals: Vec<Value> =
                            args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                        let uses =
                            collect_context_uses(&ctx.shared, &ctx.overlay, context_uses, id, frame)?;

                        let output = match &handler {
                            crate::extern_fn::ExternHandler::Sync(f) => {
                                f(arg_vals, uses, &ctx.shared.interner)?
                            }
                            crate::extern_fn::ExternHandler::Async(f) => {
                                let interner = ctx.shared.interner.clone();
                                f(arg_vals, uses, interner).await?
                            }
                        };

                        apply_context_defs(
                            &ctx.shared,
                            &mut ctx.overlay,
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
                        dispatch_call(ctx, id, arg_vals).await?
                    }
                }
                Callee::Indirect(val_id) => {
                    let fv = frame.take(*val_id).into_fn();
                    let call_args: Vec<Value> = args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                    fn_value_call(&fv, call_args).await?
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
            let is_extern = matches!(lookup_function(&ctx.shared, &callee_id), Executable::Extern(_));
            let handle = if is_extern {
                let spawn_args: Vec<Value> = args.iter().map(|a| frame.share(*a)).collect();
                let uses =
                    collect_context_uses(&ctx.shared, &ctx.overlay, context_uses, &callee_id, frame)?;
                let handler = match lookup_function(&ctx.shared, &callee_id) {
                    Executable::Extern(h) => h.clone(),
                    _ => unreachable!(),
                };
                let interner = ctx.shared.interner.clone();
                match &handler {
                    crate::extern_fn::ExternHandler::Sync(f) => {
                        let f = Arc::clone(f);
                        ctx.shared.executor.spawn_blocking(Box::new(move || {
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
                        ctx.shared.executor.spawn_async(Box::pin(async move {
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
                // Fork interpreter for Module/Builtin spawn.
                let spawn_args: Vec<Value> = args.iter().map(|a| frame.share(*a)).collect();
                let child = Interpreter {
                    shared: ctx.shared.clone(),
                    entry: callee_id,
                    overlay: ctx.overlay.spawn_fork(),
                    variables: FxHashMap::default(),
                    spawn_args,
                };
                ctx.shared.executor.spawn_interpreter(child)
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
            let result = ctx.shared.executor.eval(handle).await?;

            if !result.writes.is_empty() {
                ctx.overlay.merge_patches(result.writes);
            }

            let defs = result.defs;
            let defs_count = defs.len();
            for ((ctx_id, vid), def_value) in context_defs.iter().zip(defs) {
                let key = resolve_context_key(&ctx.shared, ctx_id)?;
                let for_overlay = def_value.share();
                frame.set(*vid, def_value);
                ctx.overlay.apply_field(&key, &[], for_overlay);
                projection_map.insert(*vid, *ctx_id);
            }

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

// ── Function dispatch ───────────────────────────────────────────────

async fn dispatch_call(
    ctx: &mut RunContext,
    id: &QualifiedRef,
    args: Args,
) -> Result<Value, RuntimeError> {
    match lookup_function(&ctx.shared, id) {
        Executable::Builtin(BuiltinHandler::Sync(f)) => {
            let f = *f;
            f(args, &ctx.shared.interner)
        }
        Executable::Builtin(BuiltinHandler::Async(f)) => {
            let f = *f;
            let interner = ctx.shared.interner.clone();
            f(args, interner).await
        }
        Executable::Module(_) => {
            let arg_values: Vec<Value> = args.into_vec();
            execute_function(ctx, id, &arg_values).await
        }
        Executable::Extern(_) => {
            panic!(
                "extern function {id:?} called via FunctionCall; \
                 use Spawn+Eval for extern functions with uses/defs",
            )
        }
    }
}

/// Execute a MIR module function by QualifiedRef with explicit args.
async fn execute_function(
    ctx: &mut RunContext,
    id: &QualifiedRef,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    let m = lookup_module(&ctx.shared, id);
    let insts: Arc<[Inst]> = m.main.insts.clone().into();
    let closures = m.closures.clone();
    let param_regs = m.main.param_regs.clone();
    let val_types = m.main.val_types.clone();
    let label_map = build_label_map(&m.main);
    let mut frame = Frame::new(&m.main.val_factory, label_map);
    for (reg, val) in param_regs.iter().zip(args.iter()) {
        frame.set(*reg, val.clone());
    }
    let mut projection_map: FxHashMap<ValueId, QualifiedRef> = FxHashMap::default();
    run_loop(ctx, &insts, &closures, &mut frame, &mut projection_map, &val_types).await
}

// ── Closure calling ─────────────────────────────────────────────────

/// Execute a FnValue with the given arguments. Self-contained — uses the
/// FnValue's own shared context and forked overlay.
pub async fn fn_value_call(
    f: &FnValue,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    let mut closure_ctx = RunContext {
        shared: f.shared.clone(),
        overlay: f.overlay.spawn_fork(),
        variables: FxHashMap::default(),
    };

    let body = &f.body;
    let label_map = build_label_map_from_insts(&body.insts);
    let mut frame = Frame::new(&body.val_factory, label_map);
    let mut projection_map = FxHashMap::default();

    for (reg, cap) in body.capture_regs.iter().zip(f.captures.iter()) {
        frame.set(*reg, cap.clone());
    }
    for (reg, arg) in body.param_regs.iter().zip(args) {
        frame.set(*reg, arg);
    }

    let empty_closures = FxHashMap::default();
    run_loop(
        &mut closure_ctx,
        &body.insts,
        &empty_closures,
        &mut frame,
        &mut projection_map,
        &body.val_types,
    )
    .await
}

// ── Iterator pulling ────────────────────────────────────────────────

/// Pull one element from an iterator. Standalone — no Interpreter needed.
/// FnValues in IterOps are self-contained and execute via fn_value_call.
pub async fn exec_next(
    iter: &mut crate::iter::IterHandle,
) -> Result<Option<Value>, RuntimeError> {
    use crate::iter::{EffectfulState, IterHandle};

    match iter {
        IterHandle::Pure { items, init, index } => {
            if let Some(collected) = items.lock().unwrap().as_ref() {
                if *index < collected.len() {
                    let val = collected[*index].clone();
                    *index += 1;
                    return Ok(Some(val));
                } else {
                    return Ok(None);
                }
            }

            let pinit = init
                .lock()
                .unwrap()
                .take()
                .expect("pure iterator: init already consumed but items not set");
            let collected = collect_through_ops(pinit.source, &pinit.ops).await?;
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
                    if let Some(0) = take_remaining {
                        *state = EffectfulState::Done;
                        return Ok(None);
                    }

                    while *offset < source.len() {
                        let val = source[*offset].clone();
                        *offset += 1;

                        let result = apply_ops(val, elem_ops).await?;

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
                        let result = apply_ops(val, elem_ops).await?;

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
                    mapped.push(f.call(item).await?);
                }
                items = mapped;
            }
            IterOp::Filter(f) => {
                let mut filtered = Vec::with_capacity(items.len());
                for item in items {
                    let keep = f.call(item.clone()).await?;
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
                    let result = f.call(item).await?;
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
    mut val: Value,
    ops: &[crate::iter::IterOp],
) -> Result<ApplyResult, RuntimeError> {
    use crate::iter::IterOp;

    for op in ops {
        match op {
            IterOp::Map(f) => {
                val = f.call(val).await?;
            }
            IterOp::Filter(f) => {
                let keep = f.call(val.clone()).await?;
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
                let result = f.call(val).await?;
                return match result {
                    Value::List(l) => Ok(ApplyResult::Expand(l.iter().cloned().collect())),
                    other => Ok(ApplyResult::Emit(other)),
                };
            }
        }
    }
    Ok(ApplyResult::Emit(val))
}

// ── Interpreter — thin entry point ──────────────────────────────────

pub struct Interpreter {
    shared: InterpreterContext,
    entry: QualifiedRef,
    overlay: ContextOverlay,
    variables: FxHashMap<Astr, Value>,
    spawn_args: Vec<Value>,
}

impl Interpreter {
    pub fn new(shared: InterpreterContext, entry: QualifiedRef, overlay: ContextOverlay) -> Self {
        Self {
            shared,
            entry,
            overlay,
            variables: FxHashMap::default(),
            spawn_args: Vec::new(),
        }
    }

    pub fn fork(&self, entry: QualifiedRef, args: Vec<Value>) -> Self {
        Self {
            shared: self.shared.clone(),
            entry,
            overlay: self.overlay.spawn_fork(),
            variables: FxHashMap::default(),
            spawn_args: args,
        }
    }

    /// Execute the entry module. Returns value + accumulated context writes.
    pub async fn execute(&mut self) -> Result<ExecResult, RuntimeError> {
        let entry = self.entry;
        let args = std::mem::take(&mut self.spawn_args);

        let mut run_ctx = RunContext {
            shared: self.shared.clone(),
            overlay: std::mem::replace(
                &mut self.overlay,
                ContextOverlay::new(
                    Arc::new(std::collections::HashMap::new()),
                    self.shared.interner.clone(),
                ),
            ),
            variables: std::mem::take(&mut self.variables),
        };

        let value = execute_function(&mut run_ctx, &entry, &args).await?;

        let writes = run_ctx.overlay.into_patches();
        Ok(ExecResult {
            value,
            writes,
            defs: Vec::new(),
        })
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
    match kind {
        CastKind::DequeToList => match val {
            Value::Deque(d) => Value::list(d.as_slice().to_vec()),
            other => panic!("DequeToList on {other:?}"),
        },
        CastKind::RangeToList => match val {
            Value::Range(r) => {
                let items: Vec<Value> = if r.inclusive {
                    (r.start..=r.end).map(Value::Int).collect()
                } else {
                    (r.start..r.end).map(Value::Int).collect()
                };
                Value::list(items)
            }
            other => panic!("RangeToList on {other:?}"),
        },
        CastKind::Extern(fn_id) => {
            panic!("ExternCast({fn_id:?}) should have been lowered to a FunctionCall")
        }
    }
}
