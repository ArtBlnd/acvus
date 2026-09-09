use std::sync::Arc;

use acvus_ast::{BinOp, Literal, UnaryOp};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, Inst, InstKind, Label, MirBody, MirModule, RefTarget, ValueId,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner, LocalFactory, LocalVec};
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::error::RuntimeError;
use crate::value::{FnValue, Value};

/// Runtime representation of a Ref instruction's target.
#[derive(Debug, Clone)]
enum RuntimeRef {
    Var {
        slot: ValueId,
        path: Vec<Astr>,
    },
    Param {
        slot: ValueId,
        path: Vec<Astr>,
    },
    Context {
        qref: QualifiedRef,
        path: Vec<Astr>,
    },
}

impl RuntimeRef {
    fn from_target(target: &RefTarget, path: Vec<Astr>) -> Self {
        match target {
            RefTarget::Var(slot) => RuntimeRef::Var { slot: *slot, path },
            RefTarget::Param(slot) => RuntimeRef::Param { slot: *slot, path },
            RefTarget::Context(qref) => RuntimeRef::Context { qref: *qref, path },
        }
    }
}

/// Load a value from a RuntimeRef.
fn load_ref(rt_ref: &RuntimeRef, ctx: &RunContext) -> Value {
    let interner = &ctx.shared.interner;

    fn walk_path(mut val: Value, path: &[Astr], interner: &Interner) -> Value {
        for f in path {
            val = match &val {
                Value::Object(obj) => obj
                    .get(f)
                    .unwrap_or_else(|| panic!("missing field {}", interner.resolve(*f)))
                    .share(),
                other => panic!("field load on non-object: {other:?}"),
            };
        }
        val
    }

    match rt_ref {
        RuntimeRef::Var { slot, path } | RuntimeRef::Param { slot, path } => {
            let val = ctx
                .variables
                .get(slot)
                .unwrap_or_else(|| panic!("undefined variable slot {:?}", slot))
                .share();
            walk_path(val, path, interner)
        }
        RuntimeRef::Context { qref, path } => {
            let name = ctx
                .shared
                .context_names
                .get(qref)
                .unwrap_or_else(|| panic!("context load: no name for {:?}", qref));
            let key = interner.resolve(*name);
            let val = ctx
                .page
                .get(key)
                .unwrap_or_else(|| panic!("context load: undefined context '{}'", key));
            walk_path(val, path, interner)
        }
    }
}

/// Set a nested field path on a mutable Value.
fn store_path(target: &mut Value, path: &[Astr], value: Value, interner: &Interner) {
    assert!(!path.is_empty(), "store_path called with empty path");
    if path.len() == 1 {
        match target {
            Value::Object(obj) => {
                Arc::make_mut(obj).insert(path[0], value);
            }
            other => panic!("field store on non-object: {other:?}"),
        }
    } else {
        match target {
            Value::Object(obj) => {
                let inner = Arc::make_mut(obj)
                    .get_mut(&path[0])
                    .unwrap_or_else(|| {
                        panic!("missing field {}", interner.resolve(path[0]))
                    });
                store_path(inner, &path[1..], value, interner);
            }
            other => panic!("field store on non-object: {other:?}"),
        }
    }
}

/// Store a value through a RuntimeRef.
fn store_ref(rt_ref: &RuntimeRef, ctx: &mut RunContext, value: Value) {
    let interner = ctx.shared.interner.clone();
    match rt_ref {
        RuntimeRef::Var { slot, path } if path.is_empty() => {
            ctx.variables.insert(*slot, value);
        }
        RuntimeRef::Var { slot, path } => {
            let entry = ctx
                .variables
                .get_mut(slot)
                .unwrap_or_else(|| panic!("undefined variable slot {:?}", slot));
            store_path(entry, path, value, &interner);
        }
        RuntimeRef::Param { slot, .. } => {
            panic!("cannot store to param slot {:?}", slot);
        }
        RuntimeRef::Context { qref, path } => {
            let name = ctx
                .shared
                .context_names
                .get(qref)
                .unwrap_or_else(|| panic!("context store: no name for {:?}", qref));
            let key = interner.resolve(*name);
            if path.is_empty() {
                ctx.page.set(key, value);
            } else {
                let mut current = ctx
                    .page
                    .get(key)
                    .unwrap_or_else(|| panic!("context store: undefined context '{}'", key));
                store_path(&mut current, path, value, &interner);
                ctx.page.set(key, current);
            }
        }
    }
}

/// Deep field set on a scalar Value::Object. Returns a new object with the field replaced.
fn field_set_deep(
    obj: Value,
    field: &Astr,
    rest: &[Astr],
    value: Value,
    interner: &Interner,
) -> Value {
    match obj {
        Value::Object(arc_map) => {
            let mut map = (*arc_map).clone();
            if rest.is_empty() {
                map.insert(*field, value);
            } else {
                let inner = map
                    .get(field)
                    .unwrap_or_else(|| panic!("missing field {}", interner.resolve(*field)))
                    .share();
                let updated = field_set_deep(inner, &rest[0], &rest[1..], value, interner);
                map.insert(*field, updated);
            }
            Value::Object(Arc::new(map))
        }
        other => panic!("FieldSet on non-object: {other:?}"),
    }
}

/// Call arguments. Stack-allocated for <=4 args.
pub type Args = SmallVec<[Value; 4]>;

// -- Frame ------------------------------------------------------------

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

// -- Label map --------------------------------------------------------

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

enum SpawnKind {
    Extern(crate::extern_fn::ExternHandler),
    Module,
}

/// Control flow after executing one instruction.
enum Flow {
    Next,
    Jump(usize),
    Return(Value),
}

use crate::journal::{ContextWrite, InMemoryContext, RuntimeContext};

// -- Public types ----------------------------------------------------

/// Result of execution - return value + context mutations.
pub struct ExecResult {
    pub value: Value,
    pub writes: Vec<ContextWrite>,
}

/// A single executable unit - MIR module or extern function.
pub enum Executable {
    Module(MirModule),
    Extern(crate::extern_fn::ExternHandler),
}

impl Executable {
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Module(_) => "Module",
            Self::Extern(_) => "Extern",
        }
    }
}

/// Readonly shared state - clone is cheap (Freeze/Arc internally).
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

// -- RunContext - mutable state bundle for run_loop -------------------

/// Per-execution mutable state passed through the run_loop call chain.
struct RunContext {
    shared: InterpreterContext,
    page: InMemoryContext,
    variables: FxHashMap<ValueId, Value>,
}

// -- Standalone helpers (extracted from Interpreter methods) ----------

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

fn lookup_function<'a>(shared: &'a InterpreterContext, id: &QualifiedRef) -> &'a Executable {
    shared.functions.get(id).unwrap_or_else(|| {
        let name = shared.interner.resolve(id.name);
        panic!("no function for {id:?} (name={name:?})")
    })
}

fn lookup_module<'a>(shared: &'a InterpreterContext, id: &QualifiedRef) -> &'a MirModule {
    match lookup_function(shared, id) {
        Executable::Module(m) => m,
        other => panic!("expected Module for {id:?}, got {}", other.variant_name()),
    }
}

// -- run_loop - standalone execution engine --------------------------

fn run_loop<'s>(
    ctx: &'s mut RunContext,
    insts: &'s [Inst],
    closures: &'s FxHashMap<Label, MirBody>,
    frame: &'s mut Frame,
    projection_map: &'s mut FxHashMap<ValueId, RuntimeRef>,
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
    projection_map: &mut FxHashMap<ValueId, RuntimeRef>,
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
    projection_map: &mut FxHashMap<ValueId, RuntimeRef>,
    val_types: &FxHashMap<ValueId, Ty>,
) -> Result<Flow, RuntimeError> {
    match &insts[pc].kind {
        // -- Constants ------------------------------------
        InstKind::Const { dst, value } => {
            frame.set(*dst, literal_to_value(value));
        }

        // -- Projection ----------------------------------
        InstKind::Ref { dst, target, path } => {
            projection_map.insert(*dst, RuntimeRef::from_target(target, path.clone()));
            frame.set(*dst, Value::Unit);
        }
        InstKind::Load { dst, src, .. } => {
            let rt_ref = projection_map
                .get(src)
                .unwrap_or_else(|| panic!("Load: no RuntimeRef for src {:?}", src))
                .clone();
            let val = load_ref(&rt_ref, ctx);
            frame.set(*dst, val);
        }
        InstKind::Store { dst, value, .. } => {
            let rt_ref = projection_map
                .get(dst)
                .unwrap_or_else(|| panic!("Store: no RuntimeRef for dst {:?}", dst))
                .clone();
            let val = frame.share(*value);
            store_ref(&rt_ref, ctx, val);
        }

        // -- Arithmetic / Logic ---------------------------
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

        // -- Field / Index access -------------------------
        InstKind::FieldGet {
            dst,
            object,
            field,
            rest,
        } => {
            let mut val = match frame.get(*object) {
                Value::Object(obj) => obj
                    .get(field)
                    .unwrap_or_else(|| {
                        panic!("missing field {}", ctx.shared.interner.resolve(*field))
                    })
                    .share(),
                other => panic!("FieldGet on non-object: {other:?}"),
            };
            // Follow rest path for multi-depth access.
            for r in rest {
                val = match &val {
                    Value::Object(obj) => obj
                        .get(r)
                        .unwrap_or_else(|| {
                            panic!("missing field {}", ctx.shared.interner.resolve(*r))
                        })
                        .share(),
                    other => panic!("FieldGet rest on non-object: {other:?}"),
                };
            }
            frame.set(*dst, val);
        }
        InstKind::FieldSet {
            dst,
            object,
            field,
            rest,
            value,
        } => {
            let obj_val = frame.share(*object);
            let new_val = frame.share(*value);
            let result = field_set_deep(obj_val, field, rest, new_val, &ctx.shared.interner);
            frame.set(*dst, result);
        }
        InstKind::TupleIndex { dst, tuple, index } => {
            let val = match frame.get(*tuple) {
                Value::Tuple(t) => t[*index].share(),
                other => panic!("TupleIndex on non-tuple: {other:?}"),
            };
            frame.set(*dst, val);
        }

        // -- Constructors ---------------------------------
        InstKind::MakeArray { dst, elements } => {
            let items: Vec<Value> = elements.iter().map(|e| frame.share(*e)).collect();
            frame.set(*dst, Value::array(items));
        }
        InstKind::MakeObject { dst, fields } => {
            let obj: FxHashMap<Astr, Value> =
                fields.iter().map(|(k, v)| (*k, frame.share(*v))).collect();
            frame.set(*dst, Value::object(obj));
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
                    page: ctx.page.fork(),
                    body: Arc::new(closure_body.clone()),
                    captures: captured.into(),
                }),
            );
        }

        // -- Variant --------------------------------------
        InstKind::MakeVariant { dst, tag, payload } => {
            let p = payload.map(|v| frame.share(v));
            frame.set(*dst, Value::variant(*tag, p));
        }
        InstKind::TestVariant { dst, src, tag } => {
            let matches = match frame.get(*src) {
                Value::Variant(v) => v.tag == *tag,
                _ => false,
            };
            frame.set(*dst, Value::bool_(matches));
        }
        InstKind::UnwrapVariant { dst, src } => {
            let val = match frame.take(*src) {
                Value::Variant(v) if v.payload.is_some() => {
                    let p = v.payload.unwrap();
                    Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share())
                }
                Value::Variant(_) => Value::Unit,
                other => panic!("UnwrapVariant on non-variant: {other:?}"),
            };
            frame.set(*dst, val);
        }

        // -- Pattern testing ------------------------------
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
        InstKind::TestObjectKey { dst, src, key } => {
            let has = match frame.get(*src) {
                Value::Object(o) => o.contains_key(key),
                _ => false,
            };
            frame.set(*dst, Value::bool_(has));
        }

        InstKind::Clone { dst, src } => {
            let val = frame.share(*src);
            frame.set(*dst, val);
        }
        InstKind::Drop { src } => {
            // Consume the value, releasing its resources.
            let _ = frame.take(*src);
        }

        // -- Control flow ---------------------------------
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

        // -- Functions -------------------------------------
        InstKind::LoadFunction { dst: _, id: _ } => {
            todo!("LoadFunction: graph-level function references not yet supported at runtime");
        }
        InstKind::FunctionCall {
            dst,
            callee,
            args,
            ..
        } => {
            let result = match callee {
                Callee::Direct(id) => {
                    let is_extern =
                        matches!(lookup_function(&ctx.shared, id), Executable::Extern(_));
                    if is_extern {
                        let handler = match lookup_function(&ctx.shared, id) {
                            Executable::Extern(h) => h.clone(),
                            _ => unreachable!(),
                        };
                        let arg_vals: Vec<Value> =
                            args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                        match &handler {
                            crate::extern_fn::ExternHandler::Sync(f) => {
                                f(arg_vals, &ctx.shared.interner)?
                            }
                            crate::extern_fn::ExternHandler::Async(f) => {
                                let interner = ctx.shared.interner.clone();
                                f(arg_vals, interner).await?
                            }
                        }
                    } else {
                        let arg_vals: Args =
                            args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                        dispatch_call(ctx, id, arg_vals).await?
                    }
                }
                Callee::Indirect(val_id) => {
                    let fv = frame.take(*val_id).into_fn();
                    let call_args: Vec<Value> =
                        args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                    fn_value_call(&fv, call_args).await?
                }
            };
            frame.set(*dst, result);
        }
        InstKind::Spawn {
            dst,
            callee,
            args,
            ..
        } => {
            let callee_id = match callee {
                Callee::Direct(id) => *id,
                Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
            };
            let spawn_kind = match lookup_function(&ctx.shared, &callee_id) {
                Executable::Extern(h) => SpawnKind::Extern(h.clone()),
                Executable::Module(_) => SpawnKind::Module,
            };
            let spawn_args: Vec<Value> = args.iter().map(|a| frame.use_val(*a, val_types)).collect();
            let handle = match spawn_kind {
                SpawnKind::Extern(handler) => {
                    let interner = ctx.shared.interner.clone();
                    match &handler {
                        crate::extern_fn::ExternHandler::Sync(f) => {
                            let f = Arc::clone(f);
                            ctx.shared.executor.spawn_blocking(Box::new(move || {
                                let value = f(spawn_args, &interner)?;
                                Ok(ExecResult {
                                    value,
                                    writes: Vec::new(),
                                })
                            }))
                        }
                        crate::extern_fn::ExternHandler::Async(f) => {
                            let f = Arc::clone(f);
                            ctx.shared.executor.spawn_async(Box::pin(async move {
                                let value = f(spawn_args, interner).await?;
                                Ok(ExecResult {
                                    value,
                                    writes: Vec::new(),
                                })
                            }))
                        }
                    }
                }
                SpawnKind::Module => {
                    let child = Interpreter {
                        shared: ctx.shared.clone(),
                        entry: callee_id,
                        page: ctx.page.fork(),
                        variables: FxHashMap::default(),
                        spawn_args,
                        };
                    ctx.shared.executor.spawn_interpreter(child)
                }
            };
            frame.set(*dst, Value::Handle(Box::new(handle)));
        }
        InstKind::Eval { dst, src } => {
            let handle = match frame.take(*src) {
                Value::Handle(h) => *h,
                other => panic!("eval: expected Handle, got {other:?}"),
            };
            let result = ctx.shared.executor.eval(handle).await?;

            for w in result.writes {
                match w {
                    ContextWrite::Set { key, value } => ctx.page.set(&key, value),
                    ContextWrite::FieldPatch { key, path, value } => {
                        let path_refs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
                        ctx.page.set_field(&key, &path_refs, value);
                    }
                }
            }

            frame.set(*dst, result.value);
        }

        // -- Object/List dynamic access -------------------
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
        InstKind::ArrayIndex { dst, array, index } => {
            let val = match frame.get(*array) {
                Value::Array(items) => items[*index].share(),
                other => panic!("ArrayIndex on non-array: {other:?}"),
            };
            frame.set(*dst, val);
        }
        InstKind::ArrayGet { dst, array, index } => {
            let idx = frame.get(*index).as_int() as usize;
            let val = match frame.get(*array) {
                Value::Array(items) => items[idx].share(),
                other => panic!("ArrayGet on non-array: {other:?}"),
            };
            frame.set(*dst, val);
        }
    }
    Ok(Flow::Next)
}

// -- Function dispatch -----------------------------------------------

async fn dispatch_call(
    ctx: &mut RunContext,
    id: &QualifiedRef,
    args: Args,
) -> Result<Value, RuntimeError> {
    match lookup_function(&ctx.shared, id) {
        Executable::Module(_) => {
            let arg_values: Vec<Value> = args.into_vec();
            execute_function(ctx, id, &arg_values).await
        }
        Executable::Extern(_) => {
            panic!("extern function {id:?} reached dispatch_call; FunctionCall handles externs directly")
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
    let val_types = m.main.val_types.clone();
    let label_map = build_label_map(&m.main);
    let mut frame = Frame::new(&m.main.val_factory, label_map);
    for ((_, reg), val) in m.main.params.iter().zip(args.iter()) {
        frame.set(*reg, val.clone());
    }
    let mut projection_map: FxHashMap<ValueId, RuntimeRef> = FxHashMap::default();
    run_loop(
        ctx,
        &insts,
        &closures,
        &mut frame,
        &mut projection_map,
        &val_types,
    )
    .await
}

// -- Closure calling -------------------------------------------------

/// Execute a FnValue with the given arguments. Self-contained - uses the
/// FnValue's own shared context and forked overlay.
pub async fn fn_value_call(f: &FnValue, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let mut closure_ctx = RunContext {
        shared: f.shared.clone(),
        page: f.page.fork(),
        variables: FxHashMap::default(),
    };

    let body = &f.body;
    let label_map = build_label_map_from_insts(&body.insts);
    let mut frame = Frame::new(&body.val_factory, label_map);
    let mut projection_map = FxHashMap::default();

    for ((_, reg), cap) in body.captures.iter().zip(f.captures.iter()) {
        frame.set(*reg, cap.clone());
    }
    for ((_, reg), arg) in body.params.iter().zip(args) {
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

// -- Interpreter - thin entry point ----------------------------------

pub struct Interpreter {
    shared: InterpreterContext,
    entry: QualifiedRef,
    page: InMemoryContext,
    variables: FxHashMap<ValueId, Value>,
    spawn_args: Vec<Value>,
}

impl Interpreter {
    pub fn new(shared: InterpreterContext, entry: QualifiedRef, page: InMemoryContext) -> Self {
        Self {
            shared,
            entry,
            page,
            variables: FxHashMap::default(),
            spawn_args: Vec::new(),
        }
    }

    pub fn fork(&self, entry: QualifiedRef, args: Vec<Value>) -> Self {
        Self {
            shared: self.shared.clone(),
            entry,
            page: self.page.fork(),
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
            page: std::mem::replace(
                &mut self.page,
                InMemoryContext::empty(self.shared.interner.clone()),
            ),
            variables: std::mem::take(&mut self.variables),
        };

        let value = execute_function(&mut run_ctx, &entry, &args).await?;

        let writes = run_ctx.page.into_writes();
        Ok(ExecResult { value, writes })
    }
}

// -- Literal -> Value --------------------------------------------------

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::string(s.as_str()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Byte(b) => Value::Byte(*b),
        Literal::List(items) => Value::array(items.iter().map(literal_to_value).collect()),
        Literal::Unit => Value::Unit,
    }
}

// -- BinOp ------------------------------------------------------------

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

// -- UnaryOp ----------------------------------------------------------

fn eval_unaryop(op: UnaryOp, val: &Value) -> Result<Value, RuntimeError> {
    match (op, val) {
        (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
        (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
        (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        _ => Err(RuntimeError::unary_op_mismatch(op, val.kind())),
    }
}

