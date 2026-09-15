use std::sync::Arc;

use acvus_ast::{BinOp, Literal, UnaryOp};
use acvus_extern::Returned;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{Callee, Inst, InstKind, Label, MirBody, MirModule, RefTarget, ValueId};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner, LocalFactory, LocalVec};
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::error::RuntimeError;
use crate::journal::{ContextWrite, InMemoryContext, RuntimeContext};
use crate::runtime::{AcvusRuntime, ExternHandler};
use crate::vtable::VtableRegistry;
use crate::value::{FnValue, HandleValue, Value, VariantValue};

/// Runtime representation of a Ref instruction's target.
#[derive(Debug, Clone)]
enum RuntimeRef {
    Var { slot: ValueId, path: Vec<Astr> },
    Param { slot: ValueId, path: Vec<Astr> },
    Context { qref: QualifiedRef, path: Vec<Astr> },
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

fn field<'a>(value: &'a Value, f: Astr, interner: &Interner) -> &'a Value {
    assert!(value.is_object(), "field load on non-object: {value:?}");
    // SAFETY: is_object checked the vtable id.
    unsafe { value.as_object() }
        .get(&f)
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(f)))
}

fn walk_path<'a>(mut value: &'a Value, path: &[Astr], interner: &Interner) -> &'a Value {
    for f in path {
        value = field(value, *f, interner);
    }
    value
}

/// Load a value from a RuntimeRef.
fn load_ref(rt_ref: &RuntimeRef, ctx: &RunContext) -> Value {
    let interner = &ctx.shared.interner;
    match rt_ref {
        RuntimeRef::Var { slot, path } | RuntimeRef::Param { slot, path } => {
            let root = ctx
                .variables
                .get(slot)
                .unwrap_or_else(|| panic!("undefined variable slot {:?}", slot));
            walk_path(root, path, interner).deep_clone()
        }
        RuntimeRef::Context { qref, path } => {
            let name = ctx
                .shared
                .context_names
                .get(qref)
                .unwrap_or_else(|| panic!("context load: no name for {:?}", qref));
            let key = interner.resolve(*name);
            let root = ctx
                .page
                .get(key)
                .unwrap_or_else(|| panic!("context load: undefined context '{}'", key));
            let out = walk_path(&root, path, interner).deep_clone();
            drop(root);
            out
        }
    }
}

/// Set a nested field path on an object value, in place.
fn store_path(target: &mut Value, path: &[Astr], value: Value, interner: &Interner) {
    assert!(!path.is_empty(), "store_path called with empty path");
    assert!(target.is_object(), "field store on non-object: {target:?}");
    // SAFETY: is_object checked the vtable id.
    let map = unsafe { target.as_object_mut() };
    if path.len() == 1 {
        if let Some(old) = map.insert(path[0], value) {
            drop(old);
        }
        return;
    }
    let inner = map
        .get_mut(&path[0])
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(path[0])));
    store_path(inner, &path[1..], value, interner);
}

/// Store a value through a RuntimeRef.
fn store_ref(rt_ref: &RuntimeRef, ctx: &mut RunContext, value: Value) {
    let interner = ctx.shared.interner.clone();
    match rt_ref {
        RuntimeRef::Var { slot, path } if path.is_empty() => {
            if let Some(old) = ctx.variables.insert(*slot, value) {
                drop(old);
            }
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

    /// An explicit copy through the value's vtable.
    #[inline]
    fn share(&self, id: ValueId) -> Value {
        self.get(id).deep_clone()
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

    fn jump(
        &mut self,
        insts: &[Inst],
        label: &Label,
        args: &[ValueId],
        val_types: &FxHashMap<ValueId, Ty>,
    ) -> usize {
        let target = self.resolve_label(label);
        self.bind_block_params(insts, target, args, val_types);
        target
    }

    fn jump_if(
        &mut self,
        insts: &[Inst],
        cond: ValueId,
        then: (&Label, &[ValueId]),
        else_: (&Label, &[ValueId]),
        val_types: &FxHashMap<ValueId, Ty>,
    ) -> usize {
        let (label, args) = if self.get(cond).as_bool() { then } else { else_ };
        self.jump(insts, label, args, val_types)
    }

    fn resolve_label(&self, label: &Label) -> usize {
        *self
            .label_map
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}"))
    }

    /// A jump moves its arguments into the params: a move-only argument
    /// leaves its register, so one owner remains.
    fn bind_block_params(
        &mut self,
        insts: &[Inst],
        target: usize,
        args: &[ValueId],
        val_types: &FxHashMap<ValueId, Ty>,
    ) {
        if let InstKind::BlockLabel { params, .. } = &insts[target].kind {
            let values: Vec<Value> = args.iter().map(|a| self.use_val(*a, val_types)).collect();
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
    Extern(ExternHandler),
    Module,
}

/// Control flow after executing one instruction.
enum Flow {
    Next,
    Jump(usize),
    Return(Value),
}

// -- Public types ----------------------------------------------------

/// Result of execution - return value + context mutations.
pub struct ExecResult {
    pub value: Value,
    pub writes: Vec<ContextWrite>,
    /// The borrowed places' values after the call (RFC-0015).
    pub lent: Vec<Value>,
}

/// A single executable unit - MIR module or extern function.
pub enum Executable {
    Module(MirModule),
    Extern(crate::runtime::ExternEntry),
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
    pub vtables: Arc<VtableRegistry>,
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
            vtables: Arc::new(VtableRegistry::default()),
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

    fn runtime(&self) -> AcvusRuntime {
        AcvusRuntime(self.clone())
    }
}

// -- RunContext - mutable state bundle for run_loop -------------------

/// Per-execution mutable state passed through the run_loop call chain.
struct RunContext {
    shared: InterpreterContext,
    /// The run's page, shared with every closure the run makes.
    page: Arc<InMemoryContext>,
    variables: FxHashMap<ValueId, Value>,
}

// -- Standalone helpers (extracted from Interpreter methods) ----------

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
    Ok(Value::unit())
}

fn type_of<'a>(val_types: &'a FxHashMap<ValueId, Ty>, id: ValueId) -> &'a Ty {
    val_types
        .get(&id)
        .unwrap_or_else(|| panic!("no type for value {id:?}"))
}

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
            frame.set(*dst, Value::unit());
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
            let ty = type_of(val_types, *left);
            let result = eval_binop(*op, ty, frame.get(*left), frame.get(*right))?;
            frame.set(*dst, result);
        }
        InstKind::UnaryOp { dst, op, operand } => {
            let ty = type_of(val_types, *operand);
            let result = eval_unaryop(*op, ty, frame.get(*operand));
            frame.set(*dst, result);
        }

        // -- Field / Index access -------------------------
        InstKind::FieldGet {
            dst,
            object,
            field: f,
            rest,
        } => {
            let interner = &ctx.shared.interner;
            let root = field(frame.get(*object), *f, interner);
            let val = walk_path(root, rest, interner).deep_clone();
            frame.set(*dst, val);
        }
        InstKind::FieldSet {
            dst,
            object,
            field: f,
            rest,
            value,
        } => {
            let mut obj = frame.share(*object);
            let new_val = frame.share(*value);
            let mut path = Vec::with_capacity(1 + rest.len());
            path.push(*f);
            path.extend_from_slice(rest);
            store_path(&mut obj, &path, new_val, &ctx.shared.interner);
            frame.set(*dst, obj);
        }
        InstKind::TupleIndex { dst, tuple, index } => {
            // SAFETY: the type checker admits only a tuple value here.
            let val = unsafe { frame.get(*tuple).as_tuple() }[*index].deep_clone();
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
                    page: Arc::clone(&ctx.page),
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
            let src_val = frame.get(*src);
            // SAFETY: is_variant checked the vtable id.
            let matches = src_val.is_variant() && unsafe { src_val.as_variant() }.tag == *tag;
            frame.set(*dst, Value::bool_(matches));
        }
        InstKind::UnwrapVariant { dst, src } => {
            let src_val = frame.take(*src);
            assert!(src_val.is_variant(), "UnwrapVariant on non-variant: {src_val:?}");
            // SAFETY: is_variant checked the vtable id.
            let variant = unsafe { src_val.materialize::<VariantValue>() };
            let val = match variant.payload {
                Some(p) => *p,
                None => Value::unit(),
            };
            frame.set(*dst, val);
        }

        // -- Pattern testing ------------------------------
        InstKind::TestLiteral { dst, src, value } => {
            let src_val = frame.get(*src);
            let matches = match value {
                Literal::Int(b) => src_val.as_int() == *b,
                Literal::Float(b) => src_val.as_float() == *b,
                Literal::Bool(b) => src_val.as_bool() == *b,
                Literal::Byte(b) => src_val.as_byte() == *b,
                // SAFETY: the type checker matches a string literal against a string.
                Literal::String(b) => (unsafe { src_val.as_str() }) == b.as_str(),
                Literal::Unit => true,
                Literal::List(_) => panic!("TestLiteral on a list literal"),
            };
            frame.set(*dst, Value::bool_(matches));
        }
        InstKind::TestObjectKey { dst, src, key } => {
            let src_val = frame.get(*src);
            // SAFETY: is_object checked the vtable id.
            let has = src_val.is_object() && unsafe { src_val.as_object() }.contains_key(key);
            frame.set(*dst, Value::bool_(has));
        }

        InstKind::Drop { src } => {
            drop(frame.take(*src));
        }

        // -- Control flow ---------------------------------
        InstKind::BlockLabel { .. } => {}
        InstKind::Jump { label, args } => {
            let target = frame.jump(insts, label, args, val_types);
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
                val_types,
            );
            return Ok(Flow::Jump(target));
        }
        InstKind::Return { value, .. } => {
            return Ok(Flow::Return(frame.take(*value)));
        }
        InstKind::Merge { dst, .. } => {
            frame.set(*dst, Value::unit());
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
            callee_ty,
            args,
            order,
            lent,
        } => {
            if let Some(edge) = order {
                frame.set(edge.after, Value::unit());
            }
            let returned = match callee {
                Callee::Direct(id) => {
                    let is_extern =
                        matches!(lookup_function(&ctx.shared, id), Executable::Extern(_));
                    if is_extern {
                        let entry = match lookup_function(&ctx.shared, id) {
                            Executable::Extern(h) => h.clone(),
                            _ => unreachable!(),
                        };
                        let handler = entry.select(callee_ty)?.clone();
                        let arg_vals: Vec<Value> =
                            args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                        match &handler {
                            ExternHandler::Sync(f) => f(&ctx.shared.runtime(), arg_vals)?,
                            ExternHandler::Async(f) => f(ctx.shared.runtime(), arg_vals).await?,
                        }
                    } else {
                        let arg_vals: Args =
                            args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                        Returned::value(dispatch_call(ctx, id, arg_vals).await?)
                    }
                }
                Callee::Indirect(val_id) => {
                    // SAFETY: the type checker admits only a closure value here.
                    let fv = unsafe { frame.take(*val_id).materialize::<FnValue>() };
                    let call_args: Vec<Value> =
                        args.iter().map(|a| frame.use_val(*a, val_types)).collect();
                    Returned::value(fn_value_call(&fv, call_args).await?)
                }
            };
            frame.set(*dst, returned.value);
            give_back(frame, lent, returned.lent)?;
        }
        InstKind::Spawn {
            dst,
            callee,
            callee_ty,
            args,
            order: _,
        } => {
            let callee_id = match callee {
                Callee::Direct(id) => *id,
                Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
            };
            let spawn_kind = match lookup_function(&ctx.shared, &callee_id) {
                Executable::Extern(entry) => SpawnKind::Extern(entry.select(callee_ty)?.clone()),
                Executable::Module(_) => SpawnKind::Module,
            };
            let spawn_args: Vec<Value> =
                args.iter().map(|a| frame.use_val(*a, val_types)).collect();
            let handle = match spawn_kind {
                SpawnKind::Extern(handler) => {
                    let rt = ctx.shared.runtime();
                    match &handler {
                        ExternHandler::Sync(f) => {
                            let f = Arc::clone(f);
                            ctx.shared.executor.spawn_blocking(Box::new(move || {
                                let returned = f(&rt, spawn_args)?;
                                Ok(ExecResult {
                                    value: returned.value,
                                    writes: Vec::new(),
                                    lent: returned.lent,
                                })
                            }))
                        }
                        ExternHandler::Async(f) => {
                            let f = Arc::clone(f);
                            ctx.shared.executor.spawn_async(Box::pin(async move {
                                let returned = f(rt, spawn_args).await?;
                                Ok(ExecResult {
                                    value: returned.value,
                                    writes: Vec::new(),
                                    lent: returned.lent,
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
            frame.set(*dst, Value::handle(handle));
        }
        InstKind::Eval {
            dst,
            src,
            order,
            lent,
        } => {
            if let Some(o) = order {
                frame.set(*o, Value::unit());
            }
            // SAFETY: the type checker admits only a handle value here.
            let handle = unsafe { frame.take(*src).materialize::<HandleValue>() };
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
            give_back(frame, lent, result.lent)?;
        }

        // -- Object/List dynamic access -------------------
        InstKind::ObjectGet { dst, object, key } => {
            let val = field(frame.get(*object), *key, &ctx.shared.interner).deep_clone();
            frame.set(*dst, val);
        }
        InstKind::ArrayIndex { dst, array, index } => {
            // SAFETY: the type checker admits only an array value here.
            let val = unsafe { frame.get(*array).as_array() }[*index].deep_clone();
            frame.set(*dst, val);
        }
        InstKind::ArrayGet { dst, array, index } => {
            let idx = frame.get(*index).as_int() as usize;
            // SAFETY: the type checker admits only an array value here.
            let val = unsafe { frame.get(*array).as_array() }[idx].deep_clone();
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
        Executable::Module(_) => execute_function(ctx, id, args.into_vec()).await,
        Executable::Extern(_) => {
            panic!(
                "extern function {id:?} reached dispatch_call; FunctionCall handles externs directly"
            )
        }
    }
}

/// Execute a MIR module function by QualifiedRef with explicit args.
async fn execute_function(
    ctx: &mut RunContext,
    id: &QualifiedRef,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    let m = lookup_module(&ctx.shared, id);
    let insts: Arc<[Inst]> = m.main.insts.clone().into();
    let closures = m.closures.clone();
    let val_types = m.main.val_types.clone();
    let label_map = build_label_map(&m.main);
    let mut frame = Frame::new(&m.main.val_factory, label_map);
    for ((_, reg), val) in m.main.params.iter().zip(args) {
        frame.set(*reg, val);
    }
    if let Some(order) = m.main.order_param {
        frame.set(order, Value::unit());
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

/// Put the values a call gave back for its borrowed places into the
/// registers the instruction names for them.
fn give_back(frame: &mut Frame, lent: &[ValueId], values: Vec<Value>) -> Result<(), RuntimeError> {
    if lent.len() != values.len() {
        return Err(RuntimeError::internal(format!(
            "call lent {} places but gave back {}",
            lent.len(),
            values.len()
        )));
    }
    for (reg, value) in lent.iter().zip(values) {
        frame.set(*reg, value);
    }
    Ok(())
}

// -- Closure calling -------------------------------------------------

pub async fn fn_value_call(f: &FnValue, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let mut closure_ctx = RunContext {
        shared: f.shared.clone(),
        page: Arc::clone(&f.page),
        variables: FxHashMap::default(),
    };

    let body = &f.body;
    let label_map = build_label_map_from_insts(&body.insts);
    let mut frame = Frame::new(&body.val_factory, label_map);
    let mut projection_map = FxHashMap::default();

    for ((_, reg), cap) in body.captures.iter().zip(f.captures.iter()) {
        frame.set(*reg, cap.deep_clone());
    }
    for ((_, reg), arg) in body.params.iter().zip(args) {
        frame.set(*reg, arg);
    }
    if let Some(order) = body.order_param {
        frame.set(order, Value::unit());
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

        let empty = InMemoryContext::empty(self.shared.interner.clone());
        let mut run_ctx = RunContext {
            shared: self.shared.clone(),
            page: Arc::new(std::mem::replace(&mut self.page, empty)),
            variables: std::mem::take(&mut self.variables),
        };

        let value = execute_function(&mut run_ctx, &entry, args).await?;

        let writes = run_ctx.page.take_writes();
        Ok(ExecResult {
            value,
            writes,
            lent: Vec::new(),
        })
    }
}

// -- Literal -> Value --------------------------------------------------

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::int(*n),
        Literal::Float(f) => Value::float(*f),
        Literal::String(s) => Value::string(s.as_str()),
        Literal::Bool(b) => Value::bool_(*b),
        Literal::Byte(b) => Value::byte(*b),
        Literal::List(items) => Value::array(items.iter().map(literal_to_value).collect()),
        Literal::Unit => Value::unit(),
    }
}

// -- BinOp ------------------------------------------------------------

/// The operands' type is the left operand's MIR type; the type checker
/// makes both sides agree.
fn eval_binop(
    op: BinOp,
    ty: &Ty,
    left: &Value,
    right: &Value,

) -> Result<Value, RuntimeError> {
    match ty {
        Ty::Int => {
            let (a, b) = (left.as_int(), right.as_int());
            Ok(match op {
                BinOp::Add => Value::int(a.wrapping_add(b)),
                BinOp::Sub => Value::int(a.wrapping_sub(b)),
                BinOp::Mul => Value::int(a.wrapping_mul(b)),
                BinOp::Div => {
                    if b == 0 {
                        return Err(RuntimeError::division_by_zero());
                    }
                    Value::int(a / b)
                }
                BinOp::Mod => {
                    if b == 0 {
                        return Err(RuntimeError::division_by_zero());
                    }
                    Value::int(a % b)
                }
                BinOp::Eq => Value::bool_(a == b),
                BinOp::Neq => Value::bool_(a != b),
                BinOp::Lt => Value::bool_(a < b),
                BinOp::Gt => Value::bool_(a > b),
                BinOp::Lte => Value::bool_(a <= b),
                BinOp::Gte => Value::bool_(a >= b),
                BinOp::BitAnd => Value::int(a & b),
                BinOp::BitOr => Value::int(a | b),
                BinOp::Xor => Value::int(a ^ b),
                BinOp::Shl => Value::int(a << b),
                BinOp::Shr => Value::int(a >> b),
                BinOp::And | BinOp::Or => panic!("And/Or on Int"),
            })
        }
        Ty::Float => {
            let (a, b) = (left.as_float(), right.as_float());
            Ok(match op {
                BinOp::Add => Value::float(a + b),
                BinOp::Sub => Value::float(a - b),
                BinOp::Mul => Value::float(a * b),
                BinOp::Div => Value::float(a / b),
                BinOp::Mod => Value::float(a % b),
                BinOp::Eq => Value::bool_(a == b),
                BinOp::Neq => Value::bool_(a != b),
                BinOp::Lt => Value::bool_(a < b),
                BinOp::Gt => Value::bool_(a > b),
                BinOp::Lte => Value::bool_(a <= b),
                BinOp::Gte => Value::bool_(a >= b),
                _ => panic!("unsupported float binop {op:?}"),
            })
        }
        Ty::String => {
            // SAFETY: the type checker admits only strings here.
            let (a, b) = unsafe { (left.as_str(), right.as_str()) };
            Ok(match op {
                BinOp::Add => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Value::string(s)
                }
                BinOp::Eq => Value::bool_(a == b),
                BinOp::Neq => Value::bool_(a != b),
                _ => panic!("unsupported string binop {op:?}"),
            })
        }
        Ty::Bool => {
            let (a, b) = (left.as_bool(), right.as_bool());
            Ok(match op {
                BinOp::And => Value::bool_(a && b),
                BinOp::Or => Value::bool_(a || b),
                BinOp::Eq => Value::bool_(a == b),
                BinOp::Neq => Value::bool_(a != b),
                BinOp::Xor => Value::bool_(a ^ b),
                _ => panic!("unsupported bool binop {op:?}"),
            })
        }
        other => panic!("binop {op:?} on {other:?}"),
    }
}

// -- UnaryOp ----------------------------------------------------------

fn eval_unaryop(op: UnaryOp, ty: &Ty, val: &Value) -> Value {
    match (op, ty) {
        (UnaryOp::Neg, Ty::Int) => Value::int(-val.as_int()),
        (UnaryOp::Neg, Ty::Float) => Value::float(-val.as_float()),
        (UnaryOp::Not, Ty::Bool) => Value::bool_(!val.as_bool()),
        (op, other) => panic!("unary {op:?} on {other:?}"),
    }
}
