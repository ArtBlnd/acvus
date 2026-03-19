use std::cmp::Ordering;

use std::sync::Arc;

use futures::future::BoxFuture;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::builtins::BuiltinId;
use acvus_mir::ir::{CastKind, Inst, InstKind, Label, MirBody, MirModule, ValueId};
use acvus_utils::Astr;
use acvus_utils::Interner;
use acvus_utils::TrackedDeque;
use rustc_hash::FxHashMap;

use acvus_mir::ty::{Effect, Ty};
use crate::builtins;
use crate::error::RuntimeError;
use crate::iter::{IterHandle, IterOp, IterRepr, SequenceChain};
use crate::value::{FnValue, LazyValue, PureValue, Tuple, TypedValue, Value};
use acvus_utils::{Coroutine, Stepped, YieldHandle};


pub struct Interpreter {
    interner: Interner,
    module: MirModule,
    variables: FxHashMap<Astr, Arc<Value>>,
}

// ---------------------------------------------------------------------------
// Frame — val store + sync instruction execution
// ---------------------------------------------------------------------------

struct Frame {
    vals: Vec<Option<Arc<Value>>>,
    label_map: FxHashMap<Label, usize>,
}

impl Frame {
    fn new(val_count: u32, label_map: FxHashMap<Label, usize>) -> Self {
        Self {
            vals: vec![None; val_count as usize],
            label_map,
        }
    }

    fn set(&mut self, id: ValueId, value: Arc<Value>) {
        self.vals[id.0 as usize] = Some(value);
    }

    fn set_new(&mut self, id: ValueId, value: Value) {
        self.vals[id.0 as usize] = Some(Arc::new(value));
    }

    fn get(&self, id: ValueId) -> &Value {
        self.vals[id.0 as usize]
            .as_deref()
            .unwrap_or_else(|| panic!("Val({}) not yet defined", id.0))
    }

    fn take(&self, id: ValueId) -> Arc<Value> {
        Arc::clone(
            self.vals[id.0 as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("Val({}) not yet defined", id.0)),
        )
    }

    fn take_owned(&self, id: ValueId) -> Value {
        Arc::unwrap_or_clone(self.take(id))
    }

    fn collect_args(&self, args: &[ValueId]) -> Vec<Value> {
        args.iter().map(|v| self.take_owned(*v)).collect()
    }

    fn collect_args_arc(&self, args: &[ValueId]) -> Vec<Arc<Value>> {
        args.iter().map(|v| self.take(*v)).collect()
    }

    // -- control flow ---------------------------------------------------------

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
        let cond_val = *self.get(cond).expect_ref::<bool>("jump_if");
        let (label, args) = if cond_val { then } else { else_ };
        let target = self.resolve_label(label);
        self.bind_block_params(insts, target, args);
        target
    }

    fn resolve_label(&self, label: &Label) -> usize {
        *self
            .label_map
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {:?}", label))
    }

    fn bind_block_params(&mut self, insts: &[Inst], target: usize, args: &[ValueId]) {
        if let InstKind::BlockLabel { params, .. } = &insts[target].kind {
            let arg_values = self.collect_args_arc(args);
            for (param, val) in params.iter().zip(arg_values) {
                self.set(*param, val);
            }
        }
    }

}

// ---------------------------------------------------------------------------
// Label map
// ---------------------------------------------------------------------------

fn build_label_map(body: &MirBody) -> FxHashMap<Label, usize> {
    body.insts
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some((*label, i)),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Interpreter — ownership-passing async exec loop
//
// All async methods take `this: Self` by value and return it back.
// This makes every future `Send + 'static` without unsafe.
// ---------------------------------------------------------------------------

impl Interpreter {
    pub fn new(interner: &Interner, module: MirModule) -> Self {
        Self {
            interner: interner.clone(),
            module,
            variables: FxHashMap::default(),
        }
    }

    pub fn execute(self) -> Coroutine<TypedValue, RuntimeError> {
        acvus_utils::coroutine(|handle| async move {
            crate::verify::verify_module(&self.module);
            crate::set_interner_ctx(&self.interner);
            let insts = self.module.main.insts.clone();
            let val_types = self.module.main.val_types.clone();
            let label_map = build_label_map(&self.module.main);
            let frame = Frame::new(self.module.main.val_count, label_map);
            Self::run(self, insts, val_types, frame, &handle).await?;
            Ok(())
        })
    }

    /// Drive the coroutine to completion with a pre-built context map.
    /// Returns all emitted values. Panics on missing context or extern calls.
    pub async fn execute_with_context(self, context: FxHashMap<Astr, TypedValue>) -> Vec<TypedValue> {
        let interner = self.interner.clone();
        let mut coroutine = self.execute();
        let mut emits = Vec::new();
        loop {
            match coroutine.resume().await {
                Stepped::Emit(value) => emits.push(value),
                Stepped::NeedContext(request) => {
                    let name = request.name();
                    let v = context
                        .get(&name)
                        .unwrap_or_else(|| {
                            panic!("undefined context @{}", interner.resolve(name))
                        });
                    request.resolve(v.clone());
                }
                Stepped::NeedExternCall(_) => {
                    panic!("unexpected extern call in execute_with_context");
                }
                Stepped::Done => break,
                Stepped::Error(e) => panic!("runtime error: {e}"),
            }
        }
        emits
    }

    // -- core exec loop -------------------------------------------------------

    fn run<'a>(
        this: Self,
        insts: Vec<Inst>,
        val_types: FxHashMap<ValueId, Ty>,
        frame: Frame,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Frame, Option<Value>), RuntimeError>> {
        Box::pin(Self::run_inner(this, insts, val_types, frame, handle))
    }

    async fn run_inner(
        mut this: Self,
        insts: Vec<Inst>,
        val_types: FxHashMap<ValueId, Ty>,
        mut frame: Frame,
        handle: &YieldHandle<TypedValue>,
    ) -> Result<(Self, Frame, Option<Value>), RuntimeError> {
        let mut pc = 0;
        while pc < insts.len() {
            match &insts[pc].kind {
                // -- yield --
                InstKind::Yield(v) => {
                    let val = frame.take(*v);
                    let ty = val_types[v].clone();
                    handle.yield_val(TypedValue::new_shared(val, ty)).await;
                }

                // -- constants / constructors --
                InstKind::Const { dst, value } => {
                    frame.set_new(*dst, literal_to_value(value));
                }
                InstKind::MakeDeque { dst, elements } => {
                    let items = frame.collect_args(elements);
                    frame.set_new(*dst, Value::deque(TrackedDeque::from_vec(items)));
                }
                InstKind::MakeObject { dst, fields } => {
                    let obj: FxHashMap<Astr, Value> = fields
                        .iter()
                        .map(|(k, v)| (*k, frame.take_owned(*v)))
                        .collect();
                    frame.set_new(*dst, Value::object(obj));
                }
                InstKind::MakeRange {
                    dst,
                    start,
                    end,
                    kind,
                } => {
                    let s = *frame.get(*start).expect_ref::<i64>("MakeRange start");
                    let e = *frame.get(*end).expect_ref::<i64>("MakeRange end");
                    frame.set_new(
                        *dst,
                        Value::range(s, e, matches!(kind, RangeKind::InclusiveEnd)),
                    );
                }
                InstKind::MakeTuple { dst, elements } => {
                    let items = frame.collect_args(elements);
                    frame.set_new(*dst, Value::tuple(items));
                }
                InstKind::MakeClosure {
                    dst,
                    body,
                    captures,
                } => {
                    let captured = frame.collect_args_arc(captures);
                    let closure_body = Arc::clone(
                        this.module.closures.get(body)
                            .unwrap_or_else(|| panic!("closure body not found: {:?}", body))
                    );
                    frame.set_new(
                        *dst,
                        Value::closure(FnValue {
                            body: closure_body,
                            captures: captured,
                        }),
                    );
                }

                // -- arithmetic / logic --
                InstKind::BinOp {
                    dst,
                    op,
                    left,
                    right,
                } => {
                    let result = eval_binop(*op, frame.get(*left), frame.get(*right))?;
                    frame.set_new(*dst, result);
                }
                InstKind::UnaryOp { dst, op, operand } => {
                    frame.set_new(*dst, eval_unaryop(*op, frame.get(*operand))?);
                }

                // -- access --
                InstKind::FieldGet { dst, object, field } => {
                    let obj = frame.get(*object).expect_ref::<FxHashMap<Astr, Value>>("field_get");
                    let v = obj.get(field)
                        .ok_or_else(|| RuntimeError::missing_field(this.interner.resolve(*field)))?
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ObjectGet { dst, object, key } => {
                    let obj = frame.get(*object).expect_ref::<FxHashMap<Astr, Value>>("object_get");
                    let v = obj.get(key)
                        .ok_or_else(|| RuntimeError::missing_field(this.interner.resolve(*key)))?
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::TupleIndex { dst, tuple, index } => {
                    let elems = frame.get(*tuple).expect_ref::<Tuple>("tuple_index");
                    let v = elems.0.get(*index)
                        .ok_or_else(|| RuntimeError::index_out_of_bounds(*index as i64, elems.0.len()))?
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ListIndex { dst, list, index } => {
                    let items = frame.get(*list).expect_ref::<[Value]>("list_index");
                    let i = if *index >= 0 {
                        *index as usize
                    } else {
                        (items.len() as i32 + *index) as usize
                    };
                    let v = items.get(i)
                        .ok_or_else(|| RuntimeError::index_out_of_bounds(*index as i64, items.len()))?
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ListGet { dst, list, index } => {
                    let items = frame.get(*list).expect_ref::<[Value]>("list_get");
                    let idx = *frame.get(*index).expect_ref::<i64>("list_get_index");
                    let v = items.get(idx as usize)
                        .ok_or_else(|| RuntimeError::index_out_of_bounds(idx, items.len()))?
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ListSlice {
                    dst,
                    list,
                    skip_head,
                    skip_tail,
                } => {
                    let items = frame.get(*list).expect_ref::<[Value]>("list_slice");
                    let end = items.len().saturating_sub(*skip_tail);
                    let start = (*skip_head).min(end);
                    frame.set_new(*dst, Value::list(items[start..end].to_vec()));
                }

                // -- variant --
                InstKind::MakeVariant { dst, tag, payload } => {
                    let p = payload.as_ref().map(|v| Box::new(frame.take_owned(*v)));
                    frame.set_new(
                        *dst,
                        Value::variant(*tag, p),
                    );
                }
                InstKind::TestVariant { dst, src, tag } => {
                    let Value::Lazy(LazyValue::Variant { tag: t, .. }) = frame.get(*src) else {
                        panic!("TestVariant: expected Variant, got {:?}", frame.get(*src));
                    };
                    frame.set_new(*dst, Value::bool_(*t == *tag));
                }
                InstKind::UnwrapVariant { dst, src } => {
                    let Value::Lazy(LazyValue::Variant {
                        payload: Some(inner),
                        ..
                    }) = frame.get(*src)
                    else {
                        panic!(
                            "UnwrapVariant: expected Variant with payload, got {:?}",
                            frame.get(*src)
                        );
                    };
                    frame.set_new(*dst, *inner.clone());
                }

                // -- pattern testing --
                InstKind::TestLiteral { dst, src, value } => {
                    let eq = frame.get(*src).structural_eq(&literal_to_value(value));
                    frame.set_new(*dst, Value::bool_(eq));
                }
                InstKind::TestListLen {
                    dst,
                    src,
                    min_len,
                    exact,
                } => {
                    let items = frame.get(*src).expect_ref::<[Value]>("test_list_len");
                    let ok = if *exact {
                        items.len() == *min_len
                    } else {
                        items.len() >= *min_len
                    };
                    frame.set_new(*dst, Value::bool_(ok));
                }
                InstKind::TestObjectKey { dst, src, key } => {
                    let ok = frame.get(*src).expect_ref::<FxHashMap<Astr, Value>>("test_object_key").contains_key(key);
                    frame.set_new(*dst, Value::bool_(ok));
                }
                InstKind::TestRange {
                    dst,
                    src,
                    start,
                    end,
                    kind,
                } => {
                    let n = *frame.get(*src).expect_ref::<i64>("test_range");
                    let ok = match kind {
                        RangeKind::Exclusive => n >= *start && n < *end,
                        RangeKind::InclusiveEnd => n >= *start && n <= *end,
                        RangeKind::ExclusiveStart => n > *start && n <= *end,
                    };
                    frame.set_new(*dst, Value::bool_(ok));
                }

                // -- iterator step (async) --
                InstKind::IterStep { dst, src } => {
                    let ih = frame.take_owned(*src).expect::<IterHandle>("IterStep");
                    let result;
                    (this, result) = Self::exec_next(this, ih, handle).await?;
                    let value = match result {
                        Some((item, rest)) => {
                            let some_tag = this.interner.intern("Some");
                            Value::variant(
                                some_tag,
                                Some(Box::new(Value::tuple(vec![item, Value::iterator(rest)]))),
                            )
                        }
                        None => {
                            let none_tag = this.interner.intern("None");
                            Value::variant(none_tag, None)
                        }
                    };
                    frame.set_new(*dst, value);
                }

                // -- context / variable I/O --
                InstKind::ContextLoad { dst, name } => {
                    let typed = handle.request_context(*name).await;
                    frame.set(*dst, Arc::new(typed.into_inner()));
                }
                InstKind::VarLoad { dst, name } => {
                    let v = this.variables.get(name).unwrap_or_else(|| {
                        panic!(
                            "VarLoad: undefined variable ${}",
                            name.display(&this.interner)
                        )
                    });
                    frame.set(*dst, Arc::clone(v));
                }
                InstKind::VarStore { name, src } => {
                    this.variables.insert(*name, frame.take(*src));
                }

                // -- calls (async, ownership-passing) --
                InstKind::BuiltinCall { dst, builtin, args } => {
                    let arg_values = frame.collect_args(args);
                    let arg_types: Vec<Ty> = args.iter()
                        .map(|a| val_types[a].clone())
                        .collect();
                    let result;
                    (this, result) =
                        Self::exec_builtin(this, *builtin, arg_values, arg_types, handle).await?;
                    frame.set_new(*dst, result);
                }
                InstKind::ExternCall { dst, name, args } => {
                    let typed_args: Vec<TypedValue> = args.iter().map(|a| {
                        TypedValue::new_shared(frame.take(*a), val_types[a].clone())
                    }).collect();
                    let result = handle.request_extern_call(*name, typed_args).await;
                    frame.set(*dst, Arc::new(result.into_inner()));
                }
                InstKind::ClosureCall { dst, closure, args } => {
                    let callee = frame.take_owned(*closure);
                    match callee {
                        Value::Lazy(LazyValue::Fn(fn_val)) => {
                            let arg_values = frame.collect_args_arc(args);
                            let result;
                            (this, result) = Self::call_closure(this, fn_val, arg_values, handle).await?;
                            frame.set_new(*dst, result);
                        }
                        Value::Lazy(LazyValue::ExternFn(name)) => {
                            let typed_args: Vec<TypedValue> = args.iter().map(|a| {
                                TypedValue::new_shared(frame.take(*a), val_types[a].clone())
                            }).collect();
                            let result = handle.request_extern_call(name, typed_args).await;
                            frame.set(*dst, Arc::new(result.into_inner()));
                        }
                        _ => panic!("ClosureCall: expected Fn or ExternFn, got {callee:?}"),
                    }
                }

                // -- control flow --
                InstKind::BlockLabel { .. } => {}
                InstKind::Jump { label, args } => {
                    pc = frame.jump(&insts, label, args);
                    continue;
                }
                InstKind::JumpIf {
                    cond,
                    then_label,
                    then_args,
                    else_label,
                    else_args,
                } => {
                    pc = frame.jump_if(
                        &insts,
                        *cond,
                        (then_label, then_args),
                        (else_label, else_args),
                    );
                    continue;
                }
                InstKind::Return(val) => {
                    let v = frame.take_owned(*val);
                    return Ok((this, frame, Some(v)));
                }
                InstKind::Nop => {}

                // -- type coercion --
                InstKind::Cast { dst, src, kind } => {
                    let value = frame.take_owned(*src);
                    let result = exec_cast(*kind, value, &val_types[src]);
                    frame.set_new(*dst, result);
                }

                InstKind::Poison { .. } => {
                    panic!("reached poison value: typechecker should have prevented this");
                }
            }
            pc += 1;
        }
        Ok((this, frame, None))
    }

    // -- builtin dispatch -----------------------------------------------------

    async fn exec_builtin<'a>(
        this: Self,
        id: BuiltinId,
        args: Vec<Value>,
        arg_types: Vec<Ty>,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        let typed_args: Vec<TypedValue> = args
            .into_iter()
            .zip(arg_types)
            .map(|(v, ty)| TypedValue::new(v, ty))
            .collect();

        // Try sync registry first
        if let Some(exec_fn) = builtins::get_builtin_impl(id) {
            let result = exec_fn(typed_args)?;
            return Ok((this, result.into_inner()));
        }

        // Try async builtins (hof.rs)
        if let Some((this, value)) =
            builtins::hof::dispatch(this, id, typed_args, handle).await?
        {
            return Ok((this, value));
        }

        unreachable!("builtin {id:?} not in sync registry and not handled as async")
    }

    // -- incremental iterator execution (thunk-based) -------------------------

    /// Pull one element from an IterHandle.
    ///
    /// Returns `None` if exhausted, `Some((item, rest))` otherwise.
    /// For Pure iterators, the result is memoized in the IterHandle state.
    pub fn exec_next<'a>(
        this: Self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Option<(Value, IterHandle)>), RuntimeError>> {
        Box::pin(Self::exec_next_inner(this, ih, handle))
    }

    async fn exec_next_inner<'a>(
        this: Self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Option<(Value, IterHandle)>), RuntimeError> {
        let state = ih.get_state();
        match state {
            IterRepr::Done => Ok((this, None)),

            IterRepr::Evaluated { head, tail } => {
                Ok((this, Some((head, tail))))
            }

            IterRepr::Chain { first, second } => {
                let (this, result) = Self::exec_next(this, first, handle).await?;
                match result {
                    Some((item, rest_first)) => {
                        let rest = rest_first.chain(second);
                        Ok((this, Some((item, rest))))
                    }
                    None => {
                        // first exhausted, continue with second
                        Self::exec_next(this, second, handle).await
                    }
                }
            }

            IterRepr::Wrapped { inner, op } => {
                // Treat as a single-op pipeline on top of inner
                let (this, result) = Self::exec_next_apply_op(
                    this, inner, op, ih.effect(), handle,
                ).await?;

                if ih.effect() == Effect::Pure {
                    match &result {
                        Some((item, tail)) => {
                            ih.set_state(IterRepr::Evaluated {
                                head: item.clone(),
                                tail: tail.clone(),
                            });
                        }
                        None => {
                            ih.set_state(IterRepr::Done);
                        }
                    }
                }

                Ok((this, result))
            }

            IterRepr::Suspended { source, ops, offset } => {
                let (this, result) = Self::exec_next_suspended(
                    this, source, ops, offset, ih.effect(), handle,
                ).await?;

                // Pure memo
                if ih.effect() == Effect::Pure {
                    match &result {
                        Some((item, tail)) => {
                            ih.set_state(IterRepr::Evaluated {
                                head: item.clone(),
                                tail: tail.clone(),
                            });
                        }
                        None => {
                            ih.set_state(IterRepr::Done);
                        }
                    }
                }

                Ok((this, result))
            }
        }
    }

    /// Execute one step of a Suspended iterator.
    async fn exec_next_suspended<'a>(
        this: Self,
        source: Vec<Value>,
        ops: Vec<IterOp>,
        offset: usize,
        effect: Effect,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Option<(Value, IterHandle)>), RuntimeError> {
        // No ops: pull directly from source
        if ops.is_empty() {
            if offset >= source.len() {
                return Ok((this, None));
            }
            let item = source[offset].clone();
            let rest = IterHandle::suspended(source, Vec::new(), offset + 1, effect);
            return Ok((this, Some((item, rest))));
        }

        // Has ops: peel off the LAST op (outermost), inner keeps remaining ops
        // Pipeline: source → ops[0] → ops[1] → ... → ops[n-1]
        // So ops[n-1] is the outermost, applied last.
        let outer_op = ops[ops.len() - 1].clone();
        let inner_ops = ops[..ops.len() - 1].to_vec();
        let inner = IterHandle::suspended(source, inner_ops, offset, effect);

        Self::exec_next_apply_op(this, inner, outer_op, effect, handle).await
    }

    /// Apply a single IterOp on top of an inner IterHandle, pulling one element.
    async fn exec_next_apply_op<'a>(
        mut this: Self,
        inner: IterHandle,
        op: IterOp,
        effect: Effect,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Option<(Value, IterHandle)>), RuntimeError> {
        match op {
            IterOp::Map(f) => {
                let (mut this, result) = Self::exec_next(this, inner, handle).await?;
                match result {
                    None => Ok((this, None)),
                    Some((item, rest_inner)) => {
                        let mapped;
                        (this, mapped) = Self::call_closure(
                            this, f.clone(), vec![Arc::new(item)], handle,
                        ).await?;
                        let rest = rest_inner.map(f);
                        Ok((this, Some((mapped, rest))))
                    }
                }
            }

            IterOp::Filter(f) => {
                let mut current_inner = inner;
                loop {
                    let result;
                    (this, result) = Self::exec_next(this, current_inner, handle).await?;
                    match result {
                        None => return Ok((this, None)),
                        Some((item, rest_inner)) => {
                            let arc_item = Arc::new(item);
                            let keep;
                            (this, keep) = Self::call_closure(
                                this, f.clone(), vec![Arc::clone(&arc_item)], handle,
                            ).await?;
                            if matches!(keep, Value::Pure(PureValue::Bool(true))) {
                                let rest = rest_inner.filter(f);
                                return Ok((this, Some((Arc::unwrap_or_clone(arc_item), rest))));
                            }
                            current_inner = rest_inner;
                        }
                    }
                }
            }

            IterOp::Take(n) => {
                if n == 0 {
                    return Ok((this, None));
                }
                let (this, result) = Self::exec_next(this, inner, handle).await?;
                match result {
                    None => Ok((this, None)),
                    Some((item, rest_inner)) => {
                        let rest = rest_inner.take(n - 1);
                        Ok((this, Some((item, rest))))
                    }
                }
            }

            IterOp::Skip(n) => {
                let mut current_inner = inner;
                for _ in 0..n {
                    let result;
                    (this, result) = Self::exec_next(this, current_inner, handle).await?;
                    match result {
                        None => return Ok((this, None)),
                        Some((_, rest_inner)) => current_inner = rest_inner,
                    }
                }
                Self::exec_next(this, current_inner, handle).await
            }

            IterOp::Chain(chain) => {
                let other = IterHandle::from_chain(chain, effect);
                let combined = inner.chain(other);
                Self::exec_next(this, combined, handle).await
            }

            IterOp::Flatten => {
                let (this, result) = Self::exec_next(this, inner, handle).await?;
                match result {
                    None => Ok((this, None)),
                    Some((item, rest_inner)) => {
                        // Flatten signature: Iterator<List<T>> → Iterator<T>.
                        // Each element is a List — convert to IterHandle for lazy chaining.
                        let items = item.expect::<Vec<Value>>("Flatten");
                        let item_iter = IterHandle::from_list(items, effect);
                        let rest_flat = rest_inner.flatten();
                        let combined = item_iter.chain(rest_flat);
                        Self::exec_next(this, combined, handle).await
                    }
                }
            }

            IterOp::FlatMap(f) => {
                let (mut this, result) = Self::exec_next(this, inner, handle).await?;
                match result {
                    None => Ok((this, None)),
                    Some((item, rest_inner)) => {
                        let mapped;
                        (this, mapped) = Self::call_closure(
                            this, f.clone(), vec![Arc::new(item)], handle,
                        ).await?;
                        // flat_map signature: Fn(T) → Iterator<U>, so mapped is Iterator.
                        let mapped_iter = mapped.expect::<IterHandle>("FlatMap");
                        let rest_fm = rest_inner.flat_map(f);
                        let combined = mapped_iter.chain(rest_fm);
                        Self::exec_next(this, combined, handle).await
                    }
                }
            }
        }
    }

    // -- closure invocation ---------------------------------------------------

    async fn call_closure<'a>(
        this: Self,
        fn_val: FnValue,
        args: Vec<Arc<Value>>,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        let closure_body = &fn_val.body;

        let label_map = build_label_map(&closure_body.body);
        let mut frame = Frame::new(closure_body.body.val_count, label_map);

        let n_captures = fn_val.captures.len();
        for (i, cap) in fn_val.captures.iter().enumerate() {
            frame.set(ValueId(i as u32), Arc::clone(cap));
        }
        for (i, arg) in args.into_iter().enumerate() {
            frame.set(ValueId((n_captures + i) as u32), arg);
        }

        let insts = closure_body.body.insts.clone();
        let val_types = closure_body.body.val_types.clone();
        let (this, _, result) = Self::run(this, insts, val_types, frame, handle).await?;
        Ok((this, result.expect("closure must return a value")))
    }
}

// ---------------------------------------------------------------------------
// ExecCtx — delegates to Interpreter's exec_next / call_closure
// ---------------------------------------------------------------------------

impl builtins::ExecCtx for Interpreter {
    fn exec_next<'a>(
        self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Option<(Value, IterHandle)>), RuntimeError>> {
        Self::exec_next(self, ih, handle)
    }

    fn call_closure<'a>(
        self,
        f: FnValue,
        args: Vec<Arc<Value>>,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Value), RuntimeError>> {
        Box::pin(Self::call_closure(self, f, args, handle))
    }
}


// ---------------------------------------------------------------------------
// Cast execution
// ---------------------------------------------------------------------------

fn exec_cast(kind: CastKind, value: Value, src_ty: &Ty) -> Value {
    match kind {
        CastKind::DequeToList => {
            let d = value.expect::<TrackedDeque<Value>>("Cast DequeToList");
            Value::list(d.into_vec())
        }
        CastKind::ListToIterator => {
            let items = value.expect::<Vec<Value>>("Cast ListToIterator");
            Value::iterator(IterHandle::from_list(items, Effect::Pure))
        }
        CastKind::DequeToIterator => {
            let d = value.expect::<TrackedDeque<Value>>("Cast DequeToIterator");
            Value::iterator(IterHandle::from_list(d.into_vec(), Effect::Pure))
        }
        CastKind::DequeToSequence => {
            let d = value.expect::<TrackedDeque<Value>>("Cast DequeToSequence");
            Value::sequence(SequenceChain::from_stored(d))
        }
        CastKind::SequenceToIterator => {
            let effect = match src_ty {
                Ty::Sequence(_, _, e) => *e,
                _ => Effect::Pure,
            };
            let sc = value.expect::<SequenceChain>("Cast SequenceToIterator");
            Value::iterator(sc.into_iter_handle(effect))
        }
        CastKind::RangeToIterator => {
            let Value::Pure(PureValue::Range { start, end, inclusive }) = value else {
                panic!("Cast RangeToIterator: expected Range, got {value:?}")
            };
            let items: Vec<Value> = if inclusive {
                (start..=end).map(Value::int).collect()
            } else {
                (start..end).map(Value::int).collect()
            };
            Value::iterator(IterHandle::from_list(items, Effect::Pure))
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::int(*n),
        Literal::Float(f) => Value::float(*f),
        Literal::String(s) => Value::string(s.clone()),
        Literal::Bool(b) => Value::bool_(*b),
        Literal::Byte(b) => Value::byte(*b),
        Literal::List(elems) => {
            Value::deque(TrackedDeque::from_vec(elems.iter().map(literal_to_value).collect()))
        }
    }
}

fn eval_binop(op: BinOp, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
    let mismatch = || RuntimeError::bin_op_mismatch(op, left.kind(), right.kind());
    match op {
        BinOp::And => {
            Ok(Value::bool_(matches!(left, Value::Pure(PureValue::Bool(true))) && matches!(right, Value::Pure(PureValue::Bool(true)))))
        }
        BinOp::Or => {
            Ok(Value::bool_(matches!(left, Value::Pure(PureValue::Bool(true))) || matches!(right, Value::Pure(PureValue::Bool(true)))))
        }
        BinOp::Add => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a.wrapping_add(*b))),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Ok(Value::float(a + b)),
            (Value::Pure(PureValue::String(a)), Value::Pure(PureValue::String(b))) => {
                let mut s = a.clone();
                s.push_str(b);
                Ok(Value::string(s))
            }
            _ => Err(mismatch()),
        },
        BinOp::Sub => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a.wrapping_sub(*b))),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Ok(Value::float(a - b)),
            _ => Err(mismatch()),
        },
        BinOp::Mul => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a.wrapping_mul(*b))),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Ok(Value::float(a * b)),
            _ => Err(mismatch()),
        },
        BinOp::Div => match (left, right) {
            (Value::Pure(PureValue::Int(_)), Value::Pure(PureValue::Int(0))) => Err(RuntimeError::division_by_zero()),
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a.wrapping_div(*b))),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Ok(Value::float(a / b)),
            _ => Err(mismatch()),
        },
        BinOp::Mod => match (left, right) {
            (Value::Pure(PureValue::Int(_)), Value::Pure(PureValue::Int(0))) => Err(RuntimeError::division_by_zero()),
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a.wrapping_rem(*b))),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Ok(Value::float(a % b)),
            _ => Err(mismatch()),
        },
        BinOp::Xor => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a ^ b)),
            _ => Err(mismatch()),
        },
        BinOp::BitAnd => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a & b)),
            _ => Err(mismatch()),
        },
        BinOp::BitOr => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a | b)),
            _ => Err(mismatch()),
        },
        BinOp::Shl => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a << b)),
            _ => Err(mismatch()),
        },
        BinOp::Shr => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Ok(Value::int(a >> b)),
            _ => Err(mismatch()),
        },
        BinOp::Eq => Ok(Value::bool_(left.structural_eq(right))),
        BinOp::Neq => Ok(Value::bool_(!left.structural_eq(right))),
        BinOp::Lt => cmp_values(left, right, Ordering::is_lt),
        BinOp::Gt => cmp_values(left, right, Ordering::is_gt),
        BinOp::Lte => cmp_values(left, right, Ordering::is_le),
        BinOp::Gte => cmp_values(left, right, Ordering::is_ge),
    }
}

fn cmp_values(left: &Value, right: &Value, f: fn(Ordering) -> bool) -> Result<Value, RuntimeError> {
    let ord = match (left, right) {
        (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => a.cmp(b),
        (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => {
            a.partial_cmp(b).ok_or_else(RuntimeError::nan_comparison)?
        }
        (Value::Pure(PureValue::String(a)), Value::Pure(PureValue::String(b))) => a.cmp(b),
        _ => return Err(RuntimeError::bin_op_mismatch(BinOp::Lt, left.kind(), right.kind())),
    };
    Ok(Value::bool_(f(ord)))
}

fn eval_unaryop(op: UnaryOp, operand: &Value) -> Result<Value, RuntimeError> {
    match op {
        UnaryOp::Neg => match operand {
            Value::Pure(PureValue::Int(n)) => Ok(Value::int(-n)),
            Value::Pure(PureValue::Float(f)) => Ok(Value::float(-f)),
            _ => Err(RuntimeError::unary_op_mismatch(op, operand.kind())),
        },
        UnaryOp::Not => match operand {
            Value::Pure(PureValue::Bool(b)) => Ok(Value::bool_(!b)),
            _ => Err(RuntimeError::unary_op_mismatch(op, operand.kind())),
        },
    }
}
