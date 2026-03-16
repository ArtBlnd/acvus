use std::cmp::Ordering;

use std::sync::Arc;

use futures::future::BoxFuture;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::builtins::BuiltinId;
use acvus_mir::ir::{Inst, InstKind, Label, MirBody, MirModule, ValueId};
use acvus_utils::Astr;
use acvus_utils::Interner;
use acvus_utils::TrackedDeque;
use rustc_hash::FxHashMap;

use acvus_mir::ty::{Effect, Ty};
use crate::iter::SequenceChain;
use crate::builtins;
use crate::error::RuntimeError;
use crate::iter::{IterHandle, IterOp, IterRepr};
use crate::value::{FnValue, LazyValue, PureValue, TypedValue, Value};
use acvus_utils::{Coroutine, Stepped, YieldHandle};

/// Runtime coercion: Deque → SequenceChain (Deque ≤ Sequence).
/// Panics if the value is neither Sequence nor Deque.
fn into_sequence_chain(value: Value, context: &str) -> SequenceChain {
    match value {
        Value::Lazy(LazyValue::Sequence(sc)) => sc,
        Value::Lazy(LazyValue::Deque(d)) => SequenceChain::from_stored(d),
        other => panic!("{context}: expected Sequence or Deque, got {other:?}"),
    }
}

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
    iters: FxHashMap<ValueId, IterState>,
}

enum IterState {
    List {
        items: Vec<Value>,
        pos: usize,
    },
    Range {
        current: i64,
        end: i64,
        inclusive: bool,
    },
}

impl Frame {
    fn new(val_count: u32, label_map: FxHashMap<Label, usize>) -> Self {
        Self {
            vals: vec![None; val_count as usize],
            label_map,
            iters: FxHashMap::default(),
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
        let (label, args) = if matches!(self.get(cond), Value::Pure(PureValue::Bool(true))) {
            then
        } else {
            else_
        };
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

    // -- iteration ------------------------------------------------------------

    fn iter_init(&mut self, dst: ValueId, src: ValueId) {
        let state = match self.take_owned(src) {
            Value::Lazy(LazyValue::List(items)) => IterState::List { items, pos: 0 },
            Value::Lazy(LazyValue::Deque(deque)) => IterState::List {
                items: deque.into_vec(),
                pos: 0,
            },
            Value::Pure(PureValue::Range {
                start,
                end,
                inclusive,
            }) => IterState::Range {
                current: start,
                end,
                inclusive,
            },
            v => panic!("IterInit: expected List or Range, got {v:?}"),
        };
        self.iters.insert(dst, state);
        self.set_new(dst, Value::unit());
    }

    fn iter_next(&mut self, dst_value: ValueId, dst_done: ValueId, iter: ValueId) {
        let state = self
            .iters
            .get_mut(&iter)
            .unwrap_or_else(|| panic!("IterNext: no iterator for Val({})", iter.0));

        let (value, done) = match state {
            IterState::List { items, pos } if *pos < items.len() => {
                let val = items[*pos].clone();
                *pos += 1;
                (val, false)
            }
            IterState::List { .. } => (Value::unit(), true),
            IterState::Range {
                current,
                end,
                inclusive,
            } if (*inclusive && *current <= *end) || (!*inclusive && *current < *end) => {
                let val = Value::int(*current);
                *current += 1;
                (val, false)
            }
            IterState::Range { .. } => (Value::unit(), true),
        };

        self.set_new(dst_value, value);
        self.set_new(dst_done, Value::bool_(done));
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
                    request.resolve(Arc::new(v.clone()));
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
                    handle.yield_val(TypedValue::new(val, ty)).await;
                }

                // -- constants / constructors --
                InstKind::Const { dst, value } => {
                    frame.set_new(*dst, literal_to_value(value));
                }
                InstKind::MakeList { dst, elements } => {
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
                    let (s, e) = (frame.get(*start), frame.get(*end));
                    match (s, e) {
                        (Value::Pure(PureValue::Int(s)), Value::Pure(PureValue::Int(e))) => frame.set_new(
                            *dst,
                            Value::range(*s, *e, matches!(kind, RangeKind::InclusiveEnd)),
                        ),
                        _ => panic!("MakeRange: expected Int bounds"),
                    }
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
                    let result = eval_binop(*op, frame.get(*left), frame.get(*right));
                    frame.set_new(*dst, result);
                }
                InstKind::UnaryOp { dst, op, operand } => {
                    frame.set_new(*dst, eval_unaryop(*op, frame.get(*operand)));
                }

                // -- access --
                InstKind::FieldGet { dst, object, field } => {
                    let v = expect_object(frame.get(*object), "FieldGet")
                        .get(field)
                        .unwrap_or_else(|| {
                            panic!(
                                "FieldGet: key '{}' not found",
                                field.display(&this.interner)
                            )
                        })
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ObjectGet { dst, object, key } => {
                    let v = expect_object(frame.get(*object), "ObjectGet")
                        .get(key)
                        .unwrap_or_else(|| {
                            panic!("ObjectGet: key '{}' not found", key.display(&this.interner))
                        })
                        .clone();
                    frame.set_new(*dst, v);
                }
                InstKind::TupleIndex { dst, tuple, index } => {
                    let v = expect_tuple(frame.get(*tuple), "TupleIndex")[*index].clone();
                    frame.set_new(*dst, v);
                }
                InstKind::ListIndex { dst, list, index } => {
                    let items = expect_list(frame.get(*list), "ListIndex");
                    let i = if *index >= 0 {
                        *index as usize
                    } else {
                        (items.len() as i32 + *index) as usize
                    };
                    frame.set_new(*dst, items[i].clone());
                }
                InstKind::ListGet { dst, list, index } => {
                    let items = expect_list(frame.get(*list), "ListGet");
                    let idx = expect_int(frame.get(*index), "ListGet index");
                    frame.set_new(*dst, items[idx as usize].clone());
                }
                InstKind::ListSlice {
                    dst,
                    list,
                    skip_head,
                    skip_tail,
                } => {
                    let items = expect_list(frame.get(*list), "ListSlice");
                    let end = items.len() - *skip_tail;
                    frame.set_new(*dst, Value::list(items[*skip_head..end].to_vec()));
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
                    let eq = values_equal(frame.get(*src), &literal_to_value(value));
                    frame.set_new(*dst, Value::bool_(eq));
                }
                InstKind::TestListLen {
                    dst,
                    src,
                    min_len,
                    exact,
                } => {
                    let items = expect_list(frame.get(*src), "TestListLen");
                    let ok = if *exact {
                        items.len() == *min_len
                    } else {
                        items.len() >= *min_len
                    };
                    frame.set_new(*dst, Value::bool_(ok));
                }
                InstKind::TestObjectKey { dst, src, key } => {
                    let ok = expect_object(frame.get(*src), "TestObjectKey").contains_key(key);
                    frame.set_new(*dst, Value::bool_(ok));
                }
                InstKind::TestRange {
                    dst,
                    src,
                    start,
                    end,
                    kind,
                } => {
                    let n = expect_int(frame.get(*src), "TestRange");
                    let ok = match kind {
                        RangeKind::Exclusive => n >= *start && n < *end,
                        RangeKind::InclusiveEnd => n >= *start && n <= *end,
                        RangeKind::ExclusiveStart => n > *start && n <= *end,
                    };
                    frame.set_new(*dst, Value::bool_(ok));
                }

                // -- iteration --
                InstKind::IterInit { dst, src } => frame.iter_init(*dst, *src),
                InstKind::IterNext {
                    dst_value,
                    dst_done,
                    iter,
                } => {
                    frame.iter_next(*dst_value, *dst_done, *iter);
                }

                // -- context / variable I/O --
                InstKind::ContextLoad { dst, name } => {
                    let typed = handle.request_context(*name).await;
                    frame.set(*dst, Arc::unwrap_or_clone(typed).into_value());
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
                    let result;
                    (this, result) =
                        Self::exec_builtin(this, *builtin, arg_values, handle).await?;
                    frame.set_new(*dst, result);
                }
                InstKind::ExternCall { dst, name, args } => {
                    let typed_args: Vec<TypedValue> = args.iter().map(|a| {
                        TypedValue::new(frame.take(*a), val_types[a].clone())
                    }).collect();
                    let arc = handle.request_extern_call(*name, typed_args).await;
                    frame.set(*dst, Arc::unwrap_or_clone(arc).into_value());
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
                                TypedValue::new(frame.take(*a), val_types[a].clone())
                            }).collect();
                            let arc = handle.request_extern_call(name, typed_args).await;
                            frame.set(*dst, Arc::unwrap_or_clone(arc).into_value());
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
        mut this: Self,
        id: BuiltinId,
        mut args: Vec<Value>,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        match id {
            BuiltinId::ToString
            | BuiltinId::ToInt
            | BuiltinId::ToFloat
            | BuiltinId::CharToInt
            | BuiltinId::IntToChar
            | BuiltinId::Len
            | BuiltinId::Contains
            | BuiltinId::ContainsStr
            | BuiltinId::Substring
            | BuiltinId::LenStr
            | BuiltinId::ToBytes
            | BuiltinId::ToUtf8
            | BuiltinId::ToUtf8Lossy
            | BuiltinId::Trim
            | BuiltinId::TrimStart
            | BuiltinId::TrimEnd
            | BuiltinId::Upper
            | BuiltinId::Lower
            | BuiltinId::ReplaceStr
            | BuiltinId::SplitStr
            | BuiltinId::StartsWithStr
            | BuiltinId::EndsWithStr
            | BuiltinId::RepeatStr
            | BuiltinId::Unwrap
            | BuiltinId::First
            | BuiltinId::FirstIter
            | BuiltinId::Last
            | BuiltinId::LastIter
            | BuiltinId::ContainsIter
            | BuiltinId::UnwrapOr => {
                let result = builtins::call_pure(id, args)?;
                Ok((this, result))
            }

            // -- Iterator constructors --
            BuiltinId::Reverse => {
                let mut items = match args.remove(0) {
                    Value::Lazy(LazyValue::List(items)) => items,
                    Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                    other => panic!("reverse: expected List or Deque, got {other:?}"),
                };
                items.reverse();
                Ok((this, Value::list(items)))
            }
            BuiltinId::Iter => {
                let items = match args.remove(0) {
                    Value::Lazy(LazyValue::List(items)) => items,
                    Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                    other => panic!("iter: expected List or Deque, got {other:?}"),
                };
                Ok((this, Value::iterator(IterHandle::from_list(items, Effect::Pure))))
            }
            BuiltinId::RevIter => {
                let mut items = match args.remove(0) {
                    Value::Lazy(LazyValue::List(items)) => items,
                    Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                    other => panic!("rev_iter: expected List or Deque, got {other:?}"),
                };
                items.reverse();
                Ok((this, Value::iterator(IterHandle::from_list(items, Effect::Pure))))
            }
            BuiltinId::Next | BuiltinId::NextSeq => {
                let none_tag = this.interner.intern("None");
                let some_tag = this.interner.intern("Some");
                let ih = args.remove(0).into_iter_handle(Effect::Pure);
                let (this, result) = Self::exec_next(this, ih, handle).await?;
                match result {
                    None => Ok((this, Value::variant(none_tag, None))),
                    Some((item, rest)) => Ok((this, Value::variant(
                        some_tag,
                        Some(Box::new(Value::tuple(vec![item, Value::iterator(rest)]))),
                    ))),
                }
            }
            BuiltinId::Collect => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                Self::exec_collect(this, shared, handle).await
            }
            BuiltinId::Take => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Pure(PureValue::Int(n)) = args.remove(0) else {
                    panic!("take: expected Int")
                };
                Ok((this, Value::iterator(shared.take(n as usize))))
            }
            BuiltinId::Skip => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Pure(PureValue::Int(n)) = args.remove(0) else {
                    panic!("skip: expected Int")
                };
                Ok((this, Value::iterator(shared.skip(n as usize))))
            }
            BuiltinId::Chain => {
                let a = args.remove(0).into_iter_handle(Effect::Pure);
                let b = args.remove(0).into_iter_handle(Effect::Pure);
                Ok((this, Value::iterator(a.chain(b))))
            }

            // -- Iterator operations --
            BuiltinId::Flatten | BuiltinId::FlattenIter => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                Ok((this, Value::iterator(shared.flatten())))
            }
            BuiltinId::FlatMap | BuiltinId::FlatMapIter => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Lazy(LazyValue::Fn(f)) = args.remove(0) else {
                    panic!("flat_map: expected Fn")
                };
                Ok((this, Value::iterator(shared.flat_map(f))))
            }
            BuiltinId::Join | BuiltinId::JoinIter => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Pure(PureValue::String(sep)) = args.remove(0) else {
                    panic!("join: expected String separator")
                };
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let parts: Vec<String> = items
                    .into_iter()
                    .map(|v| match v {
                        Value::Pure(PureValue::String(s)) => s,
                        _ => panic!("join: element is not String"),
                    })
                    .collect();
                Ok((this, Value::string(parts.join(&sep))))
            }
            // -- Sequence structural ops (origin preserved) --
            // Deque ≤ Sequence coercion: Deque values are accepted and wrapped.
            BuiltinId::TakeSeq => {
                let sc = into_sequence_chain(args.remove(0), "take_seq");
                let Value::Pure(PureValue::Int(n)) = args.remove(0) else {
                    panic!("take_seq: expected Int")
                };
                Ok((this, Value::sequence(sc.take(n as usize))))
            }
            BuiltinId::SkipSeq => {
                let sc = into_sequence_chain(args.remove(0), "skip_seq");
                let Value::Pure(PureValue::Int(n)) = args.remove(0) else {
                    panic!("skip_seq: expected Int")
                };
                Ok((this, Value::sequence(sc.skip(n as usize))))
            }
            BuiltinId::ChainSeq => {
                let sc = into_sequence_chain(args.remove(0), "chain_seq");
                let rhs = args.remove(0).into_iter_handle(Effect::Pure);
                Ok((this, Value::sequence(sc.chain(rhs))))
            }
            // Deleted: MapSeq, PmapSeq, FilterSeq, FlattenSeq, FlatMapSeq,
            // FlatMapIterSeq, CollectSeq, RevSeq — Sequence coerces to Iterator
            // via the type system for these ops.
            // -- Lazy HOFs (return Iterator) --
            BuiltinId::Map | BuiltinId::Pmap => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Lazy(LazyValue::Fn(f)) = args.remove(0) else {
                    panic!("map: expected Fn")
                };
                Ok((this, Value::iterator(shared.map(f))))
            }
            BuiltinId::Filter => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Lazy(LazyValue::Fn(f)) = args.remove(0) else {
                    panic!("filter: expected Fn")
                };
                Ok((this, Value::iterator(shared.filter(f))))
            }

            // -- Consuming HOFs (collect then apply) --
            BuiltinId::Find => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Lazy(LazyValue::Fn(f)) = fn_val else {
                    panic!("find: expected Fn")
                };
                Self::exec_hof_find_inner(this, items, f, handle).await
            }
            BuiltinId::Reduce => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Lazy(LazyValue::Fn(f)) = fn_val else {
                    panic!("reduce: expected Fn")
                };
                Self::exec_hof_reduce_inner(this, items, f, handle).await
            }
            BuiltinId::Fold => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let init = args.remove(0);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Lazy(LazyValue::Fn(f)) = fn_val else {
                    panic!("fold: expected Fn")
                };
                Self::exec_hof_fold_inner(this, items, init, f, handle).await
            }
            BuiltinId::Any => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Lazy(LazyValue::Fn(f)) = fn_val else {
                    panic!("any: expected Fn")
                };
                Self::exec_hof_any_inner(this, items, f, handle).await
            }
            BuiltinId::All => {
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Lazy(LazyValue::Fn(f)) = fn_val else {
                    panic!("all: expected Fn")
                };
                Self::exec_hof_all_inner(this, items, f, handle).await
            }

            // -- Deque ops --
            BuiltinId::Append => {
                let list = args.remove(0);
                let item = args.remove(0);
                let Value::Lazy(LazyValue::Deque(mut deque)) = list else {
                    panic!("append: expected Deque, got {list:?}")
                };
                deque.push(item);
                Ok((this, Value::deque(deque)))
            }
            BuiltinId::Extend => {
                let list = args.remove(0);
                let shared = args.remove(0).into_iter_handle(Effect::Pure);
                let Value::Lazy(LazyValue::Deque(mut deque)) = list else {
                    panic!("extend: expected Deque as first arg")
                };
                let (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                deque.extend(items);
                Ok((this, Value::deque(deque)))
            }
            BuiltinId::Consume => {
                let list = args.remove(0);
                let n_val = args.remove(0);
                let Value::Lazy(LazyValue::Deque(mut deque)) = list else {
                    panic!("consume: expected Deque as first arg")
                };
                let Value::Pure(PureValue::Int(n)) = n_val else {
                    panic!("consume: expected Int as second arg")
                };
                deque.consume(n as usize);
                Ok((this, Value::deque(deque)))
            }
        }
    }

    // -- collect helpers ------------------------------------------------------

    async fn exec_collect<'a>(
        this: Self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        let (this, items) = Self::exec_collect_vec(this, ih, handle).await?;
        Ok((this, Value::list(items)))
    }

    async fn exec_collect_vec<'a>(
        mut this: Self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Vec<Value>), RuntimeError> {
        let mut items = Vec::new();
        let mut current = ih;
        loop {
            let result;
            (this, result) = Self::exec_next(this, current, handle).await?;
            match result {
                Some((item, rest)) => {
                    items.push(item);
                    current = rest;
                }
                None => break,
            }
        }
        Ok((this, items))
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
                        let inner_items = match item {
                            Value::Lazy(LazyValue::List(items)) => items,
                            Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                            other => vec![other],
                        };
                        let items_iter = IterHandle::from_list(inner_items, effect);
                        let rest_flat = rest_inner.flatten();
                        let combined = items_iter.chain(rest_flat);
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
                        let inner_items = match mapped {
                            Value::Lazy(LazyValue::List(items)) => items,
                            Value::Lazy(LazyValue::Deque(d)) => d.into_vec(),
                            other => vec![other],
                        };
                        let items_iter = IterHandle::from_list(inner_items, effect);
                        let rest_fm = rest_inner.flat_map(f);
                        let combined = items_iter.chain(rest_fm);
                        Self::exec_next(this, combined, handle).await
                    }
                }
            }
        }
    }

    /// Collect a SequenceChain: apply structural ops to origin TrackedDeque.
    ///
    /// Chain ops may contain IterHandle with closures, so exec_next is needed.
    /// The result preserves the origin's checksum lineage — it can be diffed
    /// against the storage origin via `TrackedDeque::into_diff`.
    async fn exec_collect_sequence<'a>(
        mut this: Self,
        sc: SequenceChain,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, TrackedDeque<Value>), RuntimeError> {
        let origin_checksum = sc.origin_checksum();
        let mut deque = sc.origin().clone();

        for op in sc.ops() {
            match op {
                crate::iter::SequenceOp::Take(n) => {
                    // Keep only first n elements: remove excess from back
                    let n = *n;
                    let len = deque.len();
                    if n < len {
                        for _ in 0..(len - n) {
                            deque.pop();
                        }
                    }
                }
                crate::iter::SequenceOp::Skip(n) => {
                    deque.consume(*n);
                }
                crate::iter::SequenceOp::Chain(iter) => {
                    let items;
                    (this, items) = Self::exec_collect_vec(this, iter.clone(), handle).await?;
                    deque.extend(items);
                }
            }
        }

        // Runtime assertion: checksum must still match origin.
        // If it doesn't, something corrupted the TrackedDeque.
        assert_eq!(
            deque.checksum(),
            origin_checksum,
            "SequenceChain collect: checksum diverged from origin ({:#x} vs {:#x}). \
             This indicates a bug in structural op application.",
            deque.checksum(),
            origin_checksum,
        );

        Ok((this, deque))
    }

    // -- consuming HOF inner implementations ----------------------------------

    async fn exec_hof_find_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let arc_item = Arc::new(item);
            let matched;
            (this, matched) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::clone(&arc_item)], handle)
                    .await?;
            if matches!(matched, Value::Pure(PureValue::Bool(true))) {
                return Ok((this, Arc::unwrap_or_clone(arc_item)));
            }
        }
        Err(RuntimeError::empty_collection("find"))
    }

    async fn exec_hof_reduce_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        let mut it = items.into_iter();
        let Some(mut acc) = it.next() else {
            return Err(RuntimeError::empty_collection("reduce"));
        };
        for item in it {
            (this, acc) = Self::call_closure(
                this,
                fn_val.clone(),
                vec![Arc::new(acc), Arc::new(item)],
                handle,
            )
            .await?;
        }
        Ok((this, acc))
    }

    async fn exec_hof_fold_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        init: Value,
        fn_val: FnValue,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        let mut acc = init;
        for item in items {
            (this, acc) = Self::call_closure(
                this,
                fn_val.clone(),
                vec![Arc::new(acc), Arc::new(item)],
                handle,
            )
            .await?;
        }
        Ok((this, acc))
    }

    async fn exec_hof_any_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let result;
            (this, result) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::new(item)], handle).await?;
            if matches!(result, Value::Pure(PureValue::Bool(true))) {
                return Ok((this, Value::bool_(true)));
            }
        }
        Ok((this, Value::bool_(false)))
    }

    async fn exec_hof_all_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<TypedValue>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let result;
            (this, result) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::new(item)], handle).await?;
            if matches!(result, Value::Pure(PureValue::Bool(false))) {
                return Ok((this, Value::bool_(false)));
            }
        }
        Ok((this, Value::bool_(true)))
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
// Value extractors — flat panic on type mismatch
// ---------------------------------------------------------------------------

fn expect_list<'a>(v: &'a Value, ctx: &str) -> &'a [Value] {
    match v {
        Value::Lazy(LazyValue::List(items)) => items,
        Value::Lazy(LazyValue::Deque(d)) => d.as_slice(),
        _ => panic!("{ctx}: expected List, got {v:?}"),
    }
}

fn expect_object<'a>(v: &'a Value, ctx: &str) -> &'a FxHashMap<Astr, Value> {
    match v {
        Value::Lazy(LazyValue::Object(fields)) => fields,
        _ => panic!("{ctx}: expected Object, got {v:?}"),
    }
}

fn expect_tuple<'a>(v: &'a Value, ctx: &str) -> &'a [Value] {
    match v {
        Value::Lazy(LazyValue::Tuple(elems)) => elems,
        _ => panic!("{ctx}: expected Tuple, got {v:?}"),
    }
}

fn expect_int(v: &Value, ctx: &str) -> i64 {
    match v {
        Value::Pure(PureValue::Int(n)) => *n,
        _ => panic!("{ctx}: expected Int, got {v:?}"),
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

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => a == b,
        (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => a == b,
        (Value::Pure(PureValue::String(a)), Value::Pure(PureValue::String(b))) => a == b,
        (Value::Pure(PureValue::Bool(a)), Value::Pure(PureValue::Bool(b))) => a == b,
        (Value::Pure(PureValue::Byte(a)), Value::Pure(PureValue::Byte(b))) => a == b,
        (Value::Pure(PureValue::Unit), Value::Pure(PureValue::Unit)) => true,
        (
            Value::Lazy(LazyValue::Variant {
                tag: ta,
                payload: pa,
            }),
            Value::Lazy(LazyValue::Variant {
                tag: tb,
                payload: pb,
            }),
        ) => {
            ta == tb
                && match (pa, pb) {
                    (Some(a), Some(b)) => values_equal(a, b),
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => false,
    }
}

fn eval_binop(op: BinOp, left: &Value, right: &Value) -> Value {
    match op {
        BinOp::And => {
            Value::bool_(matches!(left, Value::Pure(PureValue::Bool(true))) && matches!(right, Value::Pure(PureValue::Bool(true))))
        }
        BinOp::Or => {
            Value::bool_(matches!(left, Value::Pure(PureValue::Bool(true))) || matches!(right, Value::Pure(PureValue::Bool(true))))
        }
        BinOp::Add => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a.wrapping_add(*b)),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Value::float(a + b),
            (Value::Pure(PureValue::String(a)), Value::Pure(PureValue::String(b))) => {
                let mut s = a.clone();
                s.push_str(b);
                Value::string(s)
            }
            (l, r) => panic!("Add: incompatible {l:?} + {r:?}"),
        },
        BinOp::Sub => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a.wrapping_sub(*b)),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Value::float(a - b),
            (l, r) => panic!("Sub: incompatible {l:?} - {r:?}"),
        },
        BinOp::Mul => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a.wrapping_mul(*b)),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Value::float(a * b),
            (l, r) => panic!("Mul: incompatible {l:?} * {r:?}"),
        },
        BinOp::Div => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a / b),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Value::float(a / b),
            (l, r) => panic!("Div: incompatible {l:?} / {r:?}"),
        },
        BinOp::Mod => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a % b),
            (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => Value::float(a % b),
            (l, r) => panic!("Mod: incompatible {l:?} % {r:?}"),
        },
        BinOp::Xor => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a ^ b),
            (l, r) => panic!("Xor: incompatible {l:?} ^ {r:?}"),
        },
        BinOp::BitAnd => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a & b),
            (l, r) => panic!("BitAnd: incompatible {l:?} & {r:?}"),
        },
        BinOp::BitOr => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a | b),
            (l, r) => panic!("BitOr: incompatible {l:?} | {r:?}"),
        },
        BinOp::Shl => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a << b),
            (l, r) => panic!("Shl: incompatible {l:?} << {r:?}"),
        },
        BinOp::Shr => match (left, right) {
            (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => Value::int(a >> b),
            (l, r) => panic!("Shr: incompatible {l:?} >> {r:?}"),
        },
        BinOp::Eq => Value::bool_(values_equal(left, right)),
        BinOp::Neq => Value::bool_(!values_equal(left, right)),
        BinOp::Lt => cmp_values(left, right, Ordering::is_lt),
        BinOp::Gt => cmp_values(left, right, Ordering::is_gt),
        BinOp::Lte => cmp_values(left, right, Ordering::is_le),
        BinOp::Gte => cmp_values(left, right, Ordering::is_ge),
    }
}

fn cmp_values(left: &Value, right: &Value, f: fn(Ordering) -> bool) -> Value {
    let ord = match (left, right) {
        (Value::Pure(PureValue::Int(a)), Value::Pure(PureValue::Int(b))) => a.cmp(b),
        (Value::Pure(PureValue::Float(a)), Value::Pure(PureValue::Float(b))) => a.partial_cmp(b).unwrap(),
        (Value::Pure(PureValue::String(a)), Value::Pure(PureValue::String(b))) => a.cmp(b),
        (l, r) => panic!("comparison: incompatible {l:?} vs {r:?}"),
    };
    Value::bool_(f(ord))
}

fn eval_unaryop(op: UnaryOp, operand: &Value) -> Value {
    match op {
        UnaryOp::Neg => match operand {
            Value::Pure(PureValue::Int(n)) => Value::int(-n),
            Value::Pure(PureValue::Float(f)) => Value::float(-f),
            v => panic!("Neg: expected numeric, got {v:?}"),
        },
        UnaryOp::Not => match operand {
            Value::Pure(PureValue::Bool(b)) => Value::bool_(!b),
            v => panic!("Not: expected Bool, got {v:?}"),
        },
    }
}
