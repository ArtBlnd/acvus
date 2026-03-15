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

use crate::builtins;
use crate::error::RuntimeError;
use crate::iter::{IterChain, IterOp, SharedIter};
use crate::value::{FnValue, Value};
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
        let (label, args) = if matches!(self.get(cond), Value::Bool(true)) {
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
            Value::List(items) => IterState::List { items, pos: 0 },
            Value::Deque(deque) => IterState::List {
                items: deque.into_vec(),
                pos: 0,
            },
            Value::Range {
                start,
                end,
                inclusive,
            } => IterState::Range {
                current: start,
                end,
                inclusive,
            },
            v => panic!("IterInit: expected List or Range, got {v:?}"),
        };
        self.iters.insert(dst, state);
        self.set_new(dst, Value::Unit);
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
            IterState::List { .. } => (Value::Unit, true),
            IterState::Range {
                current,
                end,
                inclusive,
            } if (*inclusive && *current <= *end) || (!*inclusive && *current < *end) => {
                let val = Value::Int(*current);
                *current += 1;
                (val, false)
            }
            IterState::Range { .. } => (Value::Unit, true),
        };

        self.set_new(dst_value, value);
        self.set_new(dst_done, Value::Bool(done));
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

    pub fn execute(self) -> Coroutine<Value, RuntimeError> {
        acvus_utils::coroutine(|handle| async move {
            crate::set_interner_ctx(&self.interner);
            let insts = self.module.main.insts.clone();
            let label_map = build_label_map(&self.module.main);
            let frame = Frame::new(self.module.main.val_count, label_map);
            Self::run(self, insts, frame, &handle).await?;
            Ok(())
        })
    }

    /// Drive the coroutine to completion with a pre-built context map.
    /// Returns all emitted values. Panics on missing context or extern calls.
    pub async fn execute_with_context(self, context: FxHashMap<Astr, Value>) -> Vec<Value> {
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
        frame: Frame,
        handle: &'a YieldHandle<Value>,
    ) -> BoxFuture<'a, Result<(Self, Frame, Option<Value>), RuntimeError>> {
        Box::pin(Self::run_inner(this, insts, frame, handle))
    }

    async fn run_inner(
        mut this: Self,
        insts: Vec<Inst>,
        mut frame: Frame,
        handle: &YieldHandle<Value>,
    ) -> Result<(Self, Frame, Option<Value>), RuntimeError> {
        let mut pc = 0;
        while pc < insts.len() {
            match &insts[pc].kind {
                // -- yield --
                InstKind::Yield(v) => {
                    let val = frame.take_owned(*v);
                    handle.yield_val(val).await;
                }

                // -- constants / constructors --
                InstKind::Const { dst, value } => {
                    frame.set_new(*dst, literal_to_value(value));
                }
                InstKind::MakeList { dst, elements } => {
                    let items = frame.collect_args(elements);
                    frame.set_new(*dst, Value::Deque(TrackedDeque::from_vec(items)));
                }
                InstKind::MakeObject { dst, fields } => {
                    let obj: FxHashMap<Astr, Value> = fields
                        .iter()
                        .map(|(k, v)| (*k, frame.take_owned(*v)))
                        .collect();
                    frame.set_new(*dst, Value::Object(obj));
                }
                InstKind::MakeRange {
                    dst,
                    start,
                    end,
                    kind,
                } => {
                    let (s, e) = (frame.get(*start), frame.get(*end));
                    match (s, e) {
                        (Value::Int(s), Value::Int(e)) => frame.set_new(
                            *dst,
                            Value::Range {
                                start: *s,
                                end: *e,
                                inclusive: matches!(kind, RangeKind::InclusiveEnd),
                            },
                        ),
                        _ => panic!("MakeRange: expected Int bounds"),
                    }
                }
                InstKind::MakeTuple { dst, elements } => {
                    let items = frame.collect_args(elements);
                    frame.set_new(*dst, Value::Tuple(items));
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
                        Value::Fn(FnValue {
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
                    frame.set_new(*dst, Value::List(items[*skip_head..end].to_vec()));
                }

                // -- variant --
                InstKind::MakeVariant { dst, tag, payload } => {
                    let p = payload.as_ref().map(|v| Box::new(frame.take_owned(*v)));
                    frame.set_new(
                        *dst,
                        Value::Variant {
                            tag: *tag,
                            payload: p,
                        },
                    );
                }
                InstKind::TestVariant { dst, src, tag } => {
                    let Value::Variant { tag: t, .. } = frame.get(*src) else {
                        panic!("TestVariant: expected Variant, got {:?}", frame.get(*src));
                    };
                    frame.set_new(*dst, Value::Bool(*t == *tag));
                }
                InstKind::UnwrapVariant { dst, src } => {
                    let Value::Variant {
                        payload: Some(inner),
                        ..
                    } = frame.get(*src)
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
                    frame.set_new(*dst, Value::Bool(eq));
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
                    frame.set_new(*dst, Value::Bool(ok));
                }
                InstKind::TestObjectKey { dst, src, key } => {
                    let ok = expect_object(frame.get(*src), "TestObjectKey").contains_key(key);
                    frame.set_new(*dst, Value::Bool(ok));
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
                    frame.set_new(*dst, Value::Bool(ok));
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
                    let arc = handle.request_context(*name).await;
                    frame.set(*dst, arc);
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
                    let arg_values = frame.collect_args(args);
                    let arc = handle.request_extern_call(*name, arg_values).await;
                    frame.set_new(*dst, Arc::unwrap_or_clone(arc));
                }
                InstKind::ClosureCall { dst, closure, args } => {
                    let callee = frame.take_owned(*closure);
                    match callee {
                        Value::Fn(fn_val) => {
                            let arg_values = frame.collect_args_arc(args);
                            let result;
                            (this, result) = Self::call_closure(this, fn_val, arg_values, handle).await?;
                            frame.set_new(*dst, result);
                        }
                        Value::ExternFn(name) => {
                            let arg_values = frame.collect_args(args);
                            let arc = handle.request_extern_call(name, arg_values).await;
                            frame.set_new(*dst, Arc::unwrap_or_clone(arc));
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
        handle: &'a YieldHandle<Value>,
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
                    Value::List(items) => items,
                    Value::Deque(d) => d.into_vec(),
                    other => panic!("reverse: expected List or Deque, got {other:?}"),
                };
                items.reverse();
                Ok((this, Value::List(items)))
            }
            BuiltinId::Iter => {
                let items = match args.remove(0) {
                    Value::List(items) => items,
                    Value::Deque(d) => d.into_vec(),
                    other => panic!("iter: expected List or Deque, got {other:?}"),
                };
                Ok((this, Value::Iterator(SharedIter::from_list(items))))
            }
            BuiltinId::RevIter => {
                let items = match args.remove(0) {
                    Value::List(items) => items,
                    Value::Deque(d) => d.into_vec(),
                    other => panic!("rev_iter: expected List or Deque, got {other:?}"),
                };
                Ok((this, Value::Iterator(SharedIter::from_list_rev(items))))
            }
            BuiltinId::Collect => {
                let shared = args.remove(0).into_shared_iter();
                Self::exec_collect(this, shared, handle).await
            }
            BuiltinId::Take => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Int(n) = args.remove(0) else {
                    panic!("take: expected Int")
                };
                Ok((this, Value::Iterator(shared.take(n as usize))))
            }
            BuiltinId::Skip => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Int(n) = args.remove(0) else {
                    panic!("skip: expected Int")
                };
                Ok((this, Value::Iterator(shared.skip(n as usize))))
            }
            BuiltinId::Chain => {
                let a = args.remove(0).into_shared_iter();
                let b = args.remove(0).into_shared_iter();
                Ok((this, Value::Iterator(a.chain(b))))
            }

            // -- Iterator operations --
            BuiltinId::Flatten | BuiltinId::FlattenIter => {
                let shared = args.remove(0).into_shared_iter();
                Ok((this, Value::Iterator(shared.flatten())))
            }
            BuiltinId::FlatMap | BuiltinId::FlatMapIter => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("flat_map: expected Fn")
                };
                Ok((this, Value::Iterator(shared.flat_map(f))))
            }
            BuiltinId::Join | BuiltinId::JoinIter => {
                let shared = args.remove(0).into_shared_iter();
                let Value::String(sep) = args.remove(0) else {
                    panic!("join: expected String separator")
                };
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let parts: Vec<String> = items
                    .into_iter()
                    .map(|v| match v {
                        Value::String(s) => s,
                        _ => panic!("join: element is not String"),
                    })
                    .collect();
                Ok((this, Value::String(parts.join(&sep))))
            }
            // -- Lazy Sequence HOFs (return Sequence) --
            BuiltinId::MapSeq | BuiltinId::PmapSeq => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("map: expected Fn")
                };
                Ok((this, Value::Sequence(shared.map(f))))
            }
            BuiltinId::FilterSeq => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("filter: expected Fn")
                };
                Ok((this, Value::Sequence(shared.filter(f))))
            }
            BuiltinId::TakeSeq => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Int(n) = args.remove(0) else {
                    panic!("take: expected Int")
                };
                Ok((this, Value::Sequence(shared.take(n as usize))))
            }
            BuiltinId::SkipSeq => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Int(n) = args.remove(0) else {
                    panic!("skip: expected Int")
                };
                Ok((this, Value::Sequence(shared.skip(n as usize))))
            }
            BuiltinId::ChainSeq => {
                let a = args.remove(0).into_shared_iter();
                let b = args.remove(0).into_shared_iter();
                Ok((this, Value::Sequence(a.chain(b))))
            }
            BuiltinId::FlattenSeq => {
                let shared = args.remove(0).into_shared_iter();
                Ok((this, Value::Sequence(shared.flatten())))
            }
            BuiltinId::FlatMapSeq | BuiltinId::FlatMapIterSeq => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("flat_map: expected Fn")
                };
                Ok((this, Value::Sequence(shared.flat_map(f))))
            }
            // -- Lazy HOFs (return Iterator) --
            BuiltinId::Map | BuiltinId::Pmap => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("map: expected Fn")
                };
                Ok((this, Value::Iterator(shared.map(f))))
            }
            BuiltinId::Filter => {
                let shared = args.remove(0).into_shared_iter();
                let Value::Fn(f) = args.remove(0) else {
                    panic!("filter: expected Fn")
                };
                Ok((this, Value::Iterator(shared.filter(f))))
            }

            // -- Consuming HOFs (collect then apply) --
            BuiltinId::Find => {
                let shared = args.remove(0).into_shared_iter();
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Fn(f) = fn_val else {
                    panic!("find: expected Fn")
                };
                Self::exec_hof_find_inner(this, items, f, handle).await
            }
            BuiltinId::Reduce => {
                let shared = args.remove(0).into_shared_iter();
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Fn(f) = fn_val else {
                    panic!("reduce: expected Fn")
                };
                Self::exec_hof_reduce_inner(this, items, f, handle).await
            }
            BuiltinId::Fold => {
                let shared = args.remove(0).into_shared_iter();
                let init = args.remove(0);
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Fn(f) = fn_val else {
                    panic!("fold: expected Fn")
                };
                Self::exec_hof_fold_inner(this, items, init, f, handle).await
            }
            BuiltinId::Any => {
                let shared = args.remove(0).into_shared_iter();
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Fn(f) = fn_val else {
                    panic!("any: expected Fn")
                };
                Self::exec_hof_any_inner(this, items, f, handle).await
            }
            BuiltinId::All => {
                let shared = args.remove(0).into_shared_iter();
                let fn_val = args.remove(0);
                let items;
                (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                let Value::Fn(f) = fn_val else {
                    panic!("all: expected Fn")
                };
                Self::exec_hof_all_inner(this, items, f, handle).await
            }

            // -- Deque ops --
            BuiltinId::Append => {
                let list = args.remove(0);
                let item = args.remove(0);
                let Value::Deque(mut deque) = list else {
                    panic!("append: expected Deque, got {list:?}")
                };
                deque.push(item);
                Ok((this, Value::Deque(deque)))
            }
            BuiltinId::Extend => {
                let list = args.remove(0);
                let shared = args.remove(0).into_shared_iter();
                let Value::Deque(mut deque) = list else {
                    panic!("extend: expected Deque as first arg")
                };
                let (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
                deque.extend(items);
                Ok((this, Value::Deque(deque)))
            }
            BuiltinId::Consume => {
                let list = args.remove(0);
                let n_val = args.remove(0);
                let Value::Deque(mut deque) = list else {
                    panic!("consume: expected Deque as first arg")
                };
                let Value::Int(n) = n_val else {
                    panic!("consume: expected Int as second arg")
                };
                deque.consume(n as usize);
                Ok((this, Value::Deque(deque)))
            }
        }
    }

    // -- drive_chain: eagerly execute the chain pipeline -----------------------

    fn drive_chain<'a>(
        this: Self,
        chain: &'a IterChain,
        handle: &'a YieldHandle<Value>,
    ) -> BoxFuture<'a, Result<(Self, Vec<Value>), RuntimeError>> {
        Box::pin(Self::drive_chain_inner(this, chain, handle))
    }

    async fn drive_chain_inner<'a>(
        mut this: Self,
        chain: &IterChain,
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Vec<Value>), RuntimeError> {
        let mut items: Vec<Value> = chain.source.to_vec();

        for op in &chain.ops {
            match op {
                IterOp::Map(f) => {
                    let mut result = Vec::with_capacity(items.len());
                    for item in items {
                        let mapped;
                        (this, mapped) = Self::call_closure(
                            this,
                            f.clone(),
                            vec![Arc::new(item)],
                            handle,
                        )
                        .await?;
                        result.push(mapped);
                    }
                    items = result;
                }
                IterOp::Filter(f) => {
                    let mut result = Vec::new();
                    for item in items {
                        let arc_item = Arc::new(item);
                        let keep;
                        (this, keep) = Self::call_closure(
                            this,
                            f.clone(),
                            vec![Arc::clone(&arc_item)],
                            handle,
                        )
                        .await?;
                        if matches!(keep, Value::Bool(true)) {
                            result.push(Arc::unwrap_or_clone(arc_item));
                        }
                    }
                    items = result;
                }
                IterOp::Take(n) => {
                    items.truncate(*n);
                }
                IterOp::Skip(n) => {
                    items = items.into_iter().skip(*n).collect();
                }
                IterOp::Chain(other) => {
                    let more;
                    (this, more) = Self::drive_chain(this, other, handle).await?;
                    items.extend(more);
                }
                IterOp::Flatten => {
                    let mut result = Vec::new();
                    for item in items {
                        match item {
                            Value::List(inner) => result.extend(inner),
                            Value::Deque(d) => result.extend(d.into_vec()),
                            other => result.push(other),
                        }
                    }
                    items = result;
                }
                IterOp::FlatMap(f) => {
                    let mut result = Vec::new();
                    for item in items {
                        let mapped;
                        (this, mapped) = Self::call_closure(
                            this,
                            f.clone(),
                            vec![Arc::new(item)],
                            handle,
                        )
                        .await?;
                        match mapped {
                            Value::List(inner) => result.extend(inner),
                            Value::Deque(d) => result.extend(d.into_vec()),
                            other => result.push(other),
                        }
                    }
                    items = result;
                }
            }
        }
        Ok((this, items))
    }

    // -- collect helpers ------------------------------------------------------

    async fn exec_collect<'a>(
        this: Self,
        shared: SharedIter,
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Value), RuntimeError> {
        let (this, items) = Self::exec_collect_vec(this, shared, handle).await?;
        Ok((this, Value::List(items)))
    }

    async fn exec_collect_vec<'a>(
        this: Self,
        shared: SharedIter,
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Vec<Value>), RuntimeError> {
        // Check cache first
        if let Some(cached) = shared.cached() {
            return Ok((this, cached));
        }

        let chain = shared.snapshot_chain();
        let (this, items) = Self::drive_chain(this, &chain, handle).await?;
        shared.set_cache(items.clone());
        Ok((this, items))
    }

    // -- consuming HOF inner implementations ----------------------------------

    async fn exec_hof_find_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let arc_item = Arc::new(item);
            let matched;
            (this, matched) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::clone(&arc_item)], handle)
                    .await?;
            if matches!(matched, Value::Bool(true)) {
                return Ok((this, Arc::unwrap_or_clone(arc_item)));
            }
        }
        Err(RuntimeError::empty_collection("find"))
    }

    async fn exec_hof_reduce_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<Value>,
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
        handle: &'a YieldHandle<Value>,
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
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let result;
            (this, result) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::new(item)], handle).await?;
            if matches!(result, Value::Bool(true)) {
                return Ok((this, Value::Bool(true)));
            }
        }
        Ok((this, Value::Bool(false)))
    }

    async fn exec_hof_all_inner<'a>(
        mut this: Self,
        items: Vec<Value>,
        fn_val: FnValue,
        handle: &'a YieldHandle<Value>,
    ) -> Result<(Self, Value), RuntimeError> {
        for item in items {
            let result;
            (this, result) =
                Self::call_closure(this, fn_val.clone(), vec![Arc::new(item)], handle).await?;
            if matches!(result, Value::Bool(false)) {
                return Ok((this, Value::Bool(false)));
            }
        }
        Ok((this, Value::Bool(true)))
    }

    // -- closure invocation ---------------------------------------------------

    async fn call_closure<'a>(
        this: Self,
        fn_val: FnValue,
        args: Vec<Arc<Value>>,
        handle: &'a YieldHandle<Value>,
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
        let (this, _, result) = Self::run(this, insts, frame, handle).await?;
        Ok((this, result.expect("closure must return a value")))
    }
}

// ---------------------------------------------------------------------------
// Value extractors — flat panic on type mismatch
// ---------------------------------------------------------------------------

fn expect_list<'a>(v: &'a Value, ctx: &str) -> &'a [Value] {
    match v {
        Value::List(items) => items,
        Value::Deque(d) => d.as_slice(),
        _ => panic!("{ctx}: expected List, got {v:?}"),
    }
}

fn expect_object<'a>(v: &'a Value, ctx: &str) -> &'a FxHashMap<Astr, Value> {
    match v {
        Value::Object(fields) => fields,
        _ => panic!("{ctx}: expected Object, got {v:?}"),
    }
}

fn expect_tuple<'a>(v: &'a Value, ctx: &str) -> &'a [Value] {
    match v {
        Value::Tuple(elems) => elems,
        _ => panic!("{ctx}: expected Tuple, got {v:?}"),
    }
}

fn expect_int(v: &Value, ctx: &str) -> i64 {
    match v {
        Value::Int(n) => *n,
        _ => panic!("{ctx}: expected Int, got {v:?}"),
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Byte(b) => Value::Byte(*b),
        Literal::List(elems) => {
            Value::Deque(TrackedDeque::from_vec(elems.iter().map(literal_to_value).collect()))
        }
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Byte(a), Value::Byte(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        (
            Value::Variant {
                tag: ta,
                payload: pa,
            },
            Value::Variant {
                tag: tb,
                payload: pb,
            },
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
            Value::Bool(matches!(left, Value::Bool(true)) && matches!(right, Value::Bool(true)))
        }
        BinOp::Or => {
            Value::Bool(matches!(left, Value::Bool(true)) || matches!(right, Value::Bool(true)))
        }
        BinOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.wrapping_add(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::String(a), Value::String(b)) => {
                let mut s = a.clone();
                s.push_str(b);
                Value::String(s)
            }
            (l, r) => panic!("Add: incompatible {l:?} + {r:?}"),
        },
        BinOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.wrapping_sub(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (l, r) => panic!("Sub: incompatible {l:?} - {r:?}"),
        },
        BinOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.wrapping_mul(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (l, r) => panic!("Mul: incompatible {l:?} * {r:?}"),
        },
        BinOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (l, r) => panic!("Div: incompatible {l:?} / {r:?}"),
        },
        BinOp::Mod => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a % b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a % b),
            (l, r) => panic!("Mod: incompatible {l:?} % {r:?}"),
        },
        BinOp::Xor => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a ^ b),
            (l, r) => panic!("Xor: incompatible {l:?} ^ {r:?}"),
        },
        BinOp::BitAnd => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a & b),
            (l, r) => panic!("BitAnd: incompatible {l:?} & {r:?}"),
        },
        BinOp::BitOr => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a | b),
            (l, r) => panic!("BitOr: incompatible {l:?} | {r:?}"),
        },
        BinOp::Shl => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a << b),
            (l, r) => panic!("Shl: incompatible {l:?} << {r:?}"),
        },
        BinOp::Shr => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a >> b),
            (l, r) => panic!("Shr: incompatible {l:?} >> {r:?}"),
        },
        BinOp::Eq => Value::Bool(values_equal(left, right)),
        BinOp::Neq => Value::Bool(!values_equal(left, right)),
        BinOp::Lt => cmp_values(left, right, Ordering::is_lt),
        BinOp::Gt => cmp_values(left, right, Ordering::is_gt),
        BinOp::Lte => cmp_values(left, right, Ordering::is_le),
        BinOp::Gte => cmp_values(left, right, Ordering::is_ge),
    }
}

fn cmp_values(left: &Value, right: &Value, f: fn(Ordering) -> bool) -> Value {
    let ord = match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap(),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (l, r) => panic!("comparison: incompatible {l:?} vs {r:?}"),
    };
    Value::Bool(f(ord))
}

fn eval_unaryop(op: UnaryOp, operand: &Value) -> Value {
    match op {
        UnaryOp::Neg => match operand {
            Value::Int(n) => Value::Int(-n),
            Value::Float(f) => Value::Float(-f),
            v => panic!("Neg: expected numeric, got {v:?}"),
        },
        UnaryOp::Not => match operand {
            Value::Bool(b) => Value::Bool(!b),
            v => panic!("Not: expected Bool, got {v:?}"),
        },
    }
}
