use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::ir::{Inst, InstKind, Label, MirBody, MirModule, ValueId};

use crate::builtins;
use crate::extern_fn::ExternFnRegistry;
use crate::storage::Storage;
use crate::value::{FnValue, StorageKey, Value};

pub struct Interpreter<S>
where
    S: Storage,
    S::Error: std::fmt::Debug,
{
    module: MirModule,
    storage: S,
    extern_fns: ExternFnRegistry,
}

struct Frame {
    vals: Vec<Option<Value>>,
    label_map: HashMap<Label, usize>,
    iters: HashMap<ValueId, IterState>,
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

// ---------------------------------------------------------------------------
// Frame — val storage + pure instruction execution
// ---------------------------------------------------------------------------

impl Frame {
    fn new(val_count: u32, label_map: HashMap<Label, usize>) -> Self {
        Self {
            vals: vec![None; val_count as usize],
            label_map,
            iters: HashMap::new(),
        }
    }

    fn set(&mut self, val: ValueId, value: Value) {
        self.vals[val.0 as usize] = Some(value);
    }

    fn get(&self, val: ValueId) -> &Value {
        self.vals[val.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("Val({}) not yet defined", val.0))
    }

    fn take(&mut self, val: ValueId) -> Value {
        self.get(val).clone()
    }

    fn collect_args(&mut self, args: &[ValueId]) -> Vec<Value> {
        args.iter().map(|v| self.take(*v)).collect()
    }

    fn resolve_label(&self, label: &Label) -> usize {
        *self
            .label_map
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {:?}", label))
    }

    // -- output ---------------------------------------------------------------

    fn emit_value(&self, val: ValueId, output: &mut String) {
        let v = self.get(val);
        match v {
            Value::String(s) => output.push_str(s),
            _ => panic!("EmitValue: expected String, got {v:?}"),
        }
    }

    // -- constants / constructors ---------------------------------------------

    fn exec_const(&mut self, dst: ValueId, value: &Literal) {
        self.set(dst, literal_to_value(value));
    }

    fn exec_make_list(&mut self, dst: ValueId, elements: &[ValueId]) {
        let items = self.collect_args(elements);
        self.set(dst, Value::List(items));
    }

    fn exec_make_object(&mut self, dst: ValueId, fields: &[(String, ValueId)]) {
        let obj: BTreeMap<String, Value> = fields
            .iter()
            .map(|(k, v)| (k.clone(), self.take(*v)))
            .collect();
        self.set(dst, Value::Object(obj));
    }

    fn exec_make_range(&mut self, dst: ValueId, start: ValueId, end: ValueId, kind: &RangeKind) {
        let s = self.get(start);
        let e = self.get(end);
        match (s, e) {
            (Value::Int(s), Value::Int(e)) => {
                let inclusive = matches!(kind, RangeKind::InclusiveEnd);
                self.set(dst, Value::Range { start: *s, end: *e, inclusive });
            }
            _ => panic!("MakeRange: expected Int bounds"),
        }
    }

    fn exec_make_tuple(&mut self, dst: ValueId, elements: &[ValueId]) {
        let items = self.collect_args(elements);
        self.set(dst, Value::Tuple(items));
    }

    fn exec_make_closure(&mut self, dst: ValueId, body: Label, captures: &[ValueId]) {
        let captured = self.collect_args(captures);
        self.set(dst, Value::Fn(FnValue { body, captures: captured }));
    }

    // -- arithmetic / logic ---------------------------------------------------

    fn exec_binop(&mut self, dst: ValueId, op: BinOp, left: ValueId, right: ValueId) {
        let l = self.take(left);
        let r = self.take(right);
        self.set(dst, eval_binop(op, l, r));
    }

    fn exec_unaryop(&mut self, dst: ValueId, op: UnaryOp, operand: ValueId) {
        let v = self.take(operand);
        self.set(dst, eval_unaryop(op, v));
    }

    // -- field / index access -------------------------------------------------

    fn exec_field_get(&mut self, dst: ValueId, object: ValueId, field: &str) {
        let obj = self.get(object);
        match obj {
            Value::Object(fields) => {
                let v = fields
                    .get(field)
                    .unwrap_or_else(|| panic!("FieldGet: key '{field}' not found"))
                    .clone();
                self.set(dst, v);
            }
            _ => panic!("FieldGet: expected Object, got {obj:?}"),
        }
    }

    fn exec_tuple_index(&mut self, dst: ValueId, tuple: ValueId, index: usize) {
        let t = self.get(tuple);
        match t {
            Value::Tuple(elems) => self.set(dst, elems[index].clone()),
            _ => panic!("TupleIndex: expected Tuple, got {t:?}"),
        }
    }

    fn exec_list_index(&mut self, dst: ValueId, list: ValueId, index: i32) {
        let l = self.get(list);
        match l {
            Value::List(items) => {
                let actual = if index >= 0 {
                    index as usize
                } else {
                    (items.len() as i32 + index) as usize
                };
                self.set(dst, items[actual].clone());
            }
            _ => panic!("ListIndex: expected List, got {l:?}"),
        }
    }

    fn exec_list_slice(&mut self, dst: ValueId, list: ValueId, skip_head: usize, skip_tail: usize) {
        let l = self.get(list);
        match l {
            Value::List(items) => {
                let end = items.len() - skip_tail;
                self.set(dst, Value::List(items[skip_head..end].to_vec()));
            }
            _ => panic!("ListSlice: expected List, got {l:?}"),
        }
    }

    fn exec_object_get(&mut self, dst: ValueId, object: ValueId, key: &str) {
        let obj = self.get(object);
        match obj {
            Value::Object(fields) => {
                let v = fields
                    .get(key)
                    .unwrap_or_else(|| panic!("ObjectGet: key '{key}' not found"))
                    .clone();
                self.set(dst, v);
            }
            _ => panic!("ObjectGet: expected Object, got {obj:?}"),
        }
    }

    // -- pattern testing ------------------------------------------------------

    fn exec_test_literal(&mut self, dst: ValueId, src: ValueId, value: &Literal) {
        let v = self.get(src);
        let expected = literal_to_value(value);
        self.set(dst, Value::Bool(values_equal(v, &expected)));
    }

    fn exec_test_list_len(&mut self, dst: ValueId, src: ValueId, min_len: usize, exact: bool) {
        let v = self.get(src);
        match v {
            Value::List(items) => {
                let ok = if exact { items.len() == min_len } else { items.len() >= min_len };
                self.set(dst, Value::Bool(ok));
            }
            _ => panic!("TestListLen: expected List, got {v:?}"),
        }
    }

    fn exec_test_object_key(&mut self, dst: ValueId, src: ValueId, key: &str) {
        let v = self.get(src);
        match v {
            Value::Object(fields) => self.set(dst, Value::Bool(fields.contains_key(key))),
            _ => panic!("TestObjectKey: expected Object, got {v:?}"),
        }
    }

    fn exec_test_range(&mut self, dst: ValueId, src: ValueId, start: i64, end: i64, kind: &RangeKind) {
        let v = self.get(src);
        match v {
            Value::Int(n) => {
                let in_range = match kind {
                    RangeKind::Exclusive => *n >= start && *n < end,
                    RangeKind::InclusiveEnd => *n >= start && *n <= end,
                    RangeKind::ExclusiveStart => *n > start && *n <= end,
                };
                self.set(dst, Value::Bool(in_range));
            }
            _ => panic!("TestRange: expected Int, got {v:?}"),
        }
    }

    // -- iteration ------------------------------------------------------------

    fn exec_iter_init(&mut self, dst: ValueId, src: ValueId) {
        let v = self.get(src).clone();
        let state = match v {
            Value::List(items) => IterState::List { items, pos: 0 },
            Value::Range { start, end, inclusive } => {
                IterState::Range { current: start, end, inclusive }
            }
            _ => panic!("IterInit: expected List or Range, got {v:?}"),
        };
        self.iters.insert(dst, state);
        self.set(dst, Value::Unit);
    }

    fn exec_iter_next(&mut self, dst_value: ValueId, dst_done: ValueId, iter: ValueId) {
        let state = self
            .iters
            .get_mut(&iter)
            .unwrap_or_else(|| panic!("IterNext: no iterator for Val({})", iter.0));
        match state {
            IterState::List { items, pos } => {
                if *pos < items.len() {
                    let val = items[*pos].clone();
                    *pos += 1;
                    self.set(dst_value, val);
                    self.set(dst_done, Value::Bool(false));
                } else {
                    self.set(dst_value, Value::Unit);
                    self.set(dst_done, Value::Bool(true));
                }
            }
            IterState::Range { current, end, inclusive } => {
                let done = if *inclusive { *current > *end } else { *current >= *end };
                if !done {
                    let val = *current;
                    *current += 1;
                    self.set(dst_value, Value::Int(val));
                    self.set(dst_done, Value::Bool(false));
                } else {
                    self.set(dst_value, Value::Unit);
                    self.set(dst_done, Value::Bool(true));
                }
            }
        }
    }

    // -- control flow ---------------------------------------------------------

    fn exec_jump(&mut self, insts: &[Inst], label: &Label, args: &[ValueId]) -> usize {
        let target = self.resolve_label(label);
        self.bind_block_params(insts, target, args);
        target
    }

    fn exec_jump_if(
        &mut self,
        insts: &[Inst],
        cond: ValueId,
        then_label: &Label,
        then_args: &[ValueId],
        else_label: &Label,
        else_args: &[ValueId],
    ) -> usize {
        let is_true = matches!(self.get(cond), Value::Bool(true));
        let (label, args) = if is_true {
            (then_label, then_args)
        } else {
            (else_label, else_args)
        };
        let target = self.resolve_label(label);
        self.bind_block_params(insts, target, args);
        target
    }

    fn bind_block_params(&mut self, insts: &[Inst], target: usize, args: &[ValueId]) {
        if let InstKind::BlockLabel { params, .. } = &insts[target].kind {
            let arg_values = self.collect_args(args);
            for (param, val) in params.iter().zip(arg_values) {
                self.set(*param, val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Label map
// ---------------------------------------------------------------------------

fn build_label_map(body: &MirBody) -> HashMap<Label, usize> {
    let mut map = HashMap::new();
    for (i, inst) in body.insts.iter().enumerate() {
        if let InstKind::BlockLabel { label, .. } = &inst.kind {
            map.insert(*label, i);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Interpreter — async dispatch + exec loop
// ---------------------------------------------------------------------------

impl<S> Interpreter<S>
where
    S: Storage,
    S::Error: std::fmt::Debug,
{
    pub fn new(module: MirModule, storage: S, extern_fns: ExternFnRegistry) -> Self {
        Self {
            module,
            storage,
            extern_fns,
        }
    }

    pub async fn execute(&mut self) -> String {
        let mut output = String::new();
        let label_map = build_label_map(&self.module.main);
        let mut frame = Frame::new(self.module.main.val_count, label_map);
        self.exec_body(&self.module.main.insts.clone(), &mut frame, &mut output)
            .await;
        output
    }

    fn exec_body<'a>(
        &'a mut self,
        insts: &'a [Inst],
        frame: &'a mut Frame,
        output: &'a mut String,
    ) -> Pin<Box<dyn Future<Output = Option<Value>> + 'a>> {
        Box::pin(self.exec_body_inner(insts, frame, output))
    }

    async fn exec_body_inner(
        &mut self,
        insts: &[Inst],
        frame: &mut Frame,
        output: &mut String,
    ) -> Option<Value> {
        let mut pc = 0;
        while pc < insts.len() {
            match &insts[pc].kind {
                InstKind::EmitText(idx) => output.push_str(&self.module.texts[*idx]),
                InstKind::EmitValue(val) => frame.emit_value(*val, output),
                InstKind::Const { dst, value } => frame.exec_const(*dst, value),
                InstKind::StorageLoad { dst, name } => {
                    self.exec_storage_load(frame, *dst, name).await;
                }
                InstKind::StorageStore { name, src } => {
                    self.exec_storage_store(frame, name, *src).await;
                }
                InstKind::BinOp { dst, op, left, right } => {
                    frame.exec_binop(*dst, *op, *left, *right);
                }
                InstKind::UnaryOp { dst, op, operand } => frame.exec_unaryop(*dst, *op, *operand),
                InstKind::FieldGet { dst, object, field } => {
                    frame.exec_field_get(*dst, *object, field);
                }
                InstKind::Call { dst, func, args } => {
                    let result = self.exec_call(frame, func, args, output).await;
                    frame.set(*dst, result);
                }
                InstKind::AsyncCall { dst, func, args } => {
                    let result = self.exec_async_call(frame, func, args).await;
                    frame.set(*dst, result);
                }
                InstKind::Await { dst, src } => {
                    let v = frame.take(*src);
                    frame.set(*dst, v);
                }
                InstKind::MakeList { dst, elements } => frame.exec_make_list(*dst, elements),
                InstKind::MakeObject { dst, fields } => frame.exec_make_object(*dst, fields),
                InstKind::MakeRange { dst, start, end, kind } => {
                    frame.exec_make_range(*dst, *start, *end, kind);
                }
                InstKind::MakeTuple { dst, elements } => frame.exec_make_tuple(*dst, elements),
                InstKind::TupleIndex { dst, tuple, index } => {
                    frame.exec_tuple_index(*dst, *tuple, *index);
                }
                InstKind::TestLiteral { dst, src, value } => {
                    frame.exec_test_literal(*dst, *src, value);
                }
                InstKind::TestListLen { dst, src, min_len, exact } => {
                    frame.exec_test_list_len(*dst, *src, *min_len, *exact);
                }
                InstKind::TestObjectKey { dst, src, key } => {
                    frame.exec_test_object_key(*dst, *src, key);
                }
                InstKind::TestRange { dst, src, start, end, kind } => {
                    frame.exec_test_range(*dst, *src, *start, *end, kind);
                }
                InstKind::ListIndex { dst, list, index } => {
                    frame.exec_list_index(*dst, *list, *index);
                }
                InstKind::ListSlice { dst, list, skip_head, skip_tail } => {
                    frame.exec_list_slice(*dst, *list, *skip_head, *skip_tail);
                }
                InstKind::ObjectGet { dst, object, key } => {
                    frame.exec_object_get(*dst, *object, key);
                }
                InstKind::MakeClosure { dst, body, captures } => {
                    frame.exec_make_closure(*dst, *body, captures);
                }
                InstKind::CallClosure { dst, closure, args } => {
                    let result = self.exec_call_closure(frame, *closure, args, output).await;
                    frame.set(*dst, result);
                }
                InstKind::IterInit { dst, src } => frame.exec_iter_init(*dst, *src),
                InstKind::IterNext { dst_value, dst_done, iter } => {
                    frame.exec_iter_next(*dst_value, *dst_done, *iter);
                }
                InstKind::BlockLabel { .. } => {}
                InstKind::Jump { label, args } => {
                    pc = frame.exec_jump(insts, label, args);
                    continue;
                }
                InstKind::JumpIf { cond, then_label, then_args, else_label, else_args } => {
                    pc = frame.exec_jump_if(
                        insts, *cond, then_label, then_args, else_label, else_args,
                    );
                    continue;
                }
                InstKind::Return(val) => return Some(frame.take(*val)),
                InstKind::Nop => {}
            }
            pc += 1;
        }
        None
    }

    // -- async instruction helpers --------------------------------------------

    async fn exec_storage_load(&mut self, frame: &mut Frame, dst: ValueId, name: &str) {
        let key = StorageKey::root(name);
        let pure = self
            .storage
            .get(&key)
            .await
            .unwrap_or_else(|e| panic!("StorageLoad failed for '{name}': {e:?}"));
        frame.set(dst, Value::from_pure(pure));
    }

    async fn exec_storage_store(&mut self, frame: &mut Frame, name: &str, src: ValueId) {
        let key = StorageKey::root(name);
        let v = frame.take(src);
        self.storage
            .set(&key, v.into_pure())
            .await
            .unwrap_or_else(|e| panic!("StorageStore failed for '{name}': {e:?}"));
    }

    async fn exec_call(
        &mut self,
        frame: &mut Frame,
        func: &str,
        args: &[ValueId],
        output: &mut String,
    ) -> Value {
        let arg_values = frame.collect_args(args);
        if builtins::is_builtin(func) {
            builtins::call(func, arg_values, async |fn_val, args| {
                self.call_closure(fn_val, args, output).await
            })
            .await
        } else {
            let f = self
                .extern_fns
                .get(func)
                .unwrap_or_else(|| panic!("unknown function: {func}"));
            f.call(arg_values).await
        }
    }

    async fn exec_async_call(
        &mut self,
        frame: &mut Frame,
        func: &str,
        args: &[ValueId],
    ) -> Value {
        let arg_values = frame.collect_args(args);
        let f = self
            .extern_fns
            .get(func)
            .unwrap_or_else(|| panic!("unknown async function: {func}"));
        f.call(arg_values).await
    }

    async fn exec_call_closure(
        &mut self,
        frame: &mut Frame,
        closure: ValueId,
        args: &[ValueId],
        output: &mut String,
    ) -> Value {
        let closure_val = frame.get(closure).clone();
        let arg_values = frame.collect_args(args);
        match closure_val {
            Value::Fn(fn_val) => self.call_closure(fn_val, arg_values, output).await,
            _ => panic!("CallClosure: expected Fn, got {closure_val:?}"),
        }
    }

    // -- closure invocation ---------------------------------------------------

    async fn call_closure(
        &mut self,
        fn_val: FnValue,
        args: Vec<Value>,
        output: &mut String,
    ) -> Value {
        let closure_body = self
            .module
            .closures
            .get(&fn_val.body)
            .unwrap_or_else(|| panic!("closure body not found for label {:?}", fn_val.body));

        let label_map = build_label_map(&closure_body.body);
        let mut closure_frame = Frame::new(closure_body.body.val_count, label_map);

        let n_captures = fn_val.captures.len();
        for (i, cap) in fn_val.captures.iter().enumerate() {
            closure_frame.set(ValueId(i as u32), cap.clone());
        }
        for (i, arg) in args.into_iter().enumerate() {
            closure_frame.set(ValueId((n_captures + i) as u32), arg);
        }

        let closure_insts = closure_body.body.insts.clone();
        self.exec_body(&closure_insts, &mut closure_frame, output)
            .await
            .expect("closure must return a value")
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (no interpreter/frame state needed)
// ---------------------------------------------------------------------------

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Bool(b) => Value::Bool(*b),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

fn eval_binop(op: BinOp, left: Value, right: Value) -> Value {
    match op {
        BinOp::And => {
            let l = matches!(left, Value::Bool(true));
            let r = matches!(right, Value::Bool(true));
            Value::Bool(l && r)
        }
        BinOp::Or => {
            let l = matches!(left, Value::Bool(true));
            let r = matches!(right, Value::Bool(true));
            Value::Bool(l || r)
        }
        BinOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::String(a), Value::String(b)) => Value::String(a + &b),
            (l, r) => panic!("Add: incompatible types {l:?} + {r:?}"),
        },
        BinOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (l, r) => panic!("Sub: incompatible types {l:?} - {r:?}"),
        },
        BinOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (l, r) => panic!("Mul: incompatible types {l:?} * {r:?}"),
        },
        BinOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (l, r) => panic!("Div: incompatible types {l:?} / {r:?}"),
        },
        BinOp::Eq => Value::Bool(values_equal(&left, &right)),
        BinOp::Neq => Value::Bool(!values_equal(&left, &right)),
        BinOp::Lt => cmp_values(&left, &right, |o| o.is_lt()),
        BinOp::Gt => cmp_values(&left, &right, |o| o.is_gt()),
        BinOp::Lte => cmp_values(&left, &right, |o| o.is_le()),
        BinOp::Gte => cmp_values(&left, &right, |o| o.is_ge()),
    }
}

fn cmp_values(left: &Value, right: &Value, f: impl Fn(Ordering) -> bool) -> Value {
    let ord = match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap(),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (l, r) => panic!("comparison: incompatible types {l:?} vs {r:?}"),
    };
    Value::Bool(f(ord))
}

fn eval_unaryop(op: UnaryOp, operand: Value) -> Value {
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
