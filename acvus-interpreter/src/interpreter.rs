use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use futures::future::BoxFuture;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_mir::ir::{Inst, InstKind, Label, MirBody, MirModule, ValueId};

use crate::builtins;
use crate::extern_fn::ExternFnRegistry;
use crate::value::{FnValue, Value};

pub struct Interpreter {
    module: MirModule,
    context: HashMap<String, Value>,
    variables: HashMap<String, Value>,
    extern_fns: ExternFnRegistry,
}

// ---------------------------------------------------------------------------
// Frame — val store + sync instruction execution
// ---------------------------------------------------------------------------

struct Frame {
    vals: Vec<Option<Value>>,
    label_map: HashMap<Label, usize>,
    iters: HashMap<ValueId, IterState>,
}

enum IterState {
    List { items: Vec<Value>, pos: usize },
    Range { current: i64, end: i64, inclusive: bool },
}

impl Frame {
    fn new(val_count: u32, label_map: HashMap<Label, usize>) -> Self {
        Self {
            vals: vec![None; val_count as usize],
            label_map,
            iters: HashMap::new(),
        }
    }

    fn set(&mut self, id: ValueId, value: Value) {
        self.vals[id.0 as usize] = Some(value);
    }

    fn get(&self, id: ValueId) -> &Value {
        self.vals[id.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("Val({}) not yet defined", id.0))
    }

    fn take(&mut self, id: ValueId) -> Value {
        self.get(id).clone()
    }

    fn collect_args(&mut self, args: &[ValueId]) -> Vec<Value> {
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
            let arg_values = self.collect_args(args);
            for (param, val) in params.iter().zip(arg_values) {
                self.set(*param, val);
            }
        }
    }

    // -- iteration ------------------------------------------------------------

    fn iter_init(&mut self, dst: ValueId, src: ValueId) {
        let state = match self.get(src).clone() {
            Value::List(items) => IterState::List { items, pos: 0 },
            Value::Range { start, end, inclusive } => {
                IterState::Range { current: start, end, inclusive }
            }
            v => panic!("IterInit: expected List or Range, got {v:?}"),
        };
        self.iters.insert(dst, state);
        self.set(dst, Value::Unit);
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
            IterState::Range { current, end, inclusive }
                if (*inclusive && *current <= *end) || (!*inclusive && *current < *end) =>
            {
                let val = Value::Int(*current);
                *current += 1;
                (val, false)
            }
            IterState::Range { .. } => (Value::Unit, true),
        };

        self.set(dst_value, value);
        self.set(dst_done, Value::Bool(done));
    }
}

// ---------------------------------------------------------------------------
// Label map
// ---------------------------------------------------------------------------

fn build_label_map(body: &MirBody) -> HashMap<Label, usize> {
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
    pub fn new(module: MirModule, context: HashMap<String, Value>, extern_fns: ExternFnRegistry) -> Self {
        Self { module, context, variables: HashMap::new(), extern_fns }
    }

    pub async fn execute(self) -> (Self, String) {
        let insts = self.module.main.insts.clone();
        let label_map = build_label_map(&self.module.main);
        let frame = Frame::new(self.module.main.val_count, label_map);
        let (this, _, output, _) = Self::run(self, insts, frame, String::new()).await;
        (this, output)
    }

    // -- core exec loop -------------------------------------------------------

    fn run(
        this: Self,
        insts: Vec<Inst>,
        frame: Frame,
        output: String,
    ) -> BoxFuture<'static, (Self, Frame, String, Option<Value>)> {
        Box::pin(Self::run_inner(this, insts, frame, output))
    }

    async fn run_inner(
        mut this: Self,
        insts: Vec<Inst>,
        mut frame: Frame,
        mut output: String,
    ) -> (Self, Frame, String, Option<Value>) {
        let mut pc = 0;
        while pc < insts.len() {
            match &insts[pc].kind {
                // -- emit --
                InstKind::EmitText(idx) => output.push_str(&this.module.texts[*idx]),
                InstKind::EmitValue(v) => match frame.get(*v) {
                    Value::String(s) => output.push_str(s),
                    other => panic!("EmitValue: expected String, got {other:?}"),
                },

                // -- constants / constructors --
                InstKind::Const { dst, value } => {
                    frame.set(*dst, literal_to_value(value));
                }
                InstKind::MakeList { dst, elements } => {
                    let items = frame.collect_args(elements);
                    frame.set(*dst, Value::List(items));
                }
                InstKind::MakeObject { dst, fields } => {
                    let obj: BTreeMap<String, Value> =
                        fields.iter().map(|(k, v)| (k.clone(), frame.take(*v))).collect();
                    frame.set(*dst, Value::Object(obj));
                }
                InstKind::MakeRange { dst, start, end, kind } => {
                    let (s, e) = (frame.get(*start), frame.get(*end));
                    match (s, e) {
                        (Value::Int(s), Value::Int(e)) => frame.set(
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
                    frame.set(*dst, Value::Tuple(items));
                }
                InstKind::MakeClosure { dst, body, captures } => {
                    let captured = frame.collect_args(captures);
                    frame.set(*dst, Value::Fn(FnValue { body: *body, captures: captured }));
                }

                // -- arithmetic / logic --
                InstKind::BinOp { dst, op, left, right } => {
                    let (l, r) = (frame.take(*left), frame.take(*right));
                    frame.set(*dst, eval_binop(*op, l, r));
                }
                InstKind::UnaryOp { dst, op, operand } => {
                    let v = frame.take(*operand);
                    frame.set(*dst, eval_unaryop(*op, v));
                }

                // -- access --
                InstKind::FieldGet { dst, object, field } => {
                    let v = expect_object(frame.get(*object), "FieldGet")
                        .get(field.as_str())
                        .unwrap_or_else(|| panic!("FieldGet: key '{field}' not found"))
                        .clone();
                    frame.set(*dst, v);
                }
                InstKind::ObjectGet { dst, object, key } => {
                    let v = expect_object(frame.get(*object), "ObjectGet")
                        .get(key.as_str())
                        .unwrap_or_else(|| panic!("ObjectGet: key '{key}' not found"))
                        .clone();
                    frame.set(*dst, v);
                }
                InstKind::TupleIndex { dst, tuple, index } => {
                    let v = expect_tuple(frame.get(*tuple), "TupleIndex")[*index].clone();
                    frame.set(*dst, v);
                }
                InstKind::ListIndex { dst, list, index } => {
                    let items = expect_list(frame.get(*list), "ListIndex");
                    let i = if *index >= 0 {
                        *index as usize
                    } else {
                        (items.len() as i32 + *index) as usize
                    };
                    frame.set(*dst, items[i].clone());
                }
                InstKind::ListSlice { dst, list, skip_head, skip_tail } => {
                    let items = expect_list(frame.get(*list), "ListSlice");
                    let end = items.len() - *skip_tail;
                    frame.set(*dst, Value::List(items[*skip_head..end].to_vec()));
                }

                // -- pattern testing --
                InstKind::TestLiteral { dst, src, value } => {
                    let eq = values_equal(frame.get(*src), &literal_to_value(value));
                    frame.set(*dst, Value::Bool(eq));
                }
                InstKind::TestListLen { dst, src, min_len, exact } => {
                    let items = expect_list(frame.get(*src), "TestListLen");
                    let ok = if *exact { items.len() == *min_len } else { items.len() >= *min_len };
                    frame.set(*dst, Value::Bool(ok));
                }
                InstKind::TestObjectKey { dst, src, key } => {
                    let ok = expect_object(frame.get(*src), "TestObjectKey").contains_key(key.as_str());
                    frame.set(*dst, Value::Bool(ok));
                }
                InstKind::TestRange { dst, src, start, end, kind } => {
                    let n = expect_int(frame.get(*src), "TestRange");
                    let ok = match kind {
                        RangeKind::Exclusive => n >= *start && n < *end,
                        RangeKind::InclusiveEnd => n >= *start && n <= *end,
                        RangeKind::ExclusiveStart => n > *start && n <= *end,
                    };
                    frame.set(*dst, Value::Bool(ok));
                }

                // -- iteration --
                InstKind::IterInit { dst, src } => frame.iter_init(*dst, *src),
                InstKind::IterNext { dst_value, dst_done, iter } => {
                    frame.iter_next(*dst_value, *dst_done, *iter);
                }

                // -- context / variable I/O --
                InstKind::ContextLoad { dst, name } => {
                    let v = this.context.get(name)
                        .unwrap_or_else(|| panic!("ContextLoad: undefined context @{name}"))
                        .clone();
                    frame.set(*dst, v);
                }
                InstKind::VarLoad { dst, name } => {
                    let v = this.variables.get(name)
                        .unwrap_or_else(|| panic!("VarLoad: undefined variable ${name}"))
                        .clone();
                    frame.set(*dst, v);
                }
                InstKind::VarStore { name, src } => {
                    let v = frame.take(*src);
                    this.variables.insert(name.clone(), v);
                }

                // -- calls (async, ownership-passing) --
                InstKind::Call { dst, func, args } => {
                    let arg_values = frame.collect_args(args);
                    let result;
                    if builtins::is_builtin(func) {
                        (this, output, result) =
                            Self::exec_builtin(this, func, arg_values, output).await;
                    } else {
                        let f = this.extern_fns.get(func)
                            .unwrap_or_else(|| panic!("unknown function: {func}"));
                        result = f.call(arg_values).await;
                    }
                    frame.set(*dst, result);
                }
                InstKind::AsyncCall { dst, func, args } => {
                    let arg_values = frame.collect_args(args);
                    let f = this.extern_fns.get(func)
                        .unwrap_or_else(|| panic!("unknown async function: {func}"));
                    frame.set(*dst, f.call(arg_values).await);
                }
                InstKind::CallClosure { dst, closure, args } => {
                    let fn_val = expect_fn(frame.get(*closure).clone(), "CallClosure");
                    let arg_values = frame.collect_args(args);
                    let result;
                    (this, output, result) =
                        Self::call_closure(this, fn_val, arg_values, output).await;
                    frame.set(*dst, result);
                }
                InstKind::Await { dst, src } => {
                    let v = frame.take(*src);
                    frame.set(*dst, v);
                }

                // -- control flow --
                InstKind::BlockLabel { .. } => {}
                InstKind::Jump { label, args } => {
                    pc = frame.jump(&insts, label, args);
                    continue;
                }
                InstKind::JumpIf { cond, then_label, then_args, else_label, else_args } => {
                    pc = frame.jump_if(
                        &insts, *cond,
                        (then_label, then_args),
                        (else_label, else_args),
                    );
                    continue;
                }
                InstKind::Return(val) => {
                    let v = frame.take(*val);
                    return (this, frame, output, Some(v));
                }
                InstKind::Nop => {}
            }
            pc += 1;
        }
        (this, frame, output, None)
    }

    // -- builtin dispatch -----------------------------------------------------

    async fn exec_builtin(
        this: Self,
        name: &str,
        args: Vec<Value>,
        output: String,
    ) -> (Self, String, Value) {
        match name {
            "to_string" | "to_int" | "to_float" | "char_to_int" | "int_to_char" | "len"
            | "reverse" | "join" | "contains" => {
                (this, output, builtins::call_pure(name, args))
            }
            "filter" => Self::exec_hof_filter(this, args, output).await,
            "map" | "pmap" => Self::exec_hof_map(this, args, output).await,
            "find" => Self::exec_hof_find(this, args, output).await,
            "reduce" => Self::exec_hof_reduce(this, args, output).await,
            "fold" => Self::exec_hof_fold(this, args, output).await,
            "any" => Self::exec_hof_any(this, args, output).await,
            "all" => Self::exec_hof_all(this, args, output).await,
            _ => panic!("unknown builtin: {name}"),
        }
    }

    async fn exec_hof_filter(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "filter");
        let mut result = Vec::new();
        for item in items {
            let keep;
            (this, output, keep) =
                Self::call_closure(this, fn_val.clone(), vec![item.clone()], output).await;
            if matches!(keep, Value::Bool(true)) {
                result.push(item);
            }
        }
        (this, output, Value::List(result))
    }

    async fn exec_hof_map(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "map");
        let mut result = Vec::new();
        for item in items {
            let mapped;
            (this, output, mapped) =
                Self::call_closure(this, fn_val.clone(), vec![item], output).await;
            result.push(mapped);
        }
        (this, output, Value::List(result))
    }

    async fn exec_hof_find(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "find");
        for item in items {
            let result;
            (this, output, result) =
                Self::call_closure(this, fn_val.clone(), vec![item.clone()], output).await;
            if matches!(result, Value::Bool(true)) {
                return (this, output, item);
            }
        }
        panic!("find: no element matched the predicate");
    }

    async fn exec_hof_reduce(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "reduce");
        let mut it = items.into_iter();
        let mut acc = it.next().expect("reduce: empty list");
        for item in it {
            (this, output, acc) =
                Self::call_closure(this, fn_val.clone(), vec![acc, item], output).await;
        }
        (this, output, acc)
    }

    async fn exec_hof_fold(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let mut it = args.into_iter();
        let list = it.next().unwrap();
        let init = it.next().unwrap();
        let closure = it.next().unwrap();
        let (items, fn_val) = match (list, closure) {
            (Value::List(items), Value::Fn(fn_val)) => (items, fn_val),
            (l, c) => panic!("fold: expected (List, _, Fn), got ({l:?}, _, {c:?})"),
        };
        let mut acc = init;
        for item in items {
            (this, output, acc) =
                Self::call_closure(this, fn_val.clone(), vec![acc, item], output).await;
        }
        (this, output, acc)
    }

    async fn exec_hof_any(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "any");
        for item in items {
            let result;
            (this, output, result) =
                Self::call_closure(this, fn_val.clone(), vec![item], output).await;
            if matches!(result, Value::Bool(true)) {
                return (this, output, Value::Bool(true));
            }
        }
        (this, output, Value::Bool(false))
    }

    async fn exec_hof_all(
        mut this: Self,
        args: Vec<Value>,
        mut output: String,
    ) -> (Self, String, Value) {
        let (items, fn_val) = extract_list_fn(args, "all");
        for item in items {
            let result;
            (this, output, result) =
                Self::call_closure(this, fn_val.clone(), vec![item], output).await;
            if matches!(result, Value::Bool(false)) {
                return (this, output, Value::Bool(false));
            }
        }
        (this, output, Value::Bool(true))
    }

    // -- closure invocation ---------------------------------------------------

    async fn call_closure(
        this: Self,
        fn_val: FnValue,
        args: Vec<Value>,
        output: String,
    ) -> (Self, String, Value) {
        let closure_body = this
            .module
            .closures
            .get(&fn_val.body)
            .unwrap_or_else(|| panic!("closure body not found: {:?}", fn_val.body));

        let label_map = build_label_map(&closure_body.body);
        let mut frame = Frame::new(closure_body.body.val_count, label_map);

        let n_captures = fn_val.captures.len();
        for (i, cap) in fn_val.captures.iter().enumerate() {
            frame.set(ValueId(i as u32), cap.clone());
        }
        for (i, arg) in args.into_iter().enumerate() {
            frame.set(ValueId((n_captures + i) as u32), arg);
        }

        let insts = closure_body.body.insts.clone();
        let (this, _, output, result) = Self::run(this, insts, frame, output).await;
        (this, output, result.expect("closure must return a value"))
    }
}

// ---------------------------------------------------------------------------
// Value extractors — flat panic on type mismatch
// ---------------------------------------------------------------------------

fn expect_list<'a>(v: &'a Value, ctx: &str) -> &'a [Value] {
    match v {
        Value::List(items) => items,
        _ => panic!("{ctx}: expected List, got {v:?}"),
    }
}

fn expect_object<'a>(v: &'a Value, ctx: &str) -> &'a BTreeMap<String, Value> {
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

fn expect_fn(v: Value, ctx: &str) -> FnValue {
    match v {
        Value::Fn(f) => f,
        _ => panic!("{ctx}: expected Fn, got {v:?}"),
    }
}

fn extract_list_fn(args: Vec<Value>, ctx: &str) -> (Vec<Value>, FnValue) {
    let mut it = args.into_iter();
    let list = it.next().unwrap();
    let closure = it.next().unwrap();
    match (list, closure) {
        (Value::List(items), Value::Fn(fn_val)) => (items, fn_val),
        (l, c) => panic!("{ctx}: expected (List, Fn), got ({l:?}, {c:?})"),
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
        BinOp::And => Value::Bool(
            matches!(left, Value::Bool(true)) && matches!(right, Value::Bool(true)),
        ),
        BinOp::Or => Value::Bool(
            matches!(left, Value::Bool(true)) || matches!(right, Value::Bool(true)),
        ),
        BinOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::String(a), Value::String(b)) => Value::String(a + &b),
            (l, r) => panic!("Add: incompatible {l:?} + {r:?}"),
        },
        BinOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (l, r) => panic!("Sub: incompatible {l:?} - {r:?}"),
        },
        BinOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (l, r) => panic!("Mul: incompatible {l:?} * {r:?}"),
        },
        BinOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (l, r) => panic!("Div: incompatible {l:?} / {r:?}"),
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
        BinOp::Eq => Value::Bool(values_equal(&left, &right)),
        BinOp::Neq => Value::Bool(!values_equal(&left, &right)),
        BinOp::Lt => cmp_values(&left, &right, Ordering::is_lt),
        BinOp::Gt => cmp_values(&left, &right, Ordering::is_gt),
        BinOp::Lte => cmp_values(&left, &right, Ordering::is_le),
        BinOp::Gte => cmp_values(&left, &right, Ordering::is_ge),
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
