//! Builtin function implementations.
//!
//! Each builtin is a plain `fn(Args) -> Result<Value, RuntimeError>`.
//! Registered by name via `register_all`.

use acvus_mir::graph::FunctionId;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use acvus_mir::ty::Effect;

use crate::error::RuntimeError;
use crate::interpreter::{Args, BuiltinHandler, SyncBuiltinFn};
use crate::iter::IterHandle;
use crate::value::Value;

// ── Registration ─────────────────────────────────────────────────────

/// Build all builtin handlers as a HashMap.
/// Maps builtin name → FunctionId from the compilation graph.
pub fn build_builtins(
    builtin_ids: &FxHashMap<Astr, FunctionId>,
    interner: &Interner,
) -> FxHashMap<FunctionId, BuiltinHandler> {
    let mut map = FxHashMap::default();

    let mut reg = |name: &str, handler: SyncBuiltinFn| {
        let aname = interner.intern(name);
        if let Some(&id) = builtin_ids.get(&aname) {
            map.insert(id, BuiltinHandler::Sync(handler));
        }
    };

    // String ops
    reg("len_str", builtin_len_str);
    reg("trim", builtin_trim);
    reg("trim_start", builtin_trim_start);
    reg("trim_end", builtin_trim_end);
    reg("upper", builtin_upper);
    reg("lower", builtin_lower);
    reg("contains_str", builtin_contains_str);
    reg("starts_with_str", builtin_starts_with_str);
    reg("ends_with_str", builtin_ends_with_str);
    reg("replace_str", builtin_replace_str);
    reg("split_str", builtin_split_str);
    reg("repeat_str", builtin_repeat_str);
    reg("substring", builtin_substring);

    // List ops
    reg("len", builtin_len);
    reg("reverse", builtin_reverse);

    // Conversions
    reg("to_string", builtin_to_string);
    reg("to_int", builtin_to_int);
    reg("to_float", builtin_to_float);

    // Conversion (extra)
    reg("char_to_int", builtin_char_to_int);
    reg("int_to_char", builtin_int_to_char);

    // Bytes
    reg("to_bytes", builtin_to_bytes);
    reg("to_utf8", builtin_to_utf8);
    reg("to_utf8_lossy", builtin_to_utf8_lossy);

    // Option ops
    reg("unwrap", builtin_unwrap);
    reg("unwrap_or", builtin_unwrap_or);

    // Deque ops
    reg("append", builtin_append);
    reg("extend", builtin_extend);
    reg("consume", builtin_consume);

    // Iterator constructors
    reg("iter", builtin_iter);
    reg("rev_iter", builtin_rev_iter);

    // Iterator lazy combinators
    reg("map", builtin_map);
    reg("pmap", builtin_pmap);
    reg("filter", builtin_filter);
    reg("take", builtin_take);
    reg("skip", builtin_skip);
    reg("chain", builtin_chain);
    reg("flatten", builtin_flatten);
    reg("flat_map", builtin_flat_map);

    // Sequence lazy ops
    reg("take_seq", builtin_take_seq);
    reg("skip_seq", builtin_skip_seq);
    reg("chain_seq", builtin_chain_seq);

    // Async builtins (iterator consumers)
    let mut reg_async = |name: &str, handler: crate::interpreter::AsyncBuiltinFn| {
        let aname = interner.intern(name);
        if let Some(&id) = builtin_ids.get(&aname) {
            map.insert(id, BuiltinHandler::Async(handler));
        }
    };

    reg_async("collect", builtin_collect);
    reg_async("join", builtin_join);
    reg_async("first", builtin_first);
    reg_async("last", builtin_last);
    reg_async("contains", builtin_contains);
    reg_async("next", builtin_next);
    reg_async("next_seq", builtin_next_seq);
    reg_async("find", builtin_find);
    reg_async("reduce", builtin_reduce);
    reg_async("fold", builtin_fold);
    reg_async("any", builtin_any);
    reg_async("all", builtin_all);

    map
}

// ── String builtins ──────────────────────────────────────────────────

fn builtin_len_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::Int(args[0].as_str().len() as i64))
}

fn builtin_trim(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::string(args[0].as_str().trim()))
}

fn builtin_trim_start(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::string(args[0].as_str().trim_start()))
}

fn builtin_trim_end(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::string(args[0].as_str().trim_end()))
}

fn builtin_upper(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::string(args[0].as_str().to_uppercase()))
}

fn builtin_lower(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::string(args[0].as_str().to_lowercase()))
}

fn builtin_contains_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::Bool(args[0].as_str().contains(args[1].as_str())))
}

fn builtin_starts_with_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::Bool(args[0].as_str().starts_with(args[1].as_str())))
}

fn builtin_ends_with_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::Bool(args[0].as_str().ends_with(args[1].as_str())))
}

fn builtin_replace_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let from = args[1].as_str();
    let to = args[2].as_str();
    Ok(Value::string(s.replace(from, to)))
}

fn builtin_split_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let sep = args[1].as_str();
    let items: Vec<Value> = s.split(sep).map(Value::string).collect();
    Ok(Value::list(items))
}

fn builtin_repeat_str(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let n = args[1].as_int();
    Ok(Value::string(s.repeat(n.max(0) as usize)))
}

fn builtin_substring(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let start = args[1].as_int().max(0) as usize;
    let end = args[2].as_int().max(0) as usize;
    let end = end.min(s.len());
    let start = start.min(end);
    Ok(Value::string(&s[start..end]))
}

// ── List builtins ────────────────────────────────────────────────────

fn builtin_len(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    Ok(Value::Int(args[0].as_list().len() as i64))
}

fn builtin_reverse(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let list = args[0].take().into_list();
    let mut items: Vec<Value> = Arc::try_unwrap(list)
        .unwrap_or_else(|arc| arc.as_ref().iter().map(|v| v.share()).collect());
    items.reverse();
    Ok(Value::list(items))
}

// ── Conversion builtins ──────────────────────────────────────────────

fn builtin_to_string(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = match &args[0] {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => return Ok(Value::String(Arc::clone(s))),
        Value::Byte(b) => format!("0x{b:02x}"),
        Value::Unit => "()".to_string(),
        other => format!("{other:?}"),
    };
    Ok(Value::string(s))
}

fn builtin_to_int(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    use crate::error::ValueKind;
    let n = match &args[0] {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::String(s) => s.parse::<i64>().map_err(|_| {
            RuntimeError::unexpected_type("to_int", &[ValueKind::String], ValueKind::String)
        })?,
        Value::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        other => {
            return Err(RuntimeError::unexpected_type(
                "to_int",
                &[
                    ValueKind::Int,
                    ValueKind::Float,
                    ValueKind::String,
                    ValueKind::Bool,
                ],
                other.kind(),
            ));
        }
    };
    Ok(Value::Int(n))
}

fn builtin_to_float(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    use crate::error::ValueKind;
    let f = match &args[0] {
        Value::Int(n) => *n as f64,
        Value::Float(f) => *f,
        Value::String(s) => s.parse::<f64>().map_err(|_| {
            RuntimeError::unexpected_type("to_float", &[ValueKind::String], ValueKind::String)
        })?,
        other => {
            return Err(RuntimeError::unexpected_type(
                "to_float",
                &[ValueKind::Int, ValueKind::Float, ValueKind::String],
                other.kind(),
            ));
        }
    };
    Ok(Value::Float(f))
}

// ── Option builtins ──────────────────────────────────────────────────

fn builtin_unwrap(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    match args[0].take() {
        Value::Variant {
            payload: Some(p), ..
        } => Ok(Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share())),
        Value::Variant { payload: None, .. } => {
            panic!("unwrap: called on None")
        }
        other => panic!("unwrap: expected Variant, got {other:?}"),
    }
}

fn builtin_unwrap_or(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    match args[0].take() {
        Value::Variant {
            payload: Some(p), ..
        } => Ok(Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share())),
        Value::Variant { payload: None, .. } => Ok(args[1].take()),
        other => panic!("unwrap_or: expected Variant, got {other:?}"),
    }
}

// ── Conversion (extra) ───────────────────────────────────────────────

fn builtin_char_to_int(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let ch = s.chars().next().unwrap_or('\0');
    Ok(Value::Int(ch as i64))
}

fn builtin_int_to_char(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let n = args[0].as_int();
    let ch = char::from_u32(n as u32).unwrap_or('\u{FFFD}');
    Ok(Value::string(ch.to_string()))
}

// ── Bytes builtins ───────────────────────────────────────────────────

fn builtin_to_bytes(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let s = args[0].as_str();
    let bytes: Vec<Value> = s.bytes().map(Value::byte).collect();
    Ok(Value::list(bytes))
}

fn builtin_to_utf8(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let bytes: Vec<u8> = args[0].as_list().iter().map(|v| v.as_byte()).collect();
    let s = String::from_utf8(bytes).map_err(|_| {
        RuntimeError::unexpected_type(
            "to_utf8",
            &[crate::error::ValueKind::List],
            crate::error::ValueKind::List,
        )
    })?;
    Ok(Value::string(s))
}

fn builtin_to_utf8_lossy(args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let bytes: Vec<u8> = args[0].as_list().iter().map(|v| v.as_byte()).collect();
    Ok(Value::string(String::from_utf8_lossy(&bytes).into_owned()))
}

// ── Deque builtins ───────────────────────────────────────────────────

fn builtin_append(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("append: expected Deque, got {other:?}"),
    };
    let item = args[1].take();
    deque.push(item);
    Ok(Value::Deque(Arc::new(deque)))
}

fn builtin_extend(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("extend: expected Deque, got {other:?}"),
    };
    let items = args[1].take().into_list();
    for item in items.iter() {
        deque.push(item.clone());
    }
    Ok(Value::Deque(Arc::new(deque)))
}

fn builtin_consume(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("consume: expected Deque, got {other:?}"),
    };
    let n = args[1].as_int().max(0) as usize;
    deque.consume(n.min(deque.len()));
    Ok(Value::Deque(Arc::new(deque)))
}

// ── Iterator constructors ────────────────────────────────────────────

fn builtin_iter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        Value::Deque(d) => {
            let d = Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone());
            d.into_vec()
        }
        other => panic!("iter: expected List or Deque, got {other:?}"),
    };
    Ok(Value::iterator(IterHandle::from_list(
        items,
        Effect::pure(),
    )))
}

fn builtin_rev_iter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        Value::Deque(d) => {
            let d = Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone());
            d.into_vec()
        }
        other => panic!("rev_iter: expected List or Deque, got {other:?}"),
    };
    items.reverse();
    Ok(Value::iterator(IterHandle::from_list(
        items,
        Effect::pure(),
    )))
}

// ── Iterator lazy combinators ────────────────────────────────────────

fn builtin_map(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.map(*f)))
}

fn builtin_pmap(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    // pmap = map at runtime; parallelism is a scheduling concern.
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.map(*f)))
}

fn builtin_filter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.filter(*f)))
}

fn builtin_take(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::iterator(iter.take(n)))
}

fn builtin_skip(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::iterator(iter.skip(n)))
}

fn builtin_chain(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let a = args[0].take().into_iterator();
    let b = args[1].take().into_iterator();
    Ok(Value::iterator(a.chain(*b)))
}

fn builtin_flatten(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    Ok(Value::iterator(iter.flatten()))
}

fn builtin_flat_map(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.flat_map(*f)))
}

// ── Sequence lazy ops ────────────────────────────────────────────────

fn builtin_take_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::sequence(seq.take(n)))
}

fn builtin_skip_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::sequence(seq.skip(n)))
}

fn builtin_chain_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let iter = args[1].take().into_iterator();
    Ok(Value::sequence(seq.chain(*iter)))
}

// ── Async builtins (iterator consumers) ──────────────────────────────

use crate::interpreter::{ASYNC_FUTURE_SIZE, Interpreter};
use stackfuture::StackFuture;

fn builtin_collect<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut items = Vec::new();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            items.push(val);
        }
        Ok(Value::list(items))
    })
}

fn builtin_join<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let sep = args[1].as_str();
        let mut parts = Vec::new();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            parts.push(val.as_str().to_owned());
        }
        Ok(Value::string(parts.join(sep)))
    })
}

fn builtin_first<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        match interp.exec_next(&mut iter).await? {
            Some(val) => Ok(Value::variant(interp.intern_name("Some"), Some(val))),
            None => Ok(Value::variant(interp.intern_name("None"), None)),
        }
    })
}

fn builtin_last<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut last = None;
        while let Some(val) = interp.exec_next(&mut iter).await? {
            last = Some(val);
        }
        match last {
            Some(val) => Ok(Value::variant(interp.intern_name("Some"), Some(val))),
            None => Ok(Value::variant(interp.intern_name("None"), None)),
        }
    })
}

fn builtin_contains<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let needle = &args[1];
        while let Some(val) = interp.exec_next(&mut iter).await? {
            if val.structural_eq(needle) {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    })
}

fn builtin_next<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        match interp.exec_next(&mut iter).await? {
            Some(val) => {
                let pair = Value::tuple(vec![val, Value::iterator(iter)]);
                Ok(Value::variant(interp.intern_name("Some"), Some(pair)))
            }
            None => Ok(Value::variant(interp.intern_name("None"), None)),
        }
    })
}

fn builtin_next_seq<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        // Sequence next = convert to iterator, pull one.
        let seq = *args[0].take().into_sequence();
        let mut iter = seq.into_iter_handle();
        match interp.exec_next(&mut iter).await? {
            Some(val) => {
                let pair = Value::tuple(vec![val, Value::iterator(iter)]);
                Ok(Value::variant(interp.intern_name("Some"), Some(pair)))
            }
            None => Ok(Value::variant(interp.intern_name("None"), None)),
        }
    })
}

fn builtin_find<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            let keep = interp.call_closure(&f, val.clone()).await?;
            if keep.as_bool() {
                return Ok(val);
            }
        }
        Err(RuntimeError::empty_collection(
            crate::error::CollectionOp::Find,
        ))
    })
}

fn builtin_reduce<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        let Some(mut acc) = interp.exec_next(&mut iter).await? else {
            return Err(RuntimeError::empty_collection(
                crate::error::CollectionOp::Reduce,
            ));
        };
        while let Some(val) = interp.exec_next(&mut iter).await? {
            acc = interp.call_closure_2(&f, acc, val).await?;
        }
        Ok(acc)
    })
}

fn builtin_fold<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut acc = args[1].take();
        let f = args[2].take().into_fn();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            acc = interp.call_closure_2(&f, acc, val).await?;
        }
        Ok(acc)
    })
}

fn builtin_any<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            let result = interp.call_closure(&f, val).await?;
            if result.as_bool() {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    })
}

fn builtin_all<'x>(
    mut args: Args,
    interp: &'x mut Interpreter,
) -> StackFuture<'x, Result<Value, RuntimeError>, ASYNC_FUTURE_SIZE> {
    StackFuture::from(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = interp.exec_next(&mut iter).await? {
            let result = interp.call_closure(&f, val).await?;
            if !result.as_bool() {
                return Ok(Value::Bool(false));
            }
        }
        Ok(Value::Bool(true))
    })
}
