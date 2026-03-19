use std::path::Path;

use acvus_interpreter::{Interpreter, PureValue, Stepped, TypedValue, Value};
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

// ── Core pipeline ───────────────────────────────────────────────

/// Parse + compile + execute, returning the output string.
pub async fn run(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, TypedValue>,
) -> String {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, tv)| (*k, tv.ty().clone()))
        .collect();
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(context_types);
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        &reg,
    )
    .expect("compile failed");

    let interp = Interpreter::new(interner, module);
    emits_to_string(interp.execute_with_context(context).await)
}

/// Concatenate emitted values into a string (for template execution).
fn emits_to_string(emits: Vec<TypedValue>) -> String {
    let mut output = String::new();
    for v in emits {
        match v.value() {
            Value::Pure(PureValue::String(s)) => output.push_str(s),
            other => panic!("template emit: expected String, got {other:?}"),
        }
    }
    output
}

/// Simple: no context, no extern fns.
pub async fn run_simple(source: &str) -> String {
    let interner = Interner::new();
    run(&interner, source, FxHashMap::default()).await
}

/// With context + caller-provided interner.
pub async fn run_ctx(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, TypedValue>,
) -> String {
    run(interner, source, context).await
}

/// Context call result: the yielded NeedContext info.
#[derive(Debug)]
pub struct ContextCallResult {
    pub output: String,
    pub calls: Vec<String>,
}

/// Run a template and capture context calls (names only).
pub async fn run_capturing_context_calls(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, TypedValue>,
) -> ContextCallResult {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, tv)| (*k, tv.ty().clone()))
        .collect();
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(context_types);
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        &reg,
    )
    .expect("compile failed");
    let interp = Interpreter::new(interner, module);
    let mut coroutine = interp.execute();
    let mut output = String::new();
    let mut calls = Vec::new();
    loop {
        match coroutine.resume().await {
            Stepped::Emit(value) => match value.value() {
                Value::Pure(PureValue::String(s)) => output.push_str(s),
                other => panic!("expected String, got {other:?}"),
            },
            Stepped::NeedContext(request) => {
                let name = request.name();
                calls.push(interner.resolve(name).to_string());
                let v = context
                    .get(&name)
                    .unwrap_or_else(|| panic!("undefined context @{}", interner.resolve(name)));
                request.resolve(v.clone());
            }
            Stepped::NeedExternCall(_) => panic!("unexpected extern call"),
            Stepped::Done => break,
            Stepped::Error(e) => panic!("runtime error: {e}"),
        }
    }
    ContextCallResult { output, calls }
}

/// Execute a template and return the RuntimeError if one occurs.
pub async fn run_expect_error(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, TypedValue>,
) -> acvus_interpreter::RuntimeError {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, tv)| (*k, tv.ty().clone()))
        .collect();
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(context_types);
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        &reg,
    )
    .expect("compile failed");

    let interp = Interpreter::new(interner, module);
    let mut coroutine = interp.execute();
    loop {
        match coroutine.resume().await {
            Stepped::Emit(_) => {}
            Stepped::NeedContext(request) => {
                let name = request.name();
                let v = context
                    .get(&name)
                    .unwrap_or_else(|| panic!("undefined context @{}", interner.resolve(name)));
                request.resolve(v.clone());
            }
            Stepped::NeedExternCall(_) => panic!("unexpected extern call"),
            Stepped::Done => panic!("expected error, got Done"),
            Stepped::Error(e) => return e,
        }
    }
}

// ── Fixture runner ──────────────────────────────────────────────

/// Run a single `.json` fixture file.
pub async fn run_fixture(path: &Path) -> Result<(), String> {
    let interner = Interner::new();
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let fixture: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    let template = fixture["template"]
        .as_str()
        .ok_or_else(|| format!("{}: missing 'template'", path.display()))?;
    let expected = fixture["expected"]
        .as_str()
        .ok_or_else(|| format!("{}: missing 'expected'", path.display()))?;

    let context = match fixture.get("context") {
        Some(serde_json::Value::Object(fields)) => {
            fields
                .iter()
                .map(|(k, v)| {
                    let ty = ty_from_json(&interner, v);
                    let val = value_from_json(&interner, v);
                    (interner.intern(k), TypedValue::new(val, ty))
                })
                .collect()
        }
        Some(_) => return Err(format!("{}: 'context' must be an object", path.display())),
        None => FxHashMap::default(),
    };

    let actual = run(&interner, template, context).await;

    if actual != expected {
        Err(format!(
            "output mismatch\n  expected: {expected:?}\n  actual:   {actual:?}"
        ))
    } else {
        Ok(())
    }
}

// ── JSON → Ty / Value conversion ────────────────────────────────

/// Infer `Ty` from a JSON value.
pub fn ty_from_json(interner: &Interner, v: &serde_json::Value) -> Ty {
    match v {
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                Ty::Int
            } else {
                Ty::Float
            }
        }
        serde_json::Value::String(_) => Ty::String,
        serde_json::Value::Bool(_) => Ty::Bool,
        serde_json::Value::Null => panic!("null is not a supported type"),
        serde_json::Value::Array(items) => {
            let elem_ty = items
                .first()
                .map(|v| ty_from_json(interner, v))
                .expect("empty array: cannot infer element type");
            Ty::List(Box::new(elem_ty))
        }
        serde_json::Value::Object(fields) => {
            let field_types = fields
                .iter()
                .map(|(k, v)| (interner.intern(k), ty_from_json(interner, v)))
                .collect();
            Ty::Object(field_types)
        }
    }
}

/// Convert a JSON value to `Value`.
pub fn value_from_json(interner: &Interner, v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::int(i)
            } else {
                Value::float(n.as_f64().unwrap())
            }
        }
        serde_json::Value::String(s) => Value::string(s.clone()),
        serde_json::Value::Bool(b) => Value::bool_(*b),
        serde_json::Value::Null => panic!("null is not a supported value"),
        serde_json::Value::Array(items) => {
            Value::list(items.iter().map(|v| value_from_json(interner, v)).collect())
        }
        serde_json::Value::Object(fields) => {
            let obj = fields
                .iter()
                .map(|(k, v)| (interner.intern(k), value_from_json(interner, v)))
                .collect();
            Value::object(obj)
        }
    }
}

/// Convert a JSON value to `TypedValue`.
pub fn typed_from_json(interner: &Interner, v: &serde_json::Value) -> TypedValue {
    TypedValue::new(value_from_json(interner, v), ty_from_json(interner, v))
}

// ── Helpers (used by e2e.rs) ─────────────────────────────────

pub fn int_context(
    interner: &Interner,
    name: &str,
    value: i64,
) -> FxHashMap<Astr, TypedValue> {
    FxHashMap::from_iter([(interner.intern(name), TypedValue::int(value))])
}

pub fn string_context(
    interner: &Interner,
    name: &str,
    value: &str,
) -> FxHashMap<Astr, TypedValue> {
    FxHashMap::from_iter([(interner.intern(name), TypedValue::string(value))])
}

pub fn user_context(interner: &Interner) -> FxHashMap<Astr, TypedValue> {
    let ty = Ty::Object(FxHashMap::from_iter([
        (interner.intern("name"), Ty::String),
        (interner.intern("age"), Ty::Int),
        (interner.intern("email"), Ty::String),
    ]));
    let val = Value::object(FxHashMap::from_iter([
        (interner.intern("name"), Value::string("alice")),
        (interner.intern("age"), Value::int(30)),
        (
            interner.intern("email"),
            Value::string("alice@example.com"),
        ),
    ]));
    FxHashMap::from_iter([(interner.intern("user"), TypedValue::new(val, ty))])
}

pub fn users_list_context(interner: &Interner) -> FxHashMap<Astr, TypedValue> {
    let ty = Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
        (interner.intern("name"), Ty::String),
        (interner.intern("age"), Ty::Int),
    ]))));
    let val = Value::list(vec![
        Value::object(FxHashMap::from_iter([
            (interner.intern("name"), Value::string("alice")),
            (interner.intern("age"), Value::int(30)),
        ])),
        Value::object(FxHashMap::from_iter([
            (interner.intern("name"), Value::string("bob")),
            (interner.intern("age"), Value::int(25)),
        ])),
    ]);
    FxHashMap::from_iter([(interner.intern("users"), TypedValue::new(val, ty))])
}

pub fn items_context(
    interner: &Interner,
    items: Vec<i64>,
) -> FxHashMap<Astr, TypedValue> {
    let ty = Ty::List(Box::new(Ty::Int));
    let val = Value::list(items.into_iter().map(Value::int).collect());
    FxHashMap::from_iter([(interner.intern("items"), TypedValue::new(val, ty))])
}
