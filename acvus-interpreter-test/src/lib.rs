use std::path::Path;
use std::sync::Arc;

use acvus_interpreter::{Interpreter, Stepped, Value};
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

// ── Core pipeline ───────────────────────────────────────────────

/// Parse + compile + execute, returning the output string.
pub async fn run(
    interner: &Interner,
    source: &str,
    context_types: FxHashMap<Astr, Ty>,
    context_values: FxHashMap<Astr, Value>,
) -> String {
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(context_types);
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        &reg,
    )
    .expect("compile failed");

    let interp = Interpreter::new(interner, module);
    emits_to_string(interp.execute_with_context(context_values).await)
}

/// Concatenate emitted values into a string (for template execution).
fn emits_to_string(emits: Vec<Value>) -> String {
    let mut output = String::new();
    for v in emits {
        match v {
            Value::Pure(acvus_interpreter::PureValue::String(s)) => output.push_str(&s),
            other => panic!("template emit: expected String, got {other:?}"),
        }
    }
    output
}

/// Simple: no context, no extern fns.
pub async fn run_simple(source: &str) -> String {
    let interner = Interner::new();
    run(
        &interner,
        source,
        FxHashMap::default(),
        FxHashMap::default(),
    )
    .await
}

/// With context types + values + caller-provided interner.
pub async fn run_ctx(
    interner: &Interner,
    source: &str,
    types: FxHashMap<Astr, Ty>,
    values: FxHashMap<Astr, Value>,
) -> String {
    run(
        interner,
        source,
        types,
        values,
    )
    .await
}

/// Parse + compile + obfuscate + execute, returning the output string.
pub async fn run_obfuscated(
    interner: &Interner,
    source: &str,
    context_types: FxHashMap<Astr, Ty>,
    context_values: FxHashMap<Astr, Value>,
) -> String {
    use acvus_mir_pass::TransformPass;
    use acvus_mir_pass::obfuscate::{ObfConfig, ObfuscatePass};

    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(context_types);
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        &reg,
    )
    .expect("compile failed");

    let module = ObfuscatePass {
        config: ObfConfig {
            seed: 12345,
            ..ObfConfig::default()
        },
        interner: interner.clone(),
    }
    .transform(module, ());

    let interp = Interpreter::new(interner, module);
    emits_to_string(interp.execute_with_context(context_values).await)
}

/// Simple obfuscated: no context, no extern fns.
pub async fn run_simple_obfuscated(source: &str) -> String {
    let interner = Interner::new();
    run_obfuscated(
        &interner,
        source,
        FxHashMap::default(),
        FxHashMap::default(),
    )
    .await
}

/// Obfuscated run with caller-provided interner + default extern fns.
pub async fn run_obf_ctx(
    interner: &Interner,
    source: &str,
    types: FxHashMap<Astr, Ty>,
    values: FxHashMap<Astr, Value>,
) -> String {
    run_obfuscated(
        interner,
        source,
        types,
        values,
    )
    .await
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
    types: FxHashMap<Astr, Ty>,
    values: FxHashMap<Astr, Value>,
) -> ContextCallResult {
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let reg = ContextTypeRegistry::all_system(types);
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
            Stepped::Emit(value) => match value {
                Value::Pure(acvus_interpreter::PureValue::String(s)) => output.push_str(&s),
                other => panic!("expected String, got {other:?}"),
            },
            Stepped::NeedContext(request) => {
                let name = request.name();
                calls.push(interner.resolve(name).to_string());
                let v = values
                    .get(&name)
                    .unwrap_or_else(|| panic!("undefined context @{}", interner.resolve(name)));
                request.resolve(Arc::new(v.clone()));
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
    context_types: FxHashMap<Astr, Ty>,
    context_values: FxHashMap<Astr, Value>,
) -> acvus_interpreter::RuntimeError {
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
                let v = context_values
                    .get(&name)
                    .unwrap_or_else(|| panic!("undefined context @{}", interner.resolve(name)));
                request.resolve(Arc::new(v.clone()));
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

    let (types, values) = match fixture.get("context") {
        Some(serde_json::Value::Object(fields)) => {
            let types = fields
                .iter()
                .map(|(k, v)| (interner.intern(k), ty_from_json(&interner, v)))
                .collect();
            let values = fields
                .iter()
                .map(|(k, v)| {
                    (
                        interner.intern(k),
                        value_from_json(&interner, v),
                    )
                })
                .collect();
            (types, values)
        }
        Some(_) => return Err(format!("{}: 'context' must be an object", path.display())),
        None => (FxHashMap::default(), FxHashMap::default()),
    };

    let actual = run(
        &interner,
        template,
        types,
        values,
    )
    .await;

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

// ── Helpers (used by e2e.rs) ─────────────────────────────────

pub fn int_context(
    interner: &Interner,
    name: &str,
    value: i64,
) -> (FxHashMap<Astr, Ty>, FxHashMap<Astr, Value>) {
    (
        FxHashMap::from_iter([(interner.intern(name), Ty::Int)]),
        FxHashMap::from_iter([(interner.intern(name), Value::int(value))]),
    )
}

pub fn string_context(
    interner: &Interner,
    name: &str,
    value: &str,
) -> (FxHashMap<Astr, Ty>, FxHashMap<Astr, Value>) {
    (
        FxHashMap::from_iter([(interner.intern(name), Ty::String)]),
        FxHashMap::from_iter([(interner.intern(name), Value::string(value.into()))]),
    )
}

pub fn user_context(interner: &Interner) -> (FxHashMap<Astr, Ty>, FxHashMap<Astr, Value>) {
    let ty = Ty::Object(FxHashMap::from_iter([
        (interner.intern("name"), Ty::String),
        (interner.intern("age"), Ty::Int),
        (interner.intern("email"), Ty::String),
    ]));
    let val = Value::object(FxHashMap::from_iter([
        (interner.intern("name"), Value::string("alice".into())),
        (interner.intern("age"), Value::int(30)),
        (
            interner.intern("email"),
            Value::string("alice@example.com".into()),
        ),
    ]));
    (
        FxHashMap::from_iter([(interner.intern("user"), ty)]),
        FxHashMap::from_iter([(interner.intern("user"), val)]),
    )
}

pub fn users_list_context(interner: &Interner) -> (FxHashMap<Astr, Ty>, FxHashMap<Astr, Value>) {
    let ty = Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
        (interner.intern("name"), Ty::String),
        (interner.intern("age"), Ty::Int),
    ]))));
    let val = Value::list(vec![
        Value::object(FxHashMap::from_iter([
            (interner.intern("name"), Value::string("alice".into())),
            (interner.intern("age"), Value::int(30)),
        ])),
        Value::object(FxHashMap::from_iter([
            (interner.intern("name"), Value::string("bob".into())),
            (interner.intern("age"), Value::int(25)),
        ])),
    ]);
    (
        FxHashMap::from_iter([(interner.intern("users"), ty)]),
        FxHashMap::from_iter([(interner.intern("users"), val)]),
    )
}

pub fn items_context(
    interner: &Interner,
    items: Vec<i64>,
) -> (FxHashMap<Astr, Ty>, FxHashMap<Astr, Value>) {
    let ty = Ty::List(Box::new(Ty::Int));
    let val = Value::list(items.into_iter().map(Value::int).collect());
    (
        FxHashMap::from_iter([(interner.intern("items"), ty)]),
        FxHashMap::from_iter([(interner.intern("items"), val)]),
    )
}
