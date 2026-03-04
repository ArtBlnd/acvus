use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, PureValue, Stepped, Value};
use acvus_mir::ty::Ty;

// ── Core pipeline ───────────────────────────────────────────────

/// Parse + compile + execute, returning the output string.
pub async fn run(
    source: &str,
    context_types: HashMap<String, Ty>,
    context_values: HashMap<String, Value>,
    extern_fns: ExternFnRegistry,
) -> String {
    let template = acvus_ast::parse(source).expect("parse failed");
    let mir_registry = extern_fns.to_mir_registry();
    let (module, _hints) = acvus_mir::compile(
        &template,
        context_types,
        &mir_registry,
        &acvus_mir::user_type::UserTypeRegistry::new(),
    )
    .expect("compile failed");

    let interp = Interpreter::new(module, &extern_fns);
    interp.execute_to_string(context_values).await
}

/// Simple: no context, no extern fns.
pub async fn run_simple(source: &str) -> String {
    run(
        source,
        HashMap::new(),
        HashMap::new(),
        ExternFnRegistry::new(),
    )
    .await
}

/// Parse + compile + obfuscate + execute, returning the output string.
pub async fn run_obfuscated(
    source: &str,
    context_types: HashMap<String, Ty>,
    context_values: HashMap<String, Value>,
    extern_fns: ExternFnRegistry,
) -> String {
    use acvus_mir_pass::TransformPass;
    use acvus_mir_pass::obfuscate::{ObfConfig, ObfuscatePass};

    let template = acvus_ast::parse(source).expect("parse failed");
    let mir_registry = extern_fns.to_mir_registry();
    let (module, _hints) = acvus_mir::compile(
        &template,
        context_types,
        &mir_registry,
        &acvus_mir::user_type::UserTypeRegistry::new(),
    )
    .expect("compile failed");

    let module = ObfuscatePass {
        config: ObfConfig {
            seed: 12345,
            ..ObfConfig::default()
        },
    }
    .transform(module, ());

    let interp = Interpreter::new(module, &extern_fns);
    interp.execute_to_string(context_values).await
}

/// Simple obfuscated: no context, no extern fns.
pub async fn run_simple_obfuscated(source: &str) -> String {
    run_obfuscated(
        source,
        HashMap::new(),
        HashMap::new(),
        ExternFnRegistry::new(),
    )
    .await
}

/// With context types + values.
pub async fn run_with_context(
    source: &str,
    types: HashMap<String, Ty>,
    values: HashMap<String, Value>,
) -> String {
    run(source, types, values, ExternFnRegistry::new()).await
}

/// Context call result: the yielded NeedContext info.
#[derive(Debug)]
pub struct ContextCallResult {
    pub output: String,
    pub calls: Vec<(String, HashMap<String, Value>)>,
}

/// Run a template and capture context calls with their bindings.
///
/// When a NeedContext with bindings is encountered, the bindings are recorded
/// and the context is resolved from `values` by name. This allows testing
/// that context call bindings are properly carried through the coroutine.
pub async fn run_capturing_context_calls(
    source: &str,
    types: HashMap<String, Ty>,
    values: HashMap<String, Value>,
) -> ContextCallResult {
    let template = acvus_ast::parse(source).expect("parse failed");
    let (module, _hints) = acvus_mir::compile(
        &template,
        types,
        &ExternFnRegistry::new().to_mir_registry(),
        &acvus_mir::user_type::UserTypeRegistry::new(),
    )
    .expect("compile failed");
    let interp = Interpreter::new(module, &ExternFnRegistry::new());
    let (mut coroutine, mut key) = interp.execute();
    let mut output = String::new();
    let mut calls = Vec::new();
    loop {
        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => output.push_str(&s),
                    other => panic!("expected String, got {other:?}"),
                }
                key = next_key;
            }
            Stepped::NeedContext(need) => {
                let name = need.name().to_string();
                let bindings = need.bindings().clone();
                if !bindings.is_empty() {
                    calls.push((name.clone(), bindings));
                }
                let v = values
                    .get(&name)
                    .unwrap_or_else(|| panic!("undefined context @{name}"));
                key = need.into_key(Arc::new(v.clone()));
            }
            Stepped::Done => break,
        }
    }
    ContextCallResult { output, calls }
}

// ── Fixture runner ──────────────────────────────────────────────

/// Run a single `.json` fixture file.
///
/// Expected format:
/// ```json
/// {
///   "template": "Hello, {{ @name }}!",
///   "context": { "name": "alice" },
///   "expected": "Hello, alice!"
/// }
/// ```
pub async fn run_fixture(path: &Path) -> Result<(), String> {
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
            let types: HashMap<String, Ty> = fields
                .iter()
                .map(|(k, v)| (k.clone(), ty_from_json(v)))
                .collect();
            let values: HashMap<String, Value> = fields
                .iter()
                .map(|(k, v)| (k.clone(), Value::from_pure(pv_from_json(v))))
                .collect();
            (types, values)
        }
        Some(_) => return Err(format!("{}: 'context' must be an object", path.display())),
        None => (HashMap::new(), HashMap::new()),
    };

    let actual = run(template, types, values, ExternFnRegistry::new()).await;

    if actual != expected {
        Err(format!(
            "output mismatch\n  expected: {expected:?}\n  actual:   {actual:?}"
        ))
    } else {
        Ok(())
    }
}

// ── JSON → Ty / PureValue conversion ────────────────────────────

/// Infer `Ty` from a JSON value.
pub fn ty_from_json(v: &serde_json::Value) -> Ty {
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
                .map(ty_from_json)
                .expect("empty array: cannot infer element type");
            Ty::List(Box::new(elem_ty))
        }
        serde_json::Value::Object(fields) => {
            let field_types: BTreeMap<String, Ty> = fields
                .iter()
                .map(|(k, v)| (k.clone(), ty_from_json(v)))
                .collect();
            Ty::Object(field_types)
        }
    }
}

/// Convert a JSON value to `PureValue`.
pub fn pv_from_json(v: &serde_json::Value) -> PureValue {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PureValue::Int(i)
            } else {
                PureValue::Float(n.as_f64().unwrap())
            }
        }
        serde_json::Value::String(s) => PureValue::String(s.clone()),
        serde_json::Value::Bool(b) => PureValue::Bool(*b),
        serde_json::Value::Null => panic!("null is not a supported value"),
        serde_json::Value::Array(items) => {
            PureValue::List(items.iter().map(pv_from_json).collect())
        }
        serde_json::Value::Object(fields) => {
            let obj: BTreeMap<String, PureValue> = fields
                .iter()
                .map(|(k, v)| (k.clone(), pv_from_json(v)))
                .collect();
            PureValue::Object(obj)
        }
    }
}

// ── Helpers (used by e2e.rs) ─────────────────────────────────

pub fn int_context(name: &str, value: i64) -> (HashMap<String, Ty>, HashMap<String, Value>) {
    (
        HashMap::from([(name.into(), Ty::Int)]),
        HashMap::from([(name.into(), Value::Int(value))]),
    )
}

pub fn string_context(name: &str, value: &str) -> (HashMap<String, Ty>, HashMap<String, Value>) {
    (
        HashMap::from([(name.into(), Ty::String)]),
        HashMap::from([(name.into(), Value::String(value.into()))]),
    )
}

pub fn user_context() -> (HashMap<String, Ty>, HashMap<String, Value>) {
    let ty = Ty::Object(BTreeMap::from([
        ("name".into(), Ty::String),
        ("age".into(), Ty::Int),
        ("email".into(), Ty::String),
    ]));
    let val = Value::Object(BTreeMap::from([
        ("name".into(), Value::String("alice".into())),
        ("age".into(), Value::Int(30)),
        ("email".into(), Value::String("alice@example.com".into())),
    ]));
    (
        HashMap::from([("user".into(), ty)]),
        HashMap::from([("user".into(), val)]),
    )
}

pub fn users_list_context() -> (HashMap<String, Ty>, HashMap<String, Value>) {
    let ty = Ty::List(Box::new(Ty::Object(BTreeMap::from([
        ("name".into(), Ty::String),
        ("age".into(), Ty::Int),
    ]))));
    let val = Value::List(vec![
        Value::Object(BTreeMap::from([
            ("name".into(), Value::String("alice".into())),
            ("age".into(), Value::Int(30)),
        ])),
        Value::Object(BTreeMap::from([
            ("name".into(), Value::String("bob".into())),
            ("age".into(), Value::Int(25)),
        ])),
    ]);
    (
        HashMap::from([("users".into(), ty)]),
        HashMap::from([("users".into(), val)]),
    )
}

pub fn items_context(items: Vec<i64>) -> (HashMap<String, Ty>, HashMap<String, Value>) {
    let ty = Ty::List(Box::new(Ty::Int));
    let val = Value::List(items.into_iter().map(Value::Int).collect());
    (
        HashMap::from([("items".into(), ty)]),
        HashMap::from([("items".into(), val)]),
    )
}
