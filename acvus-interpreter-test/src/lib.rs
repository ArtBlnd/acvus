use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use acvus_interpreter::{
    ExternFnRegistry, InMemoryStorage, Interpreter, PureValue, Storage, StorageKey,
};
use acvus_mir::ty::Ty;

// ── Core pipeline ───────────────────────────────────────────────

/// Parse + compile + execute, returning the output string.
pub async fn run(
    source: &str,
    storage_types: HashMap<String, Ty>,
    storage_values: HashMap<String, PureValue>,
    extern_fns: ExternFnRegistry,
) -> String {
    let template = acvus_ast::parse(source).expect("parse failed");
    let mir_registry = extern_fns.to_mir_registry();
    let (module, _hints) =
        acvus_mir::compile(&template, storage_types, &mir_registry).expect("compile failed");

    let mut storage = InMemoryStorage::new();
    for (name, value) in storage_values {
        storage
            .set(&StorageKey::root(name), value)
            .await
            .unwrap();
    }

    let mut interp = Interpreter::new(module, storage, extern_fns);
    interp.execute().await
}

/// Simple: no storage, no extern fns.
pub async fn run_simple(source: &str) -> String {
    run(source, HashMap::new(), HashMap::new(), ExternFnRegistry::new()).await
}

/// With storage types + values.
pub async fn run_with_storage(
    source: &str,
    types: HashMap<String, Ty>,
    values: HashMap<String, PureValue>,
) -> String {
    run(source, types, values, ExternFnRegistry::new()).await
}

// ── Fixture runner ──────────────────────────────────────────────

/// Run a single `.json` fixture file.
///
/// Expected format:
/// ```json
/// {
///   "template": "Hello, {{ $name }}!",
///   "storage": { "name": "alice" },
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

    let (types, values) = match fixture.get("storage") {
        Some(serde_json::Value::Object(fields)) => {
            let types: HashMap<String, Ty> = fields
                .iter()
                .map(|(k, v)| (k.clone(), ty_from_json(v)))
                .collect();
            let values: HashMap<String, PureValue> = fields
                .iter()
                .map(|(k, v)| (k.clone(), pv_from_json(v)))
                .collect();
            (types, values)
        }
        Some(_) => return Err(format!("{}: 'storage' must be an object", path.display())),
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
            let field_types: BTreeMap<String, Ty> =
                fields.iter().map(|(k, v)| (k.clone(), ty_from_json(v))).collect();
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
            let obj: BTreeMap<String, PureValue> =
                fields.iter().map(|(k, v)| (k.clone(), pv_from_json(v))).collect();
            PureValue::Object(obj)
        }
    }
}

// ── Legacy helpers (used by e2e.rs) ─────────────────────────────

pub fn int_storage(name: &str, value: i64) -> (HashMap<String, Ty>, HashMap<String, PureValue>) {
    (
        HashMap::from([(name.into(), Ty::Int)]),
        HashMap::from([(name.into(), PureValue::Int(value))]),
    )
}

pub fn string_storage(
    name: &str,
    value: &str,
) -> (HashMap<String, Ty>, HashMap<String, PureValue>) {
    (
        HashMap::from([(name.into(), Ty::String)]),
        HashMap::from([(name.into(), PureValue::String(value.into()))]),
    )
}

pub fn user_storage() -> (HashMap<String, Ty>, HashMap<String, PureValue>) {
    let ty = Ty::Object(BTreeMap::from([
        ("name".into(), Ty::String),
        ("age".into(), Ty::Int),
        ("email".into(), Ty::String),
    ]));
    let val = PureValue::Object(BTreeMap::from([
        ("name".into(), PureValue::String("alice".into())),
        ("age".into(), PureValue::Int(30)),
        ("email".into(), PureValue::String("alice@example.com".into())),
    ]));
    (
        HashMap::from([("user".into(), ty)]),
        HashMap::from([("user".into(), val)]),
    )
}

pub fn users_list_storage() -> (HashMap<String, Ty>, HashMap<String, PureValue>) {
    let ty = Ty::List(Box::new(Ty::Object(BTreeMap::from([
        ("name".into(), Ty::String),
        ("age".into(), Ty::Int),
    ]))));
    let val = PureValue::List(vec![
        PureValue::Object(BTreeMap::from([
            ("name".into(), PureValue::String("alice".into())),
            ("age".into(), PureValue::Int(30)),
        ])),
        PureValue::Object(BTreeMap::from([
            ("name".into(), PureValue::String("bob".into())),
            ("age".into(), PureValue::Int(25)),
        ])),
    ]);
    (
        HashMap::from([("users".into(), ty)]),
        HashMap::from([("users".into(), val)]),
    )
}

pub fn items_storage(items: Vec<i64>) -> (HashMap<String, Ty>, HashMap<String, PureValue>) {
    let ty = Ty::List(Box::new(Ty::Int));
    let val = PureValue::List(items.into_iter().map(PureValue::Int).collect());
    (
        HashMap::from([("items".into(), ty)]),
        HashMap::from([("items".into(), val)]),
    )
}
