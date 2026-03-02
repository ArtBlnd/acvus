use std::collections::{BTreeMap, HashMap};

use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use acvus_interpreter::{ExternFnRegistry, Interpreter, PureValue, Value};
use acvus_mir::ty::Ty;

// ── Request / Response ────────────────────────────────────────────

#[derive(Deserialize)]
struct CompileRequest {
    source: String,
    #[serde(default)]
    context: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct CompileResponse {
    output: Option<String>,
    ir: Option<String>,
    error: Option<String>,
}

// ── JSON → Ty / PureValue ────────────────────────────────────────

fn ty_from_json(v: &serde_json::Value) -> Result<Ty, String> {
    match v {
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                Ok(Ty::Int)
            } else {
                Ok(Ty::Float)
            }
        }
        serde_json::Value::String(_) => Ok(Ty::String),
        serde_json::Value::Bool(_) => Ok(Ty::Bool),
        serde_json::Value::Null => Err("null is not a supported type".into()),
        serde_json::Value::Array(items) => {
            let elem_ty = items
                .first()
                .map(ty_from_json)
                .ok_or_else(|| "empty array: cannot infer element type".to_string())?;
            Ok(Ty::List(Box::new(elem_ty?)))
        }
        serde_json::Value::Object(fields) => {
            let field_types: Result<BTreeMap<String, Ty>, String> = fields
                .iter()
                .map(|(k, v)| ty_from_json(v).map(|t| (k.clone(), t)))
                .collect();
            Ok(Ty::Object(field_types?))
        }
    }
}

fn pv_from_json(v: &serde_json::Value) -> Result<PureValue, String> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(PureValue::Int(i))
            } else {
                Ok(PureValue::Float(n.as_f64().unwrap()))
            }
        }
        serde_json::Value::String(s) => Ok(PureValue::String(s.clone())),
        serde_json::Value::Bool(b) => Ok(PureValue::Bool(*b)),
        serde_json::Value::Null => Err("null is not a supported value".into()),
        serde_json::Value::Array(items) => {
            let vals: Result<Vec<PureValue>, String> = items.iter().map(pv_from_json).collect();
            Ok(PureValue::List(vals?))
        }
        serde_json::Value::Object(fields) => {
            let obj: Result<BTreeMap<String, PureValue>, String> = fields
                .iter()
                .map(|(k, v)| pv_from_json(v).map(|pv| (k.clone(), pv)))
                .collect();
            Ok(PureValue::Object(obj?))
        }
    }
}

// ── Compilation + Execution ──────────────────────────────────────

async fn compile_and_run(
    source: &str,
    context_json: HashMap<String, serde_json::Value>,
) -> Result<(String, String), String> {
    let template =
        acvus_ast::parse(source).map_err(|e| format!("parse error: {e}"))?;

    let mut context_types: HashMap<String, Ty> = HashMap::new();
    let mut context_values: HashMap<String, Value> = HashMap::new();

    for (k, v) in &context_json {
        let ty = ty_from_json(v).map_err(|e| format!("context `{k}`: {e}"))?;
        let pv = pv_from_json(v).map_err(|e| format!("context `{k}`: {e}"))?;
        context_types.insert(k.clone(), ty);
        context_values.insert(k.clone(), Value::from_pure(pv));
    }

    let extern_fns = ExternFnRegistry::new();
    let mir_registry = extern_fns.to_mir_registry();

    let (module, _hints) =
        acvus_mir::compile(&template, context_types, &mir_registry).map_err(|errors| {
            errors
                .iter()
                .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e))
                .collect::<Vec<_>>()
                .join("\n")
        })?;

    let ir = acvus_mir::printer::dump(&module);

    let interp = Interpreter::new(module, context_values, extern_fns);
    let output = interp.execute_to_string().await;

    Ok((output, ir))
}

// ── Handlers ─────────────────────────────────────────────────────

async fn handle_compile(Json(req): Json<CompileRequest>) -> Json<CompileResponse> {
    match compile_and_run(&req.source, req.context).await {
        Ok((output, ir)) => Json(CompileResponse {
            output: Some(output),
            ir: Some(ir),
            error: None,
        }),
        Err(e) => Json(CompileResponse {
            output: None,
            ir: None,
            error: Some(e),
        }),
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

// ── Entry point ──────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/compile", post(handle_compile));

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("Acvus Playground → http://{addr}");
    axum::serve(listener, app).await.unwrap();
}
