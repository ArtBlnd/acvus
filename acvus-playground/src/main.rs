use std::collections::{BTreeMap, HashMap};

use axum::{routing::{get, post}, Json, Router};
use axum::response::Html;
use serde::{Deserialize, Serialize};

use acvus_mir::ty::Ty;

// ── Request / Response ────────────────────────────────────────────

#[derive(Deserialize)]
struct CompileRequest {
    source: String,
    #[serde(default)]
    storage: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct CompileResponse {
    ir: Option<String>,
    error: Option<String>,
}

// ── Type parser ───────────────────────────────────────────────────
//
// JSON type spec:
//   "String" | "Int" | "Float" | "Bool"
//   {"List": <type>}
//   {"Tuple": [<type>, ...]}
//   {"Object": {<key>: <type>, ...}}

fn parse_ty(v: &serde_json::Value) -> Result<Ty, String> {
    match v {
        serde_json::Value::String(s) => match s.as_str() {
            "string" => Ok(Ty::String),
            "int" => Ok(Ty::Int),
            "float" => Ok(Ty::Float),
            "bool" => Ok(Ty::Bool),
            other => Err(format!("unknown primitive type `{other}`")),
        },
        serde_json::Value::Object(map) => {
            if let Some(elem) = map.get("List") {
                Ok(Ty::List(Box::new(parse_ty(elem)?)))
            } else if let Some(elems) = map.get("Tuple") {
                let arr = elems.as_array().ok_or("Tuple value must be an array")?;
                let types: Result<Vec<Ty>, _> = arr.iter().map(parse_ty).collect();
                Ok(Ty::Tuple(types?))
            } else if let Some(fields) = map.get("Object") {
                let obj = fields.as_object()
                    .ok_or("Object value must be a JSON object")?;
                let field_types: Result<BTreeMap<String, Ty>, _> = obj.iter()
                    .map(|(k, v)| parse_ty(v).map(|t| (k.clone(), t)))
                    .collect();
                Ok(Ty::Object(field_types?))
            } else {
                Err("unknown type spec — use \"List\", \"Tuple\", or \"Object\" as key".into())
            }
        }
        _ => Err("type must be a string or object".into()),
    }
}

// ── Compilation ───────────────────────────────────────────────────

fn compile_inner(
    source: &str,
    storage_spec: HashMap<String, serde_json::Value>,
) -> Result<String, String> {
    let template = acvus_ast::parse(source)
        .map_err(|e| format!("parse error: {e}"))?;

    let storage_types: Result<HashMap<String, Ty>, _> = storage_spec
        .iter()
        .map(|(k, v)| {
            parse_ty(v)
                .map(|t| (k.clone(), t))
                .map_err(|e| format!("storage `{k}`: {e}"))
        })
        .collect();

    let (module, _hints) =
        acvus_mir::compile(&template, storage_types?, HashMap::new()).map_err(|errors| {
            errors
                .iter()
                .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e))
                .collect::<Vec<_>>()
                .join("\n")
        })?;

    Ok(acvus_mir::printer::dump(&module))
}

// ── Handlers ──────────────────────────────────────────────────────

async fn handle_compile(Json(req): Json<CompileRequest>) -> Json<CompileResponse> {
    match compile_inner(&req.source, req.storage) {
        Ok(ir) => Json(CompileResponse { ir: Some(ir), error: None }),
        Err(e) => Json(CompileResponse { ir: None, error: Some(e) }),
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

// ── Entry point ───────────────────────────────────────────────────

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
