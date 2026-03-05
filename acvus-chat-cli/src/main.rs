mod node;
mod project;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use acvus_chat::ChatEngine;
use acvus_interpreter::{ExternFnRegistry, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    ApiKind, ExprSpec, Fetch, HashMapStorage, HttpRequest, NodeKind, NodeSpec, ProviderConfig,
    Resolved, SelfSpec, Strategy, compile_nodes, compile_script,
};
use node::NodeDef;
use project::{ProjectSpec, parse_context_entry};

#[derive(Clone)]
struct HttpFetch {
    client: reqwest::Client,
}

impl Fetch for HttpFetch {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        let mut builder = self.client.post(&request.url);
        for (k, v) in &request.headers {
            builder = builder.header(k.as_str(), v.as_str());
        }
        let resp = builder
            .json(&request.body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }
}

/// Dummy fetch for `--render-only` mode. Returns provider-appropriate dummy responses.
#[derive(Clone)]
struct RenderOnlyFetch {
    /// endpoint prefix → API kind
    endpoints: HashMap<String, ApiKind>,
}

impl Fetch for RenderOnlyFetch {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        let api = self
            .endpoints
            .iter()
            .find(|(prefix, _)| request.url.starts_with(prefix.as_str()))
            .map(|(_, kind)| kind.clone())
            .unwrap_or(ApiKind::OpenAI);

        // countTokens → fixed token count
        if request.url.contains("countTokens") {
            return Ok(serde_json::json!({"totalTokens": 100}));
        }
        // cachedContents → dummy cache name
        if request.url.contains("cachedContents") {
            return Ok(serde_json::json!({"name": "cachedContents/render-only"}));
        }

        Ok(match api {
            ApiKind::OpenAI => serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "(render-only)"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }),
            ApiKind::Anthropic => serde_json::json!({
                "content": [{"type": "text", "text": "(render-only)"}],
                "role": "assistant",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }),
            ApiKind::Google => serde_json::json!({
                "candidates": [{"content": {"parts": [{"text": "(render-only)"}]}}],
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
            }),
        })
    }
}

/// Parse `key=value` pairs from CLI arguments.
fn parse_context_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
}

fn parse_api_kind(s: &str) -> ApiKind {
    ApiKind::parse(s).unwrap_or_else(|| {
        eprintln!("unknown api kind: {s}");
        process::exit(1);
    })
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let render_only = args.iter().any(|a| a == "--render-only");
    let non_flag_args: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();

    let Some(dir) = non_flag_args.first() else {
        eprintln!("usage: acvus-chat-cli <project-dir> [--render-only] [key=value ...]");
        process::exit(1);
    };
    let project_dir = PathBuf::from(dir);

    let context_args = parse_context_args(
        &non_flag_args[1..]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    );

    let project_toml = project_dir.join("project.toml");
    let project_src = std::fs::read_to_string(&project_toml).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", project_toml.display());
        process::exit(1);
    });

    let spec: ProjectSpec = toml::from_str(&project_src).unwrap_or_else(|e| {
        eprintln!("failed to parse project.toml: {e}");
        process::exit(1);
    });

    // Context types + defaults
    let mut context_types: HashMap<String, Ty> = HashMap::new();
    let mut context_defaults: HashMap<String, Value> = HashMap::new();
    for (k, v) in &spec.context {
        let entry = parse_context_entry(v);
        context_types.insert(k.clone(), entry.ty);
        if let Some(default) = entry.default {
            context_defaults.insert(k.clone(), default);
        }
    }

    // Compile expr definitions → NodeSpec with NodeKind::Expr
    let registry = ExternRegistry::new();
    let mut expr_node_specs: Vec<NodeSpec> = Vec::new();
    for expr_def in &spec.expr {
        let source = if let Some(path) = &expr_def.source {
            std::fs::read_to_string(project_dir.join(path)).unwrap_or_else(|e| {
                eprintln!("failed to read expr source {path}: {e}");
                process::exit(1);
            })
        } else if let Some(inline) = &expr_def.inline_source {
            inline.clone()
        } else {
            eprintln!(
                "expr '{}': must have source or inline_source",
                expr_def.name
            );
            process::exit(1);
        };
        let (_script, tail_ty) =
            compile_script(&source, &context_types, &registry).unwrap_or_else(|e| {
                eprintln!("expr '{}' compile error: {e}", expr_def.name);
                process::exit(1);
            });
        context_types.insert(expr_def.name.clone(), tail_ty.clone());
        expr_node_specs.push(NodeSpec {
            name: expr_def.name.clone(),
            kind: NodeKind::Expr(ExprSpec {
                source,
                output_ty: tail_ty,
            }),
            self_spec: SelfSpec {
                self_bind: "@raw".into(),
                initial_value: r#""""#.into(),
            },
            strategy: Strategy::default(),
        });
    }

    // Build provider name → ApiKind map (needed for node resolution)
    let provider_apis: HashMap<String, ApiKind> = spec
        .providers
        .iter()
        .map(|(name, config)| (name.clone(), parse_api_kind(&config.api)))
        .collect();

    let mut node_specs = Vec::new();
    for node_file in &spec.nodes {
        let node_src = std::fs::read_to_string(project_dir.join(node_file)).unwrap_or_else(|e| {
            eprintln!("failed to read {node_file}: {e}");
            process::exit(1);
        });
        let node_def: NodeDef = toml::from_str(&node_src).unwrap_or_else(|e| {
            eprintln!("failed to parse {node_file}: {e}");
            process::exit(1);
        });
        let node_spec =
            node::resolve_node(node_def, &project_dir, &provider_apis).unwrap_or_else(|e| {
                eprintln!("failed to resolve {node_file}: {e}");
                process::exit(1);
            });
        node_specs.push(node_spec);
    }

    // Merge expr node specs into node specs
    node_specs.extend(expr_node_specs);

    let compiled_nodes = match compile_nodes(&node_specs, &context_types, &registry) {
        Ok(nodes) => nodes,
        Err(errors) => {
            for e in &errors {
                eprintln!("compile error: {e}");
            }
            process::exit(1);
        }
    };

    // Storage starts empty — context is type-only
    let storage = HashMapStorage::new();

    let mut providers: HashMap<String, ProviderConfig> = HashMap::new();
    let mut endpoint_apis: HashMap<String, ApiKind> = HashMap::new();
    for (name, config) in &spec.providers {
        let api = parse_api_kind(&config.api);
        let api_key = if render_only {
            String::new()
        } else if let Some(key) = &config.api_key {
            key.clone()
        } else if let Some(env_name) = &config.api_key_env {
            std::env::var(env_name).unwrap_or_else(|_| {
                eprintln!("environment variable {env_name} not set (provider: {name})");
                process::exit(1);
            })
        } else {
            eprintln!("no api_key or api_key_env set (provider: {name})");
            process::exit(1);
        };
        endpoint_apis.insert(config.endpoint.clone(), api.clone());
        providers.insert(
            name.clone(),
            ProviderConfig {
                api,
                endpoint: config.endpoint.clone(),
                api_key,
            },
        );
    }

    let extern_fns = ExternFnRegistry::new();

    let resolver = {
        let defaults = context_defaults.clone();
        let context_args = context_args.clone();
        move |name: String| {
            let defaults = defaults.clone();
            let context_args = context_args.clone();
            async move {
                if let Some(v) = context_args.get(&name) {
                    return Resolved::Turn(Value::String(v.clone()));
                }
                if let Some(val) = defaults.get(&name) {
                    return Resolved::Turn(val.clone());
                }
                if render_only {
                    return Resolved::Turn(Value::String(format!("(@{name})")));
                }
                eprint!("{name}: ");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                Resolved::Turn(Value::String(input.trim_end().to_string()))
            }
        }
    };

    if render_only {
        let fetch = RenderOnlyFetch {
            endpoints: endpoint_apis,
        };
        let mut engine = ChatEngine::new(
            compiled_nodes,
            providers,
            fetch,
            extern_fns,
            storage,
            &spec.entrypoint,
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("engine init error: {e}");
            process::exit(1);
        });
        let response = engine.turn(&resolver).await.unwrap_or_else(|e| {
            eprintln!("turn error: {e}");
            process::exit(1);
        });
        println!("{}", format_output(&response));
    } else {
        let fetch = HttpFetch {
            client: reqwest::Client::new(),
        };
        let mut engine = ChatEngine::new(
            compiled_nodes,
            providers,
            fetch,
            extern_fns,
            storage,
            &spec.entrypoint,
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("engine init error: {e}");
            process::exit(1);
        });

        if context_args.is_empty() {
            loop {
                let response = engine.turn(&resolver).await.unwrap_or_else(|e| {
                    eprintln!("turn error: {e}");
                    process::exit(1);
                });
                println!("{}", format_output(&response));
            }
        } else {
            let response = engine.turn(&resolver).await.unwrap_or_else(|e| {
                eprintln!("turn error: {e}");
                process::exit(1);
            });
            println!("{}", format_output(&response));
        }
    }
}

fn format_output(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(obj) => match obj.get("content") {
            Some(Value::String(s)) => s.clone(),
            _ => format!("{value:?}"),
        },
        _ => format!("{value:?}"),
    }
}
