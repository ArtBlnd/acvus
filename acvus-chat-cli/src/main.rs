mod node;
mod project;

use std::path::PathBuf;
use std::process;

use acvus_chat::ChatEngine;
use acvus_interpreter::Value;
use acvus_mir::ty::Ty;
use acvus_mir::context_registry::PartialContextTypeRegistry;
use acvus_orchestration::{
    ApiKind, EntryRef, Execution, ExprSpec, Fetch, HttpRequest, Journal, NodeKind, NodeSpec,
    Persistency, ProviderConfig, Resolved, Strategy, TreeJournal, compile_nodes, compile_script,
};
use acvus_utils::{Astr, Interner};
use node::NodeDef;
use project::{ProjectSpec, parse_context_entry};
use rustc_hash::FxHashMap;

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
    endpoints: FxHashMap<String, ApiKind>,
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
fn parse_context_args(args: &[String]) -> FxHashMap<String, String> {
    let mut map = FxHashMap::default();
    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
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

    let interner = Interner::new();

    // Context types + defaults
    let mut user_types: FxHashMap<Astr, Ty> = FxHashMap::default();
    let mut context_defaults: FxHashMap<Astr, Value> = FxHashMap::default();
    for (k, v) in &spec.context {
        let entry = parse_context_entry(&interner, v);
        user_types.insert(interner.intern(k), entry.ty);
        if let Some(default) = entry.default {
            context_defaults.insert(interner.intern(k), default);
        }
    }

    // Build registry: extern fns separate, user-declared in user tier
    let extern_fns = acvus_ext::regex_context_types(&interner);
    let registry = PartialContextTypeRegistry::new(extern_fns, FxHashMap::default(), user_types)
        .unwrap_or_else(|e| {
            eprintln!("context type conflict: {e}");
            process::exit(1);
        });

    // Compile expr definitions → NodeSpec with NodeKind::Expr
    // Use a local context map for progressive compilation (each expr can reference
    // previous ones). Don't insert into registry — compute_external_context_env handles that.
    let mut expr_node_specs: Vec<NodeSpec> = Vec::new();
    let mut expr_context = registry.merged().clone();
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
        let (_script, tail_ty) = compile_script(&interner, &source, &expr_context)
            .unwrap_or_else(|e| {
                eprintln!(
                    "expr '{}' compile error: {}",
                    expr_def.name,
                    e.display(&interner)
                );
                process::exit(1);
            });
        let expr_name = interner.intern(&expr_def.name);
        expr_context.insert(expr_name, tail_ty.clone());
        expr_node_specs.push(NodeSpec {
            name: expr_name,
            kind: NodeKind::Expr(ExprSpec {
                source,
                output_ty: tail_ty,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::default(),
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        });
    }

    // Build provider name → ApiKind map (needed for node resolution)
    let provider_apis: FxHashMap<String, ApiKind> = spec
        .providers
        .iter()
        .map(|(name, config)| (name.clone(), config.api.clone()))
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
        let node_spec = node::resolve_node(&interner, node_def, &project_dir, &provider_apis)
            .unwrap_or_else(|e| {
                eprintln!("failed to resolve {node_file}: {e}");
                process::exit(1);
            });
        node_specs.push(node_spec);
    }

    // Merge expr node specs into node specs
    node_specs.extend(expr_node_specs);

    let compiled_nodes = match compile_nodes(&interner, &node_specs, registry) {
        Ok(nodes) => nodes,
        Err(errors) => {
            for e in &errors {
                eprintln!("compile error: {}", e.display(&interner));
            }
            process::exit(1);
        }
    };

    // Storage starts empty — context is type-only
    let (journal, root) = TreeJournal::new();

    let mut providers: FxHashMap<String, ProviderConfig> = FxHashMap::default();
    let mut endpoint_apis: FxHashMap<String, ApiKind> = FxHashMap::default();
    for (name, config) in &spec.providers {
        let api = config.api.clone();
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

    let resolver = {
        let defaults = context_defaults.clone();
        let context_args_astr: FxHashMap<Astr, String> = context_args
            .iter()
            .map(|(k, v)| (interner.intern(k), v.clone()))
            .collect();
        let interner_clone = interner.clone();
        move |name: Astr| {
            let defaults = defaults.clone();
            let context_args_astr = context_args_astr.clone();
            let interner_clone = interner_clone.clone();
            async move {
                if let Some(v) = context_args_astr.get(&name) {
                    return Resolved::Turn(Value::String(v.clone()));
                }
                if let Some(val) = defaults.get(&name) {
                    return Resolved::Turn(val.clone());
                }
                let name_str = interner_clone.resolve(name);
                if render_only {
                    return Resolved::Turn(Value::String(format!("(@{name_str})")));
                }
                eprint!("{name_str}: ");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                Resolved::Turn(Value::String(input.trim_end().to_string()))
            }
        }
    };

    let extern_handler = {
        let interner = interner.clone();
        move |name: Astr, args: Vec<Value>| {
            let interner = interner.clone();
            async move {
                acvus_ext::regex_call(&interner, name, args).await
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
            journal,
            root,
            &spec.entrypoint,
            &[],
            &interner,
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("engine init error: {e}");
            process::exit(1);
        });
        let (response, _) = engine.turn(&resolver, &extern_handler).await.unwrap_or_else(|e| {
            eprintln!("turn error: {e}");
            process::exit(1);
        });
        println!("{}", format_output(&interner, &response));
    } else {
        let fetch = HttpFetch {
            client: reqwest::Client::new(),
        };
        let mut engine = ChatEngine::new(
            compiled_nodes,
            providers,
            fetch,
            journal,
            root,
            &spec.entrypoint,
            &[],
            &interner,
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("engine init error: {e}");
            process::exit(1);
        });

        if context_args.is_empty() {
            loop {
                let (response, _) = engine.turn(&resolver, &extern_handler).await.unwrap_or_else(|e| {
                    eprintln!("turn error: {e}");
                    process::exit(1);
                });
                println!("{}", format_output(&interner, &response));
                println!("cursor depth: {}", engine.journal.entry(engine.cursor).await.depth());
            }
        } else {
            let (response, _) = engine.turn(&resolver, &extern_handler).await.unwrap_or_else(|e| {
                eprintln!("turn error: {e}");
                process::exit(1);
            });
            println!("{}", format_output(&interner, &response));
        }
    }
}

fn format_output(interner: &Interner, value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(obj) => {
            let content_key = interner.intern("content");
            match obj.get(&content_key) {
                Some(Value::String(s)) => s.clone(),
                _ => format!("{value:?}"),
            }
        }
        _ => format!("{value:?}"),
    }
}
