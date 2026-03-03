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
    compile_nodes, ApiKind, Fetch, HashMapStorage, HttpRequest, ProviderConfig,
};
use node::NodeDef;
use project::{toml_to_ty, ProjectSpec};

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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let project_dir_arg = args.get(1);

    let project_dir = match project_dir_arg {
        Some(dir) => PathBuf::from(dir),
        None => {
            eprintln!("usage: acvus-chat-cli <project-dir> [key=value ...]");
            process::exit(1);
        }
    };

    // key=value args after project dir
    let context_args = parse_context_args(&args[2..]);

    let project_toml = project_dir.join("project.toml");
    let project_src = std::fs::read_to_string(&project_toml).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", project_toml.display());
        process::exit(1);
    });

    let spec: ProjectSpec = toml::from_str(&project_src).unwrap_or_else(|e| {
        eprintln!("failed to parse project.toml: {e}");
        process::exit(1);
    });

    // Context is type-only — no constant values
    let context_types: HashMap<String, Ty> = spec
        .context
        .iter()
        .map(|(k, v)| (k.clone(), toml_to_ty(v)))
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
        let node_spec = node::resolve_node(node_def, &project_dir).unwrap_or_else(|e| {
            eprintln!("failed to resolve {node_file}: {e}");
            process::exit(1);
        });
        node_specs.push(node_spec);
    }

    let registry = ExternRegistry::new();
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

    let mut providers = HashMap::new();
    for (name, config) in &spec.providers {
        let api_key = if let Some(key) = &config.api_key {
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
        let api = match config.api.as_str() {
            "openai" => ApiKind::OpenAI,
            "anthropic" => ApiKind::Anthropic,
            "google" => ApiKind::Google,
            other => {
                eprintln!("unknown api kind: {other}");
                process::exit(1);
            }
        };
        providers.insert(
            name.clone(),
            ProviderConfig { api, endpoint: config.endpoint.clone(), api_key },
        );
    }

    let fetch = HttpFetch { client: reqwest::Client::new() };

    let extern_fns = ExternFnRegistry::new();
    let mut engine = ChatEngine::new(
        compiled_nodes, providers, fetch, extern_fns, storage, &spec.entrypoint,
    )
    .await
    .unwrap_or_else(|e| {
        eprintln!("engine init error: {e}");
        process::exit(1);
    });

    if context_args.is_empty() {
        // Interactive mode: resolve context keys from stdin on demand
        let resolver = |name: String| async move {
            eprint!("{name}: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            Value::String(input.trim_end().to_string())
        };

        loop {
            let response = engine.turn(&resolver).await.unwrap_or_else(|e| {
                eprintln!("turn error: {e}");
                process::exit(1);
            });
            println!("{response}");
        }
    } else {
        // One-shot: resolve from CLI args
        let resolver = |name: String| {
            let value = context_args
                .get(&name)
                .unwrap_or_else(|| panic!("unresolved context: @{name}"))
                .clone();
            async move { Value::String(value) }
        };
        let response = engine.turn(&resolver).await.unwrap_or_else(|e| {
            eprintln!("turn error: {e}");
            process::exit(1);
        });
        println!("{response}");
    }
}
