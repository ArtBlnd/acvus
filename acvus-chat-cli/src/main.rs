mod project;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use acvus_chat::ChatEngine;
use acvus_interpreter::ExternFnRegistry;
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    compile_nodes, resolve_template, ApiKind, Fetch, HashMapStorage, HttpRequest, NodeSpec,
    ProviderConfig,
};
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
        let node_spec: NodeSpec = toml::from_str(&node_src).unwrap_or_else(|e| {
            eprintln!("failed to parse {node_file}: {e}");
            process::exit(1);
        });
        node_specs.push(node_spec);
    }

    let registry = ExternRegistry::new();
    let compiled_nodes = match compile_nodes(&node_specs, &project_dir, &context_types, &registry) {
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

    // Compile output template if specified in project.toml
    let output_module = if spec.output.is_some() || spec.inline_output.is_some() {
        match resolve_template(
            &project_dir,
            spec.output.as_deref(),
            spec.inline_output.as_deref(),
            0,
            &context_types,
            &registry,
        ) {
            Ok(block) => Some(block),
            Err(e) => {
                eprintln!("output template error: {e}");
                process::exit(1);
            }
        }
    } else {
        None
    };

    let extern_fns = ExternFnRegistry::new();
    let mut engine = ChatEngine::new(compiled_nodes, providers, fetch, extern_fns, storage, output_module).await;

    if context_args.is_empty() {
        // Interactive mode: prompt per-turn keys from stdin
        loop {
            let mut per_turn = HashMap::new();
            for key in engine.per_turn_keys() {
                eprint!("{key}: ");
                std::io::stderr().flush().ok();
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).unwrap() == 0 {
                    return;
                }
                per_turn.insert(key.clone(), input.trim_end().to_string());
            }

            let response = engine.turn(per_turn).await;
            println!("{response}");
        }
    } else {
        // One-shot: use CLI args, run one turn
        let response = engine.turn(context_args).await;
        println!("{response}");
    }
}
