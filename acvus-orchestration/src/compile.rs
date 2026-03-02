use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};

use crate::dsl::{MessageSpec, NodeSpec, Strategy, StrategyMode, ToolDecl};
use crate::error::{OrchError, OrchErrorKind};

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub tools: Vec<ToolDecl>,
    pub messages: Vec<CompiledMessage>,
    pub all_context_keys: HashSet<String>,
    pub strategy: Strategy,
    pub key_module: Option<CompiledBlock>,
    pub output_module: Option<CompiledBlock>,
}

/// A compiled message entry.
#[derive(Debug, Clone)]
pub enum CompiledMessage {
    Block(CompiledBlock),
    Iterator {
        key: String,
        block: Option<CompiledBlock>,
        slice: Option<Vec<i64>>,
        bind: Option<String>,
        role: Option<String>,
    },
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub role: String,
    pub module: MirModule,
    pub context_keys: HashSet<String>,
    pub val_def: ValDefMap,
}

/// Compile a node spec into a `CompiledNode`.
///
/// Each message's template file is loaded from `base_dir`, parsed, and compiled.
/// Context keys are extracted from MIR `ContextLoad` instructions.
pub fn compile_node(
    spec: &NodeSpec,
    base_dir: &Path,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledNode, Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, msg) in spec.messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, template, inline_template } => {
                let result = resolve_template(
                    base_dir, template.as_deref(), inline_template.as_deref(),
                    i, context_types, registry,
                );
                match result {
                    Ok(block) => {
                        all_context_keys.extend(block.context_keys.iter().cloned());
                        compiled_messages.push(CompiledMessage::Block(CompiledBlock {
                            role: role.clone(),
                            ..block
                        }));
                    }
                    Err(e) => errors.push(e),
                }
            }
            MessageSpec::Iterator { iterator, template, inline_template, slice, bind, role } => {
                let key = iterator.trim_start_matches('@').to_string();

                let tmpl_source = template.as_deref().or(inline_template.as_deref());
                let block = if tmpl_source.is_some() {
                    let mut iter_types = context_types.clone();
                    if let Some(bind_name) = bind {
                        iter_types.insert(
                            bind_name.clone(),
                            Ty::Object(BTreeMap::from([
                                ("type".into(), Ty::String),
                                ("text".into(), Ty::String),
                            ])),
                        );
                    } else {
                        iter_types.insert("type".into(), Ty::String);
                        iter_types.insert("text".into(), Ty::String);
                    }

                    let result = resolve_template(
                        base_dir, template.as_deref(), inline_template.as_deref(),
                        i, &iter_types, registry,
                    );
                    match result {
                        Ok(block) => Some(block),
                        Err(e) => {
                            errors.push(e);
                            None
                        }
                    }
                } else {
                    None
                };

                compiled_messages.push(CompiledMessage::Iterator {
                    key,
                    block,
                    slice: slice.clone(),
                    bind: bind.clone(),
                    role: role.clone(),
                });
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Compile key template for if-modified strategy
    let key_module =
        if matches!(spec.strategy.mode, StrategyMode::IfModified) {
            if let Some(key_tmpl) = &spec.strategy.key {
                match compile_template(base_dir, key_tmpl, 0, context_types, registry) {
                    Ok(block) => Some(block),
                    Err(e) => {
                        return Err(vec![e]);
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

    let output_module = if spec.output.is_some() || spec.inline_output.is_some() {
        match resolve_template(
            base_dir, spec.output.as_deref(), spec.inline_output.as_deref(),
            0, context_types, registry,
        ) {
            Ok(block) => {
                all_context_keys.extend(block.context_keys.iter().cloned());
                Some(block)
            }
            Err(e) => return Err(vec![e]),
        }
    } else {
        None
    };

    Ok(CompiledNode {
        name: spec.name.clone(),
        provider: spec.provider.clone(),
        model: spec.model.clone(),
        tools: spec.tools.clone(),
        messages: compiled_messages,
        all_context_keys,
        strategy: spec.strategy.clone(),
        key_module,
        output_module,
    })
}

/// Resolve a template from either a file path or inline source.
fn resolve_template(
    base_dir: &Path,
    file: Option<&str>,
    inline: Option<&str>,
    block_idx: usize,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledBlock, OrchError> {
    match (file, inline) {
        (Some(path), _) => compile_template(base_dir, path, block_idx, context_types, registry),
        (None, Some(src)) => compile_source(src, block_idx, context_types, registry),
        (None, None) => Err(OrchError::new(OrchErrorKind::TemplateParse {
            block: block_idx,
            error: "no template or inline_template provided".into(),
        })),
    }
}

fn compile_template(
    base_dir: &Path,
    template: &str,
    block_idx: usize,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledBlock, OrchError> {
    let path = base_dir.join(template);
    let source = std::fs::read_to_string(&path).map_err(|e| {
        OrchError::new(OrchErrorKind::TemplateLoad {
            path: path.display().to_string(),
            error: e.to_string(),
        })
    })?;
    compile_source(&source, block_idx, context_types, registry)
}

fn compile_source(
    source: &str,
    block_idx: usize,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledBlock, OrchError> {
    let ast = acvus_ast::parse(source).map_err(|e| {
        OrchError::new(OrchErrorKind::TemplateParse {
            block: block_idx,
            error: format!("{e:?}"),
        })
    })?;

    let (module, _hints) = acvus_mir::compile(&ast, context_types.clone(), registry).map_err(
        |errs| {
            OrchError::new(OrchErrorKind::TemplateCompile {
                block: block_idx,
                errors: errs,
            })
        },
    )?;

    let context_keys = extract_context_keys(&module);
    let val_def = ValDefMapAnalysis.run(&module, ());

    Ok(CompiledBlock {
        role: String::new(),
        module,
        context_keys,
        val_def,
    })
}

/// Compile multiple node specs, merging their output types into context automatically.
///
/// `injected_types` are externally declared context types (from project.toml).
/// Each node name is also added as `Ty::String` context.
pub fn compile_nodes(
    specs: &[NodeSpec],
    base_dir: &Path,
    injected_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let mut context_types = injected_types.clone();
    for spec in specs {
        context_types.insert(spec.name.clone(), Ty::String);
    }

    let mut nodes = Vec::new();
    let mut errors = Vec::new();
    for spec in specs {
        match compile_node(spec, base_dir, &context_types, registry) {
            Ok(node) => nodes.push(node),
            Err(errs) => errors.extend(errs),
        }
    }
    if errors.is_empty() { Ok(nodes) } else { Err(errors) }
}

/// Extract all context keys referenced by `ContextLoad` instructions in a module.
fn extract_context_keys(module: &MirModule) -> HashSet<String> {
    let mut keys = HashSet::new();

    for inst in &module.main.insts {
        if let InstKind::ContextLoad { name, .. } = &inst.kind {
            keys.insert(name.clone());
        }
    }

    for closure in module.closures.values() {
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind {
                keys.insert(name.clone());
            }
        }
    }

    keys
}
