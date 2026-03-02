use std::collections::{HashMap, HashSet};
use std::path::Path;

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;

use crate::dsl::{NodeSpec, ToolDecl};
use crate::error::{OrchError, OrchErrorKind};

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub tools: Vec<ToolDecl>,
    pub blocks: Vec<CompiledBlock>,
    pub all_context_keys: HashSet<String>,
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub role: String,
    pub module: MirModule,
    pub context_keys: HashSet<String>,
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
    let mut compiled_blocks = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, msg) in spec.messages.iter().enumerate() {
        let path = base_dir.join(&msg.template);
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                errors.push(OrchError::new(OrchErrorKind::TemplateLoad {
                    path: path.display().to_string(),
                    error: e.to_string(),
                }));
                continue;
            }
        };

        let template = match acvus_ast::parse(&source) {
            Ok(t) => t,
            Err(e) => {
                errors.push(OrchError::new(OrchErrorKind::TemplateParse {
                    block: i,
                    error: format!("{e:?}"),
                }));
                continue;
            }
        };

        let (module, _hints) =
            match acvus_mir::compile(&template, context_types.clone(), registry) {
                Ok(m) => m,
                Err(errs) => {
                    errors.push(OrchError::new(OrchErrorKind::TemplateCompile {
                        block: i,
                        errors: errs,
                    }));
                    continue;
                }
            };

        let context_keys = extract_context_keys(&module);
        all_context_keys.extend(context_keys.iter().cloned());

        compiled_blocks.push(CompiledBlock {
            role: msg.role.clone(),
            module,
            context_keys,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(CompiledNode {
        name: spec.name.clone(),
        provider: spec.provider.clone(),
        model: spec.model.clone(),
        tools: spec.tools.clone(),
        blocks: compiled_blocks,
        all_context_keys,
    })
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
