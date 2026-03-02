use std::collections::{HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;

use crate::dsl::{BlockAttrs, ConfigBlock, DslFile};
use crate::error::{OrchError, OrchErrorKind};

/// A compiled orchestration node (one DSL file).
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub config: ConfigBlock,
    pub blocks: Vec<CompiledBlock>,
    pub all_context_keys: HashSet<String>,
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub format: String,
    pub attrs: BlockAttrs,
    pub module: MirModule,
    pub context_keys: HashSet<String>,
}

/// Compile a parsed DSL file into a `CompiledNode`.
///
/// Each message block's template is parsed and compiled via acvus-ast/mir.
/// Context keys are extracted from MIR `ContextLoad` instructions.
pub fn compile_dsl(
    dsl: &DslFile,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledNode, Vec<OrchError>> {
    let mut compiled_blocks = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, block) in dsl.blocks.iter().enumerate() {
        let template = match acvus_ast::parse(&block.template_source) {
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
            format: block.format.clone(),
            attrs: block.attrs.clone(),
            module,
            context_keys,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(CompiledNode {
        config: dsl.config.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_simple_template() {
        let dsl = crate::parse::parse_dsl(
            r#"
#![configs]
name = "out"
model = "m"

#![fmt(type = "user")]
Hello world!
"#,
        )
        .unwrap();

        let registry = ExternRegistry::new();
        let node = compile_dsl(&dsl, &HashMap::new(), &registry).unwrap();
        assert_eq!(node.blocks.len(), 1);
        assert!(node.all_context_keys.is_empty());
    }

    #[test]
    fn extract_context_keys_from_template() {
        let dsl = crate::parse::parse_dsl(
            r#"
#![configs]
name = "out"
text = "@text"
model = "m"

#![fmt(type = "user")]
{{ @text }}
"#,
        )
        .unwrap();

        let mut ctx = HashMap::new();
        ctx.insert("text".into(), Ty::String);
        let registry = ExternRegistry::new();
        let node = compile_dsl(&dsl, &ctx, &registry).unwrap();
        assert!(node.all_context_keys.contains("text"));
    }
}
