use std::collections::{BTreeMap, HashMap};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::printer::dump;
use acvus_mir::ty::Ty;
use acvus_mir::user_type::UserTypeRegistry;

/// Parse a template and compile to MIR, returning the printed IR.
pub fn compile_to_ir(
    source: &str,
    context: HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<String, String> {
    let template = acvus_ast::parse(source).map_err(|e| format!("parse error: {e}"))?;
    let (module, _hints) =
        acvus_mir::compile(&template, context, registry, &UserTypeRegistry::new()).map_err(
            |errors| {
                errors
                    .iter()
                    .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e))
                    .collect::<Vec<_>>()
                    .join("\n")
            },
        )?;
    Ok(dump(&module))
}

/// Shorthand: compile with empty context.
pub fn compile_simple(source: &str) -> Result<String, String> {
    compile_to_ir(source, HashMap::new(), &ExternRegistry::new())
}

/// Common context types for tests.
pub fn user_context() -> HashMap<String, Ty> {
    HashMap::from([(
        "user".into(),
        Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
            ("email".into(), Ty::String),
        ])),
    )])
}

pub fn users_list_context() -> HashMap<String, Ty> {
    HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])))),
    )])
}

pub fn items_context() -> HashMap<String, Ty> {
    HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))])
}
