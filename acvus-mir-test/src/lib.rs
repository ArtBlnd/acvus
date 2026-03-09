use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::printer::dump_with;
use acvus_mir::ty::Ty;
use acvus_mir::user_type::UserTypeRegistry;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

/// Parse a template and compile to MIR, returning the printed IR.
pub fn compile_to_ir(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<String, String> {
    let template = acvus_ast::parse(interner, source).map_err(|e| format!("parse error: {e}"))?;
    let (module, _hints) = acvus_mir::compile(
        interner,
        &template,
        context,
        registry,
        &UserTypeRegistry::new(),
    )
    .map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e.display(interner)))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(dump_with(interner, &module))
}

/// Shorthand: compile with empty context.
pub fn compile_simple(interner: &Interner, source: &str) -> Result<String, String> {
    compile_to_ir(
        interner,
        source,
        &FxHashMap::default(),
        &ExternRegistry::new(),
    )
}

/// Common context types for tests.
pub fn user_context(interner: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        interner.intern("user"),
        Ty::Object(FxHashMap::from_iter([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), Ty::Int),
            (interner.intern("email"), Ty::String),
        ])),
    )])
}

pub fn users_list_context(interner: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        interner.intern("users"),
        Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), Ty::Int),
        ])))),
    )])
}

pub fn items_context(interner: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(interner.intern("items"), Ty::List(Box::new(Ty::Int)))])
}
