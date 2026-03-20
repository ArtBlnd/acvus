use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{Ty, TySubst};
use acvus_mir::{typecheck_template, check_completeness, lower_checked_template};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

/// Parse a template and compile to MIR, returning the printed IR.
/// Uses the layered API: typecheck → check_completeness → lower.
pub fn compile_to_ir(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    let reg = ContextTypeRegistry::all_system(context.clone());
    let template = acvus_ast::parse(interner, source).map_err(|e| format!("parse error: {e}"))?;
    let mut subst = TySubst::new();
    let unchecked = typecheck_template(interner, &template, &reg, &mut subst)
        .map_err(|errors| format_errors(&errors, interner))?;
    let checked = check_completeness(unchecked, &subst)
        .map_err(|errors| format_errors(&errors, interner))?;
    let (module, _hints) = lower_checked_template(interner, &template, checked)
        .map_err(|errors| format_errors(&errors, interner))?;
    Ok(dump_with(interner, &module))
}

fn format_errors(errors: &[acvus_mir::error::MirError], interner: &Interner) -> String {
    errors
        .iter()
        .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e.display(interner)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Shorthand: compile with empty context.
pub fn compile_simple(interner: &Interner, source: &str) -> Result<String, String> {
    compile_to_ir(
        interner,
        source,
        &FxHashMap::default(),
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
