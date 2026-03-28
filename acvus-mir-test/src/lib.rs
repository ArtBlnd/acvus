use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{Param, Ty};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Parse a template and compile to MIR via the graph pipeline, returning the printed IR.
pub fn compile_to_ir(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    compile_to_ir_with(interner, source, context, &[])
}

/// Compile a template with both contexts and extern functions.
pub fn compile_to_ir_with(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    extern_fns: &[Function],
) -> Result<String, String> {
    let ctx: Vec<(&str, Ty)> = context
        .iter()
        .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
        .collect();
    let contexts: Vec<Context> = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Template(ast)),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    }];
    // Register stdlib ExternFn registries (string, conversion, list, option).
    for registry in [
        acvus_ext::string_registry(),
        acvus_ext::conversion_registry(),
        acvus_ext::list_registry(),
        acvus_ext::option_registry(),
    ] {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default(), Freeze::default());
    let result = graph_lower::lower(interner, &graph, &ext, &inf);
    if result.has_errors() {
        let errs: Vec<String> = result
            .errors
            .iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e.display(interner)))
            .collect();
        return Err(errs.join("\n"));
    }
    let uid = graph.functions[0].qref;
    let module = result
        .module(uid)
        .ok_or_else(|| "no module produced".to_string())?;
    Ok(dump_with(interner, module))
}

/// Shorthand: compile with empty context.
pub fn compile_simple(interner: &Interner, source: &str) -> Result<String, String> {
    compile_to_ir(interner, source, &FxHashMap::default())
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

/// Compile a **script** source via the graph pipeline and return printed IR.
pub fn compile_script_ir(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let graph = CompilationGraph {
        functions: Freeze::new(vec![Function {
            qref: test_qref,
            kind: FnKind::Local(ParsedAst::Script(ast)),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        }]),
        contexts: Freeze::new(contexts),
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default(), Freeze::default());
    let result = graph_lower::lower(interner, &graph, &ext, &inf);
    if result.has_errors() {
        let errs: Vec<String> = result
            .errors
            .iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("[{}..{}] {}", e.span.start, e.span.end, e.display(interner)))
            .collect();
        return Err(errs.join("\n"));
    }
    let uid = graph.functions[0].qref;
    let module = result
        .module(uid)
        .ok_or_else(|| "no module produced".to_string())?;
    Ok(dump_with(interner, module))
}
