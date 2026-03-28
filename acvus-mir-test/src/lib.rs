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
            name: interner.intern(name),
            namespace: None,
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let mut functions = vec![Function {
        id: FunctionId::alloc(),
        name: interner.intern("test"),
        namespace: None,
        kind: FnKind::Local(SourceCode {
            name: interner.intern("test"),
            source: interner.intern(source),
            kind: SourceKind::Template,
        }),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    }];
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        namespaces: Freeze::new(vec![]),
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default());
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
    let uid = graph.functions[0].id;
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
            name: *name,
            namespace: None,
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let graph = CompilationGraph {
        namespaces: Freeze::new(vec![]),
        functions: Freeze::new(vec![Function {
            id: FunctionId::alloc(),
            name: interner.intern("test"),
            namespace: None,
            kind: FnKind::Local(SourceCode {
                name: interner.intern("test"),
                source: interner.intern(source),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        }]),
        contexts: Freeze::new(contexts),
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default());
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
    let uid = graph.functions[0].id;
    let module = result
        .module(uid)
        .ok_or_else(|| "no module produced".to_string())?;
    Ok(dump_with(interner, module))
}
