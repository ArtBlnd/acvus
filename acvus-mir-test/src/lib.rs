use acvus_mir::cfg;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower};
use acvus_mir::ir::MirModule;
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{Param, Ty};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

/// Run extract → infer → lower, collecting errors from all passes.
fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
) -> Result<MirModule, String> {
    run_pipeline_with_registry(interner, graph, target, acvus_mir::ty::TypeRegistry::new())
}

fn run_pipeline_with_registry(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
    type_registry: acvus_mir::ty::TypeRegistry,
) -> Result<MirModule, String> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(interner, graph, &ext, &FxHashMap::default(), Freeze::new(type_registry));

    // Collect infer errors.
    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!(
                "[infer:{}] [{}..{}] {}",
                fn_name, e.span.start, e.span.end, e.display(interner)
            ));
        }
    }

    let result = graph_lower::lower(interner, graph, &ext, &inf, &FxHashSet::default());

    // Collect lower errors.
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!(
            "[lower] [{}..{}] {}",
            e.span.start, e.span.end, e.display(interner)
        ));
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let mut module = result
        .module(target)
        .cloned()
        .ok_or_else(|| "no module produced for target".to_string())?;

    // Build fn_types for SSA + validate.
    // Include both inferred (Complete) and declared (Exact constraint) function types.
    let mut fn_types: FxHashMap<QualifiedRef, acvus_mir::ty::Ty> = FxHashMap::default();
    for func in graph.functions.iter() {
        if let Constraint::Exact(ty) = &func.constraint.output {
            fn_types.insert(func.qref, ty.clone());
        }
    }
    for (qref, outcome) in &inf.outcomes {
        if let acvus_mir::graph::infer::FnInferOutcome::Complete { meta, .. } = outcome {
            fn_types.insert(*qref, meta.ty.clone());
        }
    }

    // Run SSA + validate (lower now outputs pre-SSA MIR).
    // promote → ssa → demote per body (CfgBody transition).
    let mut cfg_main = cfg::promote(std::mem::take(&mut module.main));
    acvus_mir::optimize::ssa_pass::run(&mut cfg_main, &fn_types);
    module.main = cfg::demote(cfg_main);
    for closure in module.closures.values_mut() {
        let mut cfg_closure = cfg::promote(std::mem::take(closure));
        acvus_mir::optimize::ssa_pass::run(&mut cfg_closure, &fn_types);
        *closure = cfg::demote(cfg_closure);
    }

    let validation_errors = acvus_mir::validate::validate(&module, &fn_types);
    if !validation_errors.is_empty() {
        let msgs: Vec<String> = validation_errors
            .iter() 
            .map(|e| format!("{:?}", e))
            .collect();
        return Err(msgs.join("\n"));
    }

    Ok(module)
}

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
    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    let module = run_pipeline_with_registry(interner, &graph, test_qref, type_registry)?;
    Ok(dump_with(interner, &module))
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
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Script(ast)),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    }];
    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    let module = run_pipeline_with_registry(interner, &graph, test_qref, type_registry)?;
    Ok(dump_with(interner, &module))
}

// ── Inline pipeline ─────────────────────────────────────────────────

/// Compile multiple local functions, inline, and return the printed IR for the target.
///
/// `target`: (name, script_source) — the function whose inlined IR is returned.
/// `helpers`: list of (name, script_source, signature) — local functions callable from target.
/// `contexts`: context types available to all functions.
pub fn compile_inline_ir(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Option<Signature>)],
    contexts: &[(&str, Ty)],
) -> Result<String, String> {
    compile_inline_ir_with(interner, target, helpers, contexts, &[])
}

/// Like `compile_inline_ir` but also accepts extern functions.
pub fn compile_inline_ir_with(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Option<Signature>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    let ctx_vec: Vec<Context> = contexts
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();

    let target_qref = QualifiedRef::root(interner.intern(target.0));
    let target_ast = acvus_ast::parse_script(interner, target.1)
        .map_err(|e| format!("parse error in target '{}': {e:?}", target.0))?;

    let mut functions = vec![Function {
        qref: target_qref,
        kind: FnKind::Local(ParsedAst::Script(target_ast)),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    }];

    for &(name, source, ref sig) in helpers {
        let qref = QualifiedRef::root(interner.intern(name));
        let ast = acvus_ast::parse_script(interner, source)
            .map_err(|e| format!("parse error in helper '{}': {e:?}", name))?;
        functions.push(Function {
            qref,
            kind: FnKind::Local(ParsedAst::Script(ast)),
            constraint: FnConstraint {
                signature: sig.clone(),
                output: Constraint::Inferred,
                effect: None,
            },
        });
    }

    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }
    functions.extend_from_slice(extern_fns);

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(ctx_vec),
    };

    // Run extract → infer → lower (full pipeline).
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!(
                "[infer:{}] [{}..{}] {}",
                fn_name, e.span.start, e.span.end, e.display(interner)
            ));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashSet::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!(
            "[lower] [{}..{}] {}",
            e.span.start, e.span.end, e.display(interner)
        ));
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // Extract MirModules for inlining.
    let modules: FxHashMap<QualifiedRef, MirModule> = result
        .modules
        .into_iter()
        .map(|(qref, (module, _))| (qref, module))
        .collect();

    // Inline (no recursive functions in tests — pass empty set).
    let inlined = acvus_mir::graph::inliner::inline(&modules, &FxHashSet::default());

    inlined
        .modules
        .get(&target_qref)
        .map(|m| dump_with(interner, m))
        .ok_or_else(|| "no inlined module for target".to_string())
}
