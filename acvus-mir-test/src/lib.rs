use acvus_mir::cfg;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower};
use acvus_mir::ir::MirModule;
use acvus_mir::printer::dump_with;
use acvus_mir::ty::Ty;
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
    let inf = infer::infer(
        interner,
        graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    // Collect infer errors.
    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!(
                "[infer:{}] [{}..{}] {}",
                fn_name,
                e.span.start,
                e.span.end,
                e.display(interner)
            ));
        }
    }

    let result = graph_lower::lower(interner, graph, &ext, &inf, &FxHashMap::default());

    // Collect lower errors.
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!(
            "[lower] [{}..{}] {}",
            e.span.start,
            e.span.end,
            e.display(interner)
        ));
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let mut module = result
        .module(target)
        .cloned()
        .ok_or_else(|| "no module produced for target".to_string())?;

    // Use InferResult.fn_metadata (authoritative, frozen).
    let fn_metadata = &inf.fn_metadata;

    // Init check: field-level definite assignment on CfgBody (pre-SROA).
    {
        let external_contexts: rustc_hash::FxHashSet<QualifiedRef> = graph
            .contexts
            .iter()
            .map(|c| c.qref)
            .collect();
        let cfg_main = cfg::promote(std::mem::take(&mut module.main));
        let init_errors = acvus_mir::validate::init_check::check_init(&cfg_main, fn_metadata, &external_contexts);
        module.main = cfg::demote(cfg_main);
        if !init_errors.is_empty() {
            let msgs: Vec<String> = init_errors
                .iter()
                .map(|e| format!(
                    "UninitError: {:?} fields {:?} at [{},{}]",
                    e.target, e.uninit_fields, e.span.start, e.span.end,
                ))
                .collect();
            return Err(msgs.join("\n"));
        }
    }

    // SROA: decompose field Refs into identity Refs + FieldGet/FieldSet.
    acvus_mir::optimize::sroa::run_body(&mut module.main, &inf.context_types);
    for closure in module.closures.values_mut() {
        acvus_mir::optimize::sroa::run_body(closure, &inf.context_types);
    }

    // SSA: promote identity Refs to SSA form.
    let mut cfg_main = cfg::promote(std::mem::take(&mut module.main));
    acvus_mir::optimize::ssa_pass::run(&mut cfg_main, fn_metadata);
    module.main = cfg::demote(cfg_main);
    for closure in module.closures.values_mut() {
        let mut cfg_closure = cfg::promote(std::mem::take(closure));
        acvus_mir::optimize::ssa_pass::run(&mut cfg_closure, fn_metadata);
        *closure = cfg::demote(cfg_closure);
    }

    let validation_errors = acvus_mir::validate::validate(&module, fn_metadata, &FxHashMap::default());
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
            hint: None,
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
            hint: None,
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

/// Compile a **script** with **no optimization** — raw lowered MIR.
pub fn compile_script_raw(
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
            hint: None,
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

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = result
        .module(test_qref)
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(dump_with(interner, module))
}

/// Compile a **script mode** source (keyword-based: let/if/else/for/while).
/// Returns printed IR of the pre-SSA module.
pub fn compile_script_mode_raw(
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
    let ast = match acvus_ast::parse_script_mode(interner, source) {
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
            hint: None,
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

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = result
        .module(test_qref)
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(dump_with(interner, module))
}

/// Compile a **script** with the **full optimization pipeline** (SROA → SSA → Inline → Pass2).
/// Returns printed IR of the optimized module.
pub fn compile_script_optimized(
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
            hint: None,
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

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let opt_result = acvus_mir::graph::optimize::optimize(
        result.modules,
        &inf.fn_metadata,
        &inf.context_types,
        &FxHashSet::default(),
    );

    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{}] {:?}", fn_name, e));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = opt_result
        .modules
        .get(&test_qref)
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(dump_with(interner, module))
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
            hint: None,
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
                hint: None,
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
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!(
                "[infer:{}] [{}..{}] {}",
                fn_name,
                e.span.start,
                e.span.end,
                e.display(interner)
            ));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!(
            "[lower] [{}..{}] {}",
            e.span.start,
            e.span.end,
            e.display(interner)
        ));
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // Inline (no recursive functions in tests — pass empty set).
    let inlined = acvus_mir::graph::inliner::inline(&result.modules, &FxHashSet::default());

    inlined
        .modules
        .get(&target_qref)
        .map(|m| dump_with(interner, m))
        .ok_or_else(|| "no inlined module for target".to_string())
}

/// Compile multiple local functions — **raw lower only**, no optimization.
/// Returns printed IR of ALL local modules (since no inlining happens).
pub fn compile_multi_fn_raw(
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
            hint: None,
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
                hint: None,
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

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // Print ALL local modules (sorted by name for determinism).
    let mut output = String::new();
    let mut entries: Vec<_> = result.modules.iter().collect();
    entries.sort_by_key(|(qref, _)| interner.resolve(qref.name).to_string());

    for (qref, module) in entries {
        let fn_name = interner.resolve(qref.name);
        // Skip extern functions (no meaningful body).
        if module.main.insts.is_empty() {
            continue;
        }
        output.push_str(&format!("── {} ──\n", fn_name));
        output.push_str(&dump_with(interner, module));
        output.push('\n');
    }

    Ok(output)
}

/// Compile multiple local functions through the **full optimization pipeline**.
/// Includes: SROA → SSA → DSE → Inline → Pass2 (SpawnSplit → SSA → DSE → CodeMotion → Reorder → RegColor → Validate).
///
/// `target`: (name, script_source) — the function whose optimized IR is returned.
/// `helpers`: (name, script_source, signature) — local functions callable from target.
/// `contexts`: context types.
/// `extern_fns`: additional extern function declarations.
pub fn compile_multi_fn_optimized(
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
            hint: None,
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
                hint: None,
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

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let opt_result = acvus_mir::graph::optimize::optimize(
        result.modules,
        &inf.fn_metadata,
        &inf.context_types,
        &FxHashSet::default(),
    );

    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{}] {:?}", fn_name, e));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = opt_result
        .modules
        .get(&target_qref)
        .ok_or_else(|| "no module for target".to_string())?;
    Ok(dump_with(interner, module))
}
