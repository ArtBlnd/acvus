use std::fmt;

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{Externs, Registry, TypesOnly};
use acvus_mir::cfg;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower};
use acvus_mir::ir::MirModule;
use acvus_mir::laws::LawTable;
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{ObjectTy, PolyBuilder, PolyParam, Ty, TyTerm, TypeRegistry, lift_declaration};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

/// Build an inferred `Function` with the given qref, kind, and optional params.
pub fn inferred_function(qref: QualifiedRef, kind: FnKind, params: Vec<PolyParam>) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref,
        kind,
        ty: TyTerm::Fn {
            params,
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }
}

/// Combine the standard registries: their functions join `functions`, and
/// the combined type registry is returned for the checker.
fn extend_with_std(interner: &Interner, functions: &mut Vec<Function>) -> TypeRegistry {
    extend_with_registries(interner, functions, vec![])
}

/// `extend_with_std` with registries of the caller's own beside the standard
/// ones. A test that declares an extension type reaches for this rather than
/// an `extern_fns` list, because a `Function` carries no type declaration and
/// the checker refuses a type its registry does not hold.
fn extend_with_registries(
    interner: &Interner,
    functions: &mut Vec<Function>,
    own: Vec<Registry<TypesOnly>>,
) -> TypeRegistry {
    let mut registries = acvus_ext::std_registries::<TypesOnly>();
    registries.extend(own);
    let Externs {
        functions: combined,
        types,
        handlers: _,
        ..
    } = Externs::combine(registries, interner).expect("the registries combine");
    functions.extend(combined);
    types
}

fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
) -> Result<MirModule, String> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(interner, graph, &ext);

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

    let result = graph_lower::lower(interner, graph, &ext.view(), &inf);

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

    let early_moves = acvus_mir::validate::move_check::check_moves(&module);
    if !early_moves.is_empty() {
        let msgs: Vec<String> = early_moves
            .iter()
            .map(|e| format!("[validate:{}] {}", "test", e.display(interner)))
            .collect();
        return Err(msgs.join("\n"));
    }

    // SSA: promote whole reads and writes of locals to SSA form.
    let mut cfg_main = cfg::promote(std::mem::take(&mut module.main));
    acvus_mir::optimize::ssa_pass::run(&mut cfg_main);
    acvus_mir::optimize::string_copy::run(&mut cfg_main);
    module.main = cfg::demote(cfg_main);
    for closure in module.closures.values_mut() {
        let mut cfg_closure = cfg::promote(std::mem::take(closure));
        acvus_mir::optimize::ssa_pass::run(&mut cfg_closure);
        acvus_mir::optimize::string_copy::run(&mut cfg_closure);
        *closure = cfg::demote(cfg_closure);
    }

    let validation_errors =
        acvus_mir::validate::validate(&module, &LawTable::of(graph.functions.iter()));
    if !validation_errors.is_empty() {
        let msgs: Vec<String> = validation_errors
            .iter()
            .map(|e| format!("{}", e.display(interner)))
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
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Template(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };
    let module = run_pipeline(interner, &graph, test_qref)?;
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
        Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), Ty::I64),
            (interner.intern("email"), Ty::String),
        ]))),
    )])
}

pub fn items_context(interner: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        interner.intern("items"),
        Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
    )])
}

/// Compile a **script** source via the graph pipeline and return printed IR.
pub fn compile_script_ir(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    compile_script_ir_with(interner, source, context, &[])
}

/// Compile a script with both contexts and extern functions.
pub fn compile_script_ir_with(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    extern_fns: &[Function],
) -> Result<String, String> {
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };
    let module = run_pipeline(interner, &graph, test_qref)?;
    Ok(dump_with(interner, &module))
}

/// Compile a **script** with **no optimization** - raw lowered MIR.
pub fn compile_script_raw(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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
    compile_script_mode_ir_with(interner, source, context, &[])
}

/// Compile a script-mode source with contexts and extern functions, through
/// lowering and every validation.
pub fn compile_script_mode_ir_with(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    extern_fns: &[Function],
) -> Result<String, String> {
    refuse_script_mode_ir_with(interner, source, context, extern_fns).map_err(|refusals| {
        refusals
            .iter()
            .map(Refusal::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// `compile_script_mode_ir_with`, keeping each refusal's labels.
pub fn refuse_script_mode_ir_with(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    extern_fns: &[Function],
) -> Result<String, Vec<Refusal>> {
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => {
            return Err(vec![Refusal {
                stage: "parse".to_string(),
                message: format!("parse error: {e:?}"),
                primary: None,
                span: Span::new(0, 0),
                labels: Vec::new(),
            }]);
        }
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut refusals: Vec<Refusal> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            refusals.push(Refusal {
                stage: format!("infer:{fn_name}"),
                message: e.display(interner).to_string(),
                primary: e.primary(),
                span: e.span,
                labels: e.labels.clone(),
            });
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        refusals.push(Refusal {
            stage: "lower".to_string(),
            message: format!("[{}..{}] {}", e.span.start, e.span.end, e.display(interner)),
            primary: e.primary(),
            span: e.span,
            labels: e.labels.clone(),
        });
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let module = result.module(test_qref).cloned().ok_or_else(|| {
        vec![Refusal {
            stage: "lower".to_string(),
            message: "no module produced for target".to_string(),
            primary: None,
            span: Span::new(0, 0),
            labels: Vec::new(),
        }]
    })?;
    Ok(dump_with(interner, &module))
}

/// Lower a script-mode source with the standard registries, before any
/// optimization: the module as the checker decided it. The entry's return
/// type is inferred from the body.
pub fn lowered_script_module(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
) -> Result<MirModule, String> {
    lowered_script(interner, source, extern_fns, vec![]).map(|lowered| lowered.module)
}

/// `lowered_script_module` with registries of the caller's own beside the
/// standard ones, so that a test can declare an extension type and the
/// instances it holds.
pub fn lowered_script_module_with_registries(
    interner: &Interner,
    source: &str,
    own: Vec<Registry<TypesOnly>>,
) -> Result<MirModule, String> {
    lowered_script(interner, source, &[], own).map(|lowered| lowered.module)
}

pub struct LoweredScript {
    pub module: MirModule,
    pub laws: LawTable,
}

pub fn lowered_script(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
    own: Vec<Registry<TypesOnly>>,
) -> Result<LoweredScript, String> {
    let mut pb = PolyBuilder::new();
    lower_script_returning(interner, source, extern_fns, own, pb.fresh_ty_var())
}

/// `lowered_script_module` for a host that declares what the entry returns
/// (RFC-0054): the checker holds the body to `ret`, and the module carries
/// `ret` to `validate`.
pub fn declared_script_module(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
    ret: Ty,
) -> Result<MirModule, String> {
    let mut pb = PolyBuilder::new();
    let declared = lift_declaration(&ret, &mut pb);
    lower_script_returning(interner, source, extern_fns, vec![], declared)
        .map(|lowered| lowered.module)
}

fn lower_script_returning(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
    own: Vec<Registry<TypesOnly>>,
    ret: acvus_mir::ty::PolyTy,
) -> Result<LoweredScript, String> {
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast =
        acvus_ast::parse_script(interner, source).map_err(|e| format!("parse error: {e:?}"))?;
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(ret),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }];
    let type_registry = extend_with_registries(interner, &mut functions, own);
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{fn_name}] {}", e.display(interner)));
        }
    }
    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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
    let module = result
        .module(test_qref)
        .cloned()
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(LoweredScript {
        module,
        laws: LawTable::of(graph.functions.iter()),
    })
}

/// Lower a script-mode source with the standard registries and run the full
/// optimization pipeline over it, returning the module.
pub fn optimized_script_module(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
) -> Result<MirModule, String> {
    optimized_script(interner, source, extern_fns, vec![]).map(|optimized| optimized.module)
}

/// `optimized_script_module` with registries of the caller's own beside the
/// standard ones, and the law table the pipeline read.
pub fn optimized_script(
    interner: &Interner,
    source: &str,
    extern_fns: &[Function],
    own: Vec<Registry<TypesOnly>>,
) -> Result<LoweredScript, String> {
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast =
        acvus_ast::parse_script(interner, source).map_err(|e| format!("parse error: {e:?}"))?;
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_registries(interner, &mut functions, own);
    functions.extend_from_slice(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{fn_name}] {}", e.display(interner)));
        }
    }
    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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

    let laws = LawTable::of(graph.functions.iter());
    let opt = acvus_mir::graph::optimize::optimize(interner, &laws, result.modules, Opt::Full);
    for (qref, errs) in &opt.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{fn_name}] {}", e.display(interner)));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let module = opt
        .modules
        .get(&test_qref)
        .cloned()
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(LoweredScript { module, laws })
}

/// Compile a **script** with the **full optimization pipeline** (SROA -> SSA -> Inline -> Pass2).
/// Returns printed IR of the optimized module.
pub fn compile_script_optimized(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    let module = compile_script_module_at(interner, source, context, Opt::Full)?;
    Ok(dump_with(interner, &module))
}

/// Compile a **script** through the optimization pipeline at `opt`, returning
/// the module rather than its listing.
pub fn compile_script_module_at(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    opt: Opt,
) -> Result<MirModule, String> {
    compile_script_at(interner, source, context, opt).map(|compiled| compiled.module)
}

/// `compile_script_module_at` with the law table the pipeline read, for a
/// test that runs an analysis over the result.
pub fn compile_script_at(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
    opt: Opt,
) -> Result<LoweredScript, String> {
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => return Err(format!("parse error: {e:?}")),
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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

    let laws = LawTable::of(graph.functions.iter());
    let opt_result = acvus_mir::graph::optimize::optimize(interner, &laws, result.modules, opt);

    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{}] {}", fn_name, e.display(interner)));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = opt_result
        .modules
        .get(&test_qref)
        .cloned()
        .ok_or_else(|| "no module produced for target".to_string())?;
    Ok(LoweredScript { module, laws })
}

/// One stage's refusal of a source: the stage that raised it, the words, the
/// words its primary marker carries, and the other places it points at.
pub struct Refusal {
    pub stage: String,
    pub message: String,
    pub primary: Option<String>,
    pub span: Span,
    pub labels: Vec<Label>,
}

impl Refusal {
    /// The source the primary span covers.
    pub fn at<'s>(&self, source: &'s str) -> &'s str {
        &source[self.span.start..self.span.end]
    }

    /// Each label as the source it covers and the words it carries, which is
    /// what a test asserting a second place can read.
    pub fn marked(&self, source: &str) -> Vec<Marked> {
        self.labels
            .iter()
            .map(|l| Marked {
                source: l.span.map(|s| source[s.start..s.end].to_string()),
                text: l.text.clone(),
            })
            .collect()
    }
}

/// A label as a test reads it: the source its span covers, absent for a
/// spanless note, and its words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Marked {
    pub source: Option<String>,
    pub text: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.stage, self.message)
    }
}

/// Compile a script-mode source through the full pipeline; the printed IR, or
/// every refusal with its labels.
pub fn refuse_script_mode_optimized(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, Vec<Refusal>> {
    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = match acvus_ast::parse_script(interner, source) {
        Ok(ast) => ast,
        Err(e) => {
            return Err(vec![Refusal {
                stage: "parse".to_string(),
                message: format!("{e:?}"),
                primary: None,
                span: Span::new(0, 0),
                labels: Vec::new(),
            }]);
        }
    };
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    let type_registry = extend_with_std(interner, &mut functions);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut refusals: Vec<Refusal> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            refusals.push(Refusal {
                stage: format!("infer:{fn_name}"),
                message: e.display(interner).to_string(),
                primary: e.primary(),
                span: e.span,
                labels: e.labels.clone(),
            });
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        refusals.push(Refusal {
            stage: "lower".to_string(),
            message: e.display(interner).to_string(),
            primary: e.primary(),
            span: e.span,
            labels: e.labels.clone(),
        });
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let opt_result = acvus_mir::graph::optimize::optimize(interner, &LawTable::of(graph.functions.iter()), result.modules, Opt::Full);

    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            refusals.push(Refusal {
                stage: format!("validate:{fn_name}"),
                message: e.display(interner).to_string(),
                primary: None,
                span: e.span,
                labels: e.labels().to_vec(),
            });
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let module = opt_result.modules.get(&test_qref).ok_or_else(|| {
        vec![Refusal {
            stage: "optimize".to_string(),
            message: "no module produced for target".to_string(),
            primary: None,
            span: Span::new(0, 0),
            labels: Vec::new(),
        }]
    })?;
    Ok(dump_with(interner, module))
}

/// Compile a script-mode source through the full pipeline; the printed IR, or
/// every error, one line each.
pub fn compile_script_mode_optimized(
    interner: &Interner,
    source: &str,
    context: &FxHashMap<Astr, Ty>,
) -> Result<String, String> {
    refuse_script_mode_optimized(interner, source, context).map_err(|refusals| {
        refusals
            .iter()
            .map(Refusal::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    })
}

// -- Inline pipeline -------------------------------------------------

/// Compile multiple local functions, inline, and return the printed IR for the target.
///
/// `target`: (name, script_source) - the function whose inlined IR is returned.
/// `helpers`: list of (name, script_source, signature) - local functions callable from target.
/// `contexts`: context types available to all functions.
pub fn compile_inline_ir(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
) -> Result<String, String> {
    compile_inline_ir_with(interner, target, helpers, contexts, &[])
}

/// Like `compile_inline_ir` but also accepts extern functions.
pub fn compile_inline_ir_with(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    let mut pb = PolyBuilder::new();
    let ctx_vec: Vec<Context> = contexts
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();

    let target_qref = QualifiedRef::root(interner.intern(target.0));
    let target_ast = acvus_ast::parse_script(interner, target.1)
        .map_err(|e| format!("parse error in target '{}': {e:?}", target.0))?;

    let mut functions = vec![inferred_function(
        target_qref,
        FnKind::Local(ParsedAst::Script(target_ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];

    for (name, source, params) in helpers {
        let qref = QualifiedRef::root(interner.intern(name));
        let ast = acvus_ast::parse_script(interner, source)
            .map_err(|e| format!("parse error in helper '{}': {e:?}", name))?;
        functions.push(inferred_function(
            qref,
            FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::Declared),
            params.clone(),
        ));
    }

    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(ctx_vec),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    };

    // Run extract -> infer -> lower (full pipeline).
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

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

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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

    // Inline (no recursive functions in tests - pass empty set).
    let inlined = acvus_mir::graph::inliner::inline(&result.modules, &FxHashSet::default());

    inlined
        .modules
        .get(&target_qref)
        .map(|m| dump_with(interner, m))
        .ok_or_else(|| "no inlined module for target".to_string())
}

/// Compile multiple local functions - **raw lower only**, no optimization.
/// Returns printed IR of ALL local modules (since no inlining happens).
pub fn compile_multi_fn_raw(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    let ast = acvus_ast::parse_script(interner, target.1)
        .map_err(|e| format!("parse error in target '{}': {e:?}", target.0))?;
    compile_graph_raw(
        interner,
        (target.0, ParsedAst::Script(ast)),
        helpers,
        contexts,
        extern_fns,
    )
}

pub fn compile_template_with_helpers(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    let ast = acvus_ast::parse(interner, target.1)
        .map_err(|e| format!("parse error in target '{}': {e:?}", target.0))?;
    compile_graph_raw(
        interner,
        (target.0, ParsedAst::Template(ast)),
        helpers,
        contexts,
        extern_fns,
    )
}

fn compile_graph_raw(
    interner: &Interner,
    target: (&str, ParsedAst),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    let mut pb = PolyBuilder::new();
    let ctx_vec: Vec<Context> = contexts
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();

    let target_qref = QualifiedRef::root(interner.intern(target.0));

    let mut functions = vec![inferred_function(
        target_qref,
        FnKind::Local(target.1, acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];

    for (name, source, params) in helpers {
        let qref = QualifiedRef::root(interner.intern(name));
        let ast = acvus_ast::parse_script(interner, source)
            .map_err(|e| format!("parse error in helper '{}': {e:?}", name))?;
        functions.push(inferred_function(
            qref,
            FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::Declared),
            params.clone(),
        ));
    }

    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(ctx_vec),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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
        output.push_str(&format!("-- {} --\n", fn_name));
        output.push_str(&dump_with(interner, module));
        output.push('\n');
    }

    Ok(output)
}

/// Compile multiple local functions through the **full optimization pipeline**.
/// Includes: SROA -> SSA -> DSE -> Inline -> Pass2 (SpawnSplit -> SSA -> DSE -> CodeMotion -> Reorder -> RegColor -> Validate).
///
/// `target`: (name, script_source) - the function whose optimized IR is returned.
/// `helpers`: (name, script_source, signature) - local functions callable from target.
/// `contexts`: context types.
/// `extern_fns`: additional extern function declarations.
pub fn compile_multi_fn_optimized(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    compile_multi_fn_at(interner, target, helpers, contexts, extern_fns, Opt::Full)
}

fn compile_multi_fn_at(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
    opt: Opt,
) -> Result<String, String> {
    multi_fn_module_at(interner, target, helpers, contexts, extern_fns, opt)
        .map(|compiled| dump_with(interner, &compiled.module))
}

/// `compile_multi_fn_optimized`, listing each `For` with what
/// `analysis::loop_deps` computes of its stages.
pub fn compile_multi_fn_optimized_with_facts(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
) -> Result<String, String> {
    multi_fn_module_at(interner, target, helpers, contexts, extern_fns, Opt::Full).map(
        |compiled| {
            acvus_mir::printer::dump_with_facts(interner, &compiled.module, &compiled.laws)
        },
    )
}

pub fn multi_fn_module_at(
    interner: &Interner,
    target: (&str, &str),
    helpers: &[(&str, &str, Vec<PolyParam>)],
    contexts: &[(&str, Ty)],
    extern_fns: &[Function],
    opt: Opt,
) -> Result<LoweredScript, String> {
    let mut pb = PolyBuilder::new();
    let ctx_vec: Vec<Context> = contexts
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_declaration(ty, &mut pb),
            init: None,
        })
        .collect();

    let target_qref = QualifiedRef::root(interner.intern(target.0));
    let target_ast = acvus_ast::parse_script(interner, target.1)
        .map_err(|e| format!("parse error in target '{}': {e:?}", target.0))?;

    let mut functions = vec![inferred_function(
        target_qref,
        FnKind::Local(ParsedAst::Script(target_ast), acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];

    for (name, source, params) in helpers {
        let qref = QualifiedRef::root(interner.intern(name));
        let ast = acvus_ast::parse_script(interner, source)
            .map_err(|e| format!("parse error in helper '{}': {e:?}", name))?;
        functions.push(inferred_function(
            qref,
            FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::Declared),
            params.clone(),
        ));
    }

    let type_registry = extend_with_std(interner, &mut functions);
    functions.extend_from_slice(extern_fns);

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(ctx_vec),
        types: Freeze::new(type_registry),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{}] {}", fn_name, e.display(interner)));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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

    let laws = LawTable::of(graph.functions.iter());
    let opt_result = acvus_mir::graph::optimize::optimize(interner, &laws, result.modules, opt);

    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{}] {}", fn_name, e.display(interner)));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = opt_result
        .modules
        .get(&target_qref)
        .cloned()
        .ok_or_else(|| "no module for target".to_string())?;
    Ok(LoweredScript { module, laws })
}

/// `inputs` is read off the code that survived the passes, which is what
/// makes it the set RFC-0071 rule 5 calls required.
#[derive(Debug)]
pub struct BoundModule {
    pub ir: String,
    pub inputs: Vec<ShownInput>,
    pub helper_ir_by_name: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShownInput {
    pub name: String,
    pub ty: String,
}

pub fn compile_template_bound(
    interner: &Interner,
    source: &str,
    bound: &[(&str, acvus_mir::graph::BoundValue)],
    opt: Opt,
) -> Result<BoundModule, String> {
    let ast = acvus_ast::parse(interner, source).map_err(|e| format!("parse error: {e:?}"))?;
    compile_bound(interner, ParsedAst::Template(ast), &[], bound, opt)
}

/// A script entry `test` reading its inputs from its body, beside
/// `helpers` whose inputs are their declared parameters, with `bound`
/// bound; the result is `test`'s module.
pub fn compile_script_bound(
    interner: &Interner,
    source: &str,
    helpers: &[(&str, &str, Vec<PolyParam>)],
    bound: &[(&str, acvus_mir::graph::BoundValue)],
    opt: Opt,
) -> Result<BoundModule, String> {
    let ast =
        acvus_ast::parse_script(interner, source).map_err(|e| format!("parse error: {e:?}"))?;
    compile_bound(interner, ParsedAst::Script(ast), helpers, bound, opt)
}

fn compile_bound(
    interner: &Interner,
    target: ParsedAst,
    helpers: &[(&str, &str, Vec<PolyParam>)],
    bound: &[(&str, acvus_mir::graph::BoundValue)],
    opt: Opt,
) -> Result<BoundModule, String> {
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let mut functions = vec![inferred_function(
        test_qref,
        FnKind::Local(target, acvus_mir::graph::Inputs::FromReads),
        vec![],
    )];
    for (name, source, params) in helpers {
        let ast = acvus_ast::parse_script(interner, source)
            .map_err(|e| format!("parse error in helper '{name}': {e:?}"))?;
        functions.push(inferred_function(
            QualifiedRef::root(interner.intern(name)),
            FnKind::Local(ParsedAst::Script(ast), acvus_mir::graph::Inputs::Declared),
            params.clone(),
        ));
    }
    let type_registry = extend_with_std(interner, &mut functions);
    let mut bindings = Bindings::default();
    for (name, value) in bound {
        bindings
            .bind(interner.intern(name), value.clone())
            .map_err(|refused| format!("[bind] ${name}: {refused}"))?;
    }
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(type_registry),
        bindings,
        access: acvus_mir::graph::Access::Sync,
        entries: vec![test_qref],
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);

    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[infer:{fn_name}] {}", e.display(interner)));
        }
    }
    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!("[lower] {}", e.display(interner)));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let opt_result = acvus_mir::graph::optimize::optimize(interner, &LawTable::of(graph.functions.iter()), result.modules, opt);
    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!("[validate:{fn_name}] {}", e.display(interner)));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let module = opt_result
        .modules
        .get(&test_qref)
        .ok_or_else(|| "no module produced for target".to_string())?;
    let inputs = opt_result
        .inputs
        .get(&test_qref)
        .ok_or_else(|| "no input list produced for target".to_string())?
        .iter()
        .map(|input| ShownInput {
            name: interner.resolve(input.name.name).to_string(),
            ty: input.ty.display(interner).to_string(),
        })
        .collect();
    let helper_ir_by_name = helpers
        .iter()
        .map(|(name, _, _)| {
            let qref = QualifiedRef::root(interner.intern(name));
            let module = opt_result
                .modules
                .get(&qref)
                .ok_or_else(|| format!("no module produced for helper '{name}'"))?;
            Ok((name.to_string(), dump_with(interner, module)))
        })
        .collect::<Result<_, String>>()?;
    Ok(BoundModule {
        ir: dump_with(interner, module),
        inputs,
        helper_ir_by_name,
    })
}
