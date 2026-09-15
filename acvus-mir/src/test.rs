//! Test helpers for compiling source -> MIR via the graph pipeline.
//!
//! These are convenience wrappers around extract -> infer -> lower.
//! Real callers should use the graph phases directly.

use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::graph::*;
use crate::graph::{extract, infer, lower as graph_lower};
use crate::ir::{MirBody, MirModule};
use crate::ty::{PolyBuilder, Ty, TyTerm};

/// Build a single-unit CompilationGraph for testing.
/// Returns the graph and the `QualifiedRef` of the test unit.
pub(crate) fn make_graph(
    interner: &Interner,
    source: &str,
    is_template: bool,
    ctx: &[(&str, Ty)],
) -> (CompilationGraph, QualifiedRef) {
    let mut pb = PolyBuilder::new();
    let contexts = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: crate::ty::lift_declaration(ty, &mut pb),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let parsed = if is_template {
        ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse"))
    } else {
        ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"))
    };

    let graph = CompilationGraph {
        functions: Freeze::new(vec![Function {
            qref: test_qref,
            kind: FnKind::Local(parsed),
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        }]),
        contexts: Freeze::new(contexts),
    };
    (graph, test_qref)
}

fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
) -> Result<MirModule, String> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(
        interner,
        graph,
        &ext,
        &FxHashMap::default(),
        Freeze::default(),
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

    let result = graph_lower::lower(interner, graph, &ext, &inf);

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

    // Run SROA + SSA + validate (lower outputs pre-SSA MIR).
    let mut module = result
        .modules
        .into_iter()
        .find(|(id, _)| *id == target)
        .map(|(_, m)| m)
        .ok_or_else(|| "no module produced for target".to_string())?;

    // SSA -> DCE.
    {
        let mut cfg_body = crate::cfg::promote(std::mem::replace(&mut module.main, MirBody::new()));
        crate::optimize::ssa_pass::run(&mut cfg_body);
        crate::optimize::string_copy::run(&mut cfg_body);
        crate::optimize::dce::run(&mut cfg_body);
        module.main = crate::cfg::demote(cfg_body);
    }
    for closure in module.closures.values_mut() {
        let mut cfg_body = crate::cfg::promote(std::mem::replace(closure, MirBody::new()));
        crate::optimize::ssa_pass::run(&mut cfg_body);
        crate::optimize::string_copy::run(&mut cfg_body);
        crate::optimize::dce::run(&mut cfg_body);
        *closure = crate::cfg::demote(cfg_body);
    }

    let validation_errors = crate::validate::validate(&module);
    if !validation_errors.is_empty() {
        let msgs: Vec<String> = validation_errors
            .iter()
            .map(|e| format!("{:?}", e))
            .collect();
        return Err(msgs.join("\n"));
    }

    Ok(module)
}

/// Compile a template source string through the full graph pipeline.
pub fn compile_template(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<MirModule, String> {
    let (graph, target) = make_graph(interner, source, true, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a script source string through the full graph pipeline.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<MirModule, String> {
    let (graph, target) = make_graph(interner, source, false, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a template and return the printed IR. Panics with full error on failure.
pub fn compile_and_dump(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> String {
    let module =
        compile_template(interner, source, ctx).unwrap_or_else(|e| panic!("compile failed:\n{e}"));
    crate::printer::dump(interner, &module)
}
