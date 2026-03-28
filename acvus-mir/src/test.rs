//! Test helpers for compiling source → MIR via the graph pipeline.
//!
//! These are convenience wrappers around extract → infer → lower.
//! Real callers should use the graph phases directly.

use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::graph::*;
use crate::graph::{extract, infer, lower as graph_lower};
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::ty::Ty;

/// Build a single-unit CompilationGraph for testing.
/// Returns the graph and the `QualifiedRef` of the test unit.
pub(crate) fn make_graph(
    interner: &Interner,
    source: &str,
    is_template: bool,
    ctx: &[(&str, Ty)],
) -> (CompilationGraph, QualifiedRef) {
    let contexts = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let parsed = if is_template {
        ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse"))
    } else {
        ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"))
    };
    let mut functions = crate::builtins::standard_builtins(interner);
    functions.push(Function {
        qref: test_qref,
        kind: FnKind::Local(parsed),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    });
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    (graph, test_qref)
}

fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
) -> Result<(MirModule, HintTable), String> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(interner, graph, &ext, &FxHashMap::default(), Freeze::default());

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

    result
        .modules
        .into_iter()
        .find(|(id, _)| *id == target)
        .map(|(_, pair)| pair)
        .ok_or_else(|| "no module produced for target".to_string())
}

/// Compile a template source string through the full graph pipeline.
pub fn compile_template(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), String> {
    let (graph, target) = make_graph(interner, source, true, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a script source string through the full graph pipeline.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), String> {
    let (graph, target) = make_graph(interner, source, false, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a template and return the printed IR. Panics with full error on failure.
pub fn compile_and_dump(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> String {
    let (module, _) = compile_template(interner, source, ctx)
        .unwrap_or_else(|e| panic!("compile failed:\n{e}"));
    crate::printer::dump(interner, &module)
}
