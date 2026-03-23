//! Test helpers for compiling source → MIR via the graph pipeline.
//!
//! These are convenience wrappers around extract → resolve → lower.
//! Real callers should use the graph phases directly.

use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::graph::*;
use crate::graph::{extract, infer, lower as graph_lower, resolve};
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::ty::Ty;

/// Build a single-unit CompilationGraph for testing.
/// Returns the graph and the `FunctionId` of the test unit.
fn make_graph(
    interner: &Interner,
    source: &str,
    kind: SourceKind,
    ctx: &[(&str, Ty)],
) -> (CompilationGraph, FunctionId) {
    let contexts = ctx
        .iter()
        .map(|(name, ty)| Context {
            id: ContextId::alloc(),
            name: interner.intern(name),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_id = FunctionId::alloc();
    let mut functions = crate::builtins::standard_builtins(interner);
    functions.push(Function {
        id: test_id,
        name: interner.intern("test"),
        kind: FnKind::Local(SourceCode {
            name: interner.intern("test"),
                source: interner.intern(source),
                kind,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
            },
    });
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    (graph, test_id)
}

fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: FunctionId,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(interner, graph, &ext);
    let res = resolve::resolve(interner, graph, &ext, &inf, &FxHashMap::default());
    let result = graph_lower::lower(interner, graph, &ext, &res);
    if result.has_errors() {
        return Err(result.errors.into_iter().flat_map(|e| e.errors).collect());
    }
    result
        .modules
        .into_iter()
        .find(|(id, _)| *id == target)
        .map(|(_, pair)| pair)
        .ok_or_else(Vec::new)
}

/// Compile a template source string through the full graph pipeline.
pub fn compile_template(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let (graph, target) = make_graph(interner, source, SourceKind::Template, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a script source string through the full graph pipeline.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let (graph, target) = make_graph(interner, source, SourceKind::Script, ctx);
    run_pipeline(interner, &graph, target)
}
