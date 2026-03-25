//! Phase 3: Lower
//!
//! Takes ResolvedGraph + cached ASTs and produces MirModule per function.
//! Reuses the existing MIR lowerer — this is just the orchestration layer.

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::ty::Ty;

use super::extract::{ExtractResult, ParsedSource};
use super::resolve::ResolvedGraph;
use super::types::*;

// ── Phase 3 output ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct LowerError {
    pub fn_id: FunctionId,
    pub errors: Vec<MirError>,
}

#[derive(Debug)]
pub struct LowerResult {
    pub modules: FxHashMap<FunctionId, (MirModule, HintTable)>,
    pub errors: Vec<LowerError>,
}

impl LowerResult {
    pub fn module(&self, id: FunctionId) -> Option<&MirModule> {
        self.modules.get(&id).map(|(m, _)| m)
    }

    pub fn hints(&self, id: FunctionId) -> Option<&HintTable> {
        self.modules.get(&id).map(|(_, h)| h)
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// ── Lowering ────────────────────────────────────────────────────────

/// Run Phase 3: lower each local function to MIR.
pub fn lower(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    resolved: &ResolvedGraph,
) -> LowerResult {
    let mut modules = FxHashMap::default();
    let mut errors = Vec::new();

    for func in graph.functions.iter() {
        let FnKind::Local(_source) = &func.kind else {
            continue;
        };
        let Some(parsed) = extract.parsed.get(&func.id) else {
            continue;
        };
        let Some(resolution) = resolved.try_resolution(func.id) else {
            continue;
        };

        // Build name_to_id for this function: only contexts that this function references.
        let fn_refs = extract.fn_refs.get(&func.id);
        let name_to_id: FxHashMap<Astr, (QualifiedRef, Ty)> = match fn_refs {
            Some(refs) => refs
                .context_reads
                .iter()
                .chain(refs.context_writes.iter())
                .map(|r| {
                    let qref = *r;
                    let ty = resolved.context_type(&qref).cloned().unwrap_or(Ty::error());
                    (r.name, (qref, ty))
                })
                .collect(),
            None => FxHashMap::default(),
        };

        // Clone the resolution for lowering (lower consumes it).
        let resolution_clone = resolution.clone();

        // Build function_ids from graph functions.
        let function_ids: FxHashMap<Astr, (FunctionId, Ty)> = graph
            .functions
            .iter()
            .filter_map(|f| {
                let ty = match &f.constraint.output {
                    Constraint::Exact(ty) => ty.clone(),
                    _ => Ty::error(),
                };
                Some((f.name, (f.id, ty)))
            })
            .collect();

        // Build FunctionId → Ty map for SSA pass (callee effect resolution).
        let fn_type_map: FxHashMap<FunctionId, Ty> = function_ids
            .values()
            .map(|(id, ty)| (*id, ty.clone()))
            .collect();

        let lowerer = crate::lower::Lowerer::new(
            interner,
            resolution_clone.type_map,
            resolution_clone.coercion_map,
            Freeze::new(name_to_id),
            Freeze::new(function_ids),
        );
        let (mut module, hints) = match parsed {
            ParsedSource::Script(script) => lowerer.lower_script(script),
            ParsedSource::Template(template) => lowerer.lower_template(template),
        };

        // SSA context pass: promote ContextProject/Load/Store to SSA form.
        crate::ssa_pass::run(&mut module.main, &fn_type_map);

        let validation_errors = crate::validate::validate(&module);
        let result: Result<_, Vec<crate::error::MirError>> = if validation_errors.is_empty() {
            Ok((module, hints))
        } else {
            Err(validation_errors
                .into_iter()
                .map(|e| e.into_mir_error())
                .collect())
        };

        match result {
            Ok((module, hints)) => {
                modules.insert(func.id, (module, hints));
            }
            Err(errs) => {
                errors.push(LowerError {
                    fn_id: func.id,
                    errors: errs,
                });
            }
        }
    }

    LowerResult { modules, errors }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{extract, resolve};
    use crate::ir::InstKind;
    use Ty;
    use acvus_utils::Interner;

    fn make_graph_with_ctx(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        CompilationGraph {
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
                },
            }]),
            contexts: Freeze::new(contexts),
        }
    }

    fn first_fn_id(graph: &CompilationGraph) -> FunctionId {
        graph.functions[0].id
    }

    // -- Completeness: valid programs lower to MIR --

    #[test]
    fn lower_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "1 + 2", &[]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_id(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_with_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_id(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let graph = make_graph_with_ctx(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
    }

    // -- Script: match-bind and iterate --

    #[test]
    fn lower_script_irrefutable_match_bind() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "x = @data { @out = x + 1; }; @out",
            &[("data", Ty::Int), ("out", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_id(&graph)).unwrap();
        // Irrefutable match-bind should NOT generate JumpIf.
        assert!(
            !module
                .main
                .insts
                .iter()
                .any(|i| matches!(i.kind, InstKind::JumpIf { .. })),
            "irrefutable match-bind should not generate JumpIf"
        );
    }

    #[test]
    fn lower_script_refutable_match_bind() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "42 = @val { @out = 1; }; @out",
            &[("val", Ty::Int), ("out", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_id(&graph)).unwrap();
        // Refutable match-bind MUST generate JumpIf.
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(i.kind, InstKind::JumpIf { .. })),
            "refutable match-bind should generate JumpIf"
        );
    }

    #[test]
    fn lower_script_iterate() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "x in @items { @sum = @sum + x; }; @sum",
            &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_id(&graph)).unwrap();
        // Iterate must generate IterStep.
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(i.kind, InstKind::IterStep { .. })),
            "iterate should generate IterStep"
        );
    }

    #[test]
    fn lower_script_nested_iterate() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "row in @matrix { x in row { @sum = @sum + x; }; }; @sum",
            &[
                ("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
                ("sum", Ty::Int),
            ],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());
        let result = lower(&i, &graph, &ext, &res);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_id(&graph)).unwrap();
        // Nested iterate: two IterStep instructions.
        let iter_count = module
            .main
            .insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::IterStep { .. }))
            .count();
        assert_eq!(iter_count, 2, "nested iterate should have 2 IterStep");
    }

    // -- Soundness: type errors don't produce modules --

    #[test]
    fn lower_type_error_no_module() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let res = resolve::resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        // Resolve should have errors, so no resolution for the unit.
        // Lower should produce no module for this unit.
        let result = lower(&i, &graph, &ext, &res);
        let uid = first_fn_id(&graph);
        assert!(result.module(uid).is_none());
    }
}
