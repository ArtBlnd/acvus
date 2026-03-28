//! Phase 3: Lower
//!
//! Takes InferResult (Complete outcomes) + cached ASTs and produces MirModule per function.
//! Reuses the existing MIR lowerer — this is just the orchestration layer.

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::ty::Ty;

use super::extract::{ExtractResult, ParsedSource};
use super::infer::InferResult;
use super::types::*;

// ── Phase 3 output ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct LowerError {
    pub fn_id: QualifiedRef,
    pub errors: Vec<MirError>,
}

#[derive(Debug)]
pub struct LowerResult {
    pub modules: FxHashMap<QualifiedRef, (MirModule, HintTable)>,
    pub errors: Vec<LowerError>,
}

impl LowerResult {
    pub fn module(&self, id: QualifiedRef) -> Option<&MirModule> {
        self.modules.get(&id).map(|(m, _)| m)
    }

    pub fn hints(&self, id: QualifiedRef) -> Option<&HintTable> {
        self.modules.get(&id).map(|(_, h)| h)
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// ── Lowering ────────────────────────────────────────────────────────

/// Run Phase 3: lower each Complete function to MIR.
pub fn lower(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    infer_result: &InferResult,
) -> LowerResult {
    let mut modules = FxHashMap::default();
    let mut errors = Vec::new();

    for func in graph.functions.iter() {
        if matches!(func.kind, FnKind::Extern) {
            continue;
        }
        let Some(parsed) = extract.parsed.get(&func.qref) else {
            continue;
        };
        // Only lower Complete functions.
        let Some(resolution) = infer_result.try_resolution(func.qref) else {
            continue;
        };

        // Build context_ids for this function: only contexts that this function references.
        let fn_refs = extract.fn_refs.get(&func.qref);
        let context_ids: FxHashMap<QualifiedRef, Ty> = match fn_refs {
            Some(refs) => refs
                .context_reads
                .iter()
                .chain(refs.context_writes.iter())
                .map(|&qref| {
                    let ty = infer_result
                        .context_type(&qref)
                        .cloned()
                        .unwrap_or(Ty::error());
                    (qref, ty)
                })
                .collect(),
            None => FxHashMap::default(),
        };

        // Clone the resolution for lowering (lower consumes it).
        let resolution_clone = resolution.clone();

        // Build function_ids from graph functions.
        let function_ids: FxHashMap<Astr, (QualifiedRef, Ty)> = graph
            .functions
            .iter()
            .filter_map(|f| {
                let ty = match &f.constraint.output {
                    Constraint::Exact(ty) => ty.clone(),
                    _ => Ty::error(),
                };
                Some((f.qref.name, (f.qref, ty)))
            })
            .collect();

        // Build QualifiedRef → Ty map for SSA pass (callee effect resolution).
        let fn_type_map: FxHashMap<QualifiedRef, Ty> = function_ids
            .values()
            .map(|(qref, ty)| (*qref, ty.clone()))
            .collect();

        let lowerer = crate::lower::Lowerer::new(
            interner,
            resolution_clone.type_map,
            resolution_clone.coercion_map,
            Freeze::new(context_ids),
            Freeze::new(function_ids),
        );
        let (mut module, hints) = match parsed {
            ParsedSource::Script(script) => lowerer.lower_script(script),
            ParsedSource::Template(template) => lowerer.lower_template(template),
        };

        // SSA pass: promote context/local variables to SSA form.
        crate::optimize::ssa_pass::run(&mut module.main, &fn_type_map);
        for closure in module.closures.values_mut() {
            crate::optimize::ssa_pass::run(closure, &fn_type_map);
        }

        let validation_errors = crate::validate::validate(&module, &fn_type_map);
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
                modules.insert(func.qref, (module, hints));
            }
            Err(errs) => {
                errors.push(LowerError {
                    fn_id: func.qref,
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
    use crate::graph::extract;
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
                qref: QualifiedRef::root(interner.intern(name)),
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        let fn_qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: fn_qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                },
            }]),
            contexts: Freeze::new(contexts),
        }
    }

    fn first_fn_ref(graph: &CompilationGraph) -> QualifiedRef {
        graph.functions[0].qref
    }

    // -- Completeness: valid programs lower to MIR --

    #[test]
    fn lower_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "1 + 2", &[]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_with_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let graph = make_graph_with_ctx(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);

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
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
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
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
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
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
        // Iterate must generate ListStep.
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(i.kind, InstKind::ListStep { .. })),
            "iterate should generate ListStep"
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
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        let result = lower(&i, &graph, &ext, &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
        // Nested iterate: two ListStep instructions.
        let iter_count = module
            .main
            .insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::ListStep { .. }))
            .count();
        assert_eq!(iter_count, 2, "nested iterate should have 2 ListStep");
    }

    // -- Soundness: type errors don't produce modules --

    #[test]
    fn lower_type_error_no_module() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
        // Infer should produce Incomplete for this function (type mismatch).
        // Lower should produce no module for this unit.
        let result = lower(&i, &graph, &ext, &inf);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_none());
    }
}
