//! Phase 3: Lower
//!
//! Takes InferResult (Complete outcomes) + cached ASTs and produces MirModule per function.
//! Reuses the existing MIR lowerer — this is just the orchestration layer.

use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::ir::MirModule;

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
    pub modules: FxHashMap<QualifiedRef, MirModule>,
    pub errors: Vec<LowerError>,
}

impl LowerResult {
    pub fn module(&self, id: QualifiedRef) -> Option<&MirModule> {
        self.modules.get(&id)
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// ── Lowering ────────────────────────────────────────────────────────

/// Run Phase 3: lower each Complete function to MIR.
///
/// `policies`: external constraints on contexts (volatile, read_only, etc.).
pub fn lower(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    infer_result: &InferResult,
    policies: &FxHashMap<QualifiedRef, ContextPolicy>,
) -> LowerResult {
    let mut modules = FxHashMap::default();
    let errors = Vec::new();

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

        let lowerer = crate::lower::Lowerer::new(
            interner,
            resolution,
            policies.clone(),
        );
        let module = match parsed {
            ParsedSource::Script(script) => lowerer.lower_script(script),
            ParsedSource::Template(template) => lowerer.lower_template(template),
        };

        // SSA + validate are handled by the optimize pipeline (graph/optimize.rs).
        // Lower outputs pre-SSA MIR.
        modules.insert(func.qref, module);
    }

    LowerResult { modules, errors }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::extract, ty::{Ty, TyTerm, PolyBuilder}};
    use crate::ir::InstKind;
    use acvus_utils::{Freeze, Interner};

    fn make_graph_with_ctx(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: crate::ty::lift_to_poly(ty),
            })
            .collect();
        let fn_qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: fn_qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: pb.fresh_effect_var(),
                    hint: None,
                },
                effect_constraint: None,
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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_with_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());

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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());

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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());
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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());
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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());
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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());
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
        let inf = crate::graph::infer::infer(
            &i,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );
        // Infer should produce Incomplete for this function (type mismatch).
        // Lower should produce no module for this unit.
        let result = lower(&i, &graph, &ext, &inf, &FxHashMap::default());
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_none());
    }
}
