//! Phase 3: Lower
//!
//! Takes InferResult (Complete outcomes) + cached ASTs and produces MirModule per function.
//! Reuses the existing MIR lowerer - this is just the orchestration layer.

use std::collections::BTreeSet;

use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::ir::{Callee, InstKind, MirBody, MirModule};
use crate::ty::Ty;

use super::extract::{ExtractResult, ParsedSource};
use super::infer::{FnInferOutcome, InferResult};
use super::types::*;

// -- Phase 3 output --------------------------------------------------

/// The parsed source of each function to lower, borrowed from the cache that
/// holds it. Lowering only reads the AST, so neither caller clones one.
pub type ParsedView<'a> = FxHashMap<QualifiedRef, &'a ParsedSource>;

impl ExtractResult {
    pub fn view(&self) -> ParsedView<'_> {
        self.parsed.iter().map(|(&qref, p)| (qref, p)).collect()
    }
}

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

// -- Lowering --------------------------------------------------------

/// Run Phase 3: lower each Complete function to MIR.
///
pub fn lower(
    interner: &Interner,
    graph: &CompilationGraph,
    parsed: &ParsedView<'_>,
    infer_result: &InferResult,
) -> LowerResult {
    let mut modules = FxHashMap::default();
    let mut errors = Vec::new();

    for func in graph.functions.iter() {
        let Some(source) = parsed.get(&func.qref) else {
            continue;
        };
        let Some(outcome) = infer_result.outcomes.get(&func.qref) else {
            continue;
        };
        let Some(lowered) = lower_one(interner, source, outcome, &graph.bindings) else {
            continue;
        };
        if !lowered.errors.is_empty() {
            errors.push(LowerError {
                fn_id: func.qref,
                errors: lowered.errors,
            });
        }
        modules.insert(func.qref, lowered.module);
    }
    close_fetched_first(&mut modules);

    LowerResult { modules, errors }
}

/// RFC-0025 rule 2 across calls. A module's own `fetched_first` holds the
/// contexts `main` fetches at entry. A call adds what its callee may fetch
/// first, less the contexts `main` commits around it: that `Commit` takes
/// `main`'s variable, so `main` fetched it or assigned it before the call,
/// and a variable moved out there is refused (RFC-0025 rule 3). A callee with
/// a body adds its own `fetched_first`; any other callee, and every function
/// value passed, adds what its summary touches; a call with no summary
/// touches every context some body names (RFC-0025 rule 8).
pub fn close_fetched_first(modules: &mut FxHashMap<QualifiedRef, MirModule>) {
    let contexts: BTreeSet<QualifiedRef> = modules
        .values()
        .flat_map(|module| std::iter::once(&module.main).chain(module.closures.values()))
        .flat_map(|body| committed(body))
        .collect();
    let mut fetched_first: FxHashMap<QualifiedRef, BTreeSet<QualifiedRef>> = modules
        .iter()
        .map(|(qref, module)| (*qref, module.fetched_first.iter().copied().collect()))
        .collect();
    loop {
        let mut grown = false;
        for (qref, module) in modules.iter() {
            let calls = calls_fetch_first(&module.main, &fetched_first, &contexts);
            let own = fetched_first
                .get_mut(qref)
                .expect("every module has an entry");
            for context in calls {
                grown |= own.insert(context);
            }
        }
        if !grown {
            break;
        }
    }
    for (qref, module) in modules.iter_mut() {
        let added: Vec<QualifiedRef> = fetched_first[qref]
            .iter()
            .filter(|context| !module.fetched_first.contains(context))
            .copied()
            .collect();
        module.fetched_first.extend(added);
    }
}

/// The contexts `body` names: each one it commits at its exits.
fn committed(body: &MirBody) -> impl Iterator<Item = QualifiedRef> + '_ {
    body.insts.iter().filter_map(|inst| match &inst.kind {
        InstKind::Commit { context, .. } => Some(*context),
        _ => None,
    })
}

/// What the calls of `body` may fetch first that `body` does not commit
/// around them.
fn calls_fetch_first(
    body: &MirBody,
    fetched_first: &FxHashMap<QualifiedRef, BTreeSet<QualifiedRef>>,
    contexts: &BTreeSet<QualifiedRef>,
) -> BTreeSet<QualifiedRef> {
    let named: BTreeSet<QualifiedRef> = committed(body).collect();
    let summary_touch = |ty: &Ty| -> BTreeSet<QualifiedRef> {
        match ty.effect() {
            Some(effect) => effect.reads.union(&effect.writes).copied().collect(),
            None => contexts.clone(),
        }
    };
    let mut out = BTreeSet::new();
    for inst in &body.insts {
        let (InstKind::FunctionCall {
            callee,
            callee_ty,
            args,
            ..
        }
        | InstKind::Spawn {
            callee,
            callee_ty,
            args,
            ..
        }) = &inst.kind
        else {
            continue;
        };
        let passed: BTreeSet<QualifiedRef> = args
            .iter()
            .map(|arg| {
                body.val_types
                    .get(arg)
                    .expect("every register of a lowered body has a type")
            })
            .filter(|ty| matches!(ty, Ty::Fn { .. }))
            .flat_map(|ty| summary_touch(ty))
            .collect();
        let bracketed: BTreeSet<QualifiedRef> = summary_touch(callee_ty)
            .union(&passed)
            .copied()
            .filter(|context| named.contains(context))
            .collect();
        let called: BTreeSet<QualifiedRef> = match callee {
            Callee::Direct(qref) => match fetched_first.get(qref) {
                Some(own) => own.clone(),
                None => summary_touch(callee_ty),
            },
            Callee::Extern { .. } => BTreeSet::new(),
            Callee::Indirect(_) => summary_touch(callee_ty),
        };
        out.extend(
            called
                .union(&passed)
                .filter(|context| !bracketed.contains(context)),
        );
    }
    out
}

/// A lowered body and what lowering refused in it. The module is pre-SSA:
/// `graph::optimize` runs SSA and every validation that reads the optimized
/// shape.
pub struct Lowered {
    pub module: MirModule,
    pub errors: Vec<MirError>,
}

/// `None` for a function inference left `Incomplete`, which has no resolution
/// to lower against.
pub fn lower_one(
    interner: &Interner,
    parsed: &ParsedSource,
    outcome: &FnInferOutcome,
    bindings: &Bindings,
) -> Option<Lowered> {
    let resolution = outcome.resolution()?;
    let ret = match &outcome.meta().ty {
        crate::ty::Ty::Fn { ret, .. } => (**ret).clone(),
        other => other.clone(),
    };
    let lowerer = crate::lower::Lowerer::new(interner, resolution, ret);
    let mut module = match parsed {
        ParsedSource::Script(script) => lowerer.lower_script(script),
        ParsedSource::Template(template) => lowerer.lower_template(template),
        ParsedSource::Recovered(_) => return None,
    };

    let mut errors = super::bind::substitute(interner, &mut module.main, bindings);

    // Definite assignment reads the pre-SSA shape the source wrote, so it
    // runs here and not in `validate`, which sees the optimized body.
    for body in std::iter::once(&module.main).chain(module.closures.values()) {
        let cfg = crate::cfg::promote(body.clone());
        errors.extend(crate::validate::init_check::refusals(interner, &cfg));
    }

    Some(Lowered { module, errors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::ObjectTy;
    use crate::{
        graph::extract,
        ty::{PolyBuilder, Ty, TyTerm},
    };
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
                ty: crate::ty::lift_declaration(ty, &mut pb),
            })
            .collect();
        let fn_qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: fn_qref,
                kind: FnKind::Local(
                    ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse")),
                    crate::graph::Inputs::FromReads,
                ),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            }]),
            contexts: Freeze::new(contexts),
            types: Freeze::default(),
            bindings: Bindings::default(),
            entries: Vec::new(),
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
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = lower(&i, &graph, &ext.view(), &inf);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_with_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::I64)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = lower(&i, &graph, &ext.view(), &inf);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_some());
    }

    #[test]
    fn lower_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("name"),
            Ty::String,
        )])));
        let graph = make_graph_with_ctx(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = lower(&i, &graph, &ext.view(), &inf);

        assert!(!result.has_errors(), "errors: {:?}", result.errors);
    }

    // -- Script: `if let` and iterate --

    #[test]
    fn lower_script_irrefutable_if_let() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "if let x = @data { @out = x + 1; }; @out",
            &[("data", Ty::I64), ("out", Ty::I64)],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = lower(&i, &graph, &ext.view(), &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
        // An irrefutable pattern has no test and no branch.
        assert!(
            !module
                .main
                .insts
                .iter()
                .any(|i| crate::ir::two_way(&i.kind).is_some()),
            "an irrefutable `if let` should not generate a two-way branch"
        );
    }

    #[test]
    fn lower_script_refutable_if_let() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "if let 42 = @val { @out = 1; }; @out",
            &[("val", Ty::I64), ("out", Ty::I64)],
        );
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = lower(&i, &graph, &ext.view(), &inf);
        assert!(!result.has_errors(), "errors: {:?}", result.errors);
        let module = result.module(first_fn_ref(&graph)).unwrap();
        // A refutable pattern is a test and a branch.
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| crate::ir::two_way(&i.kind).is_some()),
            "a refutable `if let` should generate a two-way branch"
        );
    }

    // -- Soundness: type errors don't produce modules --

    #[test]
    fn lower_type_error_no_module() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        // Infer should produce Incomplete for this function (type mismatch).
        // Lower should produce no module for this unit.
        let result = lower(&i, &graph, &ext.view(), &inf);
        let uid = first_fn_ref(&graph);
        assert!(result.module(uid).is_none());
    }
}
