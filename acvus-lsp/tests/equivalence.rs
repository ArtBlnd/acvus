//! Equivalence tests: LSP session must produce the same diagnostics
//! as the batch compilation pipeline.

use acvus_lsp::LspSession;
use acvus_mir::graph::types::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::Ty;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile via batch pipeline, return error messages (sorted).
fn batch_errors(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> Vec<String> {
    let contexts: Vec<Context> = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let graph = CompilationGraph {
        functions: Freeze::new(vec![Function {
            qref: test_qref,
            kind: FnKind::Local(ParsedAst::Template(template)),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        }]),
        contexts: Freeze::new(contexts),
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default(), Freeze::default());
    let mut errs: Vec<String> = Vec::new();
    let result = graph_lower::lower(interner, &graph, &ext, &inf);
    for le in &result.errors {
        for e in &le.errors {
            errs.push(format!("{}", e.display(interner)));
        }
    }
    errs.sort();
    errs
}

/// Compile via LspSession, return error messages (sorted).
fn lsp_errors(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> Vec<String> {
    let mut session = LspSession::new(interner);
    for (name, ty) in ctx {
        session.add_context(name, None, Constraint::Exact(ty.clone()));
    }
    let doc = session.open("test", source, None);
    let mut errs: Vec<String> = session
        .diagnostics(doc)
        .into_iter()
        .map(|e| e.message)
        .collect();
    errs.sort();
    errs
}

#[test]
fn no_errors_simple_template() {
    let i = Interner::new();
    let ctx = [("name", Ty::String)];
    let source = "hello {{ @name }}";
    assert_eq!(batch_errors(&i, source, &ctx), lsp_errors(&i, source, &ctx));
    assert!(lsp_errors(&i, source, &ctx).is_empty());
}

#[test]
fn valid_multi_context_equivalence() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::Int)];
    let source = "{{ @name }} and {{ @count | to_string }}";
    let batch = batch_errors(&i, source, &ctx);
    let lsp = lsp_errors(&i, source, &ctx);
    assert_eq!(batch, lsp);
    assert!(lsp.is_empty());
}

#[test]
fn type_error_equivalence() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::Int)];
    // String + Int is a type error.
    let source = "{{ @name + @count | to_string }}";
    let batch = batch_errors(&i, source, &ctx);
    let lsp = lsp_errors(&i, source, &ctx);
    assert_eq!(batch, lsp, "batch and lsp should agree on type errors");
}

#[test]
fn incremental_update_fixes_error() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("x", None, Constraint::Exact(Ty::Int));

    // Start with emit type error: Int not emittable in template.
    let doc = session.open("test", "{{ @x }}", None);
    let errs = session.diagnostics(doc);
    assert!(
        !errs.is_empty(),
        "should have emit error for Int in template"
    );

    // Fix: pipe to_string.
    session.update_source(doc, "{{ @x | to_string }}");
    let errs = session.diagnostics(doc);
    assert!(
        errs.is_empty(),
        "errors should be gone after fix, got: {:?}",
        errs
    );
}

#[test]
fn incremental_update_introduces_error() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, Constraint::Exact(Ty::String));

    // Start correct.
    let doc = session.open("test", "hello {{ @name }}", None);
    assert!(session.diagnostics(doc).is_empty());

    // Break it: unknown builtin.
    session.update_source(doc, "hello {{ @name | nonexistent }}");
    assert!(!session.diagnostics(doc).is_empty(), "should detect error");
}

#[test]
fn namespace_context_isolation() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);

    let ns = session.add_namespace("node_a");
    session.add_context("value", Some(ns), Constraint::Exact(Ty::Int));
    session.add_context("global", None, Constraint::Exact(Ty::String));


    // Root function sees @global.
    let doc_root = session.open("root_fn", "{{ @global }}", None);
    assert!(
        session.diagnostics(doc_root).is_empty(),
        "root should see @global"
    );
}
