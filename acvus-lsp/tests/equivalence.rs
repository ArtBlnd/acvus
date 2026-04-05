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
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Template(template)),
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
    let mut errs: Vec<String> = Vec::new();
    // Collect infer errors.
    for (_, fn_errs) in inf.errors() {
        for e in fn_errs {
            errs.push(format!("{}", e.display(interner)));
        }
    }
    // Collect lower errors.
    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for le in &result.errors {
        for e in &le.errors {
            errs.push(format!("{}", e.display(interner)));
        }
    }
    errs.sort();
    errs
}

/// Register standard library functions into an LspSession.
fn register_std(session: &mut LspSession) {
    let interner = session.interner().clone();
    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(&interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(&interner);
        for func in registered.functions {
            session.graph_mut().add_function(func);
        }
    }
}

/// Compile via LspSession, return error messages (sorted).
fn lsp_errors(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> Vec<String> {
    let mut session = LspSession::new(interner);
    register_std(&mut session);
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
    register_std(&mut session);
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
    register_std(&mut session);
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

// ── Completion tests ───────────────────────────────────────────────

#[test]
fn completion_context_trigger() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, Constraint::Exact(Ty::String));
    session.add_context("count", None, Constraint::Exact(Ty::Int));

    let doc = session.open("test", "{{ @n }}", None);
    // Cursor after "@n" → context trigger with prefix "n"
    let items = session.completions(doc, 5); // "{{ @n" = 5 chars
    assert!(!items.is_empty(), "should get context completions");
    assert!(
        items.iter().any(|c| c.label == "@name"),
        "should suggest @name, got: {:?}", items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
    assert!(
        !items.iter().any(|c| c.label == "@count"),
        "@count should not match prefix 'n'"
    );
}

#[test]
fn completion_pipe_trigger() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, Constraint::Exact(Ty::String));

    // Add a helper function so visible_functions returns something.
    let helper_qref = QualifiedRef::root(i.intern("helper"));
    session.graph_mut().add_function(Function {
        qref: helper_qref,
        kind: FnKind::Local(ParsedAst::Template(
            acvus_ast::parse(&i, "hello").unwrap(),
        )),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
            hint: None,
        },
    });

    let doc = session.open("test", "{{ @name | helper }}", None);
    // Cursor after "| " → pipe trigger (user is about to type after |)
    let items = session.completions(doc, 10); // "{{ @name |" = 10 chars
    assert!(!items.is_empty(), "should get pipe completions (functions)");
    assert!(
        items.iter().any(|c| c.label == "helper"),
        "should suggest helper, got: {:?}", items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_keyword_trigger() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);

    let doc = session.open("test", "{{ tr }}", None);
    // Cursor after "tr" → keyword trigger
    let items = session.completions(doc, 5); // "{{ tr" = 5 chars
    assert!(
        items.iter().any(|c| c.label == "true"),
        "should suggest 'true', got: {:?}", items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_empty_after_close() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, Constraint::Exact(Ty::String));

    let doc = session.open("test", "{{ @n }}", None);
    session.close(doc);
    let items = session.completions(doc, 5);
    assert!(items.is_empty(), "closed doc should return no completions");
}

#[test]
fn completion_updates_with_source() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, Constraint::Exact(Ty::String));
    session.add_context("age", None, Constraint::Exact(Ty::Int));

    let doc = session.open("test", "{{ @n }}", None);
    let items = session.completions(doc, 5);
    assert!(items.iter().any(|c| c.label == "@name"), "should match @name");

    // Update source to "@a"
    session.update_source(doc, "{{ @a }}");
    let items = session.completions(doc, 5);
    assert!(
        items.iter().any(|c| c.label == "@age"),
        "after update should match @age, got: {:?}", items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}
