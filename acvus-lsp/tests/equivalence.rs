//! Equivalence tests: LSP session must produce the same diagnostics
//! as the batch compilation pipeline.

use acvus_extern::{Externs, TypesOnly};
use acvus_lsp::LspSession;
use acvus_mir::graph::types::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, lift_to_poly};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile via batch pipeline, return error messages (sorted).
fn batch_errors(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> Vec<String> {
    let contexts: Vec<Context> = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_to_poly(ty),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let mut pb = PolyBuilder::new();
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Template(template)),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }];
    let Externs {
        functions: std_fns,
        types: type_registry,
        handlers: _,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), interner)
        .expect("standard registries combine");
    functions.extend(std_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
    );
    let mut errs: Vec<String> = Vec::new();
    // Collect infer errors.
    for (_, fn_errs) in inf.errors() {
        for e in fn_errs {
            errs.push(format!("{}", e.display(interner)));
        }
    }
    // Collect lower errors.
    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);
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
    let externs = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), &interner)
        .expect("standard registries combine");
    for func in externs.functions {
        session.graph_mut().add_function(func);
    }
}

/// Compile via LspSession, return error messages (sorted).
fn lsp_errors(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> Vec<String> {
    let mut session = LspSession::new(interner);
    register_std(&mut session);
    for (name, ty) in ctx {
        session.add_context(name, None, lift_to_poly(ty));
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
    let ctx = [("name", Ty::String), ("count", Ty::I64)];
    let source = "{{ @name }} and {{ @count.to_string() }}";
    let batch = batch_errors(&i, source, &ctx);
    let lsp = lsp_errors(&i, source, &ctx);
    assert_eq!(batch, lsp);
    assert!(lsp.is_empty());
}

#[test]
fn type_error_equivalence() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::I64)];
    // String + Int is a type error.
    let source = "% let out = @name + @count\n{{ out.to_string() }}";
    let batch = batch_errors(&i, source, &ctx);
    let lsp = lsp_errors(&i, source, &ctx);
    assert_eq!(batch, lsp, "batch and lsp should agree on type errors");
}

/// Lowering refuses this one, not the type checker: the session reaches the
/// stage the batch path reaches, so the two report the same refusal.
#[test]
fn definite_assignment_equivalence() {
    let i = Interner::new();
    let source = "% let x = { a: 1, }\n% if $flag\n% x.b = \"yes\"\n% end\n{{ x.b }}\n";
    let batch = batch_errors(&i, source, &[]);
    assert_eq!(batch.len(), 1, "one refusal from lowering: {batch:?}");
    assert_eq!(batch, lsp_errors(&i, source, &[]));
}

#[test]
fn incremental_update_fixes_error() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    register_std(&mut session);
    session.add_context("x", None, lift_to_poly(&Ty::I64));

    // Start with emit type error: Int not emittable in template.
    let doc = session.open("test", "{{ @x }}", None);
    let errs = session.diagnostics(doc);
    assert!(
        !errs.is_empty(),
        "should have emit error for Int in template"
    );

    // Fix: call to_string on it.
    session.update_source(doc, "{{ @x.to_string() }}");
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
    session.add_context("name", None, lift_to_poly(&Ty::String));

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
    session.add_context("value", Some(ns), lift_to_poly(&Ty::I64));
    session.add_context("global", None, lift_to_poly(&Ty::String));

    // Root function sees @global.
    let doc_root = session.open("root_fn", "{{ @global }}", None);
    assert!(
        session.diagnostics(doc_root).is_empty(),
        "root should see @global"
    );
}

// -- Completion tests -----------------------------------------------

#[test]
fn completion_context_trigger() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, lift_to_poly(&Ty::String));
    session.add_context("count", None, lift_to_poly(&Ty::I64));

    let doc = session.open("test", "{{ @n }}", None);
    // Cursor after "@n" -> context trigger with prefix "n"
    let items = session.completions(doc, 5); // "{{ @n" = 5 chars
    assert!(!items.is_empty(), "should get context completions");
    assert!(
        items.iter().any(|c| c.label == "@name"),
        "should suggest @name, got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
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
    session.add_context("name", None, lift_to_poly(&Ty::String));

    // Add a helper function so visible_functions returns something.
    let helper_qref = QualifiedRef::root(i.intern("helper"));
    let mut pb = PolyBuilder::new();
    session.graph_mut().add_function(Function {
        qref: helper_qref,
        kind: FnKind::Local(ParsedAst::Template(acvus_ast::parse(&i, "hello").unwrap())),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    });

    let doc = session.open("test", "{{ @name | helper }}", None);
    // Cursor after "| " -> pipe trigger (user is about to type after |)
    let items = session.completions(doc, 10); // "{{ @name |" = 10 chars
    assert!(!items.is_empty(), "should get pipe completions (functions)");
    assert!(
        items.iter().any(|c| c.label == "helper"),
        "should suggest helper, got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_keyword_trigger() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);

    let doc = session.open("test", "{{ tr }}", None);
    // Cursor after "tr" -> keyword trigger
    let items = session.completions(doc, 5); // "{{ tr" = 5 chars
    assert!(
        items.iter().any(|c| c.label == "true"),
        "should suggest 'true', got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_empty_after_close() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, lift_to_poly(&Ty::String));

    let doc = session.open("test", "{{ @n }}", None);
    session.close(doc);
    let items = session.completions(doc, 5);
    assert!(items.is_empty(), "closed doc should return no completions");
}

#[test]
fn completion_updates_with_source() {
    let i = Interner::new();
    let mut session = LspSession::new(&i);
    session.add_context("name", None, lift_to_poly(&Ty::String));
    session.add_context("age", None, lift_to_poly(&Ty::I64));

    let doc = session.open("test", "{{ @n }}", None);
    let items = session.completions(doc, 5);
    assert!(
        items.iter().any(|c| c.label == "@name"),
        "should match @name"
    );

    // Update source to "@a"
    session.update_source(doc, "{{ @a }}");
    let items = session.completions(doc, 5);
    assert!(
        items.iter().any(|c| c.label == "@age"),
        "after update should match @age, got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

/// RFC-0071 rule 5, at the surface the editor reads.
mod required_inputs {
    use acvus_lsp::LspSession;
    use acvus_utils::Interner;

    fn text(value: &str) -> acvus_ast::Literal {
        acvus_ast::Literal::String(value.to_string())
    }

    fn shown(session: &acvus_lsp::LspSession, id: acvus_lsp::DocId) -> Vec<String> {
        let interner = session.graph().interner().clone();
        let mut shown: Vec<String> = session
            .context_info(id)
            .iter()
            .map(|input| {
                format!(
                    "${}: {}",
                    interner.resolve(input.name.name),
                    input.ty.display(&interner)
                )
            })
            .collect();
        shown.sort();
        shown
    }

    const BY_MODE: &str = "\
% if $mode == \"review\"
Review {{ $rules }}
% else
Explain {{ $examples }}
% end
";

    const BOTH_ARMS_READ_WHO: &str = "\
% if $mode == \"review\"
Review for {{ $who }}
% else
Explain for {{ $who }}
% end
";

    #[test]
    fn a_document_shows_the_inputs_it_reads() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner);
        let doc = session.open("test", BY_MODE, None);
        assert_eq!(
            shown(&session, doc),
            vec![
                "$examples: String".to_string(),
                "$mode: String".to_string(),
                "$rules: String".to_string(),
            ]
        );
    }

    #[test]
    fn a_binding_narrows_to_what_the_surviving_arm_reads() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner);
        let doc = session.open("test", BY_MODE, None);

        session.bind_input("mode", text("review"));
        assert_eq!(shown(&session, doc), vec!["$rules: String".to_string()]);

        session.bind_input("mode", text("explain"));
        assert_eq!(shown(&session, doc), vec!["$examples: String".to_string()]);

        session.unbind_input("mode");
        assert_eq!(
            shown(&session, doc),
            vec![
                "$examples: String".to_string(),
                "$mode: String".to_string(),
                "$rules: String".to_string(),
            ]
        );
    }

    #[test]
    fn a_name_both_arms_read_survives_every_binding() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner);
        let doc = session.open("test", BY_MODE, None);
        session.update_source(doc, BOTH_ARMS_READ_WHO);

        assert_eq!(
            shown(&session, doc),
            vec!["$mode: String".to_string(), "$who: String".to_string()]
        );

        session.bind_input("mode", text("review"));
        assert_eq!(shown(&session, doc), vec!["$who: String".to_string()]);

        session.bind_input("mode", text("explain"));
        assert_eq!(shown(&session, doc), vec!["$who: String".to_string()]);
    }

    #[test]
    fn a_source_that_does_not_parse_is_one_diagnostic_and_recovers() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner);
        let doc = session.open("test", BY_MODE, None);

        session.update_source(doc, "% if\nbroken\n");
        assert_eq!(session.diagnostics(doc).len(), 1);
        assert_eq!(
            session.diagnostics(doc)[0].category,
            acvus_lsp::LspErrorCategory::Parse
        );
        assert!(shown(&session, doc).is_empty());

        session.update_source(doc, BY_MODE);
        assert!(session.diagnostics(doc).is_empty());
        assert_eq!(
            shown(&session, doc),
            vec![
                "$examples: String".to_string(),
                "$mode: String".to_string(),
                "$rules: String".to_string(),
            ]
        );
    }

    #[test]
    fn a_document_that_never_parsed_is_one_diagnostic() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner);
        let doc = session.open("test", "% if\nbroken\n", None);
        assert_eq!(session.diagnostics(doc).len(), 1);
        assert!(shown(&session, doc).is_empty());
        assert!(session.required_inputs(doc).is_empty());
    }
}
