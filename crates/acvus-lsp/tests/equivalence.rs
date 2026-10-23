//! Equivalence tests: LSP session must produce the same diagnostics
//! as the batch compilation pipeline.

use acvus_extern::{Externs, TypesOnly};
use acvus_lsp::{CompletionItem, CompletionKind, DocId, Document, LspErrorKind, LspSession, Mode};
use acvus_mir::graph::types::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, TypeRegistry, lift_to_poly};
use acvus_utils::{Freeze, Interner};

fn completed(session: &LspSession, doc: DocId, cursor: usize) -> Vec<CompletionItem> {
    session
        .completions(doc, cursor)
        .expect("the cursor is in code of an open document")
        .items
}

fn root_contexts(interner: &Interner, ctx: &[(&str, Ty)]) -> Vec<Context> {
    ctx.iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_to_poly(ty),
            init: None,
        })
        .collect()
}

fn environment(
    contexts: Vec<Context>,
    functions: Vec<Function>,
    types: TypeRegistry,
    bindings: Bindings,
) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings,
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    }
}

fn bare(contexts: Vec<Context>) -> CompilationGraph {
    environment(
        contexts,
        vec![],
        TypeRegistry::default(),
        Bindings::default(),
    )
}

fn with_std(interner: &Interner, contexts: Vec<Context>) -> CompilationGraph {
    let Externs {
        functions, types, ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), interner)
        .expect("standard registries combine");
    environment(contexts, functions, types, Bindings::default())
}

fn template_document(interner: &Interner, name: &str) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode: Mode::Template,
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
        inputs: acvus_mir::graph::Inputs::FromReads,
    }
}

fn template(interner: &Interner, name: &str, source: &str) -> Function {
    let Document {
        qref, ty, inputs, ..
    } = template_document(interner, name);
    Function {
        qref,
        kind: FnKind::Local(
            ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse failed")),
            inputs,
        ),
        ty,
    }
}

/// Compile via batch pipeline, return error messages (sorted).
fn batch_errors(interner: &Interner, environment: &CompilationGraph, source: &str) -> Vec<String> {
    let Parsed { ast, errors } = Parsed::template(acvus_ast::parse(interner, source));
    let Document {
        qref, ty, inputs, ..
    } = template_document(interner, "test");
    let functions: Vec<Function> = environment
        .functions
        .iter()
        .cloned()
        .chain(std::iter::once(Function {
            qref,
            kind: FnKind::Local(ast, inputs),
            ty,
        }))
        .collect();
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        ..environment.clone()
    };
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);
    let mut errs: Vec<String> = errors.iter().map(|e| e.kind.to_string()).collect();
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

/// Compile via LspSession, return error messages (sorted).
fn lsp_errors(interner: &Interner, environment: &CompilationGraph, source: &str) -> Vec<String> {
    let mut session = LspSession::new(interner, environment.clone());
    let doc = session
        .open(template_document(interner, "test"), source)
        .expect("the session opens no other document");
    let mut errs: Vec<String> = session
        .diagnostics(doc)
        .into_iter()
        .map(|e| e.message)
        .collect();
    errs.sort();
    errs
}

/// What parsed is checked: `a + "x"` is refused beside the two parse
/// errors. (An integer tag is no refusal: it appends through `core::display`,
/// RFC-0071 rule 3.)
#[test]
fn a_broken_template_reports_every_parse_error_and_what_parsed() {
    let i = Interner::new();
    let source = "% let a = 1\n% let = 2\n{{ a + \"x\" }}\n{{ f( }}\n";
    let env = with_std(&i, root_contexts(&i, &[]));
    let mut session = LspSession::new(&i, env.clone());
    let doc = session
        .open(template_document(&i, "test"), source)
        .expect("the session opens no other document");
    let kinds: Vec<LspErrorKind> = session
        .diagnostics(doc)
        .iter()
        .map(|error| error.kind)
        .collect();
    assert!(
        matches!(
            kinds.as_slice(),
            [
                LspErrorKind::Parse(_),
                LspErrorKind::Parse(_),
                LspErrorKind::Type(_)
            ]
        ),
        "{:?}",
        session.diagnostics(doc)
    );
    assert_eq!(batch_errors(&i, &env, source), lsp_errors(&i, &env, source));
}

#[test]
fn broken_sources_report_the_same_in_both_paths() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::I64)];
    let env = with_std(&i, root_contexts(&i, &ctx));
    for source in [
        "{{ @name }\n",
        "% for x in [1, 2]\n{{ x.to_string() }}\n",
        "% if @count > 1\nbig\n% else\nsmall\n% else\n% end\n",
        "{{ @name # }}\n{{ @count }}\n",
        "% let out = @name + @count\n% let = 1\n{{ out }}\n",
    ] {
        let batch = batch_errors(&i, &env, source);
        assert!(!batch.is_empty(), "{source:?}");
        assert_eq!(batch, lsp_errors(&i, &env, source), "{source:?}");
    }
}

#[test]
fn no_errors_simple_template() {
    let i = Interner::new();
    let ctx = [("name", Ty::String)];
    let source = "hello {{ @name }}";
    let env = with_std(&i, root_contexts(&i, &ctx));
    assert_eq!(batch_errors(&i, &env, source), lsp_errors(&i, &env, source));
    assert!(lsp_errors(&i, &env, source).is_empty());
}

#[test]
fn valid_multi_context_equivalence() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::I64)];
    let source = "{{ @name }} and {{ @count.to_string() }}";
    let env = with_std(&i, root_contexts(&i, &ctx));
    let batch = batch_errors(&i, &env, source);
    let lsp = lsp_errors(&i, &env, source);
    assert_eq!(batch, lsp);
    assert!(lsp.is_empty());
}

#[test]
fn type_error_equivalence() {
    let i = Interner::new();
    let ctx = [("name", Ty::String), ("count", Ty::I64)];
    // String + Int is a type error.
    let source = "% let out = @name + @count\n{{ out.to_string() }}";
    let env = with_std(&i, root_contexts(&i, &ctx));
    let batch = batch_errors(&i, &env, source);
    let lsp = lsp_errors(&i, &env, source);
    assert_eq!(batch, lsp, "batch and lsp should agree on type errors");
}

/// Lowering refuses this one, not the type checker: the session reaches the
/// stage the batch path reaches, so the two report the same refusal.
#[test]
fn definite_assignment_equivalence() {
    let i = Interner::new();
    let source = "% let x = { a: 1, }\n% if $flag\n% x.b = \"yes\"\n% end\n{{ x.b }}\n";
    let env = with_std(&i, root_contexts(&i, &[]));
    let batch = batch_errors(&i, &env, source);
    assert_eq!(batch.len(), 1, "one refusal from lowering: {batch:?}");
    assert_eq!(batch, lsp_errors(&i, &env, source));
}

#[test]
fn extension_type_equivalence() {
    let i = Interner::new();
    let source = "% let q = deque()\n% q.push_back(1)\n{{ q.len().to_string() }}\n";
    let env = with_std(&i, vec![]);
    let batch = batch_errors(&i, &env, source);
    assert_eq!(batch, lsp_errors(&i, &env, source));
    assert!(batch.is_empty(), "{batch:?}");
}

#[test]
fn a_new_environment_rechecks_open_documents() {
    let i = Interner::new();
    let mut session = LspSession::new(&i, bare(root_contexts(&i, &[("x", Ty::String)])));
    let doc = session
        .open(template_document(&i, "test"), "{{ @x }}")
        .expect("the session opens no other document");
    assert!(session.diagnostics(doc).is_empty());

    session.set_environment(bare(root_contexts(&i, &[("x", Ty::I64)])));
    assert!(
        !session.diagnostics(doc).is_empty(),
        "an Int is not emitted in a template"
    );

    session.set_environment(bare(root_contexts(&i, &[("x", Ty::String)])));
    assert!(session.diagnostics(doc).is_empty());
}

#[test]
fn an_open_document_replaces_the_environment_function_at_its_name() {
    let i = Interner::new();
    let qref = QualifiedRef::root(i.intern("test"));
    let env = environment(
        root_contexts(&i, &[("x", Ty::I64), ("y", Ty::String)]),
        vec![template(&i, "test", "{{ @x }}")],
        TypeRegistry::default(),
        Bindings::default(),
    );
    let mut session = LspSession::new(&i, env.clone());
    assert!(!session.graph().diagnostics(qref).is_empty());

    let doc = session
        .open(template_document(&i, "test"), "{{ @y }}")
        .expect("the session opens no other document");
    assert!(session.diagnostics(doc).is_empty());

    session.set_environment(env);
    assert!(session.diagnostics(doc).is_empty());
    assert!(session.graph().diagnostics(qref).is_empty());
}

#[test]
fn incremental_update_fixes_error() {
    let i = Interner::new();
    let mut session = LspSession::new(&i, with_std(&i, root_contexts(&i, &[("x", Ty::I64)])));

    // Start with an emit type error: an array has no `core::display`
    // instance, so a template cannot append it.
    let doc = session
        .open(template_document(&i, "test"), "{{ [@x] }}")
        .expect("the session opens no other document");
    let errs = session.diagnostics(doc);
    assert!(
        !errs.is_empty(),
        "should have emit error for an array in template"
    );

    // Fix: append the element, which has one.
    session.update_source(doc, "{{ @x }}");
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
    let mut session = LspSession::new(&i, with_std(&i, root_contexts(&i, &[("name", Ty::String)])));

    // Start correct.
    let doc = session
        .open(template_document(&i, "test"), "hello {{ @name }}")
        .expect("the session opens no other document");
    assert!(session.diagnostics(doc).is_empty());

    // Break it: unknown builtin.
    session.update_source(doc, "hello {{ @name | nonexistent }}");
    assert!(!session.diagnostics(doc).is_empty(), "should detect error");
}

#[test]
fn namespace_context_isolation() {
    let i = Interner::new();
    let mut contexts = root_contexts(&i, &[("global", Ty::String)]);
    contexts.push(Context {
        qref: QualifiedRef::qualified(i.intern("node_a"), i.intern("value")),
        ty: lift_to_poly(&Ty::I64),
        init: None,
    });
    let mut session = LspSession::new(&i, bare(contexts));

    // Root function sees @global.
    let doc_root = session
        .open(template_document(&i, "root_fn"), "{{ @global }}")
        .expect("the session opens no other document");
    assert!(
        session.diagnostics(doc_root).is_empty(),
        "root should see @global"
    );
}

// -- Completion tests -----------------------------------------------

#[test]
fn completion_offers_contexts_after_at() {
    let i = Interner::new();
    let mut session = LspSession::new(
        &i,
        bare(root_contexts(
            &i,
            &[("name", Ty::String), ("count", Ty::I64)],
        )),
    );

    let doc = session
        .open(template_document(&i, "test"), "{{ @n }}")
        .expect("the session opens no other document");
    let items = completed(&session, doc, "{{ @n".len());
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

/// A pipe stage is a call, so the functions are offered where one is
/// written.
#[test]
fn completion_offers_functions_at_a_pipe_stage() {
    let i = Interner::new();
    let mut session = LspSession::new(
        &i,
        environment(
            root_contexts(&i, &[("name", Ty::String)]),
            vec![template(&i, "helper", "hello")],
            TypeRegistry::default(),
            Bindings::default(),
        ),
    );

    let source = "{{ @name | he }}";
    let doc = session
        .open(template_document(&i, "test"), source)
        .expect("the session opens no other document");
    let items = completed(&session, doc, "{{ @name | he".len());
    assert!(
        items
            .iter()
            .any(|c| c.label == "helper" && c.kind == CompletionKind::Function),
        "should suggest helper, got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_offers_keywords_by_prefix() {
    let i = Interner::new();
    let mut session = LspSession::new(&i, bare(vec![]));

    let doc = session
        .open(template_document(&i, "test"), "{{ tr }}")
        .expect("the session opens no other document");
    let items = completed(&session, doc, "{{ tr".len());
    assert!(
        items
            .iter()
            .any(|c| c.label == "true" && c.kind == CompletionKind::Keyword),
        "should suggest 'true', got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

#[test]
fn completion_empty_after_close() {
    let i = Interner::new();
    let mut session = LspSession::new(&i, bare(root_contexts(&i, &[("name", Ty::String)])));

    let doc = session
        .open(template_document(&i, "test"), "{{ @n }}")
        .expect("the session opens no other document");
    session.close(doc);
    assert_eq!(
        session.completions(doc, "{{ @n".len()),
        None,
        "closed doc should return no completions"
    );
}

#[test]
fn completion_updates_with_source() {
    let i = Interner::new();
    let mut session = LspSession::new(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String), ("age", Ty::I64)])),
    );

    let doc = session
        .open(template_document(&i, "test"), "{{ @n }}")
        .expect("the session opens no other document");
    let items = completed(&session, doc, "{{ @n".len());
    assert!(
        items.iter().any(|c| c.label == "@name"),
        "should match @name"
    );

    session.update_source(doc, "{{ @a }}");
    let items = completed(&session, doc, "{{ @a".len());
    assert!(
        items.iter().any(|c| c.label == "@age"),
        "after update should match @age, got: {:?}",
        items.iter().map(|c| &c.label).collect::<Vec<_>>()
    );
}

/// RFC-0071 rule 5, at the surface the editor reads.
mod required_inputs {
    use acvus_lsp::LspSession;
    use acvus_mir::graph::{Bindings, BoundValue, CompilationGraph};
    use acvus_mir::ty::TypeRegistry;
    use acvus_utils::Interner;

    fn text(value: &str) -> BoundValue {
        BoundValue::String(value.to_string())
    }

    fn bound(interner: &Interner, name: &str, value: BoundValue) -> CompilationGraph {
        let mut bindings = Bindings::default();
        bindings
            .bind(interner.intern(name), value)
            .expect("text types on its own");
        super::environment(vec![], vec![], TypeRegistry::default(), bindings)
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
        let mut session = LspSession::new(&interner, super::bare(vec![]));
        let doc = session
            .open(super::template_document(&interner, "test"), BY_MODE)
            .expect("the session opens no other document");
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
        let mut session = LspSession::new(&interner, super::bare(vec![]));
        let doc = session
            .open(super::template_document(&interner, "test"), BY_MODE)
            .expect("the session opens no other document");

        session.set_environment(bound(&interner, "mode", text("review")));
        assert_eq!(shown(&session, doc), vec!["$rules: String".to_string()]);

        session.set_environment(bound(&interner, "mode", text("explain")));
        assert_eq!(shown(&session, doc), vec!["$examples: String".to_string()]);

        session.set_environment(super::bare(vec![]));
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
        let mut session = LspSession::new(&interner, super::bare(vec![]));
        let doc = session
            .open(super::template_document(&interner, "test"), BY_MODE)
            .expect("the session opens no other document");
        session.update_source(doc, BOTH_ARMS_READ_WHO);

        assert_eq!(
            shown(&session, doc),
            vec!["$mode: String".to_string(), "$who: String".to_string()]
        );

        session.set_environment(bound(&interner, "mode", text("review")));
        assert_eq!(shown(&session, doc), vec!["$who: String".to_string()]);

        session.set_environment(bound(&interner, "mode", text("explain")));
        assert_eq!(shown(&session, doc), vec!["$who: String".to_string()]);
    }

    #[test]
    fn a_source_that_does_not_parse_is_one_diagnostic_and_recovers() {
        let interner = Interner::new();
        let mut session = LspSession::new(&interner, super::bare(vec![]));
        let doc = session
            .open(super::template_document(&interner, "test"), BY_MODE)
            .expect("the session opens no other document");

        session.update_source(doc, "% if\nbroken\n");
        assert_eq!(session.diagnostics(doc).len(), 1);
        assert!(
            matches!(
                session.diagnostics(doc)[0].kind,
                acvus_lsp::LspErrorKind::Parse(_)
            ),
            "{:?}",
            session.diagnostics(doc)
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
        let mut session = LspSession::new(&interner, super::bare(vec![]));
        let doc = session
            .open(
                super::template_document(&interner, "test"),
                "% if\nbroken\n",
            )
            .expect("the session opens no other document");
        assert_eq!(session.diagnostics(doc).len(), 1);
        assert!(shown(&session, doc).is_empty());
        assert!(session.required_inputs(doc).is_empty());
    }
}
