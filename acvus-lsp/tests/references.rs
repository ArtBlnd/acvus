//! Find-references and rename through the session's and the workspace's
//! public API: what refers to what is what the checker recorded.

use std::path::{Path, PathBuf};

use acvus_extern::{Externs, TypesOnly};
use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, DocId, Document, DocumentSpec, Edit, Environment,
    Host, HostDiagnostic, Listing, Location, LspSession, Mode, RenameRefusal, Sites, Vfs,
    Workspace,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, ParamTerm, PolyBuilder, PolyParam, Ty, TyTerm, TypeRegistry, lift_to_poly,
};
use acvus_utils::{Freeze, Interner};

fn root_contexts(interner: &Interner, ctx: &[(&str, Ty)]) -> Vec<Context> {
    ctx.iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_to_poly(ty),
        })
        .collect()
}

fn environment(
    contexts: Vec<Context>,
    functions: Vec<Function>,
    types: TypeRegistry,
) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        entry: None,
    }
}

fn bare(contexts: Vec<Context>) -> CompilationGraph {
    environment(contexts, vec![], TypeRegistry::default())
}

fn with_std(interner: &Interner, contexts: Vec<Context>) -> CompilationGraph {
    let Externs {
        functions, types, ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), interner)
        .expect("standard registries combine");
    environment(contexts, functions, types)
}

fn document(interner: &Interner, name: &str, mode: Mode, params: Vec<PolyParam>) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode,
        ty: TyTerm::Fn {
            params,
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }
}

fn open(
    interner: &Interner,
    environment: CompilationGraph,
    mode: Mode,
    source: &str,
) -> (LspSession, DocId) {
    let mut session = LspSession::new(interner, environment);
    let id = session
        .open(document(interner, "test", mode, vec![]), source)
        .expect("the session opens no other document");
    (session, id)
}

/// The offset of the `nth` occurrence of `needle` in `source`.
fn nth(source: &str, needle: &str, n: usize) -> usize {
    source
        .match_indices(needle)
        .nth(n)
        .map(|(at, _)| at)
        .unwrap_or_else(|| panic!("`{needle}` occurs {} times in {source:?}", n + 1))
}

/// The span of the `nth` occurrence of `needle` in `source`.
fn span(source: &str, needle: &str, n: usize) -> (usize, usize) {
    let at = nth(source, needle, n);
    (at, at + needle.len())
}

fn texts<'s>(source: &'s str, spans: &[(usize, usize)]) -> Vec<&'s str> {
    spans
        .iter()
        .map(|&(start, end)| &source[start..end])
        .collect()
}

fn accepted(session: &LspSession, doc: DocId) {
    let diagnostics = session.diagnostics(doc);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

// -- Locals ----------------------------------------------------------

#[test]
fn a_local_is_referred_to_by_its_uses_from_the_binder_and_from_a_use() {
    let i = Interner::new();
    let source = "let a = 1; let b = a + a; b";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    let uses = vec![span(source, "a", 1), span(source, "a", 2)];
    let with_binder = vec![
        span(source, "a", 0),
        span(source, "a", 1),
        span(source, "a", 2),
    ];
    for from in [
        nth(source, "a", 0),
        nth(source, "a", 1),
        nth(source, "a", 2),
    ] {
        assert_eq!(session.references(doc, from, false), Some(uses.clone()));
        assert_eq!(
            session.references(doc, from, true),
            Some(with_binder.clone())
        );
    }
}

#[test]
fn a_shadowing_binding_is_referred_to_only_by_the_uses_it_reaches() {
    let i = Interner::new();
    let source = "let a = 1; let c = a; let a = 2; a + c";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    assert_eq!(
        session.references(doc, nth(source, "a", 2), true),
        Some(vec![span(source, "a", 2), span(source, "a", 3)])
    );
    assert_eq!(
        session.references(doc, nth(source, "a", 0), true),
        Some(vec![span(source, "a", 0), span(source, "a", 1)])
    );
}

#[test]
fn a_lambda_use_of_a_captured_name_refers_to_the_outer_binder() {
    let i = Interner::new();
    let source = "let n = 1; let f = |x| -> x + n; f(n)";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    assert_eq!(
        session.references(doc, nth(source, "n", 1), true),
        Some(vec![
            span(source, "n", 0),
            span(source, "n", 1),
            span(source, "n", 2)
        ])
    );
}

#[test]
fn an_offset_on_no_name_has_no_references() {
    let i = Interner::new();
    let source = "let a = 1; a + 2";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    assert_eq!(session.references(doc, nth(source, "2", 0), true), None);
    assert_eq!(session.references(doc, nth(source, "+", 0), true), None);
}

// -- Functions, contexts, inputs -------------------------------------

#[test]
fn a_function_called_with_method_syntax_is_referred_to_at_its_name() {
    let i = Interner::new();
    let source = "{{ @count.to_string() }}{{ to_string(&@count) }}";
    let (session, doc) = open(
        &i,
        with_std(&i, root_contexts(&i, &[("count", Ty::I64)])),
        Mode::Template,
        source,
    );
    accepted(&session, doc);
    let expected = vec![span(source, "to_string", 0), span(source, "to_string", 1)];
    for from in [nth(source, "to_string", 0), nth(source, "to_string", 1)] {
        let found = session
            .references(doc, from, true)
            .expect("the callee resolved");
        assert_eq!(found, expected);
        assert_eq!(texts(source, &found), ["to_string", "to_string"]);
    }
}

#[test]
fn an_input_is_referred_to_by_every_read_of_it() {
    let i = Interner::new();
    let params = vec![ParamTerm::new(i.intern("x"), lift_to_poly(&Ty::I64))];
    let mut session = LspSession::new(&i, bare(vec![]));
    let source = "let y = $x; $x + y";
    let doc = session
        .open(document(&i, "test", Mode::Script, params), source)
        .expect("the session opens no other document");
    accepted(&session, doc);
    let found = session
        .references(doc, nth(source, "$x", 1), true)
        .expect("the input resolved");
    assert_eq!(found, vec![span(source, "$x", 0), span(source, "$x", 1)]);
}

// -- Rename ----------------------------------------------------------

#[test]
fn a_rename_edits_the_binder_and_every_use() {
    let i = Interner::new();
    let source = "let a = 1; let b = a + a; b";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    let edit = |n| Edit {
        span: span(source, "a", n),
        text: "total".to_string(),
    };
    assert_eq!(
        session.rename(doc, nth(source, "a", 1), "total"),
        Ok(vec![edit(0), edit(1), edit(2)])
    );
}

#[test]
fn a_rename_edits_an_assignment_and_keeps_a_shorthand_key() {
    let i = Interner::new();
    let source = "let a = 1; a = a + 1; let o = { a, }; o";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    let edit = |n, text: &str| Edit {
        span: span(source, "a", n),
        text: text.to_string(),
    };
    assert_eq!(
        session.rename(doc, nth(source, "a", 0), "z"),
        Ok(vec![
            edit(0, "z"),
            edit(1, "z"),
            edit(2, "z"),
            edit(3, "a: z")
        ])
    );
}

#[test]
fn a_rename_to_a_keyword_or_a_non_identifier_is_refused() {
    let i = Interner::new();
    let source = "let a = 1; a";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    let at = nth(source, "a", 0);
    assert_eq!(
        session.rename(doc, at, "match"),
        Err(RenameRefusal::Keyword("match".to_string()))
    );
    assert_eq!(
        session.rename(doc, at, "1abc"),
        Err(RenameRefusal::NotAnIdentifier("1abc".to_string()))
    );
    assert_eq!(
        session.rename(doc, at, "a b"),
        Err(RenameRefusal::NotAnIdentifier("a b".to_string()))
    );
    assert_eq!(
        session.rename(doc, at, "$a"),
        Err(RenameRefusal::NotAnIdentifier("$a".to_string()))
    );
    assert_eq!(
        RenameRefusal::Keyword("match".to_string()).to_string(),
        "`match` is a keyword"
    );
}

#[test]
fn a_rename_an_inner_binding_would_capture_is_refused() {
    let i = Interner::new();
    let source = "let a = 1; let b = 2; a + b";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    assert_eq!(
        session.rename(doc, nth(source, "a", 0), "b"),
        Err(RenameRefusal::ResolutionChanged("b".to_string()))
    );
}

#[test]
fn a_rename_that_shadows_a_use_of_an_outer_binding_is_refused() {
    let i = Interner::new();
    let source = "let b = 1; let f = |a| -> a + b; f(2)";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    accepted(&session, doc);
    assert_eq!(
        session.rename(doc, nth(source, "a", 0), "b"),
        Err(RenameRefusal::ResolutionChanged("b".to_string()))
    );
    let edit = |n| Edit {
        span: span(source, "a", n),
        text: "c".to_string(),
    };
    assert_eq!(
        session.rename(doc, nth(source, "a", 0), "c"),
        Ok(vec![edit(0), edit(1)])
    );
}

#[test]
fn a_rename_of_a_context_a_function_or_an_input_is_refused() {
    let i = Interner::new();
    let source = "{{ @count.to_string() }}";
    let (session, doc) = open(
        &i,
        with_std(&i, root_contexts(&i, &[("count", Ty::I64)])),
        Mode::Template,
        source,
    );
    accepted(&session, doc);
    assert_eq!(
        session.rename(doc, nth(source, "count", 0), "total"),
        Err(RenameRefusal::Context)
    );
    assert_eq!(
        session.rename(doc, nth(source, "to_string", 0), "shown"),
        Err(RenameRefusal::Function)
    );
    assert_eq!(
        session.rename(doc, nth(source, "{{", 0), "shown"),
        Err(RenameRefusal::NotAName)
    );

    let params = vec![ParamTerm::new(i.intern("x"), lift_to_poly(&Ty::I64))];
    let mut session = LspSession::new(&i, bare(vec![]));
    let source = "$x + 1";
    let doc = session
        .open(document(&i, "test", Mode::Script, params), source)
        .expect("the session opens no other document");
    accepted(&session, doc);
    assert_eq!(session.rename(doc, 0, "y"), Err(RenameRefusal::Input));
}

// -- Purity and broken documents -------------------------------------

#[test]
fn references_and_rename_leave_the_graph_as_it_was() {
    let i = Interner::new();
    let source = "let a = @name; let b = 2; a + b";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    let qref = session.function_ref(doc).expect("the document is open");
    let diagnostics = session.diagnostics(doc);
    assert!(!diagnostics.is_empty(), "String + Int is refused");
    let hovers: Vec<_> = (0..source.len()).map(|at| session.hover(doc, at)).collect();
    let view = session.graph().view(qref).expect("the body is checked");

    assert!(session.references(doc, nth(source, "a", 0), true).is_some());
    assert!(
        session
            .references(doc, nth(source, "@name", 0), true)
            .is_some()
    );
    assert!(session.rename(doc, nth(source, "a", 0), "c").is_ok());
    assert!(session.rename(doc, nth(source, "a", 0), "b").is_err());

    assert_eq!(session.diagnostics(doc), diagnostics);
    let hovers_after: Vec<_> = (0..source.len()).map(|at| session.hover(doc, at)).collect();
    assert_eq!(hovers_after, hovers);
    let view_after = session.graph().view(qref).expect("the body is checked");
    assert!(
        std::ptr::eq(&*view, &*view_after),
        "the graph holds the view it held before"
    );
}

#[test]
fn references_and_rename_answer_beside_a_broken_line() {
    let i = Interner::new();
    let source = "let a = 1;\nlet = 2;\na + a";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert_eq!(session.diagnostics(doc).len(), 1);
    assert_eq!(
        session.references(doc, nth(source, "a", 2), true),
        Some(vec![
            span(source, "a", 0),
            span(source, "a", 1),
            span(source, "a", 2)
        ])
    );
    let edit = |n| Edit {
        span: span(source, "a", n),
        text: "q".to_string(),
    };
    assert_eq!(
        session.rename(doc, nth(source, "a", 0), "q"),
        Ok(vec![edit(0), edit(1), edit(2)])
    );
}

// -- Workspace -------------------------------------------------------

/// One compilation of the given template documents over the `@name`
/// context.
struct Documents {
    root: PathBuf,
    names: Vec<&'static str>,
}

impl Host for Documents {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, _vfs: &Vfs) -> Listing<()> {
        let documents = self
            .names
            .iter()
            .map(|name| DocumentSpec {
                path: self.root.join(format!("{name}.acvt")),
                document: document(interner, name, Mode::Template, vec![]),
            })
            .collect();
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("main.id")),
                environment: Ok(Environment {
                    graph: bare(root_contexts(interner, &[("name", Ty::String)])),
                    host: (),
                    sites: Sites::default(),
                }),
                documents,
            }],
            refusals: Vec::new(),
        }
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

/// A template document written to `<name>.acvt`.
struct DocumentFile<'t> {
    name: &'static str,
    text: &'t str,
}

fn workspace(interner: &Interner, root: &Path, files: &[DocumentFile<'_>]) -> Workspace<Documents> {
    for file in files {
        std::fs::write(root.join(format!("{}.acvt", file.name)), file.text)
            .expect("write a document");
    }
    let workspace = Workspace::new(
        interner,
        Documents {
            root: root.to_path_buf(),
            names: files.iter().map(|file| file.name).collect(),
        },
    );
    let diagnostics = workspace.diagnostics();
    assert!(diagnostics.values().all(Vec::is_empty), "{diagnostics:?}");
    workspace
}

#[test]
fn a_context_is_referred_to_in_every_document_of_the_compilation() {
    let i = Interner::new();
    let dir = tempfile::tempdir().expect("a temporary directory");
    let greet_source = "hello {{ @name }}";
    let other_source = "% let x = @name\n{{ x }}";
    let workspace = workspace(
        &i,
        dir.path(),
        &[
            DocumentFile {
                name: "greet",
                text: greet_source,
            },
            DocumentFile {
                name: "other",
                text: other_source,
            },
        ],
    );
    let greet = dir.path().join("greet.acvt");
    let other = dir.path().join("other.acvt");
    assert_eq!(
        workspace.references(&greet, nth(greet_source, "@name", 0), true),
        Some(vec![
            Location {
                path: greet,
                span: span(greet_source, "@name", 0),
            },
            Location {
                path: other.clone(),
                span: span(other_source, "@name", 0),
            },
        ])
    );
    assert_eq!(
        workspace.references(&other, nth(other_source, "x", 1), true),
        Some(vec![
            Location {
                path: other.clone(),
                span: span(other_source, "x", 0),
            },
            Location {
                path: other.clone(),
                span: span(other_source, "x", 1),
            },
        ])
    );
    let at = nth(other_source, "x", 0);
    assert_eq!(
        workspace.rename(&other, at, "shown"),
        Ok(vec![
            (
                other.clone(),
                Edit {
                    span: span(other_source, "x", 0),
                    text: "shown".to_string(),
                }
            ),
            (
                other.clone(),
                Edit {
                    span: span(other_source, "x", 1),
                    text: "shown".to_string(),
                }
            ),
        ])
    );
    assert_eq!(
        workspace.rename(&other, nth(other_source, "@name", 0), "shown"),
        Err(RenameRefusal::Context)
    );
}

#[test]
fn a_function_is_referred_to_at_its_calls_and_declared_at_its_document() {
    let i = Interner::new();
    let dir = tempfile::tempdir().expect("a temporary directory");
    let caller_source = "{{ greet() }}{{ greet() }}";
    let workspace = workspace(
        &i,
        dir.path(),
        &[
            DocumentFile {
                name: "greet",
                text: "hello",
            },
            DocumentFile {
                name: "caller",
                text: caller_source,
            },
        ],
    );
    let greet = dir.path().join("greet.acvt");
    let caller = dir.path().join("caller.acvt");
    let calls = vec![
        Location {
            path: caller.clone(),
            span: span(caller_source, "greet", 0),
        },
        Location {
            path: caller.clone(),
            span: span(caller_source, "greet", 1),
        },
    ];
    assert_eq!(
        workspace.references(&caller, nth(caller_source, "greet", 1), false),
        Some(calls.clone())
    );
    let mut with_declaration = calls;
    with_declaration.push(Location {
        path: greet,
        span: (0, 0),
    });
    with_declaration.sort();
    assert_eq!(
        workspace.references(&caller, nth(caller_source, "greet", 0), true),
        Some(with_declaration)
    );
    assert_eq!(
        workspace.rename(&caller, nth(caller_source, "greet", 0), "welcome"),
        Err(RenameRefusal::Function)
    );
}
