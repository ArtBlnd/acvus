//! Hover and go-to-definition through the session's and the workspace's
//! public API.

use std::path::{Path, PathBuf};

use acvus_ast::Span;
use acvus_extern::{Externs, TypesOnly};
use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Definition, Document, DocumentSpec, Environment, Host,
    HostDiagnostic, Hover, Listing, Location, LspErrorKind, LspSession, Mode, OpenRefusal,
    RecordingReader, Sites, Vfs, Workspace,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, Mutability, ParamTerm, PolyBuilder, Ty, TyTerm, TypeArg, TypeRegistry, lift_to_poly,
};
use acvus_utils::{Freeze, Interner};

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
) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
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

fn script_reading_int_x(interner: &Interner, name: &str) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode: Mode::Script,
        ty: TyTerm::Fn {
            params: vec![ParamTerm::new(interner.intern("x"), lift_to_poly(&Ty::I64))],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
        inputs: acvus_mir::graph::Inputs::Declared,
    }
}

fn document(interner: &Interner, name: &str, mode: Mode) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode,
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
        inputs: acvus_mir::graph::Inputs::FromReads,
    }
}

/// The offset of the `nth` occurrence of `needle` in `source`.
fn nth(source: &str, needle: &str, n: usize) -> usize {
    source
        .match_indices(needle)
        .nth(n)
        .map(|(at, _)| at)
        .unwrap_or_else(|| panic!("`{needle}` occurs {} times in {source:?}", n + 1))
}

fn string_display(interner: &Interner) -> String {
    Ty::String.display(interner).to_string()
}

fn open(
    interner: &Interner,
    environment: CompilationGraph,
    mode: Mode,
    source: &str,
) -> (LspSession, acvus_lsp::DocId) {
    let mut session = LspSession::new(interner, environment);
    let id = session
        .open(document(interner, "test", mode), source)
        .expect("the session opens no other document");
    (session, id)
}

#[test]
fn hover_on_a_use_shows_its_type() {
    let i = Interner::new();
    let source = "let s = @name; s";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let at = nth(source, "s", 1);
    let hover = session.hover(doc, at).expect("the use has a type");
    assert_eq!(hover.span, Span::new(at, at + 1));
    assert_eq!(hover.ty, string_display(&i));
}

#[test]
fn hover_works_on_a_refused_body() {
    let i = Interner::new();
    let source = "let s = @name; s + 1";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert!(
        !session.diagnostics(doc).is_empty(),
        "String + Int is refused"
    );
    let at = nth(source, "s", 1);
    let hover = session.hover(doc, at).expect("the use has a type");
    assert_eq!(hover.span, Span::new(at, at + 1));
    assert_eq!(hover.ty, string_display(&i));
    // `+` beside text is the concatenation whatever the other operand is,
    // so the checker records `String` for the refused sum.
    let plus = session
        .hover(doc, nth(source, "+", 0))
        .expect("the refused expression has a recorded type");
    assert_eq!(plus.span, Span::new(at, source.len()));
    assert_eq!(plus.ty, string_display(&i));
}

#[test]
fn a_poisoned_expression_shows_error() {
    let i = Interner::new();
    let source = "let s = @name; s * 2";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert!(
        !session.diagnostics(doc).is_empty(),
        "String * Int is refused"
    );
    let at = nth(source, "s", 1);
    assert_eq!(
        session.hover(doc, at).map(|hover| hover.ty),
        Some(string_display(&i))
    );
    let times = session
        .hover(doc, nth(source, "*", 0))
        .expect("the refused expression has a recorded type");
    assert_eq!(times.span, Span::new(at, source.len()));
    assert_eq!(times.ty, "<error>");
}

#[test]
fn a_cursor_right_after_a_use_hovers_and_defines_it() {
    let i = Interner::new();
    let source = "% let v = @name\n{{ v }}";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Template,
        source,
    );
    assert!(session.diagnostics(doc).is_empty());
    let binder = nth(source, "v", 0);
    let used = nth(source, "v", 1);
    let after_use = used + 1;
    assert_eq!(
        session.hover(doc, after_use),
        Some(Hover {
            span: Span::new(used, after_use),
            ty: string_display(&i),
        })
    );
    assert_eq!(
        session.definition(doc, after_use),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1),
        })
    );
}

#[test]
fn a_use_goes_to_its_let() {
    let i = Interner::new();
    let source = "let s = 1; s";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let binder = nth(source, "s", 0);
    assert_eq!(
        session.definition(doc, nth(source, "s", 1)),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

/// A binder resolves to itself, as a lambda's parameter does.
#[test]
fn a_binder_goes_to_itself() {
    let i = Interner::new();
    let source = "let s = 1; let f = |x| -> x + s; f(s)";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    for binder in [nth(source, "s", 0), nth(source, "x", 0)] {
        assert_eq!(
            session.definition(doc, binder),
            Some(Definition::Local {
                span: Span::new(binder, binder + 1)
            })
        );
    }
}

/// One function has at most one open document: a second is refused and
/// the first answers as before; once the first closes, the function is
/// gone from the graph and opens again.
#[test]
fn a_second_document_of_one_function_is_refused() {
    let i = Interner::new();
    let source = "let s = 1; s";
    let (mut session, first) = open(&i, bare(vec![]), Mode::Script, source);
    let refusal = session
        .open(document(&i, "test", Mode::Script), "let t = 2; t")
        .expect_err("`test` is the body of the open document `first`");
    assert_eq!(
        refusal,
        OpenRefusal::FunctionHeld {
            function: "test".to_string(),
            holder: first,
        }
    );
    assert_eq!(
        refusal.to_string(),
        "function `test` is already the body of an open document"
    );

    let binder = nth(source, "s", 0);
    let used = nth(source, "s", 1);
    assert_eq!(
        session.definition(first, used),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
    let hover = session.hover(first, used).expect("the use has a type");
    assert_eq!(hover.span, Span::new(used, used + 1));
    assert_eq!(hover.ty, Ty::I64.display(&i).to_string());
    // The refusal took no id: the next document is numbered after `first`.
    let other = session
        .open(document(&i, "other", Mode::Script), "1")
        .expect("no open document is `other`");
    assert_eq!(other.raw(), first.raw() + 1);

    let qref = QualifiedRef::root(i.intern("test"));
    session.close(first);
    assert!(session.graph().function(qref).is_none());
    let reopened = session
        .open(document(&i, "test", Mode::Script), source)
        .expect("the document of `test` is closed");
    assert_eq!(
        session.definition(reopened, used),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

#[test]
fn a_shadowed_name_goes_to_the_later_binder() {
    let i = Interner::new();
    let source = "let s = 1; let s = @name; s";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    let definition = session.definition(doc, nth(source, "s", 2));
    let Some(Definition::Local { span }) = definition else {
        panic!("expected a local definition, got {definition:?}");
    };
    let binder = nth(source, "s", 1);
    assert_eq!(span, Span::new(binder, binder + 1));
}

#[test]
fn a_lambda_parameter_and_a_capture() {
    let i = Interner::new();
    let source = "let n = 1; let f = |x| -> x + n; f(2)";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let param = nth(source, "x", 0);
    assert_eq!(
        session.definition(doc, nth(source, "x", 1)),
        Some(Definition::Local {
            span: Span::new(param, param + 1)
        })
    );
    let outer = nth(source, "n", 0);
    assert_eq!(
        session.definition(doc, nth(source, "n", 1)),
        Some(Definition::Local {
            span: Span::new(outer, outer + 1)
        })
    );
}

#[test]
fn a_call_of_a_local_lambda_goes_to_its_let() {
    let i = Interner::new();
    let source = "let f = |x| -> x + 1; f(2)";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let definition = session.definition(doc, nth(source, "f", 1));
    let binder = nth(source, "f", 0);
    assert_eq!(
        definition,
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

#[test]
fn a_match_arm_binder() {
    let i = Interner::new();
    let source = "let o = Some(1); match o { Some(v) => v, None => 0, }";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let binder = nth(source, "v", 0);
    assert_eq!(
        session.definition(doc, nth(source, "v", 1)),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

#[test]
fn template_context_hover_and_statement_binding() {
    let i = Interner::new();
    let source = "% let x = @name\n{{ x }}{{ @name }}";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Template,
        source,
    );
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let context = nth(source, "@name", 1);
    let hover = session
        .hover(doc, context + 1)
        .expect("the context has a type");
    assert_eq!(hover.span, Span::new(context, context + "@name".len()));
    assert_eq!(hover.ty, string_display(&i));
    let definition = session.definition(doc, nth(source, "x", 1));
    let Some(Definition::Local { span }) = definition else {
        panic!("expected a local definition, got {definition:?}");
    };
    let binder = nth(source, "x", 0);
    assert_eq!(span, Span::new(binder, binder + 1));
}

#[test]
fn a_context_is_a_definition_the_host_places_and_an_extern_call_has_none() {
    let i = Interner::new();
    let source = "{{ @count.to_string() }}";
    let (session, doc) = open(
        &i,
        with_std(&i, root_contexts(&i, &[("count", Ty::I64)])),
        Mode::Template,
        source,
    );
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    assert_eq!(
        session.definition(doc, nth(source, "count", 0)),
        Some(Definition::Context(QualifiedRef::root(i.intern("count"))))
    );
    assert_eq!(session.definition(doc, nth(source, "to_string", 0)), None);
}

#[test]
fn an_input_is_a_definition_the_host_places() {
    let i = Interner::new();
    let source = "$x + 1";
    let mut session = LspSession::new(&i, bare(vec![]));
    let doc = session
        .open(script_reading_int_x(&i, "test"), source)
        .expect("the session opens no other document");
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    assert_eq!(
        session.definition(doc, nth(source, "x", 0)),
        Some(Definition::Input(i.intern("x")))
    );
}

#[test]
fn a_binding_whose_value_does_not_parse_is_poison_and_still_defined() {
    let i = Interner::new();
    let source = "let s = ; s";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(!session.diagnostics(doc).is_empty());
    let binder = nth(source, "s", 0);
    assert_eq!(
        session
            .hover(doc, nth(source, "s", 1))
            .map(|hover| hover.ty),
        Some("<error>".to_string())
    );
    assert_eq!(
        session.definition(doc, nth(source, "s", 1)),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

#[test]
fn hover_and_definition_answer_beside_a_broken_line() {
    let i = Interner::new();
    let source = "let s = @name;\nlet = 2;\ns";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert_eq!(session.diagnostics(doc).len(), 1);
    let at = nth(source, "s", 1);
    let hover = session.hover(doc, at).expect("the use has a type");
    assert_eq!(hover.span, Span::new(at, at + 1));
    assert_eq!(hover.ty, string_display(&i));
    let binder = nth(source, "s", 0);
    assert_eq!(
        session.definition(doc, at),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
}

// -- Workspace -------------------------------------------------------

struct TwoDocuments {
    root: PathBuf,
}

impl Host for TwoDocuments {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, _reader: &RecordingReader<'_>) -> Listing<()> {
        let spec = |name: &str| DocumentSpec {
            path: self.root.join(format!("{name}.acvt")),
            document: document(interner, name, Mode::Template),
        };
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("main.id")),
                environment: Ok(Environment {
                    graph: bare(vec![]),
                    host: (),
                    sites: Sites::default(),
                }),
                documents: vec![spec("greet"), spec("caller")],
            }],
            refusals: Vec::new(),
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

#[test]
fn a_call_goes_to_the_document_that_defines_the_function() {
    let i = Interner::new();
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().to_path_buf();
    let greet = root.join("greet.acvt");
    let caller = root.join("caller.acvt");
    std::fs::write(&greet, "hello").expect("write greet");
    let source = "{{ greet() }}";
    std::fs::write(&caller, source).expect("write caller");
    let workspace = Workspace::new(&i, TwoDocuments { root });
    let diagnostics = workspace.diagnostics();
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(
        workspace.definition(&caller, nth(source, "greet", 0)),
        Some(Location {
            path: greet,
            span: Span::new(0, 0),
        })
    );
    let hover = workspace
        .hover(&caller, nth(source, "greet", 0))
        .expect("the callee has a type");
    assert!(hover.ty.contains("String"), "{}", hover.ty);
}

/// Lists two documents of the function `greet`.
struct OneFunctionTwice {
    root: PathBuf,
}

impl Host for OneFunctionTwice {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, _reader: &RecordingReader<'_>) -> Listing<()> {
        let spec = |file: &str| DocumentSpec {
            path: self.root.join(file),
            document: document(interner, "greet", Mode::Template),
        };
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("main.id")),
                environment: Ok(Environment {
                    graph: bare(vec![]),
                    host: (),
                    sites: Sites::default(),
                }),
                documents: vec![spec("first.acvt"), spec("second.acvt")],
            }],
            refusals: Vec::new(),
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

#[test]
fn a_second_document_of_one_function_is_a_host_refusal() {
    let i = Interner::new();
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().to_path_buf();
    let first = root.join("first.acvt");
    let second = root.join("second.acvt");
    let source = "% let x = \"hi\"\n{{ x }}";
    std::fs::write(&first, source).expect("write first");
    std::fs::write(&second, "{{ 2 }}").expect("write second");
    let workspace = Workspace::new(&i, OneFunctionTwice { root });
    let diagnostics = workspace.diagnostics();
    assert!(!diagnostics.contains_key(&first), "{diagnostics:?}");
    let [refusal] = diagnostics[&second].as_slice() else {
        panic!("one refusal on the second document: {diagnostics:?}");
    };
    assert_eq!(refusal.kind, LspErrorKind::Host(None));
    assert_eq!(
        refusal.message,
        format!(
            "{} and {} are both the body of function `greet`; {} is not opened",
            first.display(),
            second.display(),
            second.display(),
        )
    );
    assert_eq!(workspace.hover(&second, 3), None);

    let binder = nth(source, "x", 0);
    assert_eq!(
        workspace.definition(&first, nth(source, "x", 1)),
        Some(Location {
            path: first.clone(),
            span: Span::new(binder, binder + 1),
        })
    );
    let used = nth(source, "x", 1);
    let hover = workspace.hover(&first, used).expect("the use has a type");
    assert_eq!(hover.span, Span::new(used, used + 1));
}

struct SitedHost {
    root: PathBuf,
    with_sites: bool,
}

impl SitedHost {
    fn context_site(&self) -> Location {
        Location {
            path: self.root.join("decl.txt"),
            span: Span::new(3, 9),
        }
    }

    fn input_site(&self) -> Location {
        Location {
            path: self.root.join("decl.txt"),
            span: Span::new(12, 15),
        }
    }
}

impl Host for SitedHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, _reader: &RecordingReader<'_>) -> Listing<()> {
        let sites = match self.with_sites {
            true => Sites {
                contexts: [(
                    QualifiedRef::root(interner.intern("name")),
                    self.context_site(),
                )]
                .into_iter()
                .collect(),
                inputs: [(interner.intern("x"), self.input_site())]
                    .into_iter()
                    .collect(),
            },
            false => Sites::default(),
        };
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("main.id")),
                environment: Ok(Environment {
                    graph: bare(root_contexts(interner, &[("name", Ty::I64)])),
                    host: (),
                    sites,
                }),
                documents: vec![DocumentSpec {
                    path: self.root.join("main.acvus"),
                    document: script_reading_int_x(interner, "main"),
                }],
            }],
            refusals: Vec::new(),
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

const READS_NAME_AND_X: &str = "let y = @name + $x; @name + y";

fn sited(interner: &Interner, with_sites: bool) -> (tempfile::TempDir, Workspace<SitedHost>) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("main.acvus"), READS_NAME_AND_X).expect("write main");
    let workspace = Workspace::new(interner, SitedHost { root, with_sites });
    let diagnostics = workspace.diagnostics();
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    (dir, workspace)
}

#[test]
fn a_context_and_an_input_go_to_the_sites_the_host_gave_them() {
    let i = Interner::new();
    let (dir, workspace) = sited(&i, true);
    let host = workspace.host();
    let main = dir.path().join("main.acvus");
    let source = READS_NAME_AND_X;
    assert_eq!(
        workspace.definition(&main, nth(source, "name", 1)),
        Some(host.context_site())
    );
    assert_eq!(
        workspace.definition(&main, nth(source, "x", 0)),
        Some(host.input_site())
    );
}

#[test]
fn a_context_and_an_input_without_a_site_have_no_definition() {
    let i = Interner::new();
    let (dir, workspace) = sited(&i, false);
    let main = dir.path().join("main.acvus");
    let source = READS_NAME_AND_X;
    assert_eq!(workspace.definition(&main, nth(source, "name", 0)), None);
    assert_eq!(workspace.definition(&main, nth(source, "x", 0)), None);
}

#[test]
fn references_include_a_site_only_as_the_declaration() {
    let i = Interner::new();
    let (dir, workspace) = sited(&i, true);
    let host = workspace.host();
    let main = dir.path().join("main.acvus");
    let source = READS_NAME_AND_X;
    let at = |needle: &str, n: usize| {
        let start = nth(source, needle, n);
        Location {
            path: main.clone(),
            span: Span::new(start, start + needle.len()),
        }
    };
    let names = vec![at("@name", 0), at("@name", 1)];
    assert_eq!(
        workspace.references(&main, nth(source, "name", 0), false),
        Some(names.clone())
    );
    let mut with_site = names;
    with_site.push(host.context_site());
    with_site.sort();
    assert_eq!(
        workspace.references(&main, nth(source, "name", 0), true),
        Some(with_site)
    );
    assert_eq!(
        workspace.references(&main, nth(source, "x", 0), false),
        Some(vec![at("$x", 0)])
    );
    let mut with_site = vec![at("$x", 0), host.input_site()];
    with_site.sort();
    assert_eq!(
        workspace.references(&main, nth(source, "x", 0), true),
        Some(with_site)
    );

    let (dir, unplaced) = sited(&i, false);
    let main = dir.path().join("main.acvus");
    assert_eq!(
        unplaced
            .references(&main, nth(source, "x", 0), true)
            .map(|found| found.len()),
        Some(1)
    );
}

#[test]
fn a_string_literal_binding_is_a_str_reference() {
    let i = Interner::new();
    let source = "let s = \"a\"; s";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let str_ref: Ty = TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str)));
    for at in [nth(source, "s", 0), nth(source, "s", 1)] {
        assert_eq!(
            session.hover(doc, at),
            Some(Hover {
                span: Span::new(at, at + 1),
                ty: str_ref.display(&i).to_string(),
            })
        );
    }
}

#[test]
fn a_for_binding_use_goes_to_the_binding() {
    let i = Interner::new();
    let source = "let t = 0; for k in 0..3 { t = t + k; } t";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let binder = nth(source, "k", 0);
    assert_eq!(
        session.definition(doc, nth(source, "k", 1)),
        Some(Definition::Local {
            span: Span::new(binder, binder + 1)
        })
    );
    assert_eq!(
        session.hover(doc, binder),
        Some(Hover {
            span: Span::new(binder, binder + 1),
            ty: Ty::I64.display(&i).to_string(),
        })
    );
}

#[test]
fn hover_on_a_let_binder_shows_the_bound_type() {
    let i = Interner::new();
    let source = "let s = @name; s";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let binder = nth(source, "s", 0);
    assert_eq!(
        session.hover(doc, binder),
        Some(Hover {
            span: Span::new(binder, binder + 1),
            ty: string_display(&i),
        })
    );

    let source = "let s = \"a\"; s";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let binder = nth(source, "s", 0);
    let str_ref: Ty = TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str)));
    assert_eq!(
        session.hover(doc, binder),
        Some(Hover {
            span: Span::new(binder, binder + 1),
            ty: str_ref.display(&i).to_string(),
        })
    );
}

#[test]
fn hover_on_a_binder_in_a_refused_body() {
    let i = Interner::new();
    let source = "let s = @name; s + 1";
    let (session, doc) = open(
        &i,
        bare(root_contexts(&i, &[("name", Ty::String)])),
        Mode::Script,
        source,
    );
    assert!(
        !session.diagnostics(doc).is_empty(),
        "String + Int is refused"
    );
    let binder = nth(source, "s", 0);
    assert_eq!(
        session.hover(doc, binder),
        Some(Hover {
            span: Span::new(binder, binder + 1),
            ty: string_display(&i),
        })
    );
}

#[test]
fn hover_on_a_match_arm_binder() {
    let i = Interner::new();
    let source = "let o = Some(1); match o { Some(v) => v, None => 0, }";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        session.diagnostics(doc).is_empty(),
        "{:?}",
        session.diagnostics(doc)
    );
    let binder = nth(source, "v", 0);
    let at_binder = session.hover(doc, binder).expect("the binder has a type");
    assert_eq!(at_binder.span, Span::new(binder, binder + 1));
    let at_use = session
        .hover(doc, nth(source, "v", 1))
        .expect("the use has a type");
    assert_eq!(at_binder.ty, at_use.ty);
}

// NOTE: twenty pairs, since which of two functions without an edge is
// checked first follows the graph's hash order.
#[test]
fn a_caller_opened_before_its_callee_is_checked_after_it() {
    for pair in 0..20 {
        let i = Interner::new();
        let mut session = LspSession::new(&i, bare(vec![]));
        let caller = session
            .open(
                document(&i, &format!("caller{pair}"), Mode::Template),
                &format!("{{{{ callee{pair}() }}}}"),
            )
            .expect("the caller is the first document of its function");
        let callee = session
            .open(document(&i, &format!("callee{pair}"), Mode::Template), "hi")
            .expect("the callee is the first document of its function");
        assert_eq!(session.diagnostics(callee), [], "pair {pair}");
        assert_eq!(session.diagnostics(caller), [], "pair {pair}");
    }
}
