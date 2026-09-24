use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use acvus_ast::Span;
use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, EntryKind, Environment, Host,
    HostDiagnostic, Link, Listing, Location, LspError, LspErrorKind, LspSession, Mode,
    RecordingReader, Sites, TextError, Vfs, Workspace,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, QualifiedRef};
use acvus_mir::ty::{Effect, PolyBuilder, Ty, TyTerm, lift_to_poly};
use acvus_utils::{Freeze, Interner};

struct TestHost {
    root: PathBuf,
    loads: usize,
    checks: Cell<usize>,
    reads: RefCell<Vec<PathBuf>>,
}

fn document(interner: &Interner, name: &str) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode: Mode::Template,
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

fn environment(interner: &Interner, x: Ty) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(vec![]),
        contexts: Freeze::new(vec![Context {
            qref: QualifiedRef::root(interner.intern("x")),
            ty: lift_to_poly(&x),
            init: None,
        }]),
        types: Freeze::default(),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    }
}

impl TestHost {
    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn env_of_a(
        &self,
        interner: &Interner,
        reader: &RecordingReader<'_>,
    ) -> Result<CompilationGraph, Vec<HostDiagnostic>> {
        let path = self.path("env.txt");
        let refused = |message: String| {
            vec![HostDiagnostic {
                path: path.clone(),
                span: None,
                message,
            }]
        };
        match reader
            .read(&path)
            .map_err(|e| refused(e.to_string()))?
            .trim()
        {
            "string" => Ok(environment(interner, Ty::String)),
            "int" => Ok(environment(interner, Ty::I64)),
            other => Err(refused(format!("unknown environment `{other}`"))),
        }
    }
}

struct TestCompilation {
    name: &'static str,
    holds_a: bool,
}

impl Host for TestHost {
    type Compilation = TestCompilation;

    fn compilations(
        &mut self,
        interner: &Interner,
        reader: &RecordingReader<'_>,
    ) -> Listing<TestCompilation> {
        self.loads += 1;
        let spec = |name: &str| DocumentSpec {
            path: self.path(&format!("{name}.acvt")),
            document: document(interner, name),
        };
        let listing = self.path("listing.txt");
        let refusals = match reader.read(&listing) {
            Ok(text) if text.trim() == "broken" => vec![HostDiagnostic {
                path: listing,
                span: None,
                message: "the listing does not parse".to_string(),
            }],
            Ok(_) | Err(_) => Vec::new(),
        };
        Listing {
            compilations: vec![
                CompilationSpec {
                    id: CompilationId(self.path("a.id")),
                    environment: self.env_of_a(interner, reader).map(|graph| Environment {
                        graph,
                        host: TestCompilation {
                            name: "a.id",
                            holds_a: true,
                        },
                        sites: Sites::default(),
                    }),
                    documents: vec![spec("a"), spec("shared")],
                },
                CompilationSpec {
                    id: CompilationId(self.path("b.id")),
                    environment: Ok(Environment {
                        graph: environment(interner, Ty::String),
                        host: TestCompilation {
                            name: "b.id",
                            holds_a: false,
                        },
                        sites: Sites::default(),
                    }),
                    documents: vec![spec("shared")],
                },
            ],
            refusals,
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        self.reads.borrow_mut().push(path.to_path_buf());
        vfs.read(path)
            .map_err(|error| format!("the test host cannot read {}: {error}", path.display()))
    }

    fn check(&self, compilation: &TestCompilation, checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        self.checks.set(self.checks.get() + 1);
        let typed = checked
            .inferred_ty(&self.path("shared.acvt"))
            .and_then(Ty::effect)
            .is_some();
        let mut refusals = vec![HostDiagnostic {
            path: self.path("rules.txt"),
            span: None,
            message: format!("checked {}, shared typed: {typed}", compilation.name),
        }];
        if compilation.holds_a {
            let inputs = checked
                .required_inputs(&self.path("a.acvt"))
                .expect("a.acvt is open in an accepted compilation");
            let mut names: Vec<String> = inputs
                .iter()
                .map(|input| checked.interner().resolve(input.name.name).to_string())
                .collect();
            names.sort();
            refusals.push(HostDiagnostic {
                path: self.path("inputs.txt"),
                span: None,
                message: names.join(","),
            });
        }
        refusals
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    workspace: Workspace<TestHost>,
}

fn fixture(env: &str, a: Option<&str>, shared: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("env.txt"), env).expect("write env.txt");
    if let Some(a) = a {
        std::fs::write(root.join("a.acvt"), a).expect("write a.acvt");
    }
    std::fs::write(root.join("shared.acvt"), shared).expect("write shared.acvt");
    let host = TestHost {
        root: root.clone(),
        loads: 0,
        checks: Cell::new(0),
        reads: RefCell::new(Vec::new()),
    };
    Fixture {
        _dir: dir,
        workspace: Workspace::new(&Interner::new(), host),
        root,
    }
}

impl Fixture {
    fn at(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn messages(&self, name: &str) -> Vec<String> {
        self.workspace
            .diagnostics()
            .get(&self.at(name))
            .unwrap_or_else(|| panic!("{name} has diagnostics"))
            .iter()
            .map(|error| error.message.clone())
            .collect()
    }

    fn kinds(&self, name: &str) -> Vec<LspErrorKind> {
        self.workspace.diagnostics()[&self.at(name)]
            .iter()
            .map(|error| error.kind)
            .collect()
    }

    fn take_read_counts(&self, names: [&str; 2]) -> [usize; 2] {
        let reads = self.workspace.host().reads.take();
        names.map(|name| reads.iter().filter(|read| **read == self.at(name)).count())
    }

    fn accepted(&self, name: &str) -> bool {
        !self.workspace.diagnostics().contains_key(&self.at(name))
    }
}

#[test]
fn accepted_documents_have_no_entries_and_every_compilation_is_checked() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert!(f.accepted("a.acvt"));
    assert!(f.accepted("shared.acvt"));
    assert_eq!(
        f.messages("rules.txt"),
        [
            "checked a.id, shared typed: true",
            "checked b.id, shared typed: true"
        ]
    );
}

#[test]
fn a_refusal_two_compilations_make_is_shown_once() {
    let source = "{{ @x + 1 }}";
    let f = fixture("string", Some("{{ @x }}"), source);

    let interner = Interner::new();
    let mut alone = LspSession::new(&interner, environment(&interner, Ty::String));
    let doc = alone
        .open(document(&interner, "shared"), source)
        .expect("the session opens no other document");
    let expected: Vec<String> = alone
        .diagnostics(doc)
        .into_iter()
        .map(|e| e.message)
        .collect();

    assert!(!expected.is_empty());
    assert_eq!(f.messages("shared.acvt"), expected);
}

#[test]
fn host_rules_run_only_over_an_accepted_compilation() {
    let f = fixture("string", Some("{{ @x + 1 }}"), "{{ @x }}");
    assert!(!f.messages("a.acvt").is_empty());
    assert_eq!(
        f.messages("rules.txt"),
        ["checked b.id, shared typed: true"]
    );
}

#[test]
fn a_failed_environment_is_reported_on_its_file_and_its_documents_are_parsed() {
    let mut f = fixture("bogus", Some("{{ @x + 1 }}"), "{{ @x }}");
    assert_eq!(f.messages("env.txt"), ["unknown environment `bogus`"]);
    assert_eq!(f.kinds("env.txt"), [LspErrorKind::Host(None)]);
    assert!(f.accepted("a.acvt"));

    let a = f.at("a.acvt");
    f.workspace.set_buffer(a, "{{ ".to_string());
    assert!(
        matches!(
            f.kinds("a.acvt").as_slice(),
            [LspErrorKind::Parse(_), LspErrorKind::Parse(_)]
        ),
        "{:?}",
        f.kinds("a.acvt")
    );
}

#[test]
fn a_change_to_a_file_the_listing_read_lists_again() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert_eq!(f.workspace.host().loads, 1);

    let env = f.at("env.txt");
    f.workspace.set_buffer(env.clone(), "int".to_string());
    assert_eq!(f.workspace.host().loads, 2);
    assert!(!f.messages("a.acvt").is_empty(), "an Int is not emitted");

    f.workspace.drop_buffer(&env);
    assert_eq!(f.workspace.host().loads, 3);
    assert!(f.accepted("a.acvt"));

    std::fs::write(&env, "int").expect("rewrite env.txt");
    f.workspace.file_changed(&env);
    assert_eq!(f.workspace.host().loads, 4);
    assert!(!f.messages("a.acvt").is_empty(), "an Int is not emitted");
}

#[test]
fn a_change_to_a_path_neither_read_listed_nor_held_changes_nothing() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let before = f.workspace.diagnostics();
    assert_eq!(f.workspace.host().loads, 1);
    assert_eq!(f.workspace.host().checks.get(), 2);

    let readme = f.at("README.md");
    f.workspace
        .set_buffer(readme.clone(), "# notes".to_string());
    assert_eq!(f.workspace.host().loads, 1);
    assert_eq!(f.workspace.host().checks.get(), 2);

    f.workspace.drop_buffer(&readme);
    std::fs::write(&readme, "# notes").expect("write README.md");
    f.workspace.file_changed(&readme);
    assert_eq!(f.workspace.host().loads, 1);
    assert_eq!(f.workspace.host().checks.get(), 2);
    assert_eq!(f.workspace.diagnostics(), before);
}

#[test]
fn a_document_edit_does_not_reload() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let a = f.at("a.acvt");
    f.workspace.set_buffer(a, "{{ @x + 1 }}".to_string());
    assert_eq!(f.workspace.host().loads, 1);
    assert!(!f.messages("a.acvt").is_empty());
}

#[test]
fn a_disk_change_under_an_open_buffer_moves_nothing() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let env = f.at("env.txt");
    f.workspace.set_buffer(env.clone(), "int".to_string());
    let before = f.workspace.diagnostics();

    std::fs::write(&env, "bogus").expect("rewrite env.txt");
    f.workspace.file_changed(&env);
    assert_eq!(f.workspace.host().loads, 2);
    assert_eq!(f.workspace.diagnostics(), before);
}

#[test]
fn an_unreadable_document_opens_when_its_buffer_arrives() {
    let mut f = fixture("string", None, "{{ @x }}");
    assert_eq!(f.kinds("a.acvt"), [LspErrorKind::Host(None)]);
    assert!(f.messages("rules.txt").iter().all(|m| !m.contains("a.id")));

    let a = f.at("a.acvt");
    f.workspace.set_buffer(a, "{{ @x }}".to_string());
    assert!(f.accepted("a.acvt"));
    assert!(
        f.messages("rules.txt")
            .contains(&"checked a.id, shared typed: true".to_string())
    );
}

fn refused_by_the_test_host(path: &Path) -> LspError {
    let error = std::fs::read_to_string(path).expect_err("the document was never written");
    LspError {
        kind: LspErrorKind::Host(None),
        message: format!("the test host cannot read {}: {error}", path.display()),
        related: Vec::new(),
    }
}

#[test]
fn a_document_the_host_cannot_read_is_refused_in_the_host_s_words() {
    let f = fixture("string", None, "{{ @x }}");
    let a = f.at("a.acvt");
    assert_eq!(
        f.workspace.diagnostics()[&a],
        [refused_by_the_test_host(&a)]
    );
}

#[test]
fn a_document_of_a_refused_environment_is_read_through_the_host() {
    let f = fixture("bogus", None, "{{ @x }}");
    assert_eq!(f.messages("env.txt"), ["unknown environment `bogus`"]);
    let a = f.at("a.acvt");
    assert_eq!(
        f.workspace.diagnostics()[&a],
        [refused_by_the_test_host(&a)]
    );
}

#[test]
fn completions_answer_for_a_document_alone() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert!(f.workspace.completions(&f.at("env.txt"), 0).is_none());
    let completions = f
        .workspace
        .completions(&f.at("a.acvt"), 5)
        .expect("a.acvt is open");
    assert!(completions.items.iter().any(|item| item.label == "@x"));
}

#[test]
fn a_refusal_of_the_listing_belongs_to_no_compilation() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let listing = f.at("listing.txt");
    f.workspace.set_buffer(listing, "broken".to_string());
    assert_eq!(f.messages("listing.txt"), ["the listing does not parse"]);
    assert_eq!(f.kinds("listing.txt"), [LspErrorKind::Host(None)]);
    assert!(f.accepted("a.acvt"));
    assert_eq!(f.messages("rules.txt").len(), 2);
}

#[test]
fn a_host_reads_the_inputs_a_document_requires() {
    let f = fixture("string", Some("{{ $who }}"), "{{ @x }}");
    assert!(f.accepted("a.acvt"));
    let inputs = f.messages("inputs.txt");
    assert_eq!(inputs.len(), 1);
    assert!(inputs[0].split(',').any(|name| name == "who"), "{inputs:?}");
}

#[test]
fn a_document_two_compilations_hold_is_read_once_per_change() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let documents = ["shared.acvt", "a.acvt"];
    assert_eq!(f.take_read_counts(documents), [1, 1]);

    let shared = f.at("shared.acvt");
    f.workspace
        .set_buffer(shared, "{{ @x }}{{ @x }}".to_string());
    assert_eq!(f.take_read_counts(documents), [1, 0]);

    let env = f.at("env.txt");
    f.workspace.set_buffer(env, "int".to_string());
    assert_eq!(f.workspace.host().loads, 2);
    assert_eq!(f.take_read_counts(documents), [1, 1]);
}

#[test]
fn a_document_s_text_is_the_one_its_compilations_parsed() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let a = f.at("a.acvt");
    std::fs::write(&a, "{{ @x + 1 }}").expect("rewrite a.acvt");
    assert_eq!(f.workspace.text(&a).expect("a.acvt reads"), "{{ @x }}");
    assert!(f.accepted("a.acvt"));

    f.workspace.file_changed(&a);
    assert_eq!(f.workspace.text(&a).expect("a.acvt reads"), "{{ @x + 1 }}");
    assert!(!f.accepted("a.acvt"));

    let notes = f.at("notes.txt");
    std::fs::write(&notes, "one").expect("write notes.txt");
    assert_eq!(f.workspace.text(&notes).expect("notes.txt reads"), "one");
    std::fs::write(&notes, "two").expect("rewrite notes.txt");
    assert_eq!(f.workspace.text(&notes).expect("notes.txt reads"), "two");
}

#[test]
fn an_unreadable_document_s_text_is_its_refusal() {
    let f = fixture("string", None, "{{ @x }}");
    let a = f.at("a.acvt");
    let Err(TextError::DocumentRefused(message)) = f.workspace.text(&a) else {
        panic!("a.acvt was never written, and the host refuses it");
    };
    assert_eq!(message, refused_by_the_test_host(&a).message);
}

#[test]
fn checks_are_counted_per_accepted_compilation() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert_eq!(f.workspace.host().checks.get(), 2);
}

struct DerivingHost {
    root: PathBuf,
    loads: usize,
}

fn unit_environment(interner: &Interner, x: Ty) -> Environment<()> {
    Environment {
        graph: environment(interner, x),
        host: (),
        sites: Sites::default(),
    }
}

impl Host for DerivingHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, reader: &RecordingReader<'_>) -> Listing<()> {
        self.loads += 1;
        let d = self.root.join("d.acvt");
        let derived = match reader.read(&d) {
            Ok(text) if text.contains("int") => Ok(Ty::I64),
            Ok(_) => Ok(Ty::String),
            Err(error) => Err(vec![HostDiagnostic {
                path: d.clone(),
                span: None,
                message: error.to_string(),
            }]),
        };
        let spec = |name: &str| DocumentSpec {
            path: self.root.join(format!("{name}.acvt")),
            document: document(interner, name),
        };
        Listing {
            compilations: vec![
                CompilationSpec {
                    id: CompilationId(self.root.join("one.id")),
                    environment: Ok(unit_environment(interner, Ty::String)),
                    documents: vec![spec("d")],
                },
                CompilationSpec {
                    id: CompilationId(self.root.join("two.id")),
                    environment: derived.map(|x| unit_environment(interner, x)),
                    documents: vec![spec("e")],
                },
            ],
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
fn a_document_the_listing_read_lists_again_and_stays_a_document() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().to_path_buf();
    let d = root.join("d.acvt");
    let e = root.join("e.acvt");
    std::fs::write(&d, "{{ @x }}").expect("write d.acvt");
    std::fs::write(&e, "{{ @x }}").expect("write e.acvt");
    let mut workspace = Workspace::new(
        &Interner::new(),
        DerivingHost {
            root: root.clone(),
            loads: 0,
        },
    );
    assert_eq!(workspace.host().loads, 1);
    assert!(!workspace.diagnostics().contains_key(&e));

    let text = "int {{ @x }}";
    workspace.set_buffer(d.clone(), text.to_string());
    assert_eq!(workspace.host().loads, 2);
    assert!(!workspace.diagnostics().contains_key(&d));
    assert!(
        !workspace.diagnostics()[&e].is_empty(),
        "an Int is not emitted"
    );

    let at = text.find("@x").expect("d.acvt reads `@x`");
    let completions = workspace
        .completions(&d, at + 2)
        .expect("d.acvt is open in one.id");
    assert!(completions.items.iter().any(|item| item.label == "@x"));
    assert!(workspace.hover(&d, at + 1).is_some());

    workspace.set_buffer(d, "{{ @x }}".to_string());
    assert_eq!(workspace.host().loads, 3);
    assert!(!workspace.diagnostics().contains_key(&e));
}

struct DirectoryHost {
    src: PathBuf,
    loads: usize,
    checks: Cell<usize>,
    listed: Vec<PathBuf>,
}

impl Host for DirectoryHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, reader: &RecordingReader<'_>) -> Listing<()> {
        self.loads += 1;
        let refused = |path: &Path, message: String| HostDiagnostic {
            path: path.to_path_buf(),
            span: None,
            message,
        };
        let mut documents = Vec::new();
        let mut refusals = Vec::new();
        let entries = match reader.read_dir(&self.src) {
            Ok(entries) => entries,
            Err(error) => {
                refusals.push(refused(&self.src, error.to_string()));
                Vec::new()
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    refusals.push(refused(&self.src, error.to_string()));
                    continue;
                }
            };
            match entry.kind {
                Ok(EntryKind::File) => {}
                Ok(EntryKind::Directory | EntryKind::Symlink) => continue,
                Err(error) => {
                    refusals.push(refused(&entry.path, error.to_string()));
                    continue;
                }
            }
            if entry
                .path
                .extension()
                .is_none_or(|extension| extension != "acvt")
            {
                continue;
            }
            let stem = entry
                .path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("a listed template has a UTF-8 stem");
            documents.push(DocumentSpec {
                document: document(interner, stem),
                path: entry.path,
            });
        }
        self.listed = documents.iter().map(|spec| spec.path.clone()).collect();
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.src.join("src.id")),
                environment: Ok(unit_environment(interner, Ty::String)),
                documents,
            }],
            refusals,
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        self.checks.set(self.checks.get() + 1);
        Vec::new()
    }
}

struct Listed {
    _dir: tempfile::TempDir,
    src: PathBuf,
    workspace: Workspace<DirectoryHost>,
}

fn listed() -> Listed {
    let dir = tempfile::tempdir().expect("a temp dir");
    let src = dir.path().join("src");
    std::fs::create_dir(&src).expect("create src");
    std::fs::write(src.join("a.acvt"), "{{ @x }}").expect("write a.acvt");
    std::fs::write(src.join("notes.txt"), "notes").expect("write notes.txt");
    let workspace = Workspace::new(
        &Interner::new(),
        DirectoryHost {
            src: src.clone(),
            loads: 0,
            checks: Cell::new(0),
            listed: Vec::new(),
        },
    );
    assert_eq!(workspace.host().loads, 1);
    assert_eq!(workspace.host().checks.get(), 1);
    Listed {
        _dir: dir,
        src,
        workspace,
    }
}

impl Listed {
    fn last_listing(&self) -> Vec<PathBuf> {
        let mut listed = self.workspace.host().listed.clone();
        listed.sort();
        listed
    }
}

#[test]
fn a_buffer_new_to_a_listed_directory_lists_again() {
    let mut l = listed();
    let new = l.src.join("new.acvt");
    l.workspace.set_buffer(new.clone(), "{{ @x }}".to_string());
    assert_eq!(l.workspace.host().loads, 2);
    assert_eq!(l.last_listing(), [l.src.join("a.acvt"), new]);
}

#[test]
fn a_file_created_on_disk_in_a_listed_directory_lists_again() {
    let mut l = listed();
    let new = l.src.join("new.acvt");
    std::fs::write(&new, "{{ @x }}").expect("write new.acvt");
    l.workspace.file_changed(&new);
    assert_eq!(l.workspace.host().loads, 2);
    assert_eq!(l.last_listing(), [l.src.join("a.acvt"), new]);
}

#[test]
fn a_content_change_in_a_listed_directory_rechecks_without_listing_again() {
    let mut l = listed();
    let a = l.src.join("a.acvt");
    l.workspace
        .set_buffer(a.clone(), "{{ @x }}{{ @x }}".to_string());
    assert_eq!(l.workspace.host().loads, 1);
    assert_eq!(l.workspace.host().checks.get(), 2);

    l.workspace.drop_buffer(&a);
    assert_eq!(l.workspace.host().loads, 1);
    assert_eq!(l.workspace.host().checks.get(), 3);

    std::fs::write(&a, "{{ @x }}{{ @x }}").expect("rewrite a.acvt");
    l.workspace.file_changed(&a);
    assert_eq!(l.workspace.host().loads, 1);
    assert_eq!(l.workspace.host().checks.get(), 4);

    let notes = l.src.join("notes.txt");
    std::fs::write(&notes, "more notes").expect("rewrite notes.txt");
    l.workspace.file_changed(&notes);
    assert_eq!(l.workspace.host().loads, 1);
    assert_eq!(l.workspace.host().checks.get(), 4);
}

#[test]
fn dropping_a_buffer_only_file_of_a_listed_directory_lists_again() {
    let mut l = listed();
    let new = l.src.join("new.acvt");
    l.workspace.set_buffer(new.clone(), "{{ @x }}".to_string());
    assert_eq!(l.workspace.host().loads, 2);

    l.workspace.drop_buffer(&new);
    assert_eq!(l.workspace.host().loads, 3);
    assert_eq!(l.last_listing(), [l.src.join("a.acvt")]);
}

struct TailHost {
    root: PathBuf,
    tails: RefCell<Vec<(PathBuf, Option<Ty>)>>,
    returns: RefCell<Vec<(PathBuf, Option<Ty>)>>,
}

impl TailHost {
    fn script(&self) -> PathBuf {
        self.root.join("s.acvus")
    }

    fn template(&self) -> PathBuf {
        self.root.join("t.acvt")
    }

    fn absent(&self) -> PathBuf {
        self.root.join("absent.acvus")
    }
}

impl Host for TailHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, _reader: &RecordingReader<'_>) -> Listing<()> {
        let script = Document {
            qref: QualifiedRef::root(interner.intern("s")),
            mode: Mode::Script,
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(TyTerm::Never),
                captures: vec![],
                effect: Effect::OPAQUE.into(),
                flows: acvus_mir::ty::Flows::Every.into(),
            },
            inputs: acvus_mir::graph::Inputs::FromReads,
        };
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("tail.id")),
                environment: Ok(unit_environment(interner, Ty::String)),
                documents: vec![
                    DocumentSpec {
                        path: self.script(),
                        document: script,
                    },
                    DocumentSpec {
                        path: self.template(),
                        document: document(interner, "t"),
                    },
                ],
            }],
            refusals: Vec::new(),
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        for path in [self.script(), self.template(), self.absent()] {
            let returns = match checked.inferred_ty(&path) {
                Some(Ty::Fn { ret, .. }) => Some((**ret).clone()),
                Some(other) => panic!("a document's function is typed {other:?}"),
                None => None,
            };
            self.returns.borrow_mut().push((path.clone(), returns));
            self.tails
                .borrow_mut()
                .push((path.clone(), checked.tail_ty(&path)));
        }
        Vec::new()
    }
}

/// An unsuffixed literal no use constrains takes `i64`, the width
/// `TyVarBound::integer_default` gives while it remains among the admitted
/// ones, and a declared `!` holds the tail to nothing.
#[test]
fn a_host_reads_the_type_of_the_value_a_body_returns() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().to_path_buf();
    std::fs::write(root.join("s.acvus"), "1").expect("write s.acvus");
    std::fs::write(root.join("t.acvt"), "{{ @x }}").expect("write t.acvt");
    let workspace = Workspace::new(
        &Interner::new(),
        TailHost {
            root: root.clone(),
            tails: RefCell::new(Vec::new()),
            returns: RefCell::new(Vec::new()),
        },
    );
    let host = workspace.host();
    let (script, template, absent) = (host.script(), host.template(), host.absent());
    assert!(workspace.diagnostics().is_empty());

    assert_eq!(
        *host.tails.borrow(),
        [
            (script.clone(), Some(Ty::I64)),
            (template.clone(), Some(Ty::String)),
            (absent.clone(), None),
        ]
    );
    assert_eq!(
        *host.returns.borrow(),
        [
            (script, Some(Ty::Never)),
            (template, Some(Ty::String)),
            (absent, None),
        ]
    );
}

/// Links every spelling of a document's file name in `manifest.txt` to that
/// document's start, and every `@x` there to the site of the context `x`.
struct LinkHost {
    root: PathBuf,
    link_from_a: bool,
}

const LINKED: [&str; 2] = ["a", "ba"];

impl LinkHost {
    fn manifest(&self) -> PathBuf {
        self.root.join("manifest.txt")
    }

    fn x_site(&self) -> Location {
        Location {
            path: self.root.join("ctx.txt"),
            span: Span::new(0, 1),
        }
    }

    fn links_in(&self, manifest: &str) -> Vec<Link> {
        let from = |start: usize, spelled: &str| Location {
            path: self.manifest(),
            span: Span::new(start, start + spelled.len()),
        };
        let to_documents = LINKED.iter().flat_map(|name| {
            let file = format!("{name}.acvt");
            manifest
                .match_indices(&file)
                .map(|(start, spelled)| Link {
                    from: from(start, spelled),
                    to: Location {
                        path: self.root.join(&file),
                        span: Span::new(0, 0),
                    },
                })
                .collect::<Vec<_>>()
        });
        let to_x = manifest.match_indices("@x").map(|(start, spelled)| Link {
            from: from(start, spelled),
            to: self.x_site(),
        });
        to_documents.chain(to_x).collect()
    }
}

impl Host for LinkHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, reader: &RecordingReader<'_>) -> Listing<()> {
        let mut refusals = Vec::new();
        let mut links = match reader.read(&self.manifest()) {
            Ok(manifest) => self.links_in(&manifest),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                refusals.push(HostDiagnostic {
                    path: self.manifest(),
                    span: None,
                    message: error.to_string(),
                });
                Vec::new()
            }
        };
        if self.link_from_a {
            links.push(Link {
                from: Location {
                    path: self.root.join("a.acvt"),
                    span: Span::new(3, 5),
                },
                to: Location {
                    path: self.manifest(),
                    span: Span::new(0, 0),
                },
            });
        }
        let mut sites = Sites::default();
        sites
            .contexts
            .insert(QualifiedRef::root(interner.intern("x")), self.x_site());
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("linked.id")),
                environment: Ok(Environment {
                    graph: environment(interner, Ty::String),
                    host: (),
                    sites,
                }),
                documents: LINKED
                    .iter()
                    .map(|name| DocumentSpec {
                        path: self.root.join(format!("{name}.acvt")),
                        document: document(interner, name),
                    })
                    .collect(),
            }],
            refusals,
            links,
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path).map_err(|error| error.to_string())
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

struct Linked {
    _dir: tempfile::TempDir,
    root: PathBuf,
    workspace: Workspace<LinkHost>,
}

fn linked(manifest: &str, link_from_a: bool) -> Linked {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().to_path_buf();
    for name in LINKED {
        std::fs::write(root.join(format!("{name}.acvt")), "{{ @x }}").expect("write a document");
    }
    std::fs::write(root.join("ctx.txt"), "x").expect("write ctx.txt");
    std::fs::write(root.join("manifest.txt"), manifest).expect("write manifest.txt");
    let workspace = Workspace::new(
        &Interner::new(),
        LinkHost {
            root: root.clone(),
            link_from_a,
        },
    );
    Linked {
        _dir: dir,
        root,
        workspace,
    }
}

impl Linked {
    fn manifest(&self) -> PathBuf {
        self.root.join("manifest.txt")
    }

    fn start_of(&self, name: &str) -> Option<Location> {
        Some(Location {
            path: self.root.join(name),
            span: Span::new(0, 0),
        })
    }

    fn in_manifest(&self, start: usize, end: usize) -> Location {
        Location {
            path: self.manifest(),
            span: Span::new(start, end),
        }
    }
}

#[test]
fn definition_in_a_host_file_answers_the_link_whose_span_holds_the_offset() {
    let l = linked("use a.acvt;", false);
    let manifest = l.manifest();
    assert_eq!(l.workspace.definition(&manifest, 4), l.start_of("a.acvt"));
    assert_eq!(l.workspace.definition(&manifest, 7), l.start_of("a.acvt"));
}

#[test]
fn definition_in_a_host_file_outside_every_link_answers_nothing() {
    let l = linked("use a.acvt;", false);
    let manifest = l.manifest();
    assert_eq!(l.workspace.definition(&manifest, 0), None);
    assert_eq!(l.workspace.definition(&manifest, 3), None);
    assert_eq!(l.workspace.definition(&l.root.join("notes.txt"), 0), None);
}

#[test]
fn a_link_moves_with_the_buffer_of_the_file_it_is_read_from() {
    let mut l = linked("use a.acvt;", false);
    let manifest = l.manifest();
    l.workspace
        .set_buffer(manifest.clone(), "use it later: a.acvt;".to_string());
    assert_eq!(l.workspace.definition(&manifest, 5), None);
    assert_eq!(l.workspace.definition(&manifest, 15), l.start_of("a.acvt"));
}

#[test]
fn of_nested_links_the_narrowest_answers() {
    let l = linked("ba.acvt", false);
    let manifest = l.manifest();
    assert_eq!(l.workspace.definition(&manifest, 0), l.start_of("ba.acvt"));
    assert_eq!(l.workspace.definition(&manifest, 3), l.start_of("a.acvt"));
    assert_eq!(l.workspace.definition(&manifest, 7), l.start_of("a.acvt"));
}

#[test]
fn a_cursor_right_after_a_link_is_on_it() {
    let l = linked("a.acvt next", false);
    assert_eq!(
        l.workspace.definition(&l.manifest(), 6),
        l.start_of("a.acvt")
    );
}

#[test]
fn a_link_from_a_document_is_refused_and_the_checker_answers_there() {
    let l = linked("", true);
    let a = l.root.join("a.acvt");
    let [refusal] = l.workspace.diagnostics()[&a]
        .clone()
        .try_into()
        .unwrap_or_else(|errors: Vec<LspError>| panic!("one refusal on a.acvt: {errors:?}"));
    assert_eq!(refusal.kind, LspErrorKind::Host(Some(Span::new(3, 5))));
    assert!(
        refusal
            .message
            .ends_with("is a document, and a document's names are the checker's"),
        "{}",
        refusal.message
    );
    assert_eq!(
        l.workspace.definition(&a, 4),
        Some(l.workspace.host().x_site())
    );
}

#[test]
fn the_references_of_a_context_hold_the_links_to_its_site() {
    let l = linked("@x and @x, not a.acvt", false);
    let at = |name: &str| Location {
        path: l.root.join(name),
        span: Span::new(3, 5),
    };
    let uses = vec![
        at("a.acvt"),
        at("ba.acvt"),
        l.in_manifest(0, 2),
        l.in_manifest(7, 9),
    ];
    let a = l.root.join("a.acvt");
    assert_eq!(l.workspace.references(&a, 4, false), Some(uses.clone()));

    let mut declared = uses;
    declared.insert(2, l.workspace.host().x_site());
    assert_eq!(l.workspace.references(&a, 4, true), Some(declared));
}
