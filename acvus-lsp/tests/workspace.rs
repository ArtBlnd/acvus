use std::cell::Cell;
use std::path::PathBuf;

use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, Environment, Host,
    HostDiagnostic, Listing, LspErrorCategory, LspSession, Mode, Vfs, Workspace,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, QualifiedRef};
use acvus_mir::ty::{Effect, PolyBuilder, Ty, TyTerm, lift_to_poly};
use acvus_utils::{Freeze, Interner};

struct TestHost {
    root: PathBuf,
    loads: usize,
    checks: Cell<usize>,
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
        },
    }
}

fn environment(interner: &Interner, x: Ty) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(vec![]),
        contexts: Freeze::new(vec![Context {
            qref: QualifiedRef::root(interner.intern("x")),
            ty: lift_to_poly(&x),
        }]),
        types: Freeze::default(),
        bindings: Bindings::default(),
        entry: None,
    }
}

impl TestHost {
    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn env_of_a(
        &self,
        interner: &Interner,
        vfs: &Vfs,
    ) -> Result<CompilationGraph, Vec<HostDiagnostic>> {
        let path = self.path("env.txt");
        let refused = |message: String| {
            vec![HostDiagnostic {
                path: path.clone(),
                span: None,
                message,
            }]
        };
        match vfs.read(&path).map_err(|e| refused(e.to_string()))?.trim() {
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

    fn compilations(&mut self, interner: &Interner, vfs: &Vfs) -> Listing<TestCompilation> {
        self.loads += 1;
        let spec = |name: &str| DocumentSpec {
            path: self.path(&format!("{name}.acvt")),
            document: document(interner, name),
        };
        let listing = self.path("listing.txt");
        let refusals = match vfs.read(&listing) {
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
                    environment: self.env_of_a(interner, vfs).map(|graph| Environment {
                        graph,
                        host: TestCompilation {
                            name: "a.id",
                            holds_a: true,
                        },
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
                    }),
                    documents: vec![spec("shared")],
                },
            ],
            refusals,
        }
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
            .unwrap_or_else(|| panic!("{name} has an entry"))
            .iter()
            .map(|error| error.message.clone())
            .collect()
    }

    fn categories(&self, name: &str) -> Vec<LspErrorCategory> {
        self.workspace.diagnostics()[&self.at(name)]
            .iter()
            .map(|error| error.category)
            .collect()
    }
}

#[test]
fn accepted_documents_have_empty_entries_and_every_compilation_is_checked() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert!(f.messages("a.acvt").is_empty());
    assert!(f.messages("shared.acvt").is_empty());
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
    let doc = alone.open(document(&interner, "shared"), source);
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
    assert_eq!(f.categories("env.txt"), [LspErrorCategory::Host]);
    assert!(f.messages("a.acvt").is_empty());

    let a = f.at("a.acvt");
    f.workspace.set_buffer(a, "{{ ".to_string());
    assert_eq!(
        f.categories("a.acvt"),
        [LspErrorCategory::Parse, LspErrorCategory::Parse]
    );
}

#[test]
fn a_change_to_a_file_that_is_not_a_document_reloads() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert_eq!(f.workspace.host().loads, 1);

    let env = f.at("env.txt");
    f.workspace.set_buffer(env.clone(), "int".to_string());
    assert_eq!(f.workspace.host().loads, 2);
    assert!(!f.messages("a.acvt").is_empty(), "an Int is not emitted");

    f.workspace.drop_buffer(&env);
    assert_eq!(f.workspace.host().loads, 3);
    assert!(f.messages("a.acvt").is_empty());
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
    assert_eq!(f.categories("a.acvt"), [LspErrorCategory::Unreadable]);
    assert!(f.messages("rules.txt").iter().all(|m| !m.contains("a.id")));

    let a = f.at("a.acvt");
    f.workspace.set_buffer(a, "{{ @x }}".to_string());
    assert!(f.messages("a.acvt").is_empty());
    assert!(
        f.messages("rules.txt")
            .contains(&"checked a.id, shared typed: true".to_string())
    );
}

#[test]
fn completions_answer_for_a_document_alone() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert!(f.workspace.completions(&f.at("env.txt"), 0).is_none());
    let items = f
        .workspace
        .completions(&f.at("a.acvt"), 5)
        .expect("a.acvt is open");
    assert!(items.iter().any(|item| item.label == "@x"));
}

#[test]
fn a_refusal_of_the_listing_belongs_to_no_compilation() {
    let mut f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    let listing = f.at("listing.txt");
    f.workspace.set_buffer(listing, "broken".to_string());
    assert_eq!(f.messages("listing.txt"), ["the listing does not parse"]);
    assert_eq!(f.categories("listing.txt"), [LspErrorCategory::Host]);
    assert!(f.messages("a.acvt").is_empty());
    assert_eq!(f.messages("rules.txt").len(), 2);
}

#[test]
fn a_host_reads_the_inputs_a_document_requires() {
    let f = fixture("string", Some("{{ $who }}"), "{{ @x }}");
    assert!(f.messages("a.acvt").is_empty());
    let inputs = f.messages("inputs.txt");
    assert_eq!(inputs.len(), 1);
    assert!(inputs[0].split(',').any(|name| name == "who"), "{inputs:?}");
}

#[test]
fn checks_are_counted_per_accepted_compilation() {
    let f = fixture("string", Some("{{ @x }}"), "{{ @x }}");
    assert_eq!(f.workspace.host().checks.get(), 2);
}
