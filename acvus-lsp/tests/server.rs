//! `serve` at the protocol: a client over `Connection::memory()` against a
//! host whose documents and environment file live in a temporary directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

use acvus_ast::Span;
use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, Environment, Host,
    HostDiagnostic, Link, Listing, Location, Mode, RecordingReader, RenameRefusal, ServeError,
    Sites, Vfs, serve,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, ParamTerm, PolyBuilder, Ty, TyTerm, lift_to_poly};
use acvus_utils::{Freeze, Interner};
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types as lsp;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use serde_json::json;

struct TestHost {
    root: PathBuf,
}

const TEMPLATES: [&str; 2] = ["main", "greet"];
const SCRIPT: &str = "s";

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
    }
}

fn pad(interner: &Interner) -> Function {
    Function {
        qref: QualifiedRef::root(interner.intern("pad")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![
                ParamTerm::new(interner.intern("width"), lift_to_poly(&Ty::I64)),
                ParamTerm::new(interner.intern("_"), lift_to_poly(&Ty::String)),
            ],
            ret: Box::new(lift_to_poly(&Ty::String)),
            captures: vec![],
            effect: Effect::PURE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }
}

fn environment(interner: &Interner, x: Ty) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(vec![pad(interner)]),
        contexts: Freeze::new(vec![Context {
            qref: QualifiedRef::root(interner.intern("x")),
            ty: lift_to_poly(&x),
        }]),
        types: Freeze::default(),
        bindings: Bindings::default(),
        entry: None,
    }
}

impl Host for TestHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, reader: &RecordingReader<'_>) -> Listing<()> {
        let env = self.root.join("env.txt");
        let refused = |message: String| {
            vec![HostDiagnostic {
                path: env.clone(),
                span: None,
                message,
            }]
        };
        let graph = match reader.read(&env) {
            Ok(text) if text.trim() == "string" => Ok(environment(interner, Ty::String)),
            Ok(text) if text.trim() == "int" => Ok(environment(interner, Ty::I64)),
            Ok(text) => Err(refused(format!("unknown environment `{}`", text.trim()))),
            Err(error) => Err(refused(error.to_string())),
        };
        let templates = TEMPLATES.iter().map(|name| DocumentSpec {
            path: self.root.join(format!("{name}.acvt")),
            document: document(interner, name, Mode::Template),
        });
        let script = DocumentSpec {
            path: self.root.join(format!("{SCRIPT}.acvus")),
            document: document(interner, SCRIPT, Mode::Script),
        };
        let mut refusals = Vec::new();
        let extra = self.root.join("extra.txt");
        let extras: Vec<DocumentSpec> = match reader.read(&extra) {
            Ok(text) => text
                .lines()
                .enumerate()
                .map(|(at, relative)| DocumentSpec {
                    path: self.root.join(relative),
                    document: document(interner, &format!("extra{at}"), Mode::Template),
                })
                .collect(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                refusals.push(HostDiagnostic {
                    path: extra,
                    span: None,
                    message: error.to_string(),
                });
                Vec::new()
            }
        };
        let documents: Vec<DocumentSpec> = templates.chain([script]).chain(extras).collect();
        let manifest = self.root.join("manifest.txt");
        let links = match reader.read(&manifest) {
            Ok(text) => links_to_documents(&manifest, &text, &documents),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                refusals.push(HostDiagnostic {
                    path: manifest,
                    span: None,
                    message: error.to_string(),
                });
                Vec::new()
            }
        };
        Listing {
            compilations: vec![CompilationSpec {
                id: CompilationId(self.root.join("test.id")),
                environment: graph.map(|graph| Environment {
                    graph,
                    host: (),
                    sites: Sites::default(),
                }),
                documents,
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

/// Every spelling of a document's file name in `text` links to that
/// document's start.
fn links_to_documents(manifest: &Path, text: &str, documents: &[DocumentSpec]) -> Vec<Link> {
    documents
        .iter()
        .flat_map(|spec| {
            let name = spec
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("a document's file name is UTF-8");
            text.match_indices(name).map(|(start, spelled)| Link {
                from: Location {
                    path: manifest.to_path_buf(),
                    span: Span::new(start, start + spelled.len()),
                },
                to: Location {
                    path: spec.path.clone(),
                    span: Span::new(0, 0),
                },
            })
        })
        .collect()
}

fn uri(path: &Path) -> lsp::Uri {
    url::Url::from_file_path(path)
        .expect("a temporary path is absolute")
        .as_str()
        .parse()
        .expect("a file URL is a URI")
}

fn folders(roots: &[&Path]) -> serde_json::Value {
    roots
        .iter()
        .map(|root| json!({ "uri": uri(root), "name": root.display().to_string() }))
        .collect()
}

fn at(line: u32, character: u32) -> lsp::Position {
    lsp::Position { line, character }
}

fn range(start: lsp::Position, end: lsp::Position) -> lsp::Range {
    lsp::Range { start, end }
}

/// The client side of a test, which counts the code units of a position
/// itself so that it does not check the server by the server's own index.
fn offset_in(text: &str, position: lsp::Position, encoding: &str) -> usize {
    let line_start: usize = text
        .split_inclusive('\n')
        .take(position.line as usize)
        .map(str::len)
        .sum();
    let rest = &text[line_start..];
    let line = rest.find('\n').map_or(rest, |end| &rest[..end]);
    let mut units = 0;
    for (at, c) in line.char_indices().chain([(line.len(), '\n')]) {
        if units == position.character as usize {
            return line_start + at;
        }
        units += match encoding {
            "utf-8" => c.len_utf8(),
            "utf-16" => c.len_utf16(),
            other => panic!("the test speaks no {other}"),
        };
    }
    panic!("{position:?} is not on its line of {text:?}")
}

struct Fixture {
    _dir: Option<tempfile::TempDir>,
    root: PathBuf,
}

impl Fixture {
    fn new(env: &str) -> Self {
        let dir = tempfile::tempdir().expect("a temp dir");
        let root = dir.path().to_path_buf();
        Fixture::written(Some(dir), root, env)
    }

    fn nested(&self, name: &str, env: &str) -> Self {
        let root = self.path(name);
        std::fs::create_dir(&root).expect("create the nested project");
        Fixture::written(None, root, env)
    }

    fn written(dir: Option<tempfile::TempDir>, root: PathBuf, env: &str) -> Self {
        let fixture = Fixture { _dir: dir, root };
        fixture.write("env.txt", env);
        fixture.write("main.acvt", "{{ @x }}");
        fixture.write("greet.acvt", "hello");
        fixture.write("s.acvus", "\"s\"");
        fixture
    }

    fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root().join(name)
    }

    fn uri(&self, name: &str) -> lsp::Uri {
        uri(&self.path(name))
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.path(name), text).expect("write a fixture file");
    }
}

struct Client {
    connection: Connection,
    server: JoinHandle<Result<(), ServeError>>,
    next_id: i32,
}

struct Offer<'a> {
    root: Option<&'a Path>,
    folders: &'a [&'a Path],
    encodings: Option<&'a [&'a str]>,
    watches: bool,
}

impl Offer<'_> {
    fn params(&self) -> serde_json::Value {
        let mut capabilities = json!({});
        if let Some(encodings) = self.encodings {
            capabilities["general"] = json!({ "positionEncodings": encodings });
        }
        if self.watches {
            capabilities["workspace"] =
                json!({ "didChangeWatchedFiles": { "dynamicRegistration": true } });
        }
        let mut params = json!({ "processId": null, "capabilities": capabilities });
        if let Some(root) = self.root {
            params["rootUri"] = json!(uri(root));
        }
        if !self.folders.is_empty() {
            params["workspaceFolders"] = folders(self.folders);
        }
        params
    }
}

const PATIENCE: Duration = Duration::from_secs(60);

impl Client {
    fn start(offer: &Offer<'_>) -> (Self, Response) {
        let (server_side, connection) = Connection::memory();
        let server = std::thread::spawn(move || serve(server_side, |root| TestHost { root }));
        let mut client = Client {
            connection,
            server,
            next_id: 0,
        };
        let (before, initialized) =
            client.request_raw(lsp::request::Initialize::METHOD, offer.params());
        assert!(before.is_empty(), "{before:?}");
        (client, initialized)
    }

    /// A client the server accepted, past `initialized`, with what the
    /// server sent on it.
    fn initialized(offer: &Offer<'_>) -> (Self, Vec<Message>) {
        let (mut client, response) = Client::start(offer);
        assert!(response.response_result.is_ok(), "{response:?}");
        client.notify(lsp::notification::Initialized::METHOD, json!({}));
        let sent = client.barrier();
        (client, sent)
    }

    fn receive(&self) -> Message {
        self.connection
            .receiver
            .recv_timeout(PATIENCE)
            .expect("the server answers within the test's patience")
    }

    /// Everything the server sent before the response to the request.
    fn request_raw(&mut self, method: &str, params: serde_json::Value) -> (Vec<Message>, Response) {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        self.connection
            .sender
            .send(
                Request {
                    id: id.clone(),
                    method: method.to_string(),
                    params,
                }
                .into(),
            )
            .expect("the server listens");
        let mut before = Vec::new();
        loop {
            match self.receive() {
                Message::Response(response) if response.id == id => return (before, response),
                other => before.push(other),
            }
        }
    }

    fn request<R>(&mut self, params: R::Params) -> Result<R::Result, lsp_server::ResponseError>
    where
        R: lsp::request::Request,
    {
        let (before, response) = self.request_raw(
            R::METHOD,
            serde_json::to_value(params).expect("request params serialize"),
        );
        assert!(before.is_empty(), "{before:?}");
        response.response_result.map(|result| {
            serde_json::from_value(result).expect("the server answers the protocol's result type")
        })
    }

    fn notify(&self, method: &str, params: serde_json::Value) {
        self.connection
            .sender
            .send(Notification::new(method.to_string(), params).into())
            .expect("the server listens");
    }

    /// Everything the server sent since the last request: a request no
    /// server answers is answered in order after them.
    fn barrier(&mut self) -> Vec<Message> {
        let (before, response) = self.request_raw("acvus-test/barrier", json!(null));
        let Err(error) = response.response_result else {
            panic!("an unknown request is answered with an error");
        };
        assert_eq!(error.code, lsp_server::ErrorCode::MethodNotFound as i32);
        before
    }

    fn publications(&mut self) -> Vec<lsp::PublishDiagnosticsParams> {
        published(self.barrier())
    }

    fn open(&self, uri: &lsp::Uri, text: &str) {
        self.notify(
            lsp::notification::DidOpenTextDocument::METHOD,
            json!({ "textDocument": { "uri": uri, "languageId": "acvus", "version": 1, "text": text } }),
        );
    }

    fn change(&self, uri: &lsp::Uri, text: &str) {
        self.notify(
            lsp::notification::DidChangeTextDocument::METHOD,
            json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": [{ "text": "stale" }, { "text": text }],
            }),
        );
    }

    fn folders_changed(&self, added: &[&Path], removed: &[&Path]) {
        self.notify(
            lsp::notification::DidChangeWorkspaceFolders::METHOD,
            json!({ "event": { "added": folders(added), "removed": folders(removed) } }),
        );
    }

    fn watched(&self, uri: &lsp::Uri) {
        self.notify(
            lsp::notification::DidChangeWatchedFiles::METHOD,
            json!({ "changes": [{ "uri": uri, "type": lsp::FileChangeType::CHANGED }] }),
        );
    }

    fn shut_down(mut self) {
        let (before, response) = self.request_raw(lsp::request::Shutdown::METHOD, json!(null));
        assert!(before.is_empty(), "{before:?}");
        assert!(response.response_result.is_ok(), "{response:?}");
        self.notify(lsp::notification::Exit::METHOD, json!(null));
        let served = self
            .server
            .join()
            .expect("the server thread does not panic");
        assert!(served.is_ok(), "{served:?}");
    }
}

fn published(messages: Vec<Message>) -> Vec<lsp::PublishDiagnosticsParams> {
    messages
        .into_iter()
        .filter_map(|message| match message {
            Message::Notification(notification)
                if notification.method == lsp::notification::PublishDiagnostics::METHOD =>
            {
                Some(serde_json::from_value(notification.params).expect("a publication parses"))
            }
            _ => None,
        })
        .collect()
}

fn offer(root: &Path) -> Offer<'_> {
    Offer {
        root: Some(root),
        folders: &[],
        encodings: Some(&["utf-8", "utf-16"]),
        watches: false,
    }
}

fn position_encoding(response: Response) -> String {
    let result = response
        .response_result
        .expect("the server accepts the initialize request");
    let result: lsp::InitializeResult =
        serde_json::from_value(result).expect("an InitializeResult");
    result
        .capabilities
        .position_encoding
        .expect("the server states its position encoding")
        .as_str()
        .to_string()
}

#[test]
fn a_client_offering_utf8_is_answered_in_utf8() {
    let f = Fixture::new("string");
    let (mut client, response) = Client::start(&offer(f.root()));
    assert_eq!(position_encoding(response), "utf-8");
    client.notify(lsp::notification::Initialized::METHOD, json!({}));
    assert_eq!(client.publications(), []);
    client.shut_down();
}

#[test]
fn a_client_offering_no_encoding_is_answered_in_utf16_under_its_first_folder() {
    let f = Fixture::new("bogus");
    let (mut client, response) = Client::start(&Offer {
        root: None,
        folders: &[f.root()],
        encodings: None,
        watches: false,
    });
    assert_eq!(position_encoding(response), "utf-16");
    client.notify(lsp::notification::Initialized::METHOD, json!({}));
    let published = client.publications();
    assert_eq!(
        published.iter().map(|p| p.uri.clone()).collect::<Vec<_>>(),
        [f.uri("env.txt")]
    );
    client.shut_down();
}

#[test]
fn an_initialize_without_a_root_is_refused_and_ends_the_server() {
    let (client, response) = Client::start(&Offer {
        root: None,
        folders: &[],
        encodings: Some(&["utf-8"]),
        watches: false,
    });
    let Err(error) = response.response_result else {
        panic!("the server accepted a client without a root");
    };
    assert_eq!(error.code, lsp_server::ErrorCode::InvalidParams as i32);
    let served = client
        .server
        .join()
        .expect("the server thread does not panic");
    assert!(matches!(served, Err(ServeError::NoRoot)), "{served:?}");
}

#[test]
fn a_document_is_published_on_its_error_cleared_on_its_fix_and_not_repeated() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let main = f.uri("main.acvt");

    client.open(&main, "{{ @x + 1 }}");
    let published = client.publications();
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(published[0].uri, main);
    let [diagnostic] = published[0].diagnostics.as_slice() else {
        panic!("one type error: {published:?}");
    };
    assert_eq!(diagnostic.range, range(at(0, 3), at(0, 9)));
    assert_eq!(diagnostic.severity, Some(lsp::DiagnosticSeverity::ERROR));
    assert_eq!(diagnostic.source.as_deref(), Some("acvus"));

    client.change(&main, "{{ @x + 1 }}\n");
    assert_eq!(client.publications(), []);

    client.change(&main, "{{ @x }}");
    let published = client.publications();
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(published[0].uri, main);
    assert_eq!(published[0].diagnostics, []);

    client.change(&main, "{{ @x }}{{ @x }}");
    assert_eq!(client.publications(), []);
    client.shut_down();
}

#[test]
fn a_host_refusal_on_a_file_that_is_no_document_is_cleared_when_it_is_fixed() {
    let f = Fixture::new("bogus");
    let (mut client, sent) = Client::initialized(&Offer {
        watches: true,
        ..offer(f.root())
    });
    let registrations: Vec<&Request> = sent
        .iter()
        .filter_map(|message| match message {
            Message::Request(request) => Some(request),
            _ => None,
        })
        .collect();
    assert_eq!(registrations.len(), 1, "{sent:?}");
    assert_eq!(
        registrations[0].method,
        lsp::request::RegisterCapability::METHOD
    );
    let registration: lsp::RegistrationParams =
        serde_json::from_value(registrations[0].params.clone()).expect("registration params");
    assert_eq!(
        registration.registrations[0].method,
        lsp::notification::DidChangeWatchedFiles::METHOD
    );
    client
        .connection
        .sender
        .send(Response::new_ok(registrations[0].id.clone(), ()).into())
        .expect("the server listens");

    let env = f.uri("env.txt");
    let published = self::published(sent);
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(published[0].uri, env);
    let [refusal] = published[0].diagnostics.as_slice() else {
        panic!("one refusal: {published:?}");
    };
    assert_eq!(refusal.message, "unknown environment `bogus`");
    assert_eq!(refusal.range, range(at(0, 0), at(0, 0)));

    f.write("env.txt", "string");
    client.watched(&env);
    let published = client.publications();
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(
        (&published[0].uri, published[0].diagnostics.len()),
        (&env, 0)
    );

    f.write("env.txt", "bogus");
    client.watched(&env);
    let published = client.publications();
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(
        (&published[0].uri, published[0].diagnostics.len()),
        (&env, 1)
    );

    client.open(&env, "string");
    let published = client.publications();
    assert_eq!(published.len(), 1, "{published:?}");
    assert_eq!(
        (&published[0].uri, published[0].diagnostics.len()),
        (&env, 0)
    );
    client.shut_down();
}

#[test]
fn a_name_after_non_ascii_text_is_hovered_in_either_encoding() {
    let text = "let t = \"한🙂\"; t";
    let name = text.rfind('t').expect("the text ends in `t`");
    for (encoding, character) in [("utf-8", 19), ("utf-16", 15)] {
        let f = Fixture::new("string");
        let (mut client, sent) = Client::initialized(&Offer {
            encodings: Some(&[encoding]),
            ..offer(f.root())
        });
        assert_eq!(published(sent), []);
        let s = f.uri("s.acvus");
        client.open(&s, text);
        assert_eq!(client.publications(), [], "{encoding}");

        let hover = client
            .request::<lsp::request::HoverRequest>(lsp::HoverParams {
                text_document_position_params: lsp::TextDocumentPositionParams {
                    text_document: lsp::TextDocumentIdentifier { uri: s.clone() },
                    position: at(0, character),
                },
                work_done_progress_params: Default::default(),
            })
            .expect("hover answers")
            .unwrap_or_else(|| panic!("`t` is hovered in {encoding}"));
        let hovered = hover.range.expect("a hover states its range");
        assert_eq!(
            hovered,
            range(at(0, character), at(0, character + 1)),
            "{encoding}"
        );
        let span = offset_in(text, hovered.start, encoding)..offset_in(text, hovered.end, encoding);
        assert_eq!((span.start, &text[span]), (name, "t"), "{encoding}");
        let lsp::HoverContents::Markup(contents) = hover.contents else {
            panic!("a hover is markup");
        };
        assert_eq!(contents.kind, lsp::MarkupKind::Markdown);
        assert!(
            contents.value.starts_with("```acvus\n"),
            "{}",
            contents.value
        );
        client.shut_down();
    }
}

fn hover_at(client: &mut Client, uri: &lsp::Uri, position: lsp::Position) -> Option<lsp::Hover> {
    client
        .request::<lsp::request::HoverRequest>(lsp::HoverParams {
            text_document_position_params: position_params(uri, position),
            work_done_progress_params: Default::default(),
        })
        .expect("hover answers")
}

#[test]
fn a_document_changed_on_disk_unannounced_is_answered_over_the_text_it_was_parsed_from() {
    let f = Fixture::new("string");
    f.write("s.acvus", "let t = 1; t");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    f.write("s.acvus", "");

    let s = f.uri("s.acvus");
    let hover = hover_at(&mut client, &s, at(0, 11)).expect("`t` is hovered");
    assert_eq!(hover.range, Some(range(at(0, 11), at(0, 12))));

    client.watched(&s);
    assert_eq!(client.publications(), []);
    assert_eq!(hover_at(&mut client, &s, at(0, 11)), None);
    client.shut_down();
}

fn position_params(uri: &lsp::Uri, position: lsp::Position) -> lsp::TextDocumentPositionParams {
    lsp::TextDocumentPositionParams {
        text_document: lsp::TextDocumentIdentifier { uri: uri.clone() },
        position,
    }
}

#[test]
fn a_completion_replaces_the_identifier_the_cursor_is_in() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let main = f.uri("main.acvt");
    client.open(&main, "{{ grzz }}");
    assert_eq!(client.publications().len(), 1, "`grzz` names nothing");

    let answer = client
        .request::<lsp::request::Completion>(lsp::CompletionParams {
            text_document_position: position_params(&main, at(0, 5)),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: None,
        })
        .expect("completion answers")
        .expect("the cursor is in code");
    let lsp::CompletionResponse::Array(items) = answer else {
        panic!("completion answers a list");
    };
    assert!(items.iter().any(|item| item.label == "greet"), "{items:?}");
    for item in &items {
        let Some(lsp::CompletionTextEdit::Edit(edit)) = &item.text_edit else {
            panic!("an item replaces a range: {item:?}");
        };
        assert_eq!(edit.range, range(at(0, 3), at(0, 7)), "{item:?}");
        assert_eq!(item.filter_text.as_ref(), Some(&edit.new_text));
    }
    let order: Vec<&String> = items
        .iter()
        .map(|item| item.sort_text.as_ref().expect("an item has a sort text"))
        .collect();
    assert!(order.is_sorted(), "{order:?}");
    client.shut_down();
}

fn completions_of_main_tag(
    client: &mut Client,
    main: &lsp::Uri,
    typed: char,
) -> Vec<lsp::CompletionItem> {
    client.open(main, &format!("{{{{ {typed} }}}}"));
    assert_eq!(client.publications().len(), 1, "`{typed}` names nothing");
    let answer = client
        .request::<lsp::request::Completion>(lsp::CompletionParams {
            text_document_position: position_params(main, at(0, 4)),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: None,
        })
        .expect("completion answers")
        .expect("the cursor is in code");
    let lsp::CompletionResponse::Array(items) = answer else {
        panic!("completion answers a list");
    };
    items
}

fn label_detail(
    items: &[lsp::CompletionItem],
    label: &str,
) -> Option<lsp::CompletionItemLabelDetails> {
    items
        .iter()
        .find(|item| item.label == label)
        .unwrap_or_else(|| panic!("`{label}` is offered: {items:?}"))
        .label_details
        .clone()
}

fn call_detail(detail: &str) -> Option<lsp::CompletionItemLabelDetails> {
    Some(lsp::CompletionItemLabelDetails {
        detail: Some(detail.to_string()),
        description: None,
    })
}

#[test]
fn a_function_item_shows_the_call_of_its_first_declaration_beside_its_label() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let main = f.uri("main.acvt");

    let items = completions_of_main_tag(&mut client, &main, 'p');
    assert_eq!(
        label_detail(&items, "pad"),
        call_detail("(width: i64, String)")
    );
    let items = completions_of_main_tag(&mut client, &main, 'g');
    assert_eq!(label_detail(&items, "greet"), call_detail("()"));
    let items = completions_of_main_tag(&mut client, &main, 'i');
    assert_eq!(label_detail(&items, "if"), None);
    client.shut_down();
}

#[test]
fn a_rename_is_prepared_refused_in_its_words_and_answered_with_edits() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let main = f.uri("main.acvt");
    client.open(&main, "% let v = @x\n{{ v }}");
    assert_eq!(client.publications(), []);

    let refused = client
        .request::<lsp::request::PrepareRenameRequest>(position_params(&main, at(0, 11)))
        .expect_err("a context is not renamed");
    assert_eq!(refused.code, lsp_server::ErrorCode::RequestFailed as i32);
    assert_eq!(refused.message, RenameRefusal::Context.to_string());

    let prepared = client
        .request::<lsp::request::PrepareRenameRequest>(position_params(&main, at(1, 3)))
        .expect("a local is renamed");
    assert_eq!(
        prepared,
        Some(lsp::PrepareRenameResponse::Range(range(at(1, 3), at(1, 4))))
    );

    let renamed = client
        .request::<lsp::request::Rename>(lsp::RenameParams {
            text_document_position: position_params(&main, at(1, 3)),
            new_name: "w".to_string(),
            work_done_progress_params: Default::default(),
        })
        .expect("a local is renamed")
        .expect("a rename answers edits");
    let to_w = |start: lsp::Position, end: lsp::Position| lsp::TextEdit {
        range: range(start, end),
        new_text: "w".to_string(),
    };
    assert_eq!(
        renamed.changes,
        Some(HashMap::from([(
            main.clone(),
            vec![to_w(at(0, 6), at(0, 7)), to_w(at(1, 3), at(1, 4))]
        )]))
    );
    client.shut_down();
}

#[test]
fn a_function_is_defined_by_the_document_that_is_its_body() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let main = f.uri("main.acvt");
    client.open(&main, "{{ greet() }}");
    assert_eq!(client.publications(), []);

    let definition = client
        .request::<lsp::request::GotoDefinition>(lsp::GotoDefinitionParams {
            text_document_position_params: position_params(&main, at(0, 4)),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        })
        .expect("definition answers");
    assert_eq!(
        definition,
        Some(lsp::GotoDefinitionResponse::Scalar(lsp::Location {
            uri: f.uri("greet.acvt"),
            range: range(at(0, 0), at(0, 0)),
        }))
    );
    client.shut_down();
}

#[test]
fn a_request_on_a_uri_that_is_no_file_is_an_error() {
    let f = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);
    let untitled: lsp::Uri = "untitled:Untitled-1".parse().expect("a URI");
    let refused = client
        .request::<lsp::request::HoverRequest>(lsp::HoverParams {
            text_document_position_params: position_params(&untitled, at(0, 0)),
            work_done_progress_params: Default::default(),
        })
        .expect_err("an untitled buffer has no path");
    assert_eq!(refused.code, lsp_server::ErrorCode::InvalidParams as i32);
    client.shut_down();
}

#[test]
fn definition_in_a_host_file_answers_the_linked_uri() {
    let f = Fixture::new("string");
    f.write("manifest.txt", "see greet.acvt");
    let (mut client, sent) = Client::initialized(&offer(f.root()));
    assert_eq!(published(sent), []);

    let definition = client
        .request::<lsp::request::GotoDefinition>(lsp::GotoDefinitionParams {
            text_document_position_params: position_params(&f.uri("manifest.txt"), at(0, 6)),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        })
        .expect("definition answers");
    assert_eq!(
        definition,
        Some(lsp::GotoDefinitionResponse::Scalar(lsp::Location {
            uri: f.uri("greet.acvt"),
            range: range(at(0, 0), at(0, 0)),
        }))
    );
    client.shut_down();
}

fn definition_at(
    client: &mut Client,
    uri: &lsp::Uri,
    position: lsp::Position,
) -> Option<lsp::GotoDefinitionResponse> {
    client
        .request::<lsp::request::GotoDefinition>(lsp::GotoDefinitionParams {
            text_document_position_params: position_params(uri, position),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        })
        .expect("definition answers")
}

fn start_of(uri: lsp::Uri) -> Option<lsp::GotoDefinitionResponse> {
    Some(lsp::GotoDefinitionResponse::Scalar(lsp::Location {
        uri,
        range: range(at(0, 0), at(0, 0)),
    }))
}

#[derive(Debug, PartialEq, Eq)]
struct Published {
    uri: lsp::Uri,
    diagnostics: usize,
}

fn counted(published: Vec<lsp::PublishDiagnosticsParams>) -> Vec<Published> {
    let mut counted: Vec<Published> = published
        .into_iter()
        .map(|p| Published {
            uri: p.uri,
            diagnostics: p.diagnostics.len(),
        })
        .collect();
    counted.sort_by(|a, b| a.uri.as_str().cmp(b.uri.as_str()));
    counted
}

fn two_projects_with_a_type_error() -> [Fixture; 2] {
    let projects = [Fixture::new("string"), Fixture::new("string")];
    for project in &projects {
        project.write("s.acvus", "@x + 1");
    }
    projects
}

fn folders_offer<'a>(folders: &'a [&'a Path]) -> Offer<'a> {
    Offer {
        root: None,
        folders,
        encodings: Some(&["utf-8"]),
        watches: false,
    }
}

#[test]
fn every_folder_is_served_and_answers_in_its_own_workspace() {
    let [a, b] = two_projects_with_a_type_error();
    for project in [&a, &b] {
        project.write("main.acvt", "{{ greet() }}");
    }
    let (mut client, sent) = Client::initialized(&folders_offer(&[a.root(), b.root()]));
    let published = counted(published(sent));
    let uris: Vec<&lsp::Uri> = published.iter().map(|p| &p.uri).collect();
    let mut expected = [a.uri("s.acvus"), b.uri("s.acvus")];
    expected.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    assert_eq!(uris, expected.iter().collect::<Vec<_>>());
    assert!(published.iter().all(|p| p.diagnostics > 0), "{published:?}");

    for project in [&a, &b] {
        assert_eq!(
            definition_at(&mut client, &project.uri("main.acvt"), at(0, 4)),
            start_of(project.uri("greet.acvt"))
        );
    }
    client.shut_down();
}

#[test]
fn a_removed_folder_has_its_paths_published_empty_and_answers_nothing() {
    let [a, b] = two_projects_with_a_type_error();
    let (mut client, sent) = Client::initialized(&folders_offer(&[a.root(), b.root()]));
    assert_eq!(counted(published(sent)).len(), 2);
    let a_s = a.uri("s.acvus");
    assert!(hover_at(&mut client, &a_s, at(0, 1)).is_some());

    client.folders_changed(&[], &[b.root()]);
    assert_eq!(
        counted(client.publications()),
        [Published {
            uri: b.uri("s.acvus"),
            diagnostics: 0,
        }]
    );
    assert_eq!(hover_at(&mut client, &b.uri("s.acvus"), at(0, 1)), None);
    assert!(hover_at(&mut client, &a_s, at(0, 1)).is_some());
    client.shut_down();
}

#[test]
fn an_added_folder_is_seeded_with_the_buffers_open_under_it() {
    let a = Fixture::new("string");
    let b = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&folders_offer(&[a.root()]));
    assert_eq!(published(sent), []);

    let s = b.uri("s.acvus");
    client.open(&s, "@x + 1");
    assert_eq!(client.publications(), [], "b is under no root yet");

    client.folders_changed(&[b.root()], &[]);
    assert_eq!(
        counted(client.publications()),
        [Published {
            uri: s,
            diagnostics: 1,
        }]
    );
    client.shut_down();
}

#[test]
fn a_path_under_no_root_is_answered_with_nothing() {
    let a = Fixture::new("string");
    let elsewhere = Fixture::new("string");
    let (mut client, sent) = Client::initialized(&folders_offer(&[a.root()]));
    assert_eq!(published(sent), []);

    let s = elsewhere.uri("s.acvus");
    assert_eq!(hover_at(&mut client, &s, at(0, 1)), None);
    assert_eq!(definition_at(&mut client, &s, at(0, 1)), None);
    client.shut_down();
}

fn hovered_type(client: &mut Client, uri: &lsp::Uri, position: lsp::Position) -> String {
    let hover = hover_at(client, uri, position).expect("the name is hovered");
    let lsp::HoverContents::Markup(contents) = hover.contents else {
        panic!("a hover is markup");
    };
    contents.value
}

#[test]
fn the_innermost_root_answers_and_a_diagnostic_both_report_is_published_once() {
    let outer = Fixture::new("int");
    outer.write("main.acvt", "hello");
    let inner = outer.nested("inner", "string");
    outer.write("extra.txt", "inner/main.acvt");
    let source = "% let v = @x\n{{ nope }}";
    inner.write("main.acvt", source);
    let (mut client, sent) = Client::initialized(&folders_offer(&[outer.root(), inner.root()]));
    assert_eq!(
        counted(published(sent)),
        [Published {
            uri: inner.uri("main.acvt"),
            diagnostics: 1,
        }]
    );

    let main = inner.uri("main.acvt");
    let x = at(0, 11);
    let in_inner = hovered_type(&mut client, &main, x);
    assert!(in_inner.contains("String"), "{in_inner}");

    client.folders_changed(&[], &[inner.root()]);
    assert_eq!(
        client.publications(),
        [],
        "the outer folder reports the same error"
    );
    let in_outer = hovered_type(&mut client, &main, x);
    assert!(in_outer.contains("i64"), "{in_outer}");
    client.shut_down();
}

#[test]
fn workspace_folders_are_preferred_over_the_root_uri_and_their_changes_advertised() {
    let rooted = Fixture::new("string");
    let folder = Fixture::new("bogus");
    let (mut client, response) = Client::start(&Offer {
        root: Some(rooted.root()),
        ..folders_offer(&[folder.root()])
    });
    let result: lsp::InitializeResult = serde_json::from_value(
        response
            .response_result
            .expect("the server accepts the initialize request"),
    )
    .expect("an InitializeResult");
    assert_eq!(
        result.capabilities.workspace,
        Some(lsp::WorkspaceServerCapabilities {
            workspace_folders: Some(lsp::WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(lsp::OneOf::Left(true)),
            }),
            file_operations: None,
        })
    );

    client.notify(lsp::notification::Initialized::METHOD, json!({}));
    assert_eq!(
        counted(client.publications()),
        [Published {
            uri: folder.uri("env.txt"),
            diagnostics: 1,
        }]
    );
    assert_eq!(
        hover_at(&mut client, &rooted.uri("main.acvt"), at(0, 4)),
        None
    );
    client.shut_down();
}
