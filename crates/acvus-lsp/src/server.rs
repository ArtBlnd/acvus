//! The Language Server Protocol over one connection, for a workspace of any
//! host (RFC-0086).

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use acvus_ast::Span;
use acvus_utils::Interner;
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types as lsp;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use serde::Serialize;
use serde::de::DeserializeOwned;
use url::Url;

use crate::position::{Encoding, LineIndex, Unplaceable};
use crate::session::{CallShape, CompletionKind, Completions, LspError};
use crate::workspace::{Host, Location, Workspace};

#[derive(Debug)]
pub enum ServeError {
    Disconnected,
    MalformedInitialize(serde_json::Error),
    NoRoot,
    RootNotAFilePath(NotAFilePath),
    ExitBeforeShutdown,
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServeError::Disconnected => write!(f, "the client closed the connection before `exit`"),
            ServeError::MalformedInitialize(error) => {
                write!(f, "the initialize request does not parse: {error}")
            }
            ServeError::NoRoot => write!(
                f,
                "the client gave neither a root URI nor a workspace folder to serve"
            ),
            ServeError::RootNotAFilePath(error) => write!(f, "the root {error}"),
            ServeError::ExitBeforeShutdown => write!(f, "the client sent `exit` before `shutdown`"),
        }
    }
}

impl std::error::Error for ServeError {}

#[derive(Debug)]
pub enum NotAFilePath {
    Unparsed(String, url::ParseError),
    NotFileScheme(String),
    NoLocalPath(String),
}

impl fmt::Display for NotAFilePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotAFilePath::Unparsed(uri, error) => write!(f, "`{uri}` is no URL: {error}"),
            NotAFilePath::NotFileScheme(uri) => write!(f, "`{uri}` is no `file:` URI"),
            NotAFilePath::NoLocalPath(uri) => write!(f, "`{uri}` names no local path"),
        }
    }
}

impl std::error::Error for NotAFilePath {}

#[derive(Debug)]
enum NoUri {
    NotAbsolute(PathBuf),
    Unparsed(PathBuf, <lsp::Uri as FromStr>::Err),
}

impl fmt::Display for NoUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoUri::NotAbsolute(path) => write!(f, "{} is no absolute path", path.display()),
            NoUri::Unparsed(path, error) => {
                write!(
                    f,
                    "the file URI of {} does not parse: {error}",
                    path.display()
                )
            }
        }
    }
}

fn path_of(uri: &lsp::Uri) -> Result<PathBuf, NotAFilePath> {
    let written = uri.as_str().to_string();
    let url = match Url::parse(uri.as_str()) {
        Ok(url) => url,
        Err(error) => return Err(NotAFilePath::Unparsed(written, error)),
    };
    if url.scheme() != "file" {
        return Err(NotAFilePath::NotFileScheme(written));
    }
    url.to_file_path()
        .map_err(|()| NotAFilePath::NoLocalPath(written))
}

fn uri_of(path: &Path) -> Result<lsp::Uri, NoUri> {
    let url = Url::from_file_path(path).map_err(|()| NoUri::NotAbsolute(path.to_path_buf()))?;
    url.as_str()
        .parse()
        .map_err(|error| NoUri::Unparsed(path.to_path_buf(), error))
}

/// `make_host` makes the host of each root the client serves (RFC-0086
/// rule 1).
pub fn serve<H, F>(connection: Connection, mut make_host: F) -> Result<(), ServeError>
where
    H: Host,
    F: FnMut(PathBuf) -> H,
{
    let (id, params) = initialize_request(&connection)?;
    let roots = match roots_of(&params) {
        Ok(roots) => roots,
        Err(error) => {
            send(
                &connection,
                Response::new_err(id, ErrorCode::InvalidParams as i32, error.to_string()),
            )?;
            return Err(error);
        }
    };
    let encoding = Encoding::negotiate(
        params
            .capabilities
            .general
            .as_ref()
            .and_then(|general| general.position_encodings.as_deref()),
    );
    let watches_files = params
        .capabilities
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.did_change_watched_files)
        .and_then(|watched| watched.dynamic_registration)
        == Some(true);
    let interner = Interner::new();
    let workspaces = roots
        .into_iter()
        .map(|root| {
            let workspace = Workspace::new(&interner, make_host(root.clone()));
            (root, workspace)
        })
        .collect();
    send(
        &connection,
        Response::new_ok(
            id,
            lsp::InitializeResult {
                capabilities: capabilities(encoding),
                server_info: Some(lsp::ServerInfo {
                    name: "acvus".to_string(),
                    version: Some(env!("CARGO_PKG_VERSION").to_string()),
                }),
            },
        ),
    )?;
    Server {
        connection: &connection,
        make_host,
        interner,
        workspaces,
        open_buffers: BTreeMap::new(),
        encoding,
        watches_files,
        published: BTreeMap::new(),
        shut_down: false,
    }
    .run()
}

/// Before `initialize` the protocol has a request answered with
/// `ServerNotInitialized` and a notification other than `exit` dropped.
fn initialize_request(
    connection: &Connection,
) -> Result<(RequestId, lsp::InitializeParams), ServeError> {
    loop {
        match receive(connection)? {
            Message::Request(request) if request.method == lsp::request::Initialize::METHOD => {
                return match serde_json::from_value(request.params) {
                    Ok(params) => Ok((request.id, params)),
                    Err(error) => {
                        send(
                            connection,
                            Response::new_err(
                                request.id,
                                ErrorCode::InvalidParams as i32,
                                error.to_string(),
                            ),
                        )?;
                        Err(ServeError::MalformedInitialize(error))
                    }
                };
            }
            Message::Request(request) => send(
                connection,
                Response::new_err(
                    request.id,
                    ErrorCode::ServerNotInitialized as i32,
                    format!("`{}` before `initialize`", request.method),
                ),
            )?,
            Message::Notification(notification)
                if notification.method == lsp::notification::Exit::METHOD =>
            {
                return Err(ServeError::ExitBeforeShutdown);
            }
            Message::Notification(_) | Message::Response(_) => {}
        }
    }
}

fn roots_of(params: &lsp::InitializeParams) -> Result<Vec<PathBuf>, ServeError> {
    #[allow(deprecated)]
    let root_uri = params.root_uri.as_ref();
    let uris: Vec<&lsp::Uri> = match params.workspace_folders.as_deref() {
        Some(folders) if !folders.is_empty() => folders.iter().map(|folder| &folder.uri).collect(),
        Some(_) | None => vec![root_uri.ok_or(ServeError::NoRoot)?],
    };
    uris.into_iter()
        .map(|uri| path_of(uri).map_err(ServeError::RootNotAFilePath))
        .collect()
}

fn capabilities(encoding: Encoding) -> lsp::ServerCapabilities {
    lsp::ServerCapabilities {
        position_encoding: Some(encoding.kind()),
        text_document_sync: Some(lsp::TextDocumentSyncCapability::Options(
            lsp::TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(lsp::TextDocumentSyncKind::FULL),
                ..lsp::TextDocumentSyncOptions::default()
            },
        )),
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp::OneOf::Left(true)),
        references_provider: Some(lsp::OneOf::Left(true)),
        rename_provider: Some(lsp::OneOf::Right(lsp::RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: lsp::WorkDoneProgressOptions::default(),
        })),
        workspace: Some(lsp::WorkspaceServerCapabilities {
            workspace_folders: Some(lsp::WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(lsp::OneOf::Left(true)),
            }),
            file_operations: None,
        }),
        completion_provider: Some(lsp::CompletionOptions {
            trigger_characters: Some(["$", "@", ".", ":"].map(String::from).to_vec()),
            ..lsp::CompletionOptions::default()
        }),
        ..lsp::ServerCapabilities::default()
    }
}

fn receive(connection: &Connection) -> Result<Message, ServeError> {
    connection
        .receiver
        .recv()
        .map_err(|_| ServeError::Disconnected)
}

fn send<M>(connection: &Connection, message: M) -> Result<(), ServeError>
where
    M: Into<Message>,
{
    connection
        .sender
        .send(message.into())
        .map_err(|_| ServeError::Disconnected)
}

fn failed<E>(code: ErrorCode, error: E) -> lsp_server::ResponseError
where
    E: fmt::Display,
{
    lsp_server::ResponseError {
        code: code as i32,
        message: error.to_string(),
        data: None,
    }
}

/// Where RFC-0086 rule 4 places a diagnostic without a span.
const START_OF_FILE: lsp::Range = lsp::Range {
    start: lsp::Position {
        line: 0,
        character: 0,
    },
    end: lsp::Position {
        line: 0,
        character: 0,
    },
};

const WATCH_REGISTRATION: &str = "acvus/watched-files";

struct Server<'c, H, F>
where
    H: Host,
    F: FnMut(PathBuf) -> H,
{
    connection: &'c Connection,
    make_host: F,
    interner: Interner,
    workspaces: BTreeMap<PathBuf, Workspace<H>>,
    open_buffers: BTreeMap<PathBuf, String>,
    encoding: Encoding,
    watches_files: bool,
    published: BTreeMap<PathBuf, (lsp::Uri, Vec<lsp::Diagnostic>)>,
    shut_down: bool,
}

struct Reported<'w, H>
where
    H: Host,
{
    error: LspError,
    by: &'w Workspace<H>,
}

type Answer<T> = Result<T, lsp_server::ResponseError>;

struct RequestedDocument<'w, H>
where
    H: Host,
{
    workspace: &'w Workspace<H>,
    path: PathBuf,
    text: Cow<'w, str>,
}

impl<H, F> Server<'_, H, F>
where
    H: Host,
    F: FnMut(PathBuf) -> H,
{
    fn run(mut self) -> Result<(), ServeError> {
        loop {
            match receive(self.connection)? {
                Message::Request(request) => {
                    let response = self.request(request);
                    send(self.connection, response)?;
                }
                Message::Notification(notification)
                    if notification.method == lsp::notification::Exit::METHOD =>
                {
                    return match self.shut_down {
                        true => Ok(()),
                        false => Err(ServeError::ExitBeforeShutdown),
                    };
                }
                Message::Notification(notification) => self.notification(notification)?,
                Message::Response(response) => self.response(response)?,
            }
        }
    }

    fn request(&mut self, request: Request) -> Response {
        use lsp::request::*;

        if self.shut_down {
            return Response::new_err(
                request.id,
                ErrorCode::InvalidRequest as i32,
                format!("`{}` after `shutdown`", request.method),
            );
        }
        match request.method.as_str() {
            Shutdown::METHOD => {
                self.shut_down = true;
                Response::new_ok(request.id, ())
            }
            Initialize::METHOD => Response::new_err(
                request.id,
                ErrorCode::InvalidRequest as i32,
                "`initialize` was already answered".to_string(),
            ),
            HoverRequest::METHOD => self.answer::<HoverRequest, _>(request, Self::hover),
            GotoDefinition::METHOD => self.answer::<GotoDefinition, _>(request, Self::definition),
            References::METHOD => self.answer::<References, _>(request, Self::references),
            PrepareRenameRequest::METHOD => {
                self.answer::<PrepareRenameRequest, _>(request, Self::prepare_rename)
            }
            Rename::METHOD => self.answer::<Rename, _>(request, Self::rename),
            Completion::METHOD => self.answer::<Completion, _>(request, Self::completion),
            method => Response::new_err(
                request.id,
                ErrorCode::MethodNotFound as i32,
                format!("acvus does not answer `{method}`"),
            ),
        }
    }

    fn answer<R, A>(&self, request: Request, answer: A) -> Response
    where
        R: lsp::request::Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
        A: FnOnce(&Self, R::Params) -> Answer<R::Result>,
    {
        let params = match serde_json::from_value(request.params) {
            Ok(params) => params,
            Err(error) => {
                return Response::new_err(
                    request.id,
                    ErrorCode::InvalidParams as i32,
                    error.to_string(),
                );
            }
        };
        match answer(self, params) {
            Ok(result) => Response::new_ok(request.id, result),
            Err(error) => Response {
                id: request.id,
                response_result: Err(error),
            },
        }
    }

    fn document(&self, uri: &lsp::Uri) -> Answer<Option<RequestedDocument<'_, H>>> {
        let path = path_of(uri).map_err(|error| failed(ErrorCode::InvalidParams, error))?;
        let Some(workspace) = self.serving(&path) else {
            return Ok(None);
        };
        let text = text(workspace, &path)?;
        Ok(Some(RequestedDocument {
            workspace,
            path,
            text,
        }))
    }

    /// Enforces RFC-0086 rule 7.
    fn serving(&self, path: &Path) -> Option<&Workspace<H>> {
        self.workspaces
            .iter()
            .filter(|(root, _)| path.starts_with(root))
            .max_by_key(|(root, _)| root.components().count())
            .map(|(_, workspace)| workspace)
    }

    fn holding<'s>(&'s mut self, path: &'s Path) -> impl Iterator<Item = &'s mut Workspace<H>> {
        self.workspaces
            .iter_mut()
            .filter(move |(root, _)| path.starts_with(root))
            .map(|(_, workspace)| workspace)
    }

    fn location(&self, workspace: &Workspace<H>, location: &Location) -> Answer<lsp::Location> {
        let uri =
            uri_of(&location.path).map_err(|error| failed(ErrorCode::RequestFailed, error))?;
        let text = text(workspace, &location.path)?;
        Ok(lsp::Location {
            uri,
            range: placed_location(
                &LineIndex::new(&text, self.encoding),
                &location.path,
                location.span,
            )?,
        })
    }

    fn hover(&self, params: lsp::HoverParams) -> Answer<Option<lsp::Hover>> {
        let at = params.text_document_position_params;
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let index = LineIndex::new(&text, self.encoding);
        Ok(workspace
            .hover(&path, index.offset(at.position))
            .map(|hover| lsp::Hover {
                contents: lsp::HoverContents::Markup(lsp::MarkupContent {
                    kind: lsp::MarkupKind::Markdown,
                    value: format!("```acvus\n{}\n```", hover.ty),
                }),
                range: Some(placed_in_document(&index, hover.span)),
            }))
    }

    fn definition(
        &self,
        params: lsp::GotoDefinitionParams,
    ) -> Answer<Option<lsp::GotoDefinitionResponse>> {
        let at = params.text_document_position_params;
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let offset = LineIndex::new(&text, self.encoding).offset(at.position);
        workspace
            .definition(&path, offset)
            .map(|location| {
                self.location(workspace, &location)
                    .map(lsp::GotoDefinitionResponse::Scalar)
            })
            .transpose()
    }

    fn references(&self, params: lsp::ReferenceParams) -> Answer<Option<Vec<lsp::Location>>> {
        let at = params.text_document_position;
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let offset = LineIndex::new(&text, self.encoding).offset(at.position);
        workspace
            .references(&path, offset, params.context.include_declaration)
            .map(|locations| {
                locations
                    .iter()
                    .map(|location| self.location(workspace, location))
                    .collect()
            })
            .transpose()
    }

    fn prepare_rename(
        &self,
        at: lsp::TextDocumentPositionParams,
    ) -> Answer<Option<lsp::PrepareRenameResponse>> {
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let index = LineIndex::new(&text, self.encoding);
        let target = workspace
            .rename_target(&path, index.offset(at.position))
            .map_err(|refusal| failed(ErrorCode::RequestFailed, refusal))?;
        Ok(Some(lsp::PrepareRenameResponse::Range(placed_in_document(
            &index, target,
        ))))
    }

    fn rename(&self, params: lsp::RenameParams) -> Answer<Option<lsp::WorkspaceEdit>> {
        let at = params.text_document_position;
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let index = LineIndex::new(&text, self.encoding);
        let edits = workspace
            .rename(&path, index.offset(at.position), &params.new_name)
            .map_err(|refusal| failed(ErrorCode::RequestFailed, refusal))?;
        let uri = uri_of(&path).map_err(|error| failed(ErrorCode::RequestFailed, error))?;
        let edits = edits
            .into_iter()
            .map(|edit| lsp::TextEdit {
                range: placed_in_document(&index, edit.span),
                new_text: edit.text,
            })
            .collect();
        Ok(Some(lsp::WorkspaceEdit {
            changes: Some([(uri, edits)].into_iter().collect()),
            ..lsp::WorkspaceEdit::default()
        }))
    }

    fn completion(&self, params: lsp::CompletionParams) -> Answer<Option<lsp::CompletionResponse>> {
        let at = params.text_document_position;
        let Some(RequestedDocument {
            workspace,
            path,
            text,
        }) = self.document(&at.text_document.uri)?
        else {
            return Ok(None);
        };
        let index = LineIndex::new(&text, self.encoding);
        let Some(Completions { replaces, items }) =
            workspace.completions(&path, index.offset(at.position))
        else {
            return Ok(None);
        };
        let range = placed_in_document(&index, replaces);
        let width = items.len().to_string().len();
        let items = items
            .into_iter()
            .enumerate()
            .map(|(order, item)| lsp::CompletionItem {
                label_details: item
                    .calls
                    .first()
                    .map(|call| lsp::CompletionItemLabelDetails {
                        detail: Some(call_label_detail(call)),
                        description: None,
                    }),
                label: item.label,
                kind: Some(completion_kind(item.kind)),
                detail: Some(item.detail),
                sort_text: Some(format!("{order:0width$}")),
                filter_text: Some(item.insert_text.clone()),
                text_edit: Some(lsp::CompletionTextEdit::Edit(lsp::TextEdit {
                    range,
                    new_text: item.insert_text,
                })),
                ..lsp::CompletionItem::default()
            })
            .collect();
        Ok(Some(lsp::CompletionResponse::Array(items)))
    }

    fn notification(&mut self, notification: Notification) -> Result<(), ServeError> {
        use lsp::notification::*;

        match notification.method.as_str() {
            Initialized::METHOD => {
                if self.watches_files {
                    self.register_watcher()?;
                }
            }
            DidOpenTextDocument::METHOD => {
                let Some(params) = self.params_or_log::<DidOpenTextDocument>(notification)? else {
                    return Ok(());
                };
                let Some(path) = self.path_or_log(&params.text_document.uri)? else {
                    return Ok(());
                };
                self.set_buffer(path, params.text_document.text);
            }
            DidChangeTextDocument::METHOD => {
                let Some(params) = self.params_or_log::<DidChangeTextDocument>(notification)?
                else {
                    return Ok(());
                };
                let Some(path) = self.path_or_log(&params.text_document.uri)? else {
                    return Ok(());
                };
                let Some(change) = params.content_changes.into_iter().next_back() else {
                    return Ok(());
                };
                if change.range.is_some() {
                    return self.log(format!(
                        "acvus syncs documents in full, and the client sent a ranged change to {}",
                        path.display()
                    ));
                }
                self.set_buffer(path, change.text);
            }
            DidCloseTextDocument::METHOD => {
                let Some(params) = self.params_or_log::<DidCloseTextDocument>(notification)? else {
                    return Ok(());
                };
                let Some(path) = self.path_or_log(&params.text_document.uri)? else {
                    return Ok(());
                };
                self.open_buffers.remove(&path);
                for workspace in self.holding(&path) {
                    workspace.drop_buffer(&path);
                }
            }
            DidChangeWatchedFiles::METHOD => {
                let Some(params) = self.params_or_log::<DidChangeWatchedFiles>(notification)?
                else {
                    return Ok(());
                };
                for change in params.changes {
                    let Some(path) = self.path_or_log(&change.uri)? else {
                        continue;
                    };
                    for workspace in self.holding(&path) {
                        workspace.file_changed(&path);
                    }
                }
            }
            DidChangeWorkspaceFolders::METHOD => {
                let Some(params) = self.params_or_log::<DidChangeWorkspaceFolders>(notification)?
                else {
                    return Ok(());
                };
                for folder in params.event.removed {
                    let Some(root) = self.path_or_log(&folder.uri)? else {
                        continue;
                    };
                    self.workspaces.remove(&root);
                }
                for folder in params.event.added {
                    let Some(root) = self.path_or_log(&folder.uri)? else {
                        continue;
                    };
                    self.serve_root(root);
                }
            }
            _ => return Ok(()),
        }
        self.publish()
    }

    /// Follows RFC-0086 rule 7.
    fn set_buffer(&mut self, path: PathBuf, text: String) {
        self.open_buffers.insert(path.clone(), text.clone());
        for workspace in self.holding(&path) {
            workspace.set_buffer(path.clone(), text.clone());
        }
    }

    fn serve_root(&mut self, root: PathBuf) {
        if self.workspaces.contains_key(&root) {
            return;
        }
        let buffers: Vec<(PathBuf, String)> = self
            .open_buffers
            .iter()
            .filter(|(path, _)| path.starts_with(&root))
            .map(|(path, text)| (path.clone(), text.clone()))
            .collect();
        let host = (self.make_host)(root.clone());
        let workspace = Workspace::with_buffers(&self.interner, host, buffers);
        self.workspaces.insert(root, workspace);
    }

    fn params_or_log<N>(&self, notification: Notification) -> Result<Option<N::Params>, ServeError>
    where
        N: lsp::notification::Notification,
        N::Params: DeserializeOwned,
    {
        match serde_json::from_value(notification.params) {
            Ok(params) => Ok(Some(params)),
            Err(error) => {
                self.log(format!("`{}` does not parse: {error}", notification.method))?;
                Ok(None)
            }
        }
    }

    fn path_or_log(&self, uri: &lsp::Uri) -> Result<Option<PathBuf>, ServeError> {
        match path_of(uri) {
            Ok(path) => Ok(Some(path)),
            Err(error) => {
                self.log(error.to_string())?;
                Ok(None)
            }
        }
    }

    fn response(&self, response: Response) -> Result<(), ServeError> {
        match response.response_result {
            Ok(_) => Ok(()),
            Err(error) => self.log(format!(
                "the client refused request {}: {}",
                response.id, error.message
            )),
        }
    }

    fn log(&self, message: String) -> Result<(), ServeError> {
        send(
            self.connection,
            Notification::new(
                lsp::notification::LogMessage::METHOD.to_string(),
                lsp::LogMessageParams {
                    typ: lsp::MessageType::ERROR,
                    message,
                },
            ),
        )
    }

    fn register_watcher(&self) -> Result<(), ServeError> {
        let options = lsp::DidChangeWatchedFilesRegistrationOptions {
            watchers: vec![lsp::FileSystemWatcher {
                glob_pattern: lsp::GlobPattern::String("**/*".to_string()),
                kind: None,
            }],
        };
        send(
            self.connection,
            Request::new(
                RequestId::from(WATCH_REGISTRATION.to_string()),
                lsp::request::RegisterCapability::METHOD.to_string(),
                lsp::RegistrationParams {
                    registrations: vec![lsp::Registration {
                        id: WATCH_REGISTRATION.to_string(),
                        method: lsp::notification::DidChangeWatchedFiles::METHOD.to_string(),
                        register_options: Some(
                            serde_json::to_value(options)
                                .expect("watcher options hold only strings"),
                        ),
                    }],
                },
            ),
        )
    }

    /// Enforces RFC-0086 rule 7.
    fn publish(&mut self) -> Result<(), ServeError> {
        let mut deepest_first: Vec<(&PathBuf, &Workspace<H>)> = self.workspaces.iter().collect();
        deepest_first.sort_by_key(|(root, _)| Reverse(root.components().count()));
        let mut reported: BTreeMap<PathBuf, Vec<Reported<'_, H>>> = BTreeMap::new();
        for (_, workspace) in deepest_first {
            for (path, errors) in workspace.diagnostics() {
                let held = reported.entry(path).or_default();
                for error in errors {
                    if !held.iter().any(|reported| reported.error == error) {
                        held.push(Reported {
                            error,
                            by: workspace,
                        });
                    }
                }
            }
        }
        let mut now = BTreeMap::new();
        for (path, errors) in reported {
            let uri = match uri_of(&path) {
                Ok(uri) => uri,
                Err(error) => {
                    self.log(format!("diagnostics are not published: {error}"))?;
                    continue;
                }
            };
            let mut diagnostics = Vec::new();
            for by_one in errors.chunk_by(|a, b| std::ptr::eq(a.by, b.by)) {
                let errors: Vec<LspError> = by_one
                    .iter()
                    .map(|reported| reported.error.clone())
                    .collect();
                diagnostics.extend(self.diagnostics(by_one[0].by, &path, &uri, &errors)?);
            }
            now.insert(path, (uri, diagnostics));
        }
        for (path, (uri, diagnostics)) in &now {
            let published = self.published.get(path).map(|(_, published)| published);
            if published != Some(diagnostics) {
                self.send_diagnostics(uri, diagnostics.clone())?;
            }
        }
        for (path, (uri, _)) in &self.published {
            if !now.contains_key(path) {
                self.send_diagnostics(uri, Vec::new())?;
            }
        }
        self.published = now;
        Ok(())
    }

    fn send_diagnostics(
        &self,
        uri: &lsp::Uri,
        diagnostics: Vec<lsp::Diagnostic>,
    ) -> Result<(), ServeError> {
        send(
            self.connection,
            Notification::new(
                lsp::notification::PublishDiagnostics::METHOD.to_string(),
                lsp::PublishDiagnosticsParams {
                    uri: uri.clone(),
                    diagnostics,
                    version: None,
                },
            ),
        )
    }

    fn diagnostics(
        &self,
        workspace: &Workspace<H>,
        path: &Path,
        uri: &lsp::Uri,
        errors: &[LspError],
    ) -> Result<Vec<lsp::Diagnostic>, ServeError> {
        let text = workspace.text(path);
        let index = match &text {
            Ok(text) => Some(LineIndex::new(text, self.encoding)),
            Err(error) => {
                let spanned = errors.iter().any(|error| {
                    error.span().is_some() || error.related.iter().any(|label| label.span.is_some())
                });
                if spanned {
                    self.log(format!(
                        "the diagnostics of {} are placed at its start: {error}",
                        path.display()
                    ))?;
                }
                None
            }
        };
        let mut unplaced = Vec::new();
        let mut range = |span: Option<Span>| {
            let (Some(index), Some(span)) = (&index, span) else {
                return START_OF_FILE;
            };
            index.range(span).unwrap_or_else(|error| {
                unplaced.push(error);
                START_OF_FILE
            })
        };
        let diagnostics = errors
            .iter()
            .map(|error| {
                let related: Vec<lsp::DiagnosticRelatedInformation> = error
                    .related
                    .iter()
                    .filter_map(|label| {
                        Some(lsp::DiagnosticRelatedInformation {
                            location: lsp::Location {
                                uri: uri.clone(),
                                range: range(Some(label.span?)),
                            },
                            message: label.text.clone(),
                        })
                    })
                    .collect();
                lsp::Diagnostic {
                    range: range(error.span()),
                    severity: Some(lsp::DiagnosticSeverity::ERROR),
                    source: Some("acvus".to_string()),
                    message: message_with_notes(error),
                    related_information: (!related.is_empty()).then_some(related),
                    ..lsp::Diagnostic::default()
                }
            })
            .collect();
        for error in unplaced {
            self.log(format!(
                "a diagnostic of {} is placed at its start: {error}",
                path.display()
            ))?;
        }
        Ok(diagnostics)
    }
}

fn text<'w, H>(workspace: &'w Workspace<H>, path: &Path) -> Answer<Cow<'w, str>>
where
    H: Host,
{
    workspace.text(path).map_err(|error| {
        failed(
            ErrorCode::RequestFailed,
            format!("{} does not read: {error}", path.display()),
        )
    })
}

fn message_with_notes(error: &LspError) -> String {
    let notes = error
        .related
        .iter()
        .filter(|label| label.span.is_none())
        .map(|note| format!("\n= help: {}", note.text));
    std::iter::once(error.message.clone())
        .chain(notes)
        .collect()
}

fn placed_in_document(index: &LineIndex<'_>, span: Span) -> lsp::Range {
    index
        .range(span)
        .expect("a span answered in a document falls on the text it was parsed from")
}

/// A location's span may be one the host gave, such as a context's site,
/// taken from the text its listing read, which a file changed on disk
/// without the client announcing it no longer holds; so this one is an
/// error the client is answered with.
fn placed_location(index: &LineIndex<'_>, path: &Path, span: Span) -> Answer<lsp::Range> {
    index.range(span).map_err(|error: Unplaceable| {
        failed(
            ErrorCode::RequestFailed,
            format!("{} changed on disk unannounced: {error}", path.display()),
        )
    })
}

fn call_label_detail(call: &CallShape) -> String {
    let params: Vec<String> = call
        .params
        .iter()
        .map(|param| match &param.name {
            Some(name) => format!("{name}: {}", param.ty),
            None => param.ty.clone(),
        })
        .collect();
    format!("({})", params.join(", "))
}

fn completion_kind(kind: CompletionKind) -> lsp::CompletionItemKind {
    match kind {
        CompletionKind::Local => lsp::CompletionItemKind::VARIABLE,
        CompletionKind::Param => lsp::CompletionItemKind::VALUE,
        CompletionKind::Field => lsp::CompletionItemKind::FIELD,
        CompletionKind::Method => lsp::CompletionItemKind::METHOD,
        CompletionKind::Function => lsp::CompletionItemKind::FUNCTION,
        CompletionKind::Context => lsp::CompletionItemKind::REFERENCE,
        CompletionKind::Keyword => lsp::CompletionItemKind::KEYWORD,
    }
}
