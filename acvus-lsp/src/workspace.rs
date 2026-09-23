//! The compilations a host describes, kept checked as the editor changes
//! files.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use acvus_ast::Span;
use acvus_mir::graph::{CompilationGraph, ContextInfo};
use acvus_mir::ty::Ty;
use acvus_mir::typeck::Resolved;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::session::{
    CompletionItem, Definition, DocId, Document, Edit, Hover, LspError, LspErrorCategory,
    LspSession, Mode, RenameRefusal, parse_error_to_lsp, qualified,
};

pub trait Host {
    type Compilation;

    /// Called again whenever a file that is not a document changes, so a
    /// host reads every file it depends on through `vfs`.
    fn compilations(&mut self, interner: &Interner, vfs: &Vfs) -> Listing<Self::Compilation>;

    /// Called only for a compilation every document of which the checker
    /// accepted, as a host's batch path runs its own rules only after the
    /// compiler's.
    fn check(&self, compilation: &Self::Compilation, checked: &Checked<'_>) -> Vec<HostDiagnostic>;
}

pub struct Listing<C> {
    pub compilations: Vec<CompilationSpec<C>>,
    pub refusals: Vec<HostDiagnostic>,
}

/// The file a compilation is read from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompilationId(pub PathBuf);

pub struct CompilationSpec<C> {
    pub id: CompilationId,
    pub environment: Result<Environment<C>, Vec<HostDiagnostic>>,
    pub documents: Vec<DocumentSpec>,
}

pub struct Environment<C> {
    pub graph: CompilationGraph,
    pub host: C,
}

pub struct DocumentSpec {
    pub path: PathBuf,
    pub document: Document,
}

/// Ordered by path, then span.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
    pub path: PathBuf,
    pub span: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostDiagnostic {
    pub path: PathBuf,
    pub span: Option<Span>,
    pub message: String,
}

impl HostDiagnostic {
    fn to_lsp(&self) -> LspError {
        LspError {
            category: LspErrorCategory::Host,
            message: self.message.clone(),
            span: self.span.map(|span| (span.start, span.end)),
            related: Vec::new(),
        }
    }
}

/// The editor's unsaved buffers over the disk.
#[derive(Default)]
pub struct Vfs {
    buffers: FxHashMap<PathBuf, String>,
}

impl Vfs {
    pub fn read(&self, path: &Path) -> std::io::Result<String> {
        match self.buffers.get(path) {
            Some(text) => Ok(text.clone()),
            None => std::fs::read_to_string(path),
        }
    }
}

pub struct Checked<'a> {
    session: &'a LspSession,
    documents: &'a FxHashMap<PathBuf, Held>,
}

impl Checked<'_> {
    pub fn interner(&self) -> &Interner {
        self.session.interner()
    }

    pub fn inferred_ty(&self, path: &Path) -> Option<&Ty> {
        let qref = self.session.function_ref(self.open(path)?)?;
        self.session.graph().inferred_ty(qref)
    }

    pub fn required_inputs(&self, path: &Path) -> Option<Vec<ContextInfo>> {
        Some(self.session.required_inputs(self.open(path)?))
    }

    fn open(&self, path: &Path) -> Option<DocId> {
        match self.documents.get(path)? {
            Held::Open(id) => Some(*id),
            Held::Unreadable(_) => None,
        }
    }
}

enum Held {
    Open(DocId),
    Unreadable(LspError),
}

fn unreadable(path: &Path, error: &std::io::Error) -> LspError {
    LspError {
        category: LspErrorCategory::Unreadable,
        message: format!("cannot read {}: {error}", path.display()),
        span: None,
        related: Vec::new(),
    }
}

struct Loaded<C> {
    host: C,
    session: LspSession,
    specs: FxHashMap<PathBuf, Document>,
    documents: FxHashMap<PathBuf, Held>,
    /// One per listed document whose function an earlier listed document
    /// of the compilation defines; that document is not opened, since a
    /// session holds one open document per function.
    duplicates: Vec<HostDiagnostic>,
    refusals: Vec<HostDiagnostic>,
}

impl<C> Loaded<C> {
    fn new(
        interner: &Interner,
        environment: Environment<C>,
        documents: Vec<DocumentSpec>,
        vfs: &Vfs,
    ) -> Self {
        let mut loaded = Loaded {
            host: environment.host,
            session: LspSession::new(interner, environment.graph),
            specs: FxHashMap::default(),
            documents: FxHashMap::default(),
            duplicates: Vec::new(),
            refusals: Vec::new(),
        };
        for DocumentSpec { path, document } in documents {
            let defining = loaded
                .specs
                .iter()
                .find(|(_, held)| held.qref == document.qref)
                .map(|(defining, _)| defining.clone());
            if let Some(defining) = defining {
                loaded.duplicates.push(HostDiagnostic {
                    message: format!(
                        "{} and {} are both the body of function `{}`; {} is not opened",
                        defining.display(),
                        path.display(),
                        qualified(interner, document.qref),
                        path.display(),
                    ),
                    path,
                    span: None,
                });
                continue;
            }
            loaded.specs.insert(path.clone(), document);
            loaded.reread(&path, vfs);
        }
        loaded
    }

    fn reread(&mut self, path: &Path, vfs: &Vfs) {
        let text = vfs.read(path);
        let held = self.documents.remove(path);
        let next = match (held, text) {
            (Some(Held::Open(id)), Ok(text)) => {
                self.session.update_source(id, &text);
                Held::Open(id)
            }
            (Some(Held::Open(id)), Err(error)) => {
                self.session.close(id);
                Held::Unreadable(unreadable(path, &error))
            }
            (Some(Held::Unreadable(_)) | None, Ok(text)) => {
                let id = self
                    .session
                    .open(self.specs[path].clone(), &text)
                    .expect("`specs` holds one document per function, and only its path opens it");
                Held::Open(id)
            }
            (Some(Held::Unreadable(_)) | None, Err(error)) => {
                Held::Unreadable(unreadable(path, &error))
            }
        };
        self.documents.insert(path.to_path_buf(), next);
    }

    fn document_diagnostics(&self, path: &Path) -> Vec<LspError> {
        match &self.documents[path] {
            Held::Open(id) => self.session.diagnostics(*id),
            Held::Unreadable(error) => vec![error.clone()],
        }
    }

    /// A compilation with a document left unopened is not what the host's
    /// batch path compiles, so its rules do not run on it.
    fn accepted(&self) -> bool {
        self.duplicates.is_empty()
            && self
                .documents
                .keys()
                .all(|path| self.document_diagnostics(path).is_empty())
    }
}

struct Refused {
    refusals: Vec<HostDiagnostic>,
    modes: FxHashMap<PathBuf, Mode>,
    documents: FxHashMap<PathBuf, Vec<LspError>>,
}

impl Refused {
    fn reread(&mut self, interner: &Interner, path: &Path, vfs: &Vfs) {
        let errors = match vfs.read(path) {
            Ok(text) => self.modes[path]
                .parse(interner, &text)
                .errors
                .iter()
                .map(parse_error_to_lsp)
                .collect(),
            Err(error) => vec![unreadable(path, &error)],
        };
        self.documents.insert(path.to_path_buf(), errors);
    }
}

enum State<C> {
    Loaded(Loaded<C>),
    Refused(Refused),
}

struct Compilation<C> {
    state: State<C>,
}

impl<C> Compilation<C> {
    fn load(interner: &Interner, spec: CompilationSpec<C>, vfs: &Vfs) -> Self {
        let CompilationSpec {
            id: _,
            environment,
            documents,
        } = spec;
        let state = match environment {
            Ok(environment) => State::Loaded(Loaded::new(interner, environment, documents, vfs)),
            Err(refusals) => {
                let mut refused = Refused {
                    refusals,
                    modes: documents
                        .iter()
                        .map(|spec| (spec.path.clone(), spec.document.mode))
                        .collect(),
                    documents: FxHashMap::default(),
                };
                for spec in &documents {
                    refused.reread(interner, &spec.path, vfs);
                }
                State::Refused(refused)
            }
        };
        Compilation { state }
    }

    fn holds(&self, path: &Path) -> bool {
        match &self.state {
            State::Loaded(loaded) => loaded.specs.contains_key(path),
            State::Refused(refused) => refused.modes.contains_key(path),
        }
    }

    fn reread(&mut self, interner: &Interner, path: &Path, vfs: &Vfs) {
        match &mut self.state {
            State::Loaded(loaded) => loaded.reread(path, vfs),
            State::Refused(refused) => refused.reread(interner, path, vfs),
        }
    }

    fn recheck<H>(&mut self, host: &H)
    where
        H: Host<Compilation = C>,
    {
        let State::Loaded(loaded) = &mut self.state else {
            return;
        };
        loaded.refusals = match loaded.accepted() {
            true => host.check(
                &loaded.host,
                &Checked {
                    session: &loaded.session,
                    documents: &loaded.documents,
                },
            ),
            false => Vec::new(),
        };
    }

    fn diagnostics_into(&self, into: &mut BTreeMap<PathBuf, Vec<LspError>>) {
        let refusals = match &self.state {
            State::Loaded(loaded) => {
                for path in loaded.documents.keys() {
                    extend_unique(
                        into.entry(path.clone()).or_default(),
                        loaded.document_diagnostics(path),
                    );
                }
                for duplicate in &loaded.duplicates {
                    extend_unique(
                        into.entry(duplicate.path.clone()).or_default(),
                        vec![duplicate.to_lsp()],
                    );
                }
                &loaded.refusals
            }
            State::Refused(refused) => {
                for (path, errors) in &refused.documents {
                    extend_unique(into.entry(path.clone()).or_default(), errors.clone());
                }
                &refused.refusals
            }
        };
        for refusal in refusals {
            extend_unique(
                into.entry(refusal.path.clone()).or_default(),
                vec![refusal.to_lsp()],
            );
        }
    }
}

fn extend_unique(held: &mut Vec<LspError>, errors: Vec<LspError>) {
    for error in errors {
        if !held.contains(&error) {
            held.push(error);
        }
    }
}

pub struct Workspace<H>
where
    H: Host,
{
    host: H,
    interner: Interner,
    vfs: Vfs,
    compilations: Vec<Compilation<H::Compilation>>,
    refusals: Vec<HostDiagnostic>,
}

impl<H> Workspace<H>
where
    H: Host,
{
    pub fn new(interner: &Interner, host: H) -> Self {
        let mut workspace = Workspace {
            host,
            interner: interner.clone(),
            vfs: Vfs::default(),
            compilations: Vec::new(),
            refusals: Vec::new(),
        };
        workspace.reload();
        workspace
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    pub fn set_buffer(&mut self, path: PathBuf, text: String) {
        self.vfs.buffers.insert(path.clone(), text);
        self.touched(&path);
    }

    pub fn drop_buffer(&mut self, path: &Path) {
        if self.vfs.buffers.remove(path).is_some() {
            self.touched(path);
        }
    }

    pub fn file_changed(&mut self, path: &Path) {
        if !self.vfs.buffers.contains_key(path) {
            self.touched(path);
        }
    }

    /// Every document of every compilation has an entry, empty when it is
    /// accepted, so a client that clears what it showed before has a path
    /// to clear.
    pub fn diagnostics(&self) -> BTreeMap<PathBuf, Vec<LspError>> {
        let mut by_path: BTreeMap<PathBuf, Vec<LspError>> = BTreeMap::new();
        for compilation in &self.compilations {
            compilation.diagnostics_into(&mut by_path);
        }
        for refusal in &self.refusals {
            extend_unique(
                by_path.entry(refusal.path.clone()).or_default(),
                vec![refusal.to_lsp()],
            );
        }
        by_path
    }

    /// From the first compilation, in id order, that holds the document
    /// open; `None` when none does.
    pub fn completions(&self, path: &Path, cursor: usize) -> Option<Vec<CompletionItem>> {
        let (loaded, id) = self.open_in_first(path)?;
        Some(loaded.session.completions(id, cursor))
    }

    /// From the first compilation, in id order, that holds the document
    /// open, as `completions`.
    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        let (loaded, id) = self.open_in_first(path)?;
        loaded.session.hover(id, offset)
    }

    /// A function resolves to the document of the same compilation that
    /// defines it, at its start.
    pub fn definition(&self, path: &Path, offset: usize) -> Option<Location> {
        let (loaded, id) = self.open_in_first(path)?;
        match loaded.session.definition(id, offset)? {
            Definition::Local { span } => Some(Location {
                path: path.to_path_buf(),
                span,
            }),
            Definition::Function(qref) => loaded
                .specs
                .iter()
                .find(|(_, document)| document.qref == qref)
                .map(|(defining, _)| Location {
                    path: defining.clone(),
                    span: (0, 0),
                }),
        }
    }

    /// The places that refer to what the name at `offset` refers to, in
    /// every compilation that holds the document open: a local's or an
    /// input's in the document, a context's or a function's in every
    /// document of the compilation. A function's declaration is the start
    /// of the document that defines it. `None` where no name at `offset`
    /// resolved in any of them.
    pub fn references(
        &self,
        path: &Path,
        offset: usize,
        include_declaration: bool,
    ) -> Option<Vec<Location>> {
        let mut found: Option<BTreeSet<Location>> = None;
        for (loaded, id) in self.open_in_all(path) {
            let Some(referent) = loaded.session.referent(id, offset) else {
                continue;
            };
            let searched: Vec<(&Path, DocId)> = match referent {
                Resolved::Local(_) | Resolved::Input(_) => vec![(path, id)],
                Resolved::Function(_) | Resolved::Context(_) => loaded
                    .documents
                    .iter()
                    .filter_map(|(searched, held)| match held {
                        Held::Open(id) => Some((searched.as_path(), *id)),
                        Held::Unreadable(_) => None,
                    })
                    .collect(),
            };
            let locations = found.get_or_insert_default();
            for (searched, id) in searched {
                locations.extend(
                    loaded
                        .session
                        .references_to(id, referent, include_declaration)
                        .into_iter()
                        .map(|span| Location {
                            path: searched.to_path_buf(),
                            span,
                        }),
                );
            }
        }
        found.map(|locations| locations.into_iter().collect())
    }

    /// The edits that rename the local at `offset`, planned in the first
    /// compilation, in id order, that holds the document open, and checked
    /// in every one that does, since each may resolve its names apart.
    pub fn rename(
        &self,
        path: &Path,
        offset: usize,
        new_name: &str,
    ) -> Result<Vec<(PathBuf, Edit)>, RenameRefusal> {
        let (loaded, id) = self.open_in_first(path).ok_or(RenameRefusal::NotAName)?;
        let plan = loaded.session.plan_rename(id, offset, new_name)?;
        for (loaded, id) in self.open_in_all(path) {
            loaded.session.check_rename(id, &plan)?;
        }
        Ok(plan
            .edits()
            .iter()
            .map(|edit| (path.to_path_buf(), edit.clone()))
            .collect())
    }

    fn open_in_first(&self, path: &Path) -> Option<(&Loaded<H::Compilation>, DocId)> {
        self.open_in_all(path).next()
    }

    /// The compilations, in id order, that hold the document open.
    fn open_in_all<'w>(
        &'w self,
        path: &Path,
    ) -> impl Iterator<Item = (&'w Loaded<H::Compilation>, DocId)> {
        self.compilations.iter().filter_map(move |compilation| {
            let State::Loaded(loaded) = &compilation.state else {
                return None;
            };
            match loaded.documents.get(path)? {
                Held::Open(id) => Some((loaded, *id)),
                Held::Unreadable(_) => None,
            }
        })
    }

    fn touched(&mut self, path: &Path) {
        if !self.compilations.iter().any(|c| c.holds(path)) {
            self.reload();
            return;
        }
        for compilation in &mut self.compilations {
            if compilation.holds(path) {
                compilation.reread(&self.interner, path, &self.vfs);
                compilation.recheck(&self.host);
            }
        }
    }

    fn reload(&mut self) {
        let Listing {
            compilations: mut specs,
            refusals,
        } = self.host.compilations(&self.interner, &self.vfs);
        specs.sort_by(|a, b| a.id.cmp(&b.id));
        self.refusals = refusals;
        self.compilations = specs
            .into_iter()
            .map(|spec| Compilation::load(&self.interner, spec, &self.vfs))
            .collect();
        for compilation in &mut self.compilations {
            compilation.recheck(&self.host);
        }
    }
}
