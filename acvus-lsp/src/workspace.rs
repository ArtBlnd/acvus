//! The compilations a host describes, kept checked as the editor changes
//! files.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::{fmt, io};

use acvus_ast::Span;
use acvus_ast::locate::on_cursor;
use acvus_mir::graph::{CompilationGraph, ContextInfo, QualifiedRef};
use acvus_mir::ty::Ty;
use acvus_mir::typeck::Resolved;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::session::{
    Completions, Definition, DocId, Document, Edit, Hover, LspError, LspErrorKind, LspSession,
    Mode, RenameRefusal, parse_error_to_lsp, qualified,
};

pub trait Host {
    type Compilation;

    /// A host reads and lists only through `reader`: the workspace lists
    /// again exactly when a path the reader recorded changes (RFC-0084), so
    /// a file read around it is a dependency the workspace does not see.
    fn compilations(
        &mut self,
        interner: &Interner,
        reader: &RecordingReader<'_>,
    ) -> Listing<Self::Compilation>;

    /// A document's text as the host's batch path reads its source, or the
    /// words that path gives for a source it cannot read, which the
    /// workspace shows on the document's path (RFC-0085 rule 1).
    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String>;

    /// Called only for a compilation every document of which the checker
    /// accepted, as a host's batch path runs its own rules only after the
    /// compiler's.
    fn check(&self, compilation: &Self::Compilation, checked: &Checked<'_>) -> Vec<HostDiagnostic>;
}

pub struct Listing<C> {
    pub compilations: Vec<CompilationSpec<C>>,
    pub refusals: Vec<HostDiagnostic>,
    pub links: Vec<Link>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub from: Location,
    pub to: Location,
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
    pub sites: Sites,
}

/// Where the host declares the contexts and inputs of a compilation,
/// which the graph types but does not place.
#[derive(Debug, Default)]
pub struct Sites {
    pub contexts: FxHashMap<QualifiedRef, Location>,
    pub inputs: FxHashMap<Astr, Location>,
}

pub struct DocumentSpec {
    pub path: PathBuf,
    pub document: Document,
}

/// Ordered by path, then span.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
    pub path: PathBuf,
    pub span: Span,
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
            kind: LspErrorKind::Host(self.span),
            message: self.message.clone(),
            related: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum TextError {
    DocumentRefused(String),
    Io(io::Error),
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextError::DocumentRefused(message) => write!(f, "{message}"),
            TextError::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TextError {}

/// The editor's unsaved buffers over the disk.
pub struct Vfs {
    buffers: FxHashMap<PathBuf, String>,
}

impl Vfs {
    pub fn read(&self, path: &Path) -> io::Result<String> {
        match self.buffers.get(path) {
            Some(text) => Ok(text.clone()),
            None => std::fs::read_to_string(path),
        }
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<io::Result<Entry>>> {
        let mut entries: Vec<io::Result<Entry>> = std::fs::read_dir(dir)?
            .map(|entry| {
                entry.map(|entry| Entry {
                    name: entry.file_name(),
                    path: entry.path(),
                    kind: entry.file_type().map(EntryKind::of),
                })
            })
            .collect();
        let mut buffered: Vec<Entry> = self
            .buffers
            .keys()
            .filter(|path| path.parent() == Some(dir))
            .filter_map(|path| {
                Some(Entry {
                    name: path.file_name()?.to_os_string(),
                    path: path.clone(),
                    kind: Ok(EntryKind::File),
                })
            })
            .filter(|buffer| {
                !entries
                    .iter()
                    .any(|entry| matches!(entry, Ok(entry) if entry.name == buffer.name))
            })
            .collect();
        buffered.sort_by(|a, b| a.path.cmp(&b.path));
        entries.extend(buffered.into_iter().map(Ok));
        Ok(entries)
    }

    fn kind(&self, path: &Path) -> io::Result<Option<EntryKind>> {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) => Ok(Some(EntryKind::of(metadata.file_type()))),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Ok(self.buffers.contains_key(path).then_some(EntryKind::File))
            }
            Err(error) => Err(error),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
}

impl EntryKind {
    fn of(file_type: std::fs::FileType) -> Self {
        if file_type.is_dir() {
            EntryKind::Directory
        } else if file_type.is_symlink() {
            EntryKind::Symlink
        } else {
            EntryKind::File
        }
    }
}

#[derive(Debug)]
pub struct Entry {
    pub name: OsString,
    pub path: PathBuf,
    pub kind: io::Result<EntryKind>,
}

pub struct RecordingReader<'v> {
    vfs: &'v Vfs,
    record: RefCell<ListingDependencies>,
}

impl<'v> RecordingReader<'v> {
    fn new(vfs: &'v Vfs) -> Self {
        RecordingReader {
            vfs,
            record: RefCell::new(ListingDependencies::default()),
        }
    }

    pub fn read(&self, path: &Path) -> io::Result<String> {
        self.record.borrow_mut().read.insert(path.to_path_buf());
        self.vfs.read(path)
    }

    pub fn read_dir(&self, dir: &Path) -> io::Result<Vec<io::Result<Entry>>> {
        let entries = self.vfs.list(dir);
        let listed = match &entries {
            Ok(entries) => entries
                .iter()
                .map(|entry| {
                    let entry = entry.as_ref().ok()?;
                    let seen = match &entry.kind {
                        Ok(kind) => Seen::Kind(*kind),
                        Err(_) => Seen::Unknown,
                    };
                    Some((entry.name.clone(), seen))
                })
                .collect::<Option<BTreeMap<OsString, Seen>>>()
                .map_or(Listed::Failed, Listed::Entries),
            Err(_) => Listed::Failed,
        };
        let mut record = self.record.borrow_mut();
        let merged = match record.listed.remove(dir) {
            None => listed,
            Some(earlier) if earlier == listed => listed,
            Some(_) => Listed::Failed,
        };
        record.listed.insert(dir.to_path_buf(), merged);
        entries
    }

    fn into_dependencies(self) -> ListingDependencies {
        self.record.into_inner()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Seen {
    Kind(EntryKind),
    Unknown,
}

#[derive(Debug, PartialEq, Eq)]
enum Listed {
    Failed,
    Entries(BTreeMap<OsString, Seen>),
}

#[derive(Default)]
struct ListingDependencies {
    read: BTreeSet<PathBuf>,
    listed: BTreeMap<PathBuf, Listed>,
}

impl ListingDependencies {
    /// Enforces RFC-0084 rule 2. A path whose kind the disk cannot tell now
    /// counts as changed, so the listing runs again and meets the disk's
    /// error itself.
    fn changed_by(&self, path: &Path, vfs: &Vfs) -> bool {
        if self.read.contains(path) || self.listed.contains_key(path) {
            return true;
        }
        path.ancestors().skip(1).any(|dir| {
            let Some(listed) = self.listed.get(dir) else {
                return false;
            };
            let Listed::Entries(entries) = listed else {
                return true;
            };
            let Some(name) = path
                .strip_prefix(dir)
                .expect("an ancestor of a path is its prefix")
                .components()
                .next()
            else {
                return false;
            };
            let name = name.as_os_str();
            let Ok(now) = vfs.kind(&dir.join(name)) else {
                return true;
            };
            match entries.get(name) {
                Some(Seen::Unknown) => true,
                Some(Seen::Kind(then)) => now != Some(*then),
                None => now.is_some(),
            }
        })
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

    /// The type the checker solved the body's value at `path` to (RFC-0085
    /// rule 2), beside the function type `inferred_ty` gives. Under a
    /// declared `!` (RFC-0054 rule 5) that function type states no return,
    /// and this is where a host reads the value's type. `None` only when the
    /// compilation holds no open document at `path`.
    pub fn tail_ty(&self, path: &Path) -> Option<Ty> {
        let qref = self
            .session
            .function_ref(self.open(path)?)
            .expect("a document the compilation holds open is open in its session");
        let resolution = self.session.graph().resolution(qref).expect(
            "a document of an accepted compilation was checked without a refusal, \
             so its function's outcome is complete",
        );
        Some(resolution.tail_ty.clone())
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

fn read_document<H>(host: &H, vfs: &Vfs, path: &Path) -> Result<String, LspError>
where
    H: Host,
{
    host.read(vfs, path).map_err(|message| {
        HostDiagnostic {
            path: path.to_path_buf(),
            span: None,
            message,
        }
        .to_lsp()
    })
}

type DocumentTexts = FxHashMap<PathBuf, Result<String, LspError>>;

struct Loaded<C> {
    host: C,
    sites: Sites,
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
        texts: &DocumentTexts,
    ) -> Self {
        let mut loaded = Loaded {
            host: environment.host,
            sites: environment.sites,
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
            loaded.reread(&path, text_of(texts, &path));
        }
        loaded
    }

    fn reread(&mut self, path: &Path, text: &Result<String, LspError>) {
        let held = self.documents.remove(path);
        let next = match (held, text) {
            (Some(Held::Open(id)), Ok(text)) => {
                self.session.update_source(id, text);
                Held::Open(id)
            }
            (Some(Held::Open(id)), Err(error)) => {
                self.session.close(id);
                Held::Unreadable(error.clone())
            }
            (Some(Held::Unreadable(_)) | None, Ok(text)) => {
                let id = self
                    .session
                    .open(self.specs[path].clone(), text)
                    .expect("`specs` holds one document per function, and only its path opens it");
                Held::Open(id)
            }
            (Some(Held::Unreadable(_)) | None, Err(error)) => Held::Unreadable(error.clone()),
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
    fn reread(&mut self, interner: &Interner, path: &Path, text: &Result<String, LspError>) {
        let errors = match text {
            Ok(text) => self.modes[path]
                .parse(interner, text)
                .errors
                .iter()
                .map(parse_error_to_lsp)
                .collect(),
            Err(error) => vec![error.clone()],
        };
        self.documents.insert(path.to_path_buf(), errors);
    }
}

fn text_of<'t>(texts: &'t DocumentTexts, path: &Path) -> &'t Result<String, LspError> {
    texts
        .get(path)
        .expect("every listed document path is read before its compilations load")
}

enum State<C> {
    Loaded(Loaded<C>),
    Refused(Refused),
}

struct Compilation<C> {
    state: State<C>,
}

impl<C> Compilation<C> {
    fn load(interner: &Interner, spec: CompilationSpec<C>, texts: &DocumentTexts) -> Self {
        let CompilationSpec {
            id: _,
            environment,
            documents,
        } = spec;
        let state = match environment {
            Ok(environment) => State::Loaded(Loaded::new(interner, environment, documents, texts)),
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
                    refused.reread(interner, &spec.path, text_of(texts, &spec.path));
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

    fn reread(&mut self, interner: &Interner, path: &Path, text: &Result<String, LspError>) {
        match &mut self.state {
            State::Loaded(loaded) => loaded.reread(path, text),
            State::Refused(refused) => refused.reread(interner, path, text),
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
                    extend_unique(into, path, loaded.document_diagnostics(path));
                }
                for duplicate in &loaded.duplicates {
                    extend_unique(into, &duplicate.path, vec![duplicate.to_lsp()]);
                }
                &loaded.refusals
            }
            State::Refused(refused) => {
                for (path, errors) in &refused.documents {
                    extend_unique(into, path, errors.clone());
                }
                &refused.refusals
            }
        };
        for refusal in refusals {
            extend_unique(into, &refusal.path, vec![refusal.to_lsp()]);
        }
    }
}

/// Enforces RFC-0086 rule 8 for a link from a document.
fn refused_link(link: Link) -> HostDiagnostic {
    let Span { start, end } = link.from.span;
    HostDiagnostic {
        message: format!(
            "the link from bytes {start}..{end} of {} is refused: it is a document, \
             and a document's names are the checker's",
            link.from.path.display()
        ),
        path: link.from.path,
        span: Some(link.from.span),
    }
}

fn extend_unique(into: &mut BTreeMap<PathBuf, Vec<LspError>>, path: &Path, errors: Vec<LspError>) {
    for error in errors {
        let held = into.entry(path.to_path_buf()).or_default();
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
    links: Vec<Link>,
    dependencies: ListingDependencies,
    texts: DocumentTexts,
}

impl<H> Workspace<H>
where
    H: Host,
{
    pub fn new(interner: &Interner, host: H) -> Self {
        Self::with_buffers(interner, host, [])
    }

    pub fn with_buffers<B>(interner: &Interner, host: H, buffers: B) -> Self
    where
        B: IntoIterator<Item = (PathBuf, String)>,
    {
        let mut workspace = Workspace {
            host,
            interner: interner.clone(),
            vfs: Vfs {
                buffers: buffers.into_iter().collect(),
            },
            compilations: Vec::new(),
            refusals: Vec::new(),
            links: Vec::new(),
            dependencies: ListingDependencies::default(),
            texts: DocumentTexts::default(),
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

    /// A listed document's text is the one its compilations parsed, whatever
    /// the disk holds since, so the spans answered in it fall on it
    /// (RFC-0086 rule 2); any other path's is what the buffers or the disk
    /// hold now.
    pub fn text(&self, path: &Path) -> Result<Cow<'_, str>, TextError> {
        match self.texts.get(path) {
            Some(Ok(text)) => Ok(Cow::Borrowed(text)),
            Some(Err(refusal)) => Err(TextError::DocumentRefused(refusal.message.clone())),
            None => self.vfs.read(path).map(Cow::Owned).map_err(TextError::Io),
        }
    }

    pub fn diagnostics(&self) -> BTreeMap<PathBuf, Vec<LspError>> {
        let mut by_path: BTreeMap<PathBuf, Vec<LspError>> = BTreeMap::new();
        for compilation in &self.compilations {
            compilation.diagnostics_into(&mut by_path);
        }
        for refusal in &self.refusals {
            extend_unique(&mut by_path, &refusal.path, vec![refusal.to_lsp()]);
        }
        by_path
    }

    /// From the first compilation, in id order, that holds the document
    /// open.
    pub fn completions(&self, path: &Path, cursor: usize) -> Option<Completions> {
        let (loaded, id) = self.open_in_first(path)?;
        loaded.session.completions(id, cursor)
    }

    /// From the first compilation, in id order, that holds the document
    /// open, as `completions`.
    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        let (loaded, id) = self.open_in_first(path)?;
        loaded.session.hover(id, offset)
    }

    /// A function resolves to the document of the same compilation that
    /// defines it, at its start; a context or an input to the site its
    /// host gave it.
    pub fn definition(&self, path: &Path, offset: usize) -> Option<Location> {
        if !self.texts.contains_key(path) {
            let links = self.links.iter().filter(|link| link.from.path == path);
            return on_cursor(links, offset, |link| link.from.span).map(|link| link.to.clone());
        }
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
                    span: Span::new(0, 0),
                }),
            Definition::Context(qref) => loaded.sites.contexts.get(&qref).cloned(),
            Definition::Input(name) => loaded.sites.inputs.get(&name).cloned(),
        }
    }

    /// The places that refer to what the name at `offset` refers to, in
    /// every compilation that holds the document open: a local's or an
    /// input's in the document, a context's or a function's in every
    /// document of the compilation. A function's declaration is the start
    /// of the document that defines it; a context's or an input's is the
    /// site the compilation's host gave it, where it gave one. A host's
    /// link to that site is a use of it, held whatever
    /// `include_declaration` says (RFC-0086 rule 8). `None` where no name
    /// at `offset` resolved in any of them.
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
            let site = match referent {
                Resolved::Context(qref) => loaded.sites.contexts.get(&qref),
                Resolved::Input(name) => loaded.sites.inputs.get(&name),
                Resolved::Local(_) | Resolved::Function(_) => None,
            };
            let Some(site) = site else {
                continue;
            };
            locations.extend(
                self.links
                    .iter()
                    .filter(|link| link.to == *site)
                    .map(|link| link.from.clone()),
            );
            if include_declaration {
                locations.insert(site.clone());
            }
        }
        found.map(|locations| locations.into_iter().collect())
    }

    pub fn rename_target(&self, path: &Path, offset: usize) -> Result<Span, RenameRefusal> {
        let (loaded, id) = self.open_in_first(path).ok_or(RenameRefusal::NotAName)?;
        loaded.session.rename_target(id, offset)
    }

    /// The edits in `path` that rename the local at `offset`: a local is
    /// bound and used within one document. Planned in the first
    /// compilation, in id order, that holds the document open, and checked
    /// in every one that does, since each may resolve its names apart.
    pub fn rename(
        &self,
        path: &Path,
        offset: usize,
        new_name: &str,
    ) -> Result<Vec<Edit>, RenameRefusal> {
        let (loaded, id) = self.open_in_first(path).ok_or(RenameRefusal::NotAName)?;
        let plan = loaded.session.plan_rename(id, offset, new_name)?;
        for (loaded, id) in self.open_in_all(path) {
            loaded.session.check_rename(id, &plan)?;
        }
        Ok(plan.edits().to_vec())
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

    /// Follows RFC-0084 rules 2 and 3.
    fn touched(&mut self, path: &Path) {
        if self.dependencies.changed_by(path, &self.vfs) {
            self.reload();
            return;
        }
        let Some(text) = self.texts.get_mut(path) else {
            return;
        };
        *text = read_document(&self.host, &self.vfs, path);
        for compilation in &mut self.compilations {
            if compilation.holds(path) {
                compilation.reread(&self.interner, path, text);
                compilation.recheck(&self.host);
            }
        }
    }

    fn reload(&mut self) {
        let reader = RecordingReader::new(&self.vfs);
        let Listing {
            compilations: mut specs,
            mut refusals,
            links,
        } = self.host.compilations(&self.interner, &reader);
        self.dependencies = reader.into_dependencies();
        specs.sort_by(|a, b| a.id.cmp(&b.id));
        let paths: BTreeSet<&Path> = specs
            .iter()
            .flat_map(|spec| &spec.documents)
            .map(|document| document.path.as_path())
            .collect();
        let (from_documents, links): (Vec<Link>, Vec<Link>) = links
            .into_iter()
            .partition(|link| paths.contains(link.from.path.as_path()));
        refusals.extend(from_documents.into_iter().map(refused_link));
        self.refusals = refusals;
        self.links = links;
        self.texts = paths
            .into_iter()
            .map(|path| {
                let text = read_document(&self.host, &self.vfs, path);
                (path.to_path_buf(), text)
            })
            .collect();
        self.compilations = specs
            .into_iter()
            .map(|spec| Compilation::load(&self.interner, spec, &self.texts))
            .collect();
        for compilation in &mut self.compilations {
            compilation.recheck(&self.host);
        }
    }
}
