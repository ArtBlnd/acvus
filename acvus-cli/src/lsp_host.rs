//! The editor's view of what `acvus check <file>` checks: every source under
//! a root is one compilation, alone, whose contexts are the ones it names,
//! each at the type the compilation solves (RFC-0090 rule 1).

use std::path::{Path, PathBuf};

use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, EntryKind, Environment, Host,
    HostDiagnostic, Listing, Mode, RecordingReader, Sites, Vfs,
};
use acvus_utils::Interner;

use crate::compile;

pub struct CliHost {
    root: PathBuf,
}

impl CliHost {
    pub fn new(root: PathBuf) -> Self {
        CliHost { root }
    }
}

struct Source {
    path: PathBuf,
    mode: Mode,
}

fn mode_of(path: &Path) -> Option<Mode> {
    match path.extension()?.to_str()? {
        "acvus" => Some(Mode::Script),
        "acvt" => Some(Mode::Template),
        _ => None,
    }
}

fn skipped(name: &std::ffi::OsStr) -> bool {
    name.as_encoded_bytes().starts_with(b".") || name == "target"
}

fn sources_under(
    reader: &RecordingReader<'_>,
    dir: &Path,
    sources: &mut Vec<Source>,
    refusals: &mut Vec<HostDiagnostic>,
) {
    let entries = match reader.read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            refusals.extend(refused(dir, error.to_string()));
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                refusals.extend(refused(dir, error.to_string()));
                continue;
            }
        };
        let path = entry.path;
        let kind = match entry.kind {
            Ok(kind) => kind,
            Err(error) => {
                refusals.extend(refused(&path, error.to_string()));
                continue;
            }
        };
        match kind {
            EntryKind::Directory => {
                if !skipped(&entry.name) {
                    sources_under(reader, &path, sources, refusals);
                }
                continue;
            }
            EntryKind::File | EntryKind::Symlink => {}
        }
        if let Some(mode) = mode_of(&path) {
            sources.push(Source { path, mode });
        }
    }
}

fn refused(path: &Path, message: String) -> Vec<HostDiagnostic> {
    vec![HostDiagnostic {
        path: path.to_path_buf(),
        span: None,
        message,
    }]
}

fn compile_mode(mode: Mode) -> compile::Mode {
    match mode {
        Mode::Script => compile::Mode::Script,
        Mode::Template => compile::Mode::Template,
    }
}

/// The source is read here to learn the contexts it names, so an edit to it
/// lists its compilation again (RFC-0084 rule 4).
fn environment_of(
    interner: &Interner,
    reader: &RecordingReader<'_>,
    source: &Source,
) -> Result<Environment<()>, Vec<HostDiagnostic>> {
    let text = reader
        .read(&source.path)
        .map_err(|error| refused(&source.path, compile::unreadable_source(&source.path, &error)))?;
    let parsed = compile::parse(interner, compile_mode(source.mode), &text);
    acvus_interpreter::environment(
        interner,
        acvus_interpreter::context_refs(&parsed.ast),
        vec![compile::entry_ref(interner)],
        crate::cli_registries(),
    )
    .map(|graph| Environment {
        graph,
        host: (),
        sites: Sites::default(),
    })
    .map_err(|error| refused(&source.path, compile::combine_refusal(&error)))
}

impl Host for CliHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, reader: &RecordingReader<'_>) -> Listing<()> {
        let mut sources = Vec::new();
        let mut refusals = Vec::new();
        sources_under(reader, &self.root, &mut sources, &mut refusals);
        let compilations = sources
            .into_iter()
            .map(|source| CompilationSpec {
                id: CompilationId(source.path.clone()),
                environment: environment_of(interner, reader, &source),
                documents: vec![DocumentSpec {
                    path: source.path,
                    document: Document {
                        qref: compile::entry_ref(interner),
                        mode: source.mode,
                        ty: acvus_interpreter::untyped_entry_ty(),
                        inputs: acvus_mir::graph::Inputs::FromReads,
                    },
                }],
            })
            .collect();
        Listing {
            compilations,
            refusals,
            links: Vec::new(),
        }
    }

    fn read(&self, vfs: &Vfs, path: &Path) -> Result<String, String> {
        vfs.read(path)
            .map_err(|error| compile::unreadable_source(path, &error))
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use acvus_lsp::{LspError, LspErrorKind, Workspace};
    use acvus_mir::graph::optimize::Opt;

    use super::*;

    /// What `acvus check` says of `units`, message by message.
    fn batch(units: &[compile::Unit]) -> Result<(), Vec<String>> {
        let refused = compile::compile(
            units,
            Some(0),
            &[],
            crate::cli_registries(),
            Opt::Full,
            acvus_interpreter::SequentialExecutor,
        );
        match refused {
            Ok(_) => Ok(()),
            Err(compile::Refused::Diagnostics(diagnostics)) => {
                Err(diagnostics.into_iter().map(|d| d.message).collect())
            }
            Err(compile::Refused::Usage(message)) => panic!("no binding is given, and one was refused: {message}"),
        }
    }

    const ADDS_ONE: &str = "let m = @n + 1;\n\"{{ &m | to_string }}\"\n";
    const PURE: &str = "let m = 1 + 2;\n\"{{ &m | to_string }}\"\n";

    struct Tree {
        _dir: tempfile::TempDir,
        root: PathBuf,
    }

    impl Tree {
        fn new() -> Self {
            let dir = tempfile::tempdir().expect("a temp dir");
            let root = dir.path().to_path_buf();
            Tree { _dir: dir, root }
        }

        fn write(&self, relative: &str, text: &str) -> PathBuf {
            let path = self.root.join(relative);
            let parent = path.parent().expect("a written file has a directory");
            std::fs::create_dir_all(parent).expect("create the file's directory");
            std::fs::write(&path, text).expect("write the file");
            path
        }

        fn workspace(&self, interner: &Interner) -> Workspace<CliHost> {
            Workspace::new(interner, CliHost::new(self.root.clone()))
        }
    }

    fn messages(errors: &[LspError]) -> Vec<String> {
        errors.iter().map(|error| error.message.clone()).collect()
    }

    fn is_type(error: &LspError) -> bool {
        matches!(error.kind, LspErrorKind::Type(_))
    }

    fn is_host(error: &LspError) -> bool {
        matches!(error.kind, LspErrorKind::Host(_))
    }

    fn held_open(workspace: &Workspace<CliHost>, path: &Path) -> bool {
        let text = std::fs::read_to_string(path).expect("a source reads");
        let in_code = match mode_of(path).expect("a source is a script or a template") {
            Mode::Script => 0,
            Mode::Template => {
                let tag = text.find("{{").expect("the template has a tag");
                tag + text[tag..]
                    .find(char::is_alphanumeric)
                    .expect("the tag names something")
            }
        };
        workspace.completions(path, in_code).is_some()
    }

    #[test]
    fn a_script_that_reads_a_context_is_accepted() {
        let tree = Tree::new();
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(held_open(&workspace, &script));
    }

    #[test]
    fn a_type_error_is_reported_as_acvus_check_reports_it() {
        let tree = Tree::new();
        let source = "let m = 1;\nlet k = m + \"x\";\n\"{{ &k | to_string }}\"\n";
        let script = tree.write("a/main.acvus", source);
        let interner = Interner::new();
        let workspace = tree.workspace(&interner);

        let units = [compile::Unit {
            role: compile::Role::Entry("main".to_string()),
            space: None,
            path: script.display().to_string(),
            mode: compile::Mode::Script,
            text: source.to_string(),
        }];
        let Err(batch) = batch(&units) else {
            panic!("`acvus check` accepts a String added to an Int");
        };

        let diagnostics = workspace.diagnostics();
        let editor = &diagnostics[&script];
        assert!(!batch.is_empty());
        assert_eq!(messages(editor), batch);
        assert!(editor.iter().all(is_type));
    }

    /// `@n`'s type is what the whole compilation solves it to, in the editor
    /// as in `acvus check`, so the refusal of `m + "x"` names `i64`.
    #[test]
    fn a_type_error_over_a_solved_context_is_reported_as_acvus_check_reports_it() {
        let tree = Tree::new();
        let source = "let m = @n + 1;\nlet k = m + \"x\";\n\"{{ &k | to_string }}\"\n";
        let script = tree.write("a/main.acvus", source);
        let interner = Interner::new();
        let workspace = tree.workspace(&interner);

        let units = [compile::Unit {
            role: compile::Role::Entry("main".to_string()),
            space: None,
            path: script.display().to_string(),
            mode: compile::Mode::Script,
            text: source.to_string(),
        }];
        let Err(batch) = batch(&units) else {
            panic!("`acvus check` accepts a str added to an i64");
        };

        assert_eq!(batch, ["type mismatch in `+`: i64 vs str"]);
        assert_eq!(messages(&workspace.diagnostics()[&script]), batch);
    }

    /// RFC-0043: `&k`, whose type the refused `+` made poison, is admitted
    /// by every `len`, and the call, poison too, reports nothing of its own.
    #[test]
    fn a_poison_argument_to_len_reports_only_the_refusal_it_came_from() {
        let tree = Tree::new();
        let source = "let m = @n + 1;\nlet k = m + \"x\";\nlen(&k)\n";
        let script = tree.write("a/main.acvus", source);
        let interner = Interner::new();
        let workspace = tree.workspace(&interner);

        let units = [compile::Unit {
            role: compile::Role::Entry("main".to_string()),
            space: None,
            path: script.display().to_string(),
            mode: compile::Mode::Script,
            text: source.to_string(),
        }];
        let Err(batch) = batch(&units) else {
            panic!("`acvus check` accepts a str added to an i64");
        };

        assert_eq!(batch, ["type mismatch in `+`: i64 vs str"]);
        assert_eq!(messages(&workspace.diagnostics()[&script]), batch);
    }

    #[test]
    fn each_script_is_its_own_compilation() {
        let tree = Tree::new();
        let pure = tree.write("a/pure.acvus", PURE);
        let reads = tree.write("a/reads.acvus", ADDS_ONE);
        let broken = tree.write("a/broken.acvus", "let m = 1 + \"x\";\nm\n");
        let workspace = tree.workspace(&Interner::new());
        let diagnostics = workspace.diagnostics();

        assert_eq!(diagnostics.keys().collect::<Vec<_>>(), [&broken]);
        assert!(held_open(&workspace, &pure));
        assert!(held_open(&workspace, &reads));
        assert!(diagnostics[&broken].iter().all(is_type));
    }

    #[test]
    fn hidden_directories_and_target_hold_no_sources() {
        let tree = Tree::new();
        let script = tree.write("a/main.acvus", PURE);
        tree.write(".hidden/main.acvus", "let\n");
        tree.write("target/main.acvus", "let\n");
        tree.write("a/target/main.acvus", "let\n");
        let workspace = tree.workspace(&Interner::new());
        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(held_open(&workspace, &script));
    }

    #[test]
    fn a_template_is_checked_in_template_mode() {
        let tree = Tree::new();
        let template = tree.write("a/main.acvt", "Hello, {{ &@name }}.\n");
        let workspace = tree.workspace(&Interner::new());
        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(held_open(&workspace, &template));
    }

    struct Locked(PathBuf);

    impl Drop for Locked {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755))
                .expect("restore the locked path's permissions");
        }
    }

    #[test]
    fn an_unreadable_directory_is_a_host_diagnostic_and_the_walk_goes_on() {
        use std::os::unix::fs::PermissionsExt;

        let tree = Tree::new();
        let script = tree.write("a/main.acvus", PURE);
        tree.write("b/main.acvus", "let\n");
        let locked = Locked(tree.root.join("b"));
        std::fs::set_permissions(&locked.0, std::fs::Permissions::from_mode(0o000))
            .expect("lock the directory");
        let listing_error = std::fs::read_dir(&locked.0)
            .expect_err("a directory with mode 0o000 does not list")
            .to_string();

        let workspace = tree.workspace(&Interner::new());
        let diagnostics = workspace.diagnostics();

        assert_eq!(diagnostics.keys().collect::<Vec<_>>(), [&locked.0]);
        assert!(held_open(&workspace, &script));
        let errors = &diagnostics[&locked.0];
        assert!(
            matches!(errors.as_slice(), [error] if is_host(error)),
            "{errors:?}"
        );
        assert_eq!(messages(errors), [listing_error]);
        assert_eq!(errors[0].span(), None);
    }

    #[test]
    fn an_unreadable_source_is_refused_in_the_words_of_acvus_check() {
        use std::os::unix::fs::PermissionsExt;

        let tree = Tree::new();
        let locked = Locked(tree.write("a/main.acvus", PURE));
        std::fs::set_permissions(&locked.0, std::fs::Permissions::from_mode(0o000))
            .expect("lock the script");
        let read_error =
            std::fs::read_to_string(&locked.0).expect_err("a file with mode 0o000 does not read");

        let diagnostics = tree.workspace(&Interner::new()).diagnostics();

        assert_eq!(
            diagnostics[&locked.0],
            [LspError {
                kind: LspErrorKind::Host(None),
                message: compile::unreadable_source(&locked.0, &read_error),
                related: Vec::new(),
            }]
        );
    }

    #[test]
    fn a_context_has_no_definition_site() {
        let tree = Tree::new();
        let main = tree.write("a/main.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        let at = ADDS_ONE.find("@n").expect("the script reads `@n`") + 1;
        assert_eq!(workspace.definition(&main, at), None);
    }

    #[test]
    fn a_script_written_under_the_root_appears_after_it_changes() {
        let tree = Tree::new();
        let first = tree.write("a/main.acvus", PURE);
        let mut workspace = tree.workspace(&Interner::new());
        assert!(held_open(&workspace, &first));

        let second = tree.write("a/second.acvus", PURE);
        assert!(!held_open(&workspace, &second));
        workspace.file_changed(&second);
        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(held_open(&workspace, &first));
        assert!(held_open(&workspace, &second));

        let nested = tree.write("b/c/third.acvus", PURE);
        assert!(!held_open(&workspace, &nested));
        workspace.file_changed(&tree.root.join("b"));
        assert!(held_open(&workspace, &nested));
    }

    /// The editor refuses each example's main in the words `acvus check
    /// <file>` refuses it in, and accepts it where `acvus check` does.
    #[test]
    fn every_example_is_refused_in_the_words_acvus_check_refuses_it_in() {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
        let sources: Vec<PathBuf> = std::fs::read_dir(&examples)
            .expect("the examples directory lists")
            .map(|entry| entry.expect("an example directory entry").path())
            .flat_map(|dir| [dir.join("main.acvus"), dir.join("main.acvt")])
            .filter(|path| path.exists())
            .collect();
        let interner = Interner::new();
        let workspace = Workspace::new(&interner, CliHost::new(examples));
        let diagnostics = workspace.diagnostics();

        assert!(!sources.is_empty());
        for source in &sources {
            let mode = compile_mode(mode_of(source).expect("an example is a script or a template"));
            let units = [compile::Unit {
                role: compile::Role::Entry("main".to_string()),
                space: None,
                path: source.display().to_string(),
                mode,
                text: std::fs::read_to_string(source).expect("an example reads"),
            }];
            let batch: Vec<String> = match batch(&units) {
                Ok(()) => Vec::new(),
                Err(messages) => messages,
            };
            let editor = diagnostics.get(source).map_or_else(Vec::new, |errors| messages(errors));
            assert_eq!(editor, batch, "{source:?}: the editor and `acvus check` disagree");
        }
    }
}
