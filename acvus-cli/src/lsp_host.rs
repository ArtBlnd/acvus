//! The editor's view of what `acvus check` checks: every source under a
//! root is one compilation, in the environment `compile::environment` builds
//! from the `ctx.json` beside it.

use std::io;
use std::path::{Path, PathBuf};

use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, Environment, Host,
    HostDiagnostic, Listing, Mode, Vfs,
};
use acvus_mir::graph::Bindings;
use acvus_utils::Interner;

use crate::{compile, context};

const CONTEXT_FILE: &str = "ctx.json";

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
    context: PathBuf,
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

fn sources_under(dir: &Path, sources: &mut Vec<Source>, refusals: &mut Vec<HostDiagnostic>) {
    let entries = match std::fs::read_dir(dir) {
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
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                refusals.extend(refused(&path, error.to_string()));
                continue;
            }
        };
        if file_type.is_dir() {
            if !skipped(&entry.file_name()) {
                sources_under(&path, sources, refusals);
            }
            continue;
        }
        if let Some(mode) = mode_of(&path) {
            sources.push(Source {
                path,
                mode,
                context: dir.join(CONTEXT_FILE),
            });
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

fn environment_of(
    interner: &Interner,
    vfs: &Vfs,
    source: &Source,
) -> Result<Environment<()>, Vec<HostDiagnostic>> {
    let loaded = match vfs.read(&source.context) {
        Ok(text) => context::from_text(interner, &text)
            .map_err(|message| refused(&source.context, message))?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => context::Loaded::default(),
        Err(error) => return Err(refused(&source.context, error.to_string())),
    };
    compile::environment(
        interner,
        &loaded.types,
        Bindings::default(),
        crate::cli_registries(),
    )
    .map(|environment| Environment {
        graph: environment.graph,
        host: (),
    })
    .map_err(|error| refused(&source.path, compile::combine_refusal(&error)))
}

impl Host for CliHost {
    type Compilation = ();

    fn compilations(&mut self, interner: &Interner, vfs: &Vfs) -> Listing<()> {
        let mut sources = Vec::new();
        let mut refusals = Vec::new();
        sources_under(&self.root, &mut sources, &mut refusals);
        let compilations = sources
            .into_iter()
            .map(|source| CompilationSpec {
                id: CompilationId(source.path.clone()),
                environment: environment_of(interner, vfs, &source),
                documents: vec![DocumentSpec {
                    path: source.path,
                    document: Document {
                        qref: compile::entry_ref(interner),
                        mode: source.mode,
                        ty: compile::entry_ty(),
                    },
                }],
            })
            .collect();
        Listing {
            compilations,
            refusals,
        }
    }

    fn check(&self, _compilation: &(), _checked: &Checked<'_>) -> Vec<HostDiagnostic> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use acvus_lsp::{LspError, LspErrorCategory, Workspace};
    use acvus_mir::graph::optimize::Opt;

    use super::*;
    use crate::compile::Timed;

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

    fn categories(errors: &[LspError]) -> Vec<LspErrorCategory> {
        errors.iter().map(|error| error.category).collect()
    }

    #[test]
    fn a_script_that_reads_its_context_file_is_accepted() {
        let tree = Tree::new();
        tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        assert_eq!(
            workspace.diagnostics(),
            BTreeMap::from([(script, Vec::new())])
        );
    }

    #[test]
    fn a_type_error_is_reported_as_acvus_check_reports_it() {
        let tree = Tree::new();
        let context = r#"{ "n": 1 }"#;
        let source = "let m = @n + \"x\";\n\"{{ &m | to_string }}\"\n";
        tree.write("a/ctx.json", context);
        let script = tree.write("a/main.acvus", source);
        let interner = Interner::new();
        let workspace = tree.workspace(&interner);

        let loaded = context::from_text(&interner, context).expect("the context file loads");
        let Err(batch) = compile::check(
            &interner,
            source,
            compile::Mode::Script,
            &loaded.types,
            Bindings::default(),
            crate::cli_registries(),
            Timed::Off,
            Opt::Full,
        ) else {
            panic!("`acvus check` accepts a String added to an Int");
        };
        let batch: Vec<String> = batch.into_iter().map(|d| d.message).collect();

        let diagnostics = workspace.diagnostics();
        let editor = &diagnostics[&script];
        assert!(!batch.is_empty());
        assert_eq!(messages(editor), batch);
        assert!(
            categories(editor)
                .iter()
                .all(|category| *category == LspErrorCategory::Type)
        );
    }

    #[test]
    fn a_broken_context_file_is_a_host_diagnostic_at_that_file() {
        let tree = Tree::new();
        let unparsed = tree.write("a/ctx.json", r#"{ "n": "#);
        let unparsed_script = tree.write("a/main.acvus", ADDS_ONE);
        let untyped = tree.write("b/ctx.json", r#"{ "n": null }"#);
        let untyped_script = tree.write("b/main.acvus", ADDS_ONE);
        let diagnostics = tree.workspace(&Interner::new()).diagnostics();

        for context in [&unparsed, &untyped] {
            let errors = &diagnostics[context];
            assert_eq!(categories(errors), [LspErrorCategory::Host], "{context:?}");
            assert_eq!(errors[0].span, None);
        }
        assert_eq!(messages(&diagnostics[&untyped]), ["@n: null has no type"]);
        assert_eq!(diagnostics[&unparsed_script], []);
        assert_eq!(diagnostics[&untyped_script], []);
    }

    #[test]
    fn editing_the_context_file_rechecks_the_script() {
        let tree = Tree::new();
        let context = tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let mut workspace = tree.workspace(&Interner::new());
        assert_eq!(workspace.diagnostics()[&script], []);

        workspace.set_buffer(context.clone(), r#"{ "n": "one" }"#.to_string());
        let errors = &workspace.diagnostics()[&script];
        assert!(!errors.is_empty());
        assert!(
            categories(errors)
                .iter()
                .all(|category| *category == LspErrorCategory::Type)
        );

        workspace.set_buffer(context.clone(), r#"{ "n": 2 }"#.to_string());
        assert_eq!(workspace.diagnostics()[&script], []);

        workspace.set_buffer(context.clone(), r#"{ "n": "one" }"#.to_string());
        assert!(!workspace.diagnostics()[&script].is_empty());
        workspace.drop_buffer(&context);
        assert_eq!(workspace.diagnostics()[&script], []);
    }

    #[test]
    fn a_script_without_a_context_file_has_no_contexts() {
        let tree = Tree::new();
        let pure = tree.write("a/pure.acvus", PURE);
        let reads = tree.write("a/reads.acvus", ADDS_ONE);
        let diagnostics = tree.workspace(&Interner::new()).diagnostics();

        assert_eq!(diagnostics.keys().collect::<Vec<_>>(), [&pure, &reads]);
        assert_eq!(diagnostics[&pure], []);
        assert!(!diagnostics[&reads].is_empty());
        assert!(
            categories(&diagnostics[&reads])
                .iter()
                .all(|category| *category == LspErrorCategory::Type)
        );
    }

    #[test]
    fn hidden_directories_and_target_hold_no_sources() {
        let tree = Tree::new();
        let script = tree.write("a/main.acvus", PURE);
        tree.write(".hidden/main.acvus", "let\n");
        tree.write("target/main.acvus", "let\n");
        tree.write("a/target/main.acvus", "let\n");
        let diagnostics = tree.workspace(&Interner::new()).diagnostics();
        assert_eq!(diagnostics, BTreeMap::from([(script, Vec::new())]));
    }

    #[test]
    fn a_template_is_checked_in_template_mode() {
        let tree = Tree::new();
        tree.write("a/ctx.json", r#"{ "name": "Ada" }"#);
        let template = tree.write("a/main.acvt", "Hello, {{ &@name }}.\n");
        let diagnostics = tree.workspace(&Interner::new()).diagnostics();
        assert_eq!(diagnostics, BTreeMap::from([(template, Vec::new())]));
    }

    struct Locked(PathBuf);

    impl Drop for Locked {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755))
                .expect("restore the locked directory's permissions");
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

        let diagnostics = tree.workspace(&Interner::new()).diagnostics();

        assert_eq!(diagnostics.keys().collect::<Vec<_>>(), [&script, &locked.0]);
        assert_eq!(diagnostics[&script], []);
        let errors = &diagnostics[&locked.0];
        assert_eq!(categories(errors), [LspErrorCategory::Host]);
        assert_eq!(messages(errors), [listing_error]);
        assert_eq!(errors[0].span, None);
    }

    #[test]
    fn every_example_is_accepted() {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
        let sources: Vec<PathBuf> = std::fs::read_dir(&examples)
            .expect("the examples directory lists")
            .map(|entry| entry.expect("an example directory entry").path())
            .flat_map(|dir| [dir.join("main.acvus"), dir.join("main.acvt")])
            .filter(|path| path.exists())
            .collect();
        let diagnostics = Workspace::new(&Interner::new(), CliHost::new(examples)).diagnostics();

        assert_eq!(diagnostics.len(), sources.len());
        for source in &sources {
            assert_eq!(diagnostics[source], [], "{source:?}");
        }
    }
}
