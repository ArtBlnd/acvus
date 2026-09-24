//! The editor's view of what `acvus check` checks: every source under a
//! root is one compilation, in the environment `compile::environment` builds
//! from the `ctx.json` beside it.

use std::io;
use std::path::{Path, PathBuf};

use acvus_ast::Span;
use acvus_lsp::{
    Checked, CompilationId, CompilationSpec, Document, DocumentSpec, EntryKind, Environment, Host,
    HostDiagnostic, Listing, Location, Mode, RecordingReader, Sites, Vfs,
};
use acvus_mir::graph::{Bindings, QualifiedRef};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

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

#[derive(Debug, PartialEq, Eq)]
struct Ungrammatical {
    at: usize,
}

/// Follows the JSON grammar of RFC 8259, the one serde_json parses, so
/// every context file `context::from_text` accepts scans.
struct JsonKeyScanner<'t> {
    bytes: &'t [u8],
    at: usize,
}

impl JsonKeyScanner<'_> {
    fn refused_here(&self) -> Ungrammatical {
        Ungrammatical { at: self.at }
    }

    fn refused_last(&self) -> Ungrammatical {
        Ungrammatical { at: self.at - 1 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn next(&mut self) -> Result<u8, Ungrammatical> {
        let byte = self.peek().ok_or_else(|| self.refused_here())?;
        self.at += 1;
        Ok(byte)
    }

    fn eat(&mut self, expected: u8) -> Result<(), Ungrammatical> {
        match self.next()? {
            byte if byte == expected => Ok(()),
            _ => Err(self.refused_last()),
        }
    }

    fn eat_if(&mut self, expected: u8) -> bool {
        let eaten = self.peek() == Some(expected);
        if eaten {
            self.at += 1;
        }
        eaten
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn value(&mut self) -> Result<(), Ungrammatical> {
        self.whitespace();
        match self.peek().ok_or_else(|| self.refused_here())? {
            b'{' => self.object(|_| {}),
            b'[' => self.array(),
            b'"' => self.string_literal().map(|_| ()),
            b't' => self.literal(b"true"),
            b'f' => self.literal(b"false"),
            b'n' => self.literal(b"null"),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.refused_here()),
        }
    }

    fn object<F>(&mut self, mut key: F) -> Result<(), Ungrammatical>
    where
        F: FnMut((usize, usize)),
    {
        self.eat(b'{')?;
        self.whitespace();
        if self.eat_if(b'}') {
            return Ok(());
        }
        loop {
            self.whitespace();
            key(self.string_literal()?);
            self.whitespace();
            self.eat(b':')?;
            self.value()?;
            self.whitespace();
            match self.next()? {
                b',' => {}
                b'}' => return Ok(()),
                _ => return Err(self.refused_last()),
            }
        }
    }

    fn array(&mut self) -> Result<(), Ungrammatical> {
        self.eat(b'[')?;
        self.whitespace();
        if self.eat_if(b']') {
            return Ok(());
        }
        loop {
            self.value()?;
            self.whitespace();
            match self.next()? {
                b',' => {}
                b']' => return Ok(()),
                _ => return Err(self.refused_last()),
            }
        }
    }

    fn string_literal(&mut self) -> Result<(usize, usize), Ungrammatical> {
        let start = self.at;
        self.eat(b'"')?;
        loop {
            match self.next()? {
                b'"' => return Ok((start, self.at)),
                b'\\' => match self.next()? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                    b'u' => {
                        for _ in 0..4 {
                            if !self.next()?.is_ascii_hexdigit() {
                                return Err(self.refused_last());
                            }
                        }
                    }
                    _ => return Err(self.refused_last()),
                },
                0x00..=0x1f => return Err(self.refused_last()),
                _ => {}
            }
        }
    }

    fn literal(&mut self, word: &[u8]) -> Result<(), Ungrammatical> {
        for &expected in word {
            self.eat(expected)?;
        }
        Ok(())
    }

    fn number(&mut self) -> Result<(), Ungrammatical> {
        self.eat_if(b'-');
        match self.next()? {
            b'0' => {}
            b'1'..=b'9' => self.digits(),
            _ => return Err(self.refused_last()),
        }
        if self.eat_if(b'.') {
            self.some_digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            self.some_digits()?;
        }
        Ok(())
    }

    fn digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
    }

    fn some_digits(&mut self) -> Result<(), Ungrammatical> {
        match self.peek() {
            Some(b'0'..=b'9') => {
                self.digits();
                Ok(())
            }
            _ => Err(self.refused_here()),
        }
    }
}

fn top_level_key_literals(text: &str) -> Result<Vec<(usize, usize)>, Ungrammatical> {
    let mut scanner = JsonKeyScanner {
        bytes: text.as_bytes(),
        at: 0,
    };
    let mut keys = Vec::new();
    scanner.whitespace();
    scanner.object(|key| keys.push(key))?;
    scanner.whitespace();
    match scanner.at == text.len() {
        true => Ok(keys),
        false => Err(scanner.refused_here()),
    }
}

/// A key written twice is declared where it is written last, as serde_json
/// keeps the last value of a repeated key and that is the one loaded.
fn load_with_sites(
    interner: &Interner,
    path: &Path,
    text: &str,
) -> Result<(context::Loaded, Sites), String> {
    let loaded = context::from_text(interner, text)?;
    let keys = top_level_key_literals(text).unwrap_or_else(|Ungrammatical { at }| {
        panic!("byte {at} of a context file serde_json accepted is refused by the JSON grammar")
    });
    let contexts = keys
        .into_iter()
        .map(|(start, end)| {
            let name: String = serde_json::from_str(&text[start..end])
                .expect("a key literal serde_json accepted decodes");
            let site = Location {
                path: path.to_path_buf(),
                span: Span::new(start, end),
            };
            (QualifiedRef::root(interner.intern(&name)), site)
        })
        .collect();
    let sites = Sites {
        contexts,
        inputs: FxHashMap::default(),
    };
    Ok((loaded, sites))
}

fn environment_of(
    interner: &Interner,
    reader: &RecordingReader<'_>,
    source: &Source,
) -> Result<Environment<()>, Vec<HostDiagnostic>> {
    let (loaded, sites) = match reader.read(&source.context) {
        Ok(text) => load_with_sites(interner, &source.context, &text)
            .map_err(|message| refused(&source.context, message))?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            (context::Loaded::default(), Sites::default())
        }
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
        sites,
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
                        ty: compile::entry_ty(),
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

    use acvus_lsp::{Location, LspError, LspErrorKind, Workspace};
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
    fn a_script_that_reads_its_context_file_is_accepted() {
        let tree = Tree::new();
        tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(held_open(&workspace, &script));
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
        assert!(editor.iter().all(is_type));
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
            assert!(
                matches!(errors.as_slice(), [error] if is_host(error)),
                "{context:?}: {errors:?}"
            );
            assert_eq!(errors[0].span(), None);
        }
        assert_eq!(messages(&diagnostics[&untyped]), ["@n: null has no type"]);
        assert!(!diagnostics.contains_key(&unparsed_script));
        assert!(!diagnostics.contains_key(&untyped_script));
    }

    #[test]
    fn editing_the_context_file_rechecks_the_script() {
        let tree = Tree::new();
        let context = tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let mut workspace = tree.workspace(&Interner::new());
        assert!(!workspace.diagnostics().contains_key(&script));

        workspace.set_buffer(context.clone(), r#"{ "n": "one" }"#.to_string());
        let errors = &workspace.diagnostics()[&script];
        assert!(!errors.is_empty());
        assert!(errors.iter().all(is_type));

        workspace.set_buffer(context.clone(), r#"{ "n": 2 }"#.to_string());
        assert!(!workspace.diagnostics().contains_key(&script));

        workspace.set_buffer(context.clone(), r#"{ "n": "one" }"#.to_string());
        assert!(!workspace.diagnostics()[&script].is_empty());
        workspace.drop_buffer(&context);
        assert!(!workspace.diagnostics().contains_key(&script));
    }

    #[test]
    fn a_script_without_a_context_file_has_no_contexts() {
        let tree = Tree::new();
        let pure = tree.write("a/pure.acvus", PURE);
        let reads = tree.write("a/reads.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        let diagnostics = workspace.diagnostics();

        assert_eq!(diagnostics.keys().collect::<Vec<_>>(), [&reads]);
        assert!(held_open(&workspace, &pure));
        assert!(!diagnostics[&reads].is_empty());
        assert!(diagnostics[&reads].iter().all(is_type));
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
        tree.write("a/ctx.json", r#"{ "name": "Ada" }"#);
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

    fn definition_of_n(tree: &Tree, script: &str) -> Option<Location> {
        let main = tree.write("a/main.acvus", script);
        let workspace = tree.workspace(&Interner::new());
        assert!(!workspace.diagnostics().contains_key(&main));
        let at = script.find("@n").expect("the script reads `@n`") + 1;
        workspace.definition(&main, at)
    }

    fn spelled(location: &Location) -> String {
        let text = std::fs::read_to_string(&location.path).expect("the site's file reads");
        text[location.span.start..location.span.end].to_string()
    }

    #[test]
    fn a_context_goes_to_its_key_in_the_context_file() {
        let tree = Tree::new();
        let text = r#"{"m": 1, "n": {"a": "x\"}"}}"#;
        let context = tree.write("a/ctx.json", text);
        let site = definition_of_n(&tree, "let s = @n.a;\n\"{{ s }}\"\n").expect("`@n` has a site");
        let start = text.find(r#""n""#).expect("the file spells the key");
        assert_eq!(
            site,
            Location {
                path: context,
                span: Span::new(start, start + 3),
            }
        );
        assert_eq!(spelled(&site), r#""n""#);
    }

    #[test]
    fn a_key_spelled_with_an_escape_is_the_context_it_decodes_to() {
        let tree = Tree::new();
        tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        let site = definition_of_n(&tree, ADDS_ONE).expect("`@n` has a site");
        assert_eq!(spelled(&site), r#""n""#);
    }

    #[test]
    fn a_key_written_twice_is_declared_where_it_is_written_last() {
        let tree = Tree::new();
        let text = r#"{ "n": "one", "n": 1 }"#;
        tree.write("a/ctx.json", text);
        let site = definition_of_n(&tree, ADDS_ONE).expect("`@n` has a site");
        let start = text.rfind(r#""n""#).expect("the file spells the key");
        assert_eq!(site.span, Span::new(start, start + 3));
    }

    #[test]
    fn a_context_without_a_context_file_has_no_definition() {
        let tree = Tree::new();
        let main = tree.write("a/main.acvus", ADDS_ONE);
        let workspace = tree.workspace(&Interner::new());
        let at = ADDS_ONE.find("@n").expect("the script reads `@n`") + 1;
        assert_eq!(workspace.definition(&main, at), None);
    }

    #[test]
    fn the_key_scanner_follows_the_json_grammar() {
        let text = r#" { "a" : [1, -2.5e+3, true, null], "b\"c": {"d": {}}, "e": "" } "#;
        let keys: Vec<&str> = top_level_key_literals(text)
            .expect("the text is JSON")
            .into_iter()
            .map(|(start, end)| &text[start..end])
            .collect();
        assert_eq!(keys, [r#""a""#, r#""b\"c""#, r#""e""#]);

        let refused = [
            (r#"{"a" 1}"#, 5),
            (r#"{"a": 01}"#, 7),
            (r#"{"a": "\x"}"#, 8),
            (r#"{"a": tru}"#, 9),
            (r#"{"a": [1,]}"#, 9),
            ("{\"a\": \"\t\"}", 7),
            (r#"{"a"#, 3),
            ("{} x", 3),
            ("[]", 0),
        ];
        for (text, at) in refused {
            assert_eq!(
                top_level_key_literals(text),
                Err(Ungrammatical { at }),
                "{text:?}"
            );
        }
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

    #[test]
    fn a_context_file_created_where_none_was_types_the_script() {
        let tree = Tree::new();
        let script = tree.write("a/main.acvus", ADDS_ONE);
        let mut workspace = tree.workspace(&Interner::new());
        assert!(!workspace.diagnostics()[&script].is_empty());

        let context = tree.write("a/ctx.json", r#"{ "n": 1 }"#);
        workspace.file_changed(&context);
        assert!(!workspace.diagnostics().contains_key(&script));
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
        let workspace = Workspace::new(&Interner::new(), CliHost::new(examples));

        assert_eq!(workspace.diagnostics(), BTreeMap::new());
        assert!(!sources.is_empty());
        for source in &sources {
            assert!(held_open(&workspace, source), "{source:?}");
        }
    }
}
