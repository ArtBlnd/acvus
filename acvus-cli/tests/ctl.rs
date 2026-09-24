//! `acvus ctl` and a run over a space at the CLI's contract (RFC-0031 rules
//! 2, 3 and 9): a space holds its scripts' and inits' sources, a context
//! gets its first value from its init when a run fetches it and the space
//! lacks it, and every refusal names the command that resolves it.

use std::path::{Path, PathBuf};
use std::process::Output;

use crate::sandbox;

struct Sandbox {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let dir = sandbox::tempdir();
        let root = dir.path().to_path_buf();
        std::fs::create_dir_all(root.join("work")).expect("make the working directory");
        Sandbox { _dir: dir, root }
    }

    fn config(&self) -> PathBuf {
        self.root.join("config")
    }

    fn work(&self) -> PathBuf {
        self.root.join("work")
    }

    fn write(&self, relative: &str, text: &str) -> PathBuf {
        let path = self.work().join(relative);
        std::fs::create_dir_all(path.parent().expect("a file has a directory"))
            .expect("make the file's directory");
        std::fs::write(&path, text).expect("write a fixture");
        path
    }

    fn acvus_in(&self, cwd: &Path, args: &[&str]) -> Output {
        sandbox::with_config(&self.config())
            .current_dir(cwd)
            .args(args)
            .output()
            .expect("the binary runs")
    }

    fn acvus(&self, args: &[&str]) -> Output {
        self.acvus_in(&self.work(), args)
    }

    fn ok_in(&self, cwd: &Path, args: &[&str]) -> String {
        let out = self.acvus_in(cwd, args);
        assert_eq!(
            out.status.code(),
            Some(0),
            "`acvus {}`:\n{}",
            args.join(" "),
            text(&out.stderr)
        );
        text(&out.stdout)
    }

    fn ok(&self, args: &[&str]) -> String {
        self.ok_in(&self.work(), args)
    }

    fn refused(&self, args: &[&str], code: i32) -> String {
        let out = self.acvus(args);
        assert_eq!(
            out.status.code(),
            Some(code),
            "`acvus {}`:\n{}",
            args.join(" "),
            text(&out.stderr)
        );
        assert_eq!(text(&out.stdout), "", "`acvus {}`", args.join(" "));
        text(&out.stderr)
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

const TURN: &str = "@notes.push_back(concat(\"\", $note));\n@notes.len()\n";

fn log_lines(listing: &str) -> Vec<&str> {
    listing.lines().filter(|line| line.starts_with('@')).collect()
}

#[test]
fn a_run_on_an_empty_space_fills_the_context_from_its_init_and_commits_it() {
    let sandbox = Sandbox::new();
    let turn = sandbox.write("turn.acvus", TURN);
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "notes", "dir:notes"]);
    sandbox.ok(&["ctl", "space", "add-script", "notes", turn.to_str().unwrap()]);

    let refusal = sandbox.refused(&["run", "turn", "--space", "notes", "note=\"first\""], 2);
    assert_eq!(
        refusal,
        "error: `@notes` is not in space `notes` yet and has no init; `acvus ctl space init notes notes -e <expr>` stores one\n"
    );
    let listing = sandbox.ok(&["ctl", "space", "ls", "notes"]);
    assert!(log_lines(&listing).is_empty(), "the refused run committed nothing:\n{listing}");

    assert_eq!(
        sandbox.ok(&["ctl", "space", "init", "notes", "notes", "-e", "deque()"]),
        "space `notes` holds the init of `@notes`\n"
    );
    let out = sandbox.acvus(&["run", "turn", "--space", "notes", "note=\"first\""]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "1\n");
    let stderr = text(&out.stderr);
    let lines: Vec<&str> = stderr.lines().collect();
    assert_eq!(lines[0], "init @notes", "{stderr}");
    assert!(lines[1].starts_with("commit @notes = "), "{stderr}");
    assert_eq!(lines.len(), 2, "{stderr}");

    let out = sandbox.acvus(&["run", "turn", "--space", "notes", "note=\"second\""]);
    assert_eq!(text(&out.stdout), "2\n");
    assert!(!text(&out.stderr).contains("init @notes"), "the init ran again: {}", text(&out.stderr));

    let listing = sandbox.ok(&["ctl", "space", "ls", "notes"]);
    let lines: Vec<&str> = listing.lines().collect();
    assert_eq!(lines[0], format!("space `notes` = dir:{}", sandbox.work().join("notes").display()));
    assert_eq!(lines[1..3], ["script turn.acvus", "init @notes (acvus)"]);
    assert!(lines[3].starts_with("@notes: Deque<String> = "), "{listing}");
    assert_eq!(lines.len(), 5, "{listing}");
}

#[test]
fn an_init_is_stored_from_an_expression_or_a_file_and_removed() {
    let sandbox = Sandbox::new();
    let init = sandbox.write("elsewhere/greeting.acvus", "let g = \"hi\".to_string();\ng\n");
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    sandbox.ok(&["ctl", "space", "init", "s", "n", "-e", "41"]);
    sandbox.ok(&["ctl", "space", "init", "s", "greeting", "-f", init.to_str().unwrap()]);
    std::fs::remove_dir_all(sandbox.work().join("elsewhere")).expect("remove the source");

    assert_eq!(
        std::fs::read_to_string(sandbox.work().join("store/inits/n.acvus"))
            .expect("the space holds the expression"),
        "41"
    );
    assert_eq!(
        std::fs::read_to_string(sandbox.work().join("store/inits/greeting.acvus"))
            .expect("the space holds the file's source"),
        "let g = \"hi\".to_string();\ng\n"
    );
    assert_eq!(sandbox.ok(&["run", "-e", "@n + 1", "--space", "s"]), "42\n");
    assert_eq!(sandbox.ok(&["run", "-e", "@greeting", "--space", "s"]), "hi\n");

    assert_eq!(
        sandbox.ok(&["ctl", "space", "init", "s", "n", "-e", "7"]),
        "space `s` holds the init of `@n`, replacing the one it held\n"
    );
    assert_eq!(sandbox.ok(&["run", "-e", "@n", "--space", "s"]), "41\n", "a held value is kept");
    sandbox.ok(&["ctl", "space", "init", "--rm", "s", "greeting"]);
    let listing = sandbox.ok(&["ctl", "space", "ls", "s"]);
    assert!(listing.contains("init @n (acvus)\n"), "{listing}");
    assert!(!listing.contains("init @greeting"), "{listing}");
    let gone = sandbox.refused(&["ctl", "space", "init", "--rm", "s", "greeting"], 64);
    assert!(gone.contains("`acvus ctl space ls s`"), "{gone}");
}

#[test]
fn fill_runs_every_init_the_space_lacks_and_commits() {
    let sandbox = Sandbox::new();
    sandbox.write("sum.acvus", "@a + @b\n");
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    sandbox.ok(&["ctl", "space", "add-script", "s", "sum.acvus"]);
    sandbox.ok(&["ctl", "space", "init", "s", "a", "-e", "1"]);
    sandbox.ok(&["ctl", "space", "init", "s", "b", "-e", "2"]);

    let out = sandbox.acvus(&["ctl", "space", "fill", "s"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let stderr = text(&out.stderr);
    assert!(stderr.starts_with("init @a\ninit @b\ncommit @a = "), "{stderr}");
    let listing = sandbox.ok(&["ctl", "space", "ls", "s"]);
    assert_eq!(log_lines(&listing).len(), 2, "{listing}");

    sandbox.ok(&["ctl", "space", "init", "s", "b", "-e", "20"]);
    assert_eq!(
        sandbox.ok(&["ctl", "space", "fill", "s"]),
        "space `s` holds every context it has an init of\n"
    );
    assert_eq!(sandbox.ok(&["run", "sum", "--space", "s"]), "3\n");
}

#[test]
fn one_space_name_means_a_location_per_ctl_context() {
    let sandbox = Sandbox::new();
    let read = sandbox.write("read.acvus", "@n\n");
    let read = read.to_str().unwrap();
    for (context, dir, value) in [("a", "dir:a-store", "1"), ("b", "dir:b-store", "2")] {
        sandbox.ok(&["ctl", "use", context]);
        sandbox.ok(&["ctl", "space", "add", "s", dir]);
        sandbox.ok(&["ctl", "space", "init", "s", "n", "-e", value]);
        sandbox.ok(&["ctl", "space", "add-script", "s", read]);
    }
    assert_eq!(sandbox.ok(&["run", "read", "--space", "s"]), "2\n");
    sandbox.ok(&["ctl", "use", "a"]);
    assert_eq!(sandbox.ok(&["run", "read", "--space", "s"]), "1\n");

    for (store, value) in [("a-store", "1"), ("b-store", "2")] {
        assert_eq!(
            std::fs::read_to_string(sandbox.work().join(store).join("inits/n.acvus"))
                .expect("each location holds its own init"),
            value
        );
    }
    assert_eq!(
        sandbox.ok(&["ctl", "space", "ls"]),
        format!("ctl context `a`\ns = dir:{}\n", sandbox.work().join("a-store").display())
    );
}

#[test]
fn the_nearest_acvus_directory_names_the_space() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    sandbox.ok(&["ctl", "space", "init", "s", "n", "-e", "5"]);
    let project = sandbox.work().join("project");
    let deep = project.join("deep").join("er");
    std::fs::create_dir_all(&deep).expect("make the subdirectory");
    sandbox.ok_in(&project, &["ctl", "space", "mark", "s"]);

    assert_eq!(sandbox.ok_in(&deep, &["run", "-e", "@n + 1"]), "6\n");
    let shown = sandbox.ok_in(&deep, &["ctl", "show"]);
    assert!(
        shown.contains(&format!(
            "space: s = dir:{} (from {})",
            sandbox.work().join("store").display(),
            project.join(".acvus").display()
        )),
        "{shown}"
    );
}

/// A script that stores its context keeps today's rules: it gives the value
/// on a page that lacks it, and no init is involved.
#[test]
fn an_expression_that_stores_a_context_compiles_beside_the_space_s_scripts() {
    let sandbox = Sandbox::new();
    sandbox.write("turn.acvus", TURN);
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "notes", "dir:notes"]);
    sandbox.ok(&["ctl", "space", "add-script", "notes", "turn.acvus"]);
    sandbox.ok(&["ctl", "space", "init", "notes", "notes", "-e", "deque()"]);

    sandbox.ok(&["run", "-e", "@notes = deque();", "--space", "notes"]);
    assert_eq!(
        sandbox.ok(&["run", "turn", "--space", "notes", "note=\"x\""]),
        "1\n"
    );
    assert_eq!(sandbox.ok(&["run", "-e", "@notes.len()", "--space", "notes"]), "1\n");
}

#[test]
fn an_init_that_names_a_context_is_refused_naming_its_key() {
    let sandbox = Sandbox::new();
    sandbox.write("read.acvus", "@a + @b\n");
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    sandbox.ok(&["ctl", "space", "add-script", "s", "read.acvus"]);
    sandbox.ok(&["ctl", "space", "init", "s", "a", "-e", "@b + 1"]);
    sandbox.ok(&["ctl", "space", "init", "s", "b", "-e", "1"]);

    let refusal = sandbox.refused(&["run", "read", "--space", "s"], 1);
    assert!(
        refusal.contains("the init of `@a` names `@b`, and an init names no context"),
        "{refusal}"
    );
    assert!(refusal.contains("s/inits/a.acvus"), "{refusal}");
}

#[test]
fn a_space_s_default_is_shown_with_its_source_and_a_flag_overrides_it() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["ctl", "use", "work"]);
    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    sandbox.ok(&["ctl", "set", "parallel", "tokio", "--space", "s"]);
    sandbox.ok(&["ctl", "set", "opt", "none"]);

    let shown = sandbox.ok(&["ctl", "show", "--space", "s"]);
    assert!(shown.contains("parallel = tokio (space)\n"), "{shown}");
    assert!(shown.contains("opt = none (context)\n"), "{shown}");
    assert!(shown.contains("time = off (built-in)\n"), "{shown}");
    let shown = sandbox.ok(&["ctl", "show", "--space", "s", "--parallel=sequential"]);
    assert!(shown.contains("parallel = sequential (flag)\n"), "{shown}");
    let shown = sandbox.ok(&["ctl", "show", "--parallel"]);
    assert!(shown.contains("parallel = tokio (flag)\n"), "{shown}");
    let shown = sandbox.ok(&["ctl", "show"]);
    assert!(shown.contains("parallel = sequential (built-in)\n"), "{shown}");

    sandbox.ok(&["ctl", "set", "time", "on", "--space", "s"]);
    let timed = sandbox.acvus(&["run", "-e", "1", "--space", "s"]);
    assert_eq!(timed.status.code(), Some(0), "{}", text(&timed.stderr));
    assert!(text(&timed.stderr).contains("time: compile "), "{}", text(&timed.stderr));
    assert!(text(&timed.stderr).contains(" at opt none "), "{}", text(&timed.stderr));
    let untimed = sandbox.acvus(&["run", "-e", "1", "--space", "s", "--time=off"]);
    assert_eq!(untimed.status.code(), Some(0), "{}", text(&untimed.stderr));
    assert_eq!(text(&untimed.stderr), "");

    sandbox.ok(&["ctl", "unset", "parallel", "--space", "s"]);
    let shown = sandbox.ok(&["ctl", "show", "--space", "s"]);
    assert!(shown.contains("parallel = sequential (built-in)\n"), "{shown}");
}

#[test]
fn every_refusal_names_the_command_that_resolves_it() {
    let sandbox = Sandbox::new();
    sandbox.write("reads.acvus", "@n + 1\n");

    let no_config = sandbox.refused(&["run", "x", "--space", "s"], 64);
    assert!(no_config.contains("`acvus ctl use <context>`"), "{no_config}");

    let no_context = sandbox.refused(&["run", "reads.acvus"], 64);
    assert!(
        no_context.contains("`@n` is kept in a space, and no space is named here; `acvus ctl space add <space> dir:<path>`"),
        "{no_context}"
    );

    sandbox.ok(&["ctl", "use", "work"]);
    let unknown = sandbox.refused(&["run", "x", "--space", "nope"], 64);
    assert!(unknown.contains("`acvus ctl space add nope dir:<path>`"), "{unknown}");

    sandbox.ok(&["ctl", "space", "add", "s", "dir:store"]);
    let not_held = sandbox.refused(&["run", "x", "--space", "s"], 64);
    assert!(not_held.contains("`acvus ctl space add-script s <file>`"), "{not_held}");

    sandbox.ok(&["ctl", "space", "add-script", "s", "reads.acvus"]);
    let no_init = sandbox.refused(&["run", "reads", "--space", "s"], 2);
    assert_eq!(
        no_init,
        "error: `@n` is not in space `s` yet and has no init; `acvus ctl space init s n -e <expr>` stores one\n"
    );

    let unbound = sandbox.refused(&["run", "-e", "$x + 1"], 1);
    assert!(unbound.contains("`x=<literal>` binds it"), "{unbound}");

    let removed = sandbox.refused(&["ctl", "space", "rm-script", "s", "nope"], 64);
    assert!(removed.contains("`acvus ctl space ls s`"), "{removed}");

    let twice = sandbox.refused(&["ctl", "space", "add", "s", "dir:elsewhere"], 64);
    assert!(twice.contains("`acvus ctl space rm s`"), "{twice}");

    let no_source = sandbox.refused(&["ctl", "space", "init", "s", "n"], 64);
    assert!(no_source.contains("usage: acvus ctl space init <space> <key> -e <expr>"), "{no_source}");
    let bad_key = sandbox.refused(&["ctl", "space", "init", "s", "1n", "-e", "1"], 64);
    assert!(bad_key.contains("is not a context's name"), "{bad_key}");
    let marked = sandbox.refused(&["ctl", "space", "init", "s"], 64);
    assert!(marked.contains("usage: acvus ctl space init"), "{marked}");
}

#[test]
fn a_context_free_source_runs_without_any_config() {
    let sandbox = Sandbox::new();
    sandbox.write("pure.acvus", "let xs = [1, 2];\nxs.len() * 10\n");
    assert_eq!(sandbox.ok(&["run", "pure.acvus"]), "20\n");
    assert_eq!(sandbox.ok(&["run", "-e", "1 + 2"]), "3\n");
    assert!(!sandbox.config().exists(), "a run wrote a config");
}
