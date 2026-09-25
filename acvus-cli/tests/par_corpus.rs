//! The parallel-loop corpus (`acvus-interpreter-test/par_corpus`) at the
//! CLI's contract: every case runs to the value its row of `INDEX.md` states,
//! and `acvus mir` prints, for each of its `For`s and pull loops' `While`s,
//! the stages line, the facts `analysis::loop_deps` computes of each stage
//! and the cost `analysis::cost` computes against the interpreter's table
//! exactly as the case's `facts/<id>.facts` holds them. A change in any row's stages,
//! cycles, orders, laws or cost shows here as that row's difference.
//!
//! A case whose header states `contexts` runs in a space holding the script
//! and the inits under `inits/<id>/`, as `INDEX.md` says.

use std::path::{Path, PathBuf};
use std::process::Output;

use crate::sandbox;

/// The cases `INDEX.md`'s summary counts: 135 positive, 27 negative controls.
const CASES: usize = 135 + 27;

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lies in the workspace")
        .join("acvus-interpreter-test/par_corpus")
}

struct Case {
    id: String,
    path: PathBuf,
    contexts: bool,
    expected: String,
}

/// A table cell of `INDEX.md`: a `|` inside it is written `\|`.
fn cells(row: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut cell = String::new();
    let mut chars = row.trim().trim_start_matches('|').chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cell.push('|');
                chars.next();
            }
            '|' => found.push(std::mem::take(&mut cell).trim().to_string()),
            other => cell.push(other),
        }
    }
    found
}

/// Every case `INDEX.md` lists, with the expected result its last column
/// states (`\n` between lines).
fn cases() -> Vec<Case> {
    let dir = corpus_dir();
    let index = std::fs::read_to_string(dir.join("INDEX.md")).expect("the corpus index");
    let found: Vec<Case> = index
        .lines()
        .filter(|line| line.starts_with("| ["))
        .map(|row| {
            let cells = cells(row);
            let (id, file) = cells[0]
                .trim_start_matches('[')
                .trim_end_matches(')')
                .split_once("](")
                .expect("a case's first cell links its file");
            let expected = cells
                .last()
                .expect("a row has an expected column")
                .trim_matches('`')
                .replace("\\n", "\n");
            let path = dir.join(file);
            let source = std::fs::read_to_string(&path).expect("the case's file");
            Case {
                id: id.to_string(),
                contexts: source.lines().any(|line| line.starts_with("// contexts:")),
                path,
                expected,
            }
        })
        .collect();
    assert_eq!(
        found.len(),
        CASES,
        "INDEX.md lists every case of its summary"
    );
    found
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The `stages` line of each `For` in an `acvus mir` listing and the facts
/// printed under it, one per line, without the listing's gutter.
fn printed_stage_facts(listing: &str) -> String {
    let mut kept = String::new();
    for line in listing.lines() {
        let Some((gutter, printed)) = line.split_once('|') else {
            continue;
        };
        let gutter = gutter.trim();
        let printed = printed.trim();
        let staged = printed.starts_with("for ") || printed.starts_with("while ");
        let stages = !gutter.is_empty() && staged && printed.contains(" stages [");
        let fact = gutter.is_empty() && printed.starts_with("// ");
        if stages || fact {
            kept.push_str(printed);
            kept.push('\n');
        }
    }
    kept
}

/// A case's `acvus run` and `acvus mir`, in a space of its own where it reads
/// contexts.
struct Ran {
    run: Output,
    mir: Output,
}

fn ran(case: &Case, dir: &Path) -> Ran {
    let config = dir.join("config");
    let acvus = |args: &[&str]| {
        sandbox::with_config(&config)
            .current_dir(dir)
            .args(args)
            .output()
            .expect("the binary runs")
    };
    let path = case.path.to_str().expect("a UTF-8 path");
    if !case.contexts {
        return Ran {
            run: acvus(&["run", path]),
            mir: acvus(&["mir", path]),
        };
    }
    let setup = |args: &[&str]| {
        let out = acvus(args);
        assert_eq!(
            out.status.code(),
            Some(0),
            "`acvus {}`: {}",
            args.join(" "),
            text(&out.stderr)
        );
    };
    let space = format!("space_{}", case.id.to_lowercase());
    let store = format!("dir:store_{}", case.id.to_lowercase());
    setup(&["ctl", "use", "corpus"]);
    setup(&["ctl", "space", "add", &space, &store]);
    setup(&["ctl", "space", "add-script", &space, path]);
    let inits = corpus_dir().join("inits").join(case.id.to_lowercase());
    let mut held: Vec<PathBuf> = std::fs::read_dir(&inits)
        .expect("a case that reads contexts has its inits")
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    held.sort();
    for init in held {
        let context = init
            .file_stem()
            .expect("an init file")
            .to_string_lossy()
            .into_owned();
        setup(&[
            "ctl",
            "space",
            "init",
            &space,
            &context,
            "-f",
            init.to_str().expect("a UTF-8 path"),
        ]);
    }
    let name = case
        .path
        .file_stem()
        .expect("a case file")
        .to_string_lossy()
        .into_owned();
    Ran {
        run: acvus(&["run", &name, "--space", &space]),
        mir: acvus(&["mir", &name, "--space", &space]),
    }
}

#[test]
fn every_corpus_case_gives_its_expected_result_and_prints_its_recorded_facts() {
    let scratch = sandbox::tempdir();
    let mut failures: Vec<String> = Vec::new();
    for case in cases() {
        let dir = scratch.path().join(&case.id);
        std::fs::create_dir_all(&dir).expect("the case's directory");
        let Ran { run, mir } = ran(&case, &dir);
        let got = text(&run.stdout);
        if run.status.code() != Some(0) || got.strip_suffix('\n') != Some(case.expected.as_str()) {
            failures.push(format!(
                "{}: `acvus run` gave {:?} (exit {:?}, stderr {:?}), and INDEX.md expects {:?}",
                case.id,
                got,
                run.status.code(),
                text(&run.stderr),
                case.expected
            ));
        }
        let recorded_at = corpus_dir()
            .join("facts")
            .join(format!("{}.facts", case.id));
        let recorded = std::fs::read_to_string(&recorded_at).unwrap_or_else(|error| {
            panic!(
                "{} records no facts at {}: {error}",
                case.id,
                recorded_at.display()
            )
        });
        let printed = printed_stage_facts(&text(&mir.stdout));
        if mir.status.code() != Some(0) || printed != recorded {
            failures.push(format!(
                "{}: `acvus mir` prints (exit {:?}):\n{printed}and {} records:\n{recorded}",
                case.id,
                mir.status.code(),
                recorded_at.display()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// S06's `return` leaves both loops; with the full pipeline it moves into
/// the inner loop's own exit (RFC-0066 rule 1), and with none it stays where
/// the source wrote it. The program means one value either way.
#[test]
fn the_nested_search_runs_to_its_value_at_both_levels() {
    let scratch = sandbox::tempdir();
    let config = scratch.path().join("config");
    let case = cases()
        .into_iter()
        .find(|case| case.id == "S06")
        .expect("INDEX.md lists S06");
    let path = case.path.to_str().expect("a UTF-8 path");
    for level in ["full", "none"] {
        let run = sandbox::with_config(&config)
            .current_dir(scratch.path())
            .args(["run", path, "--opt", level])
            .output()
            .expect("the binary runs");
        assert_eq!(
            (run.status.code(), text(&run.stdout).trim_end().to_string()),
            (Some(0), case.expected.clone()),
            "--opt {level}: {}",
            text(&run.stderr)
        );
    }
}
