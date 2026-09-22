//! The contract of `examples/**`, which lives outside this crate: every
//! directory there holds `main.acvus`, `ctx.json` and `expected.txt`, and
//! the CLI run over the first two writes the third and exits 0. Adding a
//! directory therefore means adding a test here; `EXAMPLES` is what the
//! tests cover and `every_example_directory_is_covered` holds it equal to
//! what the directory contains.

use std::path::{Path, PathBuf};
use std::process::Command;

const EXAMPLES: [&str; 8] = [
    "collatz",
    "grades",
    "ledger",
    "log-parse",
    "prompt",
    "queue",
    "shapes",
    "word-count",
];

const DATA: [&str; 2] = ["ctx.json", "expected.txt"];
const SOURCES: [&str; 2] = ["main.acvus", "main.acvt"];

fn source_of(name: &str) -> String {
    let dir = root().join("examples").join(name);
    let mut sources = SOURCES.iter().filter(|s| dir.join(s).exists());
    let (Some(source), None) = (sources.next(), sources.next()) else {
        panic!("examples/{name} holds one of {SOURCES:?}, not several and not none");
    };
    format!("examples/{name}/{source}")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate directory sits in the workspace root")
        .to_path_buf()
}

fn acvus(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_acvus"))
        .current_dir(root())
        .args(args)
        .output()
        .expect("cargo built the binary this test names")
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn sorted_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap_or_else(|e| panic!("{}: {e}", dir.display())))
        .map(|entry| {
            entry
                .file_name()
                .into_string()
                .unwrap_or_else(|name| panic!("{}: {name:?} is not UTF-8", dir.display()))
        })
        .collect();
    names.sort();
    names
}

fn example(name: &str) {
    let dir = root().join("examples").join(name);
    let script = source_of(name);
    let context = format!("examples/{name}/ctx.json");
    let expected = std::fs::read(dir.join("expected.txt"))
        .unwrap_or_else(|e| panic!("examples/{name}/expected.txt: {e}"));

    let out = acvus(&["run", &script, "--context", &context]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "`acvus run {script}`:\n{}",
        stderr(&out)
    );
    assert_eq!(
        out.stdout,
        expected,
        "`acvus run {script}` wrote\n{}\nwhere examples/{name}/expected.txt holds\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&expected)
    );

    let out = acvus(&["check", &script, "--context", &context]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "`acvus check {script} --context {context}`:\n{}",
        stderr(&out)
    );
}

#[test]
fn collatz() {
    example("collatz");
}

#[test]
fn grades() {
    example("grades");
}

#[test]
fn ledger() {
    example("ledger");
}

#[test]
fn log_parse() {
    example("log-parse");
}

#[test]
fn prompt() {
    example("prompt");
}

#[test]
fn queue() {
    example("queue");
}

#[test]
fn shapes() {
    example("shapes");
}

#[test]
fn word_count() {
    example("word-count");
}

#[test]
fn every_example_directory_is_covered() {
    let examples = root().join("examples");
    let found = sorted_names(&examples);
    assert_eq!(
        found, EXAMPLES,
        "examples/ against the list the tests cover"
    );

    for name in found {
        let source = source_of(&name);
        let data: Vec<String> = sorted_names(&examples.join(&name))
            .into_iter()
            .filter(|file| !source.ends_with(file.as_str()))
            .collect();
        assert_eq!(data, DATA, "the files beside the source in examples/{name}");
    }
}
