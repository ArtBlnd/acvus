//! The corpus of wrong programs and the whole diagnostic each one gets
//! (`docs/diagnostics.md`). A program under `tests/refusals/` is refused,
//! and what `acvus check` writes to stderr is its `.expected` file byte for
//! byte, so a change of wording is a red test rather than a silent drift.

use std::path::{Path, PathBuf};

/// The corpus is the evidence that every refusal names a fix, so it does
/// not shrink below the size that evidence was gathered at.
const AT_LEAST: usize = 46;

/// Set to rewrite every `.expected` from what the compiler says now. The
/// rewritten files are read, not trusted: a pin is only evidence once a
/// reader has agreed the sentence names the fix.
const BLESS: &str = "BLESS_REFUSALS";

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("refusals")
}

fn programs() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("the corpus directory is there")
        .map(|entry| entry.expect("a corpus entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "acvus"))
        .collect();
    found.sort();
    found
}

fn named(program: &Path) -> &str {
    program
        .file_name()
        .expect("a path from read_dir names a file")
        .to_str()
        .expect("a corpus file is named in UTF-8")
}

fn text(stream: Vec<u8>) -> Result<String, std::string::FromUtf8Error> {
    String::from_utf8(stream)
}

fn refusal_of(program: &Path) -> String {
    let out = crate::sandbox::acvus()
        .current_dir(corpus_dir())
        .args(["check", named(program)])
        .output()
        .expect("the binary runs");
    assert_eq!(
        out.status.code(),
        Some(1),
        "{} was not refused; it printed {:?}",
        named(program),
        text(out.stdout)
    );
    text(out.stderr).expect("a refusal is UTF-8")
}

#[test]
fn the_corpus_is_at_least_as_large_as_the_evidence() {
    let found = programs().len();
    assert!(
        found >= AT_LEAST,
        "the corpus holds {found} programs, fewer than the {AT_LEAST} it was gathered at"
    );
}

#[test]
fn every_program_is_refused_with_the_words_it_is_pinned_to() {
    let blessing = std::env::var_os(BLESS).is_some();
    let mut wrong = Vec::new();
    for program in programs() {
        let got = refusal_of(&program);
        let pinned = program.with_extension("expected");
        if blessing {
            std::fs::write(&pinned, &got).expect("write the pinned refusal");
            continue;
        }
        match std::fs::read_to_string(&pinned) {
            Ok(want) if want == got => {}
            Ok(want) => wrong.push(format!(
                "{}: the words changed\n  pinned: {want}\n  got:    {got}",
                named(&program)
            )),
            Err(why) => wrong.push(format!(
                "{}: no pinned refusal ({why}); run with {BLESS}=1 and read what it wrote\n  got: {got}",
                named(&program)
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of the corpus is not what it is pinned to:\n\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
