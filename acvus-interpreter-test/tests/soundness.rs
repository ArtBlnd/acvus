//! Every program under `tests/soundness/` keeps one contract: the checker
//! refuses it, or it runs to one value at both optimization levels. A refusal
//! the MIR validator or `prepare` raises in the checker's place, a crash, a
//! timeout, an `<undef>` in a value, or two levels that disagree is a hole.
//!
//! A hole the branch has not closed yet is listed in `KNOWN` with the words
//! it shows. The list only shrinks: a listed program that keeps the contract
//! fails the test until its entry is removed, and a hole not listed fails it
//! too. A directory's `ctx.json` declares the contexts of its programs, as
//! `acvus run --context` reads it.
//!
//! Beside the standard registries a program can call `opaque(x)` and
//! `opaque_async(x)`, which answer `x` at effect `opaque`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

/// A program the branch still gets wrong, and the words that show it.
struct Known {
    program: &'static str,
    shows: &'static str,
}

const KNOWN: &[Known] = &[
    // A pattern on `&Option<&T>` binds its payload at the wrong depth
    // (RFC-0024 rule 3, RFC-0029 rule 3): the validator refuses it at a
    // word payload, the machine asserts at a `String` one.
    Known {
        program: "attack-control-2/d33.acvus",
        shows: "Take takes dst as &i64, and it is i64",
    },
    Known {
        program: "attack-control-2/d38.acvus",
        shows: "is not large",
    },
];

const LIMIT: Duration = Duration::from_secs(30);

/// The words the MIR type validator writes: it checks the lowered body, and
/// a refusal of its own is a program the checker should have refused.
const PAST_THE_CHECKER: &[&str] = &[
    " and got ",
    ", and it is ",
    "has no type",
    "reaches the machine",
];

/// A program's own trap (RFC-0038): an outcome like a value, the same at
/// both levels.
const TRAPS: &[&str] = &[
    "attempt to divide by zero",
    "attempt to divide with overflow",
    "attempt to calculate the remainder with a divisor of zero",
    "attempt to calculate the remainder with overflow",
    "index out of bounds: the len is",
    "substring: ",
];

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/soundness")
}

fn programs(dir: &Path, found: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("a corpus directory is readable")
        .map(|entry| entry.expect("a corpus entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            programs(&path, found);
        } else if path
            .extension()
            .is_some_and(|ext| ext == "acvus" || ext == "acvt")
        {
            found.push(path);
        }
    }
}

/// A `.acvus` file is a script and a `.acvt` file a template, as `acvus
/// run` reads them; the harness would read a script that does not parse as
/// a template's text, so such a script is left to the parser's own tests.
fn parses_as_written(program: &Path, source: &str) -> bool {
    let interner = acvus_utils::Interner::new();
    match program.extension().and_then(|ext| ext.to_str()) {
        Some("acvus") => acvus_ast::parse_script(&interner, source).is_ok(),
        _ => acvus_ast::parse(&interner, source).is_ok(),
    }
}

fn contexts_of(program: &Path) -> serde_json::Map<String, serde_json::Value> {
    let declared = program.with_file_name("ctx.json");
    match std::fs::read_to_string(&declared) {
        Ok(json) => serde_json::from_str(&json).expect("a ctx.json is a JSON object"),
        Err(_) => serde_json::Map::new(),
    }
}

fn outcome(
    source: &str,
    contexts: &serde_json::Map<String, serde_json::Value>,
    opt: Opt,
) -> Outcome {
    match acvus_interpreter_test::attempt_within!(
        source,
        contexts = contexts,
        opt,
        Stage::Run,
        LIMIT
    ) {
        Ok(outcome) => outcome,
        Err(lapse) => Outcome::RunPanicked(format!("the run lapsed: {lapse:?}")),
    }
}

/// What a program's first line says it means: `// expect <value>` or
/// `// refuse`. A value is written as `acvus run` prints it.
enum Expected {
    Value(String),
    Refused,
}

fn expected(source: &str) -> Option<Expected> {
    let first = source.lines().next()?.trim();
    if first == "// refuse" {
        return Some(Expected::Refused);
    }
    let value = first.strip_prefix("// expect ")?;
    Some(Expected::Value(value.trim().to_string()))
}

/// A program the checker admits runs to what it means; a program it
/// refuses may be one it could have run, which is not a hole.
fn contradicts(expected: &Expected, outcome: &Outcome) -> Option<String> {
    match (expected, outcome) {
        (Expected::Value(want), Outcome::Value(got))
            if *got != *want && *got != serde_json::Value::String(want.clone()).to_string() =>
        {
            Some(format!("a wrong value: expected {want}, got {got}"))
        }
        (Expected::Refused, Outcome::Value(got)) => Some(format!(
            "admitted a program it should refuse, which ran to {got}"
        )),
        _ => None,
    }
}

/// What breaks the contract in one program's two outcomes, if anything.
fn hole(none: &Outcome, full: &Outcome) -> Option<String> {
    let broken = |outcome: &Outcome| match outcome {
        Outcome::Refused(why) if PAST_THE_CHECKER.iter().any(|w| why.contains(w)) => {
            Some(format!("refused past the checker: {why}"))
        }
        Outcome::Refused(_) => None,
        Outcome::Value(value) if value.contains("<undef>") => {
            Some(format!("an undefined value reached the host: {value}"))
        }
        Outcome::Value(_) | Outcome::Prepared => None,
        Outcome::CompilePanicked(why) => Some(format!("compile panicked: {why}")),
        Outcome::PreparePanicked(why) => Some(format!("prepare panicked: {why}")),
        Outcome::RunPanicked(why) if TRAPS.iter().any(|trap| why.starts_with(trap)) => None,
        Outcome::RunPanicked(why) => Some(format!("run failed: {why}")),
    };
    if let Some(why) = broken(none).or_else(|| broken(full)) {
        return Some(why);
    }
    match (none, full) {
        (Outcome::Refused(_), Outcome::Refused(_)) => None,
        (Outcome::Value(a), Outcome::Value(b))
        | (Outcome::RunPanicked(a), Outcome::RunPanicked(b))
            if a == b =>
        {
            None
        }
        _ => Some(format!("the levels disagree: none {none:?}, full {full:?}")),
    }
}

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn every_program_is_refused_or_runs_to_one_value() {
    let mut found = Vec::new();
    programs(&corpus_dir(), &mut found);
    assert!(!found.is_empty(), "the corpus holds programs");

    let holes: Vec<(String, String)> = std::thread::scope(|scope| {
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunks: Vec<&[PathBuf]> = found.chunks(found.len().div_ceil(workers)).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|program| {
                            let source =
                                std::fs::read_to_string(program).expect("a program is readable");
                            if !parses_as_written(program, &source) {
                                return None;
                            }
                            let contexts = contexts_of(program);
                            let none = outcome(&source, &contexts, Opt::None);
                            let full = outcome(&source, &contexts, Opt::Full);
                            let name = program
                                .strip_prefix(corpus_dir())
                                .expect("a program is under the corpus")
                                .display()
                                .to_string();
                            hole(&none, &full)
                                .or_else(|| {
                                    let expected = expected(&source)?;
                                    contradicts(&expected, &none)
                                })
                                .map(|why| (name, why))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a corpus worker finishes"))
            .collect()
    });

    let mut wrong = Vec::new();
    for (program, why) in &holes {
        match KNOWN.iter().find(|known| known.program == program) {
            Some(known) if why.contains(known.shows) => {}
            Some(known) => wrong.push(format!(
                "{program}: listed as showing {:?}, now shows: {why}",
                known.shows
            )),
            None => wrong.push(format!("{program}: {why}")),
        }
    }
    for known in KNOWN {
        if !holes.iter().any(|(program, _)| program == known.program) {
            wrong.push(format!(
                "{}: keeps the contract now; remove it from KNOWN",
                known.program
            ));
        }
    }
    wrong.sort();
    assert!(
        wrong.is_empty(),
        "{} of {} programs break the contract:\n{}",
        wrong.len(),
        found.len(),
        wrong.join("\n")
    );
}
