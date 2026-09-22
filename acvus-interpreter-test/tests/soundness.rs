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
    // An `Option` of an aggregate with a `None` element, in an array.
    Known {
        program: "attack-cleanups/h01.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s22.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s27.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s33.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s35.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s36.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s37.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s40.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s46.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s47.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s48.acvus",
        shows: "a double take",
    },
    Known {
        program: "attack-cleanups/s52.acvus",
        shows: "a double take",
    },
    // `*x` of an aggregate element reference from a `for`: the checker admits it.
    Known {
        program: "attack-cleanups/h02.acvus",
        shows: "Take takes a primitive through a reference",
    },
    Known {
        program: "attack-cleanups/s38.acvus",
        shows: "Take takes a primitive through a reference",
    },
    // A `String` payload taken out of a variant still in its storage, twice.
    Known {
        program: "b18/w02.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b19/v08.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b20/u04.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b20/u05.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b21/y07.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b23/f04.acvus",
        shows: "Crashed",
    },
    Known {
        program: "b23/f06.acvus",
        shows: "Crashed",
    },
    Known {
        program: "holes/f09.acvus",
        shows: "Crashed",
    },
    Known {
        program: "holes/f11.acvus",
        shows: "Crashed",
    },
    // `for x in <array>` of `Option<String>` elements.
    Known {
        program: "b25/j03.acvus",
        shows: "a double take",
    },
    Known {
        program: "b27/q1.acvus",
        shows: "a double take",
    },
    Known {
        program: "b27/q3.acvus",
        shows: "a double take",
    },
    // A structural enum element of `for x in <array>`, matched.
    Known {
        program: "b26/k7.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b27/q9.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b27/q10.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b28/r1.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b28/r2.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b28/r3.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b28/r14.acvus",
        shows: "carries no bits",
    },
    Known {
        program: "b30/u14.acvus",
        shows: "carries no bits",
    },
];

const LIMIT: Duration = Duration::from_secs(30);

/// The words the MIR type validator writes: it checks the lowered body, and
/// a refusal of its own is a program the checker should have refused.
const PAST_THE_CHECKER: &[&str] = &[" and got ", ", and it is ", "has no type", "reaches the machine"];

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

fn contexts_of(program: &Path) -> serde_json::Map<String, serde_json::Value> {
    let declared = program.with_file_name("ctx.json");
    match std::fs::read_to_string(&declared) {
        Ok(json) => serde_json::from_str(&json).expect("a ctx.json is a JSON object"),
        Err(_) => serde_json::Map::new(),
    }
}

fn outcome(source: &str, contexts: &serde_json::Map<String, serde_json::Value>, opt: Opt) -> Outcome {
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
        (Outcome::Value(a), Outcome::Value(b)) | (Outcome::RunPanicked(a), Outcome::RunPanicked(b))
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
                            let contexts = contexts_of(program);
                            let none = outcome(&source, &contexts, Opt::None);
                            let full = outcome(&source, &contexts, Opt::Full);
                            let name = program
                                .strip_prefix(corpus_dir())
                                .expect("a program is under the corpus")
                                .display()
                                .to_string();
                            hole(&none, &full).map(|why| (name, why))
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
