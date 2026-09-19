//! A program the checker admits reaches the machine.
//!
//! `prepare` is the last stage between a checked program and the machine, and
//! it refuses by panicking. Every script the corpus holds is compiled and
//! prepared here, at both optimization levels; a panic whose message no
//! [`Site`] claims fails the test, and a `Site` no script reaches any more
//! fails it too, so the list only shrinks.
//!
//! Each `Site` is a hole in the machine or a diagnostic the checker owes, not
//! a licence.

use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Lapse, Outcome, Script, Stage};
use acvus_mir::graph::optimize::Opt;

/// A `prepare.rs` panic an admitted program reaches today.
struct Site {
    /// The message's stable prefix, up to the part that names one program's
    /// own registers or labels.
    message: &'static str,
    /// One script that reaches it, as the corpus names it.
    reached_by: &'static str,
}

const KNOWN: &[Site] = &[
    Site {
        message: "context: no name for QualifiedRef",
        reached_by: "acvus-interpreter-test/tests/arith_chain.rs:114 run_script",
    },
    Site {
        message: "prepare reached the unresolved type Error(ErrorToken",
        reached_by: "acvus-interpreter-test/tests/attention_shape.rs:155 run",
    },
    Site {
        message: "binop Add on Error(ErrorToken",
        reached_by: "acvus-mir-test/tests/e2e.rs:1305 compile_to_ir",
    },
    Site {
        message: "binop Add on Ref(Shared",
        reached_by: "acvus-mir-test/tests/e2e.rs:2466 compile_script_ir",
    },
    Site {
        message: "TestLiteral: an integer literal against a non-integer",
        reached_by: "acvus-mir-test/tests/reborrow.rs:37",
    },
    Site {
        message: "compile panicked: a type the resolution carries closes",
        reached_by: "acvus-mir-test/tests/e2e.rs:2475 compile_to_ir",
    },
    Site {
        message: "a field step `",
        reached_by: "acvus-mir-test/tests/structural_union.rs:91",
    },
];

const LIMIT: Duration = Duration::from_secs(30);

/// One script's failure to reach the machine.
struct Hole {
    message: String,
    origin: String,
    opt: Opt,
}

impl Site {
    fn claims(&self, hole: &Hole) -> bool {
        hole.message.starts_with(self.message)
    }
}

/// How far one script got at one level.
enum Reach {
    /// A stage before the machine refused it, so the contract says nothing.
    Refused,
    Prepared,
    Held(Hole),
}

fn prepare(script: &Script, opt: Opt) -> Reach {
    let held = |message: String| {
        Reach::Held(Hole {
            message,
            origin: script.origin.clone(),
            opt,
        })
    };
    match corpus::attempt_within(&script.source, opt, Stage::Prepare, LIMIT) {
        Ok(Outcome::Refused(_)) => Reach::Refused,
        Ok(Outcome::Prepared) => Reach::Prepared,
        Ok(Outcome::PreparePanicked(message)) => held(message),
        Ok(Outcome::CompilePanicked(message)) => held(format!("compile panicked: {message}")),
        Ok(other) => held(format!("prepare returned {other:?}")),
        Err(Lapse::TimedOut) => held("prepare did not finish within the limit".to_string()),
        Err(Lapse::Crashed) => held("the process died without an outcome".to_string()),
    }
}

/// The other half of `corpus::attempt_within`: this binary, asked through
/// the environment for one attempt, is the process that makes it.
#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn every_admitted_program_reaches_a_prepared_program() {
    let collection = corpus::collect();
    let mut admitted = 0;
    let mut prepared = 0;
    let mut found: Vec<Hole> = Vec::new();

    for script in &collection.scripts {
        let mut holes = Vec::new();
        let mut refused = false;
        for opt in [Opt::Full, Opt::None] {
            match prepare(script, opt) {
                Reach::Refused => refused = true,
                Reach::Prepared => {}
                Reach::Held(hole) => holes.push(hole),
            }
        }
        if refused {
            continue;
        }
        admitted += 1;
        if holes.is_empty() {
            prepared += 1;
        }
        found.append(&mut holes);
    }

    eprintln!(
        "admitted {admitted}, prepared {prepared}, holes {}",
        found.len()
    );
    for site in KNOWN {
        let count = found.iter().filter(|hole| site.claims(hole)).count();
        eprintln!("known site, {count} scripts: {}", site.message);
    }

    let fresh: Vec<String> = found
        .iter()
        .filter(|hole| !KNOWN.iter().any(|site| site.claims(hole)))
        .map(|hole| {
            format!(
                "{} at opt {:?}\n    {}",
                hole.origin, hole.opt, hole.message
            )
        })
        .collect();
    let closed: Vec<&str> = KNOWN
        .iter()
        .filter(|site| !found.iter().any(|hole| site.claims(hole)))
        .map(|site| site.reached_by)
        .collect();

    assert!(
        fresh.is_empty(),
        "{} admitted programs do not reach the machine:\n  {}",
        fresh.len(),
        fresh.join("\n  ")
    );
    assert!(
        closed.is_empty(),
        "these sites refuse no script any more, so KNOWN holds them for nothing:\n  {}",
        closed.join("\n  ")
    );
}
