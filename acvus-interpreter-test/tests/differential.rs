//! One program, one value, optimized or not.
//!
//! `code_motion::merge_pass` once returned `1001` where the program said
//! `1010`, and what found it was a slice run somebody happened to try by
//! hand. This test is that comparison over the whole corpus: every script
//! the tests hand a harness, compiled at `Opt::None` and at `Opt::Full`,
//! prepared and run, and the two values read the way `acvus run` prints
//! them.
//!
//! A script whose two levels disagree is re-run at `Opt::Full` to separate a
//! miscompile from a program whose value is not a function of the program
//! alone: a `Full` run that disagrees with itself is nondeterministic and is
//! reported as such, and one that repeats is a miscompile and fails the test.

use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Lapse, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

/// What one script's two levels came to.
enum Verdict {
    Agreed,
    /// Both levels refused the source, so there was nothing to run.
    Refused,
    /// `prepare` refused it; `prepare_contract.rs` is where that is counted.
    Unprepared,
    /// One level finished within the limit and the other did not.
    Lapsed(Lapse),
    Nondeterministic,
    Mismatch {
        full: Outcome,
        none: Outcome,
    },
    /// The checker admitted the program at one level and refused it at the
    /// other, which rule 4 forbids: the level chooses how hard the compiler
    /// works, never which programs are legal.
    RefusalMoved {
        full: Outcome,
        none: Outcome,
    },
}

const LIMIT: Duration = Duration::from_secs(30);

fn outcome(source: &str, opt: Opt) -> Result<Outcome, Lapse> {
    corpus::attempt_within(source, opt, Stage::Run, LIMIT)
}

fn verdict(source: &str) -> Verdict {
    let (full, none) = match (outcome(source, Opt::Full), outcome(source, Opt::None)) {
        (Ok(full), Ok(none)) => (full, none),
        (Err(lapse), _) | (_, Err(lapse)) => return Verdict::Lapsed(lapse),
    };
    match (&full, &none) {
        (Outcome::Refused(_), Outcome::Refused(_)) => Verdict::Refused,
        (Outcome::Refused(_), _) | (_, Outcome::Refused(_)) => Verdict::RefusalMoved { full, none },
        _ if crashed(&full) && crashed(&none) => Verdict::Unprepared,
        _ if full == none => Verdict::Agreed,
        _ => match outcome(source, Opt::Full) {
            Ok(again) if again == full => Verdict::Mismatch { full, none },
            Ok(_) => Verdict::Nondeterministic,
            Err(lapse) => Verdict::Lapsed(lapse),
        },
    }
}

/// A program that never reached the machine: `prepare_contract.rs` is where
/// that is counted, and it is a difference here only when one level reaches
/// the machine and the other does not.
fn crashed(outcome: &Outcome) -> bool {
    matches!(
        outcome,
        Outcome::CompilePanicked(_) | Outcome::PreparePanicked(_)
    )
}

fn show(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Refused(why) => format!("refused: {why}"),
        Outcome::CompilePanicked(why) => format!("compile panicked: {why}"),
        Outcome::PreparePanicked(why) => format!("prepare panicked: {why}"),
        Outcome::Prepared => "prepared".to_string(),
        Outcome::Value(value) => format!("value {value}"),
        Outcome::RunPanicked(why) => format!("panicked: {why}"),
    }
}

#[derive(Default)]
struct Tally {
    agreed: usize,
    refused: usize,
    unprepared: usize,
    lapsed: Vec<String>,
    nondeterministic: Vec<String>,
    mismatched: Vec<String>,
}

/// The other half of `corpus::attempt_within`: this binary, asked through
/// the environment for one attempt, is the process that makes it.
#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn an_optimized_run_and_an_unoptimized_run_produce_one_value() {
    let collection = corpus::collect();
    let mut tally = Tally::default();

    for script in &collection.scripts {
        match verdict(&script.source) {
            Verdict::Agreed => tally.agreed += 1,
            Verdict::Refused => tally.refused += 1,
            Verdict::Unprepared => tally.unprepared += 1,
            Verdict::Lapsed(lapse) => tally.lapsed.push(format!("{} ({lapse:?})", script.origin)),
            Verdict::Nondeterministic => tally.nondeterministic.push(script.origin.clone()),
            Verdict::Mismatch { full, none } => tally.mismatched.push(format!(
                "{}\n    opt full: {}\n    opt none: {}\n    {}",
                script.origin,
                show(&full),
                show(&none),
                script.source.replace('\n', "\n    ")
            )),
            Verdict::RefusalMoved { full, none } => tally.mismatched.push(format!(
                "{} is legal at one level and not the other\n    opt full: {}\n    opt none: {}",
                script.origin,
                show(&full),
                show(&none)
            )),
        }
    }

    eprintln!(
        "corpus: {} scripts from {} harness calls, {} gaps",
        collection.scripts.len(),
        collection.calls,
        collection.gaps.len()
    );
    eprintln!(
        "agreed {}, refused {}, unprepared {}, lapsed {}, nondeterministic {}, mismatched {}",
        tally.agreed,
        tally.refused,
        tally.unprepared,
        tally.lapsed.len(),
        tally.nondeterministic.len(),
        tally.mismatched.len()
    );
    for gap in &collection.gaps {
        eprintln!("gap: {gap}");
    }
    for origin in &tally.nondeterministic {
        eprintln!("nondeterministic: {origin}");
    }
    for origin in &tally.lapsed {
        eprintln!("lapsed: {origin}");
    }

    assert!(
        tally.mismatched.is_empty(),
        "{} scripts run differently optimized and unoptimized:\n  {}",
        tally.mismatched.len(),
        tally.mismatched.join("\n  ")
    );
}
