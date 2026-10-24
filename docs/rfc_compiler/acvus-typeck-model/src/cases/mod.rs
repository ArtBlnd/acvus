//! The cases: each `cases/**/*.case` read, built,
//! solved as its orders say, checked against its expectations and its golden
//! outcome, and reported in one table.

mod build;
mod expect;
mod parse;
mod render;
mod run;

use std::env::VarError;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use parse::{Expect, Located};
use run::{Explored, Run, Tracer};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Pass,
    Fail,
}

struct Ran {
    path: String,
    status: Status,
    messages: Vec<String>,
}

/// The expectations a block may hold: those an outcome and its final states
/// meet.
fn check_block(block: &[Located<Expect>]) -> Result<(), String> {
    for Located { line, item } in block {
        let allowed = matches!(
            item,
            Expect::Accepted(_)
                | Expect::Value { .. }
                | Expect::Decision { .. }
                | Expect::Refused(_)
                | Expect::Failing(_)
                | Expect::Check { .. }
                | Expect::Partition(_)
                | Expect::Istar { .. }
        );
        if !allowed {
            return Err(format!("line {line}: an outcome block holds no {item:?}"));
        }
    }
    Ok(())
}

/// The golden outcome of the runs: every outcome in full, with the I\*
/// violations of the final states reaching it, and a written order's trace.
fn golden(runs: &[Run]) -> String {
    let mut text = String::from("// What the model gives, written by `BLESS=1`.\n");
    for run in runs {
        let explored = match &run.explored {
            Explored::All { .. } => "every order".to_string(),
            Explored::Sampled(orders) => format!("{orders} sampled orders"),
            Explored::Written(_) => "the written order".to_string(),
        };
        writeln!(text, "run: {explored}").expect("writing to a string");
        for (k, outcome) in run.outcomes.iter().enumerate() {
            writeln!(text, "outcome {}", k + 1).expect("writing to a string");
            text.push_str(&render::outcome(outcome));
            let istars: std::collections::BTreeSet<Vec<String>> = run
                .finals
                .iter()
                .filter(|end| end.outcome.same_typing(outcome))
                .map(|end| end.istar.iter().cloned().collect())
                .collect();
            for istar in istars {
                writeln!(text, "  I* of a final state: [{}]", istar.join(", "))
                    .expect("writing to a string");
            }
        }
        if let Explored::Written(taken) = &run.explored {
            for step in taken {
                writeln!(text, "{}", step.step).expect("writing to a string");
                for report in &step.reports {
                    let depth = match &report.line {
                        Some(line) => {
                            writeln!(text, "  {line}").expect("writing to a string");
                            "    "
                        }
                        None => "  ",
                    };
                    for (at, recorded) in &report.recorded {
                        writeln!(text, "{depth}record #{at} {recorded}")
                            .expect("writing to a string");
                    }
                }
                for refused in &step.refused {
                    writeln!(text, "  refuse: {refused}").expect("writing to a string");
                }
            }
        }
    }
    text
}

/// Runs the case, and checks it against its golden outcome at `golden_path`,
/// which `bless` writes anew; `tracer` prints its runs' trace (`TRACE=1`).
fn run_case(
    source: &str,
    golden_path: &Path,
    bless: bool,
    tracer: Option<&Tracer>,
) -> Result<Ran, String> {
    let case = parse::parse(source)?;
    for block in &case.outcomes {
        check_block(&block.item)?;
    }
    let decls = build::decls(&case)?;
    let changes = expect::changes(&decls, &case.expects)?;
    if let Some(tracer) = tracer {
        let explored = match &case.orders {
            parse::Orders::All => "every order".to_string(),
            parse::Orders::Sampled(orders) => format!("{orders} sampled orders"),
            parse::Orders::Written(_) => "the written order".to_string(),
        };
        tracer.line(0, &format!("run: {explored}"));
    }
    let program = build::program(&case, &decls)?;
    let runs = vec![run::run(
        &program,
        &case.orders,
        &changes.corruptions,
        tracer,
    )?];
    let mut messages = expect::fails(
        &decls,
        &runs,
        &case.expects,
        &case.outcomes,
        &changes.expected,
    );
    let mut status = if messages.is_empty() {
        Status::Pass
    } else {
        Status::Fail
    };
    let text = golden(&runs);
    if bless {
        std::fs::write(golden_path, &text)
            .map_err(|error| format!("writing the golden: {error}"))?;
    } else {
        match std::fs::read_to_string(golden_path) {
            Ok(kept) if kept == text => {}
            Ok(kept) => {
                let differing = kept
                    .lines()
                    .zip(text.lines())
                    .position(|(a, b)| a != b)
                    .unwrap_or(kept.lines().count().min(text.lines().count()));
                let kept_line = kept.lines().nth(differing).unwrap_or("(the end)");
                let new_line = text.lines().nth(differing).unwrap_or("(the end)");
                messages.push(format!(
                    "the golden differs at its line {}: kept {kept_line:?}, now {new_line:?}",
                    differing + 1
                ));
                status = Status::Fail;
            }
            Err(error) => {
                messages.push(format!("no golden ({error}); `BLESS=1` writes it"));
                status = Status::Fail;
            }
        }
    }
    Ok(Ran {
        path: String::new(),
        status,
        messages,
    })
}

fn collect(dir: &Path, paths: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            collect(&path, paths);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "case")
        {
            paths.push(path);
        }
    }
}

fn variable(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) => Some(value),
        Err(VarError::NotPresent) => None,
        Err(error) => panic!("{name}: {error}"),
    }
}

/// The message a case's thread panicked with.
fn panicked(payload: &(dyn std::any::Any + Send)) -> String {
    match (
        payload.downcast_ref::<&str>(),
        payload.downcast_ref::<String>(),
    ) {
        (Some(message), _) => message.to_string(),
        (_, Some(message)) => message.clone(),
        (None, None) => "a panic with no message".to_string(),
    }
}

#[test]
fn cases() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cases");
    let filter = variable("CASE");
    let bless = match variable("BLESS").as_deref() {
        None => false,
        Some("1") => true,
        Some(other) => panic!("BLESS is 1 or unset, not {other}"),
    };
    let trace = match variable("TRACE").as_deref() {
        None => false,
        Some("1") => true,
        Some(other) => panic!("TRACE is 1 or unset, not {other}"),
    };
    let mut paths = Vec::new();
    collect(&root, &mut paths);
    paths.sort();
    let named: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&root)
                .expect("a case lies under the root")
                .to_string_lossy()
                .into_owned();
            (relative, path)
        })
        .filter(|(relative, _)| {
            filter
                .as_ref()
                .is_none_or(|filter| relative.contains(filter.as_str()))
        })
        .collect();
    assert!(
        !named.is_empty(),
        "no case under {} matches",
        root.display()
    );
    // One worker where the parallelism cannot be read, which only runs the
    // cases one after another.
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let queue = Mutex::new(named.iter());
    let ran = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let next = queue.lock().expect("the queue's lock").next();
                    let Some((relative, path)) = next else {
                        break;
                    };
                    let golden_path = path.with_extension("outcome");
                    let joined = std::thread::Builder::new()
                        .name(format!("cases::{relative}"))
                        .spawn_scoped(scope, move || {
                            let source = std::fs::read_to_string(path)
                                .map_err(|error| format!("reading the case: {error}"))?;
                            let tracer = trace.then(|| Tracer::new(relative));
                            run_case(&source, &golden_path, bless, tracer.as_ref())
                        })
                        .expect("a case's thread")
                        .join();
                    let outcome = match joined {
                        Ok(Ok(mut outcome)) => {
                            outcome.path = relative.clone();
                            outcome
                        }
                        Ok(Err(error)) => Ran {
                            path: relative.clone(),
                            status: Status::Fail,
                            messages: vec![error],
                        },
                        Err(payload) => Ran {
                            path: relative.clone(),
                            status: Status::Fail,
                            messages: vec![format!("panicked: {}", panicked(&*payload))],
                        },
                    };
                    ran.lock().expect("the results' lock").push(outcome);
                }
            });
        }
    });
    let mut ran = ran.into_inner().expect("the results' lock");
    ran.sort_by(|a, b| a.path.cmp(&b.path));
    let mut table = String::new();
    for case in &ran {
        let status = match case.status {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
        };
        writeln!(table, "{status} {}", case.path).expect("writing to a string");
        for message in &case.messages {
            writeln!(table, "      {message}").expect("writing to a string");
        }
    }
    println!("{table}");
    let failed: Vec<&str> = ran
        .iter()
        .filter(|case| case.status == Status::Fail)
        .map(|case| case.path.as_str())
        .collect();
    assert!(
        failed.is_empty(),
        "{} cases fail: {failed:?}\n{table}",
        failed.len()
    );
}
