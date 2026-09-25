//! The app corpus (`acvus-interpreter-test/app_corpus`) at the CLI's
//! contract: every app runs to the result its `// expect:` header states at
//! `--opt full` and at `--opt none`, and `acvus mir` prints, for each of its
//! `For`s, the stages line and the facts under it exactly as the app's
//! `facts/<stem>.facts` holds them. `INDEX.md` there compares those facts with
//! each loop's hand-written expected structure; a change in any loop's
//! stages, cycles, orders, laws or cost shows here as that app's difference.

use std::path::{Path, PathBuf};

use crate::sandbox;

/// The apps `INDEX.md` lists: ten programs and the two chat-app loops.
const APPS: usize = 12;

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lies in the workspace")
        .join("acvus-interpreter-test/app_corpus")
}

struct App {
    stem: String,
    path: PathBuf,
    expected: String,
}

/// Every `.acvus` in the corpus with its `// expect:` header (`\n` between
/// lines).
fn apps() -> Vec<App> {
    let mut found: Vec<App> = std::fs::read_dir(corpus_dir())
        .expect("the app corpus")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "acvus"))
        .map(|path| {
            let source = std::fs::read_to_string(&path).expect("the app's file");
            let expected = source
                .lines()
                .find_map(|line| line.strip_prefix("// expect: "))
                .unwrap_or_else(|| panic!("{} states no `// expect:`", path.display()))
                .replace("\\n", "\n");
            App {
                stem: path
                    .file_stem()
                    .expect("an app file")
                    .to_string_lossy()
                    .into_owned(),
                path,
                expected,
            }
        })
        .collect();
    found.sort_by(|a, b| a.stem.cmp(&b.stem));
    assert_eq!(found.len(), APPS, "the corpus holds every app INDEX.md lists");
    found
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The `stages` line of each `For` in an `acvus mir` listing and the facts
/// printed under it, one per line, without the listing's gutter. The same
/// reading as the par corpus's.
fn printed_stage_facts(listing: &str) -> String {
    let mut kept = String::new();
    for line in listing.lines() {
        let Some((gutter, printed)) = line.split_once('|') else {
            continue;
        };
        let gutter = gutter.trim();
        let printed = printed.trim();
        let stages =
            !gutter.is_empty() && printed.starts_with("for ") && printed.contains(" stages [");
        let fact = gutter.is_empty() && printed.starts_with("// ");
        if stages || fact {
            kept.push_str(printed);
            kept.push('\n');
        }
    }
    kept
}

#[test]
fn every_app_gives_its_expected_result_at_both_levels_and_prints_its_recorded_facts() {
    let scratch = sandbox::tempdir();
    let config = scratch.path().join("config");
    let acvus = |args: &[&str]| {
        sandbox::with_config(&config)
            .current_dir(scratch.path())
            .args(args)
            .output()
            .expect("the binary runs")
    };
    let mut failures: Vec<String> = Vec::new();
    for app in apps() {
        let path = app.path.to_str().expect("a UTF-8 path");
        for level in ["full", "none"] {
            let run = acvus(&["run", path, "--opt", level]);
            let got = text(&run.stdout);
            if run.status.code() != Some(0) || got.strip_suffix('\n') != Some(app.expected.as_str())
            {
                failures.push(format!(
                    "{} at --opt {level}: `acvus run` gave {:?} (exit {:?}, stderr {:?}), and its header expects {:?}",
                    app.stem,
                    got,
                    run.status.code(),
                    text(&run.stderr),
                    app.expected
                ));
            }
        }
        let recorded_at = corpus_dir()
            .join("facts")
            .join(format!("{}.facts", app.stem));
        let recorded = std::fs::read_to_string(&recorded_at).unwrap_or_else(|error| {
            panic!(
                "{} records no facts at {}: {error}",
                app.stem,
                recorded_at.display()
            )
        });
        let mir = acvus(&["mir", path]);
        let printed = printed_stage_facts(&text(&mir.stdout));
        if mir.status.code() != Some(0) || printed != recorded {
            failures.push(format!(
                "{}: `acvus mir` prints (exit {:?}):\n{printed}and {} records:\n{recorded}",
                app.stem,
                mir.status.code(),
                recorded_at.display()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
