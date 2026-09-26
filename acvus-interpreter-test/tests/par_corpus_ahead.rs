//! The parallel-loop corpus rows whose header states `registry: ahead`:
//! they call `acvus_interpreter_test::ahead`'s externs, which `acvus run`
//! does not hold, so this test runs them where `acvus-cli`'s `par_corpus`
//! runs every other row. Each gives the output its `INDEX.md` row states
//! lowered and in place under both executors, and prints the stage facts
//! its `facts/<id>.facts` holds, the lowering's line among them.

use std::path::{Path, PathBuf};

use acvus_interpreter::Lower;
use acvus_interpreter_test::ahead::{On, Wire, registries, run};
use acvus_interpreter_test::{compile_source_with_externs, split_context};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;
use std::sync::Arc;

/// The rows `INDEX.md` lists in its registry section.
const ROWS: usize = 6;

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("par_corpus")
}

struct Row {
    id: String,
    source: String,
    returns: Ty,
    expected: String,
}

fn header<'s>(source: &'s str, name: &str) -> Option<&'s str> {
    let prefix = format!("// {name}: ");
    source.lines().find_map(|line| line.strip_prefix(prefix.as_str()))
}

fn returns(written: &str) -> Ty {
    match written {
        "String" => Ty::String,
        "i64" => Ty::Int(IntTy::I64),
        "u64" => Ty::Int(IntTy::U64),
        other => panic!("a row returns {other:?}, which this test does not read"),
    }
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

fn rows() -> Vec<Row> {
    let dir = corpus_dir();
    let index = std::fs::read_to_string(dir.join("INDEX.md")).expect("the corpus index");
    let found: Vec<Row> = index
        .lines()
        .filter(|line| line.starts_with("| ["))
        .filter_map(|line| {
            let cells = cells(line);
            let (id, file) = cells[0]
                .trim_start_matches('[')
                .trim_end_matches(')')
                .split_once("](")
                .expect("a case's first cell links its file");
            let source = std::fs::read_to_string(dir.join(file)).expect("the case's file");
            header(&source, "registry")?;
            let expected = cells
                .last()
                .expect("a row has an expected column")
                .trim_matches('`')
                .replace("\\n", "\n");
            let returns = returns(header(&source, "returns").expect("a registry row states `returns`"));
            Some(Row {
                id: id.to_string(),
                source,
                returns,
                expected,
            })
        })
        .collect();
    assert_eq!(found.len(), ROWS, "INDEX.md lists every registry row");
    found
}

/// What `acvus run` would print: the lines the run emitted, then its value.
fn printed(lines: &[String], value: &serde_json::Value) -> String {
    let value = match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    lines
        .iter()
        .cloned()
        .chain(std::iter::once(value))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `stages` line of each `For` in a listing and the facts printed under
/// it, as `acvus-cli`'s `par_corpus` reads them.
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

fn facts_of(row: &Row) -> String {
    let interner = Interner::new();
    let (types, _) = split_context(&interner, Default::default());
    let ast =
        ParsedAst::Script(acvus_ast::parse_script(&interner, &row.source).expect("parse error"));
    let wire = Arc::new(Wire::default());
    let cr = compile_source_with_externs(&interner, ast, &types, registries(&wire), row.returns.clone());
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    printed_stage_facts(&acvus_mir::printer::dump_with_costs(
        &interner,
        module,
        &cr.laws,
        &acvus_interpreter::cost::INTERPRETER_COSTS,
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_registry_row_gives_its_expected_output_and_prints_its_recorded_facts() {
    let mut failures: Vec<String> = Vec::new();
    for row in rows() {
        for on in [On::Sequential, On::Tokio] {
            for lower in [Lower::Ahead, Lower::InPlace] {
                let ran = run(&row.source, row.returns.clone(), lower, on).await;
                if lower == Lower::Ahead && !ran.lowered {
                    failures.push(format!(
                        "{}: runs in place on {on:?}, declined {:?}",
                        row.id, ran.declined
                    ));
                }
                let got = match &ran.outcome.value {
                    Ok(value) => printed(&ran.outcome.lines, value),
                    Err(trap) => format!("trapped: {trap}"),
                };
                if got != row.expected {
                    failures.push(format!(
                        "{}: {lower:?} on {on:?} printed {got:?}, and INDEX.md expects {:?}",
                        row.id, row.expected
                    ));
                }
            }
        }
        let facts = facts_of(&row);
        let recorded_at = corpus_dir().join("facts").join(format!("{}.facts", row.id));
        let recorded = std::fs::read_to_string(&recorded_at).unwrap_or_else(|error| {
            panic!("{}: {error}; the facts printed are:\n{facts}", recorded_at.display())
        });
        if facts != recorded {
            failures.push(format!(
                "{}: the facts printed are\n{facts}and {} holds\n{recorded}",
                row.id,
                recorded_at.display()
            ));
        }
        if !facts.lines().any(|line| line.starts_with("// lower ahead {")) {
            failures.push(format!("{}: no loop is lowered ahead:\n{facts}", row.id));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
