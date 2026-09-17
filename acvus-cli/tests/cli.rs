//! `acvus` at its contract (RFC-0031): what reaches stdout, what reaches
//! stderr in the diagnostic shape, and the exit status of each outcome.

use std::path::Path;
use std::process::{Command, Output};

fn acvus(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_acvus"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("the binary runs")
}

fn write(dir: &Path, name: &str, text: &str) {
    std::fs::write(dir.join(name), text).expect("write a fixture");
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn a_script_prints_its_value_as_json_and_reports_its_writes() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "sum.acvus",
        "let it = as_iter(&@items);\nlet total = 0;\nwhile let Some(x) = next(&mut it) { total = total + *x; }\n@count = @count + 1;\n{ total: total, tag: \"ok\", }\n",
    );
    write(
        dir.path(),
        "ctx.json",
        "{\"items\": [1, 2, 3], \"count\": 7}",
    );
    let out = acvus(dir.path(), &["run", "sum.acvus", "--context", "ctx.json"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "{\"tag\":\"ok\",\"total\":6}\n");
    assert_eq!(text(&out.stderr), "write @count = 8\n");
}

#[test]
fn commit_rewrites_the_context_file_with_the_writes() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bump.acvus", "@count = @count + 1;\n@count\n");
    write(dir.path(), "ctx.json", "{\"count\": 1, \"name\": \"n\"}");
    let out = acvus(
        dir.path(),
        &["run", "bump.acvus", "--context", "ctx.json", "--commit"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let ctx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("ctx.json")).unwrap())
            .unwrap();
    assert_eq!(ctx["count"], 2);
    assert_eq!(ctx["name"], "n");
}

#[test]
fn a_template_prints_its_text_and_an_expression_prints_its_value() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "hi.acvt", "Hello {{ @name }}!");
    write(dir.path(), "hi.json", "{\"name\": \"acvus\"}");
    let out = acvus(dir.path(), &["run", "hi.acvt", "--context", "hi.json"]);
    assert_eq!(text(&out.stdout), "Hello acvus!\n");
    assert_eq!(text(&out.stderr), "");
    let out = acvus(dir.path(), &["run", "-e", "let xs = [1, 2]; xs.len() * 10"]);
    assert_eq!(text(&out.stdout), "20\n");
}

#[test]
fn a_compile_error_is_reported_at_its_line_and_column_with_status_1() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "bad.acvus",
        "let xs = [1, 2];\nlet n = xs + \"a\";\nn\n",
    );
    let out = acvus(dir.path(), &["run", "bad.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(text(&out.stdout), "");
    let err = text(&out.stderr);
    assert!(err.starts_with("error: "), "{err}");
    assert!(err.contains("--> bad.acvus:2:"), "{err}");
    assert!(err.contains("2 | let n = xs + \"a\";"), "{err}");
    assert!(err.contains("^"), "{err}");
    let out = acvus(dir.path(), &["check", "bad.acvus"]);
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn a_type_nothing_resolved_is_reported_as_written() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "open.acvus",
        "let f = |k, m| -> {\n  let a = len(k);\n  k < m\n};\n0\n",
    );
    let out = acvus(dir.path(), &["run", "open.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("error: type mismatch in `<`: ! vs !"),
        "a variable nothing resolved reads as `!` (RFC-0043): {err}"
    );
    assert!(
        !err.contains("<error>"),
        "`<error>` names an ErrorToken and nothing else: {err}"
    );
}

#[test]
fn a_runtime_error_is_the_operations_panic_message_with_status_2() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "boom.acvus",
        "let xs = [1, 2, 3];\n*get(&xs, 9)\n",
    );
    let out = acvus(dir.path(), &["run", "boom.acvus"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(text(&out.stdout), "");
    assert_eq!(
        text(&out.stderr),
        "error: get: index 9 is out of range for length 3\n"
    );
}

#[test]
fn a_context_value_without_a_type_is_refused_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "x.acvus", "1\n");
    write(dir.path(), "empty.json", "{\"items\": []}");
    let out = acvus(dir.path(), &["run", "x.acvus", "--context", "empty.json"]);
    assert_eq!(out.status.code(), Some(64));
    assert!(text(&out.stderr).contains("@items: an empty array has no element type"));
    write(dir.path(), "null.json", "{\"x\": null}");
    let out = acvus(dir.path(), &["run", "x.acvus", "--context", "null.json"]);
    assert!(text(&out.stderr).contains("@x: null has no type"));
}

#[test]
fn a_call_against_a_parameter_a_literal_operand_fixed_is_a_compile_error() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "cat.acvus", "let f = |k| -> k + \"a\";\nf(1)\n");
    let out = acvus(dir.path(), &["check", "cat.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert_eq!(
        err.lines().next(),
        Some("error: type mismatch: expected String, got i64")
    );
    assert!(err.contains("--> cat.acvus:2:3"), "{err}");
    assert!(!err.contains("inst #"), "{err}");

    write(
        dir.path(),
        "cat_ok.acvus",
        "let f = |k| -> k + \"a\";\nf(\"b\")\n",
    );
    let out = acvus(dir.path(), &["run", "cat_ok.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "ba\n");
}

#[test]
fn check_and_mir_compile_without_running() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "ok.acvus", "let xs = [1, 2];\nxs.len()\n");
    let out = acvus(dir.path(), &["check", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "");
    let out = acvus(dir.path(), &["mir", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0));
    assert!(text(&out.stdout).contains("=== main ==="));
    let out = acvus(dir.path(), &["frob"]);
    assert_eq!(out.status.code(), Some(64));
}

#[test]
fn a_space_directory_keeps_contexts_between_runs() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bump.acvus", "@n = @n + 10;\n@n\n");
    write(dir.path(), "seed.json", "{\"n\": 1}");
    let out = acvus(
        dir.path(),
        &[
            "run",
            "bump.acvus",
            "--space",
            "store",
            "--context",
            "seed.json",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "11\n");
    assert!(
        text(&out.stderr).contains("commit @n = "),
        "{}",
        text(&out.stderr)
    );
    let out = acvus(dir.path(), &["run", "bump.acvus", "--space", "store"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "21\n");
    let out = acvus(dir.path(), &["space", "store"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let listing = text(&out.stdout);
    assert!(listing.starts_with("@n: i64 = "), "{listing}");
    assert!(listing.contains("2 nodes"), "{listing}");
}
