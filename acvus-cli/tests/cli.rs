//! `acvus` at its contract (RFC-0031): what reaches stdout, what reaches
//! stderr in the diagnostic shape, and the exit status of each outcome.

use std::path::Path;
use std::process::{Command, Output};

use acvus_interpreter_test::Context;
use acvus_interpreter_test::listing::{script_listing_with_externs, text as listing_text};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn acvus(dir: &Path, args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_acvus"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("the binary runs");
    let reported = [&out.stdout, &out.stderr]
        .into_iter()
        .any(|stream| text(stream).contains("error: "));
    assert!(
        !(reported && out.status.code() == Some(0)),
        "`acvus {}` reported an error and exited 0:\n{}{}",
        args.join(" "),
        text(&out.stdout),
        text(&out.stderr)
    );
    out
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
    write(dir.path(), "boom.acvus", "let xs = [1, 2, 3];\nxs[9]\n");
    let out = acvus(dir.path(), &["run", "boom.acvus"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(text(&out.stdout), "");
    assert_eq!(
        text(&out.stderr),
        "error: index out of bounds: the len is 3 but the index is 9\n"
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

/// The listing `ops` prints is the interpreter's own walk: the same source
/// through `acvus-interpreter-test`'s compile path, with the registries and
/// the `!` return declaration the CLI compiles with, renders the same text.
#[test]
fn ops_prints_the_prepared_listing_and_a_broken_script_is_refused() {
    let source = "let xs = [3, 1];\nlet total = 0;\nlet i = 0;\nwhile i < 2 { total = total + xs[i]; i = i + 1; }\ntotal\n";
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "loop.acvus", source);
    let out = acvus(dir.path(), &["ops", "loop.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));

    let interner = Interner::new();
    let registries = {
        let mut r = acvus_ext::std_registries();
        r.push(acvus_ext_net::http_registry());
        r
    };
    let blocks =
        script_listing_with_externs(&interner, source, Context::default(), registries, Ty::Never);
    assert_eq!(
        text(&out.stdout),
        format!("main:\n{}", listing_text(&blocks))
    );

    write(dir.path(), "bad.acvus", "let a = 1;\na + \"x\"\n");
    let out = acvus(dir.path(), &["ops", "bad.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(text(&out.stdout), "");
    assert!(
        text(&out.stderr).contains("type mismatch"),
        "{}",
        text(&out.stderr)
    );
}

/// One script's exit status from each command.
struct Exits {
    script: &'static str,
    check: i32,
    mir: i32,
    ops: i32,
    run: i32,
}

fn exits(dir: &Path, command: &str, script: &str, code: i32) {
    let out = acvus(dir, &[command, script]);
    assert_eq!(
        out.status.code(),
        Some(code),
        "`acvus {command} {script}`:\n{}",
        text(&out.stderr)
    );
}

/// `check` and `mir` stop where checking stops; `ops` and `run` prepare, and
/// what only `prepare` refuses is theirs alone. An undeclared `@name` is
/// such a refusal: nothing before `prepare` rejects it, so `check` accepts
/// the script and the interpreter refuses it at `EXIT_RUN`.
#[test]
fn each_command_exits_by_the_stage_that_refused_the_script() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "parse.acvus", "let x = ;\n");
    write(
        dir.path(),
        "types.acvus",
        "let x = 1;\nlet y = x + \"a\";\n",
    );
    write(dir.path(), "context.acvus", "let x = @nope + 1; x\n");
    write(dir.path(), "panic.acvus", "let xs = [1, 2, 3];\nxs[9]\n");
    let expected = [
        Exits {
            script: "parse.acvus",
            check: 1,
            mir: 1,
            ops: 1,
            run: 1,
        },
        Exits {
            script: "types.acvus",
            check: 1,
            mir: 1,
            ops: 1,
            run: 1,
        },
        Exits {
            script: "context.acvus",
            check: 0,
            mir: 0,
            ops: 2,
            run: 2,
        },
        Exits {
            script: "panic.acvus",
            check: 0,
            mir: 0,
            ops: 0,
            run: 2,
        },
    ];
    for e in expected {
        exits(dir.path(), "check", e.script, e.check);
        exits(dir.path(), "mir", e.script, e.mir);
        exits(dir.path(), "ops", e.script, e.ops);
        exits(dir.path(), "run", e.script, e.run);
    }
    assert_eq!(
        acvus(dir.path(), &["frob", "parse.acvus"]).status.code(),
        Some(64)
    );
    assert_eq!(
        acvus(dir.path(), &["run", "--json", "parse.acvus"])
            .status
            .code(),
        Some(64)
    );
}

#[test]
fn a_parse_error_names_what_the_grammar_wanted_in_the_language_s_words() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "empty.acvus", "let x = ;\n");
    let out = acvus(dir.path(), &["check", "empty.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert_eq!(
        err,
        "error: expected an expression, found `;`\n  --> empty.acvus:1:9\n  |\n1 | let x = ;\n  |         ^\n"
    );
    for internal in ["int_of", "fmt_start", "$ref", "@ref", "expected one of"] {
        assert!(!err.contains(internal), "{internal} in {err}");
    }

    write(dir.path(), "stmt.acvus", "let x = 1; }\n");
    let err = text(&acvus(dir.path(), &["check", "stmt.acvus"]).stderr);
    assert_eq!(
        err.lines().next(),
        Some("error: expected a statement, found `}`")
    );

    write(dir.path(), "paren.acvus", "let x = (1;\n");
    let err = text(&acvus(dir.path(), &["check", "paren.acvus"]).stderr);
    assert_eq!(
        err.lines().next(),
        Some("error: expected `)` or `,`, found `;`")
    );

    write(dir.path(), "name.acvus", "let 1 = 2;\n");
    let err = text(&acvus(dir.path(), &["check", "name.acvus"]).stderr);
    assert_eq!(
        err.lines().next(),
        Some("error: expected a name, found `1`")
    );
}

#[test]
fn json_puts_the_diagnostics_on_stdout_and_nothing_else() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bad.acvus", "let x = 1;\nlet y = x + \"a\";\n");
    let out = acvus(dir.path(), &["check", "--json", "bad.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(text(&out.stderr), "");
    let array: Vec<serde_json::Value> = serde_json::from_str(&text(&out.stdout)).unwrap();
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["severity"], "error");
    assert_eq!(array[0]["message"], "type mismatch in `+`: i64 vs String");
    assert_eq!(array[0]["path"], "bad.acvus");
    assert_eq!(array[0]["line"], 2);
    assert_eq!(array[0]["col"], 9);
    assert_eq!(array[0]["span"], serde_json::json!([19, 26]));

    write(dir.path(), "ok.acvus", "let x = 1;\nx\n");
    for command in ["check", "mir"] {
        let out = acvus(dir.path(), &[command, "--json", "ok.acvus"]);
        assert_eq!(out.status.code(), Some(0));
        assert_eq!(text(&out.stdout), "[]\n");
        assert_eq!(text(&out.stderr), "");
    }
}

/// A template goes through the stages a script does, and a tag's diagnostic
/// carries the span the tag has in the template file.
#[test]
fn a_template_is_checked_as_a_script_is() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bad.acvt", "Hello {{ 1 + \"a\" }}!\n");
    let out = acvus(dir.path(), &["check", "bad.acvt"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        "error: type mismatch in `+`: i64 vs String\n  --> bad.acvt:1:10\n  |\n1 | Hello {{ 1 + \"a\" }}!\n  |          ^^^^^^^\n"
    );

    write(dir.path(), "ok.acvt", "Hello {{ @name }}!\n");
    write(dir.path(), "ctx.json", "{\"name\": \"acvus\"}");
    let out = acvus(dir.path(), &["check", "ok.acvt", "--context", "ctx.json"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "");
    // Without `--context` the tags are checked against an empty context, and
    // an `@name` no context declares is one only `prepare` refuses.
    let out = acvus(dir.path(), &["check", "ok.acvt"]);
    assert_eq!(out.status.code(), Some(0));
    let out = acvus(dir.path(), &["run", "ok.acvt"]);
    assert_eq!(out.status.code(), Some(2));
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

/// A refusal whose story needs a second place shows both, in source order,
/// with the lines between them elided — and `--json` carries the same labels.
#[test]
fn a_use_after_move_shows_the_move_and_the_use_with_the_lines_between_elided() {
    let dir = tempfile::tempdir().unwrap();
    let source = "let a = [1, 2];\nlet b = a;\nlet c = 1;\nlet d = 2;\nlet e = 3;\na\n";
    write(dir.path(), "mv.acvus", source);

    let out = acvus(dir.path(), &["check", "mv.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        [
            "error: `a` is used here after it was moved",
            "  --> mv.acvus:6:1",
            "  |",
            "2 | let b = a;",
            "  |         - moved here",
            "...",
            "6 | a",
            "  | ^ `a` is used here after it was moved",
            "",
        ]
        .join("\n")
    );

    let out = acvus(dir.path(), &["check", "--json", "mv.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let array: Vec<serde_json::Value> = serde_json::from_str(&text(&out.stdout)).unwrap();
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["line"], 6);
    assert_eq!(array[0]["col"], 1);
    assert_eq!(
        array[0]["labels"],
        serde_json::json!([{
            "line": 2,
            "col": 9,
            "span": [24, 25],
            "text": "moved here",
        }])
    );
}
