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
        "let it = as_iter(&@items);\nlet total = 0;\nwhile let Some(x) = next(&mut it) { total = total + *x; }\n@count = @count + 1;\n{ total: total, tag: \"ok\".to_string(), }\n",
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
fn a_closure_parameter_lent_to_a_str_parameter_runs() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        dir.path(),
        &["run", "-e", "let f = |x| -> concat(\"q\", &x); f(\"z\")"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "qz\n");
    assert_eq!(text(&out.stderr), "");
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
        err.contains("error: type mismatch in `<`: _ vs _"),
        "a variable the solve never bound reads as `_`: {err}"
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
    write(
        dir.path(),
        "cat.acvus",
        "let f = |k| -> k + \"a\".to_string();\nf(1)\n",
    );
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
        "let f = |k| -> k + \"a\".to_string();\nf(\"b\".to_string())\n",
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
        r.push(acvus_ext::regex_registry());
        r.push(acvus_ext::datetime_registry());
        r.push(acvus_ext::io_registry());
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
            check: 1,
            mir: 1,
            ops: 1,
            run: 1,
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
    assert_eq!(array[0]["message"], "type mismatch in `+`: i64 vs str");
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
        "error: type mismatch in `+`: i64 vs str\n  --> bad.acvt:1:10\n  |\n1 | Hello {{ 1 + \"a\" }}!\n  |          ^^^^^^^\n"
    );

    write(dir.path(), "ok.acvt", "Hello {{ @name }}!\n");
    write(dir.path(), "ctx.json", "{\"name\": \"acvus\"}");
    let out = acvus(dir.path(), &["check", "ok.acvt", "--context", "ctx.json"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "");
    let out = acvus(dir.path(), &["check", "ok.acvt"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        "error: `@name` is not a declared context\n  --> ok.acvt:1:10\n  |\n1 | Hello {{ @name }}!\n  |          ^^^^^\n"
    );
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

struct Stage {
    name: String,
    ms: f64,
}

fn stages(stderr: &str) -> Vec<Stage> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix("time: "))
        .map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().expect("a stage name").to_string();
            let ms = words
                .next()
                .expect("a duration")
                .parse()
                .expect("a duration in milliseconds");
            assert_eq!(words.next(), Some("ms"), "{line}");
            Stage { name, ms }
        })
        .collect()
}

#[test]
fn time_reports_every_stage_the_command_ran_after_its_output() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bump.acvus", "@count = @count + 1;\n@count\n");
    write(dir.path(), "ctx.json", "{\"count\": 7}");
    let out = acvus(
        dir.path(),
        &["run", "bump.acvus", "--context", "ctx.json", "--time"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "8\n");

    let err = text(&out.stderr);
    let timed = stages(&err);
    assert_eq!(
        timed.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["compile", "prepare", "run"],
        "{err}"
    );
    assert!(timed.iter().all(|s| s.ms >= 0.0), "{err}");
    assert_eq!(err.lines().next(), Some("write @count = 8"), "{err}");
    assert!(
        err.lines().skip(1).all(|l| l.starts_with("time: ")),
        "the times come after everything else the command printed: {err}"
    );

    let compile = err
        .lines()
        .find(|l| l.starts_with("time: compile "))
        .expect("the compile line");
    let subs = compile
        .split_once('(')
        .expect("compile carries its sub-stages")
        .1
        .trim_end_matches(')')
        .split(", ")
        .map(|sub| {
            let mut words = sub.split_whitespace();
            let name = words.next().expect("a sub-stage name").to_string();
            let ms = words
                .next()
                .expect("a duration")
                .parse()
                .expect("a duration in milliseconds");
            Stage { name, ms }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        subs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["parse", "typeck", "lower", "optimize"],
        "{compile}"
    );
    assert!(subs.iter().all(|s| s.ms >= 0.0), "{compile}");

    let out = acvus(
        dir.path(),
        &["check", "bump.acvus", "--context", "ctx.json", "--time"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let err = text(&out.stderr);
    assert_eq!(
        stages(&err)
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["compile"],
        "{err}"
    );
}

#[test]
fn json_and_time_put_the_times_in_a_trailing_object_on_stdout() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "ok.acvus", "let xs = [1, 2];\nxs.len()\n");

    let out = acvus(dir.path(), &["check", "--json", "--time", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stderr), "");
    let stdout = text(&out.stdout);
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("[]"));
    let trailing: serde_json::Value =
        serde_json::from_str(lines.next_back().expect("a trailing object")).unwrap();
    let time = &trailing["time"];
    for stage in ["total", "parse", "typeck", "lower", "optimize"] {
        assert!(
            time["compile"][stage].as_f64().is_some_and(|ms| ms >= 0.0),
            "{trailing}"
        );
    }
    assert!(time["prepare"].is_null(), "{trailing}");
    assert!(time["run"].is_null(), "{trailing}");

    let out = acvus(dir.path(), &["ops", "--json", "--time", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    let trailing: serde_json::Value =
        serde_json::from_str(stdout.lines().next_back().expect("a trailing object")).unwrap();
    assert!(
        trailing["time"]["prepare"]
            .as_f64()
            .is_some_and(|ms| ms >= 0.0),
        "{trailing}"
    );
}

#[test]
fn without_time_no_command_reports_one() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "ok.acvus", "let xs = [1, 2];\nxs.len()\n");
    for command in ["check", "mir", "ops", "run"] {
        for args in [
            vec![command, "ok.acvus"],
            vec![command, "--json", "ok.acvus"],
        ] {
            if args.contains(&"--json") && command == "run" {
                continue;
            }
            let out = acvus(dir.path(), &args);
            assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
            assert!(
                stages(&text(&out.stderr)).is_empty(),
                "`acvus {}`:\n{}",
                args.join(" "),
                text(&out.stderr)
            );
            assert!(
                !text(&out.stdout).contains("\"time\""),
                "`acvus {}`:\n{}",
                args.join(" "),
                text(&out.stdout)
            );
        }
    }
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

/// Two values of one type from two sources: the refusal names the place they
/// meet and the two places the sources begin, and `--json` carries all three.
#[test]
fn two_iterators_from_two_sources_show_where_each_source_begins() {
    let dir = tempfile::tempdir().unwrap();
    let source = "let a = [1, 2] | into_iter;\nlet b = [3, 4] | into_iter;\nlet l = [a, b];\n";
    write(dir.path(), "id.acvus", source);

    let out = acvus(dir.path(), &["check", "id.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        [
            "error: `a` and `b` are values of one type from two different sources, and one place cannot hold both",
            "  --> id.acvus:3:9",
            "  |",
            "1 | let a = [1, 2] | into_iter;",
            "  |                  --------- `a`'s source begins here",
            "2 | let b = [3, 4] | into_iter;",
            "  |                  --------- `b`'s source begins here",
            "3 | let l = [a, b];",
            "  |         ^^^^^^ `a` and `b` meet here",
            "",
        ]
        .join("\n")
    );

    let out = acvus(dir.path(), &["check", "--json", "id.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let array: Vec<serde_json::Value> = serde_json::from_str(&text(&out.stdout)).unwrap();
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["line"], 3);
    assert_eq!(array[0]["col"], 9);
    assert_eq!(array[0]["primary"], "`a` and `b` meet here");
    assert_eq!(
        array[0]["labels"],
        serde_json::json!([
            {
                "line": 1,
                "col": 18,
                "span": [17, 26],
                "text": "`a`'s source begins here",
            },
            {
                "line": 2,
                "col": 18,
                "span": [45, 54],
                "text": "`b`'s source begins here",
            },
        ])
    );
}

#[test]
fn a_comment_runs_to_the_end_of_its_line_and_a_string_keeps_its_slashes() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "commented.acvus",
        "// what this script does\nlet host = \"https://acvus.example//a\".to_string(); // the URL\nlet n = 1 + // the rest of this line is not read\n    2;\n\"{{ &host }} {{ &n | to_string }}\"\n",
    );
    let out = acvus(dir.path(), &["run", "commented.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "https://acvus.example//a 3\n");
}

#[test]
fn a_comment_inside_a_template_tag_runs_to_the_tags_line_end() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "hi.acvt",
        "Hello {{ @name // the context carries it\n}}!",
    );
    write(dir.path(), "hi.json", "{\"name\": \"world\"}");
    let out = acvus(dir.path(), &["run", "hi.acvt", "--context", "hi.json"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "Hello world!\n");
}

/// `regex` and `datetime` are in the set `acvus run` registers, so a script
/// reaches them with no flag.
#[test]
fn regex_and_datetime_need_no_flag() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        dir.path(),
        &[
            "run",
            "-e",
            "let re = regex(\"a+\".to_string())?; is_match(&re, \"baaad\")",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "true\n");

    let out = acvus(
        dir.path(),
        &[
            "run",
            "-e",
            "let d = parse_date(\"2026-09-19T09:58:03\".to_string(), \"%Y-%m-%dT%H:%M:%S\".to_string())?; timestamp(d)",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "1789811883\n");
}

/// Every printed line precedes the result line, and `--time` stays behind
/// both, on stderr.
#[test]
fn print_writes_its_lines_before_the_result_and_the_times_follow_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        dir.path(),
        &["run", "-e", "print(\"x\"); print(\"y\"); 1", "--time"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "x\ny\n1\n");

    let err = text(&out.stderr);
    assert_eq!(
        stages(&err)
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["compile", "prepare", "run"],
        "{err}"
    );
    assert!(err.lines().all(|l| l.starts_with("time: ")), "{err}");
}

/// `--opt` chooses how hard the compiler works and nothing else: the value is
/// the same at either level, and `mir --opt none` prints the program the
/// source wrote, with the loop's slice of `xs` still inside the body that
/// `--opt full` hoists it out of.
#[test]
fn opt_none_and_opt_full_run_one_program_and_print_two() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "loop.acvus",
        "let xs = [1, 2, 3];\nlet total = 0;\nlet i = 0;\nwhile i < 3 {\n    total = total + xs[i] * 2;\n    i = i + 1;\n}\ntotal\n",
    );
    for level in ["none", "full"] {
        let out = acvus(dir.path(), &["run", "loop.acvus", "--opt", level]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        assert_eq!(text(&out.stdout), "12\n", "at opt {level}");
    }

    let hoisted = |level: &str| {
        let out = acvus(dir.path(), &["mir", "loop.acvus", "--opt", level]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        let dump = text(&out.stdout);
        let at = |needle: &str| {
            dump.lines()
                .position(|line| line.contains(needle))
                .unwrap_or_else(|| panic!("no `{needle}` in the dump at opt {level}:\n{dump}"))
        };
        at("as_slice") < at("L0(")
    };
    assert!(hoisted("full"));
    assert!(!hoisted("none"));
}

#[test]
fn the_time_line_names_the_level_it_compiled_at() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "ok.acvus", "let xs = [1, 2];\nxs.len()\n");

    for level in ["none", "full"] {
        let out = acvus(dir.path(), &["check", "ok.acvus", "--opt", level, "--time"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        let err = text(&out.stderr);
        let compile = err
            .lines()
            .find(|line| line.starts_with("time: compile "))
            .expect("the compile line");
        assert!(compile.contains(&format!(" at opt {level} (")), "{compile}");
    }

    let out = acvus(
        dir.path(),
        &["check", "--json", "--time", "--opt", "none", "ok.acvus"],
    );
    let stdout = text(&out.stdout);
    let trailing: serde_json::Value =
        serde_json::from_str(stdout.lines().next_back().expect("a trailing object")).unwrap();
    assert_eq!(trailing["time"]["compile"]["opt"], "none", "{trailing}");

    let out = acvus(dir.path(), &["check", "ok.acvus", "--time"]);
    let err = text(&out.stderr);
    assert!(err.contains("at opt full ("), "the default is full: {err}");
}

#[test]
fn a_level_the_compiler_does_not_have_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "ok.acvus", "let xs = [1, 2];\nxs.len()\n");
    for args in [
        vec!["run", "ok.acvus", "--opt", "fast"],
        vec!["run", "ok.acvus", "--opt"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_acvus"))
            .current_dir(dir.path())
            .args(&args)
            .output()
            .expect("the binary runs");
        assert_eq!(out.status.code(), Some(64), "`acvus {}`", args.join(" "));
        assert!(
            text(&out.stderr).contains("--opt takes none or full"),
            "{}",
            text(&out.stderr)
        );
    }
}

/// Each shape the checker used to admit and the machine could not run: the
/// refusal is the checker's, at `check`, and `run` never reaches `prepare`.
#[test]
fn what_the_machine_cannot_run_the_checker_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let refused = [
        (
            "context.acvus",
            "@n + 1\n",
            "`@n` is not a declared context",
        ),
        (
            "namespace.acvus",
            "let xs = [1];\nnope::len(&xs)\n",
            "a body does not return a reference",
        ),
        (
            "field.acvus",
            "let x = { a: 1, };\nx.b\n",
            "`x` has no `b` stored on every path that reaches here",
        ),
        (
            "reference.acvus",
            "let a = [1, 2];\n&a\n",
            "a reference to `a` cannot leave the body",
        ),
    ];
    for (name, source, words) in refused {
        write(dir.path(), name, source);
        for command in ["check", "run"] {
            let out = acvus(dir.path(), &[command, name]);
            assert_eq!(out.status.code(), Some(1), "{command} {name}");
            assert!(
                text(&out.stderr).contains(words),
                "{command} {name}: {}",
                text(&out.stderr)
            );
        }
    }
}

/// A literal arm over a place behind a reference tests what the reference
/// names, at both optimization levels.
#[test]
fn a_literal_arm_through_a_reference_runs() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "match.acvus",
        "let prog = [2, 1];\nlet pc = 1;\nmatch &prog[pc] { 1 => { 10 }, _ => { 20 } }\n",
    );
    for level in ["full", "none"] {
        let out = acvus(dir.path(), &["run", "match.acvus", "--opt", level]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        assert_eq!(text(&out.stdout), "10\n", "at opt {level}");
    }
}
