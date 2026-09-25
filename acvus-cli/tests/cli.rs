//! `acvus` at its contract (RFC-0031): what reaches stdout, what reaches
//! stderr in the diagnostic shape, and the exit status of each outcome.

use std::path::Path;
use std::process::Output;

use acvus_interpreter_test::Context;
use acvus_interpreter_test::listing::{script_listing_with_externs, text as listing_text};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn acvus(dir: &Path, args: &[&str]) -> Output {
    let out = crate::sandbox::acvus()
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
fn a_script_prints_its_value_as_json() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "sum.acvus",
        "let items = [1, 2, 3];\nlet it = as_iter(&items);\nlet total = 0;\nwhile let Some(x) = next(&mut it) { total = total + *x; }\n{ total: total, tag: \"ok\".to_string(), }\n",
    );
    let out = acvus(dir.path(), &["run", "sum.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "{\"tag\":\"ok\",\"total\":6}\n");
    assert_eq!(text(&out.stderr), "");
}

#[test]
fn a_template_prints_its_text_and_an_expression_prints_its_value() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "hi.acvt", "Hello {{ $name }}!");
    let out = acvus(dir.path(), &["run", "hi.acvt", "name=\"acvus\""]);
    assert_eq!(text(&out.stdout), "Hello acvus!");
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

struct Panics {
    args: &'static [&'static str],
    message: &'static str,
}

#[test]
fn an_unused_division_that_can_panic_panics_at_every_level() {
    let dir = crate::sandbox::tempdir();
    let cases = [
        Panics {
            args: &["let d = 100 / 0; 5"],
            message: "attempt to divide by zero",
        },
        Panics {
            args: &["let d = 100 % 0; 5"],
            message: "attempt to calculate the remainder with a divisor of zero",
        },
        Panics {
            args: &["let d = $x / $y; 5", "x=100", "y=0"],
            message: "attempt to divide by zero",
        },
        Panics {
            args: &["let m = -9223372036854775807 - 1; let d = m / -1; 5"],
            message: "attempt to divide with overflow",
        },
        Panics {
            args: &["let m = -9223372036854775807 - 1; let d = m % -1; 5"],
            message: "attempt to calculate the remainder with overflow",
        },
    ];
    for opt in ["none", "full"] {
        for case in &cases {
            let args = [&["run", "--opt", opt, "-e"], case.args].concat();
            let out = acvus(dir.path(), &args);
            assert_eq!(out.status.code(), Some(2), "at {opt}: {:?}", case.args);
            assert_eq!(text(&out.stdout), "", "at {opt}: {:?}", case.args);
            assert_eq!(
                text(&out.stderr),
                format!("error: {}\n", case.message),
                "at {opt}: {:?}",
                case.args
            );
        }
    }
}

#[test]
fn an_unused_division_that_cannot_panic_and_an_unused_overflow_run_past() {
    let dir = crate::sandbox::tempdir();
    let values = [
        vec!["let d = $x / $y; 5", "x=100", "y=3"],
        vec!["let d = $x / 2; 5", "x=100"],
        vec!["let d = 7.0 / 0.0; 5"],
        vec!["let m = 9223372036854775807; let d = m + 1; 5"],
    ];
    for opt in ["none", "full"] {
        for source in &values {
            let args = [&["run", "--opt", opt, "-e"], source.as_slice()].concat();
            let out = acvus(dir.path(), &args);
            assert_eq!(out.status.code(), Some(0), "at {opt}: {source:?}: {}", text(&out.stderr));
            assert_eq!(text(&out.stdout), "5\n", "at {opt}: {source:?}");
        }
    }
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
        Some("error: type mismatch in `+`: String vs i64")
    );
    assert!(err.contains("--> cat.acvus:1:16"), "{err}");
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

#[test]
fn mir_prints_the_stage_facts_and_the_cost_under_each_for() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "sum.acvus", "let s = 0;\nfor x in [1, 2, 3] { s = s + x; }\ns\n");
    let out = acvus(dir.path(), &["mir", "sum.acvus"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let dump = text(&out.stdout);
    let facts: Vec<&str> = dump
        .lines()
        .filter_map(|line| line.split_once("// ").map(|(_, fact)| fact))
        .collect();
    assert_eq!(
        facts,
        [
            "L1: cycle Carried(r5) any_order law(Op(Add) exact commutative) {+}",
            "control upfront",
            "cost in place: no stage runs apart",
        ],
        "{dump}"
    );
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
/// what only a run refuses is its alone. A source that names a context runs
/// only over a space, so a run with none is a usage error the other three
/// commands do not see.
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
            ops: 0,
            run: 64,
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

    write(dir.path(), "loop.acvus", "for x in [1] { };\n");
    let err = text(&acvus(dir.path(), &["check", "loop.acvus"]).stderr);
    assert_eq!(
        err,
        "error: `;` is not allowed after a `for` block\n  --> loop.acvus:1:17\n  |\n1 | for x in [1] { };\n  |                 ^\n"
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
fn a_check_reports_every_parse_error_and_what_parsed_at_once() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "broken.acvus",
        "let a = 1;\nlet = 2;\nlet b = a * \"x\";\nfoo(;\nb\n",
    );
    let out = acvus(dir.path(), &["check", "broken.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    let headlines: Vec<&str> = err
        .lines()
        .filter(|line| line.starts_with("error:"))
        .collect();
    assert_eq!(headlines.len(), 3, "{err}");
    assert_eq!(headlines[0], "error: expected a name, found `=`");
    assert!(
        headlines[1].starts_with("error: expected an expression")
            && headlines[1].ends_with("found `;`"),
        "{err}"
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
    let out = acvus(dir.path(), &["check", "--json", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "[]\n{\"inputs\":[]}\n");
    assert_eq!(text(&out.stderr), "");

    let out = acvus(dir.path(), &["mir", "--json", "ok.acvus"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "[]\n");
    assert_eq!(text(&out.stderr), "");
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
    let out = acvus(dir.path(), &["check", "ok.acvt"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "");
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
    write(dir.path(), "bump.acvus", "let count = 7;\ncount = count + 1;\ncount\n");
    let out = acvus(dir.path(), &["run", "bump.acvus", "--time"]);
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
    assert!(
        err.lines().all(|l| l.starts_with("time: ")),
        "the command printed nothing on stderr but its times: {err}"
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

    let out = acvus(dir.path(), &["check", "bump.acvus", "--time"]);
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

/// RFC-0029: the exclusion rule is asked in pass 0 of `graph::optimize` and
/// nowhere else. Asked a second time after the rewrites, this program printed
/// the same refusal twice, the second time with no labels at all.
#[test]
fn a_write_while_a_reference_is_live_is_refused_once() {
    let dir = tempfile::tempdir().unwrap();
    let source = "let v = [1, 2];\nlet r = &v[0];\nv = [3, 4];\n*r\n";
    write(dir.path(), "wr.acvus", source);

    let out = acvus(dir.path(), &["check", "wr.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        [
            "error: `v` is written here while a reference to it is live",
            "  --> wr.acvus:3:1",
            "  |",
            "2 | let r = &v[0];",
            "  |         ----- borrowed here",
            "3 | v = [3, 4];",
            "  | ^^^^^^^^^^^ `v` is written here while a reference to it is live",
            "4 | *r",
            "  |  - the reference is used here",
            "",
        ]
        .join("\n")
    );
}

/// RFC-0064 rule 5: a lambda called after the storage it borrows was
/// written names the capture and the call. One write is one conflict, so the
/// reference the lambda captured adds its own labels to this refusal rather
/// than a refusal of its own — here it has none, because `r`'s last use in
/// `main` is the capture itself.
#[test]
fn a_write_while_a_capturing_lambda_is_live_shows_the_capture_and_the_call() {
    let dir = tempfile::tempdir().unwrap();
    let source =
        "let v = [1, 2, 3];\nlet r = &v;\nlet f = |k| -> len(r) + k;\nv = [4, 5, 6];\nf(1)\n";
    write(dir.path(), "cap.acvus", source);

    let out = acvus(dir.path(), &["check", "cap.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr),
        [
            "error: `v` is written here while a reference to it is live",
            "  --> cap.acvus:4:1",
            "  |",
            "3 | let f = |k| -> len(r) + k;",
            "  |         ----------------- captured here",
            "4 | v = [4, 5, 6];",
            "  | ^^^^^^^^^^^^^^ `v` is written here while a reference to it is live",
            "5 | f(1)",
            "  | ---- the lambda is called here",
            "",
        ]
        .join("\n")
    );

    let out = acvus(dir.path(), &["check", "--json", "cap.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let array: Vec<serde_json::Value> = serde_json::from_str(&text(&out.stdout)).unwrap();
    assert_eq!(array.len(), 1, "{}", text(&out.stdout));
    assert_eq!(
        array[0]["message"],
        "`v` is written here while a reference to it is live"
    );
    assert_eq!(array[0]["line"], 4);
    assert_eq!(
        array[0]["labels"],
        serde_json::json!([
            {
                "line": 3,
                "col": 9,
                "span": [39, 56],
                "text": "captured here",
            },
            {
                "line": 5,
                "col": 1,
                "span": [73, 77],
                "text": "the lambda is called here",
            },
        ])
    );
}

#[test]
fn a_write_while_two_references_are_live_is_one_refusal_naming_both() {
    let dir = tempfile::tempdir().unwrap();
    let source = "let a = [1, 2];\nlet r = &a;\nlet s = &a;\na = [3, 4];\nlen(r) + len(s)\n";
    write(dir.path(), "two.acvus", source);

    let out = acvus(dir.path(), &["check", "--json", "two.acvus"]);
    assert_eq!(out.status.code(), Some(1));
    let array: Vec<serde_json::Value> = serde_json::from_str(&text(&out.stdout)).unwrap();
    assert_eq!(array.len(), 1, "{}", text(&out.stdout));
    assert_eq!(
        array[0]["labels"],
        serde_json::json!([
            { "line": 2, "col": 9, "span": [24, 26], "text": "borrowed here" },
            { "line": 5, "col": 5, "span": [56, 57], "text": "the reference is used here" },
            { "line": 3, "col": 9, "span": [36, 38], "text": "borrowed here" },
            { "line": 5, "col": 14, "span": [65, 66], "text": "the reference is used here" },
        ])
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
fn a_comment_line_of_a_template_leaves_nothing_in_the_output() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "hi.acvt",
        "% // the input carries the name\nHello {{ $name }}!",
    );
    let out = acvus(dir.path(), &["run", "hi.acvt", "name=\"world\""]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "Hello world!");
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
            "let re = regex(\"a+\")?; is_match(&re, \"baaad\")",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "true\n");

    let out = acvus(
        dir.path(),
        &[
            "run",
            "-e",
            "let d = parse_date(\"2026-09-19T09:58:03\", \"%Y-%m-%dT%H:%M:%S\")?; timestamp(d)",
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
        let out = crate::sandbox::acvus()
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
fn a_vec_and_a_deque_print_as_the_json_array_of_their_items() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        ("let v = vec::new(); push(&mut v, 1); push(&mut v, 2); v", "[1,2]\n"),
        (
            "let v = vec::new(); push(&mut v, \"a\".to_string()); push(&mut v, \"b\".to_string()); v",
            "[\"a\",\"b\"]\n",
        ),
        (
            "let v = vec::new(); let w = vec::new(); push(&mut w, 1); push(&mut w, 2); push(&mut v, w); push(&mut v, vec([3])); v",
            "[[1,2],[3]]\n",
        ),
        ("let v = vec([1]); pop(&mut v); v", "[]\n"),
        ("vec([{ x: 1, }, { x: 2, }])", "[{\"x\":1},{\"x\":2}]\n"),
        ("let d = deque(); push_back(&mut d, 1); push_front(&mut d, 0); d", "[0,1]\n"),
        ("let d = deque(); push_back(&mut d, 1); pop_back(&mut d); d", "[]\n"),
    ];
    for (source, expected) in cases {
        for level in ["full", "none"] {
            let out = acvus(dir.path(), &["run", "-e", source, "--opt", level]);
            assert_eq!(out.status.code(), Some(0), "{source}: {}", text(&out.stderr));
            assert_eq!(text(&out.stdout), expected, "{source} at opt {level}");
        }
    }
}

#[test]
fn a_map_prints_as_its_key_value_pairs_and_a_set_as_its_keys_in_insertion_order() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            "let m = hash_map(); insert(&mut m, \"b\".to_string(), 2); insert(&mut m, \"a\".to_string(), 1); \
             insert(&mut m, \"c\".to_string(), 3); retain(&mut m, |k, v| -> *v != 2); \
             insert(&mut m, \"b\".to_string(), 4); m",
            "[[\"a\",1],[\"c\",3],[\"b\",4]]\n",
        ),
        (
            "let m = hash_map(); insert(&mut m, 2, vec([20])); insert(&mut m, 1, vec([10, 11])); \
             insert(&mut m, 3, vec([30])); insert(&mut m, 3, vec([31])); \
             retain(&mut m, |k, v| -> *k != 2); insert(&mut m, 2, vec([21])); m",
            "[[1,[10,11]],[3,[31]],[2,[21]]]\n",
        ),
        (
            "let m = hash_map(); insert(&mut m, 1, 10); retain(&mut m, |k, v| -> false); m",
            "[]\n",
        ),
        (
            "let s = hash_set(); insert(&mut s, \"b\".to_string()); insert(&mut s, \"a\".to_string()); \
             insert(&mut s, \"c\".to_string()); insert(&mut s, \"a\".to_string()); \
             let gone = hash_set(); insert(&mut gone, \"b\".to_string()); \
             let s = difference(s, gone); insert(&mut s, \"b\".to_string()); s",
            "[\"a\",\"c\",\"b\"]\n",
        ),
        (
            "let s = hash_set(); insert(&mut s, 1); let gone = hash_set(); insert(&mut gone, 1); difference(s, gone)",
            "[]\n",
        ),
    ];
    for (source, expected) in cases {
        for level in ["full", "none"] {
            let out = acvus(dir.path(), &["run", "-e", source, "--opt", level]);
            assert_eq!(out.status.code(), Some(0), "{source}: {}", text(&out.stderr));
            assert_eq!(text(&out.stdout), expected, "{source} at opt {level}");
        }
    }
}

#[test]
fn a_value_with_no_data_view_prints_its_name_and_a_closure_never_reaches_the_printer() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        dir.path(),
        &["run", "-e", "match regex(\"a+\") { Ok(r) => r, Err(e) => panic(\"no\".to_string()), }"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let printed = text(&out.stdout);
    assert!(
        printed.starts_with("\"<") && printed.ends_with(">\"\n") && printed.contains("Regex"),
        "{printed}"
    );
    let out = acvus(dir.path(), &["run", "-e", "(1, |x| -> x + 1)"]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out.stderr));
    assert!(
        text(&out.stderr).contains("a closure does not leave the run it was made in"),
        "{}",
        text(&out.stderr)
    );
    assert_eq!(text(&out.stdout), "");
}

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

#[test]
fn lsp_takes_no_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(dir.path(), &["lsp", "extra"]);
    assert_eq!(out.status.code(), Some(64));
    assert!(
        text(&out.stderr).contains("error: lsp takes no arguments, not `extra`"),
        "{}",
        text(&out.stderr)
    );
}

fn framed(message: serde_json::Value) -> Vec<u8> {
    let body = message.to_string();
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

#[test]
fn lsp_serves_the_client_s_root_over_stdio_and_exits_after_shutdown() {
    use std::io::Write;
    use std::process::Stdio;

    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "bad.acvus", "1 + \"a\"\n");
    let root = url::Url::from_file_path(dir.path()).expect("a temporary path is absolute");
    let script = url::Url::from_file_path(dir.path().join("bad.acvus"))
        .expect("a temporary path is absolute");
    let mut server = crate::sandbox::acvus()
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary runs");
    let messages = [
        serde_json::json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": { "processId": null, "rootUri": root.as_str(), "capabilities": {} } }),
        serde_json::json!({ "jsonrpc": "2.0", "method": "initialized", "params": {} }),
        serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "shutdown" }),
        serde_json::json!({ "jsonrpc": "2.0", "method": "exit" }),
    ];
    let mut stdin = server.stdin.take().expect("stdin is piped");
    for message in messages {
        stdin
            .write_all(&framed(message))
            .expect("the server reads stdin");
    }
    drop(stdin);
    let out = server.wait_with_output().expect("the server exits");

    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    let sent: Vec<serde_json::Value> = stdout
        .split("Content-Length: ")
        .skip(1)
        .map(|framed| {
            let (_, body) = framed.split_once("\r\n\r\n").expect("a header ends");
            serde_json::from_str(body).expect("a message is JSON")
        })
        .collect();
    let initialized = sent
        .iter()
        .find(|message| message["id"] == 1)
        .expect("initialize is answered");
    assert_eq!(
        initialized["result"]["capabilities"]["positionEncoding"],
        "utf-16"
    );
    let published: Vec<&serde_json::Value> = sent
        .iter()
        .filter(|message| message["method"] == "textDocument/publishDiagnostics")
        .collect();
    assert_eq!(published.len(), 1, "{stdout}");
    assert_eq!(published[0]["params"]["uri"], script.as_str());
    assert_eq!(
        published[0]["params"]["diagnostics"][0]["source"], "acvus",
        "{stdout}"
    );
}

/// A bound on the wait for a process that exits at once, generous for a
/// loaded machine; not a measurement.
const LSP_EXIT_PATIENCE: std::time::Duration = std::time::Duration::from_secs(30);
const LSP_EXIT_POLL: std::time::Duration = std::time::Duration::from_millis(20);

#[test]
fn lsp_exits_with_1_after_a_refused_initialize_while_the_client_holds_stdin_open() {
    use std::io::{Read, Write};
    use std::process::Stdio;
    use std::time::Instant;

    let mut server = crate::sandbox::acvus()
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary runs");
    let mut stdin = server.stdin.take().expect("stdin is piped");
    stdin
        .write_all(&framed(serde_json::json!({ "jsonrpc": "2.0", "id": 1,
            "method": "initialize", "params": { "processId": null, "capabilities": {} } })))
        .expect("the server reads stdin");
    stdin.flush().expect("stdin flushes");

    let deadline = Instant::now() + LSP_EXIT_PATIENCE;
    let status = loop {
        if let Some(status) = server.try_wait().expect("the server's status reads") {
            break status;
        }
        if Instant::now() > deadline {
            server.kill().expect("the server is killed");
            server.wait().expect("the killed server is reaped");
            panic!("`acvus lsp` did not exit after a refused initialize");
        }
        std::thread::sleep(LSP_EXIT_POLL);
    };
    drop(stdin);

    let mut stderr = String::new();
    server
        .stderr
        .take()
        .expect("stderr is piped")
        .read_to_string(&mut stderr)
        .expect("stderr reads");
    assert_eq!(status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("error: the client gave neither a root URI nor a workspace folder"),
        "{stderr}"
    );
}
