//! `--bind` at the CLI's contract: a bound `$` is a constant and the text it
//! decides against is not written, while a `$` no binding fixed and the code
//! still reads is a compile-time refusal (RFC-0071 rule 5).

use std::path::Path;
use std::process::{Command, Output};

const BY_MODE: &str = "\
% match $mode
% \"review\" =>
Review the code below.
% _ =>
Explain the code below.
% end
";

fn acvus(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_acvus"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("the binary runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn written(dir: &Path) -> &Path {
    std::fs::write(dir.join("prompt.acvt"), BY_MODE).expect("write a fixture");
    dir
}

#[test]
fn a_bound_input_renders_the_arm_it_chose() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        written(dir.path()),
        &["run", "prompt.acvt", "--bind", "mode=\"review\""],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "Review the code below.\n");
}

#[test]
fn another_binding_renders_the_other_arm() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        written(dir.path()),
        &["run", "prompt.acvt", "--bind", "mode=\"explain\""],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "Explain the code below.\n");
}

#[test]
fn an_input_no_binding_fixed_refuses_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(written(dir.path()), &["run", "prompt.acvt"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(text(&out.stdout), "");
    assert_eq!(text(&out.stderr), "error: `$mode` is required and not bound\n");
}

#[test]
fn check_reports_the_inputs_a_run_would_still_need() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(written(dir.path()), &["check", "prompt.acvt"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stderr), "input $mode: String\n");

    let out = acvus(
        written(dir.path()),
        &["check", "prompt.acvt", "--bind", "mode=\"review\""],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stderr), "");
}

#[test]
fn check_json_lists_the_inputs_after_the_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(written(dir.path()), &["check", "prompt.acvt", "--json"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(
        text(&out.stdout),
        "[]\n{\"inputs\":[{\"name\":\"mode\",\"type\":\"String\"}]}\n"
    );
}

#[test]
fn a_binding_is_a_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        written(dir.path()),
        &["run", "prompt.acvt", "--bind", "mode=[1,2]"],
    );
    assert_eq!(out.status.code(), Some(64));
    assert!(
        text(&out.stderr).contains("a binding is a scalar"),
        "{}",
        text(&out.stderr)
    );
}

#[test]
fn an_unbound_input_of_an_expression_is_refused_rather_than_read() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(dir.path(), &["run", "-e", "$x + 1"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(text(&out.stdout), "");
    assert_eq!(text(&out.stderr), "error: `$x` is required and not bound\n");

    let bound = acvus(dir.path(), &["run", "-e", "$x + 1", "--bind", "x=41"]);
    assert_eq!(bound.status.code(), Some(0), "{}", text(&bound.stderr));
    assert_eq!(text(&bound.stdout), "42\n");
}

const LENDS_TO_A_STR: &str = "\
% let on = string::contains(&$input, \"X\")
{{ if on { \"yes\" } else { \"no\" } }}
";

/// `&$input` at a `&str` parameter types the input `String`, which lends the
/// parameter its view, at both optimization levels.
#[test]
fn an_input_lent_to_a_str_parameter_is_a_string_that_prepares() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lends.acvt"), LENDS_TO_A_STR).expect("write a fixture");
    for opt in ["full", "none"] {
        let out = acvus(dir.path(), &["check", "lends.acvt", "--opt", opt]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        assert_eq!(text(&out.stderr), "input $input: String\n");

        let out = acvus(dir.path(), &["ops", "lends.acvt", "--opt", opt]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));

        for (input, shown) in [("\"aXb\"", "yes\n"), ("\"ab\"", "no\n")] {
            let binding = format!("input={input}");
            let out = acvus(
                dir.path(),
                &["run", "lends.acvt", "--opt", opt, "--bind", &binding],
            );
            assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
            assert_eq!(text(&out.stdout), shown);
        }
    }
}
