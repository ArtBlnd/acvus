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
    assert_eq!(
        text(&out.stderr),
        "error: `$mode` is required and not bound\n"
    );
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
fn a_binding_that_computes_its_value_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        written(dir.path()),
        &["run", "prompt.acvt", "--bind", "x=f()"],
    );
    assert_eq!(out.status.code(), Some(64));
    assert_eq!(
        text(&out.stderr),
        "error: --bind x: `f()` is not a value a literal writes\n"
    );
}

#[test]
fn a_binding_whose_value_has_no_type_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let out = acvus(
        written(dir.path()),
        &["run", "prompt.acvt", "--bind", "xs=[1, \"a\"]"],
    );
    assert_eq!(out.status.code(), Some(64));
    assert_eq!(
        text(&out.stderr),
        "error: --bind xs: the elements of an array in the value have no one type\n"
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

// -- Any value a literal writes (RFC-0087) ------------------------------

fn run_bound(source: &str, binding: &str) -> Output {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.acvt"), source).expect("write a fixture");
    let unbound = acvus(dir.path(), &["check", "t.acvt"]);
    assert_eq!(unbound.status.code(), Some(0), "{}", text(&unbound.stderr));
    acvus(dir.path(), &["run", "t.acvt", "--bind", binding])
}

fn prints(out: Output, expected: &str) {
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), expected);
}

const BY_VARIANT: &str = "\
% match $mode
% Mode::Review =>
Review.
% Mode::Explain =>
Explain.
% end
";

#[test]
fn a_bound_variant_renders_the_arm_it_chose() {
    prints(run_bound(BY_VARIANT, "mode=Mode::Review"), "Review.\n");
    prints(run_bound(BY_VARIANT, "mode=Mode::Explain"), "Explain.\n");
}

#[test]
fn a_bound_variant_carries_its_payload_to_the_arm() {
    let source = "\
% match $who
% Who::Named(n) =>
Hello, {{ n }}.
% Who::Anonymous =>
Hello, stranger.
% end
";
    prints(
        run_bound(source, "who=Who::Named(\"Ada\")"),
        "Hello, Ada.\n",
    );
}

#[test]
fn a_bound_object_is_read_by_field() {
    let source = "{{ $user.name }} is {{ $user.role }}.\n";
    prints(
        run_bound(source, "user={name: \"Ada\", role: \"admin\",}"),
        "Ada is admin.\n",
    );
}

/// A `for` needs its head while the body is checked, so the array is read
/// by a `for` only bound (RFC-0087 Why).
#[test]
fn a_bound_array_is_traversed() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("t.acvt"),
        "% for n in $names\n- {{ n }}\n% end\n",
    )
    .expect("write a fixture");
    prints(
        acvus(
            dir.path(),
            &["run", "t.acvt", "--bind", "names=[\"a\", \"b\"]"],
        ),
        "- a\n- b\n",
    );
}

// -- A dispatch on a bound structured value folds (RFC-0071 rule 5) ------

const ARM_BY_VARIANT: &str = "\
% match $mode
% Mode::Review =>
{{ $rules }}
% Mode::Explain =>
{{ $examples }}
% end
";

const ARM_BY_NESTED_PATTERN: &str = "\
% match $o
% {k: Some(M::R(n)), t: (1, \"x\"),} =>
{{ $a }}
% _ =>
{{ $b }}
% end
";

fn runs_at_both_levels(source: &str, bindings: &[&str], expected: &str) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.acvt"), source).expect("write a fixture");
    for opt in ["none", "full"] {
        let mut args = vec!["run", "t.acvt", "--opt", opt];
        for binding in bindings {
            args.extend(["--bind", binding]);
        }
        let out = acvus(dir.path(), &args);
        assert_eq!(out.status.code(), Some(0), "at {opt}: {}", text(&out.stderr));
        assert_eq!(text(&out.stdout), expected, "at {opt}");
    }
}

#[test]
fn a_bound_variant_runs_its_arm_without_the_other_arms_input() {
    runs_at_both_levels(
        ARM_BY_VARIANT,
        &["mode=Mode::Review", "rules=\"R\""],
        "R\n",
    );
    runs_at_both_levels(
        ARM_BY_VARIANT,
        &["mode=Mode::Explain", "examples=\"E\""],
        "E\n",
    );
}

#[test]
fn a_bound_object_runs_the_arm_its_nested_pattern_chose() {
    runs_at_both_levels(
        ARM_BY_NESTED_PATTERN,
        &["o={k: Some(M::R(3)), t: (1, \"x\"),}", "a=\"A\""],
        "A\n",
    );
    runs_at_both_levels(
        ARM_BY_NESTED_PATTERN,
        &["o={k: None, t: (1, \"x\"),}", "b=\"B\""],
        "B\n",
    );
    runs_at_both_levels(
        ARM_BY_NESTED_PATTERN,
        &["o={k: Some(M::R(3)), t: (2, \"x\"),}", "b=\"B\""],
        "B\n",
    );
}
