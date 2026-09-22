//! A bound `$` is a constant, and what it makes unreachable is `!`
//! (RFC-0071 Decision 5).

use acvus_ast::Literal;
use acvus_mir::graph::optimize::Opt;
use acvus_mir_test::{ShownInput, compile_template_bound};
use acvus_utils::Interner;

const BY_MATCH: &str = "\
% match $mode
% \"review\" =>
Review the code below and list defects.
{{ $rules }}
% _ =>
Explain the code below.
{{ $examples }}
% end
";

const BY_COMPARISON: &str = "\
% if $mode == \"review\"
Review the code below and list defects.
{{ $rules }}
% else
Explain the code below.
{{ $examples }}
% end
";

const BY_SIZE: &str = "\
% if $n > 3
many
{{ $many }}
% else
few
{{ $few }}
% end
";

const READ_IN_BOTH_ARMS: &str = "\
% match $mode
% \"review\" =>
Review for {{ $who }}.
% _ =>
Explain for {{ $who }}.
% end
";

struct Case {
    source: &'static str,
    bound: Vec<(&'static str, Literal)>,
}

fn text(held: &str) -> Literal {
    Literal::String(held.to_string())
}

fn inputs_at(source: &str, bound: &[(&str, Literal)], opt: Opt) -> Vec<ShownInput> {
    let interner = Interner::new();
    let compiled = compile_template_bound(&interner, source, bound, opt)
        .unwrap_or_else(|e| panic!("the template compiles: {e}"));
    let mut inputs = compiled.inputs;
    inputs.sort();
    inputs
}

fn required(source: &str, bound: &[(&str, Literal)]) -> Vec<String> {
    inputs_at(source, bound, Opt::Full)
        .into_iter()
        .map(|input| input.name)
        .collect()
}

fn names(held: &[&str]) -> Vec<String> {
    held.iter().map(|name| name.to_string()).collect()
}

fn shown(name: &str, ty: &str) -> ShownInput {
    ShownInput {
        name: name.to_string(),
        ty: ty.to_string(),
    }
}

#[test]
fn an_unbound_dispatch_requires_the_tag_and_both_arms() {
    assert_eq!(
        inputs_at(BY_MATCH, &[], Opt::Full),
        vec![
            shown("examples", "String"),
            shown("mode", "&str"),
            shown("rules", "String"),
        ]
    );
}

#[test]
fn binding_the_tag_leaves_the_arm_it_chose() {
    assert_eq!(
        required(BY_MATCH, &[("mode", text("review"))]),
        names(&["rules"])
    );
    assert_eq!(
        required(BY_MATCH, &[("mode", text("explain"))]),
        names(&["examples"])
    );
}

#[test]
fn binding_the_tag_leaves_the_arm_a_comparison_chose() {
    assert_eq!(
        required(BY_COMPARISON, &[("mode", text("review"))]),
        names(&["rules"])
    );
    assert_eq!(
        required(BY_COMPARISON, &[("mode", text("explain"))]),
        names(&["examples"])
    );
}

#[test]
fn an_unbound_comparison_against_text_closes_to_string_and_keeps_both_arms() {
    let inputs = inputs_at(BY_COMPARISON, &[], Opt::Full);
    let mode = inputs
        .iter()
        .find(|input| input.name == "mode")
        .unwrap_or_else(|| panic!("`$mode` is required: {inputs:?}"));
    assert_eq!(mode.ty, "String");
    assert_eq!(
        required(BY_COMPARISON, &[]),
        names(&["examples", "mode", "rules"])
    );
}

#[test]
fn binding_an_integer_decides_a_comparison() {
    assert_eq!(
        required(BY_SIZE, &[("n", Literal::Int(5))]),
        names(&["many"])
    );
    assert_eq!(required(BY_SIZE, &[("n", Literal::Int(1))]), names(&["few"]));
}

#[test]
fn a_name_both_arms_read_stays_required_under_any_binding() {
    assert_eq!(required(READ_IN_BOTH_ARMS, &[]), names(&["mode", "who"]));
    assert_eq!(
        required(READ_IN_BOTH_ARMS, &[("mode", text("review"))]),
        names(&["who"])
    );
    assert_eq!(
        required(READ_IN_BOTH_ARMS, &[("mode", text("anything else"))]),
        names(&["who"])
    );
}

#[test]
fn both_optimization_levels_give_one_required_set() {
    let cases = [
        Case {
            source: BY_MATCH,
            bound: vec![],
        },
        Case {
            source: BY_MATCH,
            bound: vec![("mode", text("review"))],
        },
        Case {
            source: BY_COMPARISON,
            bound: vec![("mode", text("explain"))],
        },
        Case {
            source: BY_SIZE,
            bound: vec![("n", Literal::Int(5))],
        },
    ];
    for case in cases {
        assert_eq!(
            inputs_at(case.source, &case.bound, Opt::None),
            inputs_at(case.source, &case.bound, Opt::Full),
            "in:\n{}",
            case.source
        );
    }
}

#[test]
fn the_dispatch_and_the_arm_it_decided_against_are_gone() {
    let interner = Interner::new();
    let compiled =
        compile_template_bound(&interner, BY_MATCH, &[("mode", text("review"))], Opt::Full)
            .unwrap_or_else(|e| panic!("the template compiles: {e}"));
    assert!(
        !compiled.ir.contains("switch"),
        "the dispatch stands:\n{}",
        compiled.ir
    );
    assert!(
        !compiled.ir.contains("Explain the code below"),
        "the arm decided against stands:\n{}",
        compiled.ir
    );
}

#[test]
fn a_decided_branch_leaves_no_test_behind() {
    let interner = Interner::new();
    let compiled = compile_template_bound(&interner, BY_SIZE, &[("n", Literal::Int(5))], Opt::Full)
        .unwrap_or_else(|e| panic!("the template compiles: {e}"));
    assert!(
        !compiled.ir.contains("if r"),
        "the branch stands:\n{}",
        compiled.ir
    );
    assert!(
        !compiled.ir.contains("\"few"),
        "the text of the arm decided against stands:\n{}",
        compiled.ir
    );
}
