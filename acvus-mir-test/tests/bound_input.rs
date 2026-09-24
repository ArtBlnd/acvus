//! A bound `$` is a constant, and what it makes unreachable is `!`
//! (RFC-0071 rule 5).

use std::collections::BTreeMap;

use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{BindingRefused, Bindings, BoundValue, QualifiedRef};
use acvus_mir::ty::{Infer, IntTy, ObjectTy};
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
    bound: Vec<(&'static str, BoundValue)>,
}

fn text(held: &str) -> BoundValue {
    BoundValue::String(held.to_string())
}

fn inputs_at(source: &str, bound: &[(&str, BoundValue)], opt: Opt) -> Vec<ShownInput> {
    let interner = Interner::new();
    let compiled = compile_template_bound(&interner, source, bound, opt)
        .unwrap_or_else(|e| panic!("the template compiles: {e}"));
    let mut inputs = compiled.inputs;
    inputs.sort();
    inputs
}

fn required(source: &str, bound: &[(&str, BoundValue)]) -> Vec<String> {
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
            shown("mode", "String"),
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
        required(BY_SIZE, &[("n", BoundValue::Int(5))]),
        names(&["many"])
    );
    assert_eq!(
        required(BY_SIZE, &[("n", BoundValue::Int(1))]),
        names(&["few"])
    );
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
            bound: vec![("n", BoundValue::Int(5))],
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
    let compiled = compile_template_bound(&interner, BY_SIZE, &[("n", BoundValue::Int(5))], Opt::Full)
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

// -- A bound `$` holds any value a literal writes (RFC-0087) -------------

const BY_ENUM: &str = "\
% match $mode
% Mode::Review =>
Review.
% Mode::Explain =>
Explain.
% end
";

const BY_PAYLOAD: &str = "\
% match $who
% Who::Named(n) =>
Hello, {{ n }}.
% Who::Anonymous =>
Hello, stranger.
% end
";

const READS_TWO_FIELDS: &str = "{{ $user.name }} is {{ $user.role }}.\n";

const READS_A_THIRD_FIELD: &str = "{{ $user.name }} writes to {{ $user.email }}.\n";

/// The branches of an `if` join as they are checked, so this use fixes the
/// payload before the value's variant joins it. A pattern on `$who` would
/// not: a pattern on a head still open is settled by the solve, after the
/// join, and refuses the pattern rather than the value.
const PAYLOAD_AS_AN_INTEGER: &str = "\
% let w = if $anonymous { Who::Named(0) } else { $who }
x
";

const BOTH: [Opt; 2] = [Opt::None, Opt::Full];

fn variant(interner: &Interner, name: &str, tag: &str, payload: Option<BoundValue>) -> BoundValue {
    BoundValue::Variant {
        name: QualifiedRef::root(interner.intern(name)),
        tag: interner.intern(tag),
        payload: payload.map(Box::new),
    }
}

fn object(interner: &Interner, fields: &[(&str, BoundValue)]) -> BoundValue {
    BoundValue::Object(
        fields
            .iter()
            .map(|(key, value)| (interner.intern(key), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    )
}

fn compiles(interner: &Interner, source: &str, bound: &[(&str, BoundValue)], opt: Opt) -> String {
    compile_template_bound(interner, source, bound, opt)
        .unwrap_or_else(|e| panic!("the template compiles at {opt:?}: {e}\nin:\n{source}"))
        .ir
}

fn refused(interner: &Interner, source: &str, bound: &[(&str, BoundValue)], opt: Opt) -> String {
    match compile_template_bound(interner, source, bound, opt) {
        Ok(compiled) => panic!("the template compiled at {opt:?}:\n{}", compiled.ir),
        Err(e) => e,
    }
}

#[test]
fn a_bound_variant_decides_its_dispatch_without_refusing_the_other_arm() {
    for opt in BOTH {
        let interner = Interner::new();
        compiles(&interner, BY_ENUM, &[], opt);
        let mode = variant(&interner, "Mode", "Review", None);
        let ir = compiles(&interner, BY_ENUM, &[("mode", mode)], opt);
        assert!(
            ir.contains("Review.") && !ir.contains("Explain."),
            "the dispatch is not decided at {opt:?}:\n{ir}"
        );
    }
}

#[test]
fn a_bound_variant_carries_its_payload() {
    for opt in BOTH {
        let interner = Interner::new();
        compiles(&interner, BY_PAYLOAD, &[], opt);
        let who = variant(&interner, "Who", "Named", Some(text("Ada")));
        compiles(&interner, BY_PAYLOAD, &[("who", who)], opt);
    }
}

#[test]
fn a_bound_object_is_read_by_field() {
    for opt in BOTH {
        let interner = Interner::new();
        compiles(&interner, READS_TWO_FIELDS, &[], opt);
        let user = object(&interner, &[("name", text("Ada")), ("role", text("admin"))]);
        compiles(&interner, READS_TWO_FIELDS, &[("user", user)], opt);
    }
}

#[test]
fn a_field_the_bound_object_lacks_is_refused_at_its_constant() {
    for opt in BOTH {
        let interner = Interner::new();
        compiles(&interner, READS_A_THIRD_FIELD, &[], opt);
        let user = object(&interner, &[("name", text("Ada")), ("role", text("admin"))]);
        let error = refused(&interner, READS_A_THIRD_FIELD, &[("user", user)], opt);
        assert!(
            error.contains("`$user` is bound to {name: \"Ada\", role: \"admin\",}, which is not a value of"),
            "{error}"
        );
    }
}

#[test]
fn a_payload_the_uses_type_otherwise_is_refused_naming_the_input() {
    for opt in BOTH {
        let interner = Interner::new();
        compiles(&interner, PAYLOAD_AS_AN_INTEGER, &[], opt);
        let who = variant(&interner, "Who", "Named", Some(text("Ada")));
        let error = refused(&interner, PAYLOAD_AS_AN_INTEGER, &[("who", who)], opt);
        assert!(
            error.contains("`$who` is bound to Who::Named(\"Ada\"), which is not a value of"),
            "{error}"
        );
    }
}

#[test]
fn every_literal_form_binds_where_the_template_compiles_unbound() {
    let interner = Interner::new();
    let int = |n: i128| BoundValue::Int(n);
    let cases: Vec<(&str, &str, BoundValue)> = vec![
        (
            "tuple",
            "% match $pair\n% (a, b) =>\n{{ a }}\n% if b > 1\nmany\n% end\n% _ =>\nnone\n% end\n",
            BoundValue::Tuple(vec![text("a"), int(2)]),
        ),
        (
            "some",
            "% match $maybe\n% Some(x) =>\n{{ x }}\n% None =>\nnone\n% end\n",
            BoundValue::Option(Some(Box::new(text("a")))),
        ),
        (
            "none",
            "% match $maybe\n% Some(x) =>\n{{ x }}\n% None =>\nnone\n% end\n",
            BoundValue::Option(None),
        ),
        (
            "ok",
            "% match $done\n% Ok(x) =>\n{{ x }}\n% Err(e) =>\n{{ e }}\n% end\n",
            BoundValue::Result(Ok(Box::new(text("a")))),
        ),
        (
            "err",
            "% match $done\n% Ok(x) =>\n{{ x }}\n% Err(e) =>\n{{ e }}\n% end\n",
            BoundValue::Result(Err(Box::new(text("e")))),
        ),
        (
            "char",
            "% if $c == 'x'\nx\n% end\n",
            BoundValue::Char('x'),
        ),
        (
            "bytes",
            "% if $b == b\"hi\"\nhi\n% end\n",
            BoundValue::Bytes(b"hi".to_vec()),
        ),
        (
            "suffixed",
            "% if $n == 3u8\nthree\n% end\n",
            BoundValue::IntOf {
                value: 3,
                width: IntTy::U8,
            },
        ),
    ];
    for (form, source, value) in cases {
        for opt in BOTH {
            let name = source
                .split('$')
                .nth(1)
                .and_then(|rest| rest.split(|c: char| !c.is_alphanumeric()).next())
                .expect("each case reads one input");
            compile_template_bound(&interner, source, &[], opt)
                .unwrap_or_else(|e| panic!("{form} compiles unbound at {opt:?}: {e}"));
            compile_template_bound(&interner, source, &[(name, value.clone())], opt)
                .unwrap_or_else(|e| panic!("{form} compiles bound at {opt:?}: {e}"));
        }
    }
}

#[test]
fn a_value_with_no_type_is_refused_where_it_is_bound() {
    let interner = Interner::new();
    let name = interner.intern("x");
    let bind = |value: BoundValue| Bindings::default().bind(name, value);

    assert_eq!(
        bind(BoundValue::Array(vec![BoundValue::Int(1), text("a")])),
        Err(BindingRefused::ElementsDisagree)
    );
    assert_eq!(
        bind(BoundValue::Array(vec![
            variant(&interner, "M", "A", Some(BoundValue::Int(1))),
            variant(&interner, "M", "A", Some(text("x"))),
        ])),
        Err(BindingRefused::ElementsDisagree)
    );
    let too_wide = (0..=ObjectTy::<Infer>::MAX_FIELDS)
        .map(|at| (interner.intern(&format!("f{at}")), BoundValue::Unit))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        bind(BoundValue::Object(too_wide)),
        Err(BindingRefused::ObjectTooWide {
            fields: ObjectTy::<Infer>::MAX_FIELDS + 1
        })
    );
    assert_eq!(
        bind(BoundValue::IntOf {
            value: 300,
            width: IntTy::U8
        }),
        Err(BindingRefused::IntOutOfRange {
            value: 300,
            width: IntTy::U8
        })
    );
    assert_eq!(
        bind(BoundValue::Array(vec![
            variant(&interner, "M", "A", None),
            variant(&interner, "M", "B", None),
        ])),
        Ok(())
    );
}

/// A `for`, an index and a list pattern each need the array's head or
/// length while the body is checked, so no read of an array compiles
/// unbound (RFC-0087 Why).
#[test]
fn a_bound_array_is_traversed() {
    let source = "% for n in $names\n- {{ n }}\n% end\n";
    for opt in BOTH {
        let interner = Interner::new();
        let names = BoundValue::Array(vec![text("a"), text("b")]);
        let ir = compiles(&interner, source, &[("names", names)], opt);
        assert!(ir.contains("list ["), "no array constant at {opt:?}:\n{ir}");
    }
}

// -- A dispatch on a known structured value folds (RFC-0071 rule 5) ------

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

const ARM_BY_OPTION: &str = "\
% match $x
% Some(v) =>
{{ $a }}
% None =>
{{ $b }}
% end
";

const ARM_BY_SCRIPT_LITERAL: &str = "\
% match M::R
% M::R =>
{{ $a }}
% _ =>
{{ $b }}
% end
";

const ARM_BY_A_SLOT_WRITTEN_TWICE: &str = "\
% let m = Mode::Review
% if $c
% m = Mode::Explain
% end
% match m
% Mode::Review =>
{{ $a }}
% Mode::Explain =>
{{ $b }}
% end
";

const ARM_BY_A_SLOT_LENT_MUTABLY: &str = "\
% let m = Mode::Review
% let r = &mut m
% if $c
% *r = Mode::Explain
% end
% match m
% Mode::Review =>
{{ $a }}
% Mode::Explain =>
{{ $b }}
% end
";

fn required_at_both(interner: &Interner, source: &str, bound: &[(&str, BoundValue)]) -> Vec<String> {
    let [none, full] = BOTH.map(|opt| {
        let mut inputs = compile_template_bound(interner, source, bound, opt)
            .unwrap_or_else(|e| panic!("the template compiles at {opt:?}: {e}\nin:\n{source}"))
            .inputs;
        inputs.sort();
        inputs
    });
    assert_eq!(none, full, "the levels disagree in:\n{source}");
    full.into_iter().map(|input| input.name).collect()
}

fn nested(interner: &Interner, k: BoundValue, first: i128) -> BoundValue {
    object(
        interner,
        &[
            ("k", k),
            ("t", BoundValue::Tuple(vec![BoundValue::Int(first), text("x")])),
        ],
    )
}

fn some(held: BoundValue) -> BoundValue {
    BoundValue::Option(Some(Box::new(held)))
}

#[test]
fn a_bound_variant_leaves_the_arm_it_chose() {
    let interner = Interner::new();
    let review = variant(&interner, "Mode", "Review", None);
    let explain = variant(&interner, "Mode", "Explain", None);
    assert_eq!(
        required_at_both(&interner, ARM_BY_VARIANT, &[("mode", review.clone())]),
        names(&["rules"])
    );
    assert_eq!(
        required_at_both(&interner, ARM_BY_VARIANT, &[("mode", explain)]),
        names(&["examples"])
    );
    assert_eq!(
        required_at_both(&interner, ARM_BY_VARIANT, &[]),
        names(&["examples", "mode", "rules"])
    );
    let ir = compiles(&interner, ARM_BY_VARIANT, &[("mode", review)], Opt::Full);
    assert!(!ir.contains("switch"), "the dispatch stands:\n{ir}");
}

#[test]
fn a_bound_object_decides_a_nested_pattern() {
    let interner = Interner::new();
    let r3 = || some(variant(&interner, "M", "R", Some(BoundValue::Int(3))));
    let matching = nested(&interner, r3(), 1);
    let no_payload = nested(&interner, BoundValue::Option(None), 1);
    let other_element = nested(&interner, r3(), 2);
    assert_eq!(
        required_at_both(&interner, ARM_BY_NESTED_PATTERN, &[("o", matching)]),
        names(&["a"])
    );
    assert_eq!(
        required_at_both(&interner, ARM_BY_NESTED_PATTERN, &[("o", no_payload)]),
        names(&["b"])
    );
    assert_eq!(
        required_at_both(&interner, ARM_BY_NESTED_PATTERN, &[("o", other_element)]),
        names(&["b"])
    );
}

#[test]
fn a_bound_option_leaves_the_arm_it_chose() {
    let interner = Interner::new();
    assert_eq!(
        required_at_both(&interner, ARM_BY_OPTION, &[("x", some(text("q")))]),
        names(&["a"])
    );
    assert_eq!(
        required_at_both(&interner, ARM_BY_OPTION, &[("x", BoundValue::Option(None))]),
        names(&["b"])
    );
}

#[test]
fn a_dispatch_on_a_script_literal_leaves_the_arm_it_chose() {
    let interner = Interner::new();
    assert_eq!(
        required_at_both(&interner, ARM_BY_SCRIPT_LITERAL, &[]),
        names(&["a"])
    );
}

#[test]
fn a_slot_written_on_one_path_decides_nothing() {
    let interner = Interner::new();
    assert_eq!(
        required_at_both(&interner, ARM_BY_A_SLOT_WRITTEN_TWICE, &[]),
        names(&["a", "b", "c"])
    );
}

#[test]
fn a_slot_lent_mutably_decides_nothing() {
    let interner = Interner::new();
    assert_eq!(
        required_at_both(&interner, ARM_BY_A_SLOT_LENT_MUTABLY, &[]),
        names(&["a", "b", "c"])
    );
}
