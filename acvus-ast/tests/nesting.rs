//! RFC-0106 rule 1: every parse refuses a source nested past `NESTING_MAX`
//! as `NestingTooDeep`, at the construct that passes the bound, and accepts
//! one nested exactly to it, for each kind of level `NESTING_MAX` names.

use acvus_ast::error::ParseErrorKind;
use acvus_ast::{NESTING_MAX, ParseError, Span, parse_expr, parse_script, parse_template};
use acvus_utils::Interner;

const MAX: usize = NESTING_MAX as usize;

fn too_deep() -> ParseErrorKind {
    ParseErrorKind::NestingTooDeep { max: NESTING_MAX }
}

fn script_ok(src: &str) {
    let interner = Interner::new();
    if let Err(recovered) = parse_script(&interner, src) {
        panic!("refused: {:?}", recovered.errors);
    }
}

fn script_refused(src: &str) -> ParseError {
    let interner = Interner::new();
    let recovered = parse_script(&interner, src).expect_err("nested past the bound");
    let [error] = recovered.errors.as_slice() else {
        panic!("more than the one refusal: {:?}", recovered.errors);
    };
    assert_eq!(error.kind, too_deep());
    error.clone()
}

fn template_ok(src: &str) {
    let interner = Interner::new();
    if let Err(recovered) = parse_template(&interner, src) {
        panic!("refused: {:?}", recovered.errors);
    }
}

fn template_refused(src: &str) -> ParseError {
    let interner = Interner::new();
    let recovered = parse_template(&interner, src).expect_err("nested past the bound");
    let [error] = recovered.errors.as_slice() else {
        panic!("more than the one refusal: {:?}", recovered.errors);
    };
    assert_eq!(error.kind, too_deep());
    error.clone()
}

fn whole(src: &str) -> Span {
    Span::new(0, src.len())
}

fn parens(levels: usize) -> String {
    format!("{}1{}", "(".repeat(levels - 1), ")".repeat(levels - 1))
}

fn format_string(parts: usize) -> String {
    let pieces: String = ["{{ 1 }}", "x"].into_iter().cycle().take(parts).collect();
    format!("\"{pieces}\"")
}

#[test]
fn parenthesized_expressions_nest_one_level_each() {
    script_ok(&parens(MAX));
    let src = parens(MAX + 1);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn a_chain_of_binary_operators_nests_one_level_per_operator() {
    let chain = |levels: usize| format!("1{}", " + 1".repeat(levels - 1));
    script_ok(&chain(MAX));
    let src = chain(MAX + 1);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn a_format_string_nests_one_level_per_part() {
    script_ok(&format_string(MAX));
    script_refused(&format_string(MAX + 1));
}

#[test]
fn blocks_nest_one_level_each() {
    let blocks = |levels: usize| format!("{}1{}", "{ ".repeat(levels - 1), " }".repeat(levels - 1));
    script_ok(&blocks(MAX));
    let src = blocks(MAX + 1);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn statements_nest_one_level_each() {
    let loops = |levels: usize| {
        format!("{}{}", "while false { ".repeat(levels - 1), "}".repeat(levels - 1))
    };
    script_ok(&loops(MAX));
    let src = loops(MAX + 1);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn lambdas_nest_one_level_each() {
    let lambdas = |levels: usize| format!("{}1", "|x| -> ".repeat(levels - 1));
    script_ok(&lambdas(MAX));
    let src = lambdas(MAX + 1);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn patterns_nest_one_level_each_below_their_match() {
    let matched = |pattern_levels: usize| {
        format!(
            "match 1 {{ {}x{} => 1, _ => 0 }}",
            "Some(".repeat(pattern_levels - 1),
            ")".repeat(pattern_levels - 1)
        )
    };
    script_ok(&matched(MAX - 1));
    let src = matched(MAX);
    assert_eq!(script_refused(&src).span, whole(&src));
}

#[test]
fn a_for_section_nests_one_level_over_its_lines() {
    let fors = |sections: usize| {
        format!("{}x\n{}", "% for x in $xs\n".repeat(sections), "% end\n".repeat(sections))
    };
    template_ok(&fors(MAX - 2));
    template_refused(&fors(MAX - 1));
}

#[test]
fn an_if_section_nests_two_levels_over_its_lines() {
    let ifs = |sections: usize| {
        format!("{}x\n{}", "% if $c\n".repeat(sections), "% end\n".repeat(sections))
    };
    template_ok(&ifs((MAX - 2) / 2));
    template_refused(&ifs((MAX - 2) / 2 + 1));
}

#[test]
fn a_tag_nests_one_level_over_its_expression() {
    let tag = |levels: usize| format!("{{{{ {} }}}}", parens(levels));
    template_ok(&tag(MAX - 1));
    template_refused(&tag(MAX));
}

#[test]
fn a_bare_expression_is_bounded_as_well() {
    let interner = Interner::new();
    parse_expr(&interner, &parens(MAX)).expect("at the bound");
    let src = parens(MAX + 1);
    let error = parse_expr(&interner, &src).expect_err("past the bound");
    assert_eq!(error.kind, too_deep());
    assert_eq!(error.span, whole(&src));
}

#[test]
fn the_refusal_is_the_innermost_construct_past_the_bound() {
    let src = parens(MAX + 4);
    let error = script_refused(&src);
    assert_eq!(error.span, Span::new(3, src.len() - 3));
}

#[test]
fn the_message_names_the_bound() {
    let message = too_deep().to_string();
    assert!(message.contains(&NESTING_MAX.to_string()), "{message}");
}

#[test]
fn a_far_deeper_source_is_refused_without_building_its_tree() {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            let levels = 100_000;
            script_refused(&parens(levels));
            script_refused(&format!("{}1", "-".repeat(levels)));
            script_refused(&format!("1{}", " + 1".repeat(levels)));
            template_refused(&format!(
                "{}x\n{}",
                "% for x in $xs\n".repeat(levels),
                "% end\n".repeat(levels)
            ));
        })
        .expect("the thread starts")
        .join()
        .expect("every parse is refused without overflowing");
}
