//! `string::` and `char::` at the script contract: every function the two
//! modules offer, read the way a script reads it, at both optimization
//! levels. The twin each one names is a Rust `str` or `char` method, and
//! the value written beside each call is the value Rust's method gives for
//! the same input.
//!
//! The refusals at the end are the other half of the contract: a `&str`
//! parameter does not take an owned `String`, and a comparison operator on
//! text names the ordering functions instead of a copy.

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn compile_and_run(i: &Interner, main: &str, ret: Ty, opt: Opt) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        &[],
        &FxHashMap::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

/// The text a script yields at both optimization levels, which is one
/// number only when the level chose how hard the compiler works and not
/// what the program means.
fn text(main: &str) -> String {
    let i = Interner::new();
    let at = |opt| {
        let v = compile_and_run(&i, main, Ty::String, opt)
            .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
        assert!(v.is_string(), "expected a String, got {v:?}");
        // SAFETY: the witness is String.
        unsafe { v.as_str() }.to_owned()
    };
    let full = at(Opt::Full);
    assert_eq!(full, at(Opt::None), "the two levels read one program");
    full
}

fn refusal(main: &str, ret: Ty) -> String {
    let i = Interner::new();
    match compile_and_run(&i, main, ret, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

/// One script per group, joining each call's result with `|`, so a wrong
/// value names the call beside it in the expectation.
fn joined(exprs: &[&str]) -> String {
    let body: Vec<String> = exprs
        .iter()
        .enumerate()
        .map(|(k, e)| {
            format!("let r{k} = {e};\nlet v{k} = r{k}.to_string();\nout = out + \"|\" + v{k};")
        })
        .collect();
    text(&format!(
        "let out = \"\".to_string();\n{}\nout\n",
        body.join("\n")
    ))
}

// -- Reading ------------------------------------------------------------

#[test]
fn length_emptiness_and_the_ascii_questions() {
    assert_eq!(
        joined(&[
            r#"len("héllo")"#,
            r#"is_empty("")"#,
            r#"is_empty("a")"#,
            r#"is_ascii("abc")"#,
            r#"is_ascii("héllo")"#,
            r#"is_char_boundary("héllo", 1u64)"#,
            r#"is_char_boundary("héllo", 2u64)"#,
        ]),
        "|6|true|false|true|false|true|false"
    );
}

#[test]
fn case_folds_the_whole_unicode_and_the_ascii_half() {
    assert_eq!(
        joined(&[
            r#"upper("straße")"#,
            r#"lower("HÉLLO")"#,
            r#"to_ascii_uppercase("héllo")"#,
            r#"to_ascii_lowercase("HÉLLO")"#,
            r#"eq_ignore_ascii_case("Abc", "aBC")"#,
            r#"eq_ignore_ascii_case("É", "é")"#,
            r#"eq_ignore_case("É", "é")"#,
            r#"capitalize("abc")"#,
        ]),
        "|STRASSE|héllo|HéLLO|hÉllo|true|false|true|Abc"
    );
}

#[test]
fn the_trim_family_cuts_whitespace_and_a_pattern() {
    assert_eq!(
        joined(&[
            r#"trim("  ab  ")"#,
            r#"trim_start("  ab  ") + "!""#,
            r#"trim_end("  ab  ") + "!""#,
            r#"trim_matches("xxabxx", "x")"#,
            r#"trim_start_matches("xxabxx", "x") + "!""#,
            r#"trim_end_matches("xxabxx", "x") + "!""#,
            r#"trim_matches("abab", "ab")"#,
            r#"trim_matches("ab", "")"#,
        ]),
        "|ab|ab  !|  ab!|ab|abxx!|xxab!||ab"
    );
}

#[test]
fn the_prefix_and_suffix_questions_carry_rust_s_names() {
    assert_eq!(
        joined(&[
            r#"contains("hello", "ell")"#,
            r#"starts_with("hello", "he")"#,
            r#"ends_with("hello", "lo")"#,
            r#"starts_with("hello", "lo")"#,
            r#"strip_prefix("hello", "he").unwrap_or("-".to_string())"#,
            r#"strip_prefix("hello", "xx").unwrap_or("-".to_string())"#,
            r#"strip_suffix("hello", "lo").unwrap_or("-".to_string())"#,
        ]),
        "|true|true|true|false|llo|-|hel"
    );
}

#[test]
fn find_and_rfind_report_a_byte_offset_or_none() {
    assert_eq!(
        joined(&[
            r#"unwrap_or(find("abcabc", "b"), 0 - 1)"#,
            r#"unwrap_or(rfind("abcabc", "b"), 0 - 1)"#,
            r#"unwrap_or(find("abc", "z"), 0 - 1)"#,
            r#"unwrap_or(find("héllo", "l"), 0 - 1)"#,
        ]),
        "|1|4|-1|3"
    );
}

// -- Building -----------------------------------------------------------

#[test]
fn replace_replacen_and_repeat() {
    assert_eq!(
        joined(&[
            r#"concat("ab", "cd")"#,
            r#"replace("aaa", "a", "b")"#,
            r#"replacen("aaa", "a", "b", 2u64)"#,
            r#"replacen("aaa", "a", "b", 0u64)"#,
            r#"repeat("ab", 3u64)"#,
            r#"repeat("ab", 0u64)"#,
            r#"pad_start("7", 3, "0")"#,
            r#"pad_end("7", 3, "0")"#,
        ]),
        "|abcd|bbb|bba|aaa|ababab||007|700"
    );
}

#[test]
fn substring_cuts_by_byte_offset() {
    assert_eq!(
        joined(&[
            r#"substring("hello", 1u64, 3u64)"#,
            r#"substring("héllo", 0u64, 3u64)"#,
            r#"substring("hello", 2u64, 2u64) + "!""#,
        ]),
        "|el|hé|!"
    );
}

#[test]
fn bytes_cross_and_come_back() {
    assert_eq!(
        joined(&[
            r#"to_utf8(to_bytes("ab".to_string())).unwrap_or("-".to_string())"#,
            r#"to_utf8_lossy(to_bytes("héllo".to_string()))"#,
            r#"char_at("héllo", 1)"#,
        ]),
        "|ab|héllo|é"
    );
}

// -- Splitting ----------------------------------------------------------

#[test]
fn the_split_family_yields_an_iterator() {
    assert_eq!(
        joined(&[
            r#"split("a,b,c", ",") | join("-".to_string())"#,
            r#"rsplit("a,b,c", ",") | join("-".to_string())"#,
            r#"splitn("a,b,c", 2u64, ",") | join("-".to_string())"#,
            r#"rsplitn("a,b,c", 2u64, ",") | join("-".to_string())"#,
            r#"split("a,b,", ",") | join("-".to_string())"#,
            r#"split_terminator("a,b,", ",") | join("-".to_string())"#,
            r#"split_whitespace("  a  b ") | join("-".to_string())"#,
            r#"lines("a\nb") | join("-".to_string())"#,
        ]),
        "|a-b-c|c-b-a|a-b,c|c-a,b|a-b-|a-b|a-b|a-b"
    );
}

#[test]
fn matches_and_match_indices_report_every_hit() {
    assert_eq!(
        joined(&[
            r#"(matches("abcabc", "bc") | count)"#,
            r#"matches("abcabc", "bc") | join("-".to_string())"#,
            r#"match_indices("abcabc", "bc") | map(|m| -> m.index.to_string()) | join("-".to_string())"#,
            r#"match_indices("abcabc", "bc") | map(|m| -> m.text) | join("-".to_string())"#,
        ]),
        "|2|bc-bc|1-4|bc-bc"
    );
}

#[test]
fn chars_and_char_indices_count_scalar_values_and_bytes() {
    assert_eq!(
        joined(&[
            r#"(chars("héllo") | count)"#,
            r#"(bytes("héllo") | count)"#,
            r#"char_indices("héllo") | map(|c| -> c.index.to_string()) | join("-".to_string())"#,
            r#"char_indices("héllo") | map(|c| -> c.ch.to_string()) | join("-".to_string())"#,
            r#"rev_iter(chars("abc") | collect) | map(|c| -> c.to_string()) | join("".to_string())"#,
        ]),
        "|5|6|0-1-3-4-5|h-é-l-l-o|cba"
    );
}

#[test]
fn split_once_gives_the_two_sides_or_none() {
    assert_eq!(
        joined(&[
            r#"if let Some(p) = split_once("a=b", "=") { p[0u64].to_string() + "/" + p[1u64] } else { "-".to_string() }"#,
            r#"if let Some(p) = split_once("ab", "=") { p[0u64].to_string() } else { "-".to_string() }"#,
        ]),
        "|a/b|-"
    );
}

// -- Ordering -----------------------------------------------------------

#[test]
fn cmp_is_rust_s_ord_for_str() {
    assert_eq!(
        joined(&[
            r#"cmp("a", "b")"#,
            r#"cmp("b", "a")"#,
            r#"cmp("a", "a")"#,
            r#"cmp("ab", "a")"#,
            r#"cmp("Z", "a")"#,
        ]),
        "|-1|1|0|1|-1"
    );
    assert_eq!("Z".cmp("a"), std::cmp::Ordering::Less);
}

#[test]
fn lt_le_gt_ge_agree_with_cmp() {
    assert_eq!(
        joined(&[
            r#"lt("a", "b")"#,
            r#"lt("b", "a")"#,
            r#"le("a", "a")"#,
            r#"gt("b", "a")"#,
            r#"ge("a", "b")"#,
        ]),
        "|true|false|true|true|false"
    );
}

// -- char ---------------------------------------------------------------

#[test]
fn the_char_questions_are_rust_s() {
    assert_eq!(
        joined(&[
            r#"is_alphabetic('a')"#,
            r#"is_alphabetic('1')"#,
            r#"is_numeric('1')"#,
            r#"is_alphanumeric('_')"#,
            r#"is_whitespace(' ')"#,
            r#"is_uppercase('A')"#,
            r#"is_lowercase('A')"#,
            r#"is_ascii('é')"#,
            r#"is_ascii_digit('7')"#,
            r#"is_ascii_alphabetic('é')"#,
        ]),
        "|true|false|true|false|true|true|false|false|true|false"
    );
}

#[test]
fn the_char_conversions_are_rust_s() {
    assert_eq!(
        joined(&[
            r#"to_ascii_uppercase('a')"#,
            r#"to_ascii_lowercase('A')"#,
            r#"to_ascii_uppercase('é')"#,
            r#"unwrap_or(to_digit('7', 10u32), 99u32)"#,
            r#"unwrap_or(to_digit('f', 16u32), 99u32)"#,
            r#"unwrap_or(to_digit('f', 10u32), 99u32)"#,
            r#"if let Some(c) = char::from_digit(11u32, 16u32) { c.to_string() } else { "-".to_string() }"#,
            r#"if let Some(c) = char::from_digit(11u32, 10u32) { c.to_string() } else { "-".to_string() }"#,
            r#"len_utf8('a')"#,
            r#"len_utf8('é')"#,
        ]),
        "|A|a|é|7|15|99|b|-|1|2"
    );
}

// -- The refusals -------------------------------------------------------

/// A `&str` parameter takes the view a literal already is, so the copy of
/// RFC-0062 rule 3 is not on the path to a parse.
#[test]
fn a_view_parameter_refuses_an_owned_string() {
    let refused = refusal(r#"i64::from_str("42".to_string())"#, Ty::I64);
    assert!(
        refused.contains("expected &str, got String"),
        "the refusal names the view the parameter asks for, got {refused}"
    );
}

/// The `<` operator has no instance for text, and `.to_string()` does not
/// give it one: the refusal names the functions that do.
#[test]
fn a_comparison_on_text_names_the_ordering_functions() {
    for main in [
        r#"if "a".to_string() < "b".to_string() { 1 } else { 0 }"#,
        r#"if "a" < "b" { 1 } else { 0 }"#,
        r#"if "a" < "b".to_string() { 1 } else { 0 }"#,
        r#"if "a".to_string() >= "b" { 1 } else { 0 }"#,
    ] {
        let refused = refusal(main, Ty::I64);
        assert!(
            refused.contains("is not defined on") && refused.contains("use `string::cmp`"),
            "{main} was refused with {refused}"
        );
        assert!(
            !refused.contains("type mismatch"),
            "{main} was refused as a mismatch: {refused}"
        );
    }
}

#[test]
fn a_comparison_on_a_type_with_no_ordering_says_so_without_the_text_advice() {
    let refused = refusal("if true < false { 1 } else { 0 }", Ty::I64);
    assert!(
        refused.contains("`<` is not defined on Bool"),
        "expected the ordering refusal, got {refused}"
    );
    assert!(
        !refused.contains("string::cmp"),
        "a Bool is not text, so the text advice does not belong: {refused}"
    );
}

/// The numeric orderings the operator does have are untouched.
#[test]
fn the_operator_still_orders_numbers_and_chars() {
    assert_eq!(
        joined(&[
            r#"(if 1 < 2 { "y" } else { "n" })"#,
            r#"(if 2.5 >= 2.5 { "y" } else { "n" })"#,
            r#"(if 'a' < 'b' { "y" } else { "n" })"#,
        ]),
        "|y|y|y"
    );
}
