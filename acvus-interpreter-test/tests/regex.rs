use acvus_extern::Registry;
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(acvus_ext::regex_registry());
    regs
}

async fn run(i: &Interner, source: &str, ret: Ty) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), registries(), ret)
        .await
        .value
}

async fn text_of(i: &Interner, source: &str) -> String {
    let v = run(i, source, Ty::String).await;
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    unsafe { v.as_str() }.to_owned()
}

async fn int_of(i: &Interner, source: &str) -> i64 {
    run(i, source, Ty::I64).await.as_int()
}

#[tokio::test]
async fn one_compiled_regex_serves_three_searches() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             let one = "a1";
             let two = "bb";
             let three = "c3";
             let a = is_match(&re, &one);
             let b = is_match(&re, &two);
             let c = is_match(&re, &three);
             a.to_string() + " " + b.to_string() + " " + c.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true false true");
}

#[tokio::test]
async fn the_method_form_reads_the_regex_as_the_receiver() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             let t = "a1b2";
             let cut = re.replace_all(&t, "~");
             let n = re.find_all(&t).count();
             cut + " " + n.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "a~b~ 2");
}

/// `string::find`, `iter::find` and `std::find` share one bare name, and the
/// receiver type is what separates them. The prefix `regex_find` is gone
/// because this resolves.
#[tokio::test]
async fn find_resolves_on_a_string_and_on_a_regex() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"let t = "price is 42 dollars";
           if let Ok(re) = regex("[0-9]+") {
             let byte = if let Some(m) = find(&re, &t) { m.start as i64 } else { -1 };
             let chars = unwrap_or(find(&t, "42"), -1);
             byte.to_string() + " " + chars.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "9 9", "one text of only ASCII, so both units agree");
}

#[tokio::test]
async fn a_match_carries_its_byte_span() {
    let i = Interner::new();
    let text = "héllo 42 wörld";
    let start = text.find("42").expect("the text holds 42");
    let s = text_of(
        &i,
        &format!(
            r#"if let Ok(re) = regex("[0-9]+") {{
                 let t = "{text}";
                 if let Some(m) = find(&re, &t) {{
                   m.start.to_string() + " " + m.end.to_string() + " " + m.text
                 }} else {{ "no match" }}
               }} else {{ "?" }}"#
        ),
    )
    .await;
    assert_eq!(
        s,
        format!("{start} {} 42", start + 2),
        "Rust's own byte offset of the same needle"
    );
    assert_eq!(s, "7 9 42", "the two-byte e moves a byte offset off 6");
}

#[tokio::test]
async fn find_at_starts_where_it_is_told() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             let t = "a1b22";
             let first = if let Some(m) = find_at(&re, &t, 0u64) { m.text } else { "-" };
             let next = if let Some(m) = find_at(&re, &t, 2u64) { m.text } else { "-" };
             let past = if let Some(m) = find_at(&re, &t, 99u64) { m.text } else { "-" };
             first + " " + next + " " + past
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "1 22 -");
}

#[tokio::test]
async fn find_all_yields_every_match() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             let t = "a1b22c333";
             find_all(&re, &t) | map(|m| -> m.text) | join(",")
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "1,22,333");
}

#[tokio::test]
async fn shortest_match_reports_a_byte_end() {
    let i = Interner::new();
    let n = int_of(
        &i,
        r#"if let Ok(re) = regex("a+") {
             let t = "xaaa";
             if let Some(end) = shortest_match(&re, &t) { end as i64 } else { -1 }
           } else { -2 }"#,
    )
    .await;
    assert_eq!(n, 2, "the leftmost match starts at 1 and one `a` suffices");
}

#[tokio::test]
async fn captures_reach_a_group_by_index_and_by_name() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("(?<key>[a-z]+)=([0-9]+)") {
             let t = "width=64";
             if let Some(c) = captures(&re, &t) {
               let whole = if let Some(m) = group(&c, 0u64) { m.text } else { "-" };
               let value = if let Some(m) = group(&c, 2u64) { m.text } else { "-" };
               let key = if let Some(m) = named(&c, "key") { m.text } else { "-" };
               let absent = if let Some(m) = named(&c, "nope") { m.text } else { "-" };
               whole + " " + key + " " + value + " " + absent
             } else { "no match" }
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "width=64 width 64 -");
}

#[tokio::test]
async fn a_group_that_did_not_participate_is_none() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("(a)|(b)") {
             let t = "b";
             if let Some(c) = captures(&re, &t) {
               let first = if let Some(m) = group(&c, 1u64) { m.text } else { "-" };
               let second = if let Some(m) = group(&c, 2u64) { m.text } else { "-" };
               first + " " + second
             } else { "no match" }
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "- b");
}

/// The rewrite of the deleted `regex_extract`: group 1 of every match.
#[tokio::test]
async fn group_one_of_every_match_is_captures_all_and_map() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("k=([0-9]+)") {
             let t = "k=1 x k=22 k=333";
             captures_all(&re, &t)
               | map(|c| -> if let Some(m) = group(&c, 1u64) { m.text } else { "" })
               | join(",")
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "1,22,333");
}

#[tokio::test]
async fn a_pattern_names_its_groups_and_counts_them() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("(?<key>[a-z]+)=([0-9]+)") {
             let n = group_count(&re);
             let names = group_names(&re) | into_iter | map(|g| -> unwrap_or(g, "-")) | join(",");
             n.to_string() + " " + names
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "2 -,key,-");
}

#[tokio::test]
async fn replace_expands_a_numbered_group() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("([a-z]+)=([0-9]+)") {
             let t = "w=1 h=2";
             let all = re.replace_all(&t, "$2:$1");
             let one = re.replace(&t, "$2:$1");
             all + " | " + one
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "1:w 2:h | 1:w h=2");
}

#[tokio::test]
async fn replace_n_replaces_the_first_n_and_nothing_at_zero() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]") {
             let t = "1 2 3";
             let none = re.replace_n(&t, 0u64, "~");
             let two = re.replace_n(&t, 2u64, "~");
             let more = re.replace_n(&t, 9u64, "~");
             none + " | " + two + " | " + more
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "1 2 3 | ~ ~ 3 | ~ ~ ~");
}

#[tokio::test]
async fn replace_with_calls_the_closure_per_match() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             let t = "a1b22c";
             replace_with(&re, &t, |m| -> "<" + m.text + "@" + m.start.to_string() + ">")
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "a<1@1>b<22@3>c");
}

#[tokio::test]
async fn split_cuts_on_every_match() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("[,;]\\s*") {
             let t = "a, b;c";
             split(&re, &t) | join("|")
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "a|b|c");
}

#[tokio::test]
async fn split_n_keeps_the_rest_in_the_last_piece() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex(",") {
             let t = "a,b,c";
             split_n(&re, &t, 2u64) | join("|")
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "a|b,c");
}

#[tokio::test]
async fn case_insensitive_changes_a_result() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"let flags = { case_insensitive: true, multi_line: false, dot_matches_new_line: false,
                        ignore_whitespace: false, unicode: true, swap_greed: false, };
           if let Ok(re) = regex_with("abc", flags) {
             if let Ok(plain) = regex("abc") {
               let t = "ABC";
               let loose = is_match(&re, &t);
               let strict = is_match(&plain, &t);
               loose.to_string() + " " + strict.to_string()
             } else { "?" }
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true false");
}

/// A derived object crosses field by field, and the checker does not refuse
/// an object literal that is missing one: two object types unify by union,
/// so the shortfall surfaces in the crossing instead. That is why
/// `regex_flags()` exists.
#[tokio::test]
#[should_panic(expected = "object field `multi_line` is missing")]
async fn a_flags_literal_missing_a_field_does_not_reach_the_builder() {
    let i = Interner::new();
    text_of(
        &i,
        r#"let partial = { case_insensitive: true, };
           if let Ok(re) = regex_with("abc", partial) {
             let t = "ABC";
             let hit = is_match(&re, &t);
             hit.to_string()
           } else { "?" }"#,
    )
    .await;
}

/// `regex_flags()` is the flag set `regex` itself compiles with, so the two
/// constructors agree when nothing is changed.
#[tokio::test]
async fn regex_flags_is_what_regex_compiles_with() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex_with("A.c", regex_flags()) {
             if let Ok(plain) = regex("A.c") {
               let upper = "AbcA";
               let lower = "abc";
               let a = is_match(&re, &upper);
               let b = is_match(&re, &lower);
               let c = is_match(&plain, &upper);
               a.to_string() + " " + b.to_string() + " " + c.to_string()
             } else { "?" }
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true false true");
}

#[tokio::test]
async fn multi_line_and_dot_match_the_switches_they_name() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"let both = { case_insensitive: false, multi_line: true, dot_matches_new_line: true,
                       ignore_whitespace: false, unicode: true, swap_greed: false, };
           if let Ok(re) = regex_with("^b.$", both) {
             let t = "a\nb\n";
             let hit = is_match(&re, &t);
             hit.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true");
}

#[tokio::test]
async fn escape_makes_a_pattern_that_matches_itself() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex(escape("a.c")) {
             let dotted = "a.c";
             let other = "abc";
             let a = is_match(&re, &dotted);
             let b = is_match(&re, &other);
             a.to_string() + " " + b.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true false");
}

#[tokio::test]
async fn is_match_at_answers_from_a_byte_offset() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("b") {
             let t = "ab";
             let at_zero = is_match_at(&re, &t, 0u64);
             let at_one = is_match_at(&re, &t, 1u64);
             let past = is_match_at(&re, &t, 2u64);
             at_zero.to_string() + " " + at_one.to_string() + " " + past.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "true true false");
}

/// `^` is the start of the whole text, so an anchored pattern does not
/// match at a `start` past 0 however the text reads from there.
#[tokio::test]
async fn a_start_offset_does_not_move_the_anchor() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Ok(re) = regex("^b") {
             let t = "ab";
             let at_zero = is_match_at(&re, &t, 0u64);
             let at_one = is_match_at(&re, &t, 1u64);
             at_zero.to_string() + " " + at_one.to_string()
           } else { "?" }"#,
    )
    .await;
    assert_eq!(s, "false false");
}

#[tokio::test]
async fn a_pattern_that_is_not_a_regex_names_itself() {
    let i = Interner::new();
    let s = text_of(
        &i,
        r#"if let Err(RegexError::Invalid(e)) = regex("(") { e.pattern } else { "compiled" }"#,
    )
    .await;
    assert_eq!(s, "(");
}
