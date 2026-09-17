//! Reading a container through the per-type fns under one bare name
//! (RFC-0028, RFC-0043) and the String producers and fns of batch 1c,
//! each at the script contract: `is_empty` over Vec, Array, Deque and
//! String; `len` of a String in characters; `char_at`; the
//! producers `chars`, `lines`, `bytes`, `split_whitespace`; `rfind`,
//! `pad_start`, `pad_end`, `strip_prefix`, `strip_suffix`, `split_once`,
//! `eq_ignore_case`, `capitalize`. A trap surfaces here as a panic of the
//! harness carrying the trap's message. A script references a variable, not
//! a literal, so every `&` argument is bound first; the lexer has no `\r`
//! escape, so `lines` is exercised on `\n` alone.

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

async fn run(source: &str) -> Value {
    let i = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    run_script_mode_with_externs(&i, source, Context::default(), registries)
        .await
        .value
}

async fn string(source: &str) -> String {
    let v = run(source).await;
    // SAFETY: every caller's script ends in a String-typed expression.
    unsafe { v.as_str() }.to_owned()
}

// -- is_empty -------------------------------------------------------------------

#[tokio::test]
async fn is_empty_of_a_vec_is_true_only_with_no_element() {
    assert!(run("let xs = vec([]); is_empty(&xs)").await.as_bool());
    assert!(!run("let xs = vec([1, 2]); is_empty(&xs)").await.as_bool());
}

#[tokio::test]
async fn is_empty_of_an_array_is_false_with_an_element() {
    assert!(!run("let xs = [1, 2]; is_empty(&xs)").await.as_bool());
}

#[tokio::test]
async fn is_empty_of_a_deque_follows_its_pushes() {
    assert!(
        !run("let d = deque(); push_back(&mut d, 1); is_empty(&d)")
            .await
            .as_bool()
    );
    assert!(
        run("let d = deque(); push_back(&mut d, 1); pop_back(&mut d); is_empty(&d)")
            .await
            .as_bool()
    );
}

#[tokio::test]
async fn is_empty_of_a_string_is_true_only_for_the_empty_string() {
    assert!(run("let s = \"\"; is_empty(&s)").await.as_bool());
    assert!(!run("let s = \"a\"; is_empty(&s)").await.as_bool());
}

// -- string::len, char_at -------------------------------------------------------

#[tokio::test]
async fn len_of_a_string_counts_characters() {
    assert_eq!(run("let s = \"héllo\"; len(&s)").await.as_int(), 5);
    assert_eq!(run("let s = \"héllo\"; string::len(&s)").await.as_int(), 5);
}

#[tokio::test]
async fn char_at_reads_one_character_by_character_index() {
    assert_eq!(string("let s = \"héllo\"; char_at(&s, 1)").await, "é");
    assert_eq!(string("let s = \"ab\"; char_at(&s, 1)").await, "b");
}

#[tokio::test]
#[should_panic(expected = "char_at")]
async fn char_at_outside_the_string_traps() {
    run("let s = \"ab\"; char_at(&s, 5)").await;
}

// -- Producers ------------------------------------------------------------------

#[tokio::test]
async fn chars_yields_one_string_per_character() {
    assert_eq!(
        run("let cs = chars(\"ab\") | collect; len(&cs)")
            .await
            .as_int(),
        2
    );
    assert_eq!(run("chars(\"héllo\") | count()").await.as_int(), 5);
    assert_eq!(string("chars(\"héllo\") | join(\"-\")").await, "h-é-l-l-o");
}

#[tokio::test]
async fn lines_splits_on_newlines_and_drops_the_terminator() {
    assert_eq!(run("lines(\"a\\nb\\nc\") | count()").await.as_int(), 3);
    assert_eq!(
        string("lines(\"a\\nb\\nc\\n\") | join(\"|\")").await,
        "a|b|c"
    );
}

#[tokio::test]
async fn bytes_yields_the_utf8_bytes_as_ints() {
    assert_eq!(run("bytes(\"hé\") | count()").await.as_int(), 3);
    assert_eq!(run("bytes(\"hé\") | sum()").await.as_int(), 104 + 195 + 169);
}

#[tokio::test]
async fn split_whitespace_drops_every_run_of_whitespace() {
    assert_eq!(
        run("split_whitespace(\"  a  b c \") | count()")
            .await
            .as_int(),
        3
    );
    assert_eq!(
        string("split_whitespace(\"  a  b c \") | join(\",\")").await,
        "a,b,c"
    );
}

// -- Searching and shaping ------------------------------------------------------

#[tokio::test]
async fn rfind_gives_the_character_index_of_the_last_match() {
    assert_eq!(
        run("let s = \"héllo\"; rfind(&s, \"l\") | unwrap_or(-1)")
            .await
            .as_int(),
        3
    );
    assert_eq!(
        run("let s = \"abcabc\"; rfind(&s, \"bc\") | unwrap_or(-1)")
            .await
            .as_int(),
        4
    );
    assert_eq!(
        run("let s = \"abc\"; rfind(&s, \"z\") | unwrap_or(-1)")
            .await
            .as_int(),
        -1
    );
}

#[tokio::test]
async fn pad_start_fills_on_the_left_to_the_width_in_characters() {
    assert_eq!(string("pad_start(\"7\", 3, \"0\")").await, "007");
    assert_eq!(string("pad_start(\"é\", 4, \"ab\")").await, "abaé");
    assert_eq!(string("pad_start(\"abc\", 2, \"0\")").await, "abc");
    assert_eq!(string("pad_start(\"abc\", 5, \"\")").await, "abc");
}

#[tokio::test]
async fn pad_end_fills_on_the_right_to_the_width_in_characters() {
    assert_eq!(string("pad_end(\"ab\", 5, \"xy\")").await, "abxyx");
    assert_eq!(string("pad_end(\"abc\", -1, \"x\")").await, "abc");
}

#[tokio::test]
async fn strip_prefix_and_strip_suffix_are_none_without_the_pattern() {
    assert_eq!(
        string("strip_prefix(\"foobar\", \"foo\") | unwrap").await,
        "bar"
    );
    assert_eq!(
        string("strip_prefix(\"foobar\", \"bar\") | unwrap_or(\"none\")").await,
        "none"
    );
    assert_eq!(
        string("strip_suffix(\"foobar\", \"bar\") | unwrap").await,
        "foo"
    );
    assert_eq!(
        string("strip_suffix(\"foobar\", \"foo\") | unwrap_or(\"none\")").await,
        "none"
    );
}

#[tokio::test]
async fn split_once_gives_the_text_around_the_first_pattern() {
    let v =
        run("let p = split_once(\"a=b=c\", \"=\") | unwrap; concat(get(&p, 0), get(&p, 1))").await;
    assert_eq!(unsafe { v.as_str() }, "ab=c");
    assert_eq!(
        run("let p = split_once(\"a=b\", \"=\") | unwrap; len(&p)")
            .await
            .as_int(),
        2
    );
    assert_eq!(
        string("let p = split_once(\"ab\", \"=\") | unwrap_or(vec([\"x\", \"y\"])); concat(get(&p, 0), get(&p, 1))")
            .await,
        "xy"
    );
}

#[tokio::test]
async fn eq_ignore_case_compares_after_lowercasing() {
    assert!(
        run("let a = \"HeLLo\"; let b = \"hello\"; eq_ignore_case(&a, &b)")
            .await
            .as_bool()
    );
    assert!(
        run("let a = \"ÉCOLE\"; let b = \"école\"; eq_ignore_case(&a, &b)")
            .await
            .as_bool()
    );
    assert!(
        !run("let a = \"a\"; let b = \"b\"; eq_ignore_case(&a, &b)")
            .await
            .as_bool()
    );
}

#[tokio::test]
async fn capitalize_uppercases_the_first_character_only() {
    assert_eq!(string("capitalize(\"hello world\")").await, "Hello world");
    assert_eq!(string("capitalize(\"école\")").await, "École");
    assert_eq!(string("capitalize(\"\")").await, "");
}
