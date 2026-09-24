//! What a literal of its own type is worth at the contract it arrives at
//! (RFC-0058). What the compiler refuses is stated in
//! `acvus-mir-test/tests/literal.rs`.
//!
//! Cross-artifact obligation: every expected `char` value here is the Rust
//! `char` written beside it, and every expected byte the Rust byte
//! literal. RFC-0058 rules that the language's spelling and Rust's name
//! the same value.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn at(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script(&i, source, FxHashMap::default(), ret).await
}

async fn int_at(source: &str, k: IntTy) -> i128 {
    k.read(at(source, Ty::Int(k)).await.bits())
}

async fn char_at_(source: &str) -> char {
    let code = at(source, Ty::Char).await.as_char();
    char::from_u32(code).expect("a Char register holds a scalar value")
}

async fn text(source: &str) -> String {
    let v = at(source, Ty::String).await;
    // SAFETY: every caller's script ends in a String-typed expression.
    unsafe { v.as_str() }.to_owned()
}

// -- Rule 1: a suffixed integer literal has that width -----------------

#[tokio::test]
async fn a_suffixed_literal_is_its_width_at_run_time() {
    assert_eq!(int_at("1u64", IntTy::U64).await, 1);
    assert_eq!(int_at("255u8", IntTy::U8).await, i128::from(255u8));
    assert_eq!(int_at("1u8 - 1u8", IntTy::U8).await, 0);
    assert_eq!(
        int_at("(0i8 - 127i8) - 1i8", IntTy::I8).await,
        i128::from(i8::MIN)
    );
}

/// A width is where the program's arithmetic stops: past it, the run traps
/// (RFC-0037 rule 3).
#[tokio::test]
#[should_panic(expected = "attempt to subtract with overflow")]
async fn a_suffixed_literal_s_width_is_where_subtraction_traps() {
    int_at("(0i8 - 127i8) - 2i8", IntTy::I8).await;
}

/// A minus that touches an integer literal is part of it, so each width's
/// minimum is a literal and arrives as the value Rust's is.
#[tokio::test]
async fn a_signed_literal_is_its_widths_minimum() {
    assert_eq!(int_at("-128i8", IntTy::I8).await, i128::from(i8::MIN));
    assert_eq!(
        int_at("-9223372036854775808", IntTy::I64).await,
        i128::from(i64::MIN)
    );
    assert_eq!(
        int_at("-128i8 + 1i8", IntTy::I8).await,
        i128::from(i8::MIN + 1)
    );
    assert_eq!(int_at("0 - 1", IntTy::I64).await, -1);
}

/// A `"…"` decodes against the same table the other literals use, so the
/// string that arrives holds the bytes the escapes name.
#[tokio::test]
async fn a_string_literal_arrives_decoded() {
    assert_eq!(text("\"\\x41\".to_string()").await, "A");
    assert_eq!(text("\"a\\tb\".to_string()").await, "a\tb");
    assert_eq!(text("\"\\u{1F600}\".to_string()").await, "\u{1F600}");
    assert_eq!(text("\"back\\\\slash\".to_string()").await, "back\\slash");
}

/// `a[i]` takes a `u64` (RFC-0047), and `1u64` is how one is written.
#[tokio::test]
async fn a_suffixed_literal_indexes_without_a_cast() {
    assert_eq!(int_at("let a = [7, 8, 9]; a[1u64]", IntTy::I64).await, 8);
}

// -- Rule 2: `char` is a type -----------------------------------------

#[tokio::test]
async fn a_char_literal_is_one_scalar_value() {
    assert_eq!(char_at_("'x'").await, 'x');
    assert_eq!(char_at_("'\\n'").await, '\n');
    assert_eq!(char_at_("'\\''").await, '\'');
    assert_eq!(char_at_("'é'").await, 'é');
    assert_eq!(char_at_("'\\u{1F600}'").await, '\u{1F600}');
}

#[tokio::test]
async fn a_char_compares_by_its_scalar_value() {
    assert_eq!(
        text("let c = 'x'; if c == 'x' { \"y\".to_string() } else { \"n\".to_string() }").await,
        "y"
    );
    assert_eq!(
        text("let c = 'x'; if c == 'y' { \"y\".to_string() } else { \"n\".to_string() }").await,
        "n"
    );
    assert_eq!(
        text("let c = 'a'; if c < 'b' { \"y\".to_string() } else { \"n\".to_string() }").await,
        "y"
    );
    assert_eq!(
        text("let c = '\\u{1F600}'; if c < 'b' { \"y\".to_string() } else { \"n\".to_string() }")
            .await,
        "n"
    );
}

/// A `char` matches as a literal pattern, at the word its scalar value is.
#[tokio::test]
async fn a_char_is_a_pattern() {
    let src = |c: char| format!("let c = '{c}'; match c {{ 'a' => 1, 'b' => 2, _ => 3 }}");
    assert_eq!(int_at(&src('a'), IntTy::I64).await, 1);
    assert_eq!(int_at(&src('b'), IntTy::I64).await, 2);
    assert_eq!(int_at(&src('z'), IntTy::I64).await, 3);
}

/// Rust's `as` at either end of a `char`: `char as T` is the `u32` cast and
/// `u8 as char` is total.
#[tokio::test]
async fn a_char_casts_as_rust_casts() {
    assert_eq!(
        int_at("'\\u{1F600}' as u32", IntTy::U32).await,
        i128::from('\u{1F600}' as u32)
    );
    assert_eq!(
        int_at("'A' as i64", IntTy::I64).await,
        i128::from('A' as i64)
    );
    assert_eq!(
        int_at("'\\u{100}' as u8", IntTy::U8).await,
        i128::from('\u{100}' as u8)
    );
    assert_eq!(char_at_("65u8 as char").await, 65u8 as char);
    assert_eq!(char_at_("255u8 as char").await, 255u8 as char);
}

/// A cast whose source is not a constant is the machine's own, not the
/// fold's: the value arrives through a register either way.
#[tokio::test]
async fn a_char_cast_of_a_register_is_the_same_value() {
    assert_eq!(
        int_at(
            "let s = \"A\"; let c = char_at(&s, 0); c as u32",
            IntTy::U32
        )
        .await,
        i128::from('A' as u32)
    );
    assert_eq!(
        char_at_("let n = 0u8; let b = n + 66u8; b as char").await,
        66u8 as char
    );
}

// -- Rule 3: `b"…"` is an `Array<u8, N>` -------------------------------

#[tokio::test]
async fn a_byte_string_is_the_bytes_it_spells() {
    assert_eq!(
        int_at("let s = b\"GET\"; s[0u64]", IntTy::U8).await,
        i128::from(b'G')
    );
    assert_eq!(
        int_at("let s = b\"GET\"; s[2u64]", IntTy::U8).await,
        i128::from(b'T')
    );
    assert_eq!(
        int_at("let s = b\"\\xFF\"; s[0u64]", IntTy::U8).await,
        i128::from(0xFFu8)
    );
    assert_eq!(
        int_at("let s = b\"a\\nb\"; s[1u64]", IntTy::U8).await,
        i128::from(b'\n')
    );
}

#[tokio::test]
async fn a_byte_literal_is_a_u8() {
    assert_eq!(int_at("b'G'", IntTy::U8).await, i128::from(b'G'));
    assert_eq!(int_at("b'\\xFF'", IntTy::U8).await, i128::from(0xFFu8));
    assert_eq!(
        text(
            "let s = b\"GET\"; if s[0u64] == b'G' { \"y\".to_string() } else { \"n\".to_string() }"
        )
        .await,
        "y"
    );
}

// -- Rule 4: the stdlib's char functions are typed with `char` ----------

#[tokio::test]
async fn the_string_boundary_yields_chars() {
    assert_eq!(
        char_at_("let s = \"héllo\"; char_at(&s, 1)").await,
        "héllo".chars().nth(1).expect("héllo has a second char")
    );
    assert_eq!(
        int_at(
            "chars(\"héllo\") | fold(0, |acc, c| -> acc + c as i64)",
            IntTy::I64
        )
        .await,
        "héllo"
            .chars()
            .map(|c| i64::from(u32::from(c)))
            .sum::<i64>() as i128
    );
}

/// `int_to_char` stays where `as` does not reach: Rust admits only
/// `u8 as char`, so a code point above U+00FF arrives through a check.
///
/// Each arm answers with a value the other cannot produce — a scalar value
/// is non-negative and the refused code point comes back negated — so the
/// assertion tells `Ok` from `Err` and not merely one word from another.
#[tokio::test]
async fn int_to_char_reaches_what_as_does_not() {
    let answer = |n: i64| {
        format!(
            "let out = 0 - {n}; \
             if let Ok(c) = int_to_char({n}) {{ out = c as i64; }}; \
             out"
        )
    };
    assert_eq!(
        int_at(&answer(128512), IntTy::I64).await,
        i128::from('\u{1F600}' as i64)
    );
    assert_eq!(int_at(&answer(55296), IntTy::I64).await, -55296);
    assert_eq!(int_at(&answer(-1), IntTy::I64).await, 1);
}
