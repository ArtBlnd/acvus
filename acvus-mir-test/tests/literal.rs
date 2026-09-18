//! What a literal's own type states and refuses (RFC-0058). The values the
//! literals produce are stated where they arrive, in
//! `acvus-interpreter-test/tests/literal.rs`.

use acvus_mir_test::compile_script_optimized;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn listing(source: &str) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &FxHashMap::default()).unwrap_or_else(|e| panic!("{e}"))
}

/// The type the listing records for the register the tail returns.
fn returned_ty(source: &str) -> String {
    let ir = listing(source);
    let returned = ir
        .lines()
        .rev()
        .find_map(|line| line.split_once("|   ")?.1.strip_prefix("return "))
        .unwrap_or_else(|| panic!("{source} has no return:\n{ir}"))
        .to_string();
    let register = match returned.split_once(" (") {
        Some((_, named)) => named.trim_end_matches(')').to_string(),
        None => returned,
    };
    ir.lines()
        .find_map(|line| {
            let (at, ty) = line.trim().strip_prefix("; ")?.split_once(" : ")?;
            let (slot, _) = at.split_once(' ')?;
            (slot == register).then(|| ty.to_string())
        })
        .unwrap_or_else(|| panic!("{source}: no type for {register}:\n{ir}"))
}

fn refusal(source: &str) -> String {
    let i = Interner::new();
    let found: Vec<String> = compile_script_optimized(&i, source, &FxHashMap::default())
        .expect_err(&format!("{source} is refused"))
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line).to_string())
        .collect();
    assert_eq!(found.len(), 1, "{source}: {found:?}");
    found.into_iter().next().expect("one refusal")
}

// -- Rule 1: a suffixed integer literal has that width -----------------

/// The width is the literal's, not the use's: `1u64` is one `u64` constant
/// with no cast, where before RFC-0058 a `u64` one took `plen / plen` or
/// `1 as u64`.
#[test]
fn a_suffixed_literal_is_one_constant_of_its_width() {
    let ir = listing("let one = 1u64; one");
    assert!(ir.contains("return 1 (r0)"), "{ir}");
    assert!(!ir.contains("cast"), "{ir}");
    assert_eq!(returned_ty("let one = 1u64; one"), "u64");
}

#[test]
fn every_width_is_a_suffix() {
    for width in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
        let source = format!("let x = 7{width}; x");
        assert_eq!(returned_ty(&source), width, "{source}");
    }
}

/// RFC-0037 checks an unsuffixed literal against the width its use decided;
/// a suffixed one carries the width, so the same message names it at the
/// literal with nothing left to decide.
#[test]
fn a_literal_that_does_not_fit_its_suffix_is_refused() {
    assert_eq!(refusal("300u8"), "literal 300 does not fit u8");
    assert_eq!(refusal("128i8"), "literal 128 does not fit i8");
    assert_eq!(refusal("0u8 - 1u8; 256u8"), "literal 256 does not fit u8");
}

// -- Rule 3: `b"…"` is the array literal's form ------------------------

#[test]
fn a_byte_string_is_an_array_of_u8() {
    let ir = listing("let s = b\"GET\"; s[0u64]");
    assert!(ir.contains("T0 = [71, 69, 84]"), "{ir}");
    assert!(ir.contains(": Array<u8, 3>"), "{ir}");
    assert_eq!(returned_ty("let s = b\"GET\"; s[0u64]"), "u8");
}

/// `b""` is an `Array<u8, 0>` and not an array of an undecided element:
/// the literal's type is the literal's, with nothing for a use to settle.
#[test]
fn an_empty_byte_string_is_still_an_array_of_u8() {
    let ir = listing("let s = b\"\"; 0");
    assert!(ir.contains("T0 = []"), "{ir}");
    assert!(ir.contains(": Array<u8, 0>"), "{ir}");
}

// -- Rule 5: the fold reads a suffixed constant at its width -----------

/// `200u8 + 100u8` wraps at `u8` (RFC-0037's widths, RFC-0055's fold), and
/// `200u8.wrapping_add(100)` in Rust is 44.
#[test]
fn a_suffixed_constant_folds_at_its_width() {
    let ir = listing("200u8 + 100u8");
    assert!(ir.contains("return 44 (r0)"), "{ir}");
    assert_eq!(returned_ty("200u8 + 100u8"), "u8");
}

#[test]
fn a_char_constant_folds_through_a_cast() {
    let ir = listing("'\\u{1F600}' as u32");
    assert!(ir.contains("return 128512 (r0)"), "{ir}");
    let ir = listing("65u8 as char as u32");
    assert!(ir.contains("return 65 (r0)"), "{ir}");
}

// -- Rule 2: `char` is a type -----------------------------------------

#[test]
fn a_char_is_its_own_type_and_compares() {
    let ir = listing("let c = 'x'; if c == 'x' { 1 } else { 0 }");
    assert!(ir.contains("r0 = 'x' (r1) == 'x' (r2)"), "{ir}");
    assert!(ir.contains("; r1 (v1) : char"), "{ir}");
}

#[test]
fn a_char_is_not_an_integer() {
    assert_eq!(refusal("'a' + 'b'"), "type mismatch in `+`: char vs char");
}
