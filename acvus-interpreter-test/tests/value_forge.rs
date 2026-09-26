//! A `Value` of a kind that is not inline is made only by the runtime, from
//! an allocation or a place it holds, never from bits (RFC-0102 rule 2,
//! `Large`). Each program here would put a word of its choosing under a
//! kind a safe reader or `Release` follows, and none compiles, even with
//! the `tooling` feature this crate turns on.

#[test]
fn no_program_outside_the_runtime_makes_a_value_of_a_large_kind_from_bits() {
    trybuild::TestCases::new().compile_fail("tests/value_forge/*.rs");
}
