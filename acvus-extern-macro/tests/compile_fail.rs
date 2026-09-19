//! What a declaration may not say. Each case is a refusal RFC-0059 carries
//! in a bound, and its `.stderr` is the message a reader gets.
#[test]
fn a_declaration_the_bounds_refuse() {
    trybuild::TestCases::new().compile_fail("tests/compile_fail/*.rs");
}
