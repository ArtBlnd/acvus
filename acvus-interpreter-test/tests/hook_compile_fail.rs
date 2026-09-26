//! What a hook's closure may not do: keep what a call lends it past the
//! call (RFC-0101 rule 2, RFC-0097 rule 1). Each `.stderr` is the error a
//! host author reads.
#[test]
fn a_closure_keeping_a_lent_argument_does_not_compile() {
    trybuild::TestCases::new().compile_fail("tests/hook_compile_fail/*.rs");
}
