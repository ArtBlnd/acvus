//! `#[extern_fn]` reads its flows off the Rust signature (RFC-0079 rule 6):
//! the declaration each signature form builds.
#[test]
fn flows_read_off_a_signature() {
    trybuild::TestCases::new().pass("tests/flows_pass/*.rs");
}
