//! A derived type's payload is proved `UniformPayload` (RFC-0076): a payload
//! whose field types the marker reaches builds with no attribute, and one
//! that holds another crate's generic type builds with
//! `unsafe(uniform_payload)`. The refusals are in `compile_fail`.
#[test]
fn a_payload_the_marker_reaches_builds() {
    trybuild::TestCases::new().pass("tests/uniform_pass/*.rs");
}
