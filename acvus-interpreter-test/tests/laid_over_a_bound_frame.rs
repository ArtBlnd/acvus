//! A window stays bound to the body it last opened a frame for, and a second
//! call of that body skips `open_frame`. A call between them whose callee
//! binds nothing in the window still lays its arguments over the window's
//! first registers, so past the bound body's parameters it unbinds it
//! (RFC-0052 rules 5 and 7).

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

/// When this was written, at `Opt::Full` `reach` took six parameters and fit
/// the window above the script's frame, and `trimmed` took seven and was
/// rooted in a `Store` of its own. Its seventh argument, an `i64`, was laid
/// over a register `reach` had opened as a `Bool`, and the second `reach`
/// handed `filled` that register.
#[test]
fn a_rooted_call_between_two_calls_of_one_body_reopens_its_frame() {
    let source = include_str!("laid_over_a_bound_frame/reach_trimmed_reach.acvus");
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(
            corpus::attempt(source, opt, Stage::Run),
            Outcome::Value("50502".to_string()),
            "{opt:?}"
        );
    }
}
