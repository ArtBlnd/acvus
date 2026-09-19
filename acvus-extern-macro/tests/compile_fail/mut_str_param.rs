//! There is no `&mut str` (RFC-0062 Decision 1), and a `&str` return waits
//! for the machine to take a pair-wide result at a call.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn blank(s: &mut str) {
    s.make_ascii_uppercase();
}

#[extern_fn(effect = pure)]
fn head(s: &str) -> &str {
    &s[..1]
}

fn main() {}
