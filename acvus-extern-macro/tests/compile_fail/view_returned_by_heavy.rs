//! A view borrows the frame the call laid its arguments on, and that frame
//! is gone when an offloaded or awaited call resumes (RFC-0023 rule 6).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, heavy)]
fn head(s: &str) -> &str {
    &s[..1]
}

fn main() {}
