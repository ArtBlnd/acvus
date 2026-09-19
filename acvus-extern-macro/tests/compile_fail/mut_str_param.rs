//! There is no `&mut str` (RFC-0062 Decision 1).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn blank(s: &mut str) {
    s.make_ascii_uppercase();
}

fn main() {}
