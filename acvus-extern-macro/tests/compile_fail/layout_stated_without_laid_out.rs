//! A type states it is laid out only through `LaidOut`, an `unsafe trait`
//! (RFC-0096 rule 2, RFC-0080 rule 1): a `Layout` of a type that does not
//! implement it cannot be built, so no safe `TyArg` impl states one.
use acvus_extern::Layout;

struct Mine;

const MINE: Layout<Mine> = Layout::laid_out();

fn main() {
    let _ = MINE;
}
