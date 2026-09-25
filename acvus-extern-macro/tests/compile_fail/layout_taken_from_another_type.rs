//! A statement is branded by its type (RFC-0096 rule 2): one type cannot
//! state another's layout by copying it.
use acvus_extern::{LaidOut, Layout, laid::ParamKind};

struct Theirs;

// SAFETY: `Theirs` is no type the macro reads; the test only builds its
// statement.
unsafe impl LaidOut for Theirs {
    const PARAMS: &'static [ParamKind] = &[];
}

struct Mine;

const MINE: Layout<Mine> = Layout::<Theirs>::laid_out();

fn main() {
    let _ = MINE;
}
