//! Sequence structural operation builtins (origin-preserving).
//!
//! Async sequence consumers (next_seq, etc.) remain in the interpreter.

use std::marker::PhantomData;

use acvus_mir::builtins::BuiltinId;

use super::handler::{sync, BuiltinExecute};
use super::types::{E, Iter, O, Seq, T};

// ── Implementations ────────────────────────────────────────────────

fn take_seq(
    seq: Seq<T<0>, O<0>, E<0>>,
    n: i64,
) -> Seq<T<0>, O<0>, E<0>> {
    Seq(seq.0.take(n.max(0) as usize), PhantomData)
}

fn skip_seq(
    seq: Seq<T<0>, O<0>, E<0>>,
    n: i64,
) -> Seq<T<0>, O<0>, E<0>> {
    Seq(seq.0.skip(n.max(0) as usize), PhantomData)
}

fn chain_seq(
    seq: Seq<T<0>, O<0>, E<0>>,
    rhs: Iter<T<0>, E<0>>,
) -> Seq<T<0>, O<0>, E<0>> {
    Seq(seq.0.chain(rhs.0), PhantomData)
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::TakeSeq,  sync(take_seq)),
        (BuiltinId::SkipSeq,  sync(skip_seq)),
        (BuiltinId::ChainSeq, sync(chain_seq)),
    ]
}
