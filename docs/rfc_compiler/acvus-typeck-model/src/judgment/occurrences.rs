//! Occurrences: the head the decided classes an equivalence class passes into
//! give it, where nothing enters it.
//!
//! This layer imports no `State` and no other layer.

use super::Head;

/// The occurrences of a class give two different heads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Contexts;

/// The one head every occurrence in `occurrences` gives.
///
/// The caller guarantees that `occurrences` is not empty, and the function
/// panics otherwise. Returns `Err(Contexts)` where two occurrences give two
/// different heads. It reads no lower bound, requirement or domain.
pub fn context(occurrences: &[Head]) -> Result<Head, Contexts> {
    let Some((first, rest)) = occurrences.split_first() else {
        panic!("a context is taken of at least one occurrence");
    };
    if rest.iter().all(|head| head == first) {
        Ok(first.clone())
    } else {
        Err(Contexts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{Constructor, IntTy, Scalar};
    use std::collections::BTreeSet;

    fn scalar_head(scalar: Scalar) -> Head {
        Head::owned(Constructor::Scalar(scalar), BTreeSet::new())
    }

    #[test]
    fn unanimous_occurrences_give_their_head() {
        let i64_head = scalar_head(Scalar::Int(IntTy::I64));
        assert_eq!(context(&[i64_head.clone(), i64_head.clone()]), Ok(i64_head));
    }

    #[test]
    fn two_heads_are_contexts() {
        let i32_head = scalar_head(Scalar::Int(IntTy::I32));
        let i64_head = scalar_head(Scalar::Int(IntTy::I64));
        assert_eq!(context(&[i32_head, i64_head]), Err(Contexts));
    }
}
