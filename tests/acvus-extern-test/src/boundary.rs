//! The only module that writes `unsafe`: the facts `WordRuntime` signs, and
//! the casts between a word and what it holds.

#![allow(unsafe_code)]

use acvus_extern::{SignedS1, SignedS2, SignedS3};

use crate::runtime::WordRuntime;

// SAFETY: to be argued by the fill: an allocation a guard lends keeps its
// address and is written only under an exclusive guard until the loan ends.
unsafe impl SignedS1 for WordRuntime {}
// SAFETY: to be argued by the fill: a reference word names its target, or its
// written-back copy, until the loan ends.
unsafe impl SignedS2 for WordRuntime {}
// SAFETY: to be argued by the fill: an allocation is released only by the
// release of the one value that owns it, after every guard over it dropped.
unsafe impl SignedS3 for WordRuntime {}
