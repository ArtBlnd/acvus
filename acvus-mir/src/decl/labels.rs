//! The lifetimes an extern fn's Rust signature names (ownership §5,
//! extern-fn §4). They exist only here: `verify` turns them into the flows
//! of the declared function type, and nothing after intake sees a lifetime.
//!
//! A *segment* of a declared type is one of: the position of a reference;
//! the position of a lifetime an extern-defined type declares; a function
//! value's one position (what its captures hold, ownership §5); a type
//! variable's part, which stands for every position of what instantiates
//! it, the k-th of the input reaching the k-th of the output. Segments are
//! read off the declared type itself, in order, so a call's instantiated
//! type has them by construction.

use crate::flows::FlowEnd;

/// A lifetime as the signature names it, by position among its lifetimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lifetime(pub u32);

/// What labels one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    Lifetime(Lifetime),
    /// A type variable's part, by the variable.
    TypeVar(u32),
}

/// The labels of one end of a call, one per segment of its declared type,
/// in segment order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndLabels {
    pub end: FlowEnd,
    pub labels: Box<[SegmentLabel]>,
}

/// `longer: shorter`: a loan labelled `longer` may flow where `shorter` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outlives {
    pub longer: Lifetime,
    pub shorter: Lifetime,
}

/// An extern fn's lifetimes: each end's labels, and their order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelledSignature {
    pub ends: Box<[EndLabels]>,
    pub outlives: Box<[Outlives]>,
}
