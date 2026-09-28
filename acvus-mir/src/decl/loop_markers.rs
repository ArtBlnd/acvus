//! Declarations the loop analysis reads. Not yet in rfc_compiler: they are
//! settled together with the loop dependences they serve, and carried here
//! unchanged until then.

use acvus_utils::QualifiedRef;

/// The keying markers a map or a set carries as its last type argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyingMarkers {
    /// A table whose keys meet by an equivalence law.
    pub equiv: QualifiedRef,
    /// A table keyed by closures, whose keys meet by no law.
    pub opaque: QualifiedRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoopMarkers {
    pub keying: Option<KeyingMarkers>,
    /// The shared signature a pull loop steps through.
    pub pull: Option<QualifiedRef>,
}
