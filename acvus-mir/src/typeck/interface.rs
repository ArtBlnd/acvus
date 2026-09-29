//! What a check sees of the units its calls reach (graph, programs): each
//! unit's interface, by the whole name the embedding gives it.

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use crate::decl::Scheme;
use crate::ty::{Poly, Types};

/// What another unit sees of a unit: its scheme, of the form of an extern
/// fn's declaration, its parameter names the inputs' names. Its types are
/// inferred within the unit alone, and a call only instantiates them
/// (programs §1, §2; types §1).
#[derive(Debug, Clone, PartialEq)]
pub struct Interface {
    pub scheme: Scheme,
}

/// The interfaces a check's calls may reach, by each unit's whole name,
/// and the one table their schemes' types are in.
#[derive(Default)]
pub struct Interfaces {
    types: Types<Poly>,
    units: FxHashMap<QualifiedRef, Interface>,
}

impl Interfaces {
    pub fn new() -> Interfaces {
        Interfaces::default()
    }

    pub fn types(&self) -> &Types<Poly> {
        &self.types
    }

    /// The table an interface's scheme is written in before it is set.
    pub fn types_mut(&mut self) -> &mut Types<Poly> {
        &mut self.types
    }

    /// Sets the interface of the unit at `unit`, replacing any it had.
    pub fn set(&mut self, unit: QualifiedRef, interface: Interface) {
        self.units.insert(unit, interface);
    }

    pub fn get(&self, unit: QualifiedRef) -> Option<&Interface> {
        self.units.get(&unit)
    }
}
