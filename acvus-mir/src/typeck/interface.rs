//! What a check sees of the units its calls reach (graph, programs): each
//! unit's interface, by the whole name the embedding gives it.

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use crate::infer::Infer;

/// What another unit sees of a unit: its scheme, generalized exactly as a
/// function value bound by `let` is (functions §2), with its parameters
/// the inputs by name. A unit is a function whose environment is empty, so
/// it and a let-bound function value are one generalization: its function
/// type over the variables it generalizes, and every decision its body
/// could not make, each a requirement made again at each call. Its types
/// are inferred within the unit alone, and a call only instantiates them
/// (programs §1, §2; types §1).
///
/// A check's terms live in its own store, and a unit is checked apart from
/// its callers, so the scheme is kept in a store of its own: the terms it
/// reaches, copied out of the unit's check when it is settled. A call
/// instantiates it by the one instantiation a let-bound scheme has, copying
/// from that store into the caller's. Two interfaces are equal when their
/// schemes are, variables compared by the order they are first reached.
pub struct Interface {
    /// The terms of the check the scheme was settled in.
    pub(crate) store: Infer,
    /// The parameter names, in the order of the function type's parameters.
    pub params: Box<[acvus_utils::Astr]>,
    /// The scheme in `store`: its function type, what it generalizes, and
    /// its requirements.
    pub(crate) scheme: super::generalize::Exported,
}

impl Interface {
    /// Whether `other` is this interface but for the names of its variables
    /// and the sites and spans its requirements carry: the same parameters,
    /// function type and requirements, variables matched by the order they
    /// are first reached. What a caller's check admits depends on nothing
    /// else; only a refusal reads a site (graph: early cutoff).
    pub fn same(&self, other: &Interface) -> bool {
        let mut table = crate::infer::CanonicalTable::new();
        let site = acvus_ast::AstId::alloc();
        self.params == other.params
            && self.scheme.canonical(&self.store, &mut table, site) == other.scheme.canonical(&other.store, &mut table, site)
    }
}

/// The interfaces a check's calls may reach, by each unit's whole name.
#[derive(Default)]
pub struct Interfaces {
    units: FxHashMap<QualifiedRef, Interface>,
}

impl Interfaces {
    pub fn new() -> Interfaces {
        Interfaces::default()
    }

    /// Sets the interface of the unit at `unit`, replacing any it had.
    pub fn set(&mut self, unit: QualifiedRef, interface: Interface) {
        self.units.insert(unit, interface);
    }

    pub fn get(&self, unit: QualifiedRef) -> Option<&Interface> {
        self.units.get(&unit)
    }

    pub fn remove(&mut self, unit: QualifiedRef) -> Option<Interface> {
        self.units.remove(&unit)
    }
}
