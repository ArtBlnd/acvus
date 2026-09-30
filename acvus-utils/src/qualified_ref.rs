use crate::{Astr, Namespace};

/// A name and the namespace it stands in, the name a program or the host
/// gives a declaration or a context.
///
/// - `QualifiedRef::root(name)` -> unqualified (root namespace)
/// - `QualifiedRef::qualified(ns, name)` -> specific namespace
///
/// The order is the namespace's, then the name's; both are ids in the order
/// their interner first interned them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QualifiedRef {
    pub namespace: Namespace,
    pub name: Astr,
}

impl QualifiedRef {
    pub fn root(name: Astr) -> Self {
        Self {
            namespace: Namespace::ROOT,
            name,
        }
    }

    pub fn qualified(namespace: Namespace, name: Astr) -> Self {
        Self { namespace, name }
    }
}
