use crate::Astr;

/// A namespace-qualified reference. Used as the identity for contexts
/// and for qualified function/context access.
///
/// - `QualifiedRef::root(name)` -> unqualified (root namespace)
/// - `QualifiedRef::qualified(ns, name)` -> specific namespace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QualifiedRef {
    /// Namespace name. `None` = root.
    pub namespace: Option<Astr>,
    /// Context or function name.
    pub name: Astr,
    pub host: Option<Astr>,
}

impl QualifiedRef {
    pub fn root(name: Astr) -> Self {
        Self {
            namespace: None,
            name,
            host: None,
        }
    }

    pub fn qualified(namespace: Astr, name: Astr) -> Self {
        Self {
            namespace: Some(namespace),
            name,
            host: None,
        }
    }

    pub fn in_host(self, host: Option<Astr>) -> Self {
        Self { host, ..self }
    }
}
