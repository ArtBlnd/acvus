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
    /// `Some` only for a function a script declares with `fn` (RFC-0100):
    /// no name a program writes carries one, so no lookup by name reaches
    /// such a function, and a call reaches it only where the script's own
    /// lift resolved the call to it.
    pub scope: Option<FnScope>,
}

/// Which script declares a `fn`, and which instance of it a function is:
/// each call site from outside the function's component has one instance
/// of that component (RFC-0100 rule 3). The declaring script is the
/// function's namespace with `script` as its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FnScope {
    pub script: Astr,
    pub instance: u32,
}

impl QualifiedRef {
    pub fn root(name: Astr) -> Self {
        Self {
            namespace: None,
            name,
            scope: None,
        }
    }

    pub fn qualified(namespace: Astr, name: Astr) -> Self {
        Self {
            namespace: Some(namespace),
            name,
            scope: None,
        }
    }

    pub fn written_in(self) -> Self {
        self.declaring_script().unwrap_or(self)
    }

    pub fn declaring_script(self) -> Option<Self> {
        let scope = self.scope?;
        Some(Self {
            namespace: self.namespace,
            name: scope.script,
            scope: None,
        })
    }
}
