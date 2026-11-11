//! The scopes classes belong to, all nested in one root scope.
//!
//! A scope is a body where it is recorded. A body is finished where nothing more
//! is to come to it: the root, which nothing is passed to, and a body where it is
//! used, which has what was passed. A body where it is written is frozen: it is
//! settled as far as what is recorded goes and kept so, as each use brings more.

use std::collections::BTreeSet;

use super::ScopeId;
use super::id::Arena;

#[derive(Clone, Debug)]
pub struct Scopes {
    /// Each scope's enclosing scope; the root has none.
    parents: Arena<ScopeId, Option<ScopeId>>,
    /// The scopes made frozen. What lies in one is frozen with it.
    frozen: BTreeSet<ScopeId>,
    root: ScopeId,
}

impl Default for Scopes {
    fn default() -> Self {
        Self::new()
    }
}

impl Scopes {
    pub fn new() -> Self {
        let mut parents = Arena::default();
        let root = parents.push(None);
        Self {
            parents,
            frozen: BTreeSet::new(),
            root,
        }
    }

    pub fn root(&self) -> ScopeId {
        self.root
    }

    /// A scope within `parent`, of a body nothing more is to come to: one where
    /// it is used.
    pub fn child(&mut self, parent: ScopeId) -> ScopeId {
        self.parents.push(Some(parent))
    }

    /// A scope within `parent`, of a body where it is written: it is frozen.
    pub fn frozen_child(&mut self, parent: ScopeId) -> ScopeId {
        let scope = self.child(parent);
        self.frozen.insert(scope);
        scope
    }

    /// Whether more is still to come to the body `scope` is of: it is frozen, or
    /// lies in a frozen one. A body used within a frozen body is finished only
    /// where that body is.
    pub fn is_frozen(&self, scope: ScopeId) -> bool {
        let mut scope = Some(scope);
        while let Some(current) = scope {
            if self.frozen.contains(&current) {
                return true;
            }
            scope = self.parent(current);
        }
        false
    }

    /// The scope `scope` lies in; the root lies in none.
    pub fn parent(&self, scope: ScopeId) -> Option<ScopeId> {
        *self.parents.get(scope)
    }

    /// The innermost scope that holds both: of two scopes one of which lies
    /// in the other, the outer.
    pub fn outer(&self, a: ScopeId, b: ScopeId) -> ScopeId {
        let mut holding_a = BTreeSet::new();
        let mut scope = Some(a);
        while let Some(current) = scope {
            holding_a.insert(current);
            scope = *self.parents.get(current);
        }
        let mut scope = b;
        while !holding_a.contains(&scope) {
            scope = self.parents.get(scope).expect("every scope lies in the root scope");
        }
        scope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn of_two_nested_scopes_the_outer_is_the_enclosing_one() {
        let mut scopes = Scopes::new();
        let body = scopes.child(scopes.root());
        let lambda = scopes.child(body);
        assert_eq!(scopes.outer(lambda, body), body);
        assert_eq!(scopes.outer(body, lambda), body);
        assert_eq!(scopes.outer(lambda, lambda), lambda);
    }

    #[test]
    fn a_scope_is_frozen_where_it_is_made_so_or_lies_in_one_that_is() {
        let mut scopes = Scopes::new();
        let used = scopes.child(scopes.root());
        let written = scopes.frozen_child(used);
        let used_within = scopes.child(written);
        assert!(!scopes.is_frozen(scopes.root()));
        assert!(!scopes.is_frozen(used));
        assert!(scopes.is_frozen(written));
        assert!(scopes.is_frozen(used_within));
    }

    #[test]
    fn of_two_sibling_scopes_the_outer_is_the_one_holding_both() {
        let mut scopes = Scopes::new();
        let body = scopes.child(scopes.root());
        let first = scopes.child(body);
        let second = scopes.child(body);
        assert_eq!(scopes.outer(first, second), body);
    }
}
