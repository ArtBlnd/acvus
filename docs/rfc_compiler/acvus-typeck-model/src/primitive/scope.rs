//! The scopes classes belong to, all nested in one root scope.

use std::collections::BTreeSet;

use super::ScopeId;
use super::id::Arena;

#[derive(Clone, Debug)]
pub struct Scopes {
    /// Each scope's enclosing scope; the root has none.
    parents: Arena<ScopeId, Option<ScopeId>>,
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
        Self { parents, root }
    }

    pub fn root(&self) -> ScopeId {
        self.root
    }

    pub fn child(&mut self, parent: ScopeId) -> ScopeId {
        self.parents.push(Some(parent))
    }

    /// The scope `scope` lies in; the root lies in none.
    pub fn parent(&self, scope: ScopeId) -> Option<ScopeId> {
        *self.parents.get(scope)
    }

    /// Whether `scope` is `of`, or lies in it at any depth.
    pub fn lies_within(&self, scope: ScopeId, of: ScopeId) -> bool {
        let mut scope = Some(scope);
        while let Some(current) = scope {
            if current == of {
                return true;
            }
            scope = *self.parents.get(current);
        }
        false
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
    fn a_scope_lies_within_itself_and_what_holds_it_and_not_within_what_it_holds() {
        let mut scopes = Scopes::new();
        let body = scopes.child(scopes.root());
        let lambda = scopes.child(body);
        let other = scopes.child(scopes.root());
        assert!(scopes.lies_within(lambda, lambda));
        assert!(scopes.lies_within(lambda, body));
        assert!(scopes.lies_within(lambda, scopes.root()));
        assert!(!scopes.lies_within(body, lambda));
        assert!(!scopes.lies_within(lambda, other));
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
