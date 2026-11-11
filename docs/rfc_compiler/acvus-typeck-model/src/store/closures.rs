//! Closures: each a lambda and the classes that stand where it captures, and
//! which closure each copy of a template is a copy of.
//!
//! A function value holds closures by name alone, so nothing is compared inside
//! one. What a name stands for is kept here, and only added to: a closure is
//! named where its lambda is defined, in the text or in a copy of the template
//! that defines it, and never changed.

use std::collections::BTreeMap;

use crate::primitive::class::ClassKind;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::{ClosureId, LambdaId, ScopeId};

/// A lambda over the classes it captures, one for each capture its template
/// writes, in that order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Closure<K>
where
    K: ClassKind,
{
    pub lambda: LambdaId,
    pub captures: Vec<K::Id>,
}

#[derive(Clone, Debug)]
pub struct Closures<K>
where
    K: ClassKind,
{
    closures: Arena<ClosureId, Closure<K>>,
    /// The scope of each copy of a closure's template, with the closure.
    copies: BTreeMap<ScopeId, ClosureId>,
}

impl<K> Default for Closures<K>
where
    K: ClassKind,
{
    fn default() -> Self {
        Self {
            closures: Arena::default(),
            copies: BTreeMap::new(),
        }
    }
}

impl<K> Closures<K>
where
    K: ClassKind,
{
    /// Names a closure anew: `lambda` over `captures`.
    pub fn name(&mut self, lambda: LambdaId, captures: Vec<K::Id>) -> ClosureId {
        self.closures.push(Closure { lambda, captures })
    }

    pub fn get(&self, closure: ClosureId) -> &Closure<K> {
        self.closures.get(closure)
    }

    /// Every closure named, in the order they were named.
    pub fn iter(&self) -> impl Iterator<Item = (ClosureId, &Closure<K>)> {
        self.closures.iter()
    }

    /// Says that the classes of `scope` are a copy of the template of `closure`.
    pub fn copied(&mut self, scope: ScopeId, closure: ClosureId) {
        let before = self.copies.insert(scope, closure);
        debug_assert_eq!(before, None, "a scope is one copy");
    }

    /// The copy of `closure` that `scope` is or lies in: the innermost such.
    pub fn within(&self, scopes: &Scopes, scope: ScopeId, closure: ClosureId) -> Option<ScopeId> {
        let mut scope = Some(scope);
        while let Some(current) = scope {
            if self.copies.get(&current) == Some(&closure) {
                return Some(current);
            }
            scope = scopes.parent(current);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::class::Live;
    use crate::primitive::id::Arena;

    fn two_closures() -> (Closures<Live>, ClosureId, ClosureId) {
        let mut lambdas = Arena::<LambdaId, ()>::default();
        let lambda = lambdas.push(());
        let mut closures = Closures::<Live>::default();
        let (first, second) = (closures.name(lambda, Vec::new()), closures.name(lambda, Vec::new()));
        (closures, first, second)
    }

    #[test]
    fn two_closures_of_one_lambda_are_two() {
        let (closures, first, second) = two_closures();
        assert_ne!(first, second);
        assert_eq!(closures.get(first), closures.get(second));
    }

    #[test]
    fn a_scope_lies_within_the_copy_it_is_and_the_copies_that_hold_it() {
        let (mut closures, first, second) = two_closures();
        let mut scopes = Scopes::new();
        let outer = scopes.child(scopes.root());
        let inner = scopes.child(outer);
        closures.copied(outer, first);
        closures.copied(inner, second);
        assert_eq!(closures.within(&scopes, inner, second), Some(inner));
        assert_eq!(closures.within(&scopes, inner, first), Some(outer));
        assert_eq!(closures.within(&scopes, outer, second), None);
        assert_eq!(closures.within(&scopes, scopes.root(), first), None);
    }

    #[test]
    fn of_two_copies_of_one_closure_the_innermost_is_found() {
        let (mut closures, first, _) = two_closures();
        let mut scopes = Scopes::new();
        let outer = scopes.child(scopes.root());
        let inner = scopes.child(outer);
        closures.copied(outer, first);
        closures.copied(inner, first);
        assert_eq!(closures.within(&scopes, inner, first), Some(inner));
    }
}
