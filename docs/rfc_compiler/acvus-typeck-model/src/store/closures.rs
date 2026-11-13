//! Closures: each a body of the store and the classes of it a call gives and
//! takes, and which closure each copy of a body is a copy of.
//!
//! A function value holds closures by name alone, so nothing is compared inside
//! one. What a name stands for is kept here, and only added to: a closure is
//! named where its body is written, in the text or in a copy of a body that
//! holds it, and never changed.

use std::collections::BTreeMap;

use crate::primitive::class::ClassKind;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::{ClosureId, ScopeId};

/// A body, the scope its text is written in, with the classes of its
/// parameters and of its result. What the body captures is whatever class
/// outside that scope its text names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Closure<K>
where
    K: ClassKind,
{
    pub body: ScopeId,
    pub parameters: Vec<K::Id>,
    pub result: K::Id,
}

/// A copy of the body of a closure: the classes its parameters and its result
/// became.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Copy<K>
where
    K: ClassKind,
{
    pub closure: ClosureId,
    pub parameters: Vec<K::Id>,
    pub result: K::Id,
}

#[derive(Clone, Debug)]
pub struct Closures<K>
where
    K: ClassKind,
{
    closures: Arena<ClosureId, Closure<K>>,
    /// Each copy, by the scope its classes lie in.
    copies: BTreeMap<ScopeId, Copy<K>>,
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
    /// Names a closure anew.
    pub fn name(&mut self, closure: Closure<K>) -> ClosureId {
        self.closures.push(closure)
    }

    pub fn get(&self, closure: ClosureId) -> &Closure<K> {
        self.closures.get(closure)
    }

    /// The closure, for [`super::template::anew`] to give it the classes its own
    /// became.
    pub(super) fn get_mut(&mut self, closure: ClosureId) -> &mut Closure<K> {
        self.closures.get_mut(closure)
    }

    /// Every copy said, with the scope its classes lie in.
    pub fn copies(&self) -> impl Iterator<Item = (ScopeId, &Copy<K>)> {
        self.copies.iter().map(|(scope, copy)| (*scope, copy))
    }

    /// Every closure named, in the order they were named.
    pub fn iter(&self) -> impl Iterator<Item = (ClosureId, &Closure<K>)> {
        self.closures.iter()
    }

    /// Says that the classes of `scope` are `copy`.
    pub fn copied(&mut self, scope: ScopeId, copy: Copy<K>) {
        let before = self.copies.insert(scope, copy);
        debug_assert_eq!(before, None, "a scope is one copy");
    }

    /// The copy of `closure` that `scope` is or lies in: the innermost such.
    pub fn within(&self, scopes: &Scopes, scope: ScopeId, closure: ClosureId) -> Option<&Copy<K>> {
        let mut scope = Some(scope);
        while let Some(current) = scope {
            if let Some(copy) = self.copies.get(&current).filter(|copy| copy.closure == closure) {
                return Some(copy);
            }
            scope = scopes.parent(current);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::ClassId;
    use crate::primitive::class::Live;
    use crate::primitive::id::Arena;

    fn two_closures() -> (Closures<Live>, ClosureId, ClosureId) {
        let scopes = Scopes::new();
        let [result, _] = classes();
        let of_one_body = || Closure {
            body: scopes.root(),
            parameters: Vec::new(),
            result,
        };
        let mut closures = Closures::<Live>::default();
        let (first, second) = (closures.name(of_one_body()), closures.name(of_one_body()));
        (closures, first, second)
    }

    #[test]
    fn two_closures_of_one_body_are_two() {
        let (closures, first, second) = two_closures();
        assert_ne!(first, second);
        assert_eq!(closures.get(first), closures.get(second));
    }

    /// A copy of `closure` with no parameter, whose result is `result`.
    fn copy(closure: ClosureId, result: ClassId) -> Copy<Live> {
        Copy {
            closure,
            parameters: Vec::new(),
            result,
        }
    }

    fn classes() -> [ClassId; 2] {
        let mut ids = Arena::<ClassId, ()>::default();
        [ids.push(()), ids.push(())]
    }

    #[test]
    fn a_scope_lies_within_the_copy_it_is_and_the_copies_that_hold_it() {
        let (mut closures, first, second) = two_closures();
        let [of_outer, of_inner] = classes();
        let mut scopes = Scopes::new();
        let outer = scopes.child(scopes.root());
        let inner = scopes.child(outer);
        closures.copied(outer, copy(first, of_outer));
        closures.copied(inner, copy(second, of_inner));
        assert_eq!(closures.within(&scopes, inner, second), Some(&copy(second, of_inner)));
        assert_eq!(closures.within(&scopes, inner, first), Some(&copy(first, of_outer)));
        assert_eq!(closures.within(&scopes, outer, second), None);
        assert_eq!(closures.within(&scopes, scopes.root(), first), None);
    }

    #[test]
    fn of_two_copies_of_one_closure_the_innermost_is_found() {
        let (mut closures, first, _) = two_closures();
        let [of_outer, of_inner] = classes();
        let mut scopes = Scopes::new();
        let outer = scopes.child(scopes.root());
        let inner = scopes.child(outer);
        closures.copied(outer, copy(first, of_outer));
        closures.copied(inner, copy(first, of_inner));
        assert_eq!(closures.within(&scopes, inner, first), Some(&copy(first, of_inner)));
    }
}
