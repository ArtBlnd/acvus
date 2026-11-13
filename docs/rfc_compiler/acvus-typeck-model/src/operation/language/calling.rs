//! Calling: a function passed to a function type.
//!
//! `f(ā)` passes `f` to `Fn(ā) → r`, a type the call writes. What is called
//! gives a signature, and the call uses it (`Call::uses`): each argument is
//! passed to the parameter in its place, and what the function gives is one type
//! with the result. An argument is passed, as it is to a declared function, and
//! does not enter: a parameter that writes no type is a hole, one type with what
//! is passed.
//!
//! A function value is a set of closures, held by their names alone, and
//! nothing is compared inside one. So to pass one to a function type is to use
//! each of its closures, and an argument keeps the bounds of every closure, and
//! the results of all are one type. A closure is a body of the store, and using
//! it is making that body anew: what lies in the body is made anew, and what
//! its text names outside it is the same class.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::class::{ClassKind, State};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;
use crate::primitive::{ClosureId, ItemId, ScopeId};

use crate::store::closures::{Closure, Copy};

use super::super::law::{Looked, Pair, Scribe};

/// A call as it is written: the function type `Fn(arguments) -> result`.
pub(super) struct Call<K>
where
    K: ClassKind,
{
    pub(super) arguments: Vec<K::Id>,
    pub(super) result: K::Id,
}

/// What a function takes and gives: the classes of its parameters and of its
/// result.
pub(super) struct Signature<K>
where
    K: ClassKind,
{
    pub(super) parameters: Vec<K::Id>,
    pub(super) result: K::Id,
}

/// One function that is called.
enum Called<K>
where
    K: ClassKind,
{
    /// A closure of a function value, by its name.
    Closure(ClosureId),
    /// A value of a function type, which writes its signature.
    OfAType(Signature<K>),
}

/// What a call comes to: the classes that are one type, and the constraints
/// recorded in its place.
pub(super) struct Follows<K>
where
    K: ClassKind,
{
    pub(super) pairs: Vec<Pair<K>>,
    pub(super) constraints: Vec<Constraint<K>>,
}

impl<K> Call<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    /// What is refused where this call is of a function that takes `parameters`
    /// parameters, by `at`: a call of another count of arguments, of the
    /// function and the function type the call writes.
    pub(super) fn count(&self, parameters: usize, of: [K::Id; 2], at: ItemId) -> Option<Refused<K>> {
        (parameters != self.arguments.len()).then(|| Refused {
            cause: Cause::Mismatch,
            items: BTreeSet::from([at]),
            of: of.to_vec(),
        })
    }

    /// Uses a function of `signature` for this call: each argument is passed to
    /// the parameter in its place, and what the function gives is one type with
    /// the result of the call.
    pub(super) fn uses(&self, signature: &Signature<K>, follows: &mut Follows<K>) {
        follows.pairs.push(Pair::Classes(signature.result, self.result));
        for (argument, parameter) in self.arguments.iter().zip(&signature.parameters) {
            follows.constraints.push(
                Bound::Change {
                    lower: *argument,
                    upper: *parameter,
                }
                .into(),
            );
        }
    }
}

/// Uses `value`, a function, as `wanted`, a function type, by `at`: each
/// function called is used for the call `wanted` writes ([`Call::uses`]).
///
/// A function value calls each of its closures ([`of_a_closure`]), and a value
/// of a function type calls the one function its type writes. A function that
/// takes another count of parameters than the call passes is refused, and then
/// no body is made anew.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, value: K::Id, wanted: K::Id, at: ItemId) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let head = |class: K::Id| {
        let State::Decided(decided) = &scribe.classes().get(scribe.classes().find(class)).state else {
            unreachable!("a call is told once both its ends have a head");
        };
        decided.shape.clone()
    };
    let Shape::Fn(arguments, result) = head(wanted) else {
        unreachable!("a call is a function passed to a function type");
    };
    let call = Call { arguments, result };
    let called: Vec<Called<K>> = match head(value) {
        Shape::FnValue(named) => named.into_iter().map(Called::Closure).collect(),
        Shape::Fn(parameters, result) => vec![Called::OfAType(Signature { parameters, result })],
        _ => unreachable!("a call is a function passed to a function type"),
    };

    let takes = |called: &Called<K>| match called {
        Called::Closure(named) => scribe.closures().get(*named).parameters.len(),
        Called::OfAType(signature) => signature.parameters.len(),
    };
    if let Some(refused) = called.iter().find_map(|called| call.count(takes(called), [value, wanted], at)) {
        return Looked::Refused(refused);
    }

    let scope = scribe.classes().get(scribe.classes().find(wanted)).scope;
    let mut follows = Follows {
        pairs: Vec::new(),
        constraints: Vec::new(),
    };
    for called in called {
        let signature = match called {
            Called::Closure(named) => of_a_closure(scribe, named, scope, at, &mut follows),
            Called::OfAType(signature) => signature,
        };
        call.uses(&signature, &mut follows);
    }
    Looked::Holds {
        pairs: follows.pairs,
        adjustments: Vec::new(),
        constraints: follows.constraints,
    }
}

/// The signature `closure` is used by where it is called in `scope`: that of
/// the copy the call lies within, where it lies within one, and else of its
/// body made anew for the call, which is then said to be a copy of it.
///
/// A call within a copy of the closure it calls, or within the body of that
/// closure, is a call of that one, and makes none: what is called is being used
/// already.
fn of_a_closure<K>(scribe: &mut Scribe<'_, K>, closure: ClosureId, scope: ScopeId, at: ItemId, follows: &mut Follows<K>) -> Signature<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let within = scribe.closures().within(scribe.scopes(), scope, closure).cloned();
    let Copy { parameters, result, .. } = match within {
        Some(in_use) => in_use,
        None => {
            let written = scribe.closures().get(closure).clone();
            let made = Used::of(scribe, &written, scope, at);
            let copy = Copy {
                closure,
                parameters: written.parameters.iter().map(|parameter| made.became(*parameter)).collect(),
                result: made.became(written.result),
            };
            scribe.copied(made.scope, copy.clone());
            follows.pairs.extend(made.pairs);
            copy
        }
    };
    Signature { parameters, result }
}

/// The body of a closure made anew for one use of it.
pub(super) struct Used<K>
where
    K: ClassKind,
{
    /// The scope the copy lies in.
    pub(super) scope: ScopeId,
    /// Each class of the body, with the class it became.
    classes: BTreeMap<K::Id, K::Id>,
    /// What is one type by the use: each class that stands for a brand the
    /// closure makes with the brand made for this use.
    pub(super) pairs: Vec<Pair<K>>,
}

impl<K> Used<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    /// Makes the body of `closure` anew within `scope`, by `at`, and a brand
    /// for each the closure makes.
    ///
    /// This is the only place a body is made anew for a use: of a declared
    /// function, whose body holds its signature alone, as of a lambda.
    pub(super) fn of(scribe: &mut Scribe<'_, K>, closure: &Closure<K>, scope: ScopeId, at: ItemId) -> Self {
        let made = scribe.anew_body(closure.body, scope);
        let pairs = closure
            .makes
            .iter()
            .map(|stands_for| Pair::Classes(made.classes.get(stands_for).copied().unwrap_or(*stands_for), scribe.brand(scope, at)))
            .collect();
        Self {
            scope: made.scope,
            classes: made.classes,
            pairs,
        }
    }

    /// The class `class`, of the body or named by it, is in the copy: the one
    /// made anew for it where it lies in the body, and itself where it lies
    /// outside.
    pub(super) fn became(&self, class: K::Id) -> K::Id {
        self.classes.get(&class).copied().unwrap_or(class)
    }
}
