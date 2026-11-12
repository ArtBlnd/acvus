//! Calling: a function passed to a function type.
//!
//! `f(ā)` passes `f` to `Fn(ā) → r`, a type the call writes. A function value is
//! a set of closures, and nothing is compared inside one, so what it is to pass
//! one to a function type is to use each of its closures: the template of each
//! is copied, each argument is passed to the copy's parameter, and the copy's result
//! is the type the function type writes. So an argument keeps the bounds of
//! every closure, and the results of all are one type. An argument is passed, as it
//! is to a declared function, and does not enter: a parameter that writes no type
//! is a hole, one type with what is passed.
//!
//! A copy is solved by the rules of any class. It is made in a scope of its
//! own, within the scope of the call, so what a call within a copy is a copy of
//! is read from where it lies.

use std::collections::BTreeSet;

use crate::primitive::{ClosureId, ItemId, ScopeId};
use crate::primitive::class::{ClassKind, State};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;

use crate::store::closures::Copy;

use super::super::law::{Looked, Pair, Scribe};
use super::lambdas::Lambdas;

/// A call as it is written: the function type `Fn(arguments) -> result`.
struct Call<K>
where
    K: ClassKind,
{
    arguments: Vec<K::Id>,
    result: K::Id,
}

/// What a function takes and gives: the classes of its parameters and of its
/// result.
struct Signature<K>
where
    K: ClassKind,
{
    parameters: Vec<K::Id>,
    result: K::Id,
}

/// One function that is called.
enum Called<K>
where
    K: ClassKind,
{
    /// A closure of a function value. Its signature is that of a copy of its
    /// template.
    Closure(ClosureId),
    /// A value of a function type, which writes its signature.
    OfAType(Signature<K>),
}

/// What a call comes to: the classes that are one type, and the constraints
/// recorded in its place.
struct Follows<K>
where
    K: ClassKind,
{
    pairs: Vec<Pair<K>>,
    constraints: Vec<Constraint<K>>,
}

impl<K> Call<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    /// Uses a function of `signature` for this call: each argument is passed to
    /// the parameter in its place, and what the function gives is one type with
    /// the result of the call.
    fn uses(&self, signature: &Signature<K>, follows: &mut Follows<K>) {
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
/// A function value calls each of its closures, and a value of a function type
/// calls the one function its type writes. A closure is used by a copy of its
/// template:
///
/// | Of the closure | In its copy |
/// |---|---|
/// | a class standing for a capture | is one type with the class the closure captures |
/// | a parameter, the result | is the signature the call uses |
/// | a lambda the body defines | is a closure named anew, over the copies of what it captures |
///
/// A function that takes another count of parameters than the call passes is
/// refused, and then no copy is made.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, lambdas: &Lambdas, value: K::Id, wanted: K::Id, at: ItemId) -> Looked<K>
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
        Shape::FnValue(closures) => closures.into_iter().map(Called::Closure).collect(),
        Shape::Fn(parameters, result) => vec![Called::OfAType(Signature { parameters, result })],
        _ => unreachable!("a call is a function passed to a function type"),
    };

    let takes = |called: &Called<K>| match called {
        Called::Closure(closure) => lambdas.get(scribe.closures().get(*closure).lambda).parameters.len(),
        Called::OfAType(signature) => signature.parameters.len(),
    };
    if called.iter().any(|called| takes(called) != call.arguments.len()) {
        return Looked::Refused(Refused {
            cause: Cause::Mismatch,
            items: BTreeSet::from([at]),
            of: vec![value, wanted],
        });
    }

    let scope = scribe.classes().get(scribe.classes().find(wanted)).scope;
    let mut follows = Follows {
        pairs: Vec::new(),
        constraints: Vec::new(),
    };
    for called in called {
        let signature = match called {
            Called::Closure(closure) => of_a_closure(scribe, lambdas, closure, scope, &mut follows),
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
/// the copy the call lies within, where it lies within one, and else of a copy
/// made for the call.
///
/// A call within a copy of the closure it calls is a call of that copy, and
/// makes none: what is called is being used already.
fn of_a_closure<K>(scribe: &mut Scribe<'_, K>, lambdas: &Lambdas, closure: ClosureId, scope: ScopeId, follows: &mut Follows<K>) -> Signature<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let within = scribe.closures().within(scribe.scopes(), scope, closure).cloned();
    let Copy { parameters, result, .. } = match within {
        Some(in_use) => in_use,
        None => copy(scribe, lambdas, closure, scope, follows),
    };
    Signature { parameters, result }
}

/// Copies the template of `closure` into a scope of its own within `scope`, and
/// gives the copy. Each class standing for a capture is one type with
/// what the closure captures, and each lambda the body defines is a closure named
/// anew, whose function value enters the class of the lambda.
fn copy<K>(scribe: &mut Scribe<'_, K>, lambdas: &Lambdas, closure: ClosureId, scope: ScopeId, follows: &mut Follows<K>) -> Copy<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let closed = scribe.closures().get(closure).clone();
    let lambda = lambdas.get(closed.lambda);
    let lies_in = scribe.scope_within(scope);
    let made = scribe.anew(&lambda.template, lies_in);
    let copy = Copy {
        closure,
        parameters: lambda.parameters.iter().map(|parameter| made[parameter]).collect(),
        result: made[&lambda.result],
    };
    scribe.copied(lies_in, copy.clone());

    let captured = lambda.captures.iter().zip(closed.captures);
    follows.pairs.extend(captured.map(|(stands_for, captured)| Pair::Classes(made[stands_for], captured)));

    for defined in &lambda.defines {
        let captures = defined.captures.iter().map(|captured| made[captured]).collect();
        let named = scribe.name(defined.lambda, captures);
        let written = scribe.written(lies_in, Shape::FnValue(BTreeSet::from([named])), BTreeSet::from([defined.by]));
        follows.constraints.push(
            Bound::Enter {
                lower: written,
                upper: made[&defined.value],
            }
            .into(),
        );
    }
    copy
}
