//! Calling: a function value passed to a function type.
//!
//! `f(ā)` passes `f` to `Fn(ā) → r`, a type the call writes. A function value is
//! a set of closures, and nothing is compared inside one, so what it is to pass
//! one to a function type is to use each of its closures: the template of each
//! is copied, each argument enters the copy's parameter, and the copy's result
//! is the type the function type writes. So an argument keeps the bounds of
//! every closure, and the results of all are one type.
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

/// Uses each closure of `value`, a function value, as `wanted`, a function
/// type, by `at`.
///
/// | Of each closure | In its copy |
/// |---|---|
/// | a class standing for a capture | is one type with the class the closure captures |
/// | a parameter | is entered by the argument in its place |
/// | the result | is one type with the result of the function type |
/// | a lambda the body defines | is a closure named anew, over the copies of what it captures |
///
/// A closure whose lambda takes another count of parameters is refused, and
/// then no copy is made.
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
    let (Shape::FnValue(called), Shape::Fn(arguments, result)) = (head(value), head(wanted)) else {
        unreachable!("a call is a function value passed to a function type");
    };
    let scope = scribe.classes().get(scribe.classes().find(wanted)).scope;
    let counts = |closure| lambdas.get(scribe.closures().get(closure).lambda).parameters.len();
    if called.iter().any(|closure| counts(*closure) != arguments.len()) {
        return Looked::Refused(Refused {
            cause: Cause::Mismatch,
            items: BTreeSet::from([at]),
            of: vec![value, wanted],
        });
    }
    let mut pairs = Vec::new();
    let mut constraints = Vec::new();
    for closure in called {
        // A call within a copy of the closure it calls is a call of that copy,
        // and makes none: what is called is being used already.
        let used = match scribe.closures().within(scribe.scopes(), scope, closure).cloned() {
            Some(within) => within,
            None => copy(scribe, lambdas, closure, scope, at, &mut pairs, &mut constraints),
        };
        pairs.push(Pair::Classes(used.result, result));
        for (parameter, argument) in used.parameters.iter().zip(&arguments) {
            constraints.push(
                Bound::Enter {
                    lower: *argument,
                    upper: *parameter,
                }
                .into(),
            );
        }
    }
    Looked::Holds {
        pairs,
        adjustments: Vec::new(),
        constraints,
    }
}

/// Copies the template of `closure` into a scope of its own within `scope`, by
/// `at`, and gives the copy. Each class standing for a capture is one type with
/// what the closure captures, and each lambda the body defines is a closure named
/// anew, whose function value enters the class of the lambda.
fn copy<K>(
    scribe: &mut Scribe<'_, K>,
    lambdas: &Lambdas,
    closure: ClosureId,
    scope: ScopeId,
    at: ItemId,
    pairs: &mut Vec<Pair<K>>,
    constraints: &mut Vec<Constraint<K>>,
) -> Copy<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let closed = scribe.closures().get(closure).clone();
    let lambda = lambdas.get(closed.lambda);
    let lies_in = scribe.scope_within(scope);
    let made = scribe.anew(&lambda.template, lies_in, at);
    let copy = Copy {
        closure,
        parameters: lambda.parameters.iter().map(|parameter| made[parameter]).collect(),
        result: made[&lambda.result],
    };
    scribe.copied(lies_in, copy.clone());
    pairs.extend(lambda.captures.iter().zip(closed.captures).map(|(stands, captured)| Pair::Classes(made[stands], captured)));
    for defined in &lambda.defines {
        let captures = defined.captures.iter().map(|captured| made[captured]).collect();
        let named = scribe.name(defined.lambda, captures);
        let written = scribe.written(lies_in, Shape::FnValue(BTreeSet::from([named])), BTreeSet::from([defined.by]));
        constraints.push(
            Bound::Enter {
                lower: written,
                upper: made[&defined.value],
            }
            .into(),
        );
    }
    copy
}
