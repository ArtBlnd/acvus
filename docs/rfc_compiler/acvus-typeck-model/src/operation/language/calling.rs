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

use crate::primitive::ItemId;
use crate::primitive::class::{ClassKind, State};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;

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
        if scribe.closures().within(scribe.scopes(), scope, closure).is_some() {
            todo!("a call of a closure within a copy of it: whether it is equal to that copy or enters it");
        }
        let closed = scribe.closures().get(closure).clone();
        let lambda = lambdas.get(closed.lambda);
        let copy = scribe.copy_of(closure, scope);
        let made = scribe.anew(&lambda.template, copy, at);
        pairs.extend(lambda.captures.iter().zip(closed.captures).map(|(stands, captured)| Pair::Classes(made[stands], captured)));
        pairs.push(Pair::Classes(made[&lambda.result], result));
        for (parameter, argument) in lambda.parameters.iter().zip(&arguments) {
            constraints.push(
                Bound::Enter {
                    lower: *argument,
                    upper: made[parameter],
                }
                .into(),
            );
        }
        for defined in &lambda.defines {
            let captures = defined.captures.iter().map(|captured| made[captured]).collect();
            let named = scribe.name(defined.lambda, captures);
            let written = scribe.written(copy, Shape::FnValue(BTreeSet::from([named])), BTreeSet::from([defined.by]));
            constraints.push(
                Bound::Enter {
                    lower: written,
                    upper: made[&defined.value],
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
