//! Calling by name: `x::m(ā)`, a function of the name `m` called with `x` as its
//! receiver.
//!
//! Which function is called is told by the receiver. The functions of the name
//! are rows of the table of declared functions and functions of modules, and the
//! receiver goes to the first parameter of exactly one of them: none goes
//! nowhere, and of two neither is chosen. It goes as any argument is passed, read through references as a
//! passing reads them, and one that is no reference is lent where the parameter
//! is a reference (`passing::receivers`).
//!
//! A row found, its signature is made anew: each argument after the receiver is
//! passed to its parameter, and its result is one type with the result the call
//! writes. A function of a module found, the module is used and the function
//! called, the receiver being its first argument.

use std::collections::BTreeSet;

use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::{ClassKind, State};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{PartKey, Shape};
use crate::primitive::{ItemId, NameId, ScopeId};

use super::super::law::{Looked, Pair, Scribe};
use super::calling;
use super::functions::{Functions, Method, Named, Row, Used};
use super::passing::{self, Received};
use super::{Then, Told, View};

/// What calling the function `name` on `receiver` comes to, `wanted` being the
/// function type the call writes.
///
/// While a part of the receiver a function has to read has no head, it waits on
/// that part. The function found, each argument is passed to a parameter and the
/// result is one type with the function's. So until then it holds the receiver
/// and each argument apart as the passing to come holds its lower end, which may
/// be decided while what lies below it is not; and the result, which is an end
/// of no passing, is not decided before.
pub(super) fn look<K>(view: &View<'_, K>, functions: &Functions, name: NameId, receiver: K::Id, wanted: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let mut read = BTreeSet::from([receiver, wanted]);
    let Some(Shape::Fn(arguments, _)) = view.head(wanted) else {
        unreachable!("a call writes its function type");
    };
    read.extend(arguments.iter().copied());
    let (mut found, undecided) = passing::receivers(view, functions, functions.name(name), receiver, &mut read);
    if !undecided.is_empty() {
        return Told::done(Looked::Waits {
            on: undecided,
            apart: read,
            pairs: Vec::new(),
        });
    }
    // Exactly one function must take the receiver.
    let (Some(received), true) = (found.pop(), found.is_empty()) else {
        return Told::done(Looked::Refused(Refused {
            cause: Cause::NoChange,
            items: BTreeSet::from([at]),
            of: vec![receiver, wanted],
        }));
    };
    Told {
        looked: Looked::holds_alone(),
        asks: Vec::new(),
        then: Some(Then::Named(received)),
    }
}

/// What the call comes to, the function `received` names being the one called.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, functions: &Functions, received: Received<K>, receiver: K::Id, wanted: K::Id, at: ItemId) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let of_wanted = scribe.classes().get(scribe.classes().find(wanted));
    let (State::Decided(decided), scope) = (&of_wanted.state, of_wanted.scope) else {
        unreachable!("a call writes its function type");
    };
    let Shape::Fn(arguments, result) = decided.shape.clone() else {
        unreachable!("a call writes its function type");
    };
    let call = Call {
        receiver,
        wanted,
        arguments,
        result,
        scope,
        at,
    };
    match received.function {
        Named::Row(row) => of_a_row(scribe, &functions.rows[row], received, call),
        Named::Method(method) => of_a_module(scribe, &functions.methods[method], received, call),
    }
}

/// A call by name as it is written: its receiver, the function type it writes
/// and what that type holds, the scope it lies in and the item it goes by.
struct Call<K>
where
    K: ClassKind,
{
    receiver: K::Id,
    wanted: K::Id,
    arguments: Vec<K::Id>,
    result: K::Id,
    scope: ScopeId,
    at: ItemId,
}

/// Makes the signature of `row` anew: each argument after the receiver is passed
/// to its parameter, and the row's result is one type with the call's. A call of
/// another count of arguments is refused.
fn of_a_row<K>(scribe: &mut Scribe<'_, K>, row: &Row, received: Received<K>, call: Call<K>) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let after_the_receiver = &row.parameters[1..];
    let written = calling::Call {
        arguments: call.arguments,
        result: call.result,
    };
    if let Some(refused) = written.count(after_the_receiver.len(), [call.receiver, call.wanted], call.at) {
        return Looked::Refused(refused);
    }
    let Used { classes: made, brands } = row.used(scribe, call.scope, call.at);

    let Received { fit: receiver, .. } = received;

    // What the receiver brings, which was told where the row was found: each
    // part of it the row reads is one type with what the row writes there, and
    // a receiver that is a function is passed to the function type the row
    // writes for it.
    let of_the_receiver = receiver.pairs.iter().map(|(part, written)| Pair::Classes(*part, made[written]));
    let the_receiver = receiver.passed.iter().map(|(function, written)| Bound::Change { lower: *function, upper: made[written] }.into());
    let mut follows = calling::Follows {
        pairs: of_the_receiver.collect(),
        constraints: the_receiver.collect(),
    };

    // The arguments after the receiver and the result, as any call uses a
    // signature.
    let signature = calling::Signature {
        parameters: after_the_receiver.iter().map(|parameter| made[parameter]).collect(),
        result: made[&row.result],
    };
    written.uses(&signature, &mut follows);

    // One type besides: each brand the row makes with the one made for this
    // call.
    follows.pairs.extend(brands);

    let called = Adjustment::Row(row.name.clone());
    Looked::Holds {
        pairs: follows.pairs,
        adjustments: receiver.adjustments.into_iter().chain([called]).collect(),
        constraints: follows.constraints,
    }
}

/// Uses the module `method` is of, and calls that function of it with the
/// receiver as its first argument, lent where the receiver is lent.
///
/// Using a module is calling its lambda, which gives its functions by their
/// names. So this writes what `M::m(x, ā)` writes: the module's lambda is passed
/// to a function type of no argument, the function is the field of its name of
/// what that gives, and it is passed to the function type of the receiver, the
/// arguments and the result. What the receiver takes besides a lend, the passing
/// of it to the function's first parameter records.
fn of_a_module<K>(scribe: &mut Scribe<'_, K>, method: &Method, received: Received<K>, call: Call<K>) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let Call { scope, at, .. } = call;
    let by = || BTreeSet::from([at]);
    let closure = scribe
        .closures()
        .iter()
        .find_map(|(closure, closed)| (closed.lambda == method.module).then_some(closure))
        .expect("a module is a lambda written where the program is, and a closure of it is named there");
    let module = scribe.written(scope, Shape::FnValue(BTreeSet::from([closure])), by());
    let functions = scribe.undecided(scope);
    let used = scribe.written(scope, Shape::Fn(Vec::new(), functions), by());
    let function = scribe.undecided(scope);
    let lent = received.fit.adjustments.iter().find_map(|adjustment| match adjustment {
        Adjustment::Lend(mutability) => Some(*mutability),
        _ => None,
    });
    let receiver = match lent {
        Some(mutability) => scribe.written(scope, Shape::Ref(mutability, call.receiver), by()),
        None => call.receiver,
    };
    let passed = [receiver].into_iter().chain(call.arguments).collect();
    let called = scribe.written(scope, Shape::Fn(passed, call.result), by());
    Looked::Holds {
        pairs: vec![Pair::Part {
            part: function,
            of: functions,
            key: PartKey::Field(method.name.clone()),
        }],
        adjustments: lent.map(Adjustment::Lend).into_iter().collect(),
        constraints: vec![
            Bound::Change {
                lower: module,
                upper: used,
            }
            .into(),
            Bound::Change {
                lower: function,
                upper: called,
            }
            .into(),
        ],
    }
}
