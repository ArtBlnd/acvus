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
//! writes. A function the table was told of found, what the call comes to is
//! said by whoever told of it (`ToldOf`).

use std::collections::BTreeSet;

use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::State;
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;
use crate::primitive::{ClassId, ItemId, NameId, ScopeId};

use super::super::law::{Looked, Pair, Scribe};
use super::calling;
use super::functions::{Functions, Named, Row};
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
pub(super) fn look(view: &View<'_>, functions: &Functions, name: NameId, receiver: ClassId, wanted: ClassId, at: ItemId) -> Told {
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

/// What calling a function the table of functions was told of comes to, as
/// whoever told of it says.
///
/// Naming knows such a function by its place in the table and by the type its
/// first parameter writes alone.
pub(super) trait ToldOf {
    /// What `call` comes to, `function` being the one called and `received` how
    /// the receiver fits its first parameter.
    fn called(&self, scribe: &mut Scribe<'_, Constraint>, function: usize, received: Received, call: Call) -> Looked<Constraint>;
}

/// What the call comes to, the function `received` names being the one called:
/// a row by its closure, and a function the table was told of as `told` says.
pub(super) fn take<T>(scribe: &mut Scribe<'_, Constraint>, functions: &Functions, told: &T, received: Received, receiver: ClassId, wanted: ClassId, at: ItemId) -> Looked<Constraint>
where
    T: ToldOf,
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
        Named::Told(function) => told.called(scribe, function, received, call),
    }
}

/// A call by name as it is written: its receiver, the function type it writes
/// and what that type holds, the scope it lies in and the item it goes by.
pub(super) struct Call {
    pub(super) receiver: ClassId,
    pub(super) wanted: ClassId,
    pub(super) arguments: Vec<ClassId>,
    pub(super) result: ClassId,
    pub(super) scope: ScopeId,
    pub(super) at: ItemId,
}

/// Makes the signature of `row` anew: each argument after the receiver is passed
/// to its parameter, and the row's result is one type with the call's. A call of
/// another count of arguments is refused.
fn of_a_row(scribe: &mut Scribe<'_, Constraint>, row: &Row, received: Received, call: Call) -> Looked<Constraint> {
    let declared = scribe.closures().get(row.closure).clone();
    let after_the_receiver = &declared.parameters[1..];
    let written = calling::Call {
        arguments: call.arguments,
        result: call.result,
    };
    if let Some(refused) = written.count(after_the_receiver.len(), [call.receiver, call.wanted], call.at) {
        return Looked::Refused(refused);
    }
    let made = calling::Used::of(scribe, &declared, call.scope, call.at);

    let Received { fit: receiver, .. } = received;

    // What the receiver brings, which was told where the row was found: each
    // part of it the row reads is one type with what the row writes there, and
    // a receiver that is a function is passed to the function type the row
    // writes for it.
    let of_the_receiver = receiver.pairs.iter().map(|(part, written)| Pair::Classes(*part, made.became(*written)));
    let the_receiver = receiver.passed.iter().map(|(function, written)| Bound::Change { lower: *function, upper: made.became(*written) }.into());
    let mut follows = calling::Follows {
        pairs: of_the_receiver.collect(),
        constraints: the_receiver.collect(),
    };

    // The arguments after the receiver and the result, as any call uses a
    // signature.
    let signature = calling::Signature {
        parameters: after_the_receiver.iter().map(|parameter| made.became(*parameter)).collect(),
        result: made.became(declared.result),
    };
    written.uses(&signature, &mut follows);

    // What making the row's template anew for this call leaves besides: each
    // brand the row makes is one type with the one made for the call.
    follows.pairs.extend(made.pairs);

    let called = Adjustment::Row(row.name.clone());
    Looked::Holds {
        pairs: follows.pairs,
        adjustments: receiver.adjustments.into_iter().chain([called]).collect(),
        constraints: follows.constraints,
    }
}
