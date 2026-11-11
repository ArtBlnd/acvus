//! Calling by name: `x::m(ā)`, a function of the name `m` called with `x` as its
//! receiver.
//!
//! Which function is called is told by the receiver. The functions of the name
//! are rows of the table of declared functions, and the receiver goes to the
//! first parameter of exactly one of them: none goes nowhere, and of two neither
//! is chosen. It goes as any argument is passed, read through references as a
//! passing reads them, and one that is no reference is lent where the parameter
//! is a reference (`passing::receivers`).
//!
//! The function found, its signature is made anew: each argument after the
//! receiver is passed to its parameter, and its result is one type with the
//! result the call writes.

use std::collections::BTreeSet;

use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::{ClassKind, State};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;
use crate::primitive::{ItemId, NameId};

use super::super::law::{Looked, Pair, Scribe};
use super::functions::Functions;
use super::passing::{self, Received};
use super::{Then, Told, View};

/// What calling the function `name` on `receiver` comes to, `wanted` being the
/// function type the call writes.
///
/// While a part of the receiver a row has to read has no head, it waits on that
/// part, and holds apart its ends and what it has read, as a passing does.
pub(super) fn look<K>(view: &View<'_, K>, functions: &Functions, name: NameId, receiver: K::Id, wanted: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let mut read = BTreeSet::from([receiver, wanted]);
    let (mut found, on) = passing::receivers(view, functions, functions.name(name), receiver, &mut read);
    if !on.is_empty() {
        return Told::done(Looked::Waits {
            on,
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

/// Makes the signature of the row `received` names anew, by `at`, and gives what
/// the call comes to. A call of another count of arguments is refused.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, functions: &Functions, received: Received<K>, receiver: K::Id, wanted: K::Id, at: ItemId) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let row = &functions.rows[received.row];
    let of_wanted = scribe.classes().get(scribe.classes().find(wanted));
    let (State::Decided(decided), scope) = (&of_wanted.state, of_wanted.scope) else {
        unreachable!("a call writes its function type");
    };
    let Shape::Fn(arguments, result) = decided.shape.clone() else {
        unreachable!("a call writes its function type");
    };
    let after_the_receiver = &row.parameters[1..];
    if after_the_receiver.len() != arguments.len() {
        return Looked::Refused(Refused {
            cause: Cause::Mismatch,
            items: BTreeSet::from([at]),
            of: vec![receiver, wanted],
        });
    }
    let made = scribe.anew(&row.template, scope, at);
    let mut pairs: Vec<Pair<K>> = received.pairs.iter().map(|(part, written)| Pair::Classes(*part, made[written])).collect();
    pairs.push(Pair::Classes(made[&row.result], result));
    let constraints = arguments
        .iter()
        .zip(after_the_receiver)
        .map(|(argument, parameter)| {
            Bound::Change {
                lower: *argument,
                upper: made[parameter],
            }
            .into()
        })
        .collect();
    let called = Adjustment::Row(row.name.clone());
    Looked::Holds {
        pairs,
        adjustments: received.adjustments.into_iter().chain([called]).collect(),
        constraints,
    }
}
