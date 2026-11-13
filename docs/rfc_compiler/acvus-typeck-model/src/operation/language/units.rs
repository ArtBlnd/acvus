//! Units: what a function called by name that no declaration writes is, said
//! from outside naming.
//!
//! Naming finds such a function by the type its first parameter writes, as it
//! finds a row, and asks what calling it comes to (`naming::ToldOf`). Here it is
//! a function of a module: the module is used, and that function of it called,
//! the receiver being its first argument.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::ClassKind;
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::ty::{PartKey, Shape};
use crate::primitive::{ClosureId, Name};

use super::super::law::{Looked, Pair, Scribe};
use super::naming::{Call, ToldOf};
use super::passing::Received;

/// A function of a module whose first parameter writes a type: one a value of
/// that type is the receiver of.
#[derive(Clone, Debug)]
pub struct Method {
    pub name: Name,
    /// The closure of the module, which gives the function when it is called.
    module: ClosureId,
}

/// Every function of a module the table of functions was told of, by the place
/// the table gave it.
#[derive(Clone, Debug, Default)]
pub struct Units {
    methods: BTreeMap<usize, Method>,
}

impl Units {
    /// Says that `function`, as the table of functions names it, is the
    /// function `name` of the module whose closure is `module`.
    pub fn method(&mut self, function: usize, name: Name, module: ClosureId) {
        let before = self.methods.insert(function, Method { name, module });
        debug_assert!(before.is_none(), "the table names one function once");
    }
}

impl<K> ToldOf<K> for Units
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    /// Uses the module the function is of, and calls that function of it with
    /// the receiver as its first argument, lent where the receiver is lent.
    ///
    /// Using a module is calling its lambda, which gives its functions by their
    /// names. So this writes what `M::m(x, ā)` writes: the module's lambda is
    /// passed to a function type of no argument, the function is the field of
    /// its name of what that gives, and it is passed to the function type of the
    /// receiver, the arguments and the result. What the receiver takes besides a
    /// lend, the passing of it to the function's first parameter records.
    fn called(&self, scribe: &mut Scribe<'_, K>, function: usize, received: Received<K>, call: Call<K>) -> Looked<K> {
        let method = &self.methods[&function];
        let Call { scope, at, .. } = call;
        let by = || BTreeSet::from([at]);
        let module = scribe.written(scope, Shape::FnValue(BTreeSet::from([method.module])), by());
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
}
