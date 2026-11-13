//! The exit of `e?`: what the error `e` exits with puts into the error of the
//! body's result.
//!
//! `e?` names two payloads of `e`, as a projection names a part: that of `Ok` is
//! its value, and that of `Err` is what it exits with. A `Result` whose error is
//! a class of its own enters the body's result at once. What enters that class is
//! told by the head of what `e` exits with, and waits for it.

use std::collections::BTreeSet;

use crate::primitive::ClassId;
use crate::primitive::Name;
use crate::primitive::constraint::Bound;
use crate::primitive::ty::Shape;

use super::super::law::Looked;
use super::{Then, Told, View};

/// What the exit with `lower` into the error `upper` comes to.
///
/// | The head `lower` has | What enters `upper` |
/// |---|---|
/// | none yet | nothing yet: it waits |
/// | a type of a name `N`: an enum, a struct, an extern-defined type | an `Error` with the one variant `N`, whose payload is `lower` |
/// | an `Error` with variants | `lower` itself, and the join unites its variants with the others |
/// | `!` | `lower` itself: nothing exits |
/// | any other | `Error⟨lower⟩` |
///
/// It asks no head of `lower`: what exits is as it is.
pub(super) fn look(view: &View<'_>, lower: ClassId, upper: ClassId) -> Told {
    let Some(head) = view.head(lower) else {
        return Told::done(Looked::Waits {
            on: BTreeSet::from([lower]),
            apart: BTreeSet::new(),
            pairs: Vec::new(),
        });
    };
    let holds = |constraints| Looked::Holds {
        pairs: Vec::new(),
        adjustments: Vec::new(),
        constraints,
    };
    let variant = |name: &Name| Shape::Error([(name.clone(), lower)].into_iter().collect());
    let wrapped = match head {
        Shape::Enum(name, _) | Shape::Struct(name, _) | Shape::Extern(name, ..) => variant(name),
        Shape::Error(_) | Shape::Never => return Told::done(holds(vec![Bound::Enter { lower, upper }.into()])),
        Shape::Ref(..)
        | Shape::Tuple(_)
        | Shape::Array(..)
        | Shape::Slice(_)
        | Shape::Option(_)
        | Shape::Result(..)
        | Shape::ErrorOf(_)
        | Shape::Fn(..)
        | Shape::FnValue(_)
        | Shape::Object(_)
        | Shape::Brand(_)
        | Shape::Scalar(_)
        | Shape::String
        | Shape::Str
        | Shape::Unit => Shape::ErrorOf(lower),
    };
    Told {
        looked: holds(Vec::new()),
        asks: Vec::new(),
        then: Some(Then::Enters(wrapped)),
    }
}
