//! The nesting bound of RFC-0106 rule 1, counted as each node is built
//! rather than by a walk after the parse: a walk over a tree past the bound
//! is the overflow the bound exists to prevent, and so is dropping one.

use crate::error::{ParseError, ParseErrorKind};
use crate::span::Span;

/// The most levels a script, a template or an expression nests.
///
/// A level is one construct as the source writes it: an expression, a
/// statement or a pattern. A construct nests one level deeper than the
/// deepest construct it holds, and one that holds none nests one level. The
/// tree mirrors the source one node for one, except that `*r = e` keeps `r`
/// and not the `*r` written, which still counts. So each of these is a level:
///
/// - an expression inside another: an operand, a call's argument, a
///   parenthesized expression, a field's value. A chain of binary operators
///   nests one level per operator, since each holds the chain before it as
///   its left operand, and a format string is such a chain of `+`;
/// - a block `{ s; e }`, which is an expression holding its statements and
///   its tail;
/// - a lambda `|x| -> e`, an expression holding its body;
/// - a pattern inside another, as in `Some(Some(x))`;
/// - a template section from `% if` to `% end`, which is a statement
///   holding an `if` expression, whose lines are statements below it.
///
/// A match arm, an `else` branch, a tuple or list element, an object field,
/// a `for` head, a body, a `fn` declaration and the script or template
/// itself hold nodes without being one, and add no level.
///
/// The value is chosen from the `nesting_wasm` example of `acvus-mir-test`,
/// which compiles a source at this depth for each kind of level on
/// `wasm32` with a 16 MiB linear stack; docs/nesting.md
/// names the command and records the stack each kind took. A change to the
/// compiler's walks that grows their frames is checked against that
/// measurement, not against this number.
pub const NESTING_MAX: u32 = 256;

pub(crate) struct Nested<T> {
    node: T,
    height: u32,
}

impl<T> Nested<T> {
    pub(crate) fn leaf(node: T) -> Self {
        Self { node, height: 1 }
    }

    pub(crate) fn empty(node: T) -> Self {
        Self { node, height: 0 }
    }

    pub(crate) fn root(self) -> T {
        self.node
    }

    pub(crate) fn get(&self) -> &T {
        &self.node
    }
}

impl<T> Nested<Vec<T>> {
    pub(crate) fn one(item: Nested<T>) -> Self {
        let mut items = Nested::empty(Vec::new());
        items.push(item);
        items
    }

    pub(crate) fn push(&mut self, item: Nested<T>) {
        self.height = self.height.max(item.height);
        self.node.push(item.node);
    }
}

pub(crate) struct Level {
    below: u32,
}

impl Level {
    pub(crate) fn new() -> Self {
        Self { below: 0 }
    }

    pub(crate) fn take<T>(&mut self, nested: Nested<T>) -> T {
        self.below = self.below.max(nested.height);
        nested.node
    }

    pub(crate) fn boxed<T>(&mut self, nested: Nested<T>) -> Box<T> {
        Box::new(self.take(nested))
    }

    pub(crate) fn opt_boxed<T>(&mut self, nested: Option<Nested<T>>) -> Option<Box<T>> {
        nested.map(|nested| self.boxed(nested))
    }

    pub(crate) fn all<T>(&mut self, nested: Vec<Nested<T>>) -> Vec<T> {
        nested.into_iter().map(|nested| self.take(nested)).collect()
    }

    pub(crate) fn node<T, E>(self, node: T, span: Span) -> Result<Nested<T>, E>
    where
        E: From<ParseError>,
    {
        let height = self.below + 1;
        match height <= NESTING_MAX {
            true => Ok(Nested { node, height }),
            false => Err(E::from(ParseError::new(
                ParseErrorKind::NestingTooDeep { max: NESTING_MAX },
                span,
            ))),
        }
    }

    pub(crate) fn part<T>(self, node: T) -> Nested<T> {
        Nested {
            node,
            height: self.below,
        }
    }
}
