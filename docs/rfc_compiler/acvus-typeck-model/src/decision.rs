//! The decisions that read the upper side of a relation: the class a value
//! passes into, which is the `α` of a lower bound or a part lower bound, the
//! `U` of a boundary, and the parameter and result types a use's arguments
//! and result pass into or out of.
//!
//! A class's value is the join of its sources, and a decision reads its
//! inputs: a use its arguments, a lend its place, a projection its operand, a
//! binding its path and position, a call of a function value its callee, and
//! a class its sources leave undecided its range. Nothing else reads the upper
//! side, so information from it reaches a typing only through a row of this
//! table, and a row states all it does there:
//!
//! - the alternatives it adds to a decision;
//! - the alternatives it removes: those under which what follows by the
//!   sources alone fails a check whatever the other decisions become;
//! - the alternative it prefers once nothing is left to remove;
//! - what it makes wait for the nodes that record the upper side it reads.
//!
//! The table is empty. Without a row, a use whose candidates differ only in
//! their results is not decided, a class nothing enters falls back, and a
//! literal takes its family's default whatever it passes into.

/// A row of the table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Row {}

impl Row {
    /// Every row of the table, in order.
    pub const ALL: [Row; 0] = [];
}
