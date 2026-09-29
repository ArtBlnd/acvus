//! Takings: how each instruction and terminator takes each value it names,
//! moved, read or cut in part (ownership §1-3, §5, §6; functions §3). This is
//! the one statement of it: `passes::analysis::Initialization` records the
//! moves it states, `passes::analysis::Loans` makes an implicit reborrow
//! where it reads a `&mut` at a by-value position, `check::borrowck` holds
//! each access as it states, and `lower::required` clones and drops by it.
//!
//! - A value that copies is read (ownership §1, §2).
//! - A `&mut` is read, by value or taken from a place by name: each use
//!   reborrows it and its holder keeps it (ownership §5).
//! - A callee called once is moved and one called many times is read, and a
//!   closure kept at signatures is moved where one of them is called once
//!   (functions §3).
//! - A part that does not copy, taken from a whole that does not copy, or a
//!   part replaced in one, cuts the whole (ownership §6).
//! - Any other value at a by-value position is moved (ownership §2). A
//!   `String` is too: `lower::required::clone_strings` has already given
//!   every use but its last a clone (ownership §3).
//!
//! A `Take`'s `taken_out` is not read: lower makes no take-out.

use smallvec::SmallVec;

use crate::ir::ValueId;
use crate::ir::cfg::{Body, Inst, Terminator};
use crate::ty::{Concrete, Ty, Types};

/// How one value is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taken {
    /// The value goes to another holder, or ends here.
    Moved,
    /// The value is read, and its holder keeps it. At a by-value position,
    /// a `&mut` read so is its implicit reborrow (ownership §5).
    Read,
    /// A part that does not copy leaves a whole that does not copy, or a
    /// part is replaced in one: the whole is cut, and each of its units is
    /// ended apart (ownership §6).
    PartMoved,
}

/// One operand of an instruction and how it is taken. `names_place` marks
/// an operand that names a storage, as a `Take`'s or an `Assign`'s target
/// does, rather than passing a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub value: ValueId,
    pub taken: Taken,
    pub names_place: bool,
}

/// A value a terminator names apart from its edges' arguments, and how it
/// is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Named {
    pub value: ValueId,
    pub taken: Taken,
}

/// How a by-value position takes a value of `ty`: an argument, an edge's
/// argument, a returned value, a value stored or built into another.
pub fn by_value(types: &Types<Concrete>, ty: Ty) -> Taken {
    todo!("a `&mut` is read, any other value moved")
}

/// Every operand of `inst` but the order token, in the order
/// `Inst::operands` gives them, with how each is taken. A `Take` of a place
/// by name reads where what it takes copies or is a `&mut`. Cost: O(its
/// operands).
pub fn inst_takings(body: &Body, types: &Types<Concrete>, inst: &Inst) -> SmallVec<[Operand; 4]> {
    todo!("each operand by the module doc's rules, in operand order")
}

/// What `terminator` does with the values it names apart from its edges'
/// arguments: a condition or a tag is read, a `For` moves an array it
/// traverses and reads any other source, and a `Return` passes its value
/// out by value. An edge's argument is taken `by_value`.
pub fn terminator_takings(body: &Body, types: &Types<Concrete>, terminator: &Terminator) -> SmallVec<[Named; 2]> {
    todo!("each named value by the rules above")
}
