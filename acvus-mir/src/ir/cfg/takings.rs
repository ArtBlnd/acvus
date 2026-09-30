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
//! - A place reached through a reference is taken as a place by name is:
//!   what a `Take` of it takes is moved out of the place the reference
//!   lends, whole or in part, unless it copies or is a `&mut` (ownership
//!   §2, §4). The reference itself is read.
//!
//! - A context is a place by name that is no value of the body: a `Take`,
//!   an `Assign` or a `Ref` of it has no operand for it, and
//!   `place_taking` states how its place is taken (context §3).
//!
//! A `Take`'s `taken_out` is not read: lower makes no take-out.

use smallvec::SmallVec;

use crate::ir::cfg::{Body, Inst, Op, Terminator};
use crate::ir::parts;
use crate::ir::{Callee, ForSource, Intrinsic, RefTarget, ValueId};
use crate::ty::{Calls, Concrete, Mutability, Ty, TyKind, Types, copies};

#[cfg(test)]
mod tests;

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
/// does, rather than passing a value. `lent` is, for the reference a
/// `Take`, a `Ref` or an `Assign` reaches its place through, how that place
/// is taken; `None` for every other operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub value: ValueId,
    pub taken: Taken,
    pub names_place: bool,
    pub lent: Option<Taken>,
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
    match copies(types, ty) || is_mut_ref(types, ty) {
        true => Taken::Read,
        false => Taken::Moved,
    }
}

fn is_mut_ref(types: &Types<Concrete>, ty: Ty) -> bool {
    matches!(types.kind(ty), TyKind::Ref(Mutability::Mut, _))
}

/// Every operand of `inst` but the order token, in the order
/// `Inst::operands` gives them, with how each is taken. A `Take` of a
/// place, by name or through a reference, reads where what it takes copies
/// or is a `&mut`. Cost: O(its operands).
pub fn inst_takings(body: &Body, types: &Types<Concrete>, inst: &Inst) -> SmallVec<[Operand; 4]> {
    let ty = |value: ValueId| body.ty_of(value);
    let copying = |value: ValueId| copies(types, ty(value));
    let taken = |value: ValueId, taken: Taken| Operand { value, taken, names_place: false, lent: None };
    let passed = |value: ValueId| taken(value, by_value(types, ty(value)));
    let read = |value: ValueId| taken(value, Taken::Read);
    let result_copies = || inst.result().is_some_and(copying);
    let extracted = |src: ValueId| match copying(src) || result_copies() {
        true => read(src),
        false => taken(src, Taken::PartMoved),
    };
    let place = |target: &RefTarget, taken: Taken| match target {
        RefTarget::Var(named) | RefTarget::Param(named) => Some(Operand { value: *named, taken, names_place: true, lent: None }),
        RefTarget::Through(reference) => Some(Operand { lent: Some(taken), ..read(*reference) }),
        RefTarget::Context(_) => None,
    };
    let callee = |callee: &Callee| match callee {
        Callee::Indirect(function) => match types.kind(ty(*function)) {
            TyKind::Fn { calls: Calls::Once, .. } => Some(taken(*function, Taken::Moved)),
            TyKind::Fn { calls: Calls::Many, .. } => Some(read(*function)),
            other => panic!("an indirect callee {function:?} is a function value, not {other:?}"),
        },
        Callee::Direct(_) | Callee::Extern { .. } => None,
    };
    let mut operands: SmallVec<[Operand; 4]> = SmallVec::new();
    let taking = || place_taking(body, types, inst).expect("a place operation takes its place");
    match inst.op() {
        Op::Const { .. }
        | Op::ConstStr { .. }
        | Op::ArrayBegin { .. }
        | Op::Undef => {}
        Op::StringConcat { parts } => operands.extend(parts.iter().map(|part| passed(*part))),
        Op::StringAppend { target, part } => operands.extend([read(*target), passed(*part)]),
        Op::StringEq { a, b } => operands.extend([read(*a), read(*b)]),
        Op::StringClone { src } => operands.push(read(*src)),
        Op::Ref { target, .. } | Op::Take { target, .. } => operands.extend(place(target, taking())),
        Op::Assign { target, value, .. } => {
            operands.extend(place(target, taking()));
            operands.push(passed(*value));
        }
        Op::AsSlice { container: lent, .. } | Op::ArrayView { src: lent, .. } => operands.push(read(*lent)),
        Op::Index { slice, index, .. } => operands.extend([read(*slice), read(*index)]),
        Op::IndexSet { slice, index, value, .. } => operands.extend([read(*slice), read(*index), passed(*value)]),
        Op::FieldGet { object: src, .. }
        | Op::ObjectGet { object: src, .. }
        | Op::TupleIndex { tuple: src, .. }
        | Op::ArrayIndex { array: src, .. } => operands.push(extracted(*src)),
        Op::UnwrapVariant { src } => operands.push(match copying(*src) || result_copies() {
            true => read(*src),
            false => taken(*src, Taken::Moved),
        }),
        Op::FieldSet { object, value, .. } => {
            let object = if copying(*object) { read(*object) } else { taken(*object, Taken::PartMoved) };
            operands.extend([object, passed(*value)]);
        }
        Op::BinOp { left, right, .. } | Op::Check { left, right, .. } => operands.extend([read(*left), read(*right)]),
        Op::UnaryOp { operand, .. } => operands.push(read(*operand)),
        Op::Cast { src, .. } => operands.push(read(*src)),
        Op::Unmark { src } => operands.push(passed(*src)),
        Op::CheckSteps { from, step, count } => operands.extend([read(*from), read(*step), read(*count)]),
        Op::FunctionCall { callee: called, args, .. } | Op::Spawn { callee: called, args, .. } => {
            operands.extend(callee(called));
            operands.extend(args.iter().map(|arg| passed(*arg)));
        }
        Op::TakeEffect { src } => operands.push(taken(*src, Taken::Moved)),
        Op::Merge { orders } => operands.extend(orders.iter().map(|order| read(*order))),
        Op::ArrayPush { array, value } => operands.extend([passed(*array), passed(*value)]),
        Op::MakeObject { fields } => operands.extend(fields.iter().map(|(_, field)| passed(*field))),
        Op::MakeTuple { elements } => operands.extend(elements.iter().map(|element| passed(*element))),
        // `Gather` moves the structure of references into the reference it
        // is; `ArrayMap` takes its array as a call takes an argument, and
        // calls `f` as an indirect callee is called.
        Op::Intrinsic { intrinsic: Intrinsic::Gather, args } => operands.extend(args.iter().map(|arg| passed(*arg))),
        Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args } => {
            let [arrays @ .., function] = &args[..] else { panic!("`ArrayMap` takes `[a_1, …, a_k, f]`") };
            operands.extend(arrays.iter().map(|array| passed(*array)));
            operands.extend(callee(&Callee::Indirect(*function)));
        }
        Op::Intrinsic { intrinsic: Intrinsic::StringClone, .. } => panic!("`StringClone` is no `Op::Intrinsic`: `Intrinsic` states no arguments for it"),
        Op::MakeVariant { payload, .. } => operands.extend(payload.iter().map(|payload| passed(*payload))),
        Op::MakeClosure { captures, .. } => operands.extend(captures.iter().map(|capture| passed(*capture))),
        // Kept at some signatures over the same environment, a closure is
        // read where every signature may be called many times, and taken
        // where one is called once (functions §3).
        Op::Specialize { closure, .. } => operands.push(match types.kind(ty(*closure)) {
            TyKind::Closures(signatures) => {
                let once = signatures.iter().any(|signature| matches!(types.kind(*signature), TyKind::Fn { calls: Calls::Once, .. }));
                match once {
                    true => taken(*closure, Taken::Moved),
                    false => read(*closure),
                }
            }
            other => panic!("a specialized value is a closure of signatures, not {other:?}"),
        }),
        Op::TestLiteral { src, .. } | Op::TestObjectKey { src, .. } | Op::TestVariant { src, .. } | Op::SliceLen { src } => {
            operands.push(read(*src))
        }
        Op::Drop { src } => operands.push(taken(*src, Taken::Moved)),
    }
    assert!(
        operands.iter().map(|operand| operand.value).eq(inst.operands().into_iter().take(operands.len())),
        "the takings of {:?} follow its operands in order",
        inst.op()
    );
    operands
}

/// How the place a `Take`, a `Ref` or an `Assign` names is taken, by name
/// or through a reference; `None` for any other operation. A `Take` reads
/// where what it takes copies or is a `&mut`, and otherwise moves the
/// place, whole or in part; a `Ref` reads it; an `Assign` of a part that
/// does not copy cuts it, and any other reads it. Cost: O(1), and the
/// size of the type taken for a `Take`.
pub fn place_taking(body: &Body, types: &Types<Concrete>, inst: &Inst) -> Option<Taken> {
    let copying = |value: ValueId| copies(types, body.ty_of(value));
    Some(match inst.op() {
        Op::Ref { .. } => Taken::Read,
        Op::Take { path, .. } => {
            let what_it_takes = inst.result().expect("a `Take` yields what it takes");
            if copying(what_it_takes) || is_mut_ref(types, body.ty_of(what_it_takes)) {
                Taken::Read
            } else if parts::owned_region(path).is_empty() {
                Taken::Moved
            } else {
                Taken::PartMoved
            }
        }
        Op::Assign { path, value, .. } => match !parts::owned_region(path).is_empty() && !copying(*value) {
            true => Taken::PartMoved,
            false => Taken::Read,
        },
        _ => return None,
    })
}

/// What `terminator` does with the values it names apart from its edges'
/// arguments: a condition or a tag is read, a `For` moves an array it
/// traverses and reads any other source, and a `Return` passes its value
/// out by value. An edge's argument is taken `by_value`.
pub fn terminator_takings(body: &Body, types: &Types<Concrete>, terminator: &Terminator) -> SmallVec<[Named; 2]> {
    let read = |value: ValueId| Named { value, taken: Taken::Read };
    match terminator {
        Terminator::Jump { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => SmallVec::new(),
        Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } | Terminator::While { cond, .. } => {
            smallvec::smallvec![read(*cond)]
        }
        Terminator::Switch { tag, .. } => smallvec::smallvec![read(*tag)],
        Terminator::For { source, .. } => match source {
            ForSource::Array(array) => smallvec::smallvec![Named { value: *array, taken: Taken::Moved }],
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => smallvec::smallvec![read(*slice)],
            ForSource::Range { at, hi } => smallvec::smallvec![read(*at), read(*hi)],
        },
        Terminator::Return { value, .. } => {
            smallvec::smallvec![Named { value: *value, taken: by_value(types, body.ty_of(*value)) }]
        }
    }
}
