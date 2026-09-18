//! Indexing a slice (RFC-0047): the one element access the machine does
//! without knowing a container.
//!
//! The slice register holds the `Large` an `AsSlice` boxed: a pointer into
//! the container's `Vec<Value>` and a length. Reading element `i` is one
//! dependent load and one compare — no call, no layout.

use acvus_extern::Elements;
use acvus_mir::ir::IndexMode;

use crate::code::{Flow, Op, OpFn};
use crate::machine::Machine;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

type Run = Elements<AcvusRuntime>;

/// The text Rust's own slice index gives; a test pins it byte for byte.
fn out_of_bounds(len: usize, index: u64) -> String {
    format!("index out of bounds: the len is {len} but the index is {index}")
}

/// The run a slice value names.
///
/// The vtable check is a `debug_assert!`, as `Value::is_array`'s is: the
/// MIR type checker gives this operand `Ref(_, Slice(T))`, and an
/// `AsSlice`'s boxed `Elements` is the only value of that type.
#[inline]
fn run(slice: &Value) -> &Run {
    debug_assert_eq!(
        slice.vtable().type_id,
        std::any::TypeId::of::<Run>(),
        "index: {slice:?} is not a slice"
    );
    // SAFETY: the value was erased from an `Elements` by `Slice::erase`.
    unsafe { slice.peek::<Run>() }
}

/// The element position `index` names, checked against `run` where the
/// instruction carries no proof of the bound.
#[inline]
fn position<const CHECKED: bool>(run: &Run, index: u64) -> usize {
    if CHECKED && index >= run.len() as u64 {
        panic!("{}", out_of_bounds(run.len(), index));
    }
    // The bound above, or the interval pass's proof (RFC-0047 §7), puts
    // `index` below a length, which is a `usize`.
    index as usize
}

/// Element `index` of the run `slice` names.
///
/// # Safety
/// The container the slice borrows is live and unmoved, which the loan the
/// slice holds keeps true for as long as the slice value exists
/// (RFC-0018).
#[inline]
unsafe fn element<'a, const CHECKED: bool>(slice: &Value, index: u64) -> &'a Value {
    let run = run(slice);
    let at = position::<CHECKED>(run, index);
    // SAFETY: the caller's contract, and `position` put `at` within the run.
    unsafe { run.at(at) }
}

pub fn index_copy<const CHECKED: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let index = machine.reg(op.c).bits();
    // SAFETY: the slice holds its container's loan.
    let value = unsafe { element::<CHECKED>(machine.reg(op.b), index) }.copy_word();
    machine.define(op.a, value);
    Flow::Next
}

pub fn index_ref<const CHECKED: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let index = machine.reg(op.c).bits();
    // SAFETY: as `index_copy`.
    let reference = Value::reference(unsafe { element::<CHECKED>(machine.reg(op.b), index) });
    machine.define(op.a, reference);
    Flow::Next
}

pub fn index_set<const CHECKED: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let index = machine.reg(op.b).bits();
    let value = machine.use_val(op.c);
    let run = run(machine.reg(op.a));
    let at = position::<CHECKED>(run, index);
    // SAFETY: the operand is a `&mut [T]`, so its run is named once here,
    // and `position` put `at` within it.
    let slot = unsafe { run.at_mut(at) };
    *slot = value;
    Flow::Next
}

pub fn checked(mode: IndexMode) -> OpFn {
    match mode {
        IndexMode::Copy => index_copy::<true>,
        IndexMode::Ref => index_ref::<true>,
    }
}

/// The same handler without the bound check (RFC-0047 §7). Nothing in
/// `prepare` reaches it: the MIR holds no unchecked instruction, and until
/// the interval pass carries its own proof the only way to run one is for a
/// probe to substitute it into a prepared body.
#[cfg(any(test, feature = "probe"))]
pub fn unchecked(mode: IndexMode) -> OpFn {
    match mode {
        IndexMode::Copy => index_copy::<false>,
        IndexMode::Ref => index_ref::<false>,
    }
}
