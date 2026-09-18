//! The decision tree's tests, and the indexed read of an array element.
//!
//! `THROUGH` is what the preparation read from the source's type: a
//! reference is read through, a value in place.

use crate::code::{Flow, Op, Step};
use crate::machine::Machine;
use crate::ops::arith::Int;
use crate::ops::payload;
use crate::ops::storage::{ReadSlots, read_at};
use crate::value::Value;

#[inline]
fn place<'a, const THROUGH: bool>(machine: &'a Machine<'_>, slot: u32) -> &'a Value {
    let value = machine.reg(slot);
    if THROUGH {
        // SAFETY: the type checker admits only a live reference here.
        unsafe { value.target() }
    } else {
        value
    }
}

pub fn test_int<T>(machine: &mut Machine<'_>, op: &Op) -> Flow
where
    T: Int,
{
    let want = *payload!(machine, op, Wide);
    let matches = T::read(machine.reg(op.b).bits()).wide() == want;
    machine.define(op.a, Value::bool_(matches));
    Flow::Next
}

pub fn test_float<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let matches = place::<THROUGH>(machine, op.b).as_float() == f64::from_bits(op.p as u64);
    machine.define(op.a, Value::bool_(matches));
    Flow::Next
}

pub fn test_bool<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let matches = place::<THROUGH>(machine, op.b).as_bool() == (op.p != 0);
    machine.define(op.a, Value::bool_(matches));
    Flow::Next
}

pub fn test_string<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let want = payload!(machine, op, Text);
    // SAFETY: the type checker matches a string literal against a string.
    let matches = unsafe { place::<THROUGH>(machine, op.b).as_str() } == want.as_str();
    machine.define(op.a, Value::bool_(matches));
    Flow::Next
}

/// A unit literal matches the one value of its type.
pub fn test_unit(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.define(op.a, Value::bool_(true));
    Flow::Next
}

pub fn test_object_key<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key = payload!(machine, op, Name);
    let source = place::<THROUGH>(machine, op.b);
    // SAFETY: is_object checked the vtable id.
    let has = source.is_object() && unsafe { source.as_object() }.contains_key(key);
    machine.define(op.a, Value::bool_(has));
    Flow::Next
}
