//! Constants: the literal's word is in the operation, and a literal that
//! is not one word is a `Konst` in the payload.

use crate::code::{Flow, Op};
use crate::machine::Machine;
use crate::ops::arith::Int;
use crate::ops::payload;
use crate::value::Value;

pub fn int<T>(machine: &mut Machine<'_>, op: &Op) -> Flow
where
    T: Int,
{
    machine.set(op.a, Value::inline(T::KIND, op.p as u64));
    Flow::Next
}

pub fn float(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.set(op.a, Value::inline(crate::value::Kind::F64, op.p as u64));
    Flow::Next
}

pub fn boolean(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.set(op.a, Value::bool_(op.p != 0));
    Flow::Next
}

pub fn unit(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.set(op.a, Value::unit());
    Flow::Next
}

pub fn konst(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = payload!(machine, op, Konst).value();
    machine.set(op.a, value);
    Flow::Next
}
