//! Composite constructors: array, object, tuple.

use rustc_hash::FxHashMap;

use crate::code::{Flow, Op};
use crate::machine::Machine;
use crate::ops::payload;
use crate::value::Value;

pub fn make_array(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let slots = payload!(machine, op, Slots);
    let items: Vec<Value> = slots.iter().map(|slot| machine.use_val(*slot)).collect();
    machine.set(op.a, Value::array(items));
    Flow::Next
}

pub fn make_tuple(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let slots = payload!(machine, op, Slots);
    let items: Vec<Value> = slots.iter().map(|slot| machine.use_val(*slot)).collect();
    machine.set(op.a, Value::tuple(items));
    Flow::Next
}

pub fn make_object(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let fields = payload!(machine, op, Fields);
    let object: FxHashMap<_, Value> = fields
        .iter()
        .map(|field| (field.key, machine.use_val(field.slot)))
        .collect();
    machine.set(op.a, Value::object(object));
    Flow::Next
}
