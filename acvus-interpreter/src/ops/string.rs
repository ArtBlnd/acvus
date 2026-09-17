//! Strings: a template's output, equality, and the clone the language's
//! `StringClone` instruction is (RFC-0020).

use crate::code::{Flow, Op};
use crate::machine::Machine;
use crate::ops::payload;
use crate::value::Value;

/// `THROUGH` is what the preparation read from the source's type: a
/// `&String` is read through its reference, a `String` in place.
pub fn clone_string<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = machine.reg(op.b);
    // SAFETY: the type checker admits only a live `String` or `&String` here.
    let text = unsafe {
        if THROUGH {
            source.target().as_str()
        } else {
            source.as_str()
        }
    };
    let value = Value::string(text.to_string());
    machine.set(op.a, value);
    Flow::Next
}

pub fn string_eq(machine: &mut Machine<'_>, op: &Op) -> Flow {
    // SAFETY: the type checker admits only live `&String`s here.
    let equal =
        unsafe { machine.reg(op.b).target().as_str() == machine.reg(op.c).target().as_str() };
    machine.set(op.a, Value::bool_(equal));
    Flow::Next
}

pub fn concat(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let parts = payload!(machine, op, Parts);
    let mut out = String::new();
    for part in parts {
        if part.through_reference {
            // SAFETY: the type checker admits only a live `&String` here.
            let target = unsafe { machine.reg(part.slot).target() };
            // SAFETY: the type checker admits only a string here.
            out.push_str(unsafe { target.as_str() });
        } else {
            let text = machine.use_val(part.slot);
            // SAFETY: the type checker admits only a string here.
            out.push_str(unsafe { text.as_str() });
        }
    }
    machine.set(op.a, Value::string(out));
    Flow::Next
}
