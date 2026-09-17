//! Variants: the language's `Option` and `Result`, and the tagged union
//! every other variant type is.
//!
//! Which of the three a constructor builds is the destination's type as the
//! preparation read it; `Ok` against `Err` and `Some` against `None` are
//! the tag resolved at preparation.

use crate::code::{Flow, Op};
use crate::machine::Machine;
use crate::ops::payload;
use crate::value::{OptionValue, ResultValue, Value, VariantValue};

#[inline]
fn payload_value<const HAS_PAYLOAD: bool>(machine: &mut Machine<'_>, op: &Op) -> Option<Value> {
    HAS_PAYLOAD.then(|| machine.use_val(op.b))
}

pub fn make_option<const HAS_PAYLOAD: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let payload = payload_value::<HAS_PAYLOAD>(machine, op);
    machine.set(op.a, Value::option(payload));
    Flow::Next
}

/// `OK` is the tag: `Ok` carries the value on the left, `Err` on the right.
pub fn make_result<const OK: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let payload = machine.use_val(op.b);
    let result = if OK { Ok(payload) } else { Err(payload) };
    machine.set(op.a, Value::result(result));
    Flow::Next
}

pub fn make_variant<const HAS_PAYLOAD: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let tag = *payload!(machine, op, Name);
    let payload = payload_value::<HAS_PAYLOAD>(machine, op);
    machine.set(op.a, Value::variant(tag, payload));
    Flow::Next
}

/// `THROUGH` is what the preparation read from the source's type. `c` and
/// `d` are what the tag resolved to at preparation: whether it is `Some`,
/// and whether it is `Ok`.
pub fn test_variant<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let tag = *payload!(machine, op, Name);
    let source = machine.reg(op.b);
    let source = if THROUGH {
        // SAFETY: the type checker admits only a live reference here.
        unsafe { source.target() }
    } else {
        source
    };
    // SAFETY: the composite check precedes each read.
    let matches = unsafe {
        if source.is_option() {
            source.as_option().is_some() == (op.c != 0)
        } else if source.is_result() {
            source.as_result().is_ok() == (op.d != 0)
        } else {
            source.is_variant() && source.as_variant().tag == tag
        }
    };
    machine.set(op.a, Value::bool_(matches));
    Flow::Next
}

pub fn unwrap_variant(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = machine.take(op.b);
    // SAFETY: the composite check precedes each materialize.
    let payload = unsafe {
        if source.is_option() {
            source.materialize::<OptionValue>()
        } else if source.is_result() {
            Some(match source.materialize::<ResultValue>() {
                Ok(v) | Err(v) => v,
            })
        } else {
            assert!(
                source.is_variant(),
                "UnwrapVariant on non-variant: {source:?}"
            );
            source.materialize::<VariantValue>().payload.map(|p| *p)
        }
    };
    machine.set(op.a, payload.unwrap_or_else(Value::unit));
    Flow::Next
}
