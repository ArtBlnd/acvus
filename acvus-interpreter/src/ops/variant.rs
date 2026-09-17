//! Variants: the language's `Option` and `Result`, and the tagged union
//! every other variant type is.
//!
//! Which shape a constructor builds and which test a scrutinee takes is the
//! destination's or the source's type as the preparation read it; `Ok`
//! against `Err` and `Some` against `None` are the tag resolved there.

use crate::code::{Flow, Op};
use crate::machine::Machine;
use crate::ops::{payload, storage};
use crate::value::{Kind, ResultValue, Value, VariantValue};

#[inline]
fn payload_value<const HAS_PAYLOAD: bool>(machine: &mut Machine<'_>, op: &Op) -> Option<Value> {
    HAS_PAYLOAD.then(|| machine.use_val(op.b))
}

#[inline]
fn place<'a, const THROUGH: bool>(machine: &'a Machine<'_>, slot: u32) -> &'a Value {
    let source = machine.reg(slot);
    if THROUGH {
        storage::through(source)
    } else {
        source
    }
}

/// A scrutinee is a value the checker gave a variant type and a definite
/// initialization, so neither the kind a `Take` leaves behind nor the SSA
/// initial value of a loop variable can reach a variant test.
#[inline]
fn scrutinee<'a, const THROUGH: bool>(machine: &'a Machine<'_>, slot: u32) -> &'a Value {
    let source = place::<THROUGH>(machine, slot);
    debug_assert!(
        source.kind() != Kind::Empty && source.kind() != Kind::Undef,
        "variant test on {source:?}"
    );
    source
}

pub fn make_option<const HAS_PAYLOAD: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = match payload_value::<HAS_PAYLOAD>(machine, op) {
        Some(payload) => Value::some(payload),
        None => Value::NONE,
    };
    machine.set(op.a, value);
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

/// `THROUGH` is what the preparation read from the source's type. `c` is
/// what the tag resolved to at preparation: whether it is `Some`.
pub fn test_option<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let is_some = !scrutinee::<THROUGH>(machine, op.b).is_none();
    machine.set(op.a, Value::bool_(is_some == (op.c != 0)));
    Flow::Next
}

/// `d` is what the tag resolved to at preparation: whether it is `Ok`.
pub fn test_result<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = scrutinee::<THROUGH>(machine, op.b);
    // SAFETY: the preparation read `Result` from the source's type.
    let is_ok = unsafe { source.as_result() }.is_ok();
    machine.set(op.a, Value::bool_(is_ok == (op.d != 0)));
    Flow::Next
}

pub fn test_variant<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let tag = *payload!(machine, op, Name);
    let source = scrutinee::<THROUGH>(machine, op.b);
    // SAFETY: the preparation read an enum from the source's type.
    let matches = unsafe { source.as_variant() }.tag == tag;
    machine.set(op.a, Value::bool_(matches));
    Flow::Next
}

pub fn unwrap_option(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = machine.take(op.b);
    machine.set(op.a, Value::some_payload(source));
    Flow::Next
}

pub fn unwrap_result(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = machine.take(op.b);
    // SAFETY: the preparation read `Result` from the source's type.
    let (Ok(payload) | Err(payload)) = unsafe { source.materialize::<ResultValue>() };
    machine.set(op.a, payload);
    Flow::Next
}

pub fn unwrap_variant(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let source = machine.take(op.b);
    // SAFETY: the preparation read an enum from the source's type.
    let payload = unsafe { source.materialize::<VariantValue>() }.payload;
    machine.set(op.a, payload.map_or_else(Value::unit, |p| *p));
    Flow::Next
}
