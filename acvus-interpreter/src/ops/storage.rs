//! Storage: references, reads, assignments, contexts, and field access
//! (RFC-0018, RFC-0024, RFC-0025).
//!
//! `CLONE` is what the preparation read from the destination's type: a
//! `String` read out of a storage is cloned, every other value is moved or
//! copied by `Value::use_from`.

use acvus_mir::ir::PathSeg;
use acvus_utils::{Astr, Interner};

use crate::code::{Flow, Op};
use crate::machine::{Frame, Machine};
use crate::ops::payload;
use crate::value::Value;

fn field<'a>(value: &'a Value, f: Astr, interner: &Interner) -> &'a Value {
    assert!(value.is_object(), "field load on non-object: {value:?}");
    // SAFETY: is_object checked the vtable id.
    unsafe { value.as_object() }
        .get(&f)
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(f)))
}

fn field_mut<'a>(value: &'a mut Value, f: Astr, interner: &Interner) -> &'a mut Value {
    assert!(value.is_object(), "field access on non-object: {value:?}");
    // SAFETY: is_object checked the vtable id.
    unsafe { value.as_object_mut() }
        .get_mut(&f)
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(f)))
}

pub fn walk_path<'a>(mut value: &'a Value, path: &[PathSeg], interner: &Interner) -> &'a Value {
    for seg in path {
        value = match seg {
            PathSeg::Field(f) => field(value, *f, interner),
            // SAFETY: the type checker admits an index only on an array or a tuple.
            PathSeg::Index(i) => unsafe {
                if value.is_array() {
                    &value.as_array()[*i]
                } else {
                    &value.as_tuple()[*i]
                }
            },
            // SAFETY: the type checker admits a payload only on an option, a result, or a variant.
            PathSeg::Payload => unsafe {
                if value.is_option() {
                    value.as_option().as_ref()
                } else if value.is_result() {
                    match value.as_result() {
                        Ok(v) | Err(v) => Some(v),
                    }
                } else {
                    value.as_variant().payload.as_deref()
                }
            }
            .expect("a payload path names a variant that carries one"),
        };
    }
    value
}

pub fn walk_path_mut<'a>(
    mut value: &'a mut Value,
    path: &[PathSeg],
    interner: &Interner,
) -> &'a mut Value {
    for seg in path {
        value = match seg {
            PathSeg::Field(f) => field_mut(value, *f, interner),
            // SAFETY: the type checker admits an index only on an array or a tuple.
            PathSeg::Index(i) => unsafe {
                if value.is_array() {
                    &mut value.as_array_mut().0[*i]
                } else {
                    &mut value.as_tuple_mut().0[*i]
                }
            },
            // SAFETY: the type checker admits a payload only on an option, a result, or a variant.
            PathSeg::Payload => unsafe {
                if value.is_option() {
                    value.as_option_mut().as_mut()
                } else if value.is_result() {
                    match value.as_result_mut() {
                        Ok(v) | Err(v) => Some(v),
                    }
                } else {
                    value.as_variant_mut().payload.as_deref_mut()
                }
            }
            .expect("a payload path names a variant that carries one"),
        };
    }
    value
}

/// RFC-0026: a `String` is cloned out of its storage, anything else moves
/// or is copied.
#[inline]
fn read_slot<const CLONE: bool>(slot: &mut Value) -> Value {
    if CLONE {
        // SAFETY: the preparation read `String` from the destination's type.
        Value::string(unsafe { slot.as_str() }.to_string())
    } else {
        Value::use_from(slot)
    }
}

#[inline]
fn read_through<const CLONE: bool>(at: &Value) -> Value {
    if CLONE {
        // SAFETY: the preparation read `String` from the destination's type.
        Value::string(unsafe { at.as_str() }.to_string())
    } else {
        at.copy_word()
    }
}

pub fn ref_var(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let reference = Value::reference(machine.reg(op.b));
    machine.set(op.a, reference);
    Flow::Next
}

pub fn ref_var_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let reference = Value::reference(walk_path(machine.reg(op.b), path, machine.interner()));
    machine.set(op.a, reference);
    Flow::Next
}

pub fn ref_through(machine: &mut Machine<'_>, op: &Op) -> Flow {
    // SAFETY: the type checker admits only a live reference here.
    let base = unsafe { machine.reg(op.b).target() };
    let reference = Value::reference(base);
    machine.set(op.a, reference);
    Flow::Next
}

pub fn ref_through_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    // SAFETY: the type checker admits only a live reference here.
    let base = unsafe { machine.reg(op.b).target() };
    let reference = Value::reference(walk_path(base, path, machine.interner()));
    machine.set(op.a, reference);
    Flow::Next
}

pub fn take_var<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = read_slot::<CLONE>(machine.slot_mut(op.b));
    machine.set(op.a, value);
    Flow::Next
}

/// Where a read puts its value and which register it reads.
pub struct ReadSlots {
    pub dst: u32,
    pub src: u32,
}

impl ReadSlots {
    fn of(op: &Op) -> Self {
        Self {
            dst: op.a,
            src: op.b,
        }
    }
}

#[inline]
pub fn read_at<const CLONE: bool>(
    machine: &mut Machine<'_>,
    slots: ReadSlots,
    path: &[PathSeg],
) -> Flow {
    let Frame { regs, interner } = machine.frame();
    let slot = walk_path_mut(&mut regs[slots.src as usize], path, interner);
    let value = read_slot::<CLONE>(slot);
    machine.set(slots.dst, value);
    Flow::Next
}

/// A read at a path under a storage: `Take` through a path, and `FieldGet`
/// at a path of more than one field.
pub fn read_path<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    read_at::<CLONE>(machine, ReadSlots::of(op), payload!(machine, op, Path))
}

pub fn read_index<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    read_at::<CLONE>(machine, ReadSlots::of(op), &[PathSeg::Index(op.p)])
}

pub fn read_field<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key = *payload!(machine, op, Name);
    read_at::<CLONE>(machine, ReadSlots::of(op), &[PathSeg::Field(key)])
}

pub fn take_through<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    // SAFETY: the type checker admits only a live reference here.
    let at = unsafe { machine.reg(op.b).target() };
    let value = read_through::<CLONE>(at);
    machine.set(op.a, value);
    Flow::Next
}

pub fn take_through_path<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    // SAFETY: the type checker admits only a live reference here.
    let base = unsafe { machine.reg(op.b).target() };
    let at = walk_path(base, path, machine.interner());
    let value = read_through::<CLONE>(at);
    machine.set(op.a, value);
    Flow::Next
}

pub fn assign_var(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = machine.use_val(op.b);
    machine.set(op.a, value);
    Flow::Next
}

pub fn assign_var_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let value = machine.use_val(op.b);
    let Frame { regs, interner } = machine.frame();
    *walk_path_mut(&mut regs[op.a as usize], path, interner) = value;
    Flow::Next
}

pub fn assign_through(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = machine.use_val(op.b);
    let reference = machine.reg(op.a).copy_word();
    // SAFETY: the type checker admits only a live `&mut` here.
    let base = unsafe { reference.target_mut() };
    *base = value;
    Flow::Next
}

pub fn assign_through_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let value = machine.use_val(op.b);
    let reference = machine.reg(op.a).copy_word();
    // SAFETY: the type checker admits only a live `&mut` here.
    let base = unsafe { reference.target_mut() };
    *walk_path_mut(base, path, machine.interner()) = value;
    Flow::Next
}

pub fn field_set(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let mut object = machine.take(op.b);
    let value = machine.use_val(op.c);
    assert!(!path.is_empty(), "FieldSet with an empty path");
    *walk_path_mut(&mut object, path, machine.interner()) = value;
    machine.set(op.a, object);
    Flow::Next
}

pub fn fetch(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key: &str = payload!(machine, op, PageKey);
    let value = machine
        .page
        .take(key)
        .unwrap_or_else(|| panic!("context fetch: '{key}' holds no value"));
    machine.set(op.a, value);
    Flow::Next
}

pub fn commit(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key: &str = payload!(machine, op, PageKey);
    let value = machine.use_val(op.a);
    machine.page.set(key, value);
    Flow::Next
}
