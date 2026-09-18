//! Storage: references, reads, assignments, contexts, and field access
//! (RFC-0018, RFC-0024, RFC-0025).
//!
//! `CLONE` is what the preparation read from the destination's type: a
//! `String` read out of a storage is cloned, every other value is moved or
//! copied by `Value::use_from`.

use acvus_utils::{Astr, Interner};

use crate::code::{Flow, Op, Step};
use crate::machine::{Frame, Machine};
use crate::ops::payload;
use crate::value::{Kind, Place, PlaceMut, Value};

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

pub fn walk_path<'a>(value: &'a Value, path: &[Step], interner: &Interner) -> Place<'a> {
    let mut at = Place::At(value);
    for seg in path {
        at = match at {
            Place::At(v) => step(v, seg, interner),
            Place::Depth(v) => depth_step(&v, seg),
        };
    }
    at
}

fn step<'a>(value: &'a Value, seg: &Step, interner: &Interner) -> Place<'a> {
    match seg {
        Step::Field(f) => Place::At(field(value, *f, interner)),
        // SAFETY (each arm): the preparation read the shape off the type.
        Step::Index(i) => Place::At(unsafe {
            if value.is_array() {
                &value.as_array()[*i]
            } else {
                &value.as_tuple()[*i]
            }
        }),
        Step::OptionPayload => value.option_payload().expect(PAYLOAD_OF_NONE),
        Step::ResultPayload => Place::At(match unsafe { value.as_result() } {
            Ok(v) | Err(v) => v,
        }),
        Step::VariantPayload => Place::At(
            unsafe { value.as_variant() }
                .payload
                .as_deref()
                .expect(PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE),
        ),
    }
}

/// A step under a `None`: the depth word is all there is, so the only step
/// it admits is the next `Some` it stands for.
fn depth_step<'a>(value: &Value, seg: &Step) -> Place<'a> {
    let Step::OptionPayload = seg else {
        panic!("{seg:?} on {value:?}")
    };
    Place::Depth(depth_payload(value).expect("a depth step lands on a depth"))
}

const PAYLOAD_OF_NONE: &str = "a payload path on None";
const PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE: &str = "a payload path names a variant that carries one";

/// The place a reference-typed slot names. A `Some` whose payload is a
/// `None` has no storage to point at, so the reference to that payload is
/// the `None` itself (RFC-0022).
#[inline]
pub fn through(slot: &Value) -> &Value {
    if slot.kind() == Kind::None {
        return slot;
    }
    // SAFETY: the type checker admits only a live reference here.
    unsafe { slot.target() }
}

/// The value a `Ref` instruction leaves in its destination.
#[inline]
fn reference_to(place: Place<'_>) -> Value {
    match place {
        Place::At(v) => Value::reference(v),
        Place::Depth(v) => v,
    }
}

pub fn walk_path_mut<'a>(value: &'a mut Value, path: &[Step], interner: &Interner) -> PlaceMut<'a> {
    let mut at = PlaceMut::At(value);
    for seg in path {
        at = match at {
            PlaceMut::At(v) => step_mut(v, seg, interner),
            PlaceMut::Depth(v) => match depth_step(&v, seg) {
                Place::At(_) => unreachable!("a depth step lands on a depth"),
                Place::Depth(v) => PlaceMut::Depth(v),
            },
        };
    }
    at
}

fn step_mut<'a>(value: &'a mut Value, seg: &Step, interner: &Interner) -> PlaceMut<'a> {
    match seg {
        Step::Field(f) => PlaceMut::At(field_mut(value, *f, interner)),
        // SAFETY (each arm): the preparation read the shape off the type.
        Step::Index(i) => PlaceMut::At(unsafe {
            if value.is_array() {
                &mut value.as_array_mut().0[*i]
            } else {
                &mut value.as_tuple_mut().0[*i]
            }
        }),
        Step::OptionPayload => {
            if let Some(depth) = depth_payload(value) {
                return PlaceMut::Depth(depth);
            }
            PlaceMut::At(value.option_payload_mut().expect(PAYLOAD_OF_NONE))
        }
        Step::ResultPayload => PlaceMut::At(match unsafe { value.as_result_mut() } {
            Ok(v) | Err(v) => v,
        }),
        Step::VariantPayload => PlaceMut::At(
            unsafe { value.as_variant_mut() }
                .payload
                .as_deref_mut()
                .expect(PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE),
        ),
    }
}

/// The value a `Some`'s payload is where it has no place of its own.
fn depth_payload(value: &Value) -> Option<Value> {
    match value.option_payload() {
        Some(Place::Depth(v)) => Some(v),
        Some(Place::At(_)) => None,
        None => panic!("{PAYLOAD_OF_NONE}"),
    }
}

/// The message a write through a `None` gives: a `Some` whose payload is a
/// `None` owns nothing, so there is no place to write into.
const NO_PLACE_UNDER_A_NONE: &str =
    "an assignment through the payload of a Some(None): a None has no storage";

/// The storage a write names.
#[inline]
fn place_mut(place: PlaceMut<'_>) -> &mut Value {
    match place {
        PlaceMut::At(v) => v,
        PlaceMut::Depth(_) => panic!("{NO_PLACE_UNDER_A_NONE}"),
    }
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
    machine.define(op.a, reference);
    Flow::Next
}

pub fn ref_var_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let reference = reference_to(walk_path(machine.reg(op.b), path, machine.interner()));
    machine.define(op.a, reference);
    Flow::Next
}

pub fn ref_through(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let base = through(machine.reg(op.b));
    let reference = Value::reference(base);
    machine.define(op.a, reference);
    Flow::Next
}

pub fn ref_through_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let base = through(machine.reg(op.b));
    let reference = reference_to(walk_path(base, path, machine.interner()));
    machine.define(op.a, reference);
    Flow::Next
}

pub fn take_var<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = read_slot::<CLONE>(machine.slot_mut(op.b));
    machine.define(op.a, value);
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
    path: &[Step],
) -> Flow {
    let Frame { regs, interner } = machine.frame();
    let value = match walk_path_mut(&mut regs[slots.src as usize], path, interner) {
        PlaceMut::At(slot) => read_slot::<CLONE>(slot),
        PlaceMut::Depth(v) => v,
    };
    machine.define(slots.dst, value);
    Flow::Next
}

/// A read at a path under a storage: `Take` through a path, and `FieldGet`
/// at a path of more than one field.
pub fn read_path<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    read_at::<CLONE>(machine, ReadSlots::of(op), payload!(machine, op, Path))
}

pub fn read_index<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    read_at::<CLONE>(machine, ReadSlots::of(op), &[Step::Index(op.p as usize)])
}

pub fn read_field<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key = *payload!(machine, op, Name);
    read_at::<CLONE>(machine, ReadSlots::of(op), &[Step::Field(key)])
}

/// The word a `*r` reads, as a function of the reference alone: what
/// `take_through` does to a register, and what a fused run's tail does to
/// the `Value` its last call returned without one.
pub fn deref_word<const CLONE: bool>(reference: &Value) -> Value {
    read_through::<CLONE>(through(reference))
}

pub fn take_through<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = deref_word::<CLONE>(machine.reg(op.b));
    machine.define(op.a, value);
    Flow::Next
}

pub fn take_through_path<const CLONE: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let base = through(machine.reg(op.b));
    let value = match walk_path(base, path, machine.interner()) {
        Place::At(at) => read_through::<CLONE>(at),
        Place::Depth(v) => v,
    };
    machine.define(op.a, value);
    Flow::Next
}

pub fn assign_var(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = machine.use_val(op.b);
    machine.assign(op.a, value);
    Flow::Next
}

pub fn assign_var_path(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let value = machine.use_val(op.b);
    let Frame { regs, interner } = machine.frame();
    *place_mut(walk_path_mut(&mut regs[op.a as usize], path, interner)) = value;
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
    *place_mut(walk_path_mut(base, path, machine.interner())) = value;
    Flow::Next
}

pub fn field_set(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let path = payload!(machine, op, Path);
    let mut object = machine.take(op.b);
    let value = machine.use_val(op.c);
    assert!(!path.is_empty(), "FieldSet with an empty path");
    *place_mut(walk_path_mut(&mut object, path, machine.interner())) = value;
    machine.define(op.a, object);
    Flow::Next
}

pub fn fetch(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key: &str = payload!(machine, op, PageKey);
    let value = machine
        .page
        .take(key)
        .unwrap_or_else(|| panic!("context fetch: '{key}' holds no value"));
    machine.define(op.a, value);
    Flow::Next
}

pub fn commit(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let key: &str = payload!(machine, op, PageKey);
    let value = machine.use_val(op.a);
    machine.page.set(key, value);
    Flow::Next
}
