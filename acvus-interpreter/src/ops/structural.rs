//! `==` and `clone` at a structural type, component by component (RFC-0020).

use acvus_extern::{Ctx, Instance, Owned, core};
use acvus_utils::Astr;

use crate::code::{Exit, Marked, Off, Op, successor};
use crate::machine::Machine;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

type Rt = AcvusRuntime;

pub enum Shape {
    /// A word: its bits, which is `f64` bit equality (RFC-0020).
    Word,
    Text,
    Leaf {
        instance_entry: Value,
    },
    Never,
    Array(Box<Shape>),
    Tuple(Box<[Shape]>),
    /// In `acvus_mir::structural::ordered_components` order, which is each
    /// field's position in the heap object.
    Object(Box<[Shape]>),
    /// A tag that carries no payload has no arm.
    Variant(Box<[VariantArm]>),
    Option(Box<Shape>),
}

pub struct VariantArm {
    pub tag: Astr,
    pub payload: Shape,
}

fn owned(value: &Value) -> &Owned<Rt> {
    // SAFETY: `Owned<Rt>` is `repr(transparent)` over `Value`.
    unsafe { &*(value as *const Value).cast::<Owned<Rt>>() }
}

impl Shape {
    fn arm<'a>(arms: &'a [VariantArm], tag: &Value) -> Option<&'a Shape> {
        // SAFETY: a variant's tag register holds a tag word.
        let tag = unsafe { tag.as_tag() };
        arms.iter()
            .find(|arm| arm.tag == tag)
            .map(|arm| &arm.payload)
    }

    /// # Safety
    /// `a` and `b` are live values of the type this shape was built from.
    unsafe fn equal(&self, ctx: &mut Ctx<'_, Rt>, a: &Value, b: &Value) -> bool {
        // SAFETY for every `as_*` below: the caller's contract names the
        // composite each arm reads.
        unsafe {
            match self {
                Shape::Word => a.bits() == b.bits(),
                Shape::Text => a.as_str() == b.as_str(),
                Shape::Leaf { instance_entry } => {
                    let eq: Instance<core::eq<Owned<Rt>, Rt>, Owned<Rt>, Rt> =
                        acvus_extern::Crossing::new(ctx.rt).instance(*instance_entry);
                    eq.call(ctx, owned(a), (owned(b),))
                }
                Shape::Never => unreachable!("no value of type `!` exists to compare"),
                Shape::Array(element) => {
                    let left = a.as_array();
                    let right = b.as_array();
                    left.len() == right.len()
                        && left
                            .iter()
                            .zip(right)
                            .all(|(x, y)| element.equal(ctx, x, y))
                }
                Shape::Tuple(parts) => parts
                    .iter()
                    .zip(a.as_tuple().iter().zip(b.as_tuple()))
                    .all(|(part, (x, y))| part.equal(ctx, x, y)),
                Shape::Object(fields) => fields
                    .iter()
                    .zip(a.as_object().iter().zip(b.as_object()))
                    .all(|(field, (x, y))| field.equal(ctx, x, y)),
                Shape::Variant(arms) => {
                    let left = a.as_variant();
                    let right = b.as_variant();
                    if left.tag().bits() != right.tag().bits() {
                        return false;
                    }
                    match Shape::arm(arms, left.tag()) {
                        Some(payload) => payload.equal(ctx, left.payload(), right.payload()),
                        None => true,
                    }
                }
                Shape::Option(payload) => match (a.option_payload(), b.option_payload()) {
                    (None, None) => true,
                    (Some(x), Some(y)) => payload.equal(ctx, &x, &y),
                    (None, Some(_)) | (Some(_), None) => false,
                },
            }
        }
    }

    /// # Safety
    /// `value` is a live value of the type this shape was built from.
    unsafe fn copy(&self, ctx: &mut Ctx<'_, Rt>, value: &Value) -> Value {
        let mut copied = |shape: &Shape, part: &Owned<Rt>| {
            // SAFETY: the caller's contract, at the part's own type; `copy`
            // makes a fresh word, which no other holder owns.
            unsafe { Owned::from_value(acvus_extern::Holding::new(), shape.copy(ctx, part)) }
        };
        // SAFETY for every `as_*` below: the caller's contract names the
        // composite each arm reads.
        unsafe {
            match self {
                Shape::Word => *value,
                Shape::Text => Value::string(value.as_str()),
                Shape::Leaf { instance_entry } => {
                    let clone: Instance<core::clone<Owned<Rt>, Rt>, Owned<Rt>, Rt> =
                        acvus_extern::Crossing::new(ctx.rt).instance(*instance_entry);
                    clone.call(ctx, owned(value), ()).into_value(acvus_extern::Holding::new())
                }
                Shape::Never => unreachable!("no value of type `!` exists to copy"),
                Shape::Array(element) => Value::array_with(|| {
                    value
                        .as_array()
                        .iter()
                        .map(|part| copied(element, part))
                        .collect()
                }),
                Shape::Tuple(parts) => Value::tuple_with(|| {
                    parts
                        .iter()
                        .zip(value.as_tuple())
                        .map(|(shape, part)| copied(shape, part))
                        .collect()
                }),
                Shape::Object(fields) => Value::object_with(value.as_shape(), || {
                    fields
                        .iter()
                        .zip(value.as_object())
                        .map(|(shape, part)| copied(shape, part))
                        .collect()
                }),
                Shape::Variant(arms) => {
                    let held = value.as_variant();
                    Value::variant_with(**held.tag(), || match Shape::arm(arms, held.tag()) {
                        Some(shape) => copied(shape, held.payload()),
                        // SAFETY (`from_value`, inside the block above):
                        // `UNDEF` owns nothing.
                        None => Owned::from_value(acvus_extern::Holding::new(), Value::UNDEF),
                    })
                }
                Shape::Option(payload) => match value.option_payload() {
                    None => *value,
                    Some(part) => Value::some(payload.copy(ctx, &part)),
                },
            }
        }
    }
}

pub struct StructuralEq {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub shape: Shape,
    pub next: Box<dyn Op>,
}

impl Op for StructuralEq {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = *regs.peek(self.a);
        let b = *regs.peek(self.b);
        // SAFETY: the checker keeps both referents live over this operation,
        // and `prepare` built the shape from the type both lend.
        let equal = unsafe { self.shape.equal(&mut m.ctx, a.target(), b.target()) };
        m.regs().set_word(self.dst, equal as u64);
        self.next.run(m, r0)
    }
}

pub struct StructuralClone<const OWNS_LARGE: bool> {
    pub dst: Marked,
    pub src: Off,
    pub shape: Shape,
    pub next: Box<dyn Op>,
}

impl<const OWNS_LARGE: bool> Op for StructuralClone<OWNS_LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let src = *m.regs().peek(self.src);
        // SAFETY: the checker keeps the referent live over this operation,
        // and `prepare` built the shape from the type it lends.
        let copy = unsafe { self.shape.copy(&mut m.ctx, src.target()) };
        m.regs().define::<OWNS_LARGE>(self.dst, copy);
        self.next.run(m, r0)
    }
}
