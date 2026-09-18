//! Storage: references, reads, assignments, contexts and field access
//! (RFC-0018, RFC-0024, RFC-0025).
//!
//! `THROUGH`, `CLONE`, `LARGE` and the `Segment` type parameter are the four
//! facts `prepare` reads off the types at every storage instruction; nothing
//! in this file re-derives them, so a wrong one is a defect in `prepare`.

use acvus_extern::{Owned, Release};
use acvus_utils::{Astr, Interner};

use std::marker::PhantomData;
use std::mem;

use crate::code::{BlockId, Off, Op, Step, successor};
use crate::machine::{Frame, Machine};
use crate::ops::arith::Unary;
use crate::ops::variant::scrutinee;
use crate::regs::Regs;
use crate::value::{Kind, Place, PlaceMut, Value};

// -- Segments ---------------------------------------------------------

/// One resolved path segment as a type, so that a path of exactly one step
/// reaches its place with no branch.
pub trait Segment: Send + Sync + 'static {
    fn at<'v>(&self, value: &'v Value, interner: &Interner) -> Place<'v>;
    fn at_mut<'v>(&self, value: &'v mut Value, interner: &Interner) -> PlaceMut<'v>;
}

pub struct Field(pub Astr);

impl Segment for Field {
    #[inline]
    fn at<'v>(&self, value: &'v Value, interner: &Interner) -> Place<'v> {
        Place::At(field(value, self.0, interner))
    }

    #[inline]
    fn at_mut<'v>(&self, value: &'v mut Value, interner: &Interner) -> PlaceMut<'v> {
        PlaceMut::At(field_mut(value, self.0, interner))
    }
}

/// An array and a tuple hold their elements in two different Rust types, and
/// `ARRAY` is which one the preparation read from the type the step stands
/// on. `code::Step::Index` does not record it, which is why the boxed walk
/// below has to ask the value and this form does not.
pub struct Index<const ARRAY: bool>(pub usize);

impl<const ARRAY: bool> Segment for Index<ARRAY> {
    #[inline]
    fn at<'v>(&self, value: &'v Value, _: &Interner) -> Place<'v> {
        // SAFETY (both arms): the preparation read the shape off the type.
        Place::At(match ARRAY {
            true => unsafe { &value.as_array()[self.0] },
            false => unsafe { &value.as_tuple()[self.0] },
        })
    }

    #[inline]
    fn at_mut<'v>(&self, value: &'v mut Value, _: &Interner) -> PlaceMut<'v> {
        // SAFETY (both arms): the preparation read the shape off the type.
        PlaceMut::At(match ARRAY {
            true => unsafe { &mut value.as_array_mut().0[self.0] },
            false => unsafe { &mut value.as_tuple_mut().0[self.0] },
        })
    }
}

pub struct OptionPayload;

impl Segment for OptionPayload {
    #[inline]
    fn at<'v>(&self, value: &'v Value, _: &Interner) -> Place<'v> {
        value.option_payload().expect(PAYLOAD_OF_NONE)
    }

    #[inline]
    fn at_mut<'v>(&self, value: &'v mut Value, _: &Interner) -> PlaceMut<'v> {
        if let Some(depth) = depth_payload(value) {
            return PlaceMut::Depth(depth);
        }
        PlaceMut::At(value.option_payload_mut().expect(PAYLOAD_OF_NONE))
    }
}

pub struct ResultPayload;

impl Segment for ResultPayload {
    #[inline]
    fn at<'v>(&self, value: &'v Value, _: &Interner) -> Place<'v> {
        // SAFETY: the preparation read `Result` from the type.
        let (Ok(payload) | Err(payload)) = unsafe { value.as_result() };
        Place::At(payload)
    }

    #[inline]
    fn at_mut<'v>(&self, value: &'v mut Value, _: &Interner) -> PlaceMut<'v> {
        // SAFETY: the preparation read `Result` from the type.
        let (Ok(payload) | Err(payload)) = unsafe { value.as_result_mut() };
        PlaceMut::At(payload)
    }
}

pub struct VariantPayload;

impl Segment for VariantPayload {
    #[inline]
    fn at<'v>(&self, value: &'v Value, _: &Interner) -> Place<'v> {
        // SAFETY: the preparation read an enum from the type.
        let held = unsafe { value.as_variant() }
            .payload
            .as_deref()
            .expect(PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE);
        Place::At(held)
    }

    #[inline]
    fn at_mut<'v>(&self, value: &'v mut Value, _: &Interner) -> PlaceMut<'v> {
        // SAFETY: the preparation read an enum from the type.
        let held = unsafe { value.as_variant_mut() }
            .payload
            .as_deref_mut()
            .expect(PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE);
        PlaceMut::At(held)
    }
}

const PAYLOAD_OF_NONE: &str = "a payload path on None";
const PAYLOAD_OF_A_TAG_THAT_CARRIES_NONE: &str = "a payload path names a variant that carries one";

fn field<'v>(value: &'v Value, f: Astr, interner: &Interner) -> &'v Value {
    assert!(value.is_object(), "field load on non-object: {value:?}");
    // SAFETY: is_object checked the vtable id.
    unsafe { value.as_object() }
        .get(&f)
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(f)))
}

fn field_mut<'v>(value: &'v mut Value, f: Astr, interner: &Interner) -> &'v mut Value {
    assert!(value.is_object(), "field access on non-object: {value:?}");
    // SAFETY: is_object checked the vtable id.
    unsafe { value.as_object_mut() }
        .get_mut(&f)
        .unwrap_or_else(|| panic!("missing field {}", interner.resolve(f)))
}

// -- The path walk ----------------------------------------------------

/// A path of more than one step, walked step by step.
///
/// The `match` per step is the one RFC-0052 leaves standing, and it is a
/// decision not to build rather than an omission: unrolling a path into a
/// type would instantiate this whole family once per path shape a program
/// writes, to save one predicted jump on a walk that already pays a
/// dependent load per step. A path of exactly one step never arrives here —
/// it is a `Segment` the preparation names.
pub fn walk<'v>(value: &'v Value, steps: &[Step], interner: &Interner) -> Place<'v> {
    let mut at = Place::At(value);
    for step in steps {
        at = match at {
            Place::At(v) => segment(v, step, interner),
            Place::Depth(v) => depth_step(&v, step),
        };
    }
    at
}

fn segment<'v>(value: &'v Value, step: &Step, interner: &Interner) -> Place<'v> {
    match step {
        Step::Field(f) => Field(*f).at(value, interner),
        Step::Index(i) => match value.is_array() {
            true => Index::<true>(*i).at(value, interner),
            false => Index::<false>(*i).at(value, interner),
        },
        Step::OptionPayload => OptionPayload.at(value, interner),
        Step::ResultPayload => ResultPayload.at(value, interner),
        Step::VariantPayload => VariantPayload.at(value, interner),
    }
}

pub fn walk_mut<'v>(value: &'v mut Value, steps: &[Step], interner: &Interner) -> PlaceMut<'v> {
    let mut at = PlaceMut::At(value);
    for step in steps {
        at = match at {
            PlaceMut::At(v) => segment_mut(v, step, interner),
            PlaceMut::Depth(v) => match depth_step(&v, step) {
                Place::At(_) => unreachable!("a depth step lands on a depth"),
                Place::Depth(v) => PlaceMut::Depth(v),
            },
        };
    }
    at
}

fn segment_mut<'v>(value: &'v mut Value, step: &Step, interner: &Interner) -> PlaceMut<'v> {
    match step {
        Step::Field(f) => Field(*f).at_mut(value, interner),
        Step::Index(i) => match value.is_array() {
            true => Index::<true>(*i).at_mut(value, interner),
            false => Index::<false>(*i).at_mut(value, interner),
        },
        Step::OptionPayload => OptionPayload.at_mut(value, interner),
        Step::ResultPayload => ResultPayload.at_mut(value, interner),
        Step::VariantPayload => VariantPayload.at_mut(value, interner),
    }
}

/// A step under a `None`: the depth word is all there is, so the only step it
/// admits is the next `Some` it stands for.
fn depth_step<'v>(value: &Value, step: &Step) -> Place<'v> {
    let Step::OptionPayload = step else {
        panic!("{step:?} on {value:?}")
    };
    Place::Depth(depth_payload(value).expect("a depth step lands on a depth"))
}

/// `None` where the payload has a place of its own; the depth word where it
/// has not (RFC-0022).
fn depth_payload(value: &Value) -> Option<Value> {
    match value.option_payload() {
        Some(Place::Depth(v)) => Some(v),
        Some(Place::At(_)) => None,
        None => panic!("{PAYLOAD_OF_NONE}"),
    }
}

// -- Reading and writing a place --------------------------------------

/// RFC-0026: a `String` is cloned out of the storage it is read from, and
/// every other value is copied.
#[inline]
fn read<const CLONE: bool>(at: &Value) -> Value {
    if CLONE {
        // SAFETY: the preparation read `String` from the destination's type.
        return Value::string(unsafe { at.as_str() }.to_string());
    }
    debug_assert_ne!(
        at.kind(),
        Kind::Large,
        "a read leaves the storage owning what it holds"
    );
    *at
}

/// A `Some` whose payload is a `None` has no storage of its own, and the
/// depth word the walk built is then the whole of what is read (RFC-0022).
#[inline]
fn read_place<const CLONE: bool>(place: Place<'_>) -> Value {
    match place {
        Place::At(at) => read::<CLONE>(at),
        Place::Depth(v) => v,
    }
}

/// RFC-0018: a part read out of a storage the frame owns *moves* — the part
/// is left `Undef`, so the storage no longer owns what the destination now
/// does. The depth word of a `Some(None)` has no storage to empty.
#[inline]
fn move_place(place: PlaceMut<'_>) -> Value {
    match place {
        PlaceMut::At(at) => mem::replace(at, Value::UNDEF),
        PlaceMut::Depth(v) => v,
    }
}

/// The word a `*r` reads, as a function of the reference alone: what
/// `TakeThrough` does to a register, and what a fused run's tail does to the
/// `Value` its last call returned without one.
pub fn deref_word<const CLONE: bool>(reference: &Value) -> Value {
    read::<CLONE>(scrutinee::<true>(reference))
}

// -- How a read leaves the place it read ------------------------------
//
// Decision (RFC-0052 §2): moving a part out through a borrow was not built.
// A borrow does not own what it names, so `Moved` carries no `THROUGH`
// parameter and that fourth combination has no name.

/// The mode a read of a place runs in: which place it reaches, what the
/// place keeps, and where the destination's frame is left owning a `Large`.
pub trait Reads: Send + Sync + 'static {
    fn at<S>(slot: &mut Value, step: &S, interner: &Interner) -> Value
    where
        S: Segment;

    fn walked(slot: &mut Value, steps: &[Step], interner: &Interner) -> Value;

    /// The destination, marked exactly where this mode hands it a `Large`.
    fn define(regs: &mut Regs, dst: Off, value: Value);
}

/// A word (or a reference) copied out of a place the read does not disturb.
/// The place keeps whatever it holds, so what is copied owns nothing.
pub struct Copied<const THROUGH: bool>;

impl<const THROUGH: bool> Reads for Copied<THROUGH> {
    #[inline]
    fn at<S>(slot: &mut Value, step: &S, interner: &Interner) -> Value
    where
        S: Segment,
    {
        read_place::<false>(step.at(scrutinee::<THROUGH>(slot), interner))
    }

    #[inline]
    fn walked(slot: &mut Value, steps: &[Step], interner: &Interner) -> Value {
        read_place::<false>(walk(scrutinee::<THROUGH>(slot), steps, interner))
    }

    #[inline]
    fn define(regs: &mut Regs, dst: Off, value: Value) {
        regs.define::<false>(dst, value);
    }
}

/// RFC-0026: a `String` read out of a place is cloned, and the place keeps
/// its own.
pub struct Cloned<const THROUGH: bool>;

impl<const THROUGH: bool> Reads for Cloned<THROUGH> {
    #[inline]
    fn at<S>(slot: &mut Value, step: &S, interner: &Interner) -> Value
    where
        S: Segment,
    {
        read_place::<true>(step.at(scrutinee::<THROUGH>(slot), interner))
    }

    #[inline]
    fn walked(slot: &mut Value, steps: &[Step], interner: &Interner) -> Value {
        read_place::<true>(walk(scrutinee::<THROUGH>(slot), steps, interner))
    }

    #[inline]
    fn define(regs: &mut Regs, dst: Off, value: Value) {
        regs.define::<true>(dst, value);
    }
}

/// A `Large` part moved out of a storage the frame owns: the part is left
/// `Undef` and the destination takes the ownership with it.
pub struct Moved;

impl Reads for Moved {
    #[inline]
    fn at<S>(slot: &mut Value, step: &S, interner: &Interner) -> Value
    where
        S: Segment,
    {
        move_place(step.at_mut(slot, interner))
    }

    #[inline]
    fn walked(slot: &mut Value, steps: &[Step], interner: &Interner) -> Value {
        move_place(walk_mut(slot, steps, interner))
    }

    #[inline]
    fn define(regs: &mut Regs, dst: Off, value: Value) {
        regs.define::<true>(dst, value);
    }
}

#[inline]
fn reference_to(place: Place<'_>) -> Value {
    match place {
        Place::At(at) => Value::reference(at),
        Place::Depth(v) => v,
    }
}

const NO_PLACE_UNDER_A_NONE: &str =
    "an assignment through the payload of a Some(None): a None has no storage";

#[inline]
fn place_mut(place: PlaceMut<'_>) -> &mut Value {
    match place {
        PlaceMut::At(at) => at,
        PlaceMut::Depth(_) => panic!("{NO_PLACE_UNDER_A_NONE}"),
    }
}

/// RFC-0045: the value a write replaces is released.
#[inline]
fn overwrite<const LARGE: bool>(at: &mut Value, value: Value) {
    let replaced = *at;
    *at = value;
    if LARGE {
        replaced.release();
    }
}

#[inline(always)]
fn write_base<const THROUGH: bool>(slot: &mut Value) -> &mut Value {
    if !THROUGH {
        return slot;
    }
    // SAFETY: the type checker admits only a live `&mut` here.
    unsafe { slot.target_mut() }
}

// -- References -------------------------------------------------------

pub struct MakeRef<const THROUGH: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for MakeRef<THROUGH> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let reference = Value::reference(scrutinee::<THROUGH>(regs.peek(self.slots.src)));
        regs.define::<false>(self.slots.dst, reference);
        self.next.run(m, r0)
    }
}

pub struct MakeRefStep<S, const THROUGH: bool>
where
    S: Segment,
{
    pub slots: Unary,
    pub step: S,
    pub next: Box<dyn Op>,
}

impl<S, const THROUGH: bool> Op for MakeRefStep<S, THROUGH>
where
    S: Segment,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let base = scrutinee::<THROUGH>(regs.peek(self.slots.src));
        let reference = reference_to(self.step.at(base, interner));
        regs.define::<false>(self.slots.dst, reference);
        self.next.run(m, r0)
    }
}

pub struct MakeRefPath<const THROUGH: bool> {
    pub slots: Unary,
    pub steps: Box<[Step]>,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for MakeRefPath<THROUGH> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let base = scrutinee::<THROUGH>(regs.peek(self.slots.src));
        let reference = reference_to(walk(base, &self.steps, interner));
        regs.define::<false>(self.slots.dst, reference);
        self.next.run(m, r0)
    }
}

// -- Reads ------------------------------------------------------------

/// The `String` form of this read is `string::CloneString<false>`, the same
/// operation under the name the clone already had.
pub struct TakeVar<const LARGE: bool> {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for TakeVar<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let value = regs.take::<LARGE>(self.slots.src);
        regs.define::<LARGE>(self.slots.dst, value);
        self.next.run(m, r0)
    }
}

/// The storage the reference names keeps whatever it owns, so this read owns
/// nothing. Its `String` form is `string::CloneString<true>`.
pub struct TakeThrough {
    pub slots: Unary,
    pub next: Box<dyn Op>,
}

impl Op for TakeThrough {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let value = deref_word::<false>(regs.peek(self.slots.src));
        regs.define::<false>(self.slots.dst, value);
        self.next.run(m, r0)
    }
}

pub struct ReadStep<S, M>
where
    S: Segment,
    M: Reads,
{
    pub slots: Unary,
    pub step: S,
    pub mode: PhantomData<M>,
    pub next: Box<dyn Op>,
}

impl<S, M> Op for ReadStep<S, M>
where
    S: Segment,
    M: Reads,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let value = M::at(regs.peek_mut(self.slots.src), &self.step, interner);
        M::define(regs, self.slots.dst, value);
        self.next.run(m, r0)
    }
}

pub struct ReadPath<M>
where
    M: Reads,
{
    pub slots: Unary,
    pub steps: Box<[Step]>,
    pub mode: PhantomData<M>,
    pub next: Box<dyn Op>,
}

impl<M> Op for ReadPath<M>
where
    M: Reads,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let value = M::walked(regs.peek_mut(self.slots.src), &self.steps, interner);
        M::define(regs, self.slots.dst, value);
        self.next.run(m, r0)
    }
}

// -- Assignments ------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Write {
    pub target: Off,
    pub value: Off,
}

pub struct AssignVar<const LARGE: bool> {
    pub slots: Write,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for AssignVar<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let value = regs.take::<LARGE>(self.slots.value);
        regs.assign::<LARGE>(self.slots.target, value);
        self.next.run(m, r0)
    }
}

pub struct AssignThrough<const LARGE: bool> {
    pub slots: Write,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for AssignThrough<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let value = regs.take::<LARGE>(self.slots.value);
        overwrite::<LARGE>(write_base::<true>(regs.peek_mut(self.slots.target)), value);
        self.next.run(m, r0)
    }
}

pub struct AssignStep<S, const THROUGH: bool, const LARGE: bool>
where
    S: Segment,
{
    pub slots: Write,
    pub step: S,
    pub next: Box<dyn Op>,
}

impl<S, const THROUGH: bool, const LARGE: bool> Op for AssignStep<S, THROUGH, LARGE>
where
    S: Segment,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let value = regs.take::<LARGE>(self.slots.value);
        let base = write_base::<THROUGH>(regs.peek_mut(self.slots.target));
        overwrite::<LARGE>(place_mut(self.step.at_mut(base, interner)), value);
        self.next.run(m, r0)
    }
}

pub struct AssignPath<const THROUGH: bool, const LARGE: bool> {
    pub slots: Write,
    pub steps: Box<[Step]>,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool, const LARGE: bool> Op for AssignPath<THROUGH, LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let value = regs.take::<LARGE>(self.slots.value);
        let base = write_base::<THROUGH>(regs.peek_mut(self.slots.target));
        overwrite::<LARGE>(place_mut(walk_mut(base, &self.steps, interner)), value);
        self.next.run(m, r0)
    }
}

// -- Field set --------------------------------------------------------

/// `LARGE` on the operations that take this is what the written field owns,
/// not what the object does: an object is a `Large` whichever field is set.
#[derive(Clone, Copy)]
pub struct Update {
    pub dst: Off,
    pub object: Off,
    pub value: Off,
}

pub struct SetStep<S, const LARGE: bool>
where
    S: Segment,
{
    pub slots: Update,
    pub step: S,
    pub next: Box<dyn Op>,
}

impl<S, const LARGE: bool> Op for SetStep<S, LARGE>
where
    S: Segment,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let mut object = regs.take::<true>(self.slots.object);
        let value = regs.take::<LARGE>(self.slots.value);
        overwrite::<LARGE>(place_mut(self.step.at_mut(&mut object, interner)), value);
        regs.define::<true>(self.slots.dst, object);
        self.next.run(m, r0)
    }
}

pub struct SetPath<const LARGE: bool> {
    pub slots: Update,
    pub steps: Box<[Step]>,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for SetPath<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let Frame { regs, interner } = m.frame();
        let mut object = regs.take::<true>(self.slots.object);
        let value = regs.take::<LARGE>(self.slots.value);
        let at = place_mut(walk_mut(&mut object, &self.steps, interner));
        overwrite::<LARGE>(at, value);
        regs.define::<true>(self.slots.dst, object);
        self.next.run(m, r0)
    }
}

// -- Contexts ---------------------------------------------------------

pub struct Fetch<const LARGE: bool> {
    pub dst: Off,
    pub key: Box<str>,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for Fetch<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let key = &self.key;
        let held = m
            .page
            .take(key)
            .unwrap_or_else(|| panic!("context fetch: '{key}' holds no value"));
        m.regs().define::<LARGE>(self.dst, held.into_value());
        self.next.run(m, r0)
    }
}

pub struct Commit<const LARGE: bool> {
    pub src: Off,
    pub key: Box<str>,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for Commit<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let value = m.regs().take::<LARGE>(self.src);
        m.page.set(&self.key, Owned::from_value(value));
        self.next.run(m, r0)
    }
}
