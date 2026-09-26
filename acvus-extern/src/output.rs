//! A `dynamic` extern's result, typed by its call site (RFC-0097 rule 3):
//! the script sees `Option<T>`, and the handler fills an `Output` at the
//! type the site settled for `T`, which it never reads. The first mismatch,
//! second write or missing field seals it, and the script sees `None`.
//!
//! That an `Output` is built by the glue alone and a `Finished` stands for
//! its own call alone is pinned by `acvus-extern-macro`'s compile-fail cases
//! `dynamic_output_built_outside_the_glue` and
//! `dynamic_finished_of_another_call`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::ty::{Ty, matches_poly};
use acvus_utils::Interner;

use crate::crossing::Crossing;
use crate::declared::Declared;
use crate::handler::{Arg, CallSite, Gives, Out, Ret, Takes};
use crate::obj::{Nothing, Obj, ObjectShape, One, OneValue, OptionOf};
use crate::owned::Owned;
use crate::runtime::Runtime;

type Brand<'call> = PhantomData<fn(&'call ()) -> &'call ()>;

enum PlaceType {
    WrittenWhole(Ty),
    Object(ObjectPlace),
}

struct ObjectPlace {
    ty: Ty,
    layout: Arc<ObjectShape>,
    /// `ObjectShape::of` orders the names by their text, so these are sorted
    /// and `field` finds a name by binary search, interning nothing.
    spelled_in_layout: Box<[Box<str>]>,
    fields: Box<[PlaceType]>,
}

impl PlaceType {
    fn of(interner: &Interner, ty: &Ty) -> PlaceType {
        let Ty::Object(object) = ty else {
            return PlaceType::WrittenWhole(ty.clone());
        };
        // Every field set is laid out alike, an `AtLeast` one included: the
        // interpreter's `prepare` lays an object and reads its fields by the
        // settled type's own keys, whatever its field set, and this layout
        // must be the one it reads.
        let layout = ObjectShape::of(interner, object.keys().copied());
        let spelled_in_layout = layout
            .names()
            .iter()
            .map(|name| Box::<str>::from(interner.resolve(*name)))
            .collect();
        let fields = layout
            .names()
            .iter()
            .map(|name| PlaceType::of(interner, &object[name]))
            .collect();
        PlaceType::Object(ObjectPlace {
            ty: ty.clone(),
            layout,
            spelled_in_layout,
            fields,
        })
    }

    fn ty(&self) -> &Ty {
        match self {
            PlaceType::WrittenWhole(ty) => ty,
            PlaceType::Object(object) => &object.ty,
        }
    }
}

#[derive(Clone)]
pub struct OutputSite {
    interner: Interner,
    settled: Arc<PlaceType>,
}

/// The marker `#[extern_fn(dynamic)]` writes for the `Output` parameter.
pub struct ByOutput;

impl<Rt> Arg<Rt> for ByOutput
where
    Rt: Runtime,
{
    type Site = OutputSite;
    type Form = Nothing;

    const ARGUMENTS: usize = 0;
    const LENDS_A_WORD: bool = false;

    /// # Panics
    /// The site carries no result type, or one that is not an `Option`.
    /// `prepare` gives every extern call site its result's type
    /// (`CallSite::returning`), and a `dynamic` declaration's result is
    /// `Option<T>`, so both are refusals of a site built by hand.
    fn site(site: &CallSite<'_, Rt>, _: usize) -> OutputSite {
        let Some(ret) = site.ret else {
            panic!(
                "a `dynamic` extern was sited with no result type; `prepare` gives every extern \
                 call site the type of its result (RFC-0097 rule 3)"
            )
        };
        let Ty::Option(settled) = ret.ty else {
            panic!(
                "a `dynamic` extern's result is typed {:?}, which is no `Option`; its declaration \
                 is `Option<T>` (RFC-0097 rule 3)",
                ret.ty
            )
        };
        OutputSite {
            interner: ret.interner.clone(),
            settled: Arc::new(PlaceType::of(ret.interner, settled)),
        }
    }

    #[inline(always)]
    unsafe fn loan_ended(_: &Rt, _: &[Rt::Value], _: &OutputSite) {}
}

// SAFETY: the output reads no word of the run, only its site's type, and
// crosses a value only at a place whose settled type the value's declaration
// matches; it keeps the capability for the call alone, never handing it out.
unsafe impl<'a, 'w, T, Rt> Takes<'a, 'w, ByOutput, Rt> for Output<'a, T, Rt>
where
    Rt: Runtime,
{
    unsafe fn take(rt: Crossing<'a, Rt>, _: &'a [Rt::Value], site: &'a OutputSite) -> Self {
        Output {
            rt,
            site,
            root: Slot::Empty,
            sealed: false,
            _brand: PhantomData,
            _settled: PhantomData,
        }
    }
}

enum Slot<Rt>
where
    Rt: Runtime,
{
    Empty,
    Whole(Owned<Rt>),
    Object(Box<[Slot<Rt>]>),
}

/// One call's result of a `dynamic` extern, filled at the type its call site
/// settled for `T`.
pub struct Output<'call, T, Rt>
where
    Rt: Runtime,
{
    rt: Crossing<'call, Rt>,
    site: &'call OutputSite,
    root: Slot<Rt>,
    sealed: bool,
    _brand: Brand<'call>,
    _settled: PhantomData<fn() -> T>,
}

/// The whole result, or a field of an object place.
pub struct Place<'o, Rt>
where
    Rt: Runtime,
{
    rt: Crossing<'o, Rt>,
    interner: &'o Interner,
    ty: &'o PlaceType,
    slot: &'o mut Slot<Rt>,
    sealed: &'o mut bool,
}

impl<'o, Rt> Place<'o, Rt>
where
    Rt: Runtime,
{
    /// Fills this place with `value`. A declared type other than the place's
    /// settled one, or a place already filled or entered by `field`, seals
    /// the output, and `value` is dropped uncrossed.
    pub fn write<V>(&mut self, value: V)
    where
        V: Declared + OneValue<Rt>,
    {
        if *self.sealed {
            return;
        }
        let fits = matches!(self.slot, Slot::Empty)
            && matches_poly(self.ty.ty(), &V::declared(self.interner));
        if !fits {
            *self.sealed = true;
            return;
        }
        *self.slot = Slot::Whole(Owned::erased(self.rt, value));
    }

    /// Lends `fill` the field `name` of this object place, which may be
    /// named again to fill more of an object field. A place that is no
    /// object, a name it lacks, or a place filled whole seals the output. A
    /// sealed output calls no `fill`.
    pub fn field<F>(&mut self, name: &str, fill: F)
    where
        F: FnOnce(&mut Place<'_, Rt>),
    {
        if *self.sealed {
            return;
        }
        let PlaceType::Object(object) = self.ty else {
            *self.sealed = true;
            return;
        };
        let Ok(at) = object
            .spelled_in_layout
            .binary_search_by(|spelled| (**spelled).cmp(name))
        else {
            *self.sealed = true;
            return;
        };
        if matches!(self.slot, Slot::Empty) {
            *self.slot = Slot::Object((0..object.fields.len()).map(|_| Slot::Empty).collect());
        }
        let Slot::Object(fields) = &mut *self.slot else {
            *self.sealed = true;
            return;
        };
        fill(&mut Place {
            rt: self.rt,
            interner: self.interner,
            ty: &object.fields[at],
            slot: &mut fields[at],
            sealed: &mut *self.sealed,
        });
    }
}

impl<'call, T, Rt> Output<'call, T, Rt>
where
    Rt: Runtime,
{
    fn root(&mut self) -> Place<'_, Rt> {
        Place {
            rt: self.rt,
            interner: &self.site.interner,
            ty: &self.site.settled,
            slot: &mut self.root,
            sealed: &mut self.sealed,
        }
    }

    pub fn write<V>(&mut self, value: V)
    where
        V: Declared + OneValue<Rt>,
    {
        self.root().write(value)
    }

    pub fn field<F>(&mut self, name: &str, fill: F)
    where
        F: FnOnce(&mut Place<'_, Rt>),
    {
        self.root().field(name, fill)
    }

    /// The value, where nothing sealed the output and every field of every
    /// object place was filled; else the `None` the script sees, every value
    /// filled so far released.
    pub fn finish(self) -> Finished<'call, T, Rt> {
        let built = match self.sealed {
            true => None,
            false => built(self.rt, &self.site.settled, self.root),
        };
        Finished {
            built,
            _brand: PhantomData,
            _settled: PhantomData,
        }
    }
}

fn built<Rt>(rt: Crossing<'_, Rt>, ty: &PlaceType, slot: Slot<Rt>) -> Option<Owned<Rt>>
where
    Rt: Runtime,
{
    match (ty, slot) {
        (_, Slot::Whole(value)) => Some(value),
        (PlaceType::Object(object), Slot::Empty) if object.fields.is_empty() => {
            Some(object_of(rt, object, Box::new([])))
        }
        (PlaceType::Object(object), Slot::Object(fields)) => {
            let values = object
                .fields
                .iter()
                .zip(fields)
                .map(|(ty, slot)| built(rt, ty, slot))
                .collect::<Option<Box<[Owned<Rt>]>>>()?;
            Some(object_of(rt, object, values))
        }
        (PlaceType::WrittenWhole(_) | PlaceType::Object(_), Slot::Empty)
        | (PlaceType::WrittenWhole(_), Slot::Object(_)) => None,
    }
}

fn object_of<Rt>(rt: Crossing<'_, Rt>, object: &ObjectPlace, values: Box<[Owned<Rt>]>) -> Owned<Rt>
where
    Rt: Runtime,
{
    // SAFETY: the language's object is `Obj<Owned<Rt>>` whose shape is its
    // type's field names in layout order (`derive::object::object_in_order`
    // writes the same), and `values` holds one value per name, each crossed
    // at that field's settled type.
    let word = unsafe { rt.erase::<Obj<Owned<Rt>>>(Obj::new(Arc::clone(&object.layout), values)) };
    // SAFETY: `erase` consumed the object and handed back its word, which no
    // other holder owns.
    unsafe { Owned::from_value(rt.holding(), word) }
}

/// What `Output::finish` built, or the sealed failure the script sees as
/// `None`.
pub struct Finished<'call, T, Rt>
where
    Rt: Runtime,
{
    built: Option<Owned<Rt>>,
    _brand: Brand<'call>,
    _settled: PhantomData<fn() -> T>,
}

/// The `Ret` marker `#[extern_fn(dynamic)]` writes for a `Finished` result.
pub struct RetFinished;

impl<Rt> Ret<Rt> for RetFinished
where
    Rt: Runtime,
{
    type Form = OptionOf<One>;
}

// SAFETY: the word written is the one `Output::finish` built at the settled
// payload type of the site's `Option`, and a failure writes nothing; the
// capability is not kept.
unsafe impl<'call, T, Rt> Gives<RetFinished, Rt> for Finished<'call, T, Rt>
where
    Rt: Runtime,
{
    fn give(self, rt: Crossing<'_, Rt>, out: Out<'_, Rt>) -> bool {
        let Some(value) = self.built else {
            return false;
        };
        out[0] = value.into_value(rt.holding());
        true
    }
}
