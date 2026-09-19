//! A borrowed aggregate crosses as its projection (RFC-0050 rule 6).
//!
//! A handler that reads an object the caller lent declares the projection
//! `#[derive(TyArg)]` emits beside the struct — `SRef<'a>` / `SMut<'a>`, one
//! field per declared field, each a borrow of the value in the object's own
//! storage — and never `&S`. Nothing here allocates.
//!
//! The brief for this work put `Ref<'a>` and `Mut<'a>` on `Cross`, and they
//! are on `Borrowed` instead. `Cross` is parameterized by the runtime, and
//! `SRef<'a>` is a struct with one lifetime and no runtime parameter, so the
//! association that names it in that struct's own definition cannot come from
//! a runtime-parameterized trait. `Borrowed` carries the types, `Project`
//! builds a field's borrow and does carry the runtime, and `Projected` is
//! what a handler's signature names.

use std::marker::PhantomData;

use crate::handler::Arg;
use crate::obj::{FieldAt, Obj, One};
use crate::owned::Owned;
use crate::runtime::Runtime;

pub trait Borrowed {
    type Ref<'a>
    where
        Self: 'a;
    type Mut<'a>
    where
        Self: 'a;
}

/// Building that borrow over the one `Value` a field occupies.
///
/// Obligation across artifacts: that a field is one value whatever its type
/// is the interpreter's. `prepare/runs.rs::Layout::lowerable` refuses any
/// object with a nested aggregate field, so such a field keeps its own
/// `Large` on the heap and occupies one value like every other; were an
/// object with an inlined nested run to become lowerable, a field would span
/// several and this trait would have to take a run rather than a value.
pub trait Project<Rt>: Borrowed
where
    Rt: Runtime,
{
    /// # Safety
    /// `value` holds what `Self`'s crossing wrote, and the storage it names
    /// is live for `'a`.
    unsafe fn project<'a>(rt: &'a Rt, value: &'a Rt::Value) -> Self::Ref<'a>;

    /// # Safety
    /// As `project`, and the storage is exclusively named for `'a`.
    unsafe fn project_mut<'a>(rt: &'a Rt, value: &'a mut Rt::Value) -> Self::Mut<'a>;
}

/// A projection type a handler names in its signature.
///
/// The glue's marker holds `Self` at `'static` because a marker type is
/// `'static`, and `At<'a>` is the same projection at the call's own lifetime.
/// The derive writes one impl for `SRef<'_>`, which reads through a shared
/// borrow, and one for `SMut<'_>`, which reads through an exclusive one, so
/// the choice between the two is the projection's own: the macro sees a type
/// with a lifetime argument and needs to know nothing further about it.
pub trait Projected<Rt>: Sized
where
    Rt: Runtime,
{
    type At<'a>;

    /// # Safety
    /// `reference` names a live object storage holding what the owner's
    /// crossing wrote, and it is exclusively named for a `Mut` projection.
    unsafe fn of<'a>(rt: &'a Rt, reference: &'a Rt::Value) -> Self::At<'a>;
}

/// The fields of an object, by the position each field's name holds in the
/// object's own shape.
///
/// This resolves a name per field per call, and RFC-0050 rule 6 says a
/// projection does no name lookup at call time. Rule 6's answer is that
/// `prepare` resolves each field against the settled type and the operation
/// carries the offsets to the glue; no extern call form carries such a table
/// today, since `prepare.rs::extern_call` builds its operation out of
/// registers and a window and nothing else. The position cannot instead be
/// the field table's own index, because rule 6 admits a projection naming a
/// subset of the object's fields and the two orders then differ.
pub struct Fields<'a, Rt>
where
    Rt: Runtime,
{
    obj: &'a Obj<Owned<Rt>>,
    rt: &'a Rt,
}

pub struct FieldsMut<'a, Rt>
where
    Rt: Runtime,
{
    obj: &'a mut Obj<Owned<Rt>>,
    rt: &'a Rt,
}

/// The position `name` holds in `obj`.
///
/// # Panics
/// The object has no such field, and the checker does not rule that out for
/// every argument. Measured in
/// `acvus-mir-test/tests/projection_parameter.rs`: a value of a declared
/// struct that lacks the field is refused, and an object **literal** that
/// lacks it is admitted with its type unchanged, because `ObjectTy::meet`
/// joins a `Written` field set with an `AtLeast` one by their union. So this
/// panic has a reachable caller, and closing it is a change to that join.
fn position_of<Rt>(rt: &Rt, obj: &Obj<Owned<Rt>>, name: &str) -> FieldAt
where
    Rt: Runtime,
{
    let Some(at) = obj.shape.at(rt.symbol(name)) else {
        panic!(
            "an object crossing into a projection lacks the field `{name}`: an object literal \
             that lacks a field the projection names is admitted by the checker, and the \
             projection cannot borrow what is not there (RFC-0050 rule 6)"
        )
    };
    at
}

impl<'a, Rt> Fields<'a, Rt>
where
    Rt: Runtime,
{
    pub fn of(rt: &'a Rt, obj: &'a Obj<Owned<Rt>>) -> Self {
        Fields { obj, rt }
    }

    pub fn at(&self, name: &str) -> FieldAt {
        position_of(self.rt, self.obj, name)
    }

    pub fn field(&self, at: FieldAt) -> &'a Rt::Value {
        &self.obj.values[at.index()]
    }

    pub fn runtime(&self) -> &'a Rt {
        self.rt
    }
}

impl<'a, Rt> FieldsMut<'a, Rt>
where
    Rt: Runtime,
{
    pub fn of(rt: &'a Rt, obj: &'a mut Obj<Owned<Rt>>) -> Self {
        FieldsMut { obj, rt }
    }

    pub fn at(&self, name: &str) -> FieldAt {
        position_of(self.rt, self.obj, name)
    }

    pub fn runtime(&self) -> &'a Rt {
        self.rt
    }

    /// # Panics
    /// Two of `ats` are equal, or one is past the object's width.
    /// `ObjectShape` holds each name once, so two equal positions would mean
    /// one name answering for two of the projection's fields.
    pub fn disjoint<const N: usize>(self, ats: [FieldAt; N]) -> [&'a mut Rt::Value; N] {
        let at = ats.map(FieldAt::index);
        let width = self.obj.values.len();
        let Ok(fields) = self.obj.values.get_disjoint_mut(at) else {
            panic!(
                "a projection named {at:?} of an object {width} fields wide, which are not {N} \
                 distinct positions"
            )
        };
        fields.map(|owned| &mut **owned)
    }
}

/// The object a reference the caller lent names.
///
/// # Safety
/// `reference` names a live storage holding what an aggregate's crossing
/// wrote, live for `'a`.
pub unsafe fn object_of<'a, Rt>(rt: &'a Rt, reference: &'a Rt::Value) -> &'a Obj<Owned<Rt>>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { rt.deref::<Obj<Owned<Rt>>>(reference) }
}

/// As `object_of`, for an exclusive projection.
///
/// # Safety
/// As `object_of`, and the storage is exclusively named for `'a`.
#[allow(clippy::mut_from_ref)]
pub unsafe fn object_of_mut<'a, Rt>(rt: &'a Rt, reference: &'a Rt::Value) -> &'a mut Obj<Owned<Rt>>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract, exclusively.
    unsafe { rt.deref_mut::<Obj<Owned<Rt>>>(reference) }
}

/// The object a nested aggregate field holds, which is that field's own
/// `Large`.
///
/// # Safety
/// `value` is what the nested aggregate's crossing wrote, live for `'a`.
pub unsafe fn object_in<'a, Rt>(rt: &'a Rt, value: &'a Rt::Value) -> &'a Obj<Owned<Rt>>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { rt.value_as_ref::<Obj<Owned<Rt>>>(value) }
}

/// As `object_in`, exclusively.
///
/// # Safety
/// As `object_in`, and `value` is exclusively named for `'a`.
pub unsafe fn object_in_mut<'a, Rt>(rt: &'a Rt, value: &'a mut Rt::Value) -> &'a mut Obj<Owned<Rt>>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract, exclusively.
    unsafe { rt.value_as_mut::<Obj<Owned<Rt>>>(value) }
}

impl<T> Borrowed for Option<T>
where
    T: Borrowed,
{
    type Ref<'a>
        = Option<<T as Borrowed>::Ref<'a>>
    where
        Self: 'a;
    type Mut<'a>
        = Option<<T as Borrowed>::Mut<'a>>
    where
        Self: 'a;
}

impl<T, Rt> Project<Rt> for Option<T>
where
    T: Project<Rt> + crate::Borrowable<Rt>,
    Rt: Runtime,
{
    unsafe fn project<'a>(rt: &'a Rt, value: &'a Rt::Value) -> Self::Ref<'a> {
        // SAFETY: the caller's contract, and the `Borrowable` bound above.
        let payload = unsafe { rt.some_at(value) }?;
        // SAFETY: the payload lies where the option does (RFC-0039).
        Some(unsafe { <T as Project<Rt>>::project(rt, payload) })
    }

    unsafe fn project_mut<'a>(rt: &'a Rt, value: &'a mut Rt::Value) -> Self::Mut<'a> {
        // SAFETY: as `project`, with the caller's exclusive loan.
        let payload = unsafe { rt.some_at_mut(value) }?;
        // SAFETY: as `project`.
        Some(unsafe { <T as Project<Rt>>::project_mut(rt, payload) })
    }
}

/// A parameter declared as a projection: the reference the caller lent is the
/// argument, and the projection is built inside the glue from it.
pub struct ByProjection<P>(PhantomData<fn() -> P>);

impl<'a, P, Rt> Arg<'a, Rt> for ByProjection<P>
where
    P: Projected<Rt> + 'static,
    Rt: Runtime,
{
    type Out = <P as Projected<Rt>>::At<'a>;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> Self::Out {
        // SAFETY: the caller's contract: `run[0]` is this parameter's own
        // value, a reference to a live object storage, exclusively named
        // where the projection is a `Mut` (RFC-0018).
        unsafe { <P as Projected<Rt>>::of(rt, &run[0]) }
    }
}

/// Why a `&S` in a handler's signature does not compile, and what to write
/// instead. The derive names this in the `where` clause of the `Borrowable`
/// impl it emits for an aggregate, and it has no impl anywhere.
///
/// Obligation across artifacts: the message below is read back by
/// `acvus-extern-macro/tests/compile_fail/borrowed_aggregate.stderr`, so a
/// change to the wording moves that golden.
#[diagnostic::on_unimplemented(
    message = "a borrowed aggregate crosses as its projection: write `{Self}Ref<'_>`",
    label = "this parameter borrows an aggregate",
    note = "`#[derive(TyArg)]` emits `{Self}Ref<'_>` and `{Self}Mut<'_>` beside `{Self}`, one field per declared field, each a borrow of the value in the object's own storage. Write the projection in the signature instead of `&{Self}`.",
    note = "`{Self}` by value is admitted as well, and materializes the fields."
)]
pub trait BorrowedWhole<Rt>
where
    Rt: Runtime,
{
}

/// `Borrowed` and `Project` for a type the runtime stores as itself, whose
/// borrow is a Rust reference into the field's own value.
#[macro_export]
macro_rules! borrowed_as_self {
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*)?> $crate::Borrowed for $t {
            type Ref<'__a> = &'__a Self where Self: '__a;
            type Mut<'__a> = &'__a mut Self where Self: '__a;
        }

        impl<$($($g)*,)? __Rt> $crate::Project<__Rt> for $t
        where
            __Rt: $crate::Runtime,
            Self: $crate::Stored<__Rt>,
        {
            unsafe fn project<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a <__Rt as $crate::Runtime>::Value,
            ) -> &'__a Self {
                // SAFETY: the caller's contract, and a `Stored` type is the
                // type its own value was erased from.
                unsafe { __rt.value_as_ref::<Self>(__value) }
            }

            unsafe fn project_mut<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a mut <__Rt as $crate::Runtime>::Value,
            ) -> &'__a mut Self {
                // SAFETY: as `project`, with the caller's exclusive loan.
                unsafe { __rt.value_as_mut::<Self>(__value) }
            }
        }
    };
}
