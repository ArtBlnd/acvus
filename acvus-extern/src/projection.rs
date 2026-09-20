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

use acvus_mir::ty::Ty;

use crate::handler::{Arg, ArgAt, Sited};
use crate::loan::{Loan, Mut, Shared};
use crate::obj::{FieldAt, Obj, ObjectShape, One, Variant};
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
    type Table: Clone + Send + Sync + 'static;

    fn table(at: ArgAt<'_>) -> Self::Table;

    /// # Safety
    /// `value` holds what `Self`'s crossing wrote, and the storage it names
    /// is live for `'a`.
    unsafe fn project<'a>(rt: &'a Rt, value: &'a Rt::Value, table: &Self::Table) -> Self::Ref<'a>;

    /// # Safety
    /// As `project`, and the storage is exclusively named for `'a`.
    unsafe fn project_mut<'a>(
        rt: &'a Rt,
        value: &'a mut Rt::Value,
        table: &Self::Table,
    ) -> Self::Mut<'a>;
}

/// Where each field an object projection names sits in the object the caller
/// lent, and what each of those fields' own crossings needs from the site.
#[derive(Clone, Copy)]
pub struct ObjectAt<const K: usize, F> {
    pub at: [FieldAt; K],
    pub fields: F,
}

/// The tag word of each variant an enum projection names, in the enum's
/// declaration order, and what each payload's own crossing needs from the
/// site.
///
/// Obligation across artifacts: a tag word is the interned name's
/// `Astr::bits`, which is what `interpreter::value::Value::tag` writes into a
/// heap variant's tag register and what `interpreter::prepare::runs::Tags::
/// word` writes into a run's. A change to that spelling moves all three.
#[derive(Clone, Copy)]
pub struct VariantAt<const K: usize, P> {
    pub tags: [u64; K],
    pub payloads: P,
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
    type Table: Clone + Send + Sync + 'static;

    fn table(at: ArgAt<'_>) -> Self::Table;

    /// # Safety
    /// `reference` names a live object storage holding what the owner's
    /// crossing wrote, and it is exclusively named for a `Mut` projection.
    unsafe fn of<'a>(rt: &'a Rt, reference: &'a Rt::Value, table: &Self::Table) -> Self::At<'a>;
}

/// The position each field an object projection names holds in the object
/// the caller lent, in the projection's own field order.
///
/// Obligation across artifacts: the positions are read out of `ObjectShape`,
/// whose order the interpreter lays the same object in —
/// `prepare/runs.rs::Layout` for a run and `layout.rs::sorted_fields` for a
/// committed object's canonical bytes. A change to rule 8's order moves all
/// three together.
///
/// The position is not the projection's own field table index, which would
/// cost nothing to compute. Rule 6 admits a projection naming a subset of
/// the object's fields, and the two orders differ as soon as it does.
///
/// # Panics
/// The settled type names no object, or the object lacks a field the
/// projection borrows. `typeck.rs::refused_projection_parameter` refuses
/// such an argument ahead of preparation, and the seven cases that fix it
/// are `acvus-mir-test/tests/projection_parameter.rs`.
pub fn object_fields_at<'a, const K: usize>(
    at: ArgAt<'a>,
    names: [&str; K],
) -> [(FieldAt, ArgAt<'a>); K] {
    let object = match at.ty {
        Ty::Ref(_, inner) => &inner.ty,
        other => other,
    };
    let Ty::Object(obj) = object else {
        panic!(
            "a projection parameter's argument is typed {object:?}, which names no object \
             (RFC-0050 rule 6)"
        )
    };
    let shape = ObjectShape::of(at.interner, obj.keys().copied());
    names.map(|name| {
        let key = at.interner.intern(name);
        let (Some(found), Some(ty)) = (shape.at(key), obj.get(&key)) else {
            panic!(
                "an object crossing into a projection lacks the field `{name}`: a projection \
                 parameter is matched at least, so the checker refuses an argument that lacks \
                 a field the projection borrows (RFC-0050 rule 6)"
            )
        };
        (
            found,
            ArgAt {
                interner: at.interner,
                ty,
            },
        )
    })
}

/// The tag word of each variant an enum projection names, in its own
/// declaration order, beside the settled type of that variant's payload.
///
/// A tag word is the interned name's `Astr::bits` and needs no type to
/// resolve it. The settled type is read all the same, so that a payload's own
/// crossing has a site to build its table from and so that an argument typed
/// as something other than the declared enum is refused here, at `prepare`,
/// rather than at the first call.
///
/// # Panics
/// The settled type names no enum, or the enum lacks a variant the
/// projection names. The checker settles the declared enum's own type on a
/// projection's argument.
pub fn variant_tags_at<'a, const K: usize>(
    at: ArgAt<'a>,
    names: [&str; K],
) -> [(u64, Option<ArgAt<'a>>); K] {
    let declared = match at.ty {
        Ty::Ref(_, inner) => &inner.ty,
        other => other,
    };
    let Ty::Enum { variants, .. } = declared else {
        panic!(
            "an enum projection parameter's argument is typed {declared:?}, which names no enum \
             (RFC-0050 rule 6)"
        )
    };
    names.map(|name| {
        let key = at.interner.intern(name);
        let Some(payload) = variants.get(&key) else {
            panic!(
                "an enum crossing into a projection has no variant `{name}`: the checker settles \
                 the declared enum's own type on the argument (RFC-0050 rule 6)"
            )
        };
        let payload = payload.as_deref().map(|ty| ArgAt {
            interner: at.interner,
            ty,
        });
        (key.bits(), payload)
    })
}

/// # Panics
/// The variant carries no payload at this site, where the projection
/// declares one.
pub fn payload_at<'a>(at: Option<ArgAt<'a>>, name: &str) -> ArgAt<'a> {
    let Some(at) = at else {
        panic!(
            "the variant `{name}` carries no payload at this call site, and the projection \
             borrows one (RFC-0050 rule 6)"
        )
    };
    at
}

/// Where a projection reaches the aggregate it borrows: `Lent` through the
/// reference the caller handed in, `Nested` in the value a field or a payload
/// holds, which is that field's own `Large`.
pub trait Reach {
    type From<'a, M, Rt>
    where
        M: Loan,
        Rt: Runtime;

    /// # Safety
    /// `from` names what `T`'s crossing wrote, live for `'a`, and for a `Mut`
    /// loan it is the only live name of it.
    unsafe fn at<'a, T, M, Rt>(rt: &'a Rt, from: Self::From<'a, M, Rt>) -> M::Of<'a, T>
    where
        T: Send + Sync + 'static,
        M: Loan,
        Rt: Runtime;
}

pub struct Lent;
pub struct Nested;

impl Reach for Lent {
    type From<'a, M, Rt>
        = &'a Rt::Value
    where
        M: Loan,
        Rt: Runtime;

    unsafe fn at<'a, T, M, Rt>(rt: &'a Rt, from: &'a Rt::Value) -> M::Of<'a, T>
    where
        T: Send + Sync + 'static,
        M: Loan,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { M::deref::<T, Rt>(rt, from) }
    }
}

impl Reach for Nested {
    type From<'a, M, Rt>
        = M::Of<'a, Rt::Value>
    where
        M: Loan,
        Rt: Runtime;

    unsafe fn at<'a, T, M, Rt>(rt: &'a Rt, from: M::Of<'a, Rt::Value>) -> M::Of<'a, T>
    where
        T: Send + Sync + 'static,
        M: Loan,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { M::value_as::<T, Rt>(rt, from) }
    }
}

/// The object a projection borrows.
///
/// # Safety
/// As `Reach::at`.
pub unsafe fn object<'a, S, M, Rt>(
    rt: &'a Rt,
    from: S::From<'a, M, Rt>,
) -> M::Of<'a, Obj<Owned<Rt>>>
where
    S: Reach,
    M: Loan,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { S::at::<Obj<Owned<Rt>>, M, Rt>(rt, from) }
}

/// The variant a projection borrows.
///
/// # Safety
/// As `Reach::at`.
pub unsafe fn variant<'a, S, M, Rt>(
    rt: &'a Rt,
    from: S::From<'a, M, Rt>,
) -> M::Of<'a, Variant<Owned<Rt>>>
where
    S: Reach,
    M: Loan,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { S::at::<Variant<Owned<Rt>>, M, Rt>(rt, from) }
}

pub struct Fields<'a, M, Rt>
where
    M: Loan,
    Rt: Runtime,
{
    obj: M::Of<'a, Obj<Owned<Rt>>>,
    rt: &'a Rt,
}

impl<'a, M, Rt> Fields<'a, M, Rt>
where
    M: Loan,
    Rt: Runtime,
{
    pub fn of(rt: &'a Rt, obj: M::Of<'a, Obj<Owned<Rt>>>) -> Self {
        Fields { obj, rt }
    }

    pub fn runtime(&self) -> &'a Rt {
        self.rt
    }
}

impl<'a, Rt> Fields<'a, Shared, Rt>
where
    Rt: Runtime,
{
    pub fn field(&self, at: FieldAt) -> &'a Rt::Value {
        &self.obj.values[at.index()]
    }
}

impl<'a, Rt> Fields<'a, Mut, Rt>
where
    Rt: Runtime,
{
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
    type Table = <T as Project<Rt>>::Table;

    fn table(at: ArgAt<'_>) -> Self::Table {
        let payload = match at.ty {
            Ty::Option(inner) => &**inner,
            other => other,
        };
        <T as Project<Rt>>::table(ArgAt {
            interner: at.interner,
            ty: payload,
        })
    }

    unsafe fn project<'a>(rt: &'a Rt, value: &'a Rt::Value, table: &Self::Table) -> Self::Ref<'a> {
        // SAFETY: the caller's contract, and the `Borrowable` bound above.
        let payload = unsafe { rt.some_at(value) }?;
        // SAFETY: the payload lies where the option does (RFC-0039).
        Some(unsafe { <T as Project<Rt>>::project(rt, payload, table) })
    }

    unsafe fn project_mut<'a>(
        rt: &'a Rt,
        value: &'a mut Rt::Value,
        table: &Self::Table,
    ) -> Self::Mut<'a> {
        // SAFETY: as `project`, with the caller's exclusive loan.
        let payload = unsafe { rt.some_at_mut(value) }?;
        // SAFETY: as `project`.
        Some(unsafe { <T as Project<Rt>>::project_mut(rt, payload, table) })
    }
}

/// A parameter declared as a projection: the reference the caller lent is the
/// argument, and the projection is built inside the glue from it. It stays its
/// own `Arg` because the macro picks a parameter's mode from the Rust type's
/// shape, and a projection is a type with a lifetime argument — neither a
/// value nor a borrow of one.
pub struct ByProjection<P>(PhantomData<fn() -> P>);

impl<P, Rt> Sited<Rt> for ByProjection<P>
where
    P: Projected<Rt> + 'static,
    Rt: Runtime,
{
    type Site = <P as Projected<Rt>>::Table;

    fn site(at: ArgAt<'_>) -> Self::Site {
        <P as Projected<Rt>>::table(at)
    }
}

impl<'a, P, Rt> Arg<'a, Rt> for ByProjection<P>
where
    P: Projected<Rt> + 'static,
    Rt: Runtime,
{
    type Out = <P as Projected<Rt>>::At<'a>;
    type Form = One;

    unsafe fn take<'s>(
        rt: &'a Rt,
        run: &'a [Rt::Value],
        site: &'s <Self as Sited<Rt>>::Site,
    ) -> Self::Out {
        // SAFETY: the caller's contract: `run[0]` is this parameter's own
        // value, a reference to a live object storage, exclusively named
        // where the projection is a `Mut` (RFC-0018).
        unsafe { <P as Projected<Rt>>::of(rt, &run[0], site) }
    }
}

/// A trait that exists to be unimplemented: its absence is the refusal, and
/// the `on_unimplemented` text below is the whole of what it does. The derive
/// names it in the `where` clause of the `Borrowable` impl it emits for an
/// aggregate, so a `&S` in a handler's signature fails to compile with this
/// message and no other.
///
/// Obligation across artifacts: the message below is read back by
/// `acvus-extern-macro/tests/compile_fail/borrowed_aggregate.stderr`, so a
/// change to the wording moves that golden.
#[diagnostic::on_unimplemented(
    message = "a borrowed aggregate crosses as its projection: write `{Self}Ref<'_>`",
    label = "this parameter borrows an aggregate",
    note = "`#[derive(TyArg)] #[projection]` emits `{Self}Ref<'_>` beside `{Self}`, one component per declared field or variant, each a borrow of the value in the aggregate's own storage. The exclusive form is `{Self}Mut<'_>` for a struct and `{Self}Mut<'_, Rt>` for an enum. Write the projection in the signature instead of `&{Self}`.",
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
            type Table = ();

            fn table(_: $crate::ArgAt<'_>) {}

            unsafe fn project<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a <__Rt as $crate::Runtime>::Value,
                _: &(),
            ) -> &'__a Self {
                // SAFETY: the caller's contract, and a `Stored` type is the
                // type its own value was erased from.
                unsafe { __rt.value_as_ref::<Self>(__value) }
            }

            unsafe fn project_mut<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a mut <__Rt as $crate::Runtime>::Value,
                _: &(),
            ) -> &'__a mut Self {
                // SAFETY: as `project`, with the caller's exclusive loan.
                unsafe { __rt.value_as_mut::<Self>(__value) }
            }
        }
    };
}
