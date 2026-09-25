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

use crate::handler::{Arg, ArgAt, CallSite, Takes};
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

    /// Whether `project_mut` can lend a storage in the runtime's value word,
    /// which a write leaves to `loan_ended` to re-encode: `false` only where
    /// a fact of the type rules it out, as `Borrowable::LENDS_A_WORD`.
    const LENDS_A_WORD: bool;

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

    /// Every borrow `project_mut` made of `value` has ended: each storage it
    /// lent in place — the value itself for a type the runtime stores as
    /// itself, each field's, a payload's — goes to `Runtime::loan_ended`,
    /// found again through the same table. This is where a `&mut i8` a
    /// projection lent has its word re-encoded (RFC-0039 rule 2), and a
    /// projection that lends a storage names it here.
    ///
    /// # Safety
    /// As `project_mut`'s, over the value `project_mut` was handed, and no
    /// borrow it made is live.
    unsafe fn loan_ended(rt: &Rt, value: &mut Rt::Value, table: &Self::Table);
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

/// A projection type a handler names in its signature, at the lifetime `'a`
/// of the borrow it is built over.
///
/// The glue's marker holds `Self` at `'static`, and the handler receives it
/// at the call's lifetime, as every crossing type. The derive writes one impl
/// for `SRef<'_>`, which reads through a shared borrow, and one for
/// `SMut<'_>`, which reads through an exclusive one, so the choice between
/// the two is the projection's own.
pub trait Projected<'a, Rt>: Sized
where
    Rt: Runtime,
{
    type Loan: Loan;
    type Table: Clone + Send + Sync + 'static;

    /// As `Project::LENDS_A_WORD`, for the projection `of` builds.
    const LENDS_A_WORD: bool;

    fn table(at: ArgAt<'_>) -> Self::Table;

    /// # Safety
    /// `reference` names a live object storage holding what the owner's
    /// crossing wrote, and it is exclusively named for a `Mut` projection.
    unsafe fn of(rt: &'a Rt, reference: &'a Rt::Value, table: &Self::Table) -> Self;

    /// The projection `of` built over `reference` has ended: a `Mut`
    /// projection hands each storage it lent to `Runtime::loan_ended`
    /// (`Project::loan_ended` over the owner's storage), a `Shared` one does
    /// nothing.
    ///
    /// # Safety
    /// As `of`'s, over the storage `of` projected, and no borrow the
    /// projection handed out is live.
    unsafe fn loan_ended(rt: &Rt, reference: &Rt::Value, table: &Self::Table);
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
        Ty::Ref(_, inner) => inner.whole(),
        other => Some(other),
    };
    let Some(Ty::Object(obj)) = object else {
        panic!(
            "a projection parameter's argument is typed {:?}, which names no object \
             (RFC-0050 rule 6)",
            at.ty
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
        Ty::Ref(_, inner) => inner.whole(),
        other => Some(other),
    };
    let Some(Ty::Enum { variants, .. }) = declared else {
        panic!(
            "an enum projection parameter's argument is typed {:?}, which names no enum \
             (RFC-0050 rule 6)",
            at.ty
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
        (crate::repr::word_of_tag(key), payload)
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

/// The values of a lent object, as the projection `#[derive(TyArg)]` writes
/// reads them: the derive's `over` and no handler.
#[doc(hidden)]
pub struct Fields<'a, M, Rt>
where
    M: Loan,
    Rt: Runtime,
{
    obj: M::Of<'a, Obj<Owned<Rt>>>,
}

impl<'a, M, Rt> Fields<'a, M, Rt>
where
    M: Loan,
    Rt: Runtime,
{
    /// # Safety
    /// `obj` is the storage a value the checker typed at the projection's
    /// object names, and what is read from it goes only to that
    /// projection's own `Project` crossings.
    pub unsafe fn of(obj: M::Of<'a, Obj<Owned<Rt>>>) -> Self {
        Fields { obj }
    }
}

impl<'a, Rt> Fields<'a, Shared, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `of`'s, at the field `at`.
    pub unsafe fn field(&self, at: FieldAt) -> &'a Rt::Value {
        &self.obj.values[at.index()]
    }
}

impl<'a, Rt> Fields<'a, Mut, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `of`'s, at the fields `ats`, and as `Owned::value_mut`'s for what
    /// is written through each result.
    ///
    /// # Panics
    /// Two of `ats` are equal, or one is past the object's width.
    /// `ObjectShape` holds each name once, so two equal positions would mean
    /// one name answering for two of the projection's fields.
    pub unsafe fn disjoint<const N: usize>(self, ats: [FieldAt; N]) -> [&'a mut Rt::Value; N] {
        let at = ats.map(FieldAt::index);
        let width = self.obj.values.len();
        let Ok(fields) = self.obj.values.get_disjoint_mut(at) else {
            panic!(
                "a projection named {at:?} of an object {width} fields wide, which are not {N} \
                 distinct positions"
            )
        };
        // SAFETY: this function's contract carries `value_mut`'s, and its
        // caller is the derive's projection glue.
        fields.map(|field| unsafe { field.value_mut(crate::Holding::new()) })
    }
}

/// A projected type that is not an option, so that a `Some` of it has a
/// payload storage for `Runtime::some_at` to name; a runtime may give a
/// nested option's payload none.
///
/// The trait is safe and has no parameter rather than being `unsafe`: the
/// orphan rule lets another crate implement it only for a type of its own,
/// which is never `Option`.
pub trait OwnStorage {}

impl<T, E> OwnStorage for Result<T, E> {}

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
    T: Project<Rt> + OwnStorage,
    Rt: Runtime,
{
    type Table = <T as Project<Rt>>::Table;

    const LENDS_A_WORD: bool = <T as Project<Rt>>::LENDS_A_WORD;

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
        // SAFETY: the caller's contract, and the `OwnStorage` bound above:
        // the payload is no option.
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

    unsafe fn loan_ended(rt: &Rt, value: &mut Rt::Value, table: &Self::Table) {
        if !<Self as Project<Rt>>::LENDS_A_WORD {
            return;
        }
        // SAFETY: as `project_mut`'s: the payload `project_mut` lent, where
        // there is one.
        if let Some(payload) = unsafe { rt.some_at_mut(value) } {
            // SAFETY: the caller's contract, over that payload.
            unsafe { <T as Project<Rt>>::loan_ended(rt, payload, table) }
        }
    }
}

impl<T, E> Borrowed for Result<T, E>
where
    T: Borrowed,
    E: Borrowed,
{
    type Ref<'a>
        = Result<<T as Borrowed>::Ref<'a>, <E as Borrowed>::Ref<'a>>
    where
        Self: 'a;
    type Mut<'a>
        = Result<<T as Borrowed>::Mut<'a>, <E as Borrowed>::Mut<'a>>
    where
        Self: 'a;
}

impl<T, E, Rt> Project<Rt> for Result<T, E>
where
    T: Project<Rt>,
    E: Project<Rt>,
    Rt: Runtime,
{
    type Table = (<T as Project<Rt>>::Table, <E as Project<Rt>>::Table);

    const LENDS_A_WORD: bool = <T as Project<Rt>>::LENDS_A_WORD || <E as Project<Rt>>::LENDS_A_WORD;

    /// # Panics
    /// The settled type is not a `Result`. The checker settles a `Result`
    /// field, and a lent value, at the type the Rust `Result` declares.
    fn table(at: ArgAt<'_>) -> Self::Table {
        let Ty::Result(ok, err) = at.ty else {
            panic!("a `Result` projection's value is typed {:?}, which is no `Result`", at.ty)
        };
        (
            <T as Project<Rt>>::table(ArgAt {
                interner: at.interner,
                ty: ok,
            }),
            <E as Project<Rt>>::table(ArgAt {
                interner: at.interner,
                ty: err,
            }),
        )
    }

    unsafe fn project<'a>(rt: &'a Rt, value: &'a Rt::Value, table: &Self::Table) -> Self::Ref<'a> {
        // SAFETY: the caller's contract: `value` holds what the `Result`
        // crossing's `erase` wrote.
        let payload = unsafe { rt.result_at(value) };
        // SAFETY: each side's payload holds what that side's crossing wrote,
        // in storage live for `'a`.
        unsafe {
            payload
                .map(|ok| <T as Project<Rt>>::project(rt, ok, &table.0))
                .map_err(|err| <E as Project<Rt>>::project(rt, err, &table.1))
        }
    }

    unsafe fn project_mut<'a>(
        rt: &'a Rt,
        value: &'a mut Rt::Value,
        table: &Self::Table,
    ) -> Self::Mut<'a> {
        // SAFETY: as `project`, with the caller's exclusive loan.
        let payload = unsafe { rt.result_at_mut(value) };
        // SAFETY: as `project`, exclusively.
        unsafe {
            payload
                .map(|ok| <T as Project<Rt>>::project_mut(rt, ok, &table.0))
                .map_err(|err| <E as Project<Rt>>::project_mut(rt, err, &table.1))
        }
    }

    unsafe fn loan_ended(rt: &Rt, value: &mut Rt::Value, table: &Self::Table) {
        if !<Self as Project<Rt>>::LENDS_A_WORD {
            return;
        }
        // SAFETY: as `project_mut`'s: the payload of the arm `project_mut`
        // lent.
        match unsafe { rt.result_at_mut(value) } {
            // SAFETY: the caller's contract, over that payload.
            Ok(ok) => unsafe { <T as Project<Rt>>::loan_ended(rt, ok, &table.0) },
            // SAFETY: as above.
            Err(err) => unsafe { <E as Project<Rt>>::loan_ended(rt, err, &table.1) },
        }
    }
}

/// A parameter declared as a projection: the reference the caller lent is the
/// argument, and the projection is built inside the glue from it. It stays its
/// own `Arg` because the macro picks a parameter's mode from the Rust type's
/// shape, and a projection is a type with a lifetime argument — neither a
/// value nor a borrow of one.
pub struct ByProjection<P>(PhantomData<fn() -> P>);

impl<P, Rt> Arg<Rt> for ByProjection<P>
where
    P: Projected<'static, Rt> + 'static,
    Rt: Runtime,
{
    type Site = <P as Projected<'static, Rt>>::Table;
    type Form = One;

    const LENDS_A_WORD: bool = <P as Projected<'static, Rt>>::LENDS_A_WORD;

    fn site(site: &CallSite<'_, Rt>, at: usize) -> Self::Site {
        <P as Projected<'static, Rt>>::table(site.args[at])
    }

    /// The projection's own end: each component's storage, found through
    /// the site's table, goes to `Runtime::loan_ended` (`Projected::
    /// loan_ended`).
    #[inline(always)]
    unsafe fn loan_ended(rt: &Rt, run: &[Rt::Value], site: &Self::Site) {
        if <Self as Arg<Rt>>::LENDS_A_WORD {
            // SAFETY: the caller's contract: `run[0]` is the reference `take`
            // built the projection over with this table, and the projection
            // has ended.
            unsafe { <P as Projected<'static, Rt>>::loan_ended(rt, &run[0], site) }
        }
    }
}

// SAFETY: the projection is `Q::of` over this parameter's own word with the
// site's table; the capability lends only its runtime and is not kept.
unsafe impl<'a, 'w, P, Q, Rt> Takes<'a, 'w, ByProjection<P>, Rt> for Q
where
    P: Projected<'static, Rt> + 'static,
    Q: Projected<'a, Rt, Table = <P as Projected<'static, Rt>>::Table> + crate::Within<'a>,
    Rt: Runtime,
{
    unsafe fn take(
        rt: crate::Crossing<'a, Rt>,
        run: &'a [Rt::Value],
        site: &<P as Projected<'static, Rt>>::Table,
    ) -> Q {
        // SAFETY: the caller's contract: `run[0]` is this parameter's own
        // value, a reference to a live object storage, exclusively named
        // where the projection is a `Mut` (RFC-0018).
        unsafe { Q::of(rt.rt(), &run[0], site) }
    }
}

/// A trait that exists to be unimplemented: its absence is the refusal, and
/// the `on_unimplemented` text below is the whole of what it does. The derive
/// names it in the `where` clause of the `Borrowable` impl it emits for an
/// aggregate, so a `&S` in a handler's signature fails to compile with this
/// message and no other.
///
/// It carries `Borrowable`'s two reads, and the derive's impl reads through
/// them: an aggregate's storage is an object, which holds no `S` to read, and
/// a trait with no impl is a body that is never called by type.
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
    /// # Safety
    /// As `Borrowable::deref`.
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self;

    /// # Safety
    /// As `Borrowable::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self;
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

        impl<$($($g)*)?> $crate::OwnStorage for $t {}

        impl<$($($g)*,)? __Rt> $crate::Project<__Rt> for $t
        where
            __Rt: $crate::Runtime,
            Self: $crate::Stored<__Rt>,
        {
            type Table = ();

            // The payload `project_mut` lends.
            const LENDS_A_WORD: bool =
                $crate::repr::may_lie_in_the_word::<<Self as $crate::Stored<__Rt>>::Payload>();

            fn table(_: $crate::ArgAt<'_>) {}

            unsafe fn project<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a <__Rt as $crate::Runtime>::Value,
                _: &(),
            ) -> &'__a Self {
                // SAFETY: the caller's contract, and a `Stored` type's value
                // was erased at its `Payload`.
                <Self as $crate::Stored<__Rt>>::from_payload(unsafe { $crate::Holding::new() }, unsafe {
                    __rt.value_as_ref::<<Self as $crate::Stored<__Rt>>::Payload>(__value)
                })
            }

            unsafe fn project_mut<'__a>(
                __rt: &'__a __Rt,
                __value: &'__a mut <__Rt as $crate::Runtime>::Value,
                _: &(),
            ) -> &'__a mut Self {
                // SAFETY: as `project`, with the caller's exclusive loan.
                <Self as $crate::Stored<__Rt>>::from_payload_mut(unsafe { $crate::Holding::new() }, unsafe {
                    __rt.value_as_mut::<<Self as $crate::Stored<__Rt>>::Payload>(__value)
                })
            }

            unsafe fn loan_ended(
                _: &__Rt,
                __value: &mut <__Rt as $crate::Runtime>::Value,
                _: &(),
            ) {
                if <Self as $crate::Project<__Rt>>::LENDS_A_WORD {
                    // `project_mut` lent the value itself.
                    <__Rt as $crate::Runtime>::loan_ended(__value)
                }
            }
        }
    };
}
