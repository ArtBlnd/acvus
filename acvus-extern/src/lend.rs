//! A value lent to a closure, whose parameters cross as a handler's do
//! (RFC-0090 rule 3).
//!
//! A host reads and edits a result or a context the way an extern handler
//! receives its parameters. The closure names each parameter's Rust type, as
//! a handler's signature does, and the parameter crosses through the marker
//! the macro would write for that type and that marker's own `Takes`. The
//! site the marker needs is built from the settled type by `Arg::site`, as
//! at a call. Nothing here reads a value: the run a parameter is taken out
//! of is the one a call hands it, a reference to the lent storage or the view
//! the language's coercion makes of one (RFC-0047 rule 6, RFC-0062 rule 3).
//!
//! The closure is generic over the borrow's lifetime, as a handler body is
//! over its call's (RFC-0079 rule 6), so what it keeps is a copy it makes,
//!
//! ```
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn copied<Rt: Runtime>() {
//!     call::<Rt, _, String, _>(|s: &str| s.to_owned());
//! }
//! ```
//!
//! and nothing it is lent outlives it:
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Ctx, Runtime};
//! fn kept<Rt, Q, F>(f: F)
//! where
//!     Rt: Runtime,
//!     F: Borrows<Rt, Q, &'static str>,
//! {}
//! fn lent<Rt: Runtime>() {
//!     kept::<Rt, _, _>(|s: &str| s);
//! }
//! ```
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn smuggle<Rt: Runtime>() {
//!     let mut kept: Vec<&str> = Vec::new();
//!     call::<Rt, _, _, _>(|s: &str| kept.push(s));
//!     drop(kept);
//! }
//! ```
//!
//! An `Option` and a `Result` are lent as their projections, at the top and
//! nested in each other alike,
//!
//! ```
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn projected<Rt: Runtime>() {
//!     call::<Rt, _, bool, _>(|r: Result<&i64, &String>| r.is_ok());
//!     call::<Rt, _, (), _>(|r: Result<&mut i64, &mut String>| {
//!         if let Ok(n) = r {
//!             *n += 1;
//!         }
//!     });
//!     call::<Rt, _, bool, _>(|o: Option<&i64>| o.is_some());
//!     call::<Rt, _, (), _>(|o: Option<&mut i64>| {
//!         if let Some(n) = o {
//!             *n += 1;
//!         }
//!     });
//!     call::<Rt, _, bool, _>(|o: Option<Result<&i64, &String>>| o.is_some());
//!     call::<Rt, _, bool, _>(|r: Result<Option<&mut i64>, &mut String>| r.is_ok());
//! }
//! ```
//!
//! and neither by reference
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn referenced<Rt: Runtime>() {
//!     call::<Rt, _, bool, _>(|r: &Result<i64, String>| r.is_ok());
//! }
//! ```
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn referenced<Rt: Runtime>() {
//!     call::<Rt, _, bool, _>(|o: &Option<i64>| o.is_some());
//! }
//! ```
//!
//! nor by value:
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn moved<Rt: Runtime>() {
//!     call::<Rt, _, bool, _>(|r: Result<i64, String>| r.is_ok());
//! }
//! ```
//!
//! ```compile_fail
//! # use acvus_extern::{Borrows, Runtime};
//! fn call<Rt, Q, O, F>(_: F) where Rt: Runtime, F: Borrows<Rt, Q, O> {}
//! fn moved<Rt: Runtime>() {
//!     call::<Rt, _, bool, _>(|o: Option<i64>| o.is_some());
//! }
//! ```

use std::marker::PhantomData;

use acvus_mir::ty::{FieldSet, Mutability, PolyTy, Ty, TypeArg, matches_poly};
use acvus_utils::{Interner, QualifiedRef};

use crate::crossing::Crossing;
use crate::ctx::Ctx;
use crate::declared::Declared;
use crate::handler::{Arg, ArgAt, Borrowable, ByRef, CallSite, Gives, Loans, Parameters, RetLent, Takes, Uniform};
use crate::len::Arr;
use crate::loan::{Loan, Mut, Shared};
use crate::erased::Erased;
use crate::obj::{Form, One, Pair};
use crate::projection::{Borrowed, ByProjection, Project, Projected};
use crate::registry::ExternTypeDecl;
use crate::runtime::{Runtime, TypesOnly};
use crate::slice::{BySlice, Slice};
use crate::str::{ByStr, RetStr};
use crate::ty_arg::{PolyVars, TyArg};
use crate::within::Within;

mod sealed {
    use acvus_mir::ty::Ty;

    use crate::crossing::Crossing;
    use crate::runtime::Runtime;

    /// Writing a lent parameter's run: the runtime's, through `lend`, and
    /// nobody else's, so no host or implementer holds it.
    pub trait Lend<Rt>
    where
        Rt: Runtime,
    {
        /// # Safety
        /// `reference` is a reference value naming a live storage settled
        /// at `held`, named by nothing else while the run is used where the
        /// loan is `Mut`; `Lendable::argument(held)` answered `Some`, and
        /// `out` is the marker's `WIDTH` long.
        unsafe fn lend<'a>(
            rt: Crossing<'a, Rt>,
            reference: &'a Rt::Value,
            held: &Ty,
            out: &mut [Rt::Value],
        );
    }

    /// Unnameable outside the crate: a method that takes one can be neither
    /// called nor implemented there.
    pub struct Token(pub(super) ());
}

/// A marker a lent value crosses through: the modes of a handler parameter
/// that borrow the caller's storage (RFC-0023 rule 5). A parameter taken by
/// value is moved into its call, and a lent value is not moved, so
/// `ByValue` is not one of them.
pub trait Lendable<Rt>: Arg<Rt> + sealed::Lend<Rt>
where
    Rt: Runtime,
{
    /// The loan the parameter holds on the storage.
    type Loan: Loan;

    /// The parameter's acvus type, as the macro states it in a declaration:
    /// built at `TypesOnly`, where a part `Erased<Rt, X>` is `X` (RFC-0076
    /// rule 1).
    fn param_ty(interner: &Interner) -> PolyTy;

    /// The argument a borrow of a storage settled at `held` is at this
    /// parameter: `&held`, or the view the language's coercion makes of it
    /// — `&String` as `&str`, `&Vec<T>` and `&Array<T, n>` as `&[T]`. `None`
    /// where the parameter borrows no storage of `held`'s kind.
    fn argument(interner: &Interner, held: &Ty) -> Option<Ty>;
}

fn reference_to(mutability: Mutability, target: Ty) -> Ty {
    Ty::Ref(mutability, Box::new(TypeArg::uniform(target)))
}

fn reference_param(mutability: Mutability, target: PolyTy) -> PolyTy {
    PolyTy::Ref(mutability, Box::new(TypeArg::uniform(target)))
}

/// The acvus type of `X` as a part `Erased<Rt, X>` names it: the type
/// `TyArg` reads at `Erased<TypesOnly, X>`, which is what the macro builds a
/// declaration's parameter at (RFC-0076 rule 1).
fn erased_ty<X>(interner: &Interner) -> PolyTy
where
    Erased<TypesOnly, X>: TyArg,
{
    <Erased<TypesOnly, X> as TyArg>::poly_ty(interner, &PolyVars::empty())
}

impl<T, M, Rt> Lendable<Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt> + Declared,
    M: Loan,
    Rt: Runtime,
{
    type Loan = M;

    fn param_ty(interner: &Interner) -> PolyTy {
        reference_param(M::MUTABILITY, T::declared(interner))
    }

    fn argument(_: &Interner, held: &Ty) -> Option<Ty> {
        Some(reference_to(M::MUTABILITY, held.clone()))
    }
}

impl<T, M, Rt> sealed::Lend<Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt> + Declared,
    M: Loan,
    Rt: Runtime,
{
    unsafe fn lend<'a>(_: Crossing<'a, Rt>, reference: &'a Rt::Value, _: &Ty, out: &mut [Rt::Value]) {
        out[0] = *reference;
    }
}

/// `&Erased<Rt, X>`: a borrow of a storage holding an `X`, read by
/// `as_ref(rt)`.
impl<X, M, Rt> Lendable<Rt> for ByRef<Erased<Rt, X>, M, Uniform>
where
    Erased<TypesOnly, X>: TyArg,
    M: Loan,
    Rt: Runtime,
{
    type Loan = M;

    fn param_ty(interner: &Interner) -> PolyTy {
        reference_param(M::MUTABILITY, erased_ty::<X>(interner))
    }

    fn argument(_: &Interner, held: &Ty) -> Option<Ty> {
        Some(reference_to(M::MUTABILITY, held.clone()))
    }
}

impl<X, M, Rt> sealed::Lend<Rt> for ByRef<Erased<Rt, X>, M, Uniform>
where
    Erased<TypesOnly, X>: TyArg,
    M: Loan,
    Rt: Runtime,
{
    unsafe fn lend<'a>(_: Crossing<'a, Rt>, reference: &'a Rt::Value, _: &Ty, out: &mut [Rt::Value]) {
        out[0] = *reference;
    }
}

impl<P, Rt> Lendable<Rt> for ByProjection<P>
where
    P: Projected<'static, Rt> + TyArg + 'static,
    Rt: Runtime,
{
    type Loan = <P as Projected<'static, Rt>>::Loan;

    fn param_ty(interner: &Interner) -> PolyTy {
        P::poly_ty(interner, &PolyVars::empty())
    }

    fn argument(_: &Interner, held: &Ty) -> Option<Ty> {
        Some(reference_to(
            <<P as Projected<'static, Rt>>::Loan as Loan>::MUTABILITY,
            held.clone(),
        ))
    }
}

impl<P, Rt> sealed::Lend<Rt> for ByProjection<P>
where
    P: Projected<'static, Rt> + TyArg + 'static,
    Rt: Runtime,
{
    unsafe fn lend<'a>(_: Crossing<'a, Rt>, reference: &'a Rt::Value, _: &Ty, out: &mut [Rt::Value]) {
        out[0] = *reference;
    }
}

/// The language's `&str` of a `String`: `core::as_str`'s glue, a `&String`
/// taken as the macro takes one and given back as its `&str`.
impl<Rt> Lendable<Rt> for ByStr
where
    Rt: Runtime,
{
    type Loan = Shared;

    fn param_ty(_: &Interner) -> PolyTy {
        reference_param(Mutability::Shared, PolyTy::Str)
    }

    fn argument(_: &Interner, held: &Ty) -> Option<Ty> {
        matches!(held, Ty::String).then(|| reference_to(Mutability::Shared, Ty::Str))
    }
}

impl<Rt> sealed::Lend<Rt> for ByStr
where
    Rt: Runtime,
{
    unsafe fn lend<'a>(rt: Crossing<'a, Rt>, reference: &'a Rt::Value, _: &Ty, out: &mut [Rt::Value]) {
        // SAFETY: the caller's contract: a live `String` storage.
        let text: &'a String = unsafe {
            <&'a String as Takes<'a, 'a, ByRef<String, Shared, Uniform>, Rt>>::take(
                rt,
                std::slice::from_ref(reference),
                &(),
            )
        };
        <&str as Gives<RetStr, Rt>>::give(text.as_str(), rt, out);
    }
}

/// Where a container the language slices keeps its elements: a `Vec`'s
/// uniform argument or an array's element.
fn slice_element<'t>(interner: &Interner, held: &'t Ty) -> Option<&'t Ty> {
    match held {
        Ty::Array(element, _) => Some(element),
        Ty::UserDefined { id, type_args, .. } if *id == vec_id(interner) => {
            match type_args.as_slice() {
                [TypeArg::Uniform(element)] => Some(element),
                _ => None,
            }
        }
        _ => None,
    }
}

fn vec_id(interner: &Interner) -> QualifiedRef {
    <Vec<()> as ExternTypeDecl>::type_decl(interner).qref
}

/// The language's `&[X]` / `&mut [X]` of a `Vec<X>` or an `Array<X, n>`, whose
/// elements a closure reads as `Erased<Rt, X>`: the container's `as_slice`
/// glue, a `&Vec<Erased<Rt, X>>` taken as the macro takes one and given back
/// as its slice.
fn slice_param<X, M>(interner: &Interner) -> PolyTy
where
    Erased<TypesOnly, X>: TyArg,
    M: Loan,
{
    reference_param(M::MUTABILITY, PolyTy::Slice(Box::new(erased_ty::<X>(interner))))
}

fn slice_argument<M>(interner: &Interner, held: &Ty) -> Option<Ty>
where
    M: Loan,
{
    let element = slice_element(interner, held)?;
    Some(reference_to(M::MUTABILITY, Ty::Slice(Box::new(element.clone()))))
}

impl<X, Rt> Lendable<Rt> for BySlice<Erased<Rt, X>, Shared>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Loan = Shared;

    fn param_ty(interner: &Interner) -> PolyTy {
        slice_param::<X, Shared>(interner)
    }

    fn argument(interner: &Interner, held: &Ty) -> Option<Ty> {
        slice_argument::<Shared>(interner, held)
    }
}

impl<X, Rt> Lendable<Rt> for BySlice<Erased<Rt, X>, Mut>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Loan = Mut;

    fn param_ty(interner: &Interner) -> PolyTy {
        slice_param::<X, Mut>(interner)
    }

    fn argument(interner: &Interner, held: &Ty) -> Option<Ty> {
        slice_argument::<Mut>(interner, held)
    }
}

impl<X, Rt> sealed::Lend<Rt> for BySlice<Erased<Rt, X>, Shared>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    unsafe fn lend<'a>(rt: Crossing<'a, Rt>, reference: &'a Rt::Value, held: &Ty, out: &mut [Rt::Value]) {
        let run = std::slice::from_ref(reference);
        let elements: &'a [Erased<Rt, X>] = match held {
            // SAFETY: the caller's contract: a live array storage.
            Ty::Array(..) => &unsafe {
                <&'a Arr<Erased<Rt, X>, ()> as Takes<
                    'a,
                    'a,
                    ByRef<Arr<Erased<Rt, X>, ()>, Shared, Uniform>,
                    Rt,
                >>::take(rt, run, &())
            }
            .0,
            // SAFETY: the caller's contract: `argument` answered, so a live
            // `Vec` storage.
            _ => unsafe {
                <&'a Vec<Erased<Rt, X>> as Takes<
                    'a,
                    'a,
                    ByRef<Vec<Erased<Rt, X>>, Shared, Uniform>,
                    Rt,
                >>::take(rt, run, &())
            },
        };
        <&[Erased<Rt, X>] as Gives<RetLent<Slice<'static, Erased<Rt, X>, Shared, Rt>>, Rt>>::give(
            elements, rt, out,
        );
    }
}

/// As the shared slice's, exclusively: `as_slice_mut`'s glue.
impl<X, Rt> sealed::Lend<Rt> for BySlice<Erased<Rt, X>, Mut>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    unsafe fn lend<'a>(rt: Crossing<'a, Rt>, reference: &'a Rt::Value, held: &Ty, out: &mut [Rt::Value]) {
        let run = std::slice::from_ref(reference);
        let elements: &'a mut [Erased<Rt, X>] = match held {
            // SAFETY: the caller's contract: a live array storage, named by
            // nothing else.
            Ty::Array(..) => &mut unsafe {
                <&'a mut Arr<Erased<Rt, X>, ()> as Takes<
                    'a,
                    'a,
                    ByRef<Arr<Erased<Rt, X>, ()>, Mut, Uniform>,
                    Rt,
                >>::take(rt, run, &())
            }
            .0,
            // SAFETY: as the array's, over a `Vec` storage.
            _ => unsafe {
                <&'a mut Vec<Erased<Rt, X>> as Takes<
                    'a,
                    'a,
                    ByRef<Vec<Erased<Rt, X>>, Mut, Uniform>,
                    Rt,
                >>::take(rt, run, &())
            },
        };
        <&mut [Erased<Rt, X>] as Gives<RetLent<Slice<'static, Erased<Rt, X>, Mut, Rt>>, Rt>>::give(
            elements, rt, out,
        );
    }
}

/// An `Option` or a `Result` is lent through this and not through `ByRef`:
/// its storage holds no Rust `Option<T>` or `Result<T, E>` for a reference to
/// name.
pub struct ByProjected<T, M>(PhantomData<fn() -> (T, M)>);

impl<T, M, Rt> Arg<Rt> for ByProjected<T, M>
where
    T: Project<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = <T as Project<Rt>>::Table;
    type Form = One;

    const LENDS_A_WORD: bool = M::EXCLUSIVE && <T as Project<Rt>>::LENDS_A_WORD;

    /// # Panics
    /// The argument is not a reference to a whole type. `lend` builds it as
    /// `reference_to` the settled type.
    fn site(site: &CallSite<'_, Rt>, at: usize) -> Self::Site {
        let lent = site.args[at];
        let Some(held) = (match lent.ty {
            Ty::Ref(_, held) => held.whole(),
            _ => None,
        }) else {
            panic!("a lent projection's argument is typed {:?}, which is no reference", lent.ty)
        };
        <T as Project<Rt>>::table(ArgAt {
            interner: lent.interner,
            ty: held,
        })
    }

    /// The projection's own end: each storage it lent — the payload of an
    /// `Option` or of a `Result`'s arm, and whatever that payload's own
    /// projection lent — goes to `Runtime::loan_ended` (`Project::
    /// loan_ended`).
    #[inline(always)]
    unsafe fn loan_ended(rt: &Rt, run: &[Rt::Value], site: &Self::Site) {
        if <Self as Arg<Rt>>::LENDS_A_WORD {
            // SAFETY: the caller's contract: `run[0]` is the reference `take`
            // projected with this table, and the projection over it has ended.
            unsafe { M::projection_ended::<T, Rt>(rt, &run[0], site) }
        }
    }
}

impl<T, M, Rt> Lendable<Rt> for ByProjected<T, M>
where
    T: Project<Rt> + Declared,
    M: Loan,
    Rt: Runtime,
{
    type Loan = M;

    fn param_ty(interner: &Interner) -> PolyTy {
        reference_param(M::MUTABILITY, T::declared(interner))
    }

    fn argument(_: &Interner, held: &Ty) -> Option<Ty> {
        Some(reference_to(M::MUTABILITY, held.clone()))
    }
}

impl<T, M, Rt> sealed::Lend<Rt> for ByProjected<T, M>
where
    T: Project<Rt> + Declared,
    M: Loan,
    Rt: Runtime,
{
    unsafe fn lend<'a>(_: Crossing<'a, Rt>, reference: &'a Rt::Value, _: &Ty, out: &mut [Rt::Value]) {
        out[0] = *reference;
    }
}

// SAFETY: the projection is `M::project` over this parameter's own word, a
// reference to the lent storage, with the table its settled type built; it
// borrows that storage at `'a` and nothing else, and the capability lends
// only its runtime and is not kept.
unsafe impl<'a, 'w, Q, T, M, Rt> Takes<'a, 'w, ByProjected<T, M>, Rt> for Q
where
    T: Project<Rt> + 'a,
    M: Loan<Projection<'a, T> = Q>,
    Rt: Runtime,
{
    unsafe fn take(rt: Crossing<'a, Rt>, run: &'a [Rt::Value], site: &<T as Project<Rt>>::Table) -> Q {
        // SAFETY: the caller's contract: `run[0]` is this parameter's own
        // value, a reference to a live storage holding what `T`'s crossing
        // wrote, exclusively named where the loan is `Mut` (RFC-0018).
        unsafe { M::project::<T, Rt>(rt.rt(), &run[0], site) }
    }
}

/// A safe trait: the part's own type is read back from `Owner` at `Loan`, so
/// an impl that names another owner makes the closure's annotation disagree
/// with what it is called with, and the closure does not compile.
pub trait Projects {
    type Owner: Borrowed + 'static;
    type Loan: Loan;
}

impl<'q, T> Projects for &'q T
where
    T: Borrowed<Ref<'q> = &'q T> + for<'s> Within<'s> + 'static,
{
    type Owner = T;
    type Loan = Shared;
}

impl<'q, T> Projects for &'q mut T
where
    T: Borrowed<Mut<'q> = &'q mut T> + for<'s> Within<'s> + 'static,
{
    type Owner = T;
    type Loan = Mut;
}

impl<A> Projects for Option<A>
where
    A: Projects,
{
    type Owner = Option<A::Owner>;
    type Loan = A::Loan;
}

impl<A, B> Projects for Result<A, B>
where
    A: Projects,
    B: Projects<Loan = A::Loan>,
{
    type Owner = Result<A::Owner, B::Owner>;
    type Loan = A::Loan;
}

impl<A, Rt> Param<Rt> for Option<A>
where
    A: Projects,
    Option<A::Owner>: Project<Rt> + Declared,
    Rt: Runtime,
{
    type Marker = ByProjected<Option<A::Owner>, A::Loan>;
    type At<'a> = <A::Loan as Loan>::Projection<'a, Option<A::Owner>>;
}

impl<A, B, Rt> Param<Rt> for Result<A, B>
where
    A: Projects,
    B: Projects<Loan = A::Loan>,
    Result<A::Owner, B::Owner>: Project<Rt> + Declared,
    Rt: Runtime,
{
    type Marker = ByProjected<Result<A::Owner, B::Owner>, A::Loan>;
    type At<'a> = <A::Loan as Loan>::Projection<'a, Result<A::Owner, B::Owner>>;
}

/// A parameter's Rust type as a closure names it, at whatever lifetime the
/// annotation leaves open: the marker the macro writes for that type, and
/// the type at another lifetime, which is what the closure is called with.
/// `&T`, `&mut T`, `&str`, `&[T]` and `&mut [T]` are here;
/// `#[derive(TyArg)] #[projection]` writes one for each projection it
/// emits.
///
/// A safe trait: `At<'a>` must take the marker's run through an impl of
/// `Takes`, which is `unsafe`, so an impl states only which crossing a type
/// already has.
pub trait Param<Rt>: Sized
where
    Rt: Runtime,
{
    type Marker: Lendable<Rt> + 'static;
    type At<'a>: Takes<'a, 'a, Self::Marker, Rt>;
}

impl<'q, T, Rt> Param<Rt> for &'q T
where
    T: Borrowable<Rt> + Declared + for<'s> Within<'s>,
    Rt: Runtime,
{
    type Marker = ByRef<T, Shared, Uniform>;
    type At<'a> = &'a T;
}

impl<'q, T, Rt> Param<Rt> for &'q mut T
where
    T: Borrowable<Rt> + Declared + for<'s> Within<'s>,
    Rt: Runtime,
{
    type Marker = ByRef<T, Mut, Uniform>;
    type At<'a> = &'a mut T;
}

impl<'q, X, Rt> Param<Rt> for &'q Erased<Rt, X>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Marker = ByRef<Erased<Rt, X>, Shared, Uniform>;
    type At<'a> = &'a Erased<Rt, X>;
}

impl<'q, X, Rt> Param<Rt> for &'q mut Erased<Rt, X>
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Marker = ByRef<Erased<Rt, X>, Mut, Uniform>;
    type At<'a> = &'a mut Erased<Rt, X>;
}

impl<'q, Rt> Param<Rt> for &'q str
where
    Rt: Runtime,
{
    type Marker = ByStr;
    type At<'a> = &'a str;
}

impl<'q, X, Rt> Param<Rt> for &'q [Erased<Rt, X>]
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Marker = BySlice<Erased<Rt, X>, Shared>;
    type At<'a> = &'a [Erased<Rt, X>];
}

impl<'q, X, Rt> Param<Rt> for &'q mut [Erased<Rt, X>]
where
    Erased<TypesOnly, X>: TyArg,
    X: for<'s> Within<'s> + 'static,
    Rt: Runtime,
{
    type Marker = BySlice<Erased<Rt, X>, Mut>;
    type At<'a> = &'a mut [Erased<Rt, X>];
}

pub type SitesOf<Rt, M> = <(M,) as Parameters<Rt>>::Sites;

/// A closure of one lent parameter `Q`, as the closure annotates it. `P` is
/// what the closure returns at the one lifetime the annotation is read at,
/// which nothing calls.
pub struct Alone<Q, P>(PhantomData<fn() -> (Q, P)>);

/// A closure of the `Ctx` a handler is called with, `C`, written first as a
/// handler writes it (RFC-0023 rule 2), and one lent parameter `Q`.
pub struct WithCtx<Q, C, P>(PhantomData<fn() -> (Q, C, P)>);

/// A closure that takes a lent value as a handler takes a parameter. The
/// bound that names the parameter at one lifetime only lets the compiler
/// read its type off the closure's annotation; the closure is called through
/// the other bound, which is generic over the borrow's lifetime.
pub trait Borrows<Rt, Q, O>: Sized
where
    Rt: Runtime,
{
    type Marker: Lendable<Rt> + 'static;

    /// The runtime's, through `lend`: a host neither calls nor implements
    /// it, since `sealed::Token` has no name outside the crate.
    ///
    /// # Safety
    /// `run` is `Marker`'s run of a borrow of a live storage the value is
    /// settled at, as `sealed::Lend::lend` wrote it, and `sites` is the
    /// table `Parameters::sites` built from that storage's settled type.
    #[doc(hidden)]
    unsafe fn call(
        self,
        token: sealed::Token,
        rt: &Rt,
        run: &[Rt::Value],
        sites: &SitesOf<Rt, Self::Marker>,
    ) -> O;
}

impl<F, O, P, Q, Rt> Borrows<Rt, Alone<Q, P>, O> for F
where
    Q: Param<Rt>,
    F: FnOnce(Q) -> P,
    F: for<'a> FnOnce(Q::At<'a>) -> O,
    Rt: Runtime,
{
    type Marker = Q::Marker;

    unsafe fn call(
        self,
        _: sealed::Token,
        rt: &Rt,
        run: &[Rt::Value],
        sites: &SitesOf<Rt, Q::Marker>,
    ) -> O {
        // SAFETY: the value crosses at the parameter's own marker, whose type
        // the caller compared with the storage's settled one.
        let crossing = unsafe { Crossing::new(rt) };
        // SAFETY: the caller's contract: `run` is this parameter's own run.
        let (lent,) = unsafe { <(Q::Marker,) as Parameters<Rt>>::take(crossing, run, sites) };
        if <(Q::Marker,) as Parameters<Rt>>::LENDS_A_WORD {
            // SAFETY: the caller's contract: `run` is this parameter's own
            // run, whose storage outlives the call, and the closure the
            // borrow goes to returns before `_loans` drops.
            let _loans = unsafe { Loans::<(Q::Marker,), Rt>::over(rt, run, sites) };
            self(lent.take::<Q::At<'_>>())
        } else {
            self(lent.take::<Q::At<'_>>())
        }
    }
}

impl<F, O, P, Q, C, Rt> Borrows<Rt, WithCtx<Q, C, P>, O> for F
where
    Q: Param<Rt>,
    F: FnOnce(C, Q) -> P,
    F: for<'a, 'c, 'w> FnOnce(&'c mut Ctx<'w, Rt>, Q::At<'a>) -> O,
    Rt: Runtime,
{
    type Marker = Q::Marker;

    unsafe fn call(
        self,
        _: sealed::Token,
        rt: &Rt,
        run: &[Rt::Value],
        sites: &SitesOf<Rt, Q::Marker>,
    ) -> O {
        // SAFETY: as the one-parameter form's.
        let crossing = unsafe { Crossing::new(rt) };
        // SAFETY: as the one-parameter form's.
        let (lent,) = unsafe { <(Q::Marker,) as Parameters<Rt>>::take(crossing, run, sites) };
        let mut rooted = rt.rooted();
        // SAFETY: the `Ctx` is lent to the closure alone, as a handler's is,
        // and safe code reaches no second one to exchange it with.
        let ctx = unsafe { Rt::ctx_of(&mut rooted) };
        if <(Q::Marker,) as Parameters<Rt>>::LENDS_A_WORD {
            // SAFETY: as the one-parameter form's.
            let _loans = unsafe { Loans::<(Q::Marker,), Rt>::over(rt, run, sites) };
            self(ctx, lent.take::<Q::At<'_>>())
        } else {
            self(ctx, lent.take::<Q::At<'_>>())
        }
    }
}

const WIDEST: usize = <Pair as Form>::WIDTH;

/// Whether a parameter of type `param` admits `argument`. A projection
/// parameter asks its object for at least the fields it borrows and admits
/// an object carrying more, declared or written (RFC-0050 rule 6); every
/// other parameter matches its argument's shape exactly.
fn admits(argument: &Ty, param: &PolyTy) -> bool {
    let (Ty::Ref(mutability, held), PolyTy::Ref(asked_mutability, asked)) = (argument, param) else {
        return matches_poly(argument, param);
    };
    let (Some(Ty::Object(object)), Some(PolyTy::Object(borrowed))) = (held.whole(), asked.whole())
    else {
        return matches_poly(argument, param);
    };
    if borrowed.field_set() != FieldSet::AtLeast {
        return matches_poly(argument, param);
    }
    mutability == asked_mutability
        && borrowed
            .iter()
            .all(|(name, asked)| object.get(name).is_some_and(|held| matches_poly(held, asked)))
}

/// Lend the value `word` holds to `f`, as a call lends a handler its
/// parameter: the closure parameter's acvus type is compared with the
/// argument a borrow of `held` is before anything is read, and a mismatch is
/// the parameter's type as the error.
///
/// The runtime's, which holds the storage; a host reaches this through the
/// runtime's `with` and `with_mut`.
///
/// # Safety
/// `word` holds a value `rt` crossed at `held`, live and unmoved while `f`
/// runs, and named by nothing else for that time where the marker's loan is
/// `Mut` (RFC-0018).
#[doc(hidden)]
pub unsafe fn lend<Rt, Q, O, F>(
    rt: &Rt,
    interner: &Interner,
    word: &Rt::Value,
    held: &Ty,
    f: F,
) -> Result<O, PolyTy>
where
    Rt: Runtime,
    F: Borrows<Rt, Q, O>,
{
    // SAFETY: the caller's contract: the storage is live and unmoved while
    // the reference is used, which is this call.
    let reference = unsafe { rt.reference(word) };
    // SAFETY: the caller's contract, and `reference` names `word`'s storage.
    unsafe { lend_through(rt, interner, &reference, held, f) }
}

/// As `lend`, through a reference value that already names the storage: a
/// call's argument settled at `&T` or `&mut T` is lent as its target
/// (RFC-0097 rule 1).
///
/// # Safety
/// `reference` is a reference value naming a live, unmoved storage `rt`
/// crossed at `held`, named by nothing else while `f` runs where the
/// marker's loan is `Mut` (RFC-0018).
pub(crate) unsafe fn lend_through<Rt, Q, O, F>(
    rt: &Rt,
    interner: &Interner,
    reference: &Rt::Value,
    held: &Ty,
    f: F,
) -> Result<O, PolyTy>
where
    Rt: Runtime,
    F: Borrows<Rt, Q, O>,
{
    let param = <F::Marker as Lendable<Rt>>::param_ty(interner);
    let Some(argument) = <F::Marker as Lendable<Rt>>::argument(interner, held) else {
        return Err(param);
    };
    if !admits(&argument, &param) {
        return Err(param);
    }
    let args = [ArgAt {
        interner,
        ty: &argument,
    }];
    let sites = <(F::Marker,) as Parameters<Rt>>::sites(&CallSite::of_args(&args));
    let width = const {
        let width = <F::Marker as Arg<Rt>>::WIDTH;
        assert!(width <= WIDEST, "a lent parameter takes at most a pair's registers");
        width
    };
    let mut run = [Rt::Value::default(); WIDEST];
    // SAFETY: the caller's contract, and `argument` answered for `held`.
    unsafe {
        <F::Marker as sealed::Lend<Rt>>::lend(Crossing::new(rt), reference, held, &mut run[..width])
    };
    // SAFETY: `run` is the marker's run over the storage, and `sites` was
    // built from the argument's settled type.
    Ok(unsafe { f.call(sealed::Token(()), rt, &run[..width], &sites) })
}
