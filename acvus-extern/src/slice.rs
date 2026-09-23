//! `Slice<T, M, Rt>`: the acvus types `&[T]` and `&mut [T]`, the one thing the
//! machine indexes (RFC-0047).
//!
//! A slice is a borrow of the container it was taken from: it holds that
//! container's loan and is never stored beyond it (RFC-0018). Every
//! sliceable container stores the runtime's own values, so the element
//! width is the runtime's `Value` and nothing else (RFC-0047 rule 1); an
//! `Elements` is therefore the only payload it carries, and the
//! machine reads it without knowing any container's layout.
//!
//! A `Slice<T, M, Rt>` is made in one place: the crossing, as
//! `Cross::from_run` of the register pair the machine keeps a slice the
//! checker typed `&[T]` / `&mut [T]` in. That is the whole ground of `T`,
//! so there is no constructor a handler can call: `of` over a `[Rt::Value]`
//! put a `T` on a run nothing checked, and `from_elements` did the same
//! over the pair (RFC-0068 rule 1).
//!
//! `with` is the one operation, and Rust proves it safe: it hands `f` a
//! `&'a [T]` for the `'a` of `&'a self`, or a `&'a mut [T]` for `&'a mut
//! self`, so the borrow cannot outlive the slice and an exclusive one is
//! the only one. What it reads is `T`s in a live run by the premise above —
//! the crossing made the slice, the loan it holds keeps the run alive
//! (RFC-0018), and `T: TransparentOver<Rt>` is the layout that lets a
//! `[Rt::Value]` be read as a `[T]`. `at` and `at_mut` returned one element
//! at a lifetime the caller chose, with the bound left to a `debug_assert!`;
//! a Rust slice indexes with Rust's own check and Rust's own lifetime.

use std::marker::PhantomData;

use acvus_mir::ty::{PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::loan::{Loan, Mut, Shared};
use crate::obj::{Cross, TransparentOver};
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

/// A run as the machine holds it: one word per register of the pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Words {
    pub ptr: u64,
    pub len: u64,
}

/// A run of the runtime's values in a storage: what a `Slice` carries and
/// what the crossing builds it from. Nothing outside this module names one.
struct Elements<Rt>
where
    Rt: Runtime,
{
    ptr: *const Rt::Value,
    len: usize,
}

// SAFETY: an `Elements` is a borrow of a storage the checker keeps alive
// for as long as the slice value exists (RFC-0018), and the storage holds
// `Rt::Value`s, which are `Send + Sync` by the `Runtime` contract. The raw
// pointer carries no capability the `&[Rt::Value]` it was taken from did
// not already have.
unsafe impl<Rt> Send for Elements<Rt> where Rt: Runtime {}
// SAFETY: as `Send`.
unsafe impl<Rt> Sync for Elements<Rt> where Rt: Runtime {}

impl<Rt> Elements<Rt>
where
    Rt: Runtime,
{
    /// Obligation across artifacts: `acvus-interpreter`'s `ops::index` keeps
    /// a slice in the register pair these two words are (RFC-0047 rule 6).
    ///
    /// # Safety
    /// `words.ptr` is the first of `words.len` live `Rt::Value`s of one run,
    /// and that run outlives every `Elements` this makes — the loan the
    /// slice holds is what keeps it so (RFC-0018).
    #[inline(always)]
    const unsafe fn from_words(words: Words) -> Self {
        Self {
            ptr: words.ptr as *const Rt::Value,
            len: words.len as usize,
        }
    }

    #[inline(always)]
    fn words(&self) -> Words {
        Words {
            ptr: self.ptr as u64,
            len: self.len as u64,
        }
    }
}

/// A borrow of a run of a container's elements. Taking a `Mut` one is an
/// exclusive take of the container, so no shared slice of it is live
/// (RFC-0047 rule 2).
pub struct Slice<T, M, Rt>(Elements<Rt>, PhantomData<(T, M)>)
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime;

impl<T, M, Rt> Slice<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }
}

impl<T, Rt> Slice<T, Shared, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    /// The elements, read in place as the `[T]` they are.
    pub fn with<'a, R>(&'a self, f: impl FnOnce(&'a [T]) -> R) -> R {
        // SAFETY: the module's head: the run is live for as long as this
        // slice, and `T: TransparentOver<Rt>` is the layout.
        f(unsafe { std::slice::from_raw_parts(self.0.ptr.cast::<T>(), self.0.len) })
    }
}

impl<T, Rt> Slice<T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    /// The elements, read and written in place as the `[T]` they are.
    pub fn with<'a, R>(&'a mut self, f: impl FnOnce(&'a mut [T]) -> R) -> R {
        // SAFETY: as the shared `with`'s, and an exclusive slice is the only
        // live name of its run (RFC-0047 rule 2), which `&'a mut self` keeps.
        f(
            unsafe {
                std::slice::from_raw_parts_mut(self.0.ptr.cast_mut().cast::<T>(), self.0.len)
            },
        )
    }
}

impl<T, M, Rt> Var<kind::Type> for Slice<T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
}

/// The acvus type: a reference to the unsized `[T]`.
impl<T, M, Rt> TyArg for Slice<T, M, Rt>
where
    T: TyArg + Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(
            M::MUTABILITY,
            Box::new(TypeArg::uniform(PolyTy::Slice(Box::new(T::poly_ty(
                i, vars,
            ))))),
        )
    }
}

/// A result declared `&[T]` / `&mut [T]` is returned as Rust's slice of a
/// parameter the caller lent (RFC-0047 rule 3), and crosses as the pair.
impl<T, Rt> crate::LentBack<Rt> for Slice<T, Shared, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a [T];
    type Form = crate::obj::Pair;

    fn into_run(value: &[T], rt: &Rt, out: &mut [Rt::Value]) {
        let words = Words {
            ptr: value.as_ptr() as u64,
            len: value.len() as u64,
        };
        rt.slice_into_run(words, out)
    }
}

impl<T, Rt> crate::LentBack<Rt> for Slice<T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a mut [T];
    type Form = crate::obj::Pair;

    fn into_run(value: &mut [T], rt: &Rt, out: &mut [Rt::Value]) {
        let words = Words {
            ptr: value.as_mut_ptr() as u64,
            len: value.len() as u64,
        };
        rt.slice_into_run(words, out)
    }
}

/// The macro emits this where `ByRef` would stand for a `&T` parameter, for
/// a parameter written `&[T]` / `&mut [T]` in Rust: the handler takes
/// Rust's slice, at the run's own lifetime, and never names a `Slice`.
pub struct BySlice<T, M>(PhantomData<fn() -> (T, M)>);

impl<T, M, Rt> crate::handler::Sited<Rt> for BySlice<T, M>
where
    T: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: &crate::handler::CallSite<'_, Rt>, _: usize) {}
}

impl<'a, T, Rt> crate::handler::Arg<'a, Rt> for BySlice<T, Shared>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Out = &'a [T];
    type Form = crate::obj::Pair;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> &'a [T] {
        // SAFETY: the caller's contract: `run` is this parameter's pair, and
        // the container it names is live for `'a` (RFC-0018); `T:
        // TransparentOver<Rt>` is the layout.
        let words = unsafe { rt.slice_from_run(run) };
        unsafe { std::slice::from_raw_parts(words.ptr as *const T, words.len as usize) }
    }
}

impl<'a, T, Rt> crate::handler::Arg<'a, Rt> for BySlice<T, Mut>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Out = &'a mut [T];
    type Form = crate::obj::Pair;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> &'a mut [T] {
        // SAFETY: as the shared form's, and a `&mut [T]` argument is the
        // only live name of its run (RFC-0047 rule 2).
        let words = unsafe { rt.slice_from_run(run) };
        unsafe { std::slice::from_raw_parts_mut(words.ptr as *mut T, words.len as usize) }
    }
}

/// A slice crosses as the register pair the machine keeps it in, and as
/// nothing else: it implements `Cross` and not `OneValue`, so a parameter, a
/// field or an element that names one is a compile error. The two words
/// themselves are the runtime's to build and to read, because only the
/// runtime knows what one of its values is made of.
impl<T, M, Rt> Cross<Rt> for Slice<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type Form = crate::obj::Pair;
    type ReturnForm = crate::obj::Pair;

    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract: `run` is the pair a slice was written
        // into, and the elements it names are live.
        Self(
            unsafe { Elements::from_words(rt.slice_from_run(run)) },
            PhantomData,
        )
    }

    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.0.words(), out)
    }

    fn into_return_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.0.words(), out)
    }
}
