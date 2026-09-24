//! `Slice<'a, T, M, Rt>`: the acvus types `&[T]` and `&mut [T]`, the one thing
//! the machine indexes (RFC-0047). `'a` is the call it was handed to
//! (RFC-0079 rule 6).
//!
//! A slice is a borrow of the container it was taken from: it holds that
//! container's loan and is never stored beyond it (RFC-0018). Every
//! sliceable container stores the runtime's own values, so the element
//! width is the runtime's `Value` and nothing else (RFC-0047 rule 1); an
//! `Elements` is therefore the only payload it carries, and the
//! machine reads it without knowing any container's layout.
//!
//! A `Slice<'a, T, M, Rt>` is made in one place: the crossing, as
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
pub struct Slice<'a, T, M, Rt>(Elements<Rt>, PhantomData<(&'a (), T, M)>)
where
    T: Send + Sync,
    M: Loan,
    Rt: Runtime;

// SAFETY: a `Slice` is at its own `'s`, and what its elements hold is at `'s`.
unsafe impl<'s, T, M, Rt> crate::Within<'s> for Slice<'s, T, M, Rt>
where
    T: Send + Sync + crate::Within<'s>,
    M: Loan,
    Rt: Runtime,
{
}

impl<'b, T, M, Rt> Slice<'b, T, M, Rt>
where
    T: Send + Sync,
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

impl<'b, T, Rt> Slice<'b, T, Shared, Rt>
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

impl<'b, T, Rt> Slice<'b, T, Mut, Rt>
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

impl<'a, T, M, Rt> Var<kind::Type> for Slice<'a, T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
}

// SAFETY: the element is its own canonical form's, and the lifetime is at
// `'static`.
unsafe impl<'a, T, M, Rt> crate::Canonical<kind::Type> for Slice<'a, T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
    type Canon = Slice<'static, T::Canon, M, Rt>;
}

// SAFETY: a `Slice` is a pointer to the runtime's values and a length at
// every `T` and `M`.
unsafe impl<'a, Mk, T, M, Rt> crate::UniformPayload<Mk> for Slice<'a, T, M, Rt>
where
    T: Send + Sync,
    M: Loan,
    Rt: Runtime,
{
}

/// The acvus type: a reference to the unsized `[T]`.
impl<T, M, Rt> TyArg for Slice<'static, T, M, Rt>
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
impl<T, M, Rt> crate::LentBack<Rt> for Slice<'static, T, M, Rt>
where
    T: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Form = crate::obj::Pair;
}

// SAFETY: the pair names exactly the slice the handler returned; nothing else
// crosses, and the capability is not kept.
unsafe impl<'x, T, D, Rt> crate::handler::Gives<crate::RetLent<Slice<'static, T, Shared, Rt>>, Rt>
    for &'x [D]
where
    T: TransparentOver<Rt>,
    D: TransparentOver<Rt>,
    Rt: Runtime,
{
    fn give(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        let words = Words {
            ptr: self.as_ptr() as u64,
            len: self.len() as u64,
        };
        rt.slice_into_run(words, out)
    }
}

// SAFETY: as the shared impl's, exclusively.
unsafe impl<'x, T, D, Rt> crate::handler::Gives<crate::RetLent<Slice<'static, T, Mut, Rt>>, Rt>
    for &'x mut [D]
where
    T: TransparentOver<Rt>,
    D: TransparentOver<Rt>,
    Rt: Runtime,
{
    fn give(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        let words = Words {
            ptr: self.as_mut_ptr() as u64,
            len: self.len() as u64,
        };
        rt.slice_into_run(words, out)
    }
}

/// The macro emits this where `ByRef` would stand for a `&T` parameter, for
/// a parameter written `&[T]` / `&mut [T]` in Rust: the handler takes
/// Rust's slice, at the run's own lifetime, and never names a `Slice`.
pub struct BySlice<T, M>(PhantomData<fn() -> (T, M)>);

impl<T, M, Rt> crate::handler::Arg<Rt> for BySlice<T, M>
where
    T: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();
    type Form = crate::obj::Pair;

    fn site(_: &crate::handler::CallSite<'_, Rt>, _: usize) {}
}

// SAFETY: the slice is read from this parameter's own pair at `D`, which
// `TransparentOver` lays out as the runtime's value; nothing else crosses, and
// the capability is not kept.
unsafe impl<'a, 'w, T, D, Rt> crate::handler::Takes<'a, 'w, BySlice<T, Shared>, Rt> for &'a [D]
where
    T: TransparentOver<Rt>,
    D: TransparentOver<Rt> + crate::Within<'a>,
    Rt: Runtime,
{
    unsafe fn take(rt: crate::Crossing<'a, Rt>, run: &'a [Rt::Value], _: &()) -> &'a [D] {
        // SAFETY: the caller's contract: `run` is this parameter's pair, and
        // the container it names is live for `'a` (RFC-0018); `D:
        // TransparentOver<Rt>` is the layout.
        let words = unsafe { rt.slice_from_run(run) };
        unsafe { std::slice::from_raw_parts(words.ptr as *const D, words.len as usize) }
    }
}

// SAFETY: as the shared impl's, exclusively.
unsafe impl<'a, 'w, T, D, Rt> crate::handler::Takes<'a, 'w, BySlice<T, Mut>, Rt> for &'a mut [D]
where
    T: TransparentOver<Rt>,
    D: TransparentOver<Rt> + crate::Within<'a>,
    Rt: Runtime,
{
    unsafe fn take(rt: crate::Crossing<'a, Rt>, run: &'a [Rt::Value], _: &()) -> &'a mut [D] {
        // SAFETY: as the shared form's, and a `&mut [D]` argument is the
        // only live name of its run (RFC-0047 rule 2).
        let words = unsafe { rt.slice_from_run(run) };
        unsafe { std::slice::from_raw_parts_mut(words.ptr as *mut D, words.len as usize) }
    }
}

/// A slice crosses as the register pair the machine keeps it in, and as
/// nothing else: it implements `Cross` and not `OneValue`, so a parameter, a
/// field or an element that names one is a compile error. The two words
/// themselves are the runtime's to build and to read, because only the
/// runtime knows what one of its values is made of.
///
/// `T: TransparentOver<Rt>` is a decision: the crossing is the one maker of
/// a `Slice`, and a slice over any other `T` would be taken and could only
/// say its length, since `with` reads the run as `[T]` under that layout
/// alone. Such a parameter is refused at its declaration, where
/// `TransparentOver`'s diagnostic names `Slice<Erased<Rt, T>, _, Rt>`.
// SAFETY: the run is the pair the runtime writes and reads for this slice's own
// elements; nothing else crosses, and the capability is not kept.
unsafe impl<'a, T, M, Rt> Cross<Rt> for Slice<'a, T, M, Rt>
where
    T: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Form = crate::obj::Pair;
    type ReturnForm = crate::obj::Pair;

    unsafe fn from_run(rt: crate::Crossing<'_, Rt>, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract: `run` is the pair a slice was written
        // into, and the elements it names are live.
        Self(
            unsafe { Elements::from_words(rt.slice_from_run(run)) },
            PhantomData,
        )
    }

    fn into_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.0.words(), out)
    }

    fn into_return_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.0.words(), out)
    }
}
