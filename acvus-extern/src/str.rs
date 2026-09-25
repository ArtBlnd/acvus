//! `StrView`: the language's `&str`, a shared borrow of a `String`'s UTF-8
//! bytes held in the two registers the machine keeps a slice in (RFC-0062).

use acvus_mir::ty::{Mutability, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::handler::{Arg, Gives, Ret, Takes};
use crate::obj::{Cross, Pair};
use crate::runtime::Runtime;
use crate::slice::Words;
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

/// A run of UTF-8 as the machine holds it: a pointer and a length in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StrView {
    ptr: *const u8,
    bytes: usize,
}

// SAFETY: a `StrView` is a borrow of bytes the checker keeps alive for as
// long as the view exists (RFC-0018), and `&str` is `Send + Sync`. The raw
// pointer carries no capability the `&str` it was taken from did not have.
unsafe impl Send for StrView {}
// SAFETY: as `Send`.
unsafe impl Sync for StrView {}

/// The crossing's and the runtime's, and no handler's: a handler takes the
/// language's `&str` as Rust's `&str` (`ByStr`) and returns it as Rust's
/// `&str` borrowed from a parameter (`RetStr`), with Rust's lifetime on
/// both. A `StrView` carries none, so every way in and out of one is
/// `unsafe`.
impl StrView {
    /// # Safety
    /// `s` outlives every use of the view, which the loan the language's
    /// `&str` holds keeps true for a view that crosses (RFC-0018).
    #[doc(hidden)]
    pub unsafe fn of(s: &str) -> Self {
        Self {
            ptr: s.as_ptr(),
            bytes: s.len(),
        }
    }

    /// # Safety
    /// `words.ptr` is the first of `words.len` live UTF-8 bytes of one run,
    /// and that run outlives every `StrView` this makes.
    #[doc(hidden)]
    #[inline(always)]
    pub const unsafe fn from_words(words: Words) -> Self {
        Self {
            ptr: words.ptr as *const u8,
            bytes: words.len as usize,
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn words(&self) -> Words {
        Words {
            ptr: self.ptr as u64,
            len: self.bytes as u64,
        }
    }

    /// Obligation across artifacts: the UTF-8 that `from_utf8_unchecked`
    /// relies on here is the checker's. A `Ty::Str` is reached only by
    /// borrowing a `String` or by a string literal (RFC-0062 rule 2),
    /// and `substring` is the one producer that cuts a run at a chosen
    /// offset, which is why it checks `is_char_boundary` at both ends
    /// before it cuts.
    ///
    /// # Safety
    /// The run this view names is live and unmoved for `'a`.
    #[doc(hidden)]
    #[inline]
    pub unsafe fn as_str<'a>(&self) -> &'a str {
        // SAFETY: the caller's contract for liveness, and the checker's
        // obligation above for the encoding.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.bytes)) }
    }
}

impl Var<kind::Type> for StrView {}

// SAFETY: a view holds no `Erased`.
unsafe impl crate::Canonical<kind::Type> for StrView {
    type Canon = Self;
}

// SAFETY: a type with no type parameter reaches none.
unsafe impl<M> crate::UniformPayload<M> for StrView {}

impl TyArg for StrView {
    fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
        PolyTy::Ref(Mutability::Shared, Box::new(TypeArg::uniform(PolyTy::Str)))
    }
}

crate::within_every!(StrView);

// SAFETY: the run is the pair the runtime writes and reads for this view's own
// bytes; nothing else crosses, and the capability is not kept.
unsafe impl<Rt> Cross<Rt> for StrView
where
    Rt: Runtime,
{
    type Form = Pair;
    type ReturnForm = Pair;

    unsafe fn from_run(rt: crate::Crossing<'_, Rt>, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract: `run` is the pair a view was
        // written into, and the bytes it names are live.
        unsafe { Self::from_words(rt.slice_from_run(run)) }
    }

    fn into_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.words(), out)
    }

    fn into_return_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        rt.slice_into_run(self.words(), out)
    }
}

/// The macro emits this where `Val` would stand for an owned result, for a
/// result written `&str` in Rust.
pub struct RetStr;

impl<Rt> Ret<Rt> for RetStr
where
    Rt: Runtime,
{
    type Form = Pair;
}

// SAFETY: the pair names exactly the `&str` the handler returned; nothing else
// crosses, and the capability is not kept.
unsafe impl<'x, Rt> Gives<RetStr, Rt> for &'x str
where
    Rt: Runtime,
{
    fn give(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        // SAFETY: the result is a borrow of a parameter the caller lent
        // for this call (RFC-0047 rule 3), which outlives the pair.
        rt.slice_into_run(unsafe { StrView::of(self) }.words(), out)
    }
}

/// The macro emits this where `ByRef` would stand for a `&T` parameter, for a
/// parameter written `&str` in Rust. It stays its own `Arg` rather than a loan
/// of `str`: its `Form` is the `Pair` a view occupies, where every `ByRef` is
/// one value, and there is no exclusive twin to fold it with.
pub struct ByStr;

impl<Rt> Arg<Rt> for ByStr
where
    Rt: Runtime,
{
    type Site = ();
    type Form = Pair;

    const LENDS_A_WORD: bool = false;

    fn site(_: &crate::handler::CallSite<'_, Rt>, _: usize) {}

    #[inline(always)]
    unsafe fn loan_ended(_: &Rt, _: &[Rt::Value], _: &Self::Site) {}
}

// SAFETY: the `&str` is read from this parameter's own pair; nothing else
// crosses, and the capability is not kept.
unsafe impl<'a, 'w, Rt> Takes<'a, 'w, ByStr, Rt> for &'a str
where
    Rt: Runtime,
{
    unsafe fn take(rt: crate::Crossing<'a, Rt>, run: &'a [Rt::Value], _: &()) -> &'a str {
        // SAFETY: the caller's contract: `run` is this parameter's pair and
        // the bytes it names are live for `'a`.
        unsafe { StrView::from_run(rt, run).as_str() }
    }
}
