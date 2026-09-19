//! `StrView`: the language's `&str`, a shared borrow of a `String`'s UTF-8
//! bytes held in the two registers the machine keeps a slice in (RFC-0062).

use acvus_mir::ty::{Mutability, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::handler::{Arg, Ret};
use crate::obj::{Cross, Pair};
use crate::runtime::Runtime;
use crate::slice::Words;
use crate::ty_arg::{PolyVars, TyArg};

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

impl StrView {
    pub fn of(s: &str) -> Self {
        Self {
            ptr: s.as_ptr(),
            bytes: s.len(),
        }
    }

    /// # Safety
    /// `words.ptr` is the first of `words.len` live UTF-8 bytes of one run,
    /// and that run outlives every `StrView` this makes.
    #[inline(always)]
    pub const unsafe fn from_words(words: Words) -> Self {
        Self {
            ptr: words.ptr as *const u8,
            bytes: words.len as usize,
        }
    }

    #[inline(always)]
    pub fn words(&self) -> Words {
        Words {
            ptr: self.ptr as u64,
            len: self.bytes as u64,
        }
    }

    /// Obligation across artifacts: the UTF-8 that `from_utf8_unchecked`
    /// relies on here is the checker's. A `Ty::Str` is reached only by
    /// borrowing a `String` or by a string literal (RFC-0062 Decision 2),
    /// and `substring` is the one producer that cuts a run at a chosen
    /// offset, which is why it checks `is_char_boundary` at both ends
    /// before it cuts.
    ///
    /// # Safety
    /// The run this view names is live and unmoved for `'a`.
    #[inline]
    pub unsafe fn as_str<'a>(&self) -> &'a str {
        // SAFETY: the caller's contract for liveness, and the checker's
        // obligation above for the encoding.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.bytes)) }
    }
}

impl TyArg for StrView {
    fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
        PolyTy::Ref(Mutability::Shared, Box::new(TypeArg::uniform(PolyTy::Str)))
    }
}

impl<Rt> Cross<Rt> for StrView
where
    Rt: Runtime,
{
    type Form = Pair;

    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract: `run` is the pair a view was
        // written into, and the bytes it names are live.
        unsafe { Self::from_words(rt.slice_from_run(run)) }
    }

    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
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
    type Of<'a> = &'a str;
    type Form = Pair;

    fn into_run(value: &str, rt: &Rt, out: &mut [Rt::Value]) {
        rt.slice_into_run(StrView::of(value).words(), out)
    }
}

/// The macro emits this where `ByRef` would stand for a `&T` parameter,
/// for a parameter written `&str` in Rust.
pub struct ByStr;

impl<'a, Rt> Arg<'a, Rt> for ByStr
where
    Rt: Runtime,
{
    type Out = &'a str;
    type Form = Pair;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> &'a str {
        // SAFETY: the caller's contract: `run` is this parameter's pair and
        // the bytes it names are live for `'a`.
        unsafe { StrView::from_run(rt, run).as_str() }
    }
}
