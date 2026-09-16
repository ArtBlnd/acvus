//! The contract a runtime signs to run every declared ExternFn.
//!
//! A runtime chooses its own value representation. Each method builds or
//! opens one shape of value; a shape that is not there is the runtime's
//! own error.

use std::future::{Future, Ready};

use crate::func::CallToken;
use crate::trap::Trap;

/// The contract a host signs to run declared ExternFns. A `Value` is opaque;
/// `materialize`/`erase` are the whole extraction/construction pair; `call_*`
/// run a value that is a closure. A host owns its `Value` representation.
pub trait Runtime: Sized + Send + Sync + 'static {
    type Value: crate::Cross<Self>;
    type Error: From<Trap> + Send + Sync + 'static;
    type CallFuture<'a>: Future<Output = Result<Self::Value, Self::Error>> + Send + 'a
    where
        Self: 'a;

    /// # Safety
    /// `T` must be the type the value was `erase`d from.
    unsafe fn materialize<T>(&self, value: Self::Value) -> T
    where
        T: Send + Sync + 'static;
    /// # Safety
    /// The value may only be `materialize`d back to this same `T`.
    unsafe fn erase<T>(&self, value: T) -> Self::Value
    where
        T: Send + Sync + 'static;

    /// The reference borrows the runtime as well as the value: a runtime
    /// may hand out a value that lives on its own stack, so this is not a
    /// free function over `Value`.
    ///
    /// # Safety
    /// `T` must be the type the value was `erase`d from.
    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a Self::Value) -> &'a T
    where
        T: Send + Sync + 'static;
    /// # Safety
    /// As `value_as_ref`.
    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut Self::Value) -> &'a mut T
    where
        T: Send + Sync + 'static;

    /// Read an `Inline` value out of the word it lives in.
    ///
    /// # Safety
    /// `T` must be the type the value was `erase`d from.
    unsafe fn inline_ref<T>(value: &Self::Value) -> &T
    where
        T: crate::obj::Inline;
    /// # Safety
    /// As `inline_ref`.
    unsafe fn inline_mut<T>(value: &mut Self::Value) -> &mut T
    where
        T: crate::obj::Inline;

    /// Read the storage a reference names (RFC-0018).
    ///
    /// # Safety
    /// `reference` is a `&T` / `&mut T` value and its storage holds a `T`
    /// erased from that type.
    unsafe fn deref<'a, T>(&self, reference: &'a Self::Value) -> &'a T
    where
        T: Send + Sync + 'static;
    /// # Safety
    /// `reference` is a `&mut T` value and its storage holds a `T` erased
    /// from that type; the checker admits no other live name of the storage.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a, T>(&self, reference: &'a Self::Value) -> &'a mut T
    where
        T: Send + Sync + 'static;

    /// The name a field key is at run time (RFC-0032).
    fn symbol(&self, name: &str) -> acvus_utils::Astr;

    /// A reference value naming `target`'s storage (RFC-0018): what a
    /// handler passes to a closure whose parameter is `&T` / `&mut T`.
    ///
    /// # Safety
    /// The reference is used only while `target` is live and unmoved; the
    /// closure it is passed to keeps it no longer than the call.
    unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

    /// Run the closure `f`; each argument moves into the callee's
    /// parameter. Only `Fn0`/`Fn1`/… reach these: the token is theirs to
    /// mint.
    fn call_0<'a>(&'a self, f: &'a Self::Value, token: CallToken) -> Self::CallFuture<'a>;
    fn call_1<'a>(
        &'a self,
        f: &'a Self::Value,
        a: Self::Value,
        token: CallToken,
    ) -> Self::CallFuture<'a>;
    fn call_n<'a>(
        &'a self,
        f: &'a Self::Value,
        args: Vec<Self::Value>,
        token: CallToken,
    ) -> Self::CallFuture<'a>;
}

/// A runtime that holds no values: for registering declarations where
/// nothing will ever run.
#[derive(Clone, Copy)]
pub struct TypesOnly;

fn no_values<T>() -> Result<T, Trap> {
    Err(Trap::internal("TypesOnly runtime holds no values"))
}

impl Runtime for TypesOnly {
    type Value = ();
    type Error = Trap;
    type CallFuture<'a> = Ready<Result<(), Trap>>;

    unsafe fn materialize<T>(&self, _: ()) -> T
    where
        T: Send + Sync + 'static,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn erase<T>(&self, _: T)
    where
        T: Send + Sync + 'static,
    {
    }
    unsafe fn value_as_ref<'a, T>(&'a self, _: &'a ()) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn value_as_mut<'a, T>(&'a self, _: &'a mut ()) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn inline_ref<T>(_: &()) -> &T
    where
        T: crate::obj::Inline,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn inline_mut<T>(_: &mut ()) -> &mut T
    where
        T: crate::obj::Inline,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn deref<'a, T>(&self, _: &'a ()) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn deref_mut<'a, T>(&self, _: &'a ()) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        panic!("TypesOnly runtime holds no values")
    }
    unsafe fn reference(&self, _: &()) {}
    fn symbol(&self, _: &str) -> acvus_utils::Astr {
        panic!("TypesOnly runtime holds no values")
    }
    fn call_0<'a>(&'a self, _: &'a (), _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
    fn call_1<'a>(&'a self, _: &'a (), _: (), _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
    fn call_n<'a>(&'a self, _: &'a (), _: Vec<()>, _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
}
