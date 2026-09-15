//! The contract a runtime signs to run every declared ExternFn.
//!
//! A runtime chooses its own value representation. Each method builds or
//! opens one shape of value; a shape that is not there is the runtime's
//! own error.

use std::future::{Future, Ready};

use crate::error::ExternError;
use crate::func::CallToken;

/// The contract a host signs to run declared ExternFns. A `Value` is opaque;
/// `materialize`/`erase` are the whole extraction/construction pair; `call_*`
/// run a value that is a closure. A host owns its `Value` representation.
pub trait Runtime: Send + Sync + 'static {
    type Value: Send + Sync + 'static;
    type Error: From<ExternError> + Send + Sync + 'static;
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

    /// Run the closure `f` on lent arguments. A value in a handler's hands is
    /// a name for storage the host owns; whether the callee takes a copy or
    /// an alias is the host's own affair, decided by the closure's type. Only
    /// `Fn0`/`Fn1`/… reach these: the token is theirs to mint. The `args`
    /// slice of `call_n` is the caller's stack; the future does not keep it.
    fn call_0<'a>(&'a self, f: &'a Self::Value, token: CallToken) -> Self::CallFuture<'a>;
    fn call_1<'a>(
        &'a self,
        f: &'a Self::Value,
        a: &'a Self::Value,
        token: CallToken,
    ) -> Self::CallFuture<'a>;
    fn call_n<'a>(
        &'a self,
        f: &'a Self::Value,
        args: &[&'a Self::Value],
        token: CallToken,
    ) -> Self::CallFuture<'a>;
}

/// A runtime that holds no values: for registering declarations where
/// nothing will ever run.
#[derive(Clone, Copy)]
pub struct TypesOnly;

fn no_values<T>() -> Result<T, ExternError> {
    Err(ExternError::internal("TypesOnly runtime holds no values"))
}

impl Runtime for TypesOnly {
    type Value = ();
    type Error = ExternError;
    type CallFuture<'a> = Ready<Result<(), ExternError>>;

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
    fn call_0<'a>(&'a self, _: &'a (), _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
    fn call_1<'a>(&'a self, _: &'a (), _: &'a (), _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
    fn call_n<'a>(&'a self, _: &'a (), _: &[&'a ()], _: CallToken) -> Self::CallFuture<'a> {
        std::future::ready(no_values())
    }
}
