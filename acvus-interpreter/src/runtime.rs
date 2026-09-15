//! This interpreter as a `Runtime`: the shared context is the host, and
//! its vtable registry is what `erase` consults for an extension type.

use std::future::Future;
use std::pin::Pin;

use acvus_extern::{CallToken, Runtime};

use crate::error::RuntimeError;
use crate::interpreter::InterpreterContext;
use crate::value::Value;

pub type ExternHandler = acvus_extern::ExternHandler<AcvusRuntime>;
pub type ExternEntry = acvus_extern::ExternEntry<AcvusRuntime>;

#[derive(Clone)]
pub struct AcvusRuntime(pub InterpreterContext);

impl Runtime for AcvusRuntime {
    type Value = Value;
    type Error = RuntimeError;
    type CallFuture<'a> = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + Send + 'a>>;

    unsafe fn materialize<T>(&self, value: Value) -> T
    where
        T: Send + Sync + 'static,
    {
        unsafe { value.materialize::<T>() }
    }

    unsafe fn erase<T>(&self, value: T) -> Value
    where
        T: Send + Sync + 'static,
    {
        unsafe { Value::erase(&self.0.vtables, value) }
    }

    fn call_0<'a>(&'a self, f: &'a Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, Vec::new())
    }

    fn call_1<'a>(&'a self, f: &'a Value, a: &'a Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, vec![a.deep_clone()])
    }

    fn call_n<'a>(&'a self, f: &'a Value, args: &[&'a Value], _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, args.iter().map(|a| a.deep_clone()).collect())
    }
}

impl AcvusRuntime {
    /// Every closure parameter is owned by the callee today, so a lent
    /// argument enters the frame as the callee's own copy; a parameter the
    /// type checker marks as borrowed will enter as an alias instead.
    fn run<'a>(&'a self, f: &'a Value, owned: Vec<Value>) -> <Self as Runtime>::CallFuture<'a> {
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { f.as_fn() };
        Box::pin(async move { crate::interpreter::fn_value_call(closure, owned).await })
    }
}
