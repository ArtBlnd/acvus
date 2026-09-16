//! This interpreter as a `Runtime`: the shared context is the host, and
//! its vtable registry is what `erase` consults for an extension type.

use std::future::Future;
use std::pin::Pin;

use acvus_extern::{CallToken, Runtime};

use crate::error::RuntimeError;
use crate::interpreter::InterpreterContext;
use crate::value::{Value, is_small};

pub type ExternHandler = acvus_extern::ExternHandler<AcvusRuntime>;

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

    unsafe fn value_as_ref<'a, T>(&'a self, value: &'a Value) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract: the value was erased from a `T`.
        unsafe { read::<T>(value) }
    }

    unsafe fn value_as_mut<'a, T>(&'a self, value: &'a mut Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract: the value was erased from a `T`.
        unsafe { read_mut::<T>(value) }
    }

    unsafe fn inline_ref<T>(value: &Value) -> &T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract: the value was erased from a `T`.
        unsafe { read::<T>(value) }
    }

    unsafe fn inline_mut<T>(value: &mut Value) -> &mut T
    where
        T: acvus_extern::Inline,
    {
        // SAFETY: the caller's contract: the value was erased from a `T`.
        unsafe { read_mut::<T>(value) }
    }

    unsafe fn deref<'a, T>(&self, reference: &'a Value) -> &'a T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract: a live reference to a `T`.
        unsafe { read::<T>(reference.target()) }
    }

    unsafe fn deref_mut<'a, T>(&self, reference: &'a Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
    {
        // SAFETY: the caller's contract: a live, exclusively named `T`.
        unsafe { read_mut::<T>(reference.target_mut()) }
    }

    unsafe fn reference(&self, target: &Value) -> Value {
        Value::reference(target)
    }

    fn symbol(&self, name: &str) -> acvus_utils::Astr {
        self.0.interner.intern(name)
    }

    fn call_0<'a>(&'a self, f: &'a Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, Vec::new())
    }

    fn call_1<'a>(&'a self, f: &'a Value, a: Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, vec![a])
    }

    fn call_n<'a>(&'a self, f: &'a Value, args: Vec<Value>, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, args)
    }
}

/// # Safety
/// `value` was erased from a `T`.
unsafe fn read<T>(value: &Value) -> &T
where
    T: 'static,
{
    if const { is_small::<T>() } {
        // SAFETY: a small `T` was written into the word by `erase`.
        unsafe { &*(value.small_ref() as *const u64 as *const T) }
    } else {
        // SAFETY: a large `T` is the payload behind the header.
        unsafe { value.peek::<T>() }
    }
}

/// # Safety
/// As `read`, exclusively.
unsafe fn read_mut<T>(value: &mut Value) -> &mut T
where
    T: 'static,
{
    if const { is_small::<T>() } {
        // SAFETY: a small `T` was written into the word by `erase`.
        unsafe { &mut *(value.small_mut() as *mut u64 as *mut T) }
    } else {
        // SAFETY: a large `T` is the payload behind the header.
        unsafe { value.peek_mut::<T>() }
    }
}

impl AcvusRuntime {
    fn run<'a>(&'a self, f: &'a Value, args: Vec<Value>) -> <Self as Runtime>::CallFuture<'a> {
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { f.as_fn() };
        Box::pin(async move { crate::interpreter::fn_value_call(closure, args).await })
    }
}

/// The runtime's own value crosses as itself: nothing to convert, and a
/// reference to one is read through the word that names it (RFC-0039).
impl acvus_extern::Cross<AcvusRuntime> for Value {
    fn erase(self, _: &AcvusRuntime) -> Value {
        self
    }

    fn materialize(_: &AcvusRuntime, value: Value) -> Self {
        value
    }

    unsafe fn deref<'a>(_: &AcvusRuntime, reference: &'a Value) -> &'a Value {
        // SAFETY: the caller's contract: a live reference.
        unsafe { reference.target() }
    }

    unsafe fn deref_mut<'a>(_: &AcvusRuntime, reference: &'a Value) -> &'a mut Value {
        // SAFETY: the caller's contract: a live, exclusively named target.
        unsafe { reference.target_mut() }
    }
}
