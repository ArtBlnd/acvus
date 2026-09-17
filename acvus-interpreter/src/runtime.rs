//! This interpreter as a `Runtime`: the shared context is the host, and
//! its vtable registry is what `erase` consults for an extension type.

use std::any::{TypeId, type_name};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_extern::{CallToken, Runtime};

use crate::interpreter::InterpreterContext;
use crate::value::{Tag, Value};

pub type ExternHandler = acvus_extern::ExternHandler<AcvusRuntime>;

#[derive(Clone)]
#[repr(transparent)]
pub struct AcvusRuntime(pub Arc<InterpreterContext>);

impl AcvusRuntime {
    /// The runtime a context already is: `AcvusRuntime` is that context
    /// and nothing else, so a caller holding one borrows a runtime from it
    /// rather than sharing the `Arc` again.
    pub fn of(shared: &Arc<InterpreterContext>) -> &AcvusRuntime {
        // SAFETY: `#[repr(transparent)]` over `Arc<InterpreterContext>`.
        unsafe { &*(shared as *const Arc<InterpreterContext>).cast::<AcvusRuntime>() }
    }
}

impl Runtime for AcvusRuntime {
    type Value = Value;
    type CallFuture<'a> = Pin<Box<dyn Future<Output = Value> + Send + 'a>>;

    fn type_of(&self, value: &Value) -> Option<TypeId> {
        match value {
            Value::Small(tag, _) => Some(tag.type_id()),
            // SAFETY: the header is live for as long as the value.
            Value::Large(p) => Some(unsafe { p.as_ref() }.vtable.type_id),
            Value::Ref(_) | Value::Empty | Value::Undef => None,
        }
    }

    fn type_name_of(&self, value: &Value) -> Option<&'static str> {
        match value {
            Value::Small(tag, _) => Some(tag.name()),
            // SAFETY: the header is live for as long as the value.
            Value::Large(p) => Some(unsafe { p.as_ref() }.vtable.name),
            Value::Ref(_) | Value::Empty | Value::Undef => None,
        }
    }

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

    fn call_is_sync(&self, f: &Value) -> bool {
        // SAFETY: the type checker admits only a closure value here.
        !unsafe { f.as_fn() }.code.may_suspend
    }

    fn call_now(&self, f: &Value, args: &mut [Value], _: CallToken) -> Value {
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { f.as_fn() };
        crate::machine::fn_value_call_now(closure, args)
    }

    fn call_0<'a>(&'a self, f: &'a Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, &mut [])
    }

    fn call_1<'a>(&'a self, f: &'a Value, a: Value, _: CallToken) -> Self::CallFuture<'a> {
        self.run(f, &mut [a])
    }

    fn call_n<'a>(
        &'a self,
        f: &'a Value,
        args: &mut [Value],
        _: CallToken,
    ) -> Self::CallFuture<'a> {
        self.run(f, args)
    }
}

/// # Safety
/// `value` was erased from a `T`.
unsafe fn read<T>(value: &Value) -> &T
where
    T: 'static,
{
    match Tag::of::<T>() {
        Some(tag) => {
            debug_assert_eq!(
                value.tag(),
                tag,
                "read: value is not a {}",
                type_name::<T>()
            );
            // SAFETY: an `Inline` `T` was written into the word by `erase`.
            unsafe { &*(value.small_ref() as *const u64 as *const T) }
        }
        // SAFETY: a large `T` is the payload behind the header.
        None => unsafe { value.peek::<T>() },
    }
}

/// # Safety
/// As `read`, exclusively.
unsafe fn read_mut<T>(value: &mut Value) -> &mut T
where
    T: 'static,
{
    match Tag::of::<T>() {
        Some(tag) => {
            debug_assert_eq!(
                value.tag(),
                tag,
                "read_mut: value is not a {}",
                type_name::<T>()
            );
            // SAFETY: an `Inline` `T` was written into the word by `erase`.
            unsafe { &mut *(value.small_mut() as *mut u64 as *mut T) }
        }
        // SAFETY: a large `T` is the payload behind the header.
        None => unsafe { value.peek_mut::<T>() },
    }
}

impl AcvusRuntime {
    fn run<'a>(&'a self, f: &'a Value, args: &mut [Value]) -> <Self as Runtime>::CallFuture<'a> {
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { f.as_fn() };
        Box::pin(crate::machine::fn_value_call(closure, args))
    }
}

impl acvus_extern::FromValue<AcvusRuntime> for Value {
    fn from_value(_: &AcvusRuntime, value: Value) -> Value {
        value
    }
}

/// The runtime's own value crosses as itself: nothing to convert, and a
/// reference to one is read through the word that names it (RFC-0039).
impl acvus_extern::Cross<AcvusRuntime> for Value {
    fn erase(self, _: &AcvusRuntime) -> Value {
        self
    }

    unsafe fn materialize(_: &AcvusRuntime, value: Value) -> Self {
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
