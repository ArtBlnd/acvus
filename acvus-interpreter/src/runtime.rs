//! This interpreter as a `Runtime`: the shared context is the host.

use std::any::{TypeId, type_name};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_extern::{CallToken, Runtime};

use crate::interpreter::InterpreterContext;
use crate::regs::{FrameState, Store};
use crate::value::{Kind, Value};

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
    type Frame<'a> = &'a mut FrameState;
    type Rooted = Store;
    type CallFuture<'a> = Pin<Box<dyn Future<Output = Value> + Send + 'a>>;

    fn rooted(&self) -> Store {
        Store::new()
    }

    fn frame_of(rooted: &mut Store) -> &mut FrameState {
        rooted.root_window()
    }

    fn type_of(&self, value: &Value) -> Option<TypeId> {
        match value.kind() {
            Kind::Large => Some(value.vtable().type_id),
            kind => kind.type_id(),
        }
    }

    fn type_name_of(&self, value: &Value) -> Option<&'static str> {
        match value.kind() {
            Kind::Large => Some((value.vtable().name)()),
            kind => kind.name(),
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
        unsafe { Value::erase(value) }
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

    fn none(&self) -> Value {
        Value::NONE
    }

    fn some(&self, payload: Value) -> Value {
        Value::some(payload)
    }

    fn is_none(&self, value: &Value) -> bool {
        value.is_none()
    }

    fn unwrap_some(&self, value: Value) -> Value {
        Value::some_payload(value)
    }

    fn symbol(&self, name: &str) -> acvus_utils::Astr {
        self.0.interner.intern(name)
    }

    fn slice_into_run(&self, words: acvus_extern::Words, out: &mut [Value]) {
        out[0] = word(words.ptr);
        out[1] = word(words.len);
    }

    unsafe fn slice_from_run(&self, run: &[Value]) -> acvus_extern::Words {
        acvus_extern::Words {
            ptr: run[0].bits(),
            len: run[1].bits(),
        }
    }

    fn call_is_sync(&self, f: &Value) -> bool {
        // SAFETY: the type checker admits only a closure value here.
        !unsafe { f.as_fn() }.entry.may_suspend()
    }

    fn call_now<A>(&self, f: &Value, frame: &mut &mut FrameState, args: A, _: CallToken) -> Value
    where
        A: acvus_extern::IntoRun<Self>,
    {
        args.into_run(self, frame.run_mut(A::WIDTH));
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { f.as_fn() };
        let arity = u16::try_from(A::WIDTH).expect("a closure takes at most one cell of arguments");
        crate::machine::fn_value_call_in_window(closure, frame, arity)
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

/// One of the machine's registers holding a bare word rather than a value of
/// the language: half of the pair a slice occupies (RFC-0047 amended), which
/// the machine reads with `Value::bits`.
pub fn word(bits: u64) -> Value {
    // SAFETY: `u64` is `Inline`, so the word is the value.
    unsafe { Value::erase(bits) }
}

/// # Safety
/// `value` was erased from a `T`.
unsafe fn read<T>(value: &Value) -> &T
where
    T: 'static,
{
    match Kind::of::<T>() {
        Some(kind) => {
            debug_assert_eq!(
                value.kind(),
                kind,
                "read: value is not a {}",
                type_name::<T>()
            );
            // SAFETY: an `Inline` `T` was written into the word by `erase`.
            unsafe { &*(value.bits_ref() as *const u64 as *const T) }
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
    match Kind::of::<T>() {
        Some(kind) => {
            debug_assert_eq!(
                value.kind(),
                kind,
                "read_mut: value is not a {}",
                type_name::<T>()
            );
            // SAFETY: an `Inline` `T` was written into the word by `erase`.
            unsafe { &mut *(value.bits_mut() as *mut u64 as *mut T) }
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

acvus_extern::cross_one_value!(Value, at AcvusRuntime);

/// The runtime's own value crosses as itself: nothing to convert, and a
/// reference to one is read through the word that names it (RFC-0039).
impl acvus_extern::OneValue<AcvusRuntime> for Value {
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

impl acvus_extern::Borrowable<AcvusRuntime> for Value {}
