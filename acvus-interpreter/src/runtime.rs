//! This interpreter as a `Runtime`: the shared context is the host.

use std::any::{TypeId, type_name};
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;

use acvus_extern::{Ctx, Owned, Runtime, Variant};

use crate::interpreter::InterpreterContext;
use crate::ops::call;
use crate::regs::{FrameState, RootCells, RootFrame};
use crate::value::{Kind, Value, VariantValue};

pub type ExternHandler = acvus_extern::ExternHandler<AcvusRuntime>;

/// One run as a `Runtime`: the state its functions read and the page its
/// contexts live on (RFC-0014). `Interpreter` makes the pair and a spawned
/// run is handed its parent's, so a closure value carries neither
/// (RFC-0069 rule 1): every caller of one holds a `&AcvusRuntime`.
#[derive(Clone)]
pub struct AcvusRuntime {
    pub shared: Arc<InterpreterContext>,
    pub page: Arc<dyn crate::journal::RuntimeContext>,
}

impl AcvusRuntime {
    pub fn new(
        shared: Arc<InterpreterContext>,
        page: Arc<dyn crate::journal::RuntimeContext>,
    ) -> AcvusRuntime {
        AcvusRuntime { shared, page }
    }
}

/// The Rust type `acvus-extern`'s `OneValue for Result<T, E>` erases into this
/// runtime and materializes back out of it (RFC-0038). It is the crossing's
/// shape, not the held one: RFC-0050 rule 8 gives a `Result` the flat variant
/// every other variant type has, and `erase` and `materialize` below are the
/// one boundary that sees both spellings. Teach `acvus-extern` to erase a
/// `Variant` directly and the two arms that name this type go away.
type CrossedResult = Result<Owned<AcvusRuntime>, Owned<AcvusRuntime>>;

impl AcvusRuntime {
    fn variant_of_result(&self, crossed: CrossedResult) -> Value {
        let (tag, payload) = match crossed {
            Ok(v) => ("Ok", v),
            Err(e) => ("Err", e),
        };
        Value::variant(self.shared.interner.intern(tag), Some(payload))
    }

    fn result_of_variant(&self, value: Value) -> CrossedResult {
        // SAFETY: `materialize`'s contract: the value came from the `erase`
        // above, which wrote a variant.
        let held = unsafe { value.materialize::<VariantValue>() };
        let Variant {
            values: [tag, payload],
        } = held;
        // SAFETY: the first register of a variant this runtime wrote is its tag.
        let tag = unsafe { tag.into_value(acvus_extern::Holding::new()).as_tag() };
        if tag == self.shared.interner.intern("Ok") {
            return Ok(payload);
        }
        debug_assert!(
            tag == self.shared.interner.intern("Err"),
            "a Result crossed back holding the tag `{}`, and `variant_of_result` — the only \
             writer `materialize`'s contract admits — writes `Ok` or `Err`",
            tag.display(&self.shared.interner)
        );
        Err(payload)
    }
}

/// The `Ctx` of a call whose frame outlives the frame it was made on: the
/// cells at the root of its own chain, and the context over them.
pub struct RootedCtx<'a> {
    ctx: Ctx<'a, AcvusRuntime>,
    _cells: RootCells,
}

impl<'a> RootedCtx<'a> {
    fn new(rt: &'a AcvusRuntime) -> RootedCtx<'a> {
        let RootFrame { state, cells } = RootFrame::new();
        RootedCtx {
            // SAFETY: `state` names `cells`, which this `RootedCtx` keeps
            // beside the `Ctx` for as long as the `Ctx` lives, and nothing
            // else holds `state`.
            ctx: unsafe { Ctx::new(rt, state) },
            _cells: cells,
        }
    }
}

impl Runtime for AcvusRuntime {
    type Value = Value;
    type Frame<'a> = FrameState;
    type Rooted<'a> = RootedCtx<'a>;
    type CallFuture<'a> = Pin<Box<dyn Future<Output = Value> + Send + 'a>>;
    type Op = Box<dyn crate::code::Op>;
    type CallShape = call::CallShape;
    type AsyncShape = call::AsyncShape;
    type FusedCall = call::Call;
    type FusedShape = call::FusedShape;

    fn instance_value(entry: &acvus_extern::InstanceEntry<Self>) -> Value {
        Value::instance(entry)
    }

    unsafe fn instance_entry<'a>(value: &'a Value) -> &'a acvus_extern::InstanceEntry<Self> {
        // SAFETY: the caller's contract: `instance_value` wrote this value.
        unsafe { value.as_instance_entry() }
    }

    fn rooted(&self) -> RootedCtx<'_> {
        RootedCtx::new(self)
    }

    unsafe fn ctx_of<'a, 'r>(rooted: &'r mut RootedCtx<'a>) -> &'r mut Ctx<'a, AcvusRuntime>
    where
        'a: 'r,
    {
        &mut rooted.ctx
    }

    fn op<H>(handler: H, shape: call::CallShape) -> Box<dyn crate::code::Op>
    where
        H: acvus_extern::Handler<AcvusRuntime>,
    {
        call::op(handler, shape)
    }

    fn fused<H>(handler: H, shape: call::FusedShape) -> call::Call
    where
        H: acvus_extern::Handler<AcvusRuntime>,
    {
        call::fused_call(handler, shape)
    }

    fn async_extern_op<H>(handler: H, shape: call::AsyncShape) -> Box<dyn crate::code::Op>
    where
        H: acvus_extern::AsyncCall<AcvusRuntime>,
    {
        call::async_extern_op(handler, shape)
    }


    unsafe fn materialize<T>(&self, value: Value) -> T
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<CrossedResult>() {
            let crossed = self.result_of_variant(value);
            // SAFETY: the `TypeId` above says `T` is `CrossedResult`; the copy
            // takes ownership and the original is forgotten.
            let out: T = unsafe { mem::transmute_copy(&crossed) };
            mem::forget(crossed);
            return out;
        }
        unsafe { value.materialize::<T>() }
    }

    unsafe fn erase<T>(&self, value: T) -> Value
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<CrossedResult>() {
            // SAFETY: as `materialize`'s.
            let crossed: CrossedResult = unsafe { mem::transmute_copy(&value) };
            mem::forget(value);
            return self.variant_of_result(crossed);
        }
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

    fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send + use<> {
        self.shared.executor.sleep(d)
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

    unsafe fn some_at<'a>(&self, value: &'a Value) -> Option<&'a Value> {
        (!value.is_none()).then_some(value)
    }

    unsafe fn some_at_mut<'a>(&self, value: &'a mut Value) -> Option<&'a mut Value> {
        (!value.is_none()).then_some(value)
    }

    fn symbol(&self, name: &str) -> acvus_utils::Astr {
        self.shared.interner.intern(name)
    }

    fn variant_tag(&self, name: &str) -> Value {
        Value::tag(self.shared.interner.intern(name))
    }

    unsafe fn tag_symbol(&self, tag: &Value) -> acvus_utils::Astr {
        // SAFETY: the caller's contract.
        unsafe { tag.as_tag() }
    }

    fn undef(&self) -> Value {
        Value::UNDEF
    }

    fn is_undef(&self, value: &Value) -> bool {
        value.kind() == Kind::Undef
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
        !unsafe { f.code_of() }.code().may_suspend()
    }

    unsafe fn call_now<A>(&self, f: &Value, ctx: &mut acvus_extern::Ctx<'_, Self>, args: A) -> Value
    where
        A: acvus_extern::IntoRun<Self>,
    {
        const {
            assert!(
                A::WIDTH <= crate::regs::MAX_ARG_SLOTS,
                "a closure takes at most one cell of arguments"
            )
        };
        let rt = ctx.rt;
        // SAFETY: the frame is read and written in place, never moved out
        // or replaced.
        let frame = unsafe { ctx.frame_mut() };
        // SAFETY: the runtime crosses a closure's arguments at the types its
        // declaration names, which `IntoRun` carries.
        args.into_run(unsafe { acvus_extern::Crossing::new(self) }, frame.run_mut(A::WIDTH));
        // SAFETY: the type checker admits only a closure value here, so its
        // code word names the `Code` this enters and the captures the entry
        // reads.
        unsafe { f.code_of().code().call(*f, rt, frame, A::WIDTH as u16) }
    }

    unsafe fn call_n<'a>(&'a self, f: &'a Value, args: &mut [Value]) -> Self::CallFuture<'a> {
        self.run(f, args)
    }
}

/// One of the machine's registers holding a bare word rather than a value of
/// the language: half of the pair a slice occupies (RFC-0047 rule 6), which
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
        Box::pin(crate::machine::fn_value_call(f, self, args))
    }
}

acvus_extern::cross_one_value!(Value, at AcvusRuntime);

/// The runtime's own value crosses as itself: nothing to convert, and a
/// reference to one is read through the word that names it (RFC-0039).
// SAFETY: the runtime's own value crosses as itself: `erase` and `materialize`
// hand the word through unchanged, and the capability is not used.
unsafe impl acvus_extern::OneValue<AcvusRuntime> for Value {
    const STORED_AS_VALUE: bool = true;

    fn erase(self, _: acvus_extern::Crossing<'_, AcvusRuntime>) -> Value {
        self
    }

    unsafe fn materialize(_: acvus_extern::Crossing<'_, AcvusRuntime>, value: Value) -> Self {
        value
    }
}

impl acvus_extern::Borrowable<AcvusRuntime> for Value {
    unsafe fn deref<'a>(_: &AcvusRuntime, reference: &'a Value) -> &'a Value {
        // SAFETY: the caller's contract: a live reference.
        unsafe { reference.target() }
    }

    unsafe fn deref_mut<'a>(_: &AcvusRuntime, reference: &'a Value) -> &'a mut Value {
        // SAFETY: the caller's contract: a live, exclusively named target.
        unsafe { reference.target_mut() }
    }
}
