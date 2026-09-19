//! The contract a runtime signs to run every declared ExternFn.
//!
//! A runtime chooses its own value representation. Each method builds or
//! opens one shape of value; a shape that is not there is the runtime's
//! own error.

use std::any::TypeId;
use std::future::{Future, Ready};

use crate::func::CallToken;

/// The contract a host signs to run declared ExternFns. A `Value` is opaque;
/// `materialize`/`erase` are the whole extraction/construction pair; `call_*`
/// run a value that is a closure. A host owns its `Value` representation.
pub trait Runtime: Sized + Send + Sync + 'static {
    type Value: crate::OneValue<Self> + crate::FromValue<Self> + crate::Release + Copy + Default;
    /// The frame a handler calls a closure on: the window above the calling
    /// frame, lent for the call's duration (RFC-0050 rule 6).
    type Frame<'a>: Send
    where
        Self: 'a;
    /// The cells a call that outlives the frame it was made on runs its
    /// closures in. A future the caller waits for cannot borrow the caller's
    /// window — it is `'static` and the caller's frame holds it — so the
    /// `async` glue owns one of these and lends a `Frame` out of it per call.
    type Rooted: Send + Sync;
    type CallFuture<'a>: Future<Output = Self::Value> + Send + 'a
    where
        Self: 'a;
    /// What one extern call site runs as.
    type Op;
    /// What the host decided about a call site before the handler's type is
    /// known: where the result goes, where the arguments are, what follows.
    type CallShape;
    type AsyncShape;
    /// One call of a run of extern calls the host fused.
    type FusedCall;
    type FusedShape;

    fn rooted(&self) -> Self::Rooted;
    fn frame_of(rooted: &mut Self::Rooted) -> Self::Frame<'_>;

    /// One entry per call form (RFC-0044 stage 2c, RFC-0047 amended rule 2).
    /// A declaration's arity names its entry where the glue is written, so a
    /// runtime instantiates each handler's operation at that form alone.
    fn op_no_argument<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;
    fn op_one_argument<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;
    fn op_two_arguments<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;
    fn op_three_arguments<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;
    fn op_wide<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;
    fn op_slice<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;

    fn fused_no_argument<H>(handler: H, shape: Self::FusedShape) -> Self::FusedCall
    where
        H: crate::handler::Handler<Self>;
    fn fused_one_argument<H>(handler: H, shape: Self::FusedShape) -> Self::FusedCall
    where
        H: crate::handler::Handler<Self>;
    fn fused_two_arguments<H>(handler: H, shape: Self::FusedShape) -> Self::FusedCall
    where
        H: crate::handler::Handler<Self>;

    fn async_extern_op<H>(handler: H, shape: Self::AsyncShape) -> Self::Op
    where
        H: crate::handler::AsyncCall<Self>;

    /// The `T` of the `erase::<T>` that made this value, when the value
    /// records it. `downcast` and `Erased::from_value` trust this answer
    /// with a `materialize::<T>`, so a runtime answers only from the record.
    fn type_of(&self, value: &Self::Value) -> Option<TypeId>;
    /// The runtime's name for the type `type_of` reports, for a panic
    /// message; a runtime that keeps no name answers `None`.
    fn type_name_of(&self, value: &Self::Value) -> Option<&'static str>;

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

    /// The language's `Option` (RFC-0022).
    fn none(&self) -> Self::Value;
    fn some(&self, payload: Self::Value) -> Self::Value;
    fn is_none(&self, value: &Self::Value) -> bool;
    /// # Panics
    /// The value is `None`.
    fn unwrap_some(&self, value: Self::Value) -> Self::Value;

    /// The name a field key is at run time (RFC-0032).
    fn symbol(&self, name: &str) -> acvus_utils::Astr;

    /// A slice written into the run its result is: two of the runtime's
    /// values, one per register of the pair the machine keeps a slice in
    /// (RFC-0047 amended). A runtime whose value cannot carry a bare word
    /// holds no slice and says so here.
    fn slice_into_run(&self, words: crate::slice::Words, out: &mut [Self::Value]);

    /// The same slice read back out of that run.
    ///
    /// # Safety
    /// `run` is what `slice_into_run` wrote, and the elements it names are
    /// live and unmoved — the loan the slice holds is what keeps them so
    /// (RFC-0018).
    unsafe fn slice_from_run(&self, run: &[Self::Value]) -> crate::slice::Words;

    /// A reference value naming `target`'s storage (RFC-0018): what a
    /// handler passes to a closure whose parameter is `&T` / `&mut T`.
    ///
    /// # Safety
    /// The reference is used only while `target` is live and unmoved; the
    /// closure it is passed to keeps it no longer than the call.
    unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

    /// Whether running `f` reaches its result without suspending.
    /// `Fn0`/`Fn1`/… ask once, when they are built, and a runtime whose
    /// closures can always suspend answers `false`.
    fn call_is_sync(&self, f: &Self::Value) -> bool;
    /// Run `f` to its result now, reached only where `call_is_sync`
    /// answered true for this same value. Each argument crosses straight into
    /// the parameter register `frame` holds for it (RFC-0052 §7).
    fn call_now<A>(
        &self,
        f: &Self::Value,
        frame: &mut Self::Frame<'_>,
        args: A,
        token: CallToken,
    ) -> Self::Value
    where
        A: crate::IntoRun<Self>;

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
        args: &mut [Self::Value],
        token: CallToken,
    ) -> Self::CallFuture<'a>;
}

/// A runtime that holds no values: for registering declarations where
/// nothing will ever run.
#[derive(Clone, Copy)]
pub struct TypesOnly;

impl crate::Release for () {
    fn release(self) {}
}

fn no_values() -> ! {
    panic!("TypesOnly runtime holds no values")
}

impl crate::FromValue<TypesOnly> for () {
    fn from_value(_: &TypesOnly, value: ()) {
        value
    }
}

impl Runtime for TypesOnly {
    type Value = ();
    type Frame<'a> = ();
    type Rooted = ();
    type CallFuture<'a> = Ready<()>;
    type Op = crate::handler::DirectOp<TypesOnly>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = crate::handler::DirectOp<TypesOnly>;
    type FusedShape = ();

    fn rooted(&self) {}
    fn frame_of(_: &mut ()) {}

    crate::direct_call_forms!();

    fn type_of(&self, _: &()) -> Option<TypeId> {
        None
    }
    fn type_name_of(&self, _: &()) -> Option<&'static str> {
        None
    }
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
    fn none(&self) {
        no_values()
    }
    fn some(&self, _: ()) {
        no_values()
    }
    fn is_none(&self, _: &()) -> bool {
        no_values()
    }
    fn unwrap_some(&self, _: ()) {
        no_values()
    }
    fn symbol(&self, _: &str) -> acvus_utils::Astr {
        panic!("TypesOnly runtime holds no values")
    }
    fn slice_into_run(&self, _: crate::slice::Words, _: &mut [()]) {
        no_values()
    }
    unsafe fn slice_from_run(&self, _: &[()]) -> crate::slice::Words {
        no_values()
    }
    fn call_is_sync(&self, _: &()) -> bool {
        false
    }
    fn call_now<A>(&self, _: &(), _: &mut (), _: A, _: CallToken)
    where
        A: crate::IntoRun<Self>,
    {
        no_values()
    }
    fn call_0<'a>(&'a self, _: &'a (), _: CallToken) -> Self::CallFuture<'a> {
        no_values()
    }
    fn call_1<'a>(&'a self, _: &'a (), _: (), _: CallToken) -> Self::CallFuture<'a> {
        no_values()
    }
    fn call_n<'a>(&'a self, _: &'a (), _: &mut [()], _: CallToken) -> Self::CallFuture<'a> {
        no_values()
    }
}
