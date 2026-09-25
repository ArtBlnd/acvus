//! The contract a runtime signs to run every declared ExternFn.
//!
//! A runtime chooses its own value representation. Each method builds or
//! opens one shape of value; a shape that is not there is the runtime's
//! own error.

use std::future::{Future, Ready};

/// The contract a host signs to run declared ExternFns. A `Value` is opaque;
/// `materialize`/`erase` are the whole extraction/construction pair; `call_*`
/// run a value that is a closure. A host owns its `Value` representation.
pub trait Runtime: Sized + Send + Sync + 'static {
    type Value: crate::Cross<Self, Form = crate::One>
        + crate::Borrowable<Self>
        + crate::Release
        + Copy
        + Default;
    /// The frame a handler calls a closure on: the cells above the calling
    /// frame (RFC-0052 rule 7). This is the state **itself**, not a borrow of
    /// it: a `Ctx` owns one, and every crossing hands out `&mut Ctx` from the
    /// owner of that `Ctx`. `Ctx`'s frame field is private and its one
    /// accessor, `Ctx::frame_mut`, is `unsafe`; the one other `&mut Ctx` a
    /// caller can hold beside it, `ctx_of`'s, is `unsafe` too, as is
    /// `Ctx::new`, so no safe code swaps two frames or two `Ctx`s. The state
    /// therefore never moves out of the frame below and
    /// `Regs::take_window`'s "one handle to these cells" holds unchanged.
    ///
    /// The lifetime is here for hosts whose window borrows something; a host
    /// whose state borrows nothing ignores it.
    type Frame<'a>: Send;
    /// The cells a call that outlives the frame it was made on runs its
    /// closures in, with the `Ctx` over them. A future the caller waits for
    /// cannot borrow the caller's window — it is `'static` and the caller's
    /// frame holds it — so the `async` glue owns one of these and lends its
    /// `Ctx` per call.
    ///
    /// `Sync` is not asked for: a `Ctx` holds `Rt::Frame`, which is only
    /// `Send`, and every future here is `dyn Future + Send`.
    type Rooted<'a>: Send;
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

    /// Obligation across artifacts: `Signature::call_later` branches on
    /// the task this value carries to choose the `fn` type of the word.
    fn instance_value(entry: &crate::InstanceEntry<Self>) -> Self::Value;

    /// # Safety
    /// `value` was made by `instance_value` from an entry that outlives
    /// `'a`.
    unsafe fn instance_entry<'a>(value: &'a Self::Value) -> &'a crate::InstanceEntry<Self>;

    fn rooted(&self) -> Self::Rooted<'_>;
    /// The `Ctx` the rooted cells carry, lent from its owner. A handler
    /// that calls a closure on cells of its own does so through
    /// `Closure::call_rooted`, which never lends this `Ctx` out.
    ///
    /// # Safety
    /// The `Ctx` is not exchanged with another: the caller moves no `Ctx`
    /// or frame out through the reference and writes no other into it, so
    /// the `Ctx` a handler was called with keeps the frame its owner built
    /// it over.
    unsafe fn ctx_of<'a, 'r>(rooted: &'r mut Self::Rooted<'a>) -> &'r mut crate::Ctx<'a, Self>
    where
        'a: 'r;

    /// The operation one extern call site runs as. `H::WIDTH` says which
    /// form the call takes — how many of the runtime's values its arguments
    /// are and where its result goes (RFC-0044 rule 3, RFC-0047 rule 6)
    /// — and a runtime that lays registers by form reads it there;
    /// one that runs every call where it stands ignores it.
    fn op<H>(handler: H, shape: Self::CallShape) -> Self::Op
    where
        H: crate::handler::Handler<Self>;

    /// One call of a run the host fused (RFC-0044 rule 7): a call whose
    /// arguments are at most two of the runtime's values and whose result
    /// is one. A runtime that fuses refuses any other form here.
    fn fused<H>(handler: H, shape: Self::FusedShape) -> Self::FusedCall
    where
        H: crate::handler::Handler<Self>;

    fn async_extern_op<H>(handler: H, shape: Self::AsyncShape) -> Self::Op
    where
        H: crate::handler::AsyncCall<Self>;

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
    /// As `value_as_ref`, and `loan_ended` is handed the value once the
    /// borrow ends.
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
    /// As `inline_ref`, and `loan_ended` is handed the value once the
    /// borrow ends.
    unsafe fn inline_mut<T>(value: &mut Self::Value) -> &mut T
    where
        T: crate::obj::Inline;

    /// An exclusive borrow of what `storage` holds, lent in place by
    /// `inline_mut`, `value_as_mut` or `deref_mut`, has ended: the storage
    /// holds the runtime's own encoding of whatever the borrow wrote. A
    /// runtime whose borrow is its storage's own Rust value does nothing; one
    /// that keeps an inline value in a word re-encodes the word
    /// (`repr::settle`), since a `&mut i8` writes one byte of it.
    fn loan_ended(storage: &mut Self::Value);

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
    /// `loan_ended` is handed the storage once the borrow ends.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a, T>(&self, reference: &'a Self::Value) -> &'a mut T
    where
        T: Send + Sync + 'static;

    /// The language's `Option` (RFC-0039 rule 6).
    fn none(&self) -> Self::Value;
    fn some(&self, payload: Self::Value) -> Self::Value;
    fn is_none(&self, value: &Self::Value) -> bool;
    /// # Panics
    /// The value is `None`.
    fn unwrap_some(&self, value: Self::Value) -> Self::Value;

    /// The payload of a `Some`, where it lies: an option is its payload
    /// (RFC-0039), so a `Some(v)`'s storage is `v`'s own.
    ///
    /// # Safety
    /// `value` holds an option whose payload is not itself an option. A
    /// runtime is free to distinguish `Some(None)` from `None` by a word the
    /// payload does not carry, and then the payload has no storage to name.
    unsafe fn some_at<'a>(&self, value: &'a Self::Value) -> Option<&'a Self::Value>;

    /// # Safety
    /// As `some_at`, and the storage is exclusively named for `'a`.
    unsafe fn some_at_mut<'a>(&self, value: &'a mut Self::Value) -> Option<&'a mut Self::Value>;

    /// Obligation across artifacts: what `erase` of a
    /// `Result<Owned<Self>, Owned<Self>>` writes is the runtime's own choice.
    /// `acvus-interpreter` writes the flat variant `[tag, payload]` (RFC-0050
    /// rule 8), and a runtime that stores what it is handed keeps the Rust
    /// value. This method reads whichever its own `erase` wrote.
    ///
    /// # Safety
    /// `value` holds what `erase` of a `Result<Owned<Self>, Owned<Self>>`
    /// wrote, live for `'a`.
    unsafe fn result_at<'a>(&self, value: &'a Self::Value) -> Result<&'a Self::Value, &'a Self::Value>;

    /// # Safety
    /// As `result_at`, and the storage is exclusively named for `'a`. A word
    /// written through the payload is owned by no other holder, and the word
    /// it replaces is released or moved out by the writer, as
    /// `Owned::value_mut`'s.
    unsafe fn result_at_mut<'a>(
        &self,
        value: &'a mut Self::Value,
    ) -> Result<&'a mut Self::Value, &'a mut Self::Value>;

    /// The name a field key is at run time (RFC-0050 rule 8).
    fn symbol(&self, name: &str) -> acvus_utils::Astr;

    /// How a tag lies in a register is the runtime's contract (RFC-0059),
    /// so `acvus-extern-macro`'s derived enum crossing calls these two and
    /// never reads the word itself.
    fn variant_tag(&self, name: &str) -> Self::Value;

    /// # Safety
    /// `tag` is the tag register of a variant this runtime wrote.
    unsafe fn tag_symbol(&self, tag: &Self::Value) -> acvus_utils::Astr;

    /// The word rule 8 leaves at a register whose component this value does
    /// not have; `interpreter::ops::pattern::TestObjectKey` is what reads one
    /// back.
    fn undef(&self) -> Self::Value;

    fn is_undef(&self, value: &Self::Value) -> bool;

    /// A slice written into the run its result is: two of the runtime's
    /// values, one per register of the pair the machine keeps a slice in
    /// (RFC-0047 rule 6). A runtime whose value cannot carry a bare word
    /// holds no slice and says so here.
    fn slice_into_run(&self, words: crate::repr::Words, out: &mut [Self::Value]);

    /// The same slice read back out of that run.
    ///
    /// # Safety
    /// `run` is what `slice_into_run` wrote, and the elements it names are
    /// live and unmoved — the loan the slice holds is what keeps them so
    /// (RFC-0018).
    unsafe fn slice_from_run(&self, run: &[Self::Value]) -> crate::repr::Words;

    /// A reference value naming `target`'s storage (RFC-0018): what a
    /// handler passes to a closure whose parameter is `&T` / `&mut T`.
    ///
    /// # Safety
    /// The reference is used only while `target` is live and unmoved; the
    /// closure it is passed to keeps it no longer than the call.
    unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

    /// `value` laid out as canonical bytes by `ty` and appended to `out`
    /// (RFC-0033), for `Args::encode`.
    ///
    /// # Safety
    /// `value` holds a value this runtime crossed at `ty`, live for the call.
    unsafe fn encode(
        &self,
        ty: &acvus_mir::ty::Ty,
        value: &Self::Value,
        out: &mut Vec<u8>,
    ) -> crate::SpaceResult<()>;

    /// A future that waits `d` (RFC-0075 rule 1). It borrows nothing of
    /// `&self`: `use<Self>` is its whole capture, so a handler holds it past
    /// the call that made it. A timer is the host's, so there is no default.
    fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send + use<Self>;

    /// Whether running `f` reaches its result without suspending.
    /// `Closure` asks once, when it is built. A runtime whose closures
    /// can always suspend answers `false`, which is the default.
    fn call_is_sync(&self, _f: &Self::Value) -> bool {
        false
    }

    /// Run `f` to its result now, reached only where `call_is_sync`
    /// answered true for this same value. Each argument crosses straight into
    /// the parameter register `frame` holds for it (RFC-0052 rule 7). A runtime
    /// that answers `false` above is never asked, and keeps the default.
    ///
    /// # Safety
    /// The call comes through `Closure`, which is where a value known to be
    /// one of this runtime's closures, called at the types its declaration
    /// names, is the only thing that reaches here.
    unsafe fn call_now<A>(
        &self,
        _f: &Self::Value,
        _ctx: &mut crate::Ctx<'_, Self>,
        _args: A,
    ) -> Self::Value
    where
        A: crate::IntoRun<Self>,
    {
        unreachable!("`call_is_sync` answered false for every closure of this runtime")
    }

    /// Run the closure `f`; each argument moves into the callee's
    /// parameter. The two fixed arities are `call_n` at that many values,
    /// which a runtime overrides only where it lays them differently.
    ///
    /// # Safety
    /// As `call_now`'s.
    unsafe fn call_0<'a>(&'a self, f: &'a Self::Value) -> Self::CallFuture<'a> {
        // SAFETY: the caller's contract.
        unsafe { self.call_n(f, &mut []) }
    }
    /// # Safety
    /// As `call_now`'s.
    unsafe fn call_1<'a>(&'a self, f: &'a Self::Value, a: Self::Value) -> Self::CallFuture<'a> {
        // SAFETY: the caller's contract.
        unsafe { self.call_n(f, &mut [a]) }
    }
    /// # Safety
    /// As `call_now`'s.
    unsafe fn call_n<'a>(
        &'a self,
        f: &'a Self::Value,
        args: &mut [Self::Value],
    ) -> Self::CallFuture<'a>;
}

/// A runtime that holds no values: for registering declarations where
/// nothing will ever run.
#[derive(Clone, Copy)]
pub struct TypesOnly;

mod sealed {
    /// Unnameable outside this crate, so `TypesOnly` is its one impl: what
    /// reads an `Erased`'s `T` for the checker holds at no runtime that
    /// makes values (`Canonical`).
    pub trait HoldsNoValues: crate::Runtime {}
}

pub(crate) use sealed::HoldsNoValues;

impl HoldsNoValues for TypesOnly {}

impl crate::Release for () {
    fn release(self) {}
}

fn no_values() -> ! {
    panic!("TypesOnly runtime holds no values")
}

impl Runtime for TypesOnly {
    type Value = ();
    type Frame<'a> = ();
    type Rooted<'a> = crate::Ctx<'a, TypesOnly>;
    type CallFuture<'a> = Ready<()>;
    type Op = crate::handler::DirectOp<TypesOnly>;
    type CallShape = ();
    type AsyncShape = ();
    type FusedCall = crate::handler::DirectOp<TypesOnly>;
    type FusedShape = ();

    fn instance_value(_: &crate::InstanceEntry<TypesOnly>) {
        no_values()
    }

    unsafe fn instance_entry<'a>(_: &'a ()) -> &'a crate::InstanceEntry<TypesOnly> {
        no_values()
    }

    fn rooted(&self) -> crate::Ctx<'_, TypesOnly> {
        // SAFETY: the frame is `()`, which names no cells.
        unsafe { crate::Ctx::new(self, ()) }
    }
    unsafe fn ctx_of<'a, 'r>(
        rooted: &'r mut crate::Ctx<'a, TypesOnly>,
    ) -> &'r mut crate::Ctx<'a, TypesOnly>
    where
        'a: 'r,
    {
        rooted
    }

    crate::direct_call_forms!();

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
    fn loan_ended(_: &mut ()) {
        no_values()
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
    unsafe fn encode(&self, _: &acvus_mir::ty::Ty, _: &(), _: &mut Vec<u8>) -> crate::SpaceResult<()> {
        Err(crate::SpaceError::new("TypesOnly runtime holds no values"))
    }
    /// Nothing runs here, so nothing waits beside it; the thread sleeps.
    fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send + use<> {
        async move { std::thread::sleep(d) }
    }
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
    unsafe fn some_at<'a>(&self, _: &'a ()) -> Option<&'a ()> {
        no_values()
    }
    unsafe fn some_at_mut<'a>(&self, _: &'a mut ()) -> Option<&'a mut ()> {
        no_values()
    }
    unsafe fn result_at<'a>(&self, _: &'a ()) -> Result<&'a (), &'a ()> {
        no_values()
    }
    unsafe fn result_at_mut<'a>(&self, _: &'a mut ()) -> Result<&'a mut (), &'a mut ()> {
        no_values()
    }
    fn symbol(&self, _: &str) -> acvus_utils::Astr {
        panic!("TypesOnly runtime holds no values")
    }
    fn variant_tag(&self, _: &str) {
        no_values()
    }
    unsafe fn tag_symbol(&self, _: &()) -> acvus_utils::Astr {
        panic!("TypesOnly runtime holds no values")
    }
    fn undef(&self) {
        no_values()
    }
    fn is_undef(&self, _: &()) -> bool {
        no_values()
    }
    fn slice_into_run(&self, _: crate::repr::Words, _: &mut [()]) {
        no_values()
    }
    unsafe fn slice_from_run(&self, _: &[()]) -> crate::repr::Words {
        no_values()
    }
    unsafe fn call_n<'a>(&'a self, _: &'a (), _: &mut [()]) -> Self::CallFuture<'a> {
        no_values()
    }
}
