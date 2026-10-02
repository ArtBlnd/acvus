use acvus_mir::decl::Scheme;
use acvus_mir::ty::{Poly, Types};
use acvus_utils::{Interner, QualifiedRef};

use crate::kind::{EnumKind, FnKind, ObjectKind, OpaqueKind, OptionKind, RefKind, ResultKind, ScalarKind, StringKind, TupleKind};
use crate::repr::{Boxed, Words};

/// What every runtime implements and every extern fn is written against. A
/// value is the runtime's `Value`, reached only through the kind of its type,
/// each kind an associated type with functions of its own.
pub trait Runtime: Sized + 'static {
    type Value: Copy;

    type Scalar: ScalarKind<Self, i8>
        + ScalarKind<Self, i16>
        + ScalarKind<Self, i32>
        + ScalarKind<Self, i64>
        + ScalarKind<Self, u8>
        + ScalarKind<Self, u16>
        + ScalarKind<Self, u32>
        + ScalarKind<Self, u64>
        + ScalarKind<Self, f64>
        + ScalarKind<Self, bool>
        + ScalarKind<Self, char>
        + ScalarKind<Self, ()>;
    type String: StringKind<Self>;
    type Tuple: TupleKind<Self>;
    type Object: ObjectKind<Self>;
    type Option: OptionKind<Self>;
    type Result: ResultKind<Self>;
    type Enum: EnumKind<Self>;
    type Fn: FnKind<Self>;
    type Opaque: OpaqueKind<Self>;
    type Ref: RefKind<Self>;

    /// A second owner of `value`, for a boundary that must keep it alive on
    /// its own, such as a spawn.
    fn share(&self, value: &Self::Value) -> Self::Value;

    /// Ends one owner of a value the runtime holds, and with the last owner
    /// the value and every part it owns. It takes no runtime because an owner
    /// that releases its value on drop has none to give.
    fn release(value: Self::Value);
}

/// The storage the extern crate's unsafe kinds stand on, which a runtime
/// implements in safe code: it holds a `Boxed` behind a `Value`, lends it
/// back shared or exclusive, and gives it back whole; and it holds a
/// reference's `Words`.
pub trait Stores: Runtime {
    fn hold(&self, boxed: Boxed) -> Self::Value;
    fn lend<O>(&self, value: &Self::Value, lent: impl FnOnce(&Boxed) -> O) -> O;
    fn lend_mut<O>(&self, value: &mut Self::Value, lent: impl FnOnce(&mut Boxed) -> O) -> O;
    fn unhold(&self, value: Self::Value) -> Boxed;
    fn hold_words(&self, words: Words) -> Self::Value;
    fn words(&self, value: &Self::Value) -> Words;
}

/// An extern fn as a registry takes it: a type rather than an erased
/// function, so that a runtime can build its own operations around the call.
/// The macros implement it.
pub trait ExternFn<R>: 'static
where
    R: Runtime,
{
    /// The call's arguments, one `Value` each: a `[R::Value; N]`.
    type Args: Arguments<R>;

    /// The declaration the compiler takes, its types interned in `types`.
    fn declare(types: &mut Types<Poly>, names: &Interner) -> (QualifiedRef, Scheme);

    /// Runs the call. Every argument is the call's own, and so is the result
    /// it gives back.
    fn call(rt: &R, args: Self::Args) -> R::Value;
}

/// A call's argument run, which a runtime fills knowing only its extern fn.
pub trait Arguments<R>: Sized
where
    R: Runtime,
{
    fn from_fn(value_at: impl FnMut(usize) -> R::Value) -> Self;
}

impl<R, const N: usize> Arguments<R> for [R::Value; N]
where
    R: Runtime,
{
    fn from_fn(value_at: impl FnMut(usize) -> R::Value) -> Self {
        std::array::from_fn(value_at)
    }
}

/// A runtime that takes extern fns.
pub trait Registry: Runtime {
    fn register<F>(&mut self)
    where
        F: ExternFn<Self>;
}
