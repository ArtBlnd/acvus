//! A model of the runtime contract as kinds, and of a nested type crossed as
//! its layers' kinds composed, the innermost first. `contract` is the
//! contract, `boundary` the extern side's crossing, and `slots` and `boxes`
//! two runtimes over the one contract. The kinds and their operations are the
//! probe's means, not decided primitives.

pub mod boundary;
pub mod boxes;
pub mod contract;
mod laid;
pub mod slots;

pub use boundary::{ByMut, ByRef, Calling, CallingMut, CallingOnce, Cross, Effect0, EffectVar, Generic, Lend, Mut, Open, Ref, TypeVar, Uniform, Uniformly, scope, scope_mut, scope_once};
pub use contract::{
    EnumLayout, EnumTable, Enums, Functions, ObjectLayout, ObjectTable, Objects, Opaques, Options, Programs, Results, Runtime, Scalar,
    Scalars, Strings, Structures, Tuples, VariantTable, Views,
};

/// A runtime whose held values can be counted, so that a test sees a leak or
/// a release of a value still held.
pub trait Counted: Runtime {
    /// The number of values held apart from a `Value`'s own bits.
    fn live() -> usize;
}
