//! Stand-ins for what an `#[opaque(..)]` type's fields name: the runtime,
//! a `Uniform`, the sealed `TypeVar`, the effect and identity markers, a
//! kept function's type and an `Instance`. The traits and types are the
//! probe's means, not decided primitives.

use core::marker::PhantomData;

pub trait Runtime {
    type Value: Copy;
}

pub struct Uniform<R: Runtime, X> {
    pub value: R::Value,
    pub marker: PhantomData<X>,
}

mod seal {
    pub trait TypeVar {}
}

impl<R: Runtime, X> seal::TypeVar for Uniform<R, X> {}

/// Holds for every `Uniform` and for nothing else (variables §3).
pub trait TypeVar<R: Runtime>: seal::TypeVar {}

impl<R: Runtime, X> TypeVar<R> for Uniform<R, X> {}

pub trait EffectVar {}

pub trait IdentityVar {}

/// The one fixed uninhabited marker a type variable is monomorphized at.
pub enum Generic {}

pub enum Effect0 {}

impl EffectVar for Effect0 {}

pub enum Identity0 {}

impl IdentityVar for Identity0 {}

/// A kept function's type, its effect a variable (functions §2).
pub struct Pred<'f, T, E: EffectVar>(pub PhantomData<(&'f (), T, E)>);

/// The name of a generic function `next<It, T, E>`, whose effect is its
/// variable `E` (variables §1).
pub enum Next {}

/// A value and its impl of one generic function, as one (impls §2). `C` is
/// the choice of the generic function's variables, its effect variable's
/// included.
pub struct Instance<R: Runtime, G, V: TypeVar<R>, C> {
    pub value: V,
    pub witness: R::Value,
    pub marker: PhantomData<(G, C)>,
}

pub enum Probe {}

impl Runtime for Probe {
    type Value = usize;
}
