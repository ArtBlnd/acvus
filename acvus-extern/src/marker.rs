use crate::runtime::Runtime;
use crate::uniform::Uniform;

/// What a type variable is monomorphized at: `Uniform<R, Never>`. One fixed
/// uninhabited marker, apart from Rust's `!`, which stands for acvus's `!`.
pub enum Never {}

/// The bound that makes a generic parameter a type variable. It holds for
/// every `Uniform<R, X>`, so an impl that generic code reaches through it
/// covers every view of a value alike, and coherence refuses a second impl
/// for one view. It has no associated type and no supertrait.
pub trait TypeVar<R>
where
    R: Runtime,
{
}

impl<R, X> TypeVar<R> for Uniform<R, X> where R: Runtime {}

/// The bound that makes a generic parameter an effect variable.
pub trait EffectVar {}

/// The bound that makes a generic parameter an identity variable.
pub trait IdentityVar {}

/// What an effect variable is monomorphized at.
pub enum SomeEffect {}

impl EffectVar for SomeEffect {}

/// What an identity variable is monomorphized at.
pub enum SomeIdentity {}

impl IdentityVar for SomeIdentity {}
