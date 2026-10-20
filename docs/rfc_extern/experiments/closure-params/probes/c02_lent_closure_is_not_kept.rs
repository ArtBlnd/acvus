// expect: E0310
// A closure lent for the call cannot be kept past it.
use model::{EffectVar, TypeVar};
pub fn keep<T: TypeVar, E: EffectVar>(f: impl Fn(&T) -> bool) -> Box<dyn Fn(&T) -> bool> { Box::new(f) }
