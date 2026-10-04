// expect: E0310
// A closure lent for the call cannot be kept past it.
use model::TypeVar;
pub fn keep<T: TypeVar>(f: impl Fn(&T) -> bool) -> Box<dyn Fn(&T) -> bool> { Box::new(f) }
