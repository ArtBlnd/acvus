// expect: E0621
// The result cannot borrow the argument it does not name.
use model::{EffectVar, Ref, TypeVar};
pub fn pick_wrong<'x, T: TypeVar, E: EffectVar>(a: &'x T, b: &T, f: impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'q, T>) -> Ref<'x, T> { f(a, b) }
