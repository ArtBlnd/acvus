// expect: pass
// `push<T>(v: &mut Stack<T>, x: T)` and its `#` member `push_f64` at `T = f64`.
use pair::*;
// The uniform member's family, every position put through its native form.
pub type PushSig<T> = for<'l0> fn(<&'l0 mut Stack<T> as Native>::Out, <T as Native>::Out) -> <() as Native>::Out;
// `#[extern_fn(specializes = push<f64>)] fn push_f64(v: &mut Stack<f64>, x: f64)`.
pub fn check() { same::<PushSig<f64>, for<'l0> fn(&'l0 mut Stack<f64>, f64)>(); }
