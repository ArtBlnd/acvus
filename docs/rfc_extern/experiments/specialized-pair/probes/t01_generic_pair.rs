// expect: pass
// `push<T>(v: &mut Stack<T>, x: T)` and its `#` member `push_f64` at `T = f64`.
use pair::*;
// The uniform member's family: its parameters and result as one tuple, its
// lifetimes in the order they first appear, put through its native form.
pub type PushSig<'l0, T> = <((&'l0 mut Stack<T>, T), ()) as Native>::Out;
// `#[extern_fn(specializes = push<f64>)] fn push_f64(v: &mut Stack<f64>, x: f64)`.
pub fn check<'l0>() { same::<PushSig<'l0, f64>, ((&'l0 mut Stack<f64>, f64), ())>(); }
