// expect: pass
// The model's own `Uniform` has `as_mut` on `X: Lend` and on `Exclusive`; this
// calls both on their markers.
use model::{Exclusive, Uniform};
pub fn f() { let mut owner = Uniform::new(1i64); let mut held = Uniform::<Exclusive<'_, i64>>::from_mut(owner.as_mut()); held.as_mut().set(2); }
