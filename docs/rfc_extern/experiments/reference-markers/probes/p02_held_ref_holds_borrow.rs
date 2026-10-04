// expect: E0502
// A reference held uniformly has `Uniform`'s drop, so its borrow lasts until it drops.
use model::Uniform;
pub fn f() { let mut owner = Uniform::new(1i64); let held = Uniform::from_ref(owner.as_ref()); let _ = held.bits(); let _m = owner.as_mut(); }
