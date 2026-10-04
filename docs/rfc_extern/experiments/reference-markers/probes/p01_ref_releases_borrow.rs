// expect: pass
// A `Ref` has no drop, so its borrow ends at its last use.
use model::Uniform;
pub fn f() { let mut owner = Uniform::new(1i64); let r = owner.as_ref(); let _ = r.bits(); let _m = owner.as_mut(); }
