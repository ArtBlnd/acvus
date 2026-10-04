// expect: E0597
// `Uniform`'s drop has no `may_dangle`, so a held reference's lifetime lasts until the Vec drops.
use model::{Shared, Uniform};
pub fn f() { let mut v: Vec<Uniform<Shared<'_, i64>>> = Vec::new(); let x = Uniform::new(5i64); v.push(Uniform::from_ref(x.as_ref())); }
