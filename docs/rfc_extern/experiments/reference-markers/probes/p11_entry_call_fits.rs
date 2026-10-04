// expect: pass
// A body written with its own lifetimes, called by an entry that opens every loan at one `'call`.
use model::{Ref, Shared, Uniform};
pub fn first<'x, 'a>(v: &'x Vec<Uniform<Shared<'a, i64>>>) -> Ref<'a, i64> { v[0].get() }
pub fn entry<'call>(v: &'call Vec<Uniform<Shared<'call, i64>>>) -> Ref<'call, i64> { first(v) }
