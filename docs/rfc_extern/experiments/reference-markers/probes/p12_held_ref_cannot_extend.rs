// expect: lifetime may not live long enough
use model::{Ref, Shared, Uniform};
pub fn extend<'a>(v: &Vec<Uniform<Shared<'a, i64>>>) -> Ref<'static, i64> { v[0].get() }
