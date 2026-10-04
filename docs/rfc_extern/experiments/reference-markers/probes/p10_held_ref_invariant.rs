// expect: lifetime may not live long enough
use model::{Shared, Uniform};
pub fn shorten<'x, 'a: 'x>(v: &'x Vec<Uniform<Shared<'a, i64>>>) -> &'x Vec<Uniform<Shared<'x, i64>>> { v }
