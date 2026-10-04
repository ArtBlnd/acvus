// expect: lifetime may not live long enough
use model::{Shared, Uniform};
pub fn keep<'a>(held: Uniform<Shared<'a, i64>>) -> Box<dyn std::any::Any> { Box::new(held) }
