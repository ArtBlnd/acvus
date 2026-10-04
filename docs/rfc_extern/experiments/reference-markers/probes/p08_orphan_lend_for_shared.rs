// expect: E0117
use model::{Crossing, Shared, Value};
pub struct Local;
impl Crossing for Shared<'static, Local> { fn give(self) -> Value { Value { p: 0, m: 0 } } }
