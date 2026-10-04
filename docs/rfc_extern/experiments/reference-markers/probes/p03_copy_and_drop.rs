// expect: E0184
#[derive(Clone, Copy)]
pub struct H(u64);
impl Drop for H { fn drop(&mut self) {} }
