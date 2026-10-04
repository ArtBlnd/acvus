// expect: E0366
use std::marker::PhantomData;
pub struct Owned;
pub struct H<M>(u64, PhantomData<M>);
impl Drop for H<Owned> { fn drop(&mut self) {} }
