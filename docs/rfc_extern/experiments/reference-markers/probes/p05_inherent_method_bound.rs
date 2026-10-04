// expect: E0592
use std::marker::PhantomData;
pub struct U<X>(PhantomData<fn(X) -> X>);
pub struct Shared<'a, X>(PhantomData<fn() -> &'a X>);
pub trait Lend {}
impl Lend for i64 {}
impl<X> U<X> { pub fn get(&self) -> u8 where X: Lend { 0 } }
impl<'a, X> U<Shared<'a, X>> { pub fn get(&self) -> u16 { 0 } }
