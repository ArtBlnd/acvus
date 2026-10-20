// expect: E0277
// A run's elements are `Value`s, held by a container the runtime reads, so a
// run of native data is not lent as a view (references §2): its element is
// no variable and no `Uniform`.
use model::{Cross, Runtime};

pub fn give_numbers<R>(numbers: &[i64]) -> R::Value
where
    R: Runtime,
{
    <&[i64] as Cross<R>>::give(numbers)
}
