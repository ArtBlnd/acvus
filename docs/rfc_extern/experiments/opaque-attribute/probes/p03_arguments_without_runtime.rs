// expect: E0425
// A type argument's bound names the runtime, so a type with a type
// argument and no runtime parameter cannot be bounded.
use model::*;

pub struct Stack<T>
where
    T: TypeVar<R>,
{
    pub items: Vec<T>,
}
