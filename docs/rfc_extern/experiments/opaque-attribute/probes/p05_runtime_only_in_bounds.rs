// expect: E0392
// A runtime named only so that the type arguments can be bounded, and used
// by no field, is a parameter Rust refuses as unused: whatever writes the
// bounds must also make the runtime used.
use model::*;

pub struct Stack<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    pub items: Vec<T>,
}
