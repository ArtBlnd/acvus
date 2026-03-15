mod builtins;
mod error;
mod interner_ctx;
pub mod iter;
mod interpreter;
mod value;

pub use acvus_utils::{ContextRequest, Coroutine, ExternCallRequest, Stepped, YieldHandle};

pub type ValueCoroutine = Coroutine<TypedValue, RuntimeError>;
pub type ValueStepped = Stepped<TypedValue, RuntimeError>;
pub use builtins::{FromValue, IntoValue};
pub use error::{RuntimeError, RuntimeErrorKind};
pub use interpreter::Interpreter;
pub use iter::{IterHandle, SequenceChain};
pub use value::{ConcreteValue, FnValue, OpaqueValue, TypedValue, UnpureValue,PureValue, LazyValue, Value};

/// Set the thread-local interner context for `IntoValue<Option>` / `FromValue<Option>`
/// and `builtin_unwrap`. Must be called before executing extern fns that return Option.
pub fn set_interner_ctx(interner: &acvus_utils::Interner) {
    interner_ctx::set_interner(interner);
}
