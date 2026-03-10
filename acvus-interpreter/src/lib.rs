mod builtins;
mod error;
pub mod extern_fn;
mod interner_ctx;
mod interpreter;
mod value;

pub use acvus_utils::{ContextRequest, Coroutine, Stepped, YieldHandle};

/// Concrete coroutine type used throughout the interpreter/orchestration.
pub type ValueCoroutine = Coroutine<Value, RuntimeError>;
/// Concrete stepped type.
pub type ValueStepped = Stepped<Value, RuntimeError>;
pub use builtins::{FromValue, IntoValue};
pub use error::{RuntimeError, RuntimeErrorKind};
pub use extern_fn::{ExternFn, ExternFnBody, ExternFnRegistry, ExternFnSig, IntoExternFnBody};
pub use interpreter::Interpreter;
pub use value::{ConcreteValue, FnValue, OpaqueValue, PureValue, Value};

/// Set the thread-local interner context for `IntoValue<Option>` / `FromValue<Option>`
/// and `builtin_unwrap`. Must be called before executing extern fns that return Option.
pub fn set_interner_ctx(interner: &acvus_utils::Interner) {
    interner_ctx::set_interner(interner);
}
