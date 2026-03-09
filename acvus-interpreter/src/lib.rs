mod builtins;
mod error;
pub mod extern_fn;
mod interpreter;
mod value;

pub use acvus_utils::{
    Coroutine, EmitStepped, NeedContextStepped, ResumeKey, Stepped, YieldHandle,
};

/// Concrete coroutine type used throughout the interpreter/orchestration.
pub type ValueCoroutine = Coroutine<Value, RuntimeError>;
/// Concrete stepped type.
pub type ValueStepped = Stepped<Value, RuntimeError>;
pub use builtins::{FromValue, IntoValue};
pub use error::{RuntimeError, RuntimeErrorKind};
pub use extern_fn::{ExternFn, ExternFnBody, ExternFnRegistry, ExternFnSig, IntoExternFnBody};
pub use interpreter::Interpreter;
pub use value::{FnValue, OpaqueValue, PureValue, Value};
