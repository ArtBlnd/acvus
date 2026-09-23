pub mod code;
pub mod executor;
mod interpreter;
pub mod journal;
pub mod layout;
/// The walk reads `Op::successor`, `Op::owns` and `Named::name`, which the
/// release trait does not carry.
#[cfg(any(debug_assertions, feature = "probe"))]
pub mod listing;
pub mod machine;
mod ops;
pub mod prepare;
pub mod regs;
mod runtime;
pub mod space;
mod value;
pub mod vtable;

pub use code::{Code, CodeBody, Prepared};
pub use executor::{Executor, SequentialExecutor, TokioExecutor};
pub use interpreter::{Args, Executable, Interpreter, InterpreterContext};
pub use journal::{ContextWrite, InMemoryContext, RuntimeContext};
pub use layout::Hooks as SpaceHooksByType;
pub use machine::fn_value_call;
pub use ops::chain::{ChainTy, LeafRead, Node as ChainNode, Nodes as ChainNodes, Reads};
/// The handlers an `Index` runs, checked and, for a bound the MIR proves,
/// unchecked (RFC-0047 rule 7).
pub use ops::index as index_handlers;
pub use prepare::{PrepareCtx, prepare_module};
pub use runtime::{AcvusRuntime, ExternHandler};
pub use space::{
    Commit, DirStore, Head, Log, MemoryStore, Mode, Node, NodeKind, Plain, Record, Space,
    SpacePage, Store, hex,
};
pub use value::{Array, FnValue, HandleValue, Kind, Object, Place, Tuple, Value, VariantValue};
pub use vtable::{Composite, Vtable};
