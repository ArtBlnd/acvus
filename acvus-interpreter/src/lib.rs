//! The acvus runtime. A host compiles scripts with `Host`, opens a page over
//! a storage and runs entries on it inside `Program::scope`, reads a result
//! through an `Output`, and reads and writes contexts through the page's
//! typed methods (RFC-0090).
//!
//! The runtime's own value word, the machine that runs it, and a run's raw
//! writes are the runtime's and its tooling's (RFC-0090 rule 6): a module or
//! an item that names them is public only under the `tooling` feature, and
//! crate-private otherwise. `host.rs` shows that a crate without the
//! feature cannot name them.

/// Expands the macro `$name`, which takes the visibility its items are
/// declared at, as `pub` under the `tooling` feature and as `pub(crate)`
/// otherwise.
macro_rules! tooling_vis {
    ($name:ident) => {
        #[cfg(feature = "tooling")]
        $name!(pub);
        #[cfg(not(feature = "tooling"))]
        $name!(pub(crate));
    };
}

#[cfg(feature = "tooling")]
pub mod code;
#[cfg(not(feature = "tooling"))]
mod code;
pub mod cost;
pub mod executor;
mod flight;
mod host;
mod host_graph;
mod init;
mod interpreter;
#[cfg(feature = "tooling")]
pub mod layout;
#[cfg(not(feature = "tooling"))]
mod layout;
/// The walk reads `Op::successor`, `Op::owns` and `Named::name`, which the
/// release trait does not carry.
#[cfg(all(feature = "tooling", any(debug_assertions, feature = "probe")))]
pub mod listing;
/// Without `tooling` the walk is compiled for the unit tests alone, which
/// call `last_path_segment` and nothing else; it is not split out of the
/// module for them, so the rest of the walk is allowed to go unused.
#[cfg(all(not(feature = "tooling"), test, any(debug_assertions, feature = "probe")))]
#[allow(dead_code)]
mod listing;
#[cfg(feature = "tooling")]
pub mod machine;
#[cfg(not(feature = "tooling"))]
mod machine;
mod ops;
mod port;
#[cfg(feature = "tooling")]
pub mod prepare;
#[cfg(not(feature = "tooling"))]
mod prepare;
#[cfg(feature = "tooling")]
pub mod regs;
#[cfg(not(feature = "tooling"))]
mod regs;
mod runtime;
pub mod space;
mod value;
#[cfg(feature = "tooling")]
pub mod vtable;
#[cfg(not(feature = "tooling"))]
mod vtable;

pub use executor::{AsyncJob, BlockingJob, Done, Executor, Handle, SequentialExecutor, TokioExecutor};
pub use host::{
    Access, AsyncAccess, AsyncStorage, Cause, Codec, Entry, Host, HostError, InputShape, Inputs,
    MemoryStorage, Named, Origin, Output, Page, Part, Program, Refusal, RunInputs, Scope, Source,
    Storage, StorageError, SyncAccess,
};
#[cfg(feature = "tooling")]
pub use host::{
    CompileTimes, InputListing, Listing, UntypedEntry, UntypedOutput, context_refs, environment,
    untyped_entry_ty,
};
pub use host_graph::HostGraph;
pub use port::Held;
pub use runtime::AcvusRuntime;
pub use space::{
    Commit, Committed, DirStore, Head, Identity, Log, MemoryStore, Mode, Node, NodeKind, Plain,
    Record, Space, SpaceStorage, Store, hex,
};

#[cfg(feature = "tooling")]
pub use code::{Code, CodeBody, Prepared};
#[cfg(feature = "tooling")]
pub use interpreter::{Args, Executable, Interpreter, InterpreterContext};
#[cfg(feature = "tooling")]
pub use port::ContextWrite;
#[cfg(feature = "tooling")]
pub use layout::Hooks as SpaceHooksByType;
#[cfg(feature = "tooling")]
pub use machine::fn_value_call;
#[cfg(feature = "tooling")]
pub use ops::chain::{ChainTy, LeafRead, Node as ChainNode, Nodes as ChainNodes, Reads};
/// The handlers an `Index` runs, checked and, for a bound the MIR proves,
/// unchecked (RFC-0047 rule 7).
#[cfg(feature = "tooling")]
pub use ops::index as index_handlers;
#[cfg(feature = "tooling")]
pub use prepare::{BodyRole, FrameRefusal, PrepareCtx, RegisterBound, prepare_module};
#[cfg(feature = "tooling")]
pub use runtime::ExternHandler;
#[cfg(feature = "tooling")]
pub use value::{Array, FnValue, Kind, Object, Place, Tuple, Value, VariantValue};
#[cfg(feature = "tooling")]
pub use vtable::{Composite, Vtable};
