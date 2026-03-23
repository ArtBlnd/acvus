pub mod assert;
pub mod init;
pub mod interpreter;
pub mod openai;
pub mod anthropic;
pub mod google;
pub mod google_cache;
pub(crate) mod schema;

use acvus_interpreter::{Coroutine, RuntimeError, TypedValue};
use acvus_mir::graph::ContextId;
use acvus_mir::ty::Ty;
use rustc_hash::FxHashMap;

pub trait Unit: Send + Sync {
    fn spawn(
        &self,
        local_context: FxHashMap<ContextId, TypedValue>,
    ) -> Coroutine<TypedValue, RuntimeError, ContextId>;
}

pub use assert::AssertNode;
pub use init::InitNode;
pub use interpreter::InterpreterUnit;
