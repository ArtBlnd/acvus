mod handler;
pub mod types;
pub mod exec_ctx;
pub mod hof;
mod string;
mod convert;
mod bytes;
mod list;
mod option;
mod deque;
mod iter;
mod seq;

pub use handler::{IntoValue, BuiltinExecute};
pub use exec_ctx::ExecCtx;


use std::sync::LazyLock;
use rustc_hash::FxHashMap;
use acvus_mir::builtins::BuiltinId;

// ── ImplRegistry ─────────────────────────────────────────────────

/// Runtime implementation registry — maps BuiltinId to execute function.
struct ImplRegistry {
    impls: FxHashMap<BuiltinId, BuiltinExecute>,
}

impl ImplRegistry {
    fn new() -> Self {
        Self { impls: FxHashMap::default() }
    }

    fn register(&mut self, entries: Vec<(BuiltinId, BuiltinExecute)>) {
        for (id, exec) in entries {
            self.impls.insert(id, exec);
        }
    }

    fn get(&self, id: BuiltinId) -> Option<&BuiltinExecute> {
        self.impls.get(&id)
    }
}

static IMPL_REGISTRY: LazyLock<ImplRegistry> = LazyLock::new(|| {
    let mut r = ImplRegistry::new();
    r.register(string::entries());
    r.register(convert::entries());
    r.register(bytes::entries());
    r.register(list::entries());
    r.register(option::entries());
    r.register(deque::entries());
    r.register(iter::entries());
    r.register(seq::entries());
    r
});

/// Look up a builtin implementation by ID.
pub fn get_builtin_impl(id: BuiltinId) -> Option<&'static BuiltinExecute> {
    IMPL_REGISTRY.get(id)
}

#[cfg(test)]
mod tests;
