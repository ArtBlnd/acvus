use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::kind::NodeKind;

/// Node specification — pure compilation input, no Serde.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub name: Astr,
    pub kind: NodeKind,
    pub strategy: Strategy,
    /// Maximum retry count on RuntimeError. 0 = no retry.
    pub retry: u32,
    /// Assert script (must evaluate to Bool). If false, triggers retry.
    pub assert: Option<Astr>,
    /// Whether this node is a function node.
    pub is_function: bool,
    /// Function parameters (name, type) pairs.
    pub fn_params: Vec<(Astr, Ty)>,
}

impl NodeSpec {
    /// Build the context type map visible inside this node's scripts.
    ///
    /// Starts from `base` (typically `registry.merged()` or `context_types`)
    /// and injects fn_params (for function nodes) and optionally `@self`.
    ///
    /// All node-internal context assembly MUST go through this method
    /// to prevent fn_params from being accidentally omitted.
    pub fn build_node_context(
        &self,
        interner: &Interner,
        base: &FxHashMap<Astr, Ty>,
        self_ty: Option<Ty>,
    ) -> FxHashMap<Astr, Ty> {
        let mut ctx = base.clone();
        if self.is_function {
            for (name, ty) in &self.fn_params {
                ctx.insert(*name, ty.clone());
            }
        }
        if let Some(ty) = self_ty {
            ctx.insert(interner.intern("self"), ty);
        }
        ctx
    }
}

/// Execution strategy — determines execution timing and @self storage location.
///
/// Context hierarchy:
///   turn_context  — per-turn. Empty at turn start, discarded at turn end.
///   storage       — persistent. Survives across turns.
#[derive(Debug, Clone, Default)]
pub enum Strategy {
    /// Execute every invocation. @self stored in turn_context, overwritten each time.
    Always,
    /// Execute once per turn. @self stored in storage (persistent).
    /// Next turn can reference previous @self.
    #[default]
    OncePerTurn,
    /// Execute only when key changes. @self stored in storage (persistent).
    /// Unchanged key → previous @self retained.
    IfModified { key: Astr },
    /// Execute once per turn. @self stored in storage (persistent).
    /// Evaluates history_bind (@self + other context → entry) and appends to @turn.history.{name}.
    History { history_bind: Astr },
}

/// A message entry: either a template block or an iterator over a context key.
#[derive(Debug, Clone)]
pub enum MessageSpec {
    Block {
        role: Astr,
        source: String,
    },
    Iterator {
        key: Astr,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        slice: Option<Vec<i64>>,
        /// Override the role for all messages from this iterator.
        role: Option<Astr>,
        /// Token budget for this iterator.
        token_budget: Option<TokenBudget>,
    },
}

/// Token budget for a single iterator.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Lower = fills first (0 is highest priority).
    pub priority: u32,
    /// Minimum guaranteed tokens (reserved from the shared pool).
    pub min: Option<u32>,
    /// Maximum tokens this iterator may use.
    pub max: Option<u32>,
}
