mod anthropic;
mod expression;
mod google;
mod google_cache;
pub(crate) mod helpers;
mod iterator;
mod openai;
mod plain;
mod schema;

use acvus_utils::Interner;
pub use anthropic::AnthropicNode;
pub use expression::ExpressionNode;
pub use google::GoogleAINode;
pub use google_cache::GoogleAICacheNode;
pub use iterator::IteratorNode;
pub use openai::OpenAICompatibleNode;
pub use plain::PlainNode;

use std::sync::Arc;

use acvus_interpreter::{Coroutine, RuntimeError, TypedValue};
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::{CompiledNode, CompiledNodeKind};
use crate::http::Fetch;

/// Node = function. kind-agnostic uniform interface.
/// spawn creates a coroutine -> resolver drives it uniformly.
pub trait Node: Send + Sync {
    fn spawn(
        &self,
        local_context: FxHashMap<Astr, TypedValue>,
    ) -> Coroutine<TypedValue, RuntimeError>;
}

/// Build a node table from compiled nodes.
/// Match once here -> uniform `Arc<dyn Node>` everywhere else.
pub fn build_node_table<F>(
    compiled: &[CompiledNode],
    fetch: Arc<F>,
    interner: &Interner,
) -> Vec<Arc<dyn Node>>
where
    F: Fetch + 'static,
{
    compiled
        .iter()
        .map(|node| -> Arc<dyn Node> {
            match &node.kind {
                CompiledNodeKind::Plain(plain) => Arc::new(PlainNode::new(
                    plain.block.module.clone(),
                    interner,
                )),
                CompiledNodeKind::Expression(expr) => Arc::new(ExpressionNode::new(
                    expr.script.module.clone(),
                    interner,
                )),
                CompiledNodeKind::OpenAICompatible(c) => Arc::new(OpenAICompatibleNode::new(
                    c,
                    Arc::clone(&fetch),
                    interner,
                )),
                CompiledNodeKind::Anthropic(c) => Arc::new(AnthropicNode::new(
                    c,
                    Arc::clone(&fetch),
                    interner,
                )),
                CompiledNodeKind::GoogleAI(c) => Arc::new(GoogleAINode::new(
                    c,
                    Arc::clone(&fetch),
                    interner,
                )),
                CompiledNodeKind::GoogleAICache(c) => Arc::new(GoogleAICacheNode::new(
                    c,
                    Arc::clone(&fetch),
                    interner,
                )),
                CompiledNodeKind::Iterator { sources, unordered } => {
                    Arc::new(IteratorNode::new(
                        sources.clone(),
                        *unordered,
                        &node.output_ty,
                        interner,
                    ))
                }
            }
        })
        .collect()
}
