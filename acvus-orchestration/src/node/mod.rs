mod expr;
pub(crate) mod helpers;
mod llm;
mod llm_cache;
mod plain;

use acvus_utils::{Astr, Interner};
pub use expr::ExprNode;
pub use llm::LlmNode;
pub use llm_cache::LlmCacheNode;
pub use plain::PlainNode;
use rustc_hash::FxHashMap;

use std::sync::Arc;

use acvus_interpreter::{Coroutine, RuntimeError, Value};

use crate::provider::{Fetch, ProviderConfig};

/// Node = 함수. kind에 무관하게 동일한 인터페이스.
/// spawn으로 coroutine 생성 → resolver가 uniform하게 drive.
pub trait Node: Send + Sync {
    fn spawn(
        &self,
        local_context: FxHashMap<Astr, Arc<Value>>,
    ) -> Coroutine<Value, RuntimeError>;
}

/// Build a node table from compiled nodes.
/// Match once here → uniform `Arc<dyn Node>` everywhere else.
pub fn build_node_table<F>(
    compiled: &[crate::compile::CompiledNode],
    providers: &FxHashMap<String, ProviderConfig>,
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
                crate::kind::CompiledNodeKind::Plain(plain) => Arc::new(PlainNode::new(
                    plain.block.module.clone(),
                    interner,
                )),
                crate::kind::CompiledNodeKind::Expr(expr) => Arc::new(ExprNode::new(
                    expr.script.module.clone(),
                    interner,
                )),
                crate::kind::CompiledNodeKind::Llm(llm) => {
                    let provider_config = providers
                        .get(&llm.provider)
                        .cloned()
                        .unwrap_or_else(|| panic!("unknown provider: {}", llm.provider));
                    Arc::new(LlmNode::new(
                        llm,
                        provider_config,
                        Arc::clone(&fetch),
                        interner,
                    ))
                }
                crate::kind::CompiledNodeKind::LlmCache(cache) => {
                    let provider_config = providers
                        .get(&cache.provider)
                        .cloned()
                        .unwrap_or_else(|| panic!("unknown provider: {}", cache.provider));
                    Arc::new(LlmCacheNode::new(
                        cache,
                        provider_config,
                        Arc::clone(&fetch),
                        interner,
                    ))
                }
            }
        })
        .collect()
}
