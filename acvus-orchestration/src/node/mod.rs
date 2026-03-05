mod expr;
pub(crate) mod helpers;
mod llm;
mod llm_cache;
mod plain;

pub use expr::ExprNode;
pub use llm::LlmNode;
pub use llm_cache::LlmCacheNode;
pub use plain::PlainNode;

use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{Coroutine, ExternFnRegistry, ResumeKey, Value};

use crate::provider::{Fetch, ProviderConfig};

/// Node = 함수. kind에 무관하게 동일한 인터페이스.
/// spawn으로 coroutine 생성 → resolver가 uniform하게 drive.
pub trait Node: Send + Sync {
    fn spawn(
        &self,
        local_context: HashMap<String, Arc<Value>>,
    ) -> (Coroutine<Value>, ResumeKey<Value>);
}

/// Build a node table from compiled nodes.
/// Match once here → uniform `Arc<dyn Node>` everywhere else.
pub fn build_node_table<F>(
    compiled: &[crate::compile::CompiledNode],
    providers: &HashMap<String, ProviderConfig>,
    fetch: Arc<F>,
    extern_fns: &ExternFnRegistry,
) -> Vec<Arc<dyn Node>>
where
    F: Fetch + 'static,
{
    compiled
        .iter()
        .map(|node| -> Arc<dyn Node> {
            match &node.kind {
                crate::kind::CompiledNodeKind::Plain(plain) => {
                    Arc::new(PlainNode::new(plain.block.module.clone(), extern_fns))
                }
                crate::kind::CompiledNodeKind::Expr(expr) => {
                    Arc::new(ExprNode::new(expr.script.module.clone(), extern_fns))
                }
                crate::kind::CompiledNodeKind::Llm(llm) => {
                    let provider_config = providers
                        .get(&llm.provider)
                        .cloned()
                        .unwrap_or_else(|| panic!("unknown provider: {}", llm.provider));
                    Arc::new(LlmNode::new(
                        llm,
                        provider_config,
                        Arc::clone(&fetch),
                        extern_fns,
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
                        extern_fns,
                    ))
                }
            }
        })
        .collect()
}
