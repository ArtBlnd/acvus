use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{Coroutine, ExternFnRegistry, ResumeKey, RuntimeError, Value};
use acvus_utils::{Astr, Interner};

use tracing::{debug, info};

use super::Node;
use super::helpers::render_block_in_coroutine;
use crate::compile::CompiledMessage;
use crate::message::{Content, Message};
use crate::provider::{Fetch, ProviderConfig};

pub struct LlmCacheNode<F> {
    provider_config: ProviderConfig,
    model: String,
    messages: Vec<CompiledMessage>,
    ttl: String,
    cache_config: HashMap<String, serde_json::Value>,
    fetch: Arc<F>,
    extern_fns: ExternFnRegistry,
    interner: Interner,
}

impl<F> LlmCacheNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        cache: &crate::kind::CompiledLlmCache,
        provider_config: ProviderConfig,
        fetch: Arc<F>,
        extern_fns: &ExternFnRegistry,
        interner: &Interner,
    ) -> Self {
        Self {
            provider_config,
            model: cache.model.clone(),
            messages: cache.messages.clone(),
            ttl: cache.ttl.clone(),
            cache_config: cache.cache_config.clone(),
            fetch,
            extern_fns: extern_fns.clone(),
            interner: interner.clone(),
        }
    }
}

impl<F> Node for LlmCacheNode<F>
where
    F: Fetch + 'static,
{
    fn spawn(&self, local: HashMap<Astr, Arc<Value>>) -> (Coroutine<Value, RuntimeError>, ResumeKey<Value>) {
        let messages = self.messages.clone();
        let model = self.model.clone();
        let ttl = self.ttl.clone();
        let cache_config = self.cache_config.clone();
        let provider_config = self.provider_config.clone();
        let fetch = Arc::clone(&self.fetch);
        let extern_fns = self.extern_fns.clone();
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let mut rendered = Vec::new();
            for msg in &messages {
                let CompiledMessage::Block(block) = msg else {
                    continue;
                };
                let text =
                    render_block_in_coroutine(&interner, &block.module, &local, &extern_fns, &handle).await;
                rendered.push(Message::Content {
                    role: interner.resolve(block.role).to_string(),
                    content: Content::Text(text),
                });
            }

            info!(model = %model, ttl = %ttl, messages = rendered.len(), "llm_cache request");
            let request = crate::provider::build_cache_request(
                &provider_config,
                &model,
                &rendered,
                &ttl,
                &cache_config,
            );
            let json = fetch
                .fetch(&request)
                .await
                .map_err(|e| RuntimeError::fetch(e))?;
            let cache_name = crate::provider::parse_cache_response(&provider_config.api, &json)
                .map_err(|e| RuntimeError::fetch(e))?;
            debug!(cache_name = %cache_name, "llm_cache created");

            handle.yield_val(Value::String(cache_name)).await;
            Ok(())
        })
    }
}
