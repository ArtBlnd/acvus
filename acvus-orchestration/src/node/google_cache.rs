use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue};
use acvus_utils::{Astr, Interner};

use rustc_hash::FxHashMap;
use tracing::{debug, info};

use super::Node;
use super::helpers::render_block_in_coroutine;
use crate::compile::CompiledMessage;
use crate::http::Fetch;
use crate::spec::CompiledGoogleAICache;
use crate::message::{Content, Message};

/// Shared config for Google AI cache node.
struct GoogleAICacheConfig {
    endpoint: String,
    api_key: String,
    model: String,
    messages: Vec<CompiledMessage>,
    ttl: String,
    cache_config: FxHashMap<String, serde_json::Value>,
}

pub struct GoogleAICacheNode<F> {
    config: Arc<GoogleAICacheConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> GoogleAICacheNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        cache: &CompiledGoogleAICache,
        fetch: Arc<F>,
        interner: &Interner,
    ) -> Self {
        Self {
            config: Arc::new(GoogleAICacheConfig {
                endpoint: cache.endpoint.clone(),
                api_key: cache.api_key.clone(),
                model: cache.model.clone(),
                messages: cache.messages.clone(),
                ttl: cache.ttl.clone(),
                cache_config: cache.cache_config.clone(),
            }),
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F> Node for GoogleAICacheNode<F>
where
    F: Fetch + 'static,
{
    fn spawn(
        &self,
        local: FxHashMap<Astr, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let config = Arc::clone(&self.config);
        let fetch = Arc::clone(&self.fetch);
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let mut rendered = Vec::new();
            for msg in &config.messages {
                let CompiledMessage::Block(block) = msg else {
                    continue;
                };
                let text = render_block_in_coroutine(
                    &interner,
                    &block.module,
                    &local,
                    &handle,
                )
                .await?;
                rendered.push(Message::Content {
                    role: interner.resolve(block.role).to_string(),
                    content: Content::Text(text),
                });
            }

            info!(model = %config.model, ttl = %config.ttl, messages = rendered.len(), "google_cache request");
            let request = super::google::build_cache_request(
                &config.endpoint,
                &config.api_key,
                &config.model,
                &rendered,
                &config.ttl,
                &config.cache_config,
            ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
            let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
            let cache_name = super::google::parse_cache_response(&json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;
            debug!(cache_name = %cache_name, "google_cache created");

            handle.yield_val(TypedValue::string(cache_name)).await;
            Ok(())
        })
    }
}
