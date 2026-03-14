use std::sync::Arc;

use acvus_interpreter::{RuntimeError, Value};
use acvus_utils::{Astr, Interner};

use rustc_hash::FxHashMap;
use tracing::{debug, info, warn};

use super::Node;
use super::helpers::{
    MessageSegment, allocate_token_budgets, content_to_value, eval_script_in_coroutine,
    expand_iterator_in_coroutine, flatten_segments, make_tool_specs, render_block_in_coroutine,
    value_to_tool_result,
};
use crate::compile::{CompiledMessage, CompiledScript};
use crate::kind::{CompiledToolBinding, GenerationParams, MaxTokens};
use crate::message::{Content, Message, ModelResponse};
use crate::provider::{ApiKind, Fetch, ProviderConfig, create_llm_model};

const MAX_TOOL_ROUNDS: usize = 10;

pub struct LlmNode<F> {
    api: ApiKind,
    provider_config: ProviderConfig,
    model: String,
    messages: Vec<CompiledMessage>,
    tools: Vec<CompiledToolBinding>,
    generation: GenerationParams,
    max_tokens: MaxTokens,
    cache_key: Option<CompiledScript>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> LlmNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        llm: &crate::kind::CompiledLlm,
        provider_config: ProviderConfig,
        fetch: Arc<F>,
        interner: &Interner,
    ) -> Self {
        Self {
            api: llm.api.clone(),
            provider_config,
            model: llm.model.clone(),
            messages: llm.messages.clone(),
            tools: llm.tools.clone(),
            generation: llm.generation.clone(),
            max_tokens: llm.max_tokens.clone(),
            cache_key: llm.cache_key.clone(),
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F> Node for LlmNode<F>
where
    F: Fetch + 'static,
{
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<Value>>,
    ) -> acvus_utils::Coroutine<Value, RuntimeError> {
        let messages = self.messages.clone();
        let tools = self.tools.clone();
        let api = self.api.clone();
        let model = self.model.clone();
        let generation = self.generation.clone();
        let max_tokens = self.max_tokens.clone();
        let cache_key_script = self.cache_key.clone();
        let provider_config = self.provider_config.clone();
        let fetch = Arc::clone(&self.fetch);
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let model_name = model.clone();
            let llm_model = create_llm_model(provider_config, model);

            let cached_content = if let Some(ref ck_script) = cache_key_script {
                let val =
                    eval_script_in_coroutine(&interner, &ck_script.module, &local, &handle)
                        .await?;
                match val {
                    Value::String(s) => Some(s),
                    _ => None,
                }
            } else {
                None
            };

            // Render messages
            let mut segments: Vec<MessageSegment> = Vec::new();
            for msg in &messages {
                match msg {
                    CompiledMessage::Block(block) => {
                        let text = render_block_in_coroutine(
                            &interner,
                            &block.module,
                            &local,
                            &handle,
                        )
                        .await?;
                        segments.push(MessageSegment::Single(Message::Content {
                            role: interner.resolve(block.role).to_string(),
                            content: Content::Text(text),
                        }));
                    }
                    CompiledMessage::Iterator {
                        expr,
                        slice,
                        role,
                        token_budget,
                    } => {
                        let expanded = expand_iterator_in_coroutine(
                            &api,
                            expr,
                            slice,
                            role,
                            &interner,
                            &local,
                            &handle,
                        )
                        .await?;
                        segments.push(MessageSegment::Iterator {
                            messages: expanded,
                            budget: token_budget.clone(),
                        });
                    }
                }
            }

            allocate_token_budgets(&llm_model, &*fetch, &mut segments, max_tokens.input).await;

            let mut rendered = flatten_segments(segments);
            let specs = make_tool_specs(&tools);

            info!(model = %model_name, messages = rendered.len(), tools = specs.len(), "llm request");
            let request = llm_model.build_request(
                &rendered,
                &specs,
                &generation,
                max_tokens.output,
                cached_content.as_deref(),
            ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
            let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
            let (mut response, _usage) = llm_model
                .parse_response(&json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;
            debug!(
                input_tokens = _usage.input_tokens,
                output_tokens = _usage.output_tokens,
                "llm response received",
            );

            let mut tool_rounds = 0usize;
            loop {
                match response {
                    ModelResponse::Content(items) => {
                        debug!(items = items.len(), "llm returned content");
                        handle.yield_val(content_to_value(&interner, &items)).await;
                        return Ok(());
                    }
                    ModelResponse::ToolCalls(calls) => {
                        tool_rounds += 1;
                        info!(round = tool_rounds, count = calls.len(), "llm tool calls");
                        if tool_rounds > MAX_TOOL_ROUNDS {
                            panic!("tool call limit exceeded");
                        }

                        rendered.push(Message::ToolCalls(calls.clone()));

                        for call in &calls {
                            debug!(tool = %call.name, id = %call.id, "invoking tool");
                            let binding = tools.iter().find(|t| t.name == call.name);
                            let result_text = if let Some(binding) = binding {
                                let tool_args: FxHashMap<Astr, Value> = match &call.arguments {
                                    serde_json::Value::Object(obj) => obj
                                        .iter()
                                        .map(|(k, v)| {
                                            (
                                                interner.intern(k),
                                                crate::convert::json_to_value(&interner, v),
                                            )
                                        })
                                        .collect(),
                                    _ => FxHashMap::default(),
                                };
                                let tool_value = Value::Object(tool_args);
                                let result = handle
                                    .request_extern_call(
                                        interner.intern(&binding.node),
                                        vec![tool_value],
                                    )
                                    .await;
                                value_to_tool_result(&result)
                            } else {
                                warn!(tool = %call.name, "tool not found");
                                format!("tool '{}' not found", call.name)
                            };

                            rendered.push(Message::ToolResult {
                                call_id: call.id.clone(),
                                content: result_text,
                            });
                        }

                        debug!(
                            round = tool_rounds,
                            "llm follow-up request after tool results"
                        );
                        let request = llm_model.build_request(
                            &rendered,
                            &specs,
                            &generation,
                            max_tokens.output,
                            cached_content.as_deref(),
                        ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
                        let parsed = llm_model
                            .parse_response(&json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        response = parsed.0;
                    }
                }
            }
        })
    }
}
