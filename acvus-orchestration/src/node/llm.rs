use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{Coroutine, ExternFnRegistry, ResumeKey, Value};

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
    extern_fns: ExternFnRegistry,
}

impl<F> LlmNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        llm: &crate::kind::CompiledLlm,
        provider_config: ProviderConfig,
        fetch: Arc<F>,
        extern_fns: &ExternFnRegistry,
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
            extern_fns: extern_fns.clone(),
        }
    }
}

impl<F> Node for LlmNode<F>
where
    F: Fetch + 'static,
{
    fn spawn(&self, local: HashMap<String, Arc<Value>>) -> (Coroutine<Value>, ResumeKey<Value>) {
        let messages = self.messages.clone();
        let tools = self.tools.clone();
        let api = self.api.clone();
        let model = self.model.clone();
        let generation = self.generation.clone();
        let max_tokens = self.max_tokens.clone();
        let cache_key_script = self.cache_key.clone();
        let provider_config = self.provider_config.clone();
        let fetch = Arc::clone(&self.fetch);
        let extern_fns = self.extern_fns.clone();

        acvus_coroutine::coroutine(move |handle| async move {
            let llm_model = create_llm_model(provider_config, model);

            let cached_content = if let Some(ref ck_script) = cache_key_script {
                let val = eval_script_in_coroutine(ck_script, &local, &extern_fns, &handle).await;
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
                        let text =
                            render_block_in_coroutine(&block.module, &local, &extern_fns, &handle)
                                .await;
                        segments.push(MessageSegment::Single(Message::Content {
                            role: block.role.clone(),
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
                            &local,
                            &extern_fns,
                            &handle,
                        )
                        .await;
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

            let request = llm_model.build_request(
                &rendered,
                &specs,
                &generation,
                max_tokens.output,
                cached_content.as_deref(),
            );
            let json = fetch.fetch(&request).await.expect("llm fetch failed");
            let (mut response, _usage) = llm_model
                .parse_response(&json)
                .expect("llm response parse failed");

            let mut tool_rounds = 0usize;
            loop {
                match response {
                    ModelResponse::Content(items) => {
                        handle.yield_val(content_to_value(&items)).await;
                        return;
                    }
                    ModelResponse::ToolCalls(calls) => {
                        tool_rounds += 1;
                        if tool_rounds > MAX_TOOL_ROUNDS {
                            panic!("tool call limit exceeded");
                        }

                        rendered.push(Message::ToolCalls(calls.clone()));

                        for call in &calls {
                            let binding = tools.iter().find(|t| t.name == call.name);
                            let result_text = if let Some(binding) = binding {
                                let tool_args: HashMap<String, Value> = match &call.arguments {
                                    serde_json::Value::Object(obj) => obj
                                        .iter()
                                        .map(|(k, v)| (k.clone(), crate::convert::json_to_value(v)))
                                        .collect(),
                                    _ => HashMap::new(),
                                };
                                let result = handle
                                    .request_context_with(binding.node.clone(), tool_args)
                                    .await;
                                value_to_tool_result(&result)
                            } else {
                                format!("tool '{}' not found", call.name)
                            };

                            rendered.push(Message::ToolResult {
                                call_id: call.id.clone(),
                                content: result_text,
                            });
                        }

                        let request = llm_model.build_request(
                            &rendered,
                            &specs,
                            &generation,
                            max_tokens.output,
                            cached_content.as_deref(),
                        );
                        let json = fetch.fetch(&request).await.expect("llm fetch failed");
                        let parsed = llm_model
                            .parse_response(&json)
                            .expect("llm response parse failed");
                        response = parsed.0;
                    }
                }
            }
        })
    }
}
