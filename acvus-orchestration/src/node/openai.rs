use std::sync::Arc;

use acvus_interpreter::{PureValue, RuntimeError, TypedValue, Value};
use acvus_utils::{Astr, Interner};

use rust_decimal::Decimal;
use rustc_hash::FxHashMap;
use tracing::{debug, info};

use super::Node;
use super::helpers::{
    content_to_value, eval_script_in_coroutine,
    execute_tool_calls, flatten_segments, make_tool_specs,
    render_messages,
};
use super::schema::openai as schema;
use crate::compile::{CompiledMessage, CompiledScript};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::spec::{CompiledOpenAICompatible, CompiledToolBinding, MaxTokens, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

// ── Provider logic (absorbed from provider/openai.rs) ───────────────

fn convert_message(m: &Message) -> schema::RequestMessage {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => schema::RequestMessage::Content {
                role: role.clone(),
                content: text.clone(),
            },
            Content::Blob { mime_type, data } => schema::RequestMessage::ContentArray {
                role: role.clone(),
                content: vec![schema::ContentPart::ImageUrl {
                    image_url: schema::ImageUrlData {
                        url: format!("data:{mime_type};base64,{data}"),
                    },
                }],
            },
        },
        Message::ToolCalls(calls) => schema::RequestMessage::ToolCalls {
            role: "assistant".into(),
            tool_calls: calls
                .iter()
                .map(|tc| schema::RequestToolCall {
                    id: tc.id.clone(),
                    call_type: "function".into(),
                    function: schema::RequestToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect(),
        },
        Message::ToolResult { call_id, content } => schema::RequestMessage::ToolResult {
            role: "tool".into(),
            tool_call_id: call_id.clone(),
            content: content.clone(),
        },
    }
}

fn resolve_reasoning_effort(thinking: &Option<ThinkingConfig>) -> Result<Option<String>, RequestError> {
    match thinking {
        None | Some(ThinkingConfig::Off) => Ok(None),
        Some(ThinkingConfig::Low) => Ok(Some("low".into())),
        Some(ThinkingConfig::Medium) => Ok(Some("medium".into())),
        Some(ThinkingConfig::High) => Ok(Some("high".into())),
        Some(config @ ThinkingConfig::Custom(_)) => {
            Err(RequestError::UnsupportedThinkingConfig {
                provider: "openai",
                config: config.clone(),
            })
        }
    }
}

fn build_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    temperature: Option<Decimal>,
    top_p: Option<Decimal>,
    thinking: &Option<ThinkingConfig>,
    max_output_tokens: Option<u32>,
) -> Result<HttpRequest, RequestError> {
    let msgs: Vec<schema::RequestMessage> = messages.iter().map(convert_message).collect();

    let tools_vec = if tools.is_empty() {
        None
    } else {
        let tools_vec: Vec<schema::Tool> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .map(|(name, param)| {
                        let prop = schema::PropertySchema {
                            ty: param.ty.clone(),
                            description: param.description.clone(),
                        };
                        Ok((
                            name.clone(),
                            serde_json::to_value(prop).map_err(|e| RequestError::Serialization { detail: e.to_string() })?,
                        ))
                    })
                    .collect::<Result<_, RequestError>>()?;

                Ok(schema::Tool {
                    tool_type: "function".into(),
                    function: schema::FunctionDecl {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: schema::FunctionParams {
                            schema_type: "object".into(),
                            properties,
                        },
                    },
                })
            })
            .collect::<Result<_, RequestError>>()?;
        Some(tools_vec)
    };

    let request = schema::Request {
        model: model.into(),
        messages: msgs,
        tools: tools_vec,
        temperature,
        top_p,
        max_tokens: max_output_tokens,
        reasoning_effort: resolve_reasoning_effort(thinking)?,
    };

    let body = serde_json::to_value(request).map_err(|e| RequestError::Serialization { detail: e.to_string() })?;
    let url = format!("{endpoint}/v1/chat/completions");
    Ok(HttpRequest {
        url,
        headers: vec![("Authorization".into(), format!("Bearer {api_key}"))],
        body,
    })
}

fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let response: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;

    let usage = match response.usage {
        Some(u) => Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        },
        None => Usage::default(),
    };

    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or(RequestError::MissingField { field: "choices" })?;
    let message = choice.message;

    if let Some(tool_calls) = message.tool_calls {
        let calls: Result<Vec<ToolCall>, RequestError> = tool_calls
            .into_iter()
            .map(|tc| {
                let id = tc.id.ok_or(RequestError::MissingField { field: "tool_call.id" })?.to_string();
                let func = tc.function.ok_or(RequestError::MissingField { field: "tool_call.function" })?;
                let name = func.name.ok_or(RequestError::MissingField { field: "tool_call.function.name" })?.to_string();
                let args_str = func.arguments.as_deref()
                    .ok_or(RequestError::MissingField { field: "tool_call.function.arguments" })?;
                let arguments: serde_json::Value = serde_json::from_str(args_str)
                    .map_err(|e| RequestError::ResponseParse {
                        detail: format!("tool_call arguments: {e}"),
                    })?;

                Ok(ToolCall {
                    id,
                    name,
                    arguments,
                    extras: None,
                })
            })
            .collect();

        return Ok((ModelResponse::ToolCalls(calls?), usage));
    }

    let role = message.role.unwrap_or_else(|| "assistant".into());

    let parts = match message.content {
        Some(schema::ResponseContent::Text(s)) => vec![Content::Text(s)],
        Some(schema::ResponseContent::Parts(arr)) => arr
            .into_iter()
            .filter_map(|block| {
                let ty = block.part_type.as_deref()?;
                match ty {
                    "text" => {
                        let text = block.text?;
                        Some(Content::Text(text))
                    }
                    _ => None,
                }
            })
            .collect(),
        // Model returned no content and no tool calls — yield empty content list
        // rather than fabricating an empty text that wasn't in the response.
        None => vec![],
    };

    let items = parts
        .into_iter()
        .map(|content| ContentItem {
            role: role.clone(),
            content,
        })
        .collect();
    Ok((ModelResponse::Content(items), usage))
}

// ── Node ────────────────────────────────────────────────────────────

const MAX_TOOL_ROUNDS: usize = 10;

struct OpenAIConfig {
    endpoint: String,
    api_key: String,
    model: String,
    messages: Vec<CompiledMessage>,
    tools: Vec<CompiledToolBinding>,
    temperature: Option<rust_decimal::Decimal>,
    top_p: Option<rust_decimal::Decimal>,
    max_tokens: MaxTokens,
    cache_key: Option<CompiledScript>,
}

pub struct OpenAICompatibleNode<F> {
    config: Arc<OpenAIConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> OpenAICompatibleNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        compiled: &CompiledOpenAICompatible,
        fetch: Arc<F>,
        interner: &Interner,
    ) -> Self {
        Self {
            config: Arc::new(OpenAIConfig {
                endpoint: compiled.endpoint.clone(),
                api_key: compiled.api_key.clone(),
                model: compiled.model.clone(),
                messages: compiled.messages.clone(),
                tools: compiled.tools.clone(),
                temperature: compiled.temperature,
                top_p: compiled.top_p,
                max_tokens: compiled.max_tokens.clone(),
                cache_key: compiled.cache_key.clone(),
            }),
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F> Node for OpenAICompatibleNode<F>
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
            let model_name = config.model.clone();

            let cached_content = if let Some(ref ck_script) = config.cache_key {
                let val =
                    eval_script_in_coroutine(&interner, &ck_script.module, &local, &handle)
                        .await?;
                match val.value() {
                    Value::Pure(PureValue::String(s)) => Some(s.clone()),
                    _ => None,
                }
            } else {
                None
            };
            // OpenAI doesn't support cached_content — ignore it
            let _ = cached_content;

            // Render messages
            let segments = render_messages(&config.messages, &interner, &local, &handle).await?;

            // No token budgeting for OpenAI (no count_tokens API)
            let mut rendered = flatten_segments(segments);
            let specs = make_tool_specs(&config.tools);

            info!(model = %model_name, messages = rendered.len(), tools = specs.len(), "openai request");
            let request = build_request(
                &config.endpoint,
                &config.api_key,
                &config.model,
                &rendered,
                &specs,
                config.temperature,
                config.top_p,
                &None, // OpenAI thinking (reasoning_effort handled inside)
                config.max_tokens.output,
            ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
            let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
            let (mut response, _usage) = parse_response(&json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;
            debug!(
                input_tokens = _usage.input_tokens,
                output_tokens = _usage.output_tokens,
                "openai response received",
            );

            let mut tool_rounds = 0usize;
            loop {
                match response {
                    ModelResponse::Content(items) => {
                        debug!(items = items.len(), "openai returned content");
                        handle.yield_val(content_to_value(&interner, &items)).await;
                        return Ok(());
                    }
                    ModelResponse::ToolCalls(calls) => {
                        tool_rounds += 1;
                        info!(round = tool_rounds, count = calls.len(), "openai tool calls");
                        if tool_rounds > MAX_TOOL_ROUNDS {
                            return Err(RuntimeError::tool_call_limit(MAX_TOOL_ROUNDS));
                        }

                        rendered.push(Message::ToolCalls(calls.clone()));
                        let tool_results = execute_tool_calls(&calls, &config.tools, &interner, &handle).await;
                        rendered.extend(tool_results);

                        debug!(
                            round = tool_rounds,
                            "openai follow-up request after tool results"
                        );
                        let request = build_request(
                            &config.endpoint,
                            &config.api_key,
                            &config.model,
                            &rendered,
                            &specs,
                            config.temperature,
                            config.top_p,
                            &None,
                            config.max_tokens.output,
                        ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
                        let parsed = parse_response(&json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        response = parsed.0;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

    use crate::message::{Message, ToolSpec, ToolSpecParam};

    use super::*;

    fn build_body(
        model: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_output_tokens: Option<u32>,
    ) -> serde_json::Value {
        let req = build_request(
            "https://api.openai.com",
            "test-key",
            model,
            messages,
            tools,
            None,
            None,
            &None,
            max_output_tokens,
        ).unwrap();
        req.body
    }

    #[test]
    fn format_basic_messages() {
        let body = build_body(
            "gpt-4o",
            &[
                Message::Content {
                    role: "system".into(),
                    content: Content::Text("You are helpful.".into()),
                },
                Message::Content {
                    role: "user".into(),
                    content: Content::Text("Hello".into()),
                },
            ],
            &[],
            None,
        );
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn format_with_tools() {
        let body = build_body(
            "gpt-4o",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            &[ToolSpec {
                name: "search".into(),
                description: "Search the web".into(),
                params: FxHashMap::from_iter([(
                    "query".into(),
                    ToolSpecParam {
                        ty: "string".into(),
                        description: None,
                    },
                )]),
            }],
            None,
        );
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "search");
    }

    #[test]
    fn format_tool_call_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
            extras: None,
        }]);
        let converted = convert_message(&msg);
        let value = serde_json::to_value(converted).unwrap();
        assert_eq!(value["role"], "assistant");
        let calls = value["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "search");
    }

    #[test]
    fn format_tool_result_message() {
        let msg = Message::ToolResult {
            call_id: "call_1".into(),
            content: "result data".into(),
        };
        let converted = convert_message(&msg);
        let value = serde_json::to_value(converted).unwrap();
        assert_eq!(value["role"], "tool");
        assert_eq!(value["tool_call_id"], "call_1");
        assert_eq!(value["content"], "result data");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                }
            }]
        });
        let (resp, _) = parse_response(&json).unwrap();
        let ModelResponse::Content(items) = resp else {
            panic!("expected Content");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].role, "assistant");
        assert!(matches!(&items[0].content, Content::Text(s) if s == "Hello there!"));
    }

    #[test]
    fn parse_tool_call_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\": \"hello\"}"
                        }
                    }]
                }
            }]
        });
        let (resp, _) = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_usage_fields() {
        let json = serde_json::json!({
            "choices": [{ "message": { "content": "hi" } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        });
        let (_, usage) = parse_response(&json).unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));
    }
}
