use std::sync::Arc;

use acvus_interpreter::{PureValue, RuntimeError, TypedValue, Value};
use acvus_utils::{Astr, Interner};

use rust_decimal::Decimal;
use rustc_hash::FxHashMap;
use tracing::{debug, info};

use super::Node;
use super::helpers::{
    allocate_token_budgets, content_to_value, eval_script_in_coroutine,
    execute_tool_calls, flatten_segments, make_tool_specs,
    render_messages, split_system_messages,
};
use super::schema::anthropic as schema;
use crate::compile::{CompiledMessage, CompiledScript};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::spec::{CompiledAnthropic, CompiledToolBinding, MaxTokens, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

// ── Provider logic (absorbed from provider/anthropic.rs) ────────────

const DEFAULT_MAX_TOKENS: u32 = 4096;
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_TOKEN_COUNTING_BETA: &str = "token-counting-2024-11-01";

fn convert_message(m: &Message) -> schema::RequestMessage {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => schema::RequestMessage {
                role: role.clone(),
                content: schema::RequestContent::Text(text.clone()),
            },
            Content::Blob { mime_type, data } => schema::RequestMessage {
                role: role.clone(),
                content: schema::RequestContent::Blocks(vec![schema::ContentBlock::Image {
                    source: schema::ImageSource {
                        source_type: "base64".into(),
                        media_type: mime_type.clone(),
                        data: data.clone(),
                    },
                }]),
            },
        },
        Message::ToolCalls(calls) => schema::RequestMessage {
            role: "assistant".into(),
            content: schema::RequestContent::Blocks(
                calls
                    .iter()
                    .map(|tc| schema::ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input: tc.arguments.clone(),
                    })
                    .collect(),
            ),
        },
        Message::ToolResult { call_id, content } => schema::RequestMessage {
            role: "user".into(),
            content: schema::RequestContent::Blocks(vec![schema::ContentBlock::ToolResult {
                tool_use_id: call_id.clone(),
                content: content.clone(),
            }]),
        },
    }
}

fn convert_thinking(thinking: &ThinkingConfig) -> Result<schema::ThinkingParam, RequestError> {
    match thinking {
        ThinkingConfig::Off => Ok(schema::ThinkingParam::Disabled {}),
        ThinkingConfig::Custom(n) => Ok(schema::ThinkingParam::Enabled { budget_tokens: *n }),
        other => Err(RequestError::UnsupportedThinkingConfig {
            provider: "anthropic",
            config: other.clone(),
        }),
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
    top_k: Option<u32>,
    thinking: &Option<ThinkingConfig>,
    max_output_tokens: Option<u32>,
) -> Result<HttpRequest, RequestError> {
    let (system, rest) = split_system_messages(messages)?;
    let msgs: Vec<schema::RequestMessage> = rest.into_iter().map(convert_message).collect();

    let tools_param = if tools.is_empty() {
        None
    } else {
        let tools_vec: Vec<schema::Tool> = tools
            .iter()
            .map(|t| {
                let properties = t
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
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: schema::InputSchema {
                        schema_type: "object".into(),
                        properties,
                    },
                })
            })
            .collect::<Result<_, RequestError>>()?;
        Some(tools_vec)
    };

    let thinking_param = thinking.as_ref().map(convert_thinking).transpose()?;

    let request = schema::Request {
        model: model.to_string(),
        messages: msgs,
        max_tokens: max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        tools: tools_param,
        temperature,
        top_p,
        top_k,
        thinking: thinking_param,
    };

    let body = serde_json::to_value(request).map_err(|e| RequestError::Serialization { detail: e.to_string() })?;
    let url = format!("{endpoint}/v1/messages");
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), api_key.to_string()),
            ("anthropic-version".into(), ANTHROPIC_API_VERSION.into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub(super) fn build_count_tokens_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
) -> Result<HttpRequest, RequestError> {
    let (system, rest) = split_system_messages(messages)?;
    let msgs: Vec<schema::RequestMessage> = rest.into_iter().map(convert_message).collect();

    let request = schema::CountTokensRequest {
        model: model.to_string(),
        messages: msgs,
        system,
    };

    let body = serde_json::to_value(request).map_err(|e| RequestError::Serialization { detail: e.to_string() })?;
    let url = format!("{endpoint}/v1/messages/count_tokens");
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), api_key.to_string()),
            ("anthropic-version".into(), ANTHROPIC_API_VERSION.into()),
            ("anthropic-beta".into(), ANTHROPIC_TOKEN_COUNTING_BETA.into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub(super) fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, RequestError> {
    let resp: schema::CountTokensResponse =
        serde_json::from_value(json.clone()).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;
    resp.input_tokens
        .ok_or(RequestError::MissingField { field: "input_tokens" })
}

fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;

    let usage = match resp.usage {
        Some(u) => Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        },
        None => Usage::default(),
    };

    let mut tool_calls = Vec::new();
    let mut parts = Vec::new();

    for block in resp.content {
        match block {
            schema::ResponseContentBlock::Text { text } => {
                parts.push(Content::Text(text));
            }
            schema::ResponseContentBlock::Image { source } => {
                let media_type = source.media_type.ok_or(RequestError::MissingField { field: "media_type" })?;
                let data = source.data.ok_or(RequestError::MissingField { field: "data" })?;
                parts.push(Content::Blob {
                    mime_type: media_type,
                    data,
                });
            }
            schema::ResponseContentBlock::ToolUse { id, name, input } => {
                let arguments = input.ok_or(RequestError::MissingField { field: "tool_use.input" })?;
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                    extras: None,
                });
            }
            schema::ResponseContentBlock::Unknown => {}
        }
    }

    if !tool_calls.is_empty() {
        Ok((ModelResponse::ToolCalls(tool_calls), usage))
    } else {
        let role = resp.role.unwrap_or_else(|| "assistant".into());
        let items = parts
            .into_iter()
            .map(|content| ContentItem {
                role: role.clone(),
                content,
            })
            .collect();
        Ok((ModelResponse::Content(items), usage))
    }
}

// ── Node ────────────────────────────────────────────────────────────

const MAX_TOOL_ROUNDS: usize = 10;

struct AnthropicConfig {
    endpoint: String,
    api_key: String,
    model: String,
    messages: Vec<CompiledMessage>,
    tools: Vec<CompiledToolBinding>,
    temperature: Option<rust_decimal::Decimal>,
    top_p: Option<rust_decimal::Decimal>,
    top_k: Option<u32>,
    max_tokens: MaxTokens,
    thinking: Option<ThinkingConfig>,
    cache_key: Option<CompiledScript>,
}

pub struct AnthropicNode<F> {
    config: Arc<AnthropicConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> AnthropicNode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        compiled: &CompiledAnthropic,
        fetch: Arc<F>,
        interner: &Interner,
    ) -> Self {
        Self {
            config: Arc::new(AnthropicConfig {
                endpoint: compiled.endpoint.clone(),
                api_key: compiled.api_key.clone(),
                model: compiled.model.clone(),
                messages: compiled.messages.clone(),
                tools: compiled.tools.clone(),
                temperature: compiled.temperature,
                top_p: compiled.top_p,
                top_k: compiled.top_k,
                max_tokens: compiled.max_tokens.clone(),
                thinking: compiled.thinking.clone(),
                cache_key: compiled.cache_key.clone(),
            }),
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F> Node for AnthropicNode<F>
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
            // Anthropic doesn't support cached_content — ignore it
            let _ = cached_content;

            // Render messages
            let mut segments = render_messages(&config.messages, &interner, &local, &handle).await?;

            allocate_token_budgets(
                &config.endpoint, &config.api_key, &config.model,
                "anthropic",
                &*fetch, &mut segments, config.max_tokens.input,
            ).await;

            let mut rendered = flatten_segments(segments);
            let specs = make_tool_specs(&config.tools);

            info!(model = %model_name, messages = rendered.len(), tools = specs.len(), "anthropic request");
            let request = build_request(
                &config.endpoint,
                &config.api_key,
                &config.model,
                &rendered,
                &specs,
                config.temperature,
                config.top_p,
                config.top_k,
                &config.thinking,
                config.max_tokens.output,
            ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
            let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
            let (mut response, _usage) = parse_response(&json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;
            debug!(
                input_tokens = _usage.input_tokens,
                output_tokens = _usage.output_tokens,
                "anthropic response received",
            );

            let mut tool_rounds = 0usize;
            loop {
                match response {
                    ModelResponse::Content(items) => {
                        debug!(items = items.len(), "anthropic returned content");
                        handle.yield_val(content_to_value(&interner, &items)).await;
                        return Ok(());
                    }
                    ModelResponse::ToolCalls(calls) => {
                        tool_rounds += 1;
                        info!(round = tool_rounds, count = calls.len(), "anthropic tool calls");
                        if tool_rounds > MAX_TOOL_ROUNDS {
                            return Err(RuntimeError::tool_call_limit(MAX_TOOL_ROUNDS));
                        }

                        rendered.push(Message::ToolCalls(calls.clone()));
                        let tool_results = execute_tool_calls(&calls, &config.tools, &interner, &handle).await;
                        rendered.extend(tool_results);

                        debug!(
                            round = tool_rounds,
                            "anthropic follow-up request after tool results"
                        );
                        let request = build_request(
                            &config.endpoint,
                            &config.api_key,
                            &config.model,
                            &rendered,
                            &specs,
                            config.temperature,
                            config.top_p,
                            config.top_k,
                            &config.thinking,
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
            "https://api.anthropic.com",
            "test-key",
            model,
            messages,
            tools,
            None,
            None,
            None,
            &None,
            max_output_tokens,
        ).unwrap();
        req.body
    }

    #[test]
    fn format_system_separated() {
        let body = build_body(
            "claude-sonnet-4-6",
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
        assert_eq!(body["system"], "You are helpful.");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn format_with_tools() {
        let body = build_body(
            "claude-sonnet-4-6",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            &[ToolSpec {
                name: "search".into(),
                description: "Search".into(),
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
        assert_eq!(tools[0]["name"], "search");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn format_tool_use_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "toolu_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
            extras: None,
        }]);
        let converted = convert_message(&msg);
        let formatted = serde_json::to_value(converted).unwrap();
        assert_eq!(formatted["role"], "assistant");
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "toolu_1");
    }

    #[test]
    fn format_tool_result_message() {
        let msg = Message::ToolResult {
            call_id: "toolu_1".into(),
            content: "result data".into(),
        };
        let converted = convert_message(&msg);
        let formatted = serde_json::to_value(converted).unwrap();
        assert_eq!(formatted["role"], "user");
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn"
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
    fn parse_tool_use_response() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "Let me search." },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"query": "hello"}
                }
            ],
            "stop_reason": "tool_use"
        });
        let (resp, _) = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].id, "toolu_123");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_usage_fields() {
        let json = serde_json::json!({
            "content": [{ "type": "text", "text": "hi" }],
            "usage": { "input_tokens": 15, "output_tokens": 8 }
        });
        let (_, usage) = parse_response(&json).unwrap();
        assert_eq!(usage.input_tokens, Some(15));
        assert_eq!(usage.output_tokens, Some(8));
    }

    #[test]
    fn count_tokens_request_format() {
        let req = build_count_tokens_request(
            "https://api.anthropic.com",
            "test-key",
            "claude-sonnet-4-6",
            &[
                Message::Content {
                    role: "system".into(),
                    content: Content::Text("You are helpful.".into()),
                },
                Message::Content {
                    role: "user".into(),
                    content: Content::Text("hello".into()),
                },
            ],
        ).unwrap();
        assert!(req.url.contains("/v1/messages/count_tokens"));
        assert_eq!(req.body["model"], "claude-sonnet-4-6");
        assert_eq!(req.body["system"], "You are helpful.");
        assert_eq!(req.body["messages"].as_array().unwrap().len(), 1);
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "anthropic-beta" && v.contains("token-counting"))
        );
    }

    #[test]
    fn count_tokens_response_parsing() {
        let json = serde_json::json!({ "input_tokens": 37 });
        assert_eq!(parse_count_tokens_response(&json).unwrap(), 37);
    }
}
