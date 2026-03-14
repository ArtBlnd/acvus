use crate::kind::{GenerationParams, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::schema::anthropic as schema;
use super::{split_system_messages, HttpRequest, ProviderConfig, ProviderError};

pub struct AnthropicModel {
    config: ProviderConfig,
    model: String,
}

impl AnthropicModel {
    pub fn new(config: ProviderConfig, model: String) -> Self {
        Self { config, model }
    }

    pub fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        max_output_tokens: Option<u32>,
        cached_content: Option<&str>,
    ) -> Result<HttpRequest, ProviderError> {
        let _ = cached_content;
        build_request(
            &self.config,
            &self.model,
            messages,
            tools,
            generation,
            max_output_tokens,
        )
    }

    pub fn parse_response(
        &self,
        json: &serde_json::Value,
    ) -> Result<(ModelResponse, Usage), ProviderError> {
        parse_response(json)
    }

    pub fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        Some(build_count_tokens_request(
            &self.config,
            &self.model,
            messages,
        ))
    }

    pub fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, ProviderError> {
        parse_count_tokens_response(json)
    }
}

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

fn convert_thinking(thinking: &ThinkingConfig) -> Result<schema::ThinkingParam, ProviderError> {
    match thinking {
        ThinkingConfig::Off => Ok(schema::ThinkingParam::Disabled {}),
        ThinkingConfig::Custom(n) => Ok(schema::ThinkingParam::Enabled { budget_tokens: *n }),
        other => Err(ProviderError::UnsupportedThinkingConfig {
            provider: "anthropic",
            config: other.clone(),
        }),
    }
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> Result<HttpRequest, ProviderError> {
    let (system, rest) = split_system_messages(messages);
    let msgs: Vec<schema::RequestMessage> = rest.into_iter().map(convert_message).collect();

    let tools_param = if tools.is_empty() {
        None
    } else {
        Some(
            tools
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
                            (
                                name.clone(),
                                serde_json::to_value(prop).expect("PropertySchema serialization"),
                            )
                        })
                        .collect();

                    schema::Tool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: schema::InputSchema {
                            schema_type: "object".into(),
                            properties,
                        },
                    }
                })
                .collect(),
        )
    };

    let thinking = generation.thinking.as_ref().map(convert_thinking).transpose()?;

    let request = schema::Request {
        model: model.to_string(),
        messages: msgs,
        max_tokens: max_output_tokens.unwrap_or(4096),
        system,
        tools: tools_param,
        temperature: generation.temperature,
        top_p: generation.top_p,
        top_k: generation.top_k,
        thinking,
    };

    let body = serde_json::to_value(request).expect("Request serialization");
    let url = format!("{}/v1/messages", config.endpoint);
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub fn build_count_tokens_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
) -> HttpRequest {
    let (system, rest) = split_system_messages(messages);
    let msgs: Vec<schema::RequestMessage> = rest.into_iter().map(convert_message).collect();

    let request = schema::CountTokensRequest {
        model: model.to_string(),
        messages: msgs,
        system,
    };

    let body = serde_json::to_value(request).expect("CountTokensRequest serialization");
    let url = format!("{}/v1/messages/count_tokens", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("anthropic-beta".into(), "token-counting-2024-11-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, ProviderError> {
    let resp: schema::CountTokensResponse =
        serde_json::from_value(json.clone()).map_err(|e| ProviderError::ResponseParse {
            detail: e.to_string(),
        })?;
    resp.input_tokens
        .ok_or(ProviderError::MissingField { field: "input_tokens" })
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), ProviderError> {
    let resp: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| ProviderError::ResponseParse {
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
                parts.push(Content::Blob {
                    mime_type: source.media_type.unwrap_or_default(),
                    data: source.data.unwrap_or_default(),
                });
            }
            schema::ResponseContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input.unwrap_or(serde_json::json!({})),
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

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

    use crate::ApiKind;
    use crate::kind::GenerationParams;
    use crate::message::{Message, ToolSpec, ToolSpecParam};

    use super::*;

    fn build_body(
        model: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        max_output_tokens: Option<u32>,
    ) -> serde_json::Value {
        let config = ProviderConfig {
            api: ApiKind::Anthropic,
            endpoint: "https://api.anthropic.com".into(),
            api_key: "test-key".into(),
        };
        let req = build_request(&config, model, messages, tools, generation, max_output_tokens)
            .unwrap();
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
            &GenerationParams::default(),
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
            &GenerationParams::default(),
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
        let config = ProviderConfig {
            api: ApiKind::Anthropic,
            endpoint: "https://api.anthropic.com".into(),
            api_key: "test-key".into(),
        };
        let req = build_count_tokens_request(
            &config,
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
        );
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
