use crate::kind::{GenerationParams, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::schema::openai as schema;
use super::{HttpRequest, ProviderConfig, ProviderError};

pub struct OpenAiModel {
    config: ProviderConfig,
    model: String,
}

impl OpenAiModel {
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

    pub fn build_count_tokens_request(&self, _messages: &[Message]) -> Option<HttpRequest> {
        None
    }

    pub fn parse_count_tokens_response(&self, _json: &serde_json::Value) -> Result<u32, ProviderError> {
        Err(ProviderError::Unsupported {
            provider: "openai",
            operation: "count_tokens",
        })
    }
}

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

fn resolve_reasoning_effort(thinking: &Option<ThinkingConfig>) -> Result<Option<String>, ProviderError> {
    match thinking {
        None | Some(ThinkingConfig::Off) => Ok(None),
        Some(ThinkingConfig::Low) => Ok(Some("low".into())),
        Some(ThinkingConfig::Medium) => Ok(Some("medium".into())),
        Some(ThinkingConfig::High) => Ok(Some("high".into())),
        Some(config @ ThinkingConfig::Custom(_)) => {
            Err(ProviderError::UnsupportedThinkingConfig {
                provider: "openai",
                config: config.clone(),
            })
        }
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
    let msgs: Vec<schema::RequestMessage> = messages.iter().map(convert_message).collect();

    let tools_vec = if tools.is_empty() {
        None
    } else {
        Some(
            tools
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
                            (
                                name.clone(),
                                serde_json::to_value(prop).expect("PropertySchema serialization"),
                            )
                        })
                        .collect();

                    schema::Tool {
                        tool_type: "function".into(),
                        function: schema::FunctionDecl {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: schema::FunctionParams {
                                schema_type: "object".into(),
                                properties,
                            },
                        },
                    }
                })
                .collect(),
        )
    };

    let request = schema::Request {
        model: model.into(),
        messages: msgs,
        tools: tools_vec,
        temperature: generation.temperature,
        top_p: generation.top_p,
        max_tokens: max_output_tokens,
        reasoning_effort: resolve_reasoning_effort(&generation.thinking)?,
    };

    let body = serde_json::to_value(request).expect("Request serialization");
    let url = format!("{}/v1/chat/completions", config.endpoint);
    Ok(HttpRequest {
        url,
        headers: vec![("Authorization".into(), format!("Bearer {}", config.api_key))],
        body,
    })
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), ProviderError> {
    let response: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| ProviderError::ResponseParse {
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
        .ok_or(ProviderError::MissingField { field: "choices" })?;
    let message = choice.message;

    if let Some(tool_calls) = message.tool_calls {
        let calls: Result<Vec<ToolCall>, ProviderError> = tool_calls
            .into_iter()
            .map(|tc| {
                let id = tc.id.ok_or(ProviderError::MissingField { field: "tool_call.id" })?.to_string();
                let func = tc.function.ok_or(ProviderError::MissingField { field: "tool_call.function" })?;
                let name = func.name.ok_or(ProviderError::MissingField { field: "tool_call.function.name" })?.to_string();
                let arguments = func
                    .arguments
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(Default::default()));

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
        None => vec![Content::Text(String::new())],
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

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

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
            api: super::super::ApiKind::OpenAI,
            endpoint: "https://api.openai.com".into(),
            api_key: "test-key".into(),
        };
        let req = build_request(&config, model, messages, tools, generation, max_output_tokens).unwrap();
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
            &GenerationParams::default(),
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
            &GenerationParams::default(),
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
