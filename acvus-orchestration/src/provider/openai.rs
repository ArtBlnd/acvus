use crate::kind::GenerationParams;
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::{HttpRequest, ProviderConfig};

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
    ) -> HttpRequest {
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
    ) -> Result<(ModelResponse, Usage), String> {
        parse_response(json)
    }

    pub fn build_count_tokens_request(&self, _messages: &[Message]) -> Option<HttpRequest> {
        None
    }

    pub fn parse_count_tokens_response(&self, _json: &serde_json::Value) -> Result<u32, String> {
        Err("count tokens not supported for OpenAI".into())
    }
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> HttpRequest {
    let body = format_body(model, messages, tools, generation, max_output_tokens);
    let url = format!("{}/v1/chat/completions", config.endpoint);
    HttpRequest {
        url,
        headers: vec![("Authorization".into(), format!("Bearer {}", config.api_key))],
        body,
    }
}

fn format_message(m: &Message) -> serde_json::Value {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => {
                serde_json::json!({ "role": role, "content": text })
            }
            Content::Blob { mime_type, data } => {
                serde_json::json!({
                    "role": role,
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{mime_type};base64,{data}"),
                        }
                    }]
                })
            }
        },
        Message::ToolCalls(calls) => {
            let tool_calls: Vec<serde_json::Value> = calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string(),
                        }
                    })
                })
                .collect();
            serde_json::json!({
                "role": "assistant",
                "tool_calls": tool_calls,
            })
        }
        Message::ToolResult { call_id, content } => {
            serde_json::json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })
        }
    }
}

fn format_body(
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> serde_json::Value {
    let msgs: Vec<serde_json::Value> = messages.iter().map(format_message).collect();

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
    });

    if let Some(t) = generation.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = generation.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(m) = max_output_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }

    if !tools.is_empty() {
        let tool_specs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .map(|(name, type_name)| {
                        (name.clone(), serde_json::json!({ "type": type_name }))
                    })
                    .collect();

                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                        }
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), String> {
    let usage = parse_usage(json);

    let choices = json
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or("missing 'choices' in response")?;

    let choice = choices.first().ok_or("empty choices array")?;
    let message = choice.get("message").ok_or("missing 'message' in choice")?;

    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        let calls: Result<Vec<ToolCall>, String> = tool_calls
            .iter()
            .map(|tc| {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool call id")?
                    .to_string();
                let func = tc.get("function").ok_or("missing function")?;
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing function name")?
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(Default::default()));

                Ok(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();

        return Ok((ModelResponse::ToolCalls(calls?), usage));
    }

    let role = message
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("assistant")
        .to_string();

    let parts = match message.get("content") {
        Some(serde_json::Value::String(s)) => vec![Content::Text(s.clone())],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|block| {
                let ty = block.get("type").and_then(|t| t.as_str())?;
                match ty {
                    "text" => {
                        let text = block.get("text").and_then(|t| t.as_str())?;
                        Some(Content::Text(text.to_string()))
                    }
                    _ => None,
                }
            })
            .collect(),
        _ => vec![Content::Text(String::new())],
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

fn parse_usage(json: &serde_json::Value) -> Usage {
    let u = match json.get("usage") {
        Some(u) => u,
        None => return Usage::default(),
    };
    Usage {
        input_tokens: u
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        output_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::kind::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_basic_messages() {
        let body = format_body(
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
        let body = format_body(
            "gpt-4o",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            &[ToolSpec {
                name: "search".into(),
                description: "Search the web".into(),
                params: HashMap::from([("query".into(), "string".into())]),
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
        }]);
        let formatted = format_message(&msg);
        assert_eq!(formatted["role"], "assistant");
        let calls = formatted["tool_calls"].as_array().unwrap();
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
        let formatted = format_message(&msg);
        assert_eq!(formatted["role"], "tool");
        assert_eq!(formatted["tool_call_id"], "call_1");
        assert_eq!(formatted["content"], "result data");
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
