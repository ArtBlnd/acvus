use crate::kind::GenerationParams;
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::{HttpRequest, ProviderConfig};

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

    pub fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        Some(build_count_tokens_request(
            &self.config,
            &self.model,
            messages,
        ))
    }

    pub fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, String> {
        parse_count_tokens_response(json)
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
    let url = format!("{}/v1/messages", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn build_count_tokens_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
) -> HttpRequest {
    let mut system_text = String::new();
    let mut msgs = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            msgs.push(format_message(m));
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
    });
    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }

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

pub fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, String> {
    json.get("input_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| "missing 'input_tokens' in count tokens response".into())
}

fn format_body(
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> serde_json::Value {
    let mut system_text = String::new();
    let mut msgs = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            msgs.push(format_message(m));
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
        "max_tokens": max_output_tokens.unwrap_or(4096),
    });

    if let Some(t) = generation.temperature {
        body["temperature"] = serde_json::to_value(t).unwrap();
    }
    if let Some(p) = generation.top_p {
        body["top_p"] = serde_json::to_value(p).unwrap();
    }
    if let Some(k) = generation.top_k {
        body["top_k"] = serde_json::json!(k);
    }

    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
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
                    "name": t.name,
                    "description": t.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
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
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        }
                    }]
                })
            }
        },
        Message::ToolCalls(calls) => {
            let content: Vec<serde_json::Value> = calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                })
                .collect();
            serde_json::json!({
                "role": "assistant",
                "content": content,
            })
        }
        Message::ToolResult { call_id, content } => {
            serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                }]
            })
        }
    }
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), String> {
    let usage = parse_usage(json);

    let content = json
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or("missing 'content' array in response")?;

    let mut tool_calls = Vec::new();
    let mut parts = Vec::new();

    for block in content {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    parts.push(Content::Text(text.to_string()));
                }
            }
            "image" => {
                if let Some(source) = block.get("source") {
                    let media_type = source
                        .get("media_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let data = source
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    parts.push(Content::Blob {
                        mime_type: media_type,
                        data,
                    });
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use id")?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use name")?
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }

    if !tool_calls.is_empty() {
        Ok((ModelResponse::ToolCalls(tool_calls), usage))
    } else {
        let role = json
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("assistant")
            .to_string();
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

fn parse_usage(json: &serde_json::Value) -> Usage {
    let u = match json.get("usage") {
        Some(u) => u,
        None => return Usage::default(),
    };
    Usage {
        input_tokens: u
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        output_tokens: u
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

    use crate::kind::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_system_separated() {
        let body = format_body(
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
        let body = format_body(
            "claude-sonnet-4-6",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            &[ToolSpec {
                name: "search".into(),
                description: "Search".into(),
                params: FxHashMap::from_iter([("query".into(), "string".into())]),
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
        }]);
        let formatted = format_message(&msg);
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
        let formatted = format_message(&msg);
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
            api: crate::provider::ApiKind::Anthropic,
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
